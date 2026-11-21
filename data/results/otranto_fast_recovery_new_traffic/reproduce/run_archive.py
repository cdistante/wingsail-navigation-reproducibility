"""Replay fixed marine forcing with a predefined new received-AIS encounter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

if __name__ == "__main__":
    sys.path.insert(0, str(Path.cwd()))

import numpy as np
import xarray as xr

from world import Weather
from world._geo import normalise_longitude, offset_latitude_longitude
from world.animate_episode import SaildroneAnimation
from world.weather import Wind


ORIGINAL_SAMPLE = SaildroneAnimation.sample_wind_grid
KNOTS_PER_METRE_SECOND = 1.94384449
OTRANTO_BOUNDS = {"latitude_bounds": (39.5, 41.0), "longitude_bounds": (18.0, 19.5)}


def directed_provider(direction):
    class DirectedWeather(Weather):
        _controlled_wind_from_deg = float(direction) % 360.0

        def __init__(self):
            super().__init__(**OTRANTO_BOUNDS)

        def wind(self, latitude, longitude, when):
            native = super().wind(latitude, longitude, when)
            angle = np.radians(self._controlled_wind_from_deg)
            return Wind(float(-native.speed * np.sin(angle)), float(-native.speed * np.cos(angle)),
                        native.speed, self._controlled_wind_from_deg)

    return DirectedWeather


def sample_wind_grid(cls, state, observation, weather=None):
    """Batch native speed interpolation, then apply the same FROM rotation."""
    if not isinstance(weather, Weather) or not hasattr(weather, "_controlled_wind_from_deg"):
        return ORIGINAL_SAMPLE(state, observation, weather)
    dataset = weather._require_dataset()
    if any(dataset.sizes[name] < 2 for name in ("latitude", "longitude", "valid_time")):
        return ORIGINAL_SAMPLE(state, observation, weather)
    try:
        valid_time = weather._normalise_time(observation.time)
        weather._ensure_time_in_bounds(dataset, valid_time)
    except ValueError:
        return ORIGINAL_SAMPLE(state, observation, weather)
    fallback = observation.wind
    assert fallback is not None
    east, north = cls._wind_grid_offsets().T
    latitudes, longitudes = offset_latitude_longitude(state.latitude, state.longitude, east, north)
    longitudes = normalise_longitude(longitudes)
    inside = np.ones(latitudes.shape, dtype=bool)
    for axis, points in (("latitude", latitudes), ("longitude", longitudes)):
        coordinates = np.asarray(dataset.coords[axis].values)
        inside &= np.isfinite(points) & (coordinates.min() <= points) & (points <= coordinates.max())
    speed = np.full(latitudes.shape, fallback.speed, dtype=float)
    if inside.any():
        native = dataset[["u10", "v10"]].interp(
            latitude=xr.DataArray(latitudes[inside], dims="wind_point"),
            longitude=xr.DataArray(longitudes[inside], dims="wind_point"), valid_time=valid_time)
        samples = np.hypot(native.u10.values, native.v10.values)
        good = np.isfinite(samples)
        speed[np.flatnonzero(inside)[good]] = samples[good]
    angle = np.radians(weather._controlled_wind_from_deg)
    return -speed * np.sin(angle) * KNOTS_PER_METRE_SECOND, -speed * np.cos(angle) * KNOTS_PER_METRE_SECOND


def regional_provider(provider):
    class RegionalProvider(provider):
        def __init__(self):
            super().__init__(**OTRANTO_BOUNDS)
    return RegionalProvider


def main():
    from world import run_legacy_fossen as runner

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--batched-wind-grid", action="store_true")
    parser.add_argument("--regional-otranto", action="store_true")
    parser.add_argument("--wind-from", type=float, required=True)
    parser.add_argument("--traffic-fixture",type=Path,required=True)
    options, runner_args = parser.parse_known_args()
    if not options.regional_otranto or not np.isfinite(options.wind_from):
        parser.error("Use the fixed regional archive crop and a finite wind FROM direction.")
    if options.batched_wind_grid:
        SaildroneAnimation.sample_wind_grid = classmethod(sample_wind_grid)
    from traffic_fixture import fixture_provider
    traffic_provider,traffic_config,traffic_hash=fixture_provider(options.traffic_fixture)
    weather = directed_provider(options.wind_from)
    native_lines = runner.navigation_method_lines

    def method_lines(arguments):
        return tuple(line if line != "Archived wind, waves and current" else
                     f"Wind FROM {options.wind_from % 360:g}°; archived speed, waves and current"
                     for line in native_lines(arguments)) + (
                         f"Predefined traffic encounter · {traffic_config.get('encounter_id', options.traffic_fixture.parent.name)}",
                     )

    print("Marine archive source grids:", json.dumps(OTRANTO_BOUNDS), flush=True)
    print("Controlled wind FROM degrees:", options.wind_from % 360.0, flush=True)
    print("Wind override scope: Weather.wind/sample, plant, future tacks/hold, temporal and barbs", flush=True)
    print("Traffic fixture:", str(options.traffic_fixture.resolve()),flush=True)
    print("Traffic fixture SHA256:",traffic_hash,flush=True)
    print("Traffic runtime input: analytical parameters only; no truth CSV",flush=True)
    print("Traffic fixture ID:",traffic_config.get("encounter_id",traffic_config.get("fixture","new mixed")),flush=True)
    print("Runner arguments:", json.dumps(runner_args), flush=True)
    with (patch.object(runner,"SyntheticAIS",traffic_provider),
          patch.object(runner, "Weather", weather),
          patch.object(runner, "Waves", regional_provider(runner.Waves)),
          patch.object(runner, "Currents", regional_provider(runner.Currents)),
          patch.object(runner, "navigation_method_lines", method_lines),
          patch.object(sys, "argv", [sys.argv[0], *runner_args])):
        runner.main()


if __name__ == "__main__":
    main()
