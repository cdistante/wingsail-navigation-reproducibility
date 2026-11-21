"""Replay fixed marine forcing with a predefined new received-AIS encounter."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

if __name__ == "__main__":
    sys.path.insert(0, str(Path.cwd()))

import numpy as np
import xarray as xr

from world import Weather
from world.ais_departure_gate import DepartureGateSettings, assess_departure
from world._geo import normalise_longitude, offset_latitude_longitude
from world.animate_episode import SaildroneAnimation
from world.boat import BoatState
from world.weather import Wind
from world.world import WorldObservation


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


def select_departure(runner, arguments, provider):
    """Advance only the AIS receiver clock while the craft is in protected staging."""
    settings = DepartureGateSettings()
    start = runner.normalise_time(arguments.time)
    traffic_epoch = start + np.timedelta64(round(arguments.ais_launch_lead * 1e9), "ns")
    reporting = runner.synthetic_ais_reporting(arguments)
    if (reporting is None or not arguments.ais_predictive_fallback
            or arguments.ais_preview_age_budget != settings.preview_age_budget_s
            or reporting.delay_seconds > settings.maximum_report_age_s
            or arguments.route_gpx is None):
        raise ValueError("Departure retest requires the specified delayed AIS receiver and strict post-launch checks.")
    route = runner.Route.load_gpx(arguments.route_gpx, route_name=arguments.route_name)
    boat = BoatState(route.departure.latitude, route.departure.longitude, arguments.heading, 0.0,
                     flap=arguments.flap)
    receiver = provider(arguments.ais_scenario, boat.latitude, boat.longitude, reporting=reporting,
                        launch_lead_seconds=arguments.ais_launch_lead,
                        sharp_manoeuvres=arguments.ais_maneuvering_sharp).load()
    output = Path(arguments.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    log = output / "departure_gate.jsonl"
    streak = 0
    roster = set()
    previous_generations = {}
    launched = False
    maximum_wall = 0.0
    with log.open("x", encoding="utf-8") as stream:
        for step in range(int(settings.maximum_wait_s / arguments.dt) + 1):
            started = perf_counter()
            elapsed = step * arguments.dt
            when = start + np.timedelta64(round(elapsed * 1e9), "ns")
            reports = tuple(receiver.targets(elapsed))
            observation = WorldObservation(when, boat, None, None, None, None, None, None, reports)
            assessment = assess_departure(
                observation, settings, arguments.dt,
                acceleration_bound_mps2=arguments.ais_acceleration_bound,
                clearance_reserve_metres=arguments.ais_clearance_reserve,
            )
            report_ids = {target.mmsi for target in reports}
            roster.update(report_ids)
            present = bool(roster) and roster == report_ids
            generations = {target.mmsi: elapsed - target.report_age_seconds for target in reports
                           if target.report_age_seconds is not None}
            renewed = all(epoch > previous_generations.get(mmsi, -float("inf"))
                          for mmsi, epoch in generations.items()) and len(generations) == len(reports)
            previous_generations.update(generations)
            streak = streak + 1 if assessment.eligible and present and renewed else 0
            launched = streak >= settings.required_consecutive_observations
            wall = perf_counter() - started
            maximum_wall = max(maximum_wall, wall)
            row = {
                "step": step, "monitoring_elapsed_s": elapsed, "observation_utc": utc(when),
                "own_state": asdict(boat), "received_reports": [asdict(target) for target in reports],
                **asdict(assessment), "roster_continuous": present, "reports_renewed": renewed,
                "clear_streak": streak, "launched": launched, "wall_seconds": wall,
            }
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            if launched:
                break
    receiver.close()
    summary = {
        "monitoring_start_utc": utc(start), "traffic_epoch_utc": utc(traffic_epoch),
        "actual_departure_utc": utc(when) if launched else None,
        "gate_wait_seconds": elapsed, "parameters": asdict(settings) | {
            "interval_s": arguments.dt, "acceleration_bound_mps2": arguments.ais_acceleration_bound,
            "clearance_reserve_metres": arguments.ais_clearance_reserve,
        },
        "attempts": step + 1, "maximum_gate_decision_s": maximum_wall,
        "log_sha256": sha256(log.read_bytes()).hexdigest(),
        "staging": "Protected external staging boundary. No Fossen drift or in-water berth clearance is assumed before release.",
        "certificate": row if launched else None,
    }
    (output / "departure_gate.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    if not launched:
        raise RuntimeError("AIS departure gate did not find a release within the monitoring limit.")
    print("AIS departure gate result:", json.dumps(summary), flush=True)
    return elapsed, summary["actual_departure_utc"]


def utc(when):
    return np.datetime_as_string(when, unit="s") + "Z"


def main():
    from world import run_legacy_fossen as runner

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--batched-wind-grid", action="store_true")
    parser.add_argument("--regional-otranto", action="store_true")
    parser.add_argument("--wind-from", type=float, required=True)
    parser.add_argument("--traffic-fixture",type=Path,required=True)
    parser.add_argument("--ais-departure-gate", action="store_true")
    parser.add_argument("--departure-gate-only", action="store_true", help=argparse.SUPPRESS)
    options, runner_args = parser.parse_known_args()
    if not options.regional_otranto or not np.isfinite(options.wind_from):
        parser.error("Use the fixed regional archive crop and a finite wind FROM direction.")
    if options.batched_wind_grid:
        SaildroneAnimation.sample_wind_grid = classmethod(sample_wind_grid)
    from traffic_fixture import fixture_provider
    traffic_provider,traffic_config,traffic_hash=fixture_provider(options.traffic_fixture)
    if options.ais_departure_gate:
        parsed = runner.parser().parse_args(runner_args)
        wait, departure = select_departure(runner, parsed, traffic_provider)
        if options.departure_gate_only:
            return
        runner_args[runner_args.index("--time") + 1] = departure

        class TrafficOnFixedClock(traffic_provider):
            def targets(self, when=None):
                return super().targets(wait + (0.0 if when is None else float(when)))

        traffic_provider = TrafficOnFixedClock
    weather = directed_provider(options.wind_from)
    native_lines = runner.navigation_method_lines

    def method_lines(arguments):
        original = native_lines(arguments)
        if options.ais_departure_gate:
            original = tuple(line for line in original if not line.startswith("Saildrone launch "))
            original += (f"AIS-gated departure {departure} · protected staging {wait:g} s",)
        return tuple(line if line != "Archived wind, waves and current" else
                     f"Wind FROM {options.wind_from % 360:g}°; archived speed, waves and current"
                     for line in original) + (
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
