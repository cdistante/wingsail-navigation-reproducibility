"""Read synchronized seasonal subsets without exposing future forcing to guidance."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
import xarray as xr

from world.boat import BoatState
from world.currents import Current
from world.weather import Wind
from world.waves import Wave
from world.world import WorldObservation
from world.temporal_recovery import HISTORY_STEPS, WEATHER_INTERVAL_SECONDS


STUDY = Path(__file__).resolve().parents[1]
FORCING = STUDY / "forcing"
COORDINATES = {"time": ("valid_time", "time"), "latitude": ("latitude", "lat"),
               "longitude": ("longitude", "lon")}
VARIABLES = {"wind": (("u10", "10m_u_component_of_wind"), ("v10", "10m_v_component_of_wind")),
             "current": (("uo", "water_u"), ("vo", "water_v")),
             "wave": (("VHM0", "significant_wave_height"), ("VTPK", "wave_peak_period"),
                      ("VMDR", "mean_wave_direction"))}


def _find(dataset, aliases):
    for name in aliases:
        if name in dataset:
            return name
    raise ValueError(f"Required seasonal archive field unavailable: {aliases}.")


def _utc_ns(when):
    day = np.datetime64(str(when).removesuffix("Z"), "ns")
    if np.isnat(day):
        raise ValueError("Environmental sampling requires a valid UTC timestamp.")
    return int(day.astype(np.int64))


def _read(path, kind):
    with xr.open_dataset(path) as source:
        names = {_find(source, aliases): target for target, aliases in COORDINATES.items()}
        ds = source.rename({name: target for name, target in names.items() if name != target})
        for axis in ("time", "latitude", "longitude"):
            ds = ds.sortby(axis)
        fields = []
        for aliases in VARIABLES[kind]:
            field = ds[_find(ds, aliases)]
            for axis in set(field.dims) - {"time", "latitude", "longitude"}:
                if field.sizes[axis] == 1:
                    field = field.isel({axis: 0})
                elif axis.lower() in ("depth", "depthu", "depthv", "lev"):
                    field = field.isel({axis: int(np.abs(field[axis]).argmin())})
                else:
                    raise ValueError(f"Unresolved forcing dimension {axis} in {path}.")
            fields.append(np.asarray(field.transpose("time", "latitude", "longitude"), dtype=float))
        times = ds["time"].values.astype("datetime64[ns]").astype("int64")
        latitude = np.asarray(ds["latitude"], dtype=float)
        longitude = np.asarray(ds["longitude"], dtype=float)
    if len(times) < 2 or np.any(np.diff(times) <= 0) or np.any(np.diff(latitude) <= 0) or np.any(np.diff(longitude) <= 0):
        raise ValueError(f"{path} has nonunique or insufficient coordinates.")
    if kind == "wave":
        height, period, direction = fields
        angle = np.radians(direction)
        fields = [height, period, np.sin(angle), np.cos(angle)]
    return times, latitude, longitude, np.stack(fields, axis=-1)


class _Grid:
    def __init__(self, path, kind):
        self.times, self.latitudes, self.longitudes, self.fields = _read(path, kind)
        self.origin_ns = int(self.times[0])
        self.seconds = (self.times - self.origin_ns) / 1e9
        axes = self.seconds, self.latitudes, self.longitudes
        valid = np.isfinite(self.fields).all(axis=-1)
        self._weighted = RegularGridInterpolator(axes, np.where(valid[..., None], self.fields, 0.))
        self._coverage = RegularGridInterpolator(axes, valid.astype(float))
        # Retain wet-cell fallback for a near-coastal all-dry stencil.
        self._valid = valid
        self._nearest = {}
        self._fallback_count = 0

    def _nearest_wet(self, seconds, latitude, longitude):
        index = int(np.searchsorted(self.seconds, seconds, side="left"))
        before, after = max(0, index - 1), min(index, len(self.seconds) - 1)
        paired = self._valid[before] & self._valid[after]
        if not paired.any():
            raise ValueError("Both bracketing times must contain a valid coastal forcing cell.")
        key = before, after
        if key not in self._nearest:
            j, i = np.nonzero(paired)
            scale = np.cos(np.radians(float(np.mean(self.latitudes))))
            tree = cKDTree(np.column_stack((self.latitudes[j], self.longitudes[i] * scale)))
            self._nearest[key] = tree, j, i, scale
        tree, j, i, scale = self._nearest[key]
        distance, near = tree.query((latitude, longitude * scale))
        if distance > .15:
            raise ValueError(f"No trusted wet archive cell within 0.15° of ({latitude}, {longitude}).")
        left = self.fields[before, j[near], i[near]]
        right = self.fields[after, j[near], i[near]]
        fraction = ((seconds - self.seconds[before]) / (self.seconds[after] - self.seconds[before])
                    if after != before else 0.)
        self._fallback_count += 1
        return (1 - fraction) * left + fraction * right

    def values(self, latitude, longitude, when_ns):
        seconds = (when_ns - self.origin_ns) / 1e9
        query = (seconds, latitude, longitude)
        coverage = float(self._coverage(query))
        if coverage > 1e-9:
            return self._weighted(query) / coverage
        return self._nearest_wet(*query)


@dataclass(frozen=True)
class Forcing:
    wind: Wind
    current: Current
    wave: Wave

    def vector(self):
        return np.array((self.wind.u, self.wind.v, self.current.u, self.current.v, self.wave.height))


class SeasonalArchive:
    """Hourly source-grid interpolator shared by physics and learned arms."""

    def __init__(self, folder):
        folder = Path(folder)
        self._wind = _Grid(folder / "wind.nc", "wind")
        self._current = _Grid(folder / "current.nc", "current")
        self._wave = _Grid(folder / "wave.nc", "wave")

    @lru_cache(maxsize=8192)
    def sample(self, latitude, longitude, when_ns: int) -> Forcing:
        wind_u, wind_v = map(float, self._wind.values(latitude, longitude, when_ns))
        cur_u, cur_v = map(float, self._current.values(latitude, longitude, when_ns))
        height, period, east, north = map(float, self._wave.values(latitude, longitude, when_ns))
        wind = Wind(wind_u, wind_v, float(np.hypot(wind_u, wind_v)),
                    float(np.degrees(np.arctan2(-wind_u, -wind_v)) % 360))
        current = Current(cur_u, cur_v, float(np.hypot(cur_u, cur_v)),
                          float(np.degrees(np.arctan2(cur_u, cur_v)) % 360), 0.)
        wave = Wave(height, float(np.degrees(np.arctan2(east, north)) % 360), period)
        return Forcing(wind, current, wave)

    def at(self, latitude, longitude, when):
        return self.sample(float(latitude), float(longitude), _utc_ns(when))

    def observe(self, state: BoatState, when, reports=()):
        stamp = _utc_ns(when)
        forcing = self.sample(state.latitude, state.longitude, stamp)
        return WorldObservation(np.datetime64(stamp, "ns"), state, forcing.wind, forcing.wave,
                                forcing.current, None, None, None, tuple(reports))

    def weather_history(self, latitude, longitude, when):
        utc = _utc_ns(when)
        quantum = WEATHER_INTERVAL_SECONDS * 1_000_000_000
        current_window = utc // quantum
        return np.stack([self.sample(float(latitude), float(longitude), index * quantum).vector()
                         for index in range(current_window + 1 - HISTORY_STEPS, current_window + 1)])

    def coastal_fallbacks(self):
        return {name: grid._fallback_count for name, grid in
                (("wind", self._wind), ("current", self._current), ("wave", self._wave))}
