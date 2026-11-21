"""Pair seasonal weather with a causal receiver for route-relative AIS fixtures."""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from hashlib import sha256
import math
from pathlib import Path

import numpy as np

from world._geo import normalise_time
from world.global_map import GlobalOccupancyMap
from world.synthetic_ais import SyntheticAISManoeuvre, SyntheticAISReporting, SyntheticAISTarget


ROOT = Path(__file__).resolve().parents[4]
CHART_PATH = ROOT / "data/bathymetry/gebco_2026_n44.0_s37.0_w14.0_e22.0.nc"
CHART_MINIMUM_DEPTH_M = 2.


@lru_cache(maxsize=4)
def _chart_grid(path):
    chart = GlobalOccupancyMap(CHART_MINIMUM_DEPTH_M, path).load()
    bounds = []
    for axis in (chart.latitudes, chart.longitudes):
        if axis is None or len(axis) < 2:
            raise ValueError("The chart needs a GEBCO coordinate raster.")
        spacing = float(np.median(np.diff(axis)))
        if spacing == 0. or not np.allclose(np.diff(axis), spacing, atol=1e-8, rtol=1e-6):
            raise ValueError("The chart needs regular latitude and longitude grid axes.")
        bounds.append((float(axis[0]), spacing))
    return chart, np.asarray(bounds, dtype=float)


def _location(point):
    latitude = float(point["latitude"])
    longitude = float(point["longitude"])
    if not (-90. <= latitude <= 90. and -180. <= longitude <= 180.):
        raise ValueError("The traffic origin must be a valid latitude/longitude.")
    return latitude, longitude


def _vector(course, speed):
    angle = math.radians(course)
    return speed * math.sin(angle), speed * math.cos(angle)


def _true_bearing(origin, target):
    if origin == target:
        raise ValueError("The traffic fixture needs a nonzero first GPX leg.")
    start, finish = math.radians(origin[0]), math.radians(target[0])
    delta = math.radians(target[1] - origin[1])
    return math.degrees(math.atan2(math.sin(delta) * math.cos(finish),
                                  math.cos(start) * math.sin(finish)
                                  - math.sin(start) * math.cos(finish) * math.cos(delta))) % 360.


class ScenarioEnvironment:
    """Read observed weather and delivered AIS, with a separate truth audit API.

    The receiver starts at the scenario's requested departure epoch, including
    any negative prehistory needed for a delayed first report. Branches can
    query times in any order. A delayed or sheltered launch must not reset the
    traffic clock. Positions and compass bearings are anchored to the first GPX
    leg in a true east/north local frame, independent of ownship heading.
    """

    def __init__(self, archive, scenario, route, traffic_profile, *, chart_path=CHART_PATH):
        self.archive = archive
        self.chart_path = Path(chart_path).resolve() if chart_path is not None else None
        self.scenario_id = scenario["id"]
        self.start_utc = normalise_time(scenario["departure_utc"])
        points = route["waypoints"]
        self.origin = _location(scenario["departure"])
        if self.origin != _location(points[0]) or scenario["route_gpx"] != route["route_gpx"]:
            raise ValueError(f"{self.scenario_id}: traffic origin differs from the GPX departure.")
        self.route_bearing_deg = float(route["first_leg_bearing_deg"])
        true_bearing = _true_bearing(self.origin, _location(points[1]))
        discrepancy = abs((self.route_bearing_deg - true_bearing + 180.) % 360. - 180.)
        if not math.isfinite(self.route_bearing_deg) or discrepancy > 1e-7:
            raise ValueError(f"{self.scenario_id}: traffic orientation differs from the first GPX leg.")

        self.reporting = SyntheticAISReporting(
            float(traffic_profile["report_interval_s"]), float(traffic_profile["delay_s"]),
            tuple((float(start), float(stop)) for start, stop in traffic_profile["drop_windows_s"]),
        )
        self.targets = self._targets(scenario["traffic_profile"], traffic_profile["targets"])
        self.manoeuvres, self.max_target_acceleration_mps2 = self._ramps(traffic_profile.get("velocity_ramps", ()))

    @classmethod
    def from_manifest(cls, archive, manifest, scenario, *, chart_path=CHART_PATH):
        return cls(archive, scenario, manifest["routes"][scenario["route_id"]],
                   manifest["traffic_profiles"][scenario["traffic_profile"]], chart_path=chart_path)

    def _targets(self, profile, raw):
        angle = math.radians(self.route_bearing_deg)
        along_east, along_north = math.sin(angle), math.cos(angle)
        # Facing north, positive starboard is east. Rotating this pair by the
        # first-leg bearing preserves relative approach and crossing geometry.
        right_east, right_north = along_north, -along_east
        namespace = int.from_bytes(sha256(profile.encode("utf-8")).digest()[:2], "big")
        targets = []
        for index, row in enumerate(raw, 1):
            along, right, offset = map(float, (row["along_m"], row["starboard_m"], row["course_offset_deg"]))
            length, beam = float(row["length_m"]), float(row["beam_m"])
            if not all(map(math.isfinite, (along, right, offset, length, beam))):
                raise ValueError(f"{self.scenario_id}: nonfinite traffic geometry.")
            targets.append(SyntheticAISTarget(
                mmsi=980_000_000 + namespace * 100 + index,
                name=row["name"], callsign=f"N{namespace:04X}{index:02d}", ship_type=int(row["ship_type"]),
                initial_east_metres=along * along_east + right * right_east,
                initial_north_metres=along * along_north + right * right_north,
                speed_metres_per_second=float(row["speed_mps"]),
                course_degrees=(self.route_bearing_deg + offset) % 360.,
                length_metres=length, beam_metres=beam,
            ))
        return tuple(targets)

    def _ramps(self, rows):
        ramps = {}
        maximum = 0.
        for row in rows:
            index = int(row["ship_index"])
            start, stop, offset, speed = map(float, (row["start_s"], row["end_s"],
                                                     row["final_offset_deg"], row["final_speed_mps"]))
            if (not 0 <= index < len(self.targets) or index in ramps or speed < 0. or not 0. <= start < stop
                    or not all(map(math.isfinite, (start, stop, offset, speed)))):
                raise ValueError(f"{self.scenario_id}: malformed or duplicate synthetic velocity ramp.")
            course = (self.route_bearing_deg + offset) % 360.
            ramps[index] = SyntheticAISManoeuvre(start, stop, course, speed)
            ship = self.targets[index]
            vector0 = _vector(ship.course_degrees, ship.speed_metres_per_second)
            vector1 = _vector(course, speed)
            maximum = max(maximum, math.dist(vector0, vector1) / (stop - start))
        return tuple(ramps.get(index) for index in range(len(self.targets))), maximum

    def elapsed_seconds(self, when):
        return float((normalise_time(when) - self.start_utc) / np.timedelta64(1, "s"))

    def _at_epoch(self, elapsed_seconds):
        return tuple(target._at_epoch(*self.origin, elapsed_seconds, motion)
                     for target, motion in zip(self.targets, self.manoeuvres, strict=True))

    @lru_cache(maxsize=2048)
    def _emitted(self, epoch):
        return self._at_epoch(epoch)

    def reports_at(self, when):
        elapsed = self.elapsed_seconds(when)
        epoch = self.reporting.latest_epoch(elapsed)
        age = elapsed - epoch
        return tuple(replace(target, report_age_seconds=age) for target in self._emitted(epoch))

    def observe(self, state, when):
        instant = normalise_time(when)
        return self.archive.observe(state, instant, reports=self.reports_at(instant))

    def weather_history(self, latitude, longitude, when):
        return self.archive.weather_history(latitude, longitude, normalise_time(when))

    def chart_path_is_clear(self, path, observation=None):
        """Check each supplied (latitude, longitude) native integration point."""
        if self.chart_path is None:
            return True
        points = np.asarray(path, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all():
            return False
        chart, axes = _chart_grid(self.chart_path)
        indices = np.rint((points - axes[:, 0]) / axes[:, 1])
        if not np.all((indices >= 0) & (indices < np.asarray(chart.shape))):
            return False
        rows, columns = indices.astype(np.intp).T
        return bool(np.isfinite(chart.elevation[rows, columns]).all()
                    and not chart.occupancy[rows, columns].any())

    def truth_targets(self, when):
        """Return analytic current ship positions for offline swept auditing."""
        return self.truth(self.elapsed_seconds(when))

    def truth(self, elapsed_seconds):
        """Audit interface with floating elapsed seconds on the fixed traffic clock."""
        elapsed = float(elapsed_seconds)
        if not math.isfinite(elapsed):
            raise ValueError("Synthetic traffic truth requires finite elapsed seconds.")
        return self._at_epoch(elapsed)
