"""Replay the same planned tack references across controller comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, isfinite, radians, sin
from pathlib import Path

import numpy as np

from ._geo import local_east_north_metres, offset_latitude_longitude
from .scenario import Route, Waypoint


@dataclass(frozen=True)
class TackPlanStep:
    start: Waypoint
    target: Waypoint
    course: float
    bow: float | None


class TackPlanReplay:
    """Index recorded guidance by active waypoint and control updates on its leg."""

    def __init__(self, path: Path, route: Route, departure: np.datetime64, dt: float,
                 arrival_radius: float = 50.0) -> None:
        from .navigation_replay import load_navigation_replay

        replay = load_navigation_replay(path)
        if replay.route != route.waypoints:
            raise ValueError("The tack plan replay must contain the same ordered route.")
        if replay.frames[0].observation.time != departure:
            raise ValueError("The tack plan replay must start at the same departure time.")
        self._legs: dict[tuple[str, str], list[TackPlanStep]] = {}
        waypoints = {point.name: point for point in route.waypoints}
        if len(waypoints) != len(route.waypoints):
            raise ValueError("Tack plan replay requires unique route waypoint names.")

        step_time = np.timedelta64(int(round(dt * 1_000_000_000)), "ns")
        direct_starts: dict[tuple[str, str], Waypoint] = {}
        for step, (previous, frame) in enumerate(zip(replay.frames, replay.frames[1:]), 1):
            if frame.observation.time != departure + step * step_time:
                raise ValueError("The tack plan replay must use the same controller update interval.")
            navigation = frame.navigation
            name = navigation["waypoint"]
            label = navigation["phase"], name
            course = navigation.get("heading_reference_deg")
            bow = navigation.get("bow_reference_deg")
            if course is None or not isfinite(course) or (bow is not None and not isfinite(bow)):
                raise ValueError(f"Invalid tack reference at replay step {step}.")
            start = Waypoint(previous.state.latitude, previous.state.longitude)
            waypoint = waypoints[name]
            east, north = local_east_north_metres(
                waypoint.latitude, waypoint.longitude, start.latitude, start.longitude,
            )
            bearing = degrees(atan2(east, north)) % 360.0
            if bow is None and abs((course - bearing + 180.0) % 360.0 - 180.0) < 0.05:
                # Exact geographic targets use a fixed line through the start
                # of the direct segment. A tack reference is a short track ray.
                start = direct_starts.setdefault(label, start)
                target = waypoint
            else:
                direct_starts.pop(label, None)
                angle = radians(course)
                tangent_east, tangent_north = sin(angle), cos(angle)
                along = east * tangent_east + north * tangent_north
                across = east * tangent_north - north * tangent_east
                length = 150.0
                if 0.0 < along < length and abs(across) < arrival_radius:
                    # End at the point on the recorded course closest to the
                    # waypoint whenever this short segment reaches the gate.
                    length = along
                target = Waypoint(*offset_latitude_longitude(
                    start.latitude, start.longitude, length * tangent_east, length * tangent_north,
                ))
            self._legs.setdefault(label, []).append(TackPlanStep(start, target, course, bow))

    def reference(self, phase: str, name: str, leg_step: int) -> TackPlanStep | None:
        leg = self._legs.get((phase, name))
        if leg:
            # Boat position still gates the next waypoint. Continue the last
            # reference if the controller takes longer than the source leg.
            return leg[min(leg_step, len(leg) - 1)]
        return None
