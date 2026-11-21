"""Limit displacement during winds that cannot support a waypoint approach."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import atan2, degrees

import numpy as np

from ._geo import local_east_north_metres
from .forecast_tacks import steady_sailing_trim
from .fossen import Fossen3DOFSaildroneDynamics
from .scenario import Waypoint
from .world import WorldObservation


@dataclass(frozen=True)
class DriftMode:
    velocity: np.ndarray
    bow: float
    rudder: float
    flap: float

    @property
    def course(self) -> float:
        east, north = self.velocity
        return degrees(atan2(east, north)) % 360.0


def gate_velocity(velocities: np.ndarray, target: np.ndarray) -> float | None:
    """Best velocity on the target axis from at most two steady modes."""
    distance = np.linalg.norm(target)
    if distance <= 1e-8 or not len(velocities):
        return None
    axis = target / distance
    along = velocities @ axis
    across = velocities @ np.array([axis[1], -axis[0]])
    aligned = along[np.abs(across) < 1e-8]
    left, right = across < -1e-8, across > 1e-8
    if np.any(left) and np.any(right):
        left_cross, right_cross = across[left, None], across[right]
        weights = right_cross / (right_cross - left_cross)
        mixed = weights * along[left, None] + (1.0 - weights) * along[right]
        aligned = np.append(aligned, np.max(mixed))
    return float(aligned.max()) if aligned.size else None


class DriftGuidance:
    """Monitor sailability and rank real sailing modes around a holding point."""

    interval_seconds = 300
    displacement_horizon_seconds = 600.0
    enter_speed_mps = 0.01
    release_speed_mps = 0.04

    def __init__(self, dynamics: Fossen3DOFSaildroneDynamics, no_go_angle: float) -> None:
        self.dynamics = dynamics
        angles = np.arange(no_go_angle + 0.5, 180.0, 7.5)
        self.offsets = np.r_[-angles, angles]
        p = dynamics.parameters
        self.flaps = np.linspace(p.flap_min, p.flap_max, 7)
        self.goal: Waypoint | None = None
        self.anchor: Waypoint | None = None
        self.mode: str | None = None
        self.released = False
        self.gate_vmg: float | None = None
        self._window: int | None = None
        self._modes: tuple[DriftMode, ...] = ()
        self._velocities = np.empty((0, 2))

    def update(
        self, observation: WorldObservation, goal: Waypoint, target: np.ndarray, *,
        choose_hold: Callable[[bool, bool], bool] | None = None,
    ) -> None:
        """Update the holding state at each new weather window or waypoint."""
        window = int(observation.time.astype("datetime64[s]").astype(np.int64)) // self.interval_seconds
        new_goal = self.goal != goal
        if new_goal:
            self.goal, self.anchor, self.mode = goal, None, None
        if self._window == window and not new_goal:
            return
        if self._window != window:
            self._window = window
            self._modes = tuple(self._steady_modes(observation))
            self._velocities = np.asarray([mode.velocity for mode in self._modes], dtype=float).reshape(-1, 2)
        self.gate_vmg = gate_velocity(self._velocities, target)
        was_holding = self.mode == "holding"
        threshold = self.release_speed_mps if was_holding else self.enter_speed_mps
        hold = self.gate_vmg is None or self.gate_vmg < threshold
        if choose_hold is not None:
            hold = choose_hold(was_holding, hold)
        if not self._modes:
            return
        if hold and not was_holding:
            self.anchor = Waypoint(observation.boat.latitude, observation.boat.longitude)
            self.mode = "holding"
        elif not hold and was_holding:
            self.mode = "recovering"
            self.released = True

    def suspend(self) -> None:
        """Invalidate the holding point when traffic guidance takes over."""
        self.anchor = None
        self._window = None
        if self.mode == "holding":
            self.mode = None

    def distance_from_anchor(self, observation: WorldObservation) -> float | None:
        if self.anchor is None:
            return None
        boat = observation.boat
        return float(np.linalg.norm(local_east_north_metres(
            boat.latitude, boat.longitude, self.anchor.latitude, self.anchor.longitude,
        )))

    def ranked_modes(self, observation: WorldObservation, *, anchor: Waypoint | None = None):
        """Consider drift during the turn and then displacement from the anchor."""
        anchor = self.anchor if anchor is None else anchor
        if anchor is None or not self._modes:
            return ()
        boat, current = observation.boat, observation.current
        assert current is not None
        displacement = np.asarray(local_east_north_metres(
            boat.latitude, boat.longitude, anchor.latitude, anchor.longitude,
        ))
        bows = np.asarray([mode.bow for mode in self._modes])
        # About one second per degree is the turn estimate used for the
        # ordinary ground layline planner too.
        turn = np.abs((bows - boat.heading + 180.0) % 360.0 - 180.0)
        ends = (
            displacement + turn[:, None] * np.array([current.u, current.v])
            + self._velocities * (self.displacement_horizon_seconds - turn[:, None])
        )
        order = np.lexsort((turn, np.sum(ends**2, axis=1)))
        return (self._modes[index] for index in order)

    def _steady_modes(self, observation: WorldObservation):
        wind, current = observation.wind, observation.current
        assert wind is not None and current is not None
        drift = np.array([current.u, current.v])
        for bow in (wind.direction + self.offsets) % 360.0:
            rotation = self.dynamics._body_to_earth(bow)
            for flap in self.flaps:
                candidate = steady_sailing_trim(observation, self.dynamics, bow, flap)
                if candidate is not None:
                    water_velocity, rudder = candidate
                    yield DriftMode(rotation @ water_velocity + drift, float(bow), rudder, float(flap))
