"""Compare sailing out of hold now with one more hold interval."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .drift_guidance import DriftMode
from .forecast_hold import _Paths, _Polar, _signed_angle
from .regatta_guidance import RegattaHoldPlanner


@dataclass(frozen=True)
class ForecastRelease:
    trim: DriftMode
    time_s: int
    wait_s: int
    sail_eta_s: float
    hold_eta_s: float
    sail_range_m: float
    hold_range_m: float


class ForecastETARelease(RegattaHoldPlanner):
    """Rank two first-window decisions with a common time-to-mark forecast."""

    horizon_seconds = 12 * 3600
    forecast_interval_seconds = 30 * 60
    beam_width = 24
    minimum_gain_seconds = 5.

    def __init__(self, world, dynamics, no_go_angle_degrees, arrival_radius):
        super().__init__(world, dynamics, no_go_angle_degrees, arrival_radius)
        self._release_key = None
        self._release_plan: ForecastRelease | None = None

    def invalidate_selection(self) -> None:
        super().invalidate_selection()
        self._release_key = None
        self._release_plan = None

    @staticmethod
    def _window_ranges(observation, target, polar, seconds):
        bow = observation.boat.heading
        same_side = (_signed_angle(bow, polar.wind_from) < 0.) == (
            _signed_angle(polar.bows, polar.wind_from) < 0.
        )
        turn = np.minimum(seconds, np.abs(_signed_angle(polar.bows, bow)) + 30. * ~same_side)
        endpoints = turn[:, None] * polar.flow + (seconds - turn)[:, None] * polar.velocity
        return np.linalg.norm(target - endpoints, axis=1)

    def _first_arrival(self, observation, goal, target, live, time_s, first_seconds):
        paths = _Paths(np.zeros((1, 2)), np.array([observation.boat.heading]),
                       np.array([-1]), np.zeros(1))
        paths, hit = self._expand(paths, live, first_seconds, target)
        if hit is not None:
            return hit
        end = time_s + self.horizon_seconds
        stamp = time_s + first_seconds
        stride = self.forecast_interval_seconds
        while stamp < end:
            slot = stamp // stride * stride
            polar = self._sample(observation, goal, slot + stride // 2)
            if polar is None or not len(polar.modes):
                return None
            next_stamp = min(slot + stride, end)
            paths, hit = self._expand(paths, polar, next_stamp - stamp, target)
            if hit is not None:
                return hit[0], stamp - time_s + hit[1]
            stamp = next_stamp
        return None

    def release_reference(self, observation, goal, target, hold, modes, *, chart_allows):
        """Use the chosen trim only with a faster finite ETA and less first-window drift."""
        time_s = int(observation.time.astype('datetime64[s]').astype(np.int64))
        window = time_s // self.interval_seconds
        key = goal, window, hold.bow, hold.flap
        if self._release_key != key:
            if self._goal != goal:
                self._goal = goal
                self._cache.clear()
            self._cache = {stamp: polar for stamp, polar in self._cache.items()
                           if stamp >= time_s - self.forecast_interval_seconds}
            self._release_key = key
            self._release_plan = self._compare(observation, goal, target, hold, modes, time_s, chart_allows)
        self.telemetry = {}
        plan = self._release_plan
        if plan is None or not chart_allows(plan.trim):
            return None
        self.telemetry = {
            "forecast_release_policy": "forecast_eta",
            "forecast_release_decision": "release",
            "forecast_release_plan_utc": str(np.datetime64(plan.time_s, 's')) + "Z",
            "forecast_release_wait_s": plan.wait_s,
            "forecast_release_horizon_s": self.horizon_seconds,
            "forecast_release_gate_radius_m": self.arrival_radius,
            "forecast_release_now_eta_s": plan.sail_eta_s,
            "forecast_release_wait_eta_s": plan.hold_eta_s,
            "forecast_release_eta_gain_s": plan.hold_eta_s - plan.sail_eta_s,
            "forecast_release_now_range_m": plan.sail_range_m,
            "forecast_release_wait_range_m": plan.hold_range_m,
        }
        return plan.trim

    def _compare(self, observation, goal, target, hold, modes, time_s, chart_allows):
        duration = self.interval_seconds - time_s % self.interval_seconds
        kept = _Polar.from_modes(observation, (hold,))
        if not kept.modes:
            return None
        hold_range = float(self._window_ranges(observation, target, kept, duration)[0])
        possible = _Polar.from_modes(observation, (mode for mode in modes if chart_allows(mode)))
        ranges = self._window_ranges(observation, target, possible, duration)
        alternatives = _Polar.from_modes(
            observation, (mode for mode, distance in zip(possible.modes, ranges, strict=True)
                          if distance < hold_range - 1e-6),
        )
        if not alternatives.modes:
            return None
        waiting = self._first_arrival(observation, goal, target, kept, time_s, duration)
        if waiting is None:
            return None
        leaving = self._first_arrival(observation, goal, target, alternatives, time_s, duration)
        if leaving is None or waiting[1] - leaving[1] <= self.minimum_gain_seconds:
            return None
        index, arrival = leaving
        selected = alternatives.modes[index]
        first_range = self._window_ranges(observation, target, alternatives, duration)[index]
        return ForecastRelease(selected, time_s, duration, arrival, waiting[1], float(first_range), hold_range)
