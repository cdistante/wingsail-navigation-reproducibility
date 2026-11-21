"""Plan attainable holding trims with retrospective marine archive look-ahead."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .drift_guidance import DriftGuidance, DriftMode, gate_velocity
from .fossen import Fossen3DOFSaildroneDynamics
from .scenario import Waypoint
from .world import MaritimeWorld, WorldObservation


@dataclass(frozen=True)
class HoldForecast:
    trim: DriftMode
    end_time_s: int
    end_range_m: float
    release_vmg_mps: float | None


@dataclass(frozen=True)
class _Polar:
    modes: tuple[DriftMode, ...]
    velocity: np.ndarray
    bows: np.ndarray
    flow: np.ndarray
    wind_from: float
    hull: np.ndarray

    @classmethod
    def from_modes(cls, observation: WorldObservation, modes: Iterable[DriftMode]) -> _Polar:
        assert observation.wind is not None and observation.current is not None
        valid = tuple(mode for mode in modes if np.isfinite(mode.velocity).all() and np.isfinite(mode.bow))
        velocity = np.array([mode.velocity for mode in valid], dtype=float).reshape(-1, 2)
        boundary = velocity
        if len(boundary) > 2:
            try:
                boundary = velocity[ConvexHull(velocity).vertices]
            except QhullError:
                pass
        flow = np.array([observation.current.u, observation.current.v])
        return cls(valid, velocity, np.array([mode.bow for mode in valid]), flow, observation.wind.direction, boundary)


@dataclass(frozen=True)
class _Paths:
    position: np.ndarray
    bow: np.ndarray
    action: np.ndarray
    turn_cost: np.ndarray

    def subset(self, selected) -> _Paths:
        return _Paths(self.position[selected], self.bow[selected], self.action[selected], self.turn_cost[selected])


def _signed_angle(bow, origin):
    return (bow - origin + 180.0) % 360.0 - 180.0


class ForecastHoldPlanner:
    """Select the first trim from a receding forecast of the holding phase."""

    interval_seconds = DriftGuidance.interval_seconds
    forecast_interval_seconds = 900
    horizon_seconds = 3 * 3600
    beam_width = 24

    def __init__(self, world: MaritimeWorld, dynamics: Fossen3DOFSaildroneDynamics, no_go_angle_degrees: float) -> None:
        self.world = world
        self._sampler = DriftGuidance(dynamics, no_go_angle_degrees)
        # Keep the full bow grid, including the close-hauled and beam-reach
        # alternatives. Sample five flap trims in each future weather block.
        self._sampler.flaps = self._sampler.flaps[[0, 2, 3, 4, 6]]
        self._goal: Waypoint | None = None
        self._window: int | None = None
        self._cache: dict[int, _Polar | None] = {}
        self._selected: HoldForecast | None = None
        self.telemetry: dict[str, str | float | bool | None] = {}

    def invalidate_selection(self) -> None:
        """Discard the chosen live trim while retaining the archive cache."""
        self._window = None
        self._selected = None
        self.telemetry = {}

    def _sample(self, observation: WorldObservation, goal: Waypoint, time_s: int) -> _Polar | None:
        if time_s in self._cache:
            return self._cache[time_s]
        weather, currents, waves = self.world.weather, self.world.currents, self.world.waves
        polar = None
        if weather is not None and currents is not None and waves is not None:
            when = np.datetime64(time_s, 's')
            latitude, longitude = goal.latitude, goal.longitude
            try:
                sample = replace(observation, time=when,
                                 wind=weather.sample(latitude, longitude, when),
                                 current=currents.sample(latitude, longitude, when, 0.0),
                                 wave=waves.sample(latitude, longitude, when))
                polar = _Polar.from_modes(sample, self._sampler._steady_modes(sample))
            except ValueError:
                pass  # Match the forecast tack planner's handling of unavailable archive windows.
        self._cache[time_s] = polar
        return polar

    @staticmethod
    def _range(paths: _Paths, target: np.ndarray) -> np.ndarray:
        return np.linalg.norm(target - paths.position, axis=-1)

    def _advance(self, paths: _Paths, polar: _Polar, duration: float, target: np.ndarray) -> _Paths:
        start_bows = paths.bow[:, None]
        next_bows = polar.bows[None, :]
        turn = np.abs(_signed_angle(next_bows, start_bows))
        cross_wind = (_signed_angle(start_bows, polar.wind_from) < 0) != (_signed_angle(next_bows, polar.wind_from) < 0)
        turn = np.minimum(duration, turn + np.where(cross_wind, 30.0, 0.0))
        position = (paths.position[:, None, :] + turn[..., None] * polar.flow
                    + (duration - turn)[..., None] * polar.velocity)
        shape = turn.shape
        first_action = np.where(paths.action[:, None] < 0, np.arange(shape[1]), paths.action[:, None])
        expanded = _Paths(position.reshape(-1, 2), np.broadcast_to(next_bows, shape).ravel(), first_action.ravel(),
                          (paths.turn_cost[:, None] + turn).ravel())
        # Avoid spending the beam on nearly identical displacements while
        # preserving separate options for approaching from different bows.
        ranges = self._range(expanded, target)
        order = np.lexsort((expanded.turn_cost, ranges))
        cells = np.column_stack((np.rint(expanded.position / 15.0), np.rint(expanded.bow / 10.0)))
        _, unique = np.unique(cells[order], axis=0, return_index=True)
        selected = order[np.sort(unique)[:self.beam_width]]
        return expanded.subset(selected)

    def _search(self, observation: WorldObservation, goal: Waypoint, time_s: int, target: np.ndarray,
                live: _Polar) -> HoldForecast | None:
        end = time_s + self.horizon_seconds
        following_window = (time_s // self.interval_seconds + 1) * self.interval_seconds
        paths = _Paths(np.zeros((1, 2)), np.array([observation.boat.heading]), np.array([-1]), np.zeros(1))
        paths = self._advance(paths, live, following_window - time_s, target)
        # Use the live estimate through the first steering window, then hold
        # each sampled quarter-hour estimate over a bounded forecast block.
        releases = []
        elapsed = following_window
        has_forecast = False
        while elapsed < end:
            stride = self.forecast_interval_seconds
            next_window = min(end, (elapsed // stride + 1) * stride)
            polar = self._sample(observation, goal, (elapsed // stride) * stride + stride // 2)
            if polar is None or not len(polar.modes):
                break
            has_forecast = True
            viable = np.array([gate_velocity(polar.hull, target - position) or 0.0 for position in paths.position])
            released = viable >= DriftGuidance.release_speed_mps
            if released.any():
                ranges = self._range(paths, target)
                for index in np.flatnonzero(released):
                    releases.append((float(ranges[index]), elapsed, float(paths.turn_cost[index]),
                                     int(paths.action[index]), float(viable[index])))
                paths = paths.subset(~released)
                if not len(paths.action):
                    break
            paths = self._advance(paths, polar, next_window - elapsed, target)
            elapsed = next_window
        if not has_forecast:
            return None
        if releases:
            distance, terminal, _, action, vmg = min(releases)
            return HoldForecast(live.modes[action], terminal, distance, vmg)
        ranges = self._range(paths, target)
        index = int(np.lexsort((paths.turn_cost, ranges))[0])
        return HoldForecast(live.modes[paths.action[index]], elapsed, float(ranges[index]), None)

    def reference(
        self, observation: WorldObservation, goal: Waypoint, target: np.ndarray, modes: Iterable[DriftMode], *,
        chart_allows: Callable[[DriftMode], bool],
    ) -> DriftMode | None:
        """Choose a sampled Fossen trim and recheck its immediate chart corridor."""
        time_s = int(observation.time.astype('datetime64[s]').astype(np.int64))
        window = time_s // self.interval_seconds
        if goal != self._goal:
            self._goal, self._window = goal, None
            self._cache.clear()
        if window != self._window:
            self._window = window
            self._cache = {when: stage for when, stage in self._cache.items()
                           if when >= time_s - self.forecast_interval_seconds}
            live = _Polar.from_modes(observation, (mode for mode in modes if chart_allows(mode)))
            self._selected = self._search(observation, goal, time_s, target, live) if len(live.modes) else None
            reason = 'archive forecast unavailable' if len(live.modes) else 'no chart-clear observed trim'
        else:
            reason = self.telemetry.get('forecast_hold_fallback_reason')
        self.telemetry = {'forecast_hold_window_utc': str(np.datetime64(window * self.interval_seconds, 's')) + 'Z'}
        selected = self._selected
        if selected is not None and chart_allows(selected.trim):
            self.telemetry.update(
                forecast_hold_source='archive forecast',
                forecast_hold_trim_bow_deg=selected.trim.bow,
                forecast_hold_trim_flap_deg=selected.trim.flap,
                forecast_hold_range_at_end_m=selected.end_range_m,
                forecast_hold_end_utc=str(np.datetime64(selected.end_time_s, 's')) + 'Z',
                forecast_hold_predicted_release=selected.release_vmg_mps is not None,
                forecast_hold_gate_vmg_mps=selected.release_vmg_mps,
            )
            return selected.trim
        self.telemetry.update(forecast_hold_source='classical fallback',
                              forecast_hold_fallback_reason='planned trim blocked' if selected is not None else reason)
        return None
