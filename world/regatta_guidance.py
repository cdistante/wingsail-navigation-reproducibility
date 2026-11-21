"""Choose an open-water trim by predicted arrival at the active mark."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .forecast_hold import ForecastHoldPlanner, HoldForecast, _Paths, _signed_angle


@dataclass(frozen=True)
class MarkForecast(HoldForecast):
    """A trim and the first simulated entry time, when the search reaches the mark."""
    arrival_utc_seconds: float | None = None


def _entry_fraction(origin, displacement, target, radius):
    """First intersection with a circular mark along each linear motion segment."""
    relative = origin - target
    radius_error = np.sum(relative * relative, axis=-1) - radius ** 2
    length = np.sum(displacement * displacement, axis=-1)
    projection = np.sum(relative * displacement, axis=-1)
    discriminant = projection ** 2 - length * radius_error
    with np.errstate(divide="ignore", invalid="ignore"):
        entry = (-projection - np.sqrt(np.maximum(discriminant, 0.0))) / length
    valid = (length > 1e-10) & (discriminant >= 0.0) & (entry >= 0.0) & (entry <= 1.0)
    return np.where(radius_error <= 0.0, 0.0, np.where(valid, entry, np.inf))


class RegattaHoldPlanner(ForecastHoldPlanner):
    """Search for a time-to-mark trim while the short-range planner awaits sailability.

    The open-water forecast is a ranking heuristic. The chart check in the
    parent planner and the runner's native applied AIS and chart checks are
    still responsible for admissibility.
    """

    horizon_seconds = 10 * 3600
    beam_width = 40

    def __init__(self, world, dynamics, no_go_angle_degrees, arrival_radius):
        super().__init__(world, dynamics, no_go_angle_degrees)
        self.arrival_radius = arrival_radius
        # Retain both close-hauled, reaching and downwind modes. The live
        # catalog still comes from the ordinary drift guidance at full density.
        angles = np.arange(no_go_angle_degrees + .5, 180., 10.)
        self._sampler.offsets = np.r_[-angles, angles]
        self._sampler.flaps = self._sampler.flaps[[0, 2, 4]]
        self._last_search_seconds = None

    def _expand(self, paths, polar, seconds, target):
        bows = polar.bows[None, :]
        current_bows = paths.bow[:, None]
        heading_change = _signed_angle(bows, current_bows)
        cross_wind = ((_signed_angle(current_bows, polar.wind_from) < 0)
                      != (_signed_angle(bows, polar.wind_from) < 0))
        turn_required = np.abs(heading_change) + 30.0 * cross_wind
        turn = np.minimum(seconds, turn_required)
        turn_fraction = np.divide(turn, turn_required, out=np.ones_like(turn), where=turn_required > 0.0)
        reached_bow = (current_bows + turn_fraction * heading_change) % 360.0
        turned = paths.position[:, None, :] + turn[..., None] * polar.flow
        travel = (seconds - turn)[..., None] * polar.velocity
        position = turned + travel
        first = np.where(paths.action[:, None] < 0, np.arange(len(polar.modes)), paths.action[:, None])
        turn_cost = paths.turn_cost[:, None] + turn

        turning_hit = _entry_fraction(paths.position[:, None, :], turn[..., None] * polar.flow,
                                      target, self.arrival_radius)
        sailing_hit = _entry_fraction(turned, travel, target, self.arrival_radius)
        with np.errstate(invalid="ignore"):
            elapsed = np.minimum(np.where(np.isfinite(turning_hit), turning_hit * turn, np.inf),
                                 np.where(np.isfinite(sailing_hit), turn + sailing_hit * (seconds - turn), np.inf))
        if np.isfinite(elapsed).any():
            best = int(np.lexsort((turn_cost.ravel(), elapsed.ravel()))[0])
            return None, (int(first.ravel()[best]), float(elapsed.ravel()[best]))

        expanded = _Paths(position.reshape(-1, 2), reached_bow.ravel(),
                          first.ravel(), turn_cost.ravel())
        # Keep multiple approaches by deduplicating space and bow together.
        # Future forces can favor a path approaching the mark from either side.
        range_left = self._range(expanded, target)
        order = np.lexsort((expanded.turn_cost, range_left))
        cells = np.column_stack((np.rint(expanded.position / 30.0), np.rint(expanded.bow / 15.0)))
        _, unique = np.unique(cells[order], axis=0, return_index=True)
        return expanded.subset(order[np.sort(unique)[:self.beam_width]]), None

    def _search(self, observation, goal, time_s, target, live):
        started = perf_counter()
        try:
            return self._race_search(observation, goal, time_s, target, live)
        finally:
            self._last_search_seconds = perf_counter() - started

    def _race_search(self, observation, goal, time_s, target, live):
        end = time_s + self.horizon_seconds
        following = (time_s // self.interval_seconds + 1) * self.interval_seconds
        paths = _Paths(np.zeros((1, 2)), np.array([observation.boat.heading]),
                       np.array([-1]), np.zeros(1))
        paths, hit = self._expand(paths, live, following - time_s, target)
        if hit is not None:
            first, delay = hit
            arrival = time_s + delay
            return MarkForecast(live.modes[first], int(np.ceil(arrival)), self.arrival_radius, None, arrival)
        has_forecast = False
        elapsed = following
        while elapsed < end:
            step = self.forecast_interval_seconds
            next_window = min(end, (elapsed // step + 1) * step)
            polar = self._sample(observation, goal, (elapsed // step) * step + step // 2)
            if polar is None or not len(polar.modes):
                break
            has_forecast = True
            paths, hit = self._expand(paths, polar, next_window - elapsed, target)
            if hit is not None:
                first, delay = hit
                arrival = elapsed + delay
                return MarkForecast(live.modes[first], int(np.ceil(arrival)), self.arrival_radius, None, arrival)
            elapsed = next_window
        if not has_forecast:
            return None
        ranges = self._range(paths, target)
        best = int(np.lexsort((paths.turn_cost, ranges))[0])
        return MarkForecast(live.modes[paths.action[best]], elapsed, float(ranges[best]), None)

    def reference(self, observation, goal, target, modes, *, chart_allows):
        trim = super().reference(observation, goal, target, modes, chart_allows=chart_allows)
        if trim is not None and self._selected is not None:
            arrival = self._selected.arrival_utc_seconds
            current = float(observation.time.astype('datetime64[ns]').astype(np.int64)) / 1e9
            self.telemetry.update(
                race_guidance='earliest forecast mark entry',
                race_predicted_time_to_mark_s=None if arrival is None else max(0.0, arrival - current),
                race_forecast_search_wall_s=self._last_search_seconds,
                race_forecast_reaches_mark=arrival is not None,
            )
        return trim
