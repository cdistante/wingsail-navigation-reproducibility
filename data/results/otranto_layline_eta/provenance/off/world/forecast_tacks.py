"""Plan open-water tacks with time-varying archived marine conditions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import atan2, degrees, exp, hypot, radians
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import root

from ._geo import local_east_north_metres, offset_latitude_longitude
from .fossen import Fossen3DOFSaildroneDynamics
from .scenario import Waypoint
from .world import MaritimeWorld, WorldObservation

if TYPE_CHECKING:
    from .learned_tack_arrival import LearnedTackArrivalSelector
    from .learned_tacks import LearnedTackSelector


@dataclass(frozen=True)
class ForecastTackReference:
    course: float
    bow: float
    closest_metres: float
    tack_in_minutes: float | None


_Schedule = tuple[tuple[float, ...], int, int, int, float]


class ForecastTackPlanner:
    """Search single-leg and one-tack schedules, then replan from the actual boat.

    Each forecast interval uses a forward steady Fossen polar sampled at the
    active waypoint. The tactical planner still checks the immediate track,
    and MPC alone selects the actuators. This is an archive look-ahead.
    """

    interval_seconds = 120
    horizon_intervals = 24
    tack_seconds = 60.0

    def __init__(self, world: MaritimeWorld, dynamics: Fossen3DOFSaildroneDynamics,
                 no_go_angle: float, arrival_radius: float, *,
                 learned_selector: LearnedTackSelector | None = None,
                 arrival_selector: LearnedTackArrivalSelector | None = None,
                 refine_modes: bool = False, tracking_feedback: bool = False,
                 interval_seconds: int | None = None, gate_aim_fraction: float = 0.0,
                 gate_aim_upwind_angle: float = 180.0, maneuver_commit_seconds: float = 0.0) -> None:
        if learned_selector is not None and arrival_selector is not None:
            raise ValueError("Choose either the progress model or the arrival-time model.")
        if not np.isfinite(gate_aim_fraction) or not 0.0 <= gate_aim_fraction <= 0.5:
            raise ValueError("Gate aim fraction must be finite and between zero and one half.")
        if not np.isfinite(gate_aim_upwind_angle) or not 0.0 <= gate_aim_upwind_angle <= 180.0:
            raise ValueError("Gate aim upwind angle must be finite and between zero and 180 degrees.")
        if not np.isfinite(maneuver_commit_seconds) or maneuver_commit_seconds < 0.0:
            raise ValueError("Maneuver commitment must be finite and nonnegative.")
        if interval_seconds is not None:
            if (isinstance(interval_seconds, bool) or not isinstance(interval_seconds, int)
                    or interval_seconds < 60 or 2880 % interval_seconds):
                raise ValueError("Forecast interval must be an integer of at least 60 s dividing the 48-minute horizon.")
            self.interval_seconds = interval_seconds
            self.horizon_intervals = 2880 // interval_seconds
        self.world = world
        self.dynamics = dynamics
        self.arrival_radius = arrival_radius
        self.gate_aim_fraction = gate_aim_fraction
        self.gate_aim_upwind_angle = gate_aim_upwind_angle
        self.continuation_goal: Waypoint | None = None
        self.gate_telemetry: dict[str, str | float] = {}
        self.maneuver_commit_seconds = maneuver_commit_seconds
        self.maneuver_telemetry: dict[str, str | float | int] = {}
        self._commit_until = float('-inf')
        self._committed_mode: int | None = None
        self._committed_side: int | None = None
        self._last_command_bow: float | None = None
        self._last_command_side: int | None = None
        self.learned_selector = learned_selector
        self.arrival_selector = arrival_selector
        self.learned_selection: dict[str, str | float | int] = {}
        self.learned_selection_time: str | None = None
        offsets = tuple(min(no_go_angle + offset, 179.0) for offset in (0.5, 5, 15, 30, 50, 80, 115))
        if refine_modes:
            # Small close-hauled bow changes strongly affect ground laylines
            # in adverse current. Resolve this gap using actual equilibria,
            # retaining every original mode and the same no-go margin.
            offsets += tuple(np.linspace(offsets[0], offsets[1], 5)[1:-1])
        self.offsets = np.array(sorted({-angle for angle in offsets} | set(offsets)))
        self.tracking_feedback = tracking_feedback
        self.tracking_telemetry: dict[str, str | float | int] = {
            'forecast_tracking_feedback': int(tracking_feedback),
            'forecast_tracking_source': 'warming up' if tracking_feedback else 'disabled',
            'forecast_tracking_samples': 0,
        }
        # Wind-relative residuals belong to the controller, rather than a
        # waypoint. Retain both tack estimates when the active goal changes.
        self._tracking_residuals: dict[int, np.ndarray] = {}
        self._tracking_reference: float | None = None
        self._tracking_stable_since: float | None = None
        self._tracking_last_observation: float | None = None
        self._active_goal: Waypoint | None = None
        self._active_continuation: Waypoint | None = None
        self._polars: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._window: int | None = None
        self._choice: tuple[int, int, int, float] | None = None

    def suspend(self, reason: str = "nominal reference not applied") -> None:
        """Discard history tied to a nominal command that was not applied.

        Traffic avoidance, chart rejection and holding trims can replace the
        proposed sailing reference. A later resume must plan from the actual
        boat rather than execute an old tack or learn that override's velocity
        as the error of the nominal NMPC reference.
        """
        self._polars.clear()
        self._window = None
        self._choice = None
        self._tracking_residuals.clear()
        self._tracking_reference = None
        self._tracking_stable_since = None
        self._tracking_last_observation = None
        self._commit_until = -np.inf
        self._committed_mode = None
        self._committed_side = None
        self._last_command_bow = None
        self._last_command_side = None
        self.learned_selection = {}
        self.learned_selection_time = None
        self.tracking_telemetry = {
            'forecast_tracking_feedback': int(self.tracking_feedback),
            'forecast_tracking_source': 'suspended',
            'forecast_tracking_samples': 0,
            'forecast_suspension_reason': reason,
        }
        self.gate_telemetry = {}
        self.maneuver_telemetry = {}

    def reference(self, observation: WorldObservation, goal: Waypoint) -> ForecastTackReference | None:
        if observation.wind is None or observation.current is None:
            return None
        if self.tracking_telemetry.pop('forecast_suspension_reason', None) is not None:
            self.tracking_telemetry['forecast_tracking_source'] = (
                'warming up' if self.tracking_feedback else 'disabled'
            )
        second = observation.time.astype('datetime64[ns]').astype(np.int64) / 1e9
        self._maneuver_telemetry(second)
        if self.tracking_feedback:
            self._observe_tracking(observation, second)
        continuation = self.continuation_goal if self.gate_aim_fraction else None
        goal_changed = goal != self._active_goal
        if goal_changed or continuation != self._active_continuation:
            self._active_goal = goal
            self._active_continuation = continuation
            self._polars.clear()
            self._window = None
            self._choice = None

        window = int(second // self.interval_seconds)
        if self._window != window:
            previous = self._choice
            self._window = window
            self._polars = {slot: polar for slot, polar in self._polars.items() if slot >= window}
            first_mode = None
            if previous is not None and previous[2] == window and previous[0] != previous[1]:
                # Execute a scheduled switch before evaluating another one.
                first_mode = previous[1]
            first_side = None
            if self.maneuver_commit_seconds and second < self._commit_until:
                # A forecast boundary must not reverse an in-flight turn.
                # The side constraint also protects the fallback when the
                # exact committed mode loses its valid equilibrium.
                first_mode = self._committed_mode
                first_side = self._committed_side
            self._choice = self._plan(observation, goal, window, second,
                                      first_mode=first_mode, first_side=first_side)
        if self._choice is None or window not in self._polars:
            return None

        first, second_mode, switch, closest = self._choice
        modes, headings, _ = self._polars[window]
        mode = second_mode if window >= switch else first
        east, north = modes[mode]
        if not np.isfinite(east + north) or hypot(east, north) < 0.01:
            return None
        bow = float(headings[mode])
        if self.maneuver_commit_seconds:
            side = 1 if self.offsets[mode] >= 0.0 else -1
            side_change = self._last_command_side is not None and side != self._last_command_side
            new_large_turn = (
                abs((bow - observation.boat.heading + 180.0) % 360.0 - 180.0) > 60.0
                and (self._last_command_bow is None or goal_changed
                     or abs((bow - self._last_command_bow + 180.0) % 360.0 - 180.0) > 60.0)
            )
            if second >= self._commit_until and (side_change or new_large_turn):
                self._commit_until = second + self.maneuver_commit_seconds
                self._committed_mode, self._committed_side = mode, side
            self._last_command_bow, self._last_command_side = bow, side
            self._maneuver_telemetry(second)
        if self.tracking_feedback:
            if (self._tracking_reference is None or
                    abs((bow - self._tracking_reference + 180.0) % 360.0 - 180.0) > 3.0):
                self._tracking_stable_since = second
            self._tracking_reference = bow
        return ForecastTackReference(
            course=degrees(atan2(east, north)) % 360.0,
            bow=bow, closest_metres=closest,
            tack_in_minutes=(switch * self.interval_seconds - second) / 60 if switch > window else None,
        )

    def _plan(self, observation: WorldObservation, goal: Waypoint, window: int,
              second: float, *, first_mode: int | None = None,
              first_side: int | None = None) -> tuple[int, int, int, float] | None:
        use_learning = self.learned_selector is not None or self.arrival_selector is not None
        if use_learning:
            self.learned_selection_time = f'{observation.time}Z'
            self.learned_selection = {'learned_tack_selection': 'no forecast schedule'}
        velocity, current = [], []
        for slot in range(window, window + self.horizon_intervals):
            if slot not in self._polars:
                time = np.datetime64(slot * self.interval_seconds + self.interval_seconds // 2, 's')
                try:
                    sample = self._sample(observation, goal, time)
                except ValueError:
                    break
                self._polars[slot] = self._polar(sample)
            modes, _, drift = self._polars[slot]
            velocity.append(modes)
            current.append(drift)
        if not velocity:
            return None

        velocities, drift = np.asarray(velocity), np.asarray(current)
        if self.tracking_feedback and self._tracking_residuals:
            # Correct only the schedule model. The emitted bow/course retain
            # the original nominal polar, whose closed-loop error was observed.
            velocities = self._tracking_velocities(velocities, window)
        durations = np.full(len(velocities), float(self.interval_seconds))
        durations[0] -= second % self.interval_seconds
        interval_starts = np.r_[0.0, np.cumsum(durations[:-1])]
        valid = np.isfinite(velocities).all(axis=2)
        start_bow = self._polars[window][1]
        turn = np.abs((start_bow - observation.boat.heading + 180) % 360 - 180) > 60
        initial_delay = np.where(turn, self.tack_seconds, 0.0)
        if observation.boat.surge_speed is None:
            initial_delay = np.maximum(initial_delay, 30.0)
        target, planning_radius = self._gate_target(observation, goal)
        first_modes = np.flatnonzero(valid[0])
        if first_side is not None:
            first_modes = first_modes[self.offsets[first_modes] * first_side > 0.0]
        if first_mode is not None and first_mode in first_modes:
            first_modes = (first_mode,)
        best: _Schedule | None = None
        best_by_first: dict[int, _Schedule] = {}

        for first in first_modes:
            first_invalid = next((i for i, usable in enumerate(valid[:, first]) if not usable), len(valid))
            candidates = [(first, first_invalid, first_invalid)]
            for switch in range(1, min(first_invalid + 1, len(valid))):
                if interval_starts[switch] < initial_delay[first]:
                    continue
                if first_side is not None and interval_starts[switch] < self._commit_until - second:
                    continue
                opposite = (self.offsets * self.offsets[first] < 0) & valid[switch]
                for following in np.flatnonzero(opposite):
                    end = next((i for i in range(switch, len(valid)) if not valid[i, following]), len(valid))
                    candidates.append((following, switch, end))

            for following, switch, end in candidates:
                sailing = np.vstack((velocities[:switch, first], velocities[switch:end, following]))
                turn = np.clip(initial_delay[first] - interval_starts[:end], 0.0, durations[:end])
                turning_time = self.tack_seconds if following != first else 0.0
                if turning_time:
                    turn[switch] = turning_time
                sail_time = durations[:end] - turn
                # Score drift and forward sailing as distinct segments, including
                # when a maneuver begins near the end of a forecast interval.
                travel = np.stack((drift[:end] * turn[:, None], sailing * sail_time[:, None]), axis=1)
                segment_time = np.column_stack((turn, sail_time)).ravel()
                score, closest = self._score_path(
                    target, travel.reshape(-1, 2), segment_time, turning_time, len(valid) - end,
                    arrival_radius=planning_radius,
                )
                switch_window = window + switch if following != first else window
                schedule = (score, first, following, switch_window, closest)
                if best is None or score < best[0]:
                    best = schedule
                if use_learning and (
                    first not in best_by_first or score < best_by_first[first][0]
                ):
                    best_by_first[first] = schedule

        if self.learned_selector is not None and best is not None:
            best = self._choose_first_leg(
                observation, goal, best, best_by_first, velocities[0], start_bow,
                scheduled_switch=first_mode is not None,
            )
        if self.arrival_selector is not None and best is not None:
            best = self._choose_arrival_schedule(
                observation, goal, best, best_by_first, window, second,
                scheduled_switch=first_mode is not None,
            )

        return None if best is None else best[1:]

    def _maneuver_telemetry(self, second: float) -> None:
        if not self.maneuver_commit_seconds:
            self.maneuver_telemetry = {}
            return
        self.maneuver_telemetry = {
            'forecast_maneuver_commit_seconds': float(self.maneuver_commit_seconds),
            'forecast_maneuver_commit_active': int(second < self._commit_until),
            'forecast_maneuver_commit_remaining_s': max(0.0, self._commit_until - second),
        }
        if np.isfinite(self._commit_until):
            self.maneuver_telemetry['forecast_maneuver_commit_until_utc'] = (
                str(np.datetime64(int(self._commit_until), 's')) + 'Z'
            )
            self.maneuver_telemetry['forecast_maneuver_commit_side'] = int(self._committed_side)

    def _gate_target(self, observation: WorldObservation, goal: Waypoint) -> tuple[np.ndarray, float]:
        """Aim within the original gate toward the next leg, with a contained disk.

        This only changes nominal schedule scoring. The mission still tests
        capture against the original GPX center and its unchanged radius.
        Archive samples remain at that original center as well.
        """
        target = np.array(local_east_north_metres(
            goal.latitude, goal.longitude, observation.boat.latitude, observation.boat.longitude,
        ))
        self.gate_telemetry = {}
        following = self.continuation_goal
        if not self.gate_aim_fraction or following is None:
            return target, self.arrival_radius
        direction = np.array(local_east_north_metres(
            following.latitude, following.longitude, goal.latitude, goal.longitude,
        ))
        length = float(np.linalg.norm(direction))
        if length <= 1e-6:
            return target, self.arrival_radius
        wind = observation.wind
        bearing = degrees(atan2(direction[0], direction[1])) % 360.0
        angle = None if wind is None else abs((bearing - wind.direction + 180.0) % 360.0 - 180.0)
        reason = 'applied'
        if angle is None:
            reason = 'wind unavailable'
        elif angle > self.gate_aim_upwind_angle:
            reason = 'continuation outside upwind cone'
        self.gate_telemetry = {
            'forecast_gate_aim_fraction': float(self.gate_aim_fraction),
            'forecast_gate_aim_guard_reason': reason,
            'forecast_gate_aim_upwind_angle_limit_deg': float(self.gate_aim_upwind_angle),
        }
        if angle is not None:
            self.gate_telemetry['forecast_gate_continuation_wind_angle_deg'] = float(angle)
        if reason != 'applied':
            self.gate_telemetry.update(forecast_gate_aim_offset_m=0.0,
                                       forecast_gate_aim_radius_m=float(self.arrival_radius),
                                       forecast_gate_true_radius_m=float(self.arrival_radius))
            return target, self.arrival_radius
        offset = min(self.gate_aim_fraction * self.arrival_radius, 0.5 * length)
        displacement = direction * (offset / length)
        aim = target + displacement
        radius = self.arrival_radius - offset
        latitude, longitude = offset_latitude_longitude(
            goal.latitude, goal.longitude, *displacement,
        )
        self.gate_telemetry.update({
            'forecast_gate_aim_fraction': float(self.gate_aim_fraction),
            'forecast_gate_aim_offset_m': float(offset),
            'forecast_gate_aim_east_m': float(displacement[0]),
            'forecast_gate_aim_north_m': float(displacement[1]),
            'forecast_gate_aim_latitude': float(latitude),
            'forecast_gate_aim_longitude': float(longitude),
            'forecast_gate_aim_radius_m': float(radius),
            'forecast_gate_true_radius_m': float(self.arrival_radius),
        })
        return aim, radius

    def _observe_tracking(self, observation: WorldObservation, second: float) -> None:
        """Estimate settled NMPC velocity error on each close-hauled tack."""
        previous_time = self._tracking_last_observation
        self._tracking_last_observation = second
        if previous_time is not None and not 0.0 < second - previous_time <= 60.0:
            self._tracking_stable_since = second
        bow = self._tracking_reference
        boat, wind, current = observation.boat, observation.wind, observation.current
        assert wind is not None and current is not None
        if bow is None:
            return
        offset = (bow - wind.direction + 180.0) % 360.0 - 180.0
        bow_error = abs((boat.heading - bow + 180.0) % 360.0 - 180.0)
        if (boat.surge_speed is None or boat.surge_speed <= 0.2
                or abs(boat.yaw_rate) > radians(0.5) or bow_error > 10.0
                or not max(0.0, float(np.min(np.abs(self.offsets))) - 5.0) <= abs(offset) <= 60.0):
            self._tracking_stable_since = None
            self.tracking_telemetry['forecast_tracking_source'] = 'unsettled tracking'
            return
        if self._tracking_stable_since is None:
            self._tracking_stable_since = second
        if second - self._tracking_stable_since < 60.0:
            self.tracking_telemetry['forecast_tracking_source'] = 'settling'
            return
        modes, _, _ = steady_ground_polar(observation, self.dynamics, np.array([offset]))
        nominal = modes[0]
        actual = self.dynamics._body_to_earth(boat.heading) @ np.array([
            boat.surge_speed, boat.sway_speed,
        ]) + np.array([current.u, current.v])
        residual = actual - nominal
        if not np.isfinite(residual).all():
            self.tracking_telemetry['forecast_tracking_source'] = 'invalid local polar'
            return
        residual *= min(1.0, 0.15 / max(float(np.linalg.norm(residual)), 1e-12))
        rotation = self.dynamics._body_to_earth(wind.direction)
        wind_residual = rotation.T @ residual
        side = 1 if offset >= 0.0 else -1
        previous = self._tracking_residuals.get(side)
        elapsed = 0.0 if previous_time is None else max(0.0, min(second - previous_time, 60.0))
        alpha = 1.0 if previous is None else 1.0 - exp(-elapsed / 60.0)
        updated = wind_residual if previous is None else (1.0 - alpha) * previous + alpha * wind_residual
        self._tracking_residuals[side] = updated
        current_residual = rotation @ updated
        self.tracking_telemetry.update({
            'forecast_tracking_source': 'settled velocity residual',
            'forecast_tracking_samples': int(self.tracking_telemetry['forecast_tracking_samples']) + 1,
            'forecast_tracking_side': side,
            'forecast_tracking_residual_east_mps': float(current_residual[0]),
            'forecast_tracking_residual_north_mps': float(current_residual[1]),
        })

    def _tracking_velocities(self, velocities: np.ndarray, window: int) -> np.ndarray:
        """Rotate bounded measured errors into each future wind frame."""
        corrected = velocities.copy()
        for interval in range(len(velocities)):
            headings = self._polars[window + interval][1]
            for side, residual in self._tracking_residuals.items():
                eligible = (self.offsets * side > 0.0) & (np.abs(self.offsets) <= 60.0)
                for mode in np.flatnonzero(eligible):
                    wind_from = float(headings[mode] - self.offsets[mode])
                    correction = self.dynamics._body_to_earth(wind_from) @ residual
                    corrected[interval, mode] += correction
        self.tracking_telemetry['forecast_tracking_corrected_sides'] = len(self._tracking_residuals)
        return corrected

    def _choose_first_leg(
        self, observation: WorldObservation, goal: Waypoint, baseline: _Schedule,
        best_by_first: dict[int, _Schedule], modes: np.ndarray, headings: np.ndarray,
        *, scheduled_switch: bool,
    ) -> _Schedule:
        """Rank only comparable first legs and keep the physics-based schedule for each."""
        assert self.learned_selector is not None
        first = baseline[1]
        eligible = self._comparable_schedules(baseline, best_by_first, headings, scheduled_switch)
        details = self.learned_selection
        if len(eligible) < 2:
            return baseline

        def predict(mode: int) -> float | None:
            east, north = modes[mode]
            course = degrees(atan2(east, north)) % 360.0
            return self.learned_selector.predict_progress(observation, goal, course, float(headings[mode]))

        initial_progress = predict(first)
        if initial_progress is None:
            details['learned_tack_selection'] = 'classical input outside training support'
            return baseline
        selected, best_progress = baseline, initial_progress
        for schedule in eligible:
            if schedule[1] == first:
                continue
            progress = predict(schedule[1])
            if progress is not None and progress > best_progress:
                selected, best_progress = schedule, progress
        gain = best_progress - initial_progress
        details['learned_predicted_progress_gain_m'] = gain
        if gain < 2.0:
            details['learned_tack_selection'] = 'classical first leg retained'
            return baseline
        details['learned_tack_selection'] = 'learned first leg'
        details['learned_selected_bow_deg'] = float(headings[selected[1]])
        return selected

    def _comparable_schedules(
        self, baseline: _Schedule, best_by_first: dict[int, _Schedule], headings: np.ndarray,
        scheduled_switch: bool,
    ) -> list[_Schedule]:
        initial_score, first, *_ = baseline
        details = {
            'learned_tack_selection': 'no comparable first leg',
            'learned_classical_bow_deg': float(headings[first]),
            'learned_selected_bow_deg': float(headings[first]),
        }
        self.learned_selection = details
        if scheduled_switch:
            details['learned_tack_selection'] = 'scheduled switch'
            return []
        if initial_score[0] == 2.0:
            details['learned_tack_selection'] = 'classical no-progress forecast'
            return []

        # Keep the same outcome class from the 48-minute physical forecast.
        # Its second score component is time to arrival or closest approach.
        tolerance = 120.0 if initial_score[0] == 0.0 else 10.0
        eligible = [schedule for schedule in best_by_first.values()
                    if schedule[0][0] == initial_score[0]
                    and schedule[0][1] <= initial_score[1] + tolerance]
        details['learned_comparable_first_legs'] = len(eligible)
        return eligible

    def _choose_arrival_schedule(
        self, observation: WorldObservation, goal: Waypoint, baseline: _Schedule,
        best_by_first: dict[int, _Schedule], window: int, second: float, *, scheduled_switch: bool,
    ) -> _Schedule:
        """Re-rank by expected time spent before capture or the rollout limit."""
        assert self.arrival_selector is not None
        eligible = self._comparable_schedules(baseline, best_by_first, self._polars[window][1], scheduled_switch)
        details = self.learned_selection
        if len(eligible) < 2:
            return baseline
        horizon = float(self.horizon_intervals * self.interval_seconds)

        def bearings(slot: int, mode: int) -> tuple[float, float]:
            modes, headings, _ = self._polars[slot]
            east, north = modes[mode]
            return degrees(atan2(east, north)) % 360.0, float(headings[mode])

        def rank(schedule: _Schedule) -> tuple[float, float] | None:
            _, first, following, switch, _ = schedule
            first_course, first_bow = bearings(window, first)
            next_course = next_bow = switch_seconds = None
            if following != first:
                next_course, next_bow = bearings(switch, following)
                switch_seconds = switch * self.interval_seconds - second
            prediction = self.arrival_selector.predict_arrival(
                observation, goal, first_course, first_bow, next_course, next_bow,
                switch_seconds, self.arrival_radius, horizon,
            )
            if prediction is None:
                return None
            probability = prediction.capture_probability
            capped_time = probability * prediction.arrival_seconds + (1.0 - probability) * horizon
            return capped_time, probability

        classical = rank(baseline)
        if classical is None:
            details['learned_tack_selection'] = 'classical input outside training support'
            return baseline
        initial_time, initial_probability = classical
        selected, best_time, best_probability = baseline, initial_time, initial_probability
        details['learned_classical_capped_arrival_s'] = initial_time
        details['learned_classical_capture_probability'] = initial_probability
        for schedule in eligible:
            if schedule[1] == baseline[1]:
                continue
            alternative = rank(schedule)
            if alternative is None:
                continue
            time, probability = alternative
            if probability >= max(0.5, initial_probability - 0.05) and time < best_time:
                selected, best_time, best_probability = schedule, time, probability
        gain = initial_time - best_time
        details['learned_predicted_arrival_gain_s'] = gain
        if gain < 30.0:
            details['learned_tack_selection'] = 'classical first leg retained'
            return baseline
        details['learned_tack_selection'] = 'learned first leg'
        details['learned_selected_bow_deg'] = bearings(window, selected[1])[1]
        details['learned_selected_capped_arrival_s'] = best_time
        details['learned_selected_capture_probability'] = best_probability
        return selected

    def _score_path(self, target: np.ndarray, travel: np.ndarray, durations: np.ndarray,
                    turning_time: float, unavailable_intervals: int, *,
                    arrival_radius: float | None = None) -> tuple[tuple[float, ...], float]:
        radius = self.arrival_radius if arrival_radius is None else arrival_radius
        origins = np.vstack((np.zeros(2), np.cumsum(travel[:-1], axis=0)))
        to_target = target - origins
        length_squared = np.maximum(np.sum(travel * travel, axis=1), 1e-12)
        projection = np.sum(to_target * travel, axis=1) / length_squared
        fraction = np.clip(projection, 0.0, 1.0)
        distances = np.linalg.norm(to_target - fraction[:, None] * travel, axis=1)
        closest_index = np.argmin(distances)
        closest = float(distances[closest_index])
        hits = np.flatnonzero(distances <= radius)
        if len(hits):
            index = hits[0]
            across = to_target[index] - projection[index] * travel[index]
            clearance_squared = radius**2 - across @ across
            entry = projection[index] - np.sqrt(max(0.0, clearance_squared) / length_squared[index])
            arrival_time = durations[:index].sum() + np.clip(entry, 0.0, 1.0) * durations[index]
            return (0.0, arrival_time, turning_time, closest), closest
        if closest < np.linalg.norm(target) - 2.0:
            closest_time = durations[:closest_index].sum() + fraction[closest_index] * durations[closest_index]
            return (1.0, closest, turning_time, closest_time), closest
        # Prefer a complete horizon before comparing its final distance.
        final_distance = np.linalg.norm(target - origins[-1] - travel[-1])
        return (2.0, float(unavailable_intervals), final_distance, turning_time), closest

    def _sample(self, observation: WorldObservation, goal: Waypoint, time: np.datetime64) -> WorldObservation:
        weather, currents, waves = self.world.weather, self.world.currents, self.world.waves
        assert weather is not None and currents is not None and waves is not None
        latitude, longitude = goal.latitude, goal.longitude
        return replace(
            observation, time=time, wind=weather.sample(latitude, longitude, time),
            current=currents.sample(latitude, longitude, time, 0.0),
            wave=waves.sample(latitude, longitude, time),
        )

    def _polar(self, observation: WorldObservation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return steady_ground_polar(observation, self.dynamics, self.offsets)


def steady_ground_polar(
    observation: WorldObservation, dynamics: Fossen3DOFSaildroneDynamics, offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve forward steady Fossen velocities over ground for true-wind bow offsets."""
    assert observation.wind is not None and observation.current is not None
    p = dynamics.parameters
    drift = np.array([observation.current.u, observation.current.v])
    headings = (observation.wind.direction + offsets) % 360
    velocities = np.full((len(headings), 2), np.nan)
    for index, heading in enumerate(headings):
        rotation = dynamics._body_to_earth(heading)
        best_surge = 0.05
        for flap in (p.flap_max, p.flap_min, 0.0):
            trim = steady_sailing_trim(observation, dynamics, heading, flap, minimum_surge=best_surge)
            if trim is not None:
                water_velocity, _ = trim
                best_surge = water_velocity[0]
                velocities[index] = rotation @ water_velocity + drift
    return velocities, headings, drift


def steady_sailing_trim(
    observation: WorldObservation, dynamics: Fossen3DOFSaildroneDynamics,
    heading: float, flap: float, *, minimum_surge: float = 0.05,
) -> tuple[np.ndarray, float] | None:
    """Return the forward equilibrium and bounded rudder for this bow and flap."""
    p = dynamics.parameters
    rudder_pressure = 0.5 * p.water_density * p.rudder_area

    def balance(velocity: np.ndarray) -> tuple[np.ndarray, float]:
        surge, sway = velocity
        nu = np.array([surge, sway, 0.0])
        wing = dynamics._wing_load(nu, heading, flap, observation)
        residual = wing - dynamics._coriolis(nu) @ nu - dynamics._damping(nu, observation) @ nu
        # Derive the rudder required for zero residual yaw moment.
        pressure = max(rudder_pressure * (surge * surge + sway * sway), 1e-6)
        lift = -residual[2] / (p.rudder_longitudinal_position * pressure)
        rudder = degrees(atan2(sway, max(abs(surge), 0.05)) + lift / p.rudder_lift_slope)
        drag = pressure * (p.rudder_profile_drag_coefficient + p.rudder_induced_drag_factor * lift * lift)
        return residual[:2] + np.array([-drag, pressure * lift]), rudder

    fit = root(lambda uv: balance(uv)[0], (0.5, 0.0))
    surge, sway = fit.x
    residual, rudder = balance(fit.x)
    if (surge <= minimum_surge or np.max(np.abs(residual)) > 1e-4
            or not p.rudder_min <= rudder <= p.rudder_max):
        return None
    nu = np.array([surge, sway, 0.0])
    # Check the actual clipped rudder model before keeping a solution.
    actual = (dynamics._wing_load(nu, heading, flap, observation) + dynamics._rudder_load(nu, rudder)
              - dynamics._coriolis(nu) @ nu - dynamics._damping(nu, observation) @ nu)
    if np.max(np.abs(actual)) <= 1e-3:
        return fit.x, rudder
    return None
