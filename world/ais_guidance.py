"""Tactical courses and bounded Fossen fallback for time-indexed AIS traffic."""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, cos, radians, sin

import numpy as np

from ._geo import local_east_north_metres, offset_latitude_longitude
from .ais_avoidance import AISPrediction, AISRolloutAssessment
from .boat import BoatAction, BoatState
from .fossen import Fossen3DOFSaildroneDynamics
from .legacy_planner_adapter import LegacyPlannerAdapter
from .mpc import SaildroneMPC
from .occupancy import LocalGrid
from .scenario import Waypoint
from .world import WorldObservation

GUIDANCE_HORIZON_SECONDS = 180.0
_GUIDANCE_TIMES = np.arange(0.0, GUIDANCE_HORIZON_SECONDS + 5.0, 5.0)
_CLEARANCE_RESERVE_METRES = 8.0


def _signed_angle(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _blocked_segments(grid: LocalGrid, blocked: np.ndarray, positions: np.ndarray) -> bool:
    centre = (grid.cells - 1) / 2.0
    cells = np.column_stack((centre - positions[:, 1] / grid.resolution,
                             centre + positions[:, 0] / grid.resolution))
    if np.any(cells < -0.5) or np.any(cells > grid.cells - 0.5):
        return True
    if not np.any(blocked):
        return False
    return any(not LegacyPlannerAdapter._segment_is_clear(blocked, start, end)
               for start, end in zip(cells[:-1], cells[1:]))


@dataclass(frozen=True)
class AISReference:
    heading: float | None
    waypoint: Waypoint | None = None
    predicted_margin_metres: float | None = None
    escape_heading: float | None = None


@dataclass
class AISGuidance:
    """Screen sailing courses on a timed lookahead before asking the actuator solver."""

    no_go_angle_degrees: float
    avoidance_heading: float | None = None

    def choose_reference(
        self,
        observation: WorldObservation,
        grid: LocalGrid,
        blocked: np.ndarray,
        traffic: AISPrediction,
        nominal_heading: float | None,
        prefer_starboard: bool,
    ) -> AISReference:
        if nominal_heading is None:
            return AISReference(None)
        if not traffic.targets:
            self.avoidance_heading = None
            return AISReference(nominal_heading)
        nominal = self._course_preview(observation, nominal_heading)
        margin = traffic.assess_local_rollout(
            nominal, _GUIDANCE_TIMES, compute_contact_times=False
        ).minimum_clearance_metres
        reserve = _CLEARANCE_RESERVE_METRES if self.avoidance_heading is None else 2 * _CLEARANCE_RESERVE_METRES
        if margin >= reserve and not prefer_starboard:
            self.avoidance_heading = None
            return AISReference(nominal_heading, predicted_margin_metres=margin)

        wind = observation.wind
        offsets = (15, 30, 45, 60, 75, 90, 110, 135, 165)
        candidates = [nominal_heading, *((nominal_heading + side * delta) % 360.0
                                        for delta in offsets for side in (1, -1))]
        if self.avoidance_heading is not None:
            candidates.append(self.avoidance_heading)
        if wind is not None:
            close_hauled = self.no_go_angle_degrees + 4.0
            candidates.extend((wind.direction + side * close_hauled) % 360.0 for side in (1, -1))
        best = None
        escape = None
        for heading in candidates:
            if wind is not None and abs(_signed_angle(heading - wind.direction)) < self.no_go_angle_degrees:
                continue
            distance = min(max(2.0 * grid.resolution, 45.0), 120.0)
            endpoint = distance * np.array((sin(radians(heading)), cos(radians(heading))))
            if _blocked_segments(grid, blocked, np.array(((0.0, 0.0), endpoint))):
                continue
            preview = self._course_preview(observation, heading)
            assessment = traffic.assess_local_rollout(preview, _GUIDANCE_TIMES, compute_contact_times=False)
            heading_margin = assessment.minimum_clearance_metres
            turn = _signed_angle(heading - nominal_heading)
            bow_change = abs(_signed_angle(heading - observation.boat.heading))
            escape_score = heading_margin - 0.22 * bow_change
            if escape is None or escape_score > escape[0]:
                escape = (escape_score, heading)
            if not assessment.safe:
                continue
            # Prefer efficient detours and retain a selected side through the pass.
            # A give-way preference changes ranking, never excludes an escape side.
            turn_cost = abs(turn) + (55.0 if prefer_starboard and turn < -1.0 else 0.0)
            if prefer_starboard and abs(turn) < 10.0:
                turn_cost += 25.0
            if self.avoidance_heading is not None:
                turn_cost += 0.28 * abs(_signed_angle(heading - self.avoidance_heading))
            score = turn_cost + 0.40 * bow_change + max(0.0, 2.0 * reserve - heading_margin) * 4.0
            if best is None or score < best[0]:
                best = (score, heading, endpoint, heading_margin)

        if best is None:
            return AISReference(None, predicted_margin_metres=margin,
                                escape_heading=None if escape is None else escape[1])
        _, heading, endpoint, heading_margin = best
        if abs(_signed_angle(heading - nominal_heading)) < 0.1:
            self.avoidance_heading = None
            return AISReference(nominal_heading, predicted_margin_metres=heading_margin)
        self.avoidance_heading = heading
        waypoint = Waypoint(*offset_latitude_longitude(
            observation.boat.latitude, observation.boat.longitude, *endpoint,
        ))
        return AISReference(heading, waypoint, heading_margin)

    @staticmethod
    def _course_preview(
        observation: WorldObservation, heading: float, times: np.ndarray = _GUIDANCE_TIMES,
    ) -> np.ndarray:
        """Turn-limited ground-track preview, followed by nonlinear controller checks."""
        state, wind, current = observation.boat, observation.wind, observation.current
        wind_speed = 0.0 if wind is None else wind.speed
        # Keep the measured Earth-fixed motion at the start, then converge to a
        # wind-scaled sailing speed and the proposed course over the next leg.
        body = np.array((state.speed if state.surge_speed is None else state.surge_speed, state.sway_speed))
        actual = Fossen3DOFSaildroneDynamics._body_to_earth(state.heading) @ body
        flow = np.zeros(2) if current is None else np.array((current.u, current.v))
        ground = actual + flow
        ground_speed = np.linalg.norm(ground)
        if ground_speed > 0.08:
            course = np.degrees(np.arctan2(*ground)) % 360.0
        else:
            course = state.heading
        turn = _signed_angle(heading - course)
        headings = np.radians(course + np.sign(turn) * np.minimum(abs(turn), 6.0 * times))
        speed = min(2.5, 0.13 * wind_speed)
        if wind is not None:
            angle = abs(_signed_angle(heading - wind.direction))
            speed *= float(np.interp(angle, (45.0, 80.0, 125.0, 180.0), (0.72, 1.0, 1.12, 0.65)))
        direction = np.column_stack((np.sin(headings), np.cos(headings)))
        target_speed = np.maximum(0.0, speed + direction @ flow)
        speed = target_speed + (ground_speed - target_speed) * np.exp(-times / 12.0)
        velocity = speed[:, None] * direction
        displacement = np.diff(times)[:, None] * (velocity[1:] + velocity[:-1]) / 2.0
        return np.vstack((np.zeros(2), np.cumsum(displacement, axis=0)))


def _integrate_action(
    dynamics: Fossen3DOFSaildroneDynamics,
    states: list[BoatState],
    times: list[float],
    observation: WorldObservation,
    action: BoatAction,
    duration: float,
) -> None:
    segments = max(1, ceil(duration / dynamics.parameters.max_integration_step))
    interval = duration / segments
    for _ in range(segments):
        states.append(dynamics.step(states[-1], action, observation, interval))
        times.append(times[-1] + interval)


def _chart_clear(grid: LocalGrid, blocked: np.ndarray, states: list[BoatState]) -> bool:
    east, north = local_east_north_metres(
        [state.latitude for state in states], [state.longitude for state in states],
        grid.origin_latitude, grid.origin_longitude,
    )
    return not _blocked_segments(grid, blocked, np.column_stack((east, north)))


def chart_action_is_clear(
    dynamics: Fossen3DOFSaildroneDynamics, state: BoatState, observation: WorldObservation,
    action: BoatAction, duration: float, grid: LocalGrid, blocked: np.ndarray,
) -> bool:
    """Check a constant control along the integration substeps on the local chart."""
    states, times = [state], [0.0]
    _integrate_action(dynamics, states, times, observation, action, duration)
    return _chart_clear(grid, blocked, states)


def assess_action(
    dynamics: Fossen3DOFSaildroneDynamics,
    state: BoatState,
    observation: WorldObservation,
    action: BoatAction,
    duration: float,
    grid: LocalGrid | None,
    blocked: np.ndarray | None,
    traffic: AISPrediction,
) -> tuple[bool, AISRolloutAssessment]:
    """Check the applied control on integration substeps, independently of the solver."""
    states, times = [state], [0.0]
    _integrate_action(dynamics, states, times, observation, action, duration)
    chart_clear = grid is None or blocked is None or _chart_clear(grid, blocked, states)
    return chart_clear, traffic.assess_rollout(states, times, compute_contact_times=False)


@dataclass(frozen=True)
class AISFallback:
    action: BoatAction
    mode: str
    assessment: AISRolloutAssessment


def fallback_action(
    dynamics: Fossen3DOFSaildroneDynamics,
    mpc: SaildroneMPC,
    state: BoatState,
    observation: WorldObservation,
    grid: LocalGrid,
    blocked: np.ndarray,
    traffic: AISPrediction,
    dt_seconds: float,
    *,
    escape_heading: float | None = None,
    full_clearance_only: bool = False,
) -> AISFallback | None:
    """Try to depower and drift clear, or apply the least-risk feasible first step."""
    parameters = dynamics.parameters

    def rudder_towards(value: float, initial: float) -> float:
        slew = mpc.max_rudder_change_per_update_degrees
        return float(np.clip(value, max(parameters.rudder_min, initial - slew),
                             min(parameters.rudder_max, initial + slew)))

    def flap_towards(value: float, initial: float) -> float:
        slew = mpc.max_flap_change_per_update_degrees
        return float(np.clip(value, max(parameters.flap_min, initial - slew),
                             min(parameters.flap_max, initial + slew)))

    first_hold = BoatAction(rudder_towards(0.0, state.rudder), flap_towards(0.0, state.flap))
    candidates = dict.fromkeys([
        (first_hold, True),
        *((BoatAction(rudder_towards(rudder, state.rudder), flap_towards(flap, state.flap)), False)
          for rudder in (parameters.rudder_min, 0.0, parameters.rudder_max)
          for flap in (parameters.flap_min, 0.0, parameters.flap_max)),
    ])
    best = None
    for action, depower in candidates:
        states, times = [state], [0.0]
        _integrate_action(dynamics, states, times, observation, action, dt_seconds)
        if not _chart_clear(grid, blocked, states):
            continue
        first = traffic.assess_rollout(states, times, compute_contact_times=False)
        if full_clearance_only and not first.safe:
            continue
        # Compare steady depowering against a powered escape in the same plant.
        # Also screen a longer coast at the resulting ground velocity.
        for _ in range(6):
            last = states[-1]
            target = last.heading if escape_heading is None else escape_heading
            error = _signed_angle(target - last.heading)
            steering = -0.75 * error + 0.65 * np.degrees(last.yaw_rate)
            wind_angle = None if observation.wind is None else _signed_angle(last.heading - observation.wind.direction)
            trim = 0.0 if wind_angle is None else (parameters.flap_max if wind_angle >= 0.0 else parameters.flap_min)
            follow = BoatAction(
                rudder_towards(0.0 if depower else steering, last.rudder),
                flap_towards(0.0 if depower else trim, last.flap),
            )
            _integrate_action(dynamics, states, times, observation, follow, min(5.0, dt_seconds))
        whole = traffic.assess_rollout(states, times, compute_contact_times=False)
        chart_safe = _chart_clear(grid, blocked, states)
        final = states[-1]
        last_position = np.array(local_east_north_metres(
            final.latitude, final.longitude, state.latitude, state.longitude,
        ))
        final_velocity = dynamics._body_to_earth(final.heading) @ np.array((final.surge_speed, final.sway_speed))
        if observation.current is not None:
            final_velocity += (observation.current.u, observation.current.v)
        coast_duration = max(0.0, 120.0 - times[-1])
        last_coast = last_position + coast_duration * final_velocity
        coast = traffic.assess_local_rollout(
            np.array((last_position, last_coast)), (times[-1], times[-1] + max(0.1, coast_duration)),
            compute_contact_times=False,
        )
        coast_chart_clear = not _blocked_segments(grid, blocked, np.array((last_position, last_coast)))
        first_acceptable = first.safe or first.recovering
        if whole.safe and coast.safe and chart_safe and coast_chart_clear and depower:
            return AISFallback(action, "depower", first)
        safe_escape = chart_safe and coast_chart_clear and whole.safe and coast.safe
        mode = "escape" if first_acceptable and safe_escape else "minimum-risk"
        rating = (first.minimum_hull_clearance_metres > 0.0, first_acceptable, safe_escape,
                  chart_safe, whole.minimum_hull_clearance_metres > 0.0,
                  min(whole.minimum_clearance_metres, coast.minimum_clearance_metres),
                  first.minimum_clearance_metres)
        if best is None or rating > best[0]:
            best = (rating, AISFallback(action, mode, first))
    return None if best is None else best[1]
