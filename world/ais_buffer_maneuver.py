"""Plan and retain a physically screened AIS maneuver through a traffic encounter."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from math import ceil, radians
from time import perf_counter

import numpy as np

from ._geo import local_east_north_metres
from .ais_avoidance import AISPrediction, DEFAULT_AIS_SAFETY_BUFFER_METRES
from .ais_guidance import AISGuidance, _blocked_segments, _signed_angle
from .boat import BoatAction
from .fossen import Fossen3DOFSaildroneDynamics
from .mpc import SaildroneMPC
from .occupancy import LocalGrid
from .world import WorldObservation


_LOOKAHEAD_SECONDS = 360.0
_PREDICTION_INTERVAL_SECONDS = 5.0
_TRIGGER_MARGIN_METRES = 12.0
_MARGIN_RESERVE_METRES = 6.0
_SWEEP_INTERVAL_SECONDS = 1.0
_REPORT_COURSE_REVISION_DEGREES = 1.5
_REPORT_SPEED_REVISION_MPS = 0.075
_REPLAN_MARGIN_GAIN_METRES = 3.0
_RECOVERY_SECONDS = 300.0
_FAST_POLISH_ITERATIONS = 15
_FAST_LOCAL_EVALUATIONS = 150


@dataclass(frozen=True)
class _Policy:
    heading: float
    trim: float
    steering_gain: float = 0.65
    initial_trim: float | None = None
    next_heading: float | None = None
    next_trim: float | None = None
    switch_after_seconds: float = 0.0
    controls: tuple[BoatAction, ...] = ()
    recovery_controls: bool = False
    recovered_departure_turn_deg: float | None = None
    colreg_soft_override: bool = False
    recovery_backend: str | None = None

    def demand(self, elapsed: float, dt: float) -> tuple[float, float]:
        if self.next_heading is not None and elapsed >= self.switch_after_seconds:
            assert self.next_trim is not None
            return self.next_heading, self.next_trim
        if self.initial_trim is not None and elapsed < dt:
            return self.heading, self.initial_trim
        return self.heading, self.trim


@dataclass(frozen=True)
class AISBufferManeuver:
    action: BoatAction
    bow_heading: float
    nominal_margin_metres: float
    predicted_margin_metres: float
    refreshed_margin_metres: float | None = None
    refined_controls: bool = False
    recovery_controls: bool = False
    recovery_remaining_seconds: float = 0.0
    recovered_departure_turn_deg: float | None = None
    colreg_soft_override: bool = False
    recovery_backend: str | None = None
    predictive_fallback: bool = False


@dataclass(frozen=True)
class _Rollout:
    first: BoatAction
    positions: np.ndarray
    times: np.ndarray
    controls: tuple[BoatAction, ...]


@lru_cache(maxsize=8)
def _local_transition(parameters, interval: float):
    """Use the existing fast transcription if installed, with a native fallback."""
    try:
        import casadi as ca
    except ImportError:
        return None
    from .acados_fossen import discrete_fossen

    state, action, forcing = ca.SX.sym("state", 9), ca.SX.sym("action", 2), ca.SX.sym("forcing", 5)
    dynamics = Fossen3DOFSaildroneDynamics(parameters)
    step = discrete_fossen(dynamics, state, action, forcing, interval)
    return ca.Function("ais_maneuver_step", [state, action, forcing], [step])


class AISBufferPlanner:
    """Choose a multi-contact maneuver before the actuator controller runs out of choices."""

    def __init__(self, dynamics: Fossen3DOFSaildroneDynamics, mpc: SaildroneMPC,
                 dt_seconds: float, no_go_angle_degrees: float, *,
                 reports_refreshed_each_step: bool = False, predictive_fallback: bool = False) -> None:
        self.dynamics = dynamics
        self.dt = dt_seconds
        self.no_go_angle = no_go_angle_degrees
        self.reports_refreshed_each_step = reports_refreshed_each_step
        self.predictive_fallback = predictive_fallback
        self.escape_seed: np.ndarray | None = None
        parameters = dynamics.parameters
        self.lower = np.array((parameters.rudder_min, parameters.flap_min))
        self.upper = np.array((parameters.rudder_max, parameters.flap_max))
        self.slew = np.array((mpc.max_rudder_change_per_update_degrees, mpc.max_flap_change_per_update_degrees))
        self.policy: _Policy | None = None
        self.duration = 0.0
        self.elapsed = 0.0
        self.report_motion: dict[int, tuple[int, float, float | None]] = {}
        self.recovery_search_profile: dict[str, float | int | str] = {}

    def blocks_open_water_recovery(
        self, grid: LocalGrid, traffic: AISPrediction, prefer_starboard: bool,
        *, candidate_handoff: bool = False,
    ) -> bool:
        """Decide when to admit a weather reference for subsequent AIS screening."""
        if not traffic.targets:
            return False
        if prefer_starboard or (self.policy is not None and self.elapsed < self.duration):
            return True
        if candidate_handoff:
            # Predictive escape can have a continuation seed without a policy.
            # An admitted reference still goes through every AIS and chart check.
            return self.escape_seed is not None
        # Bound the square chart by a circle. Sweeping each chord bounds the
        # uncapped convex growth without assuming a new report arrives.
        expanded = replace(
            traffic,
            clearance_radii_metres=traffic.clearance_radii_metres + grid.span / np.sqrt(2.0),
            preview_growth_seconds=None,
        )
        times = np.arange(0.0, _LOOKAHEAD_SECONDS + _PREDICTION_INTERVAL_SECONDS, _PREDICTION_INTERVAL_SECONDS)
        assessment = expanded.assess_local_rollout(
            np.zeros((times.size, 2)), times, compute_contact_times=False,
        )
        return not assessment.safe

    @staticmethod
    def nominal_traffic(traffic: AISPrediction) -> AISPrediction:
        """The 25 m hull buffer before AIS position and forecast uncertainty."""
        return replace(traffic, clearance_radii_metres=traffic.hull_radii_metres + DEFAULT_AIS_SAFETY_BUFFER_METRES,
                       uncertainty_growth_mps=np.zeros_like(traffic.uncertainty_growth_mps),
                       acceleration_bound_mps2=0.0, clearance_reserve_metres=0.0)

    def _refreshed_traffic(self, traffic: AISPrediction) -> AISPrediction | None:
        # Synthetic AIS supplies a fresh deterministic report each control
        # update. Budget a full update of growth at every point in its preview.
        # The applied action still uses the original, swept uncertainty check.
        if not self.reports_refreshed_each_step or np.any(traffic.report_ages_seconds > 0.0):
            return None
        return replace(
            traffic,
            clearance_radii_metres=traffic.radii_at(self.dt),
            uncertainty_growth_mps=np.zeros_like(traffic.uncertainty_growth_mps),
            acceleration_bound_mps2=0.0,
        )

    def choose(
        self, observation: WorldObservation, grid: LocalGrid, blocked: np.ndarray,
        traffic: AISPrediction, nominal_course: float | None, prefer_starboard: bool,
    ) -> AISBufferManeuver | None:
        revised_course = self._report_course_revised(traffic)
        if not traffic.targets or observation.wind is None:
            self.policy = None
            self.escape_seed = None
            return None
        # A missing nominal route is not a reason to skip emergency control
        # search. This bow is only a seed/ranking reference; the same chart
        # and full traffic checks still govern every accepted recovery.
        if nominal_course is None:
            nominal_course = observation.boat.heading
        nominal = self.nominal_traffic(traffic)
        refreshed = self._refreshed_traffic(traffic)
        incumbent = None
        active = self.policy is not None and self.elapsed < self.duration
        if active:
            assert self.policy is not None
            rollout = self._rollout(observation, self.policy, self.duration - self.elapsed, self.elapsed, dense=True)
            incumbent = self._screen(rollout, self.policy, self.elapsed, traffic, nominal, grid, blocked, refreshed)
            if incumbent is not None and (not revised_course or self.policy.recovery_controls):
                self.elapsed += self.dt
                return incumbent

        reassess = active and revised_course
        times = np.arange(0.0, _LOOKAHEAD_SECONDS + _PREDICTION_INTERVAL_SECONDS, _PREDICTION_INTERVAL_SECONDS)
        preview = AISGuidance._course_preview(observation, nominal_course, times)
        encounter = traffic.assess_local_rollout(preview, times, compute_contact_times=False)
        close = [contact for contact in encounter.contacts
                 if reassess or contact.minimum_clearance_metres < _TRIGGER_MARGIN_METRES]
        if not close:
            self.policy = None
            self.escape_seed = None
            return None
        if self.predictive_fallback and traffic.preview_growth_seconds is not None:
            return self._predictive_escape(observation, grid, blocked, traffic, prefer_starboard)
        last_encounter = max(contact.minimum_clearance_time_seconds for contact in close)
        duration = ceil(max(180.0, min(_LOOKAHEAD_SECONDS, last_encounter + 55.0)) / self.dt) * self.dt
        candidates = self._policies(observation, nominal_course)
        # With several close contacts and no guaranteed update cadence, the
        # single-report forecast can favor a tighter pass just because it
        # occurs earlier. Compare the reserve with the current report ages so
        # the choice leaves room for additional aging at either pass.
        aged = None
        if refreshed is None and len(close) > 1:
            aged = replace(traffic, uncertainty_growth_mps=np.zeros_like(traffic.uncertainty_growth_mps),
                           acceleration_bound_mps2=0.0)
            candidates.extend(self._refine_downwind_turns(observation))
        viable = []
        batched = (self._coarse_rollouts(observation, candidates, duration)
                   if traffic.preview_growth_seconds is not None else None)
        for index, policy in enumerate(candidates):
            rollout = (batched[index] if batched is not None else
                       self._rollout(observation, policy, duration, 0.0, dense=False))
            scored = self._screen(rollout, policy, 0.0, traffic, nominal, grid, blocked, refreshed)
            if scored is not None:
                # Favor a reserve against uncertainty before progress or the
                # directional preference. Avoid a needless long diversion once clear.
                horizon_clear = scored.predicted_margin_metres > 0.0
                clearance = scored.predicted_margin_metres
                if not horizon_clear and scored.refreshed_margin_metres is not None:
                    clearance = scored.refreshed_margin_metres
                elif not horizon_clear and aged is not None:
                    clearance = aged.assess_local_rollout(
                        rollout.positions, rollout.times, compute_contact_times=False,
                    ).minimum_clearance_metres
                if not reassess:
                    clearance = min(clearance, _MARGIN_RESERVE_METRES)
                turn = _signed_angle(scored.bow_heading - nominal_course)
                cost = abs(turn) + (15.0 if prefer_starboard and turn < 0.0 else 0.0)
                progress = float(rollout.positions[-1] @ np.array(
                    (np.sin(radians(nominal_course)), np.cos(radians(nominal_course)))))
                viable.append(((horizon_clear, clearance, progress - cost), policy))
        # A heading policy that is already nominally unsafe can still seed a
        # feasible actuator sequence. Do not discard all such seeds before
        # giving the nonlinear control search a chance to build steerage.
        # The legacy uncapped-preview path keeps its existing semantics.
        recovery_attempted = False
        if traffic.preview_growth_seconds is not None and not any(score[0] for score, _ in viable):
            recovery_attempted = True
            recovery = self._recover_policy(observation, traffic, nominal, grid, blocked,
                                            nominal_course=nominal_course, prefer_starboard=prefer_starboard)
            if recovery is not None:
                policy, scored = recovery
                self.policy, self.duration, self.elapsed = policy, _RECOVERY_SECONDS, self.dt
                return scored
        for _, policy in sorted(viable, key=lambda candidate: candidate[0], reverse=True):
            rollout = self._rollout(observation, policy, duration, 0.0, dense=True)
            scored = self._screen(rollout, policy, 0.0, traffic, nominal, grid, blocked, refreshed)
            if scored is None:
                continue
            if incumbent is not None and (
                scored.predicted_margin_metres < incumbent.predicted_margin_metres + _REPLAN_MARGIN_GAIN_METRES
            ):
                break
            if not active and aged is not None and traffic.acceleration_bound_mps2:
                optimized = self._refine_policy(
                    observation, policy, rollout, traffic, nominal, aged, grid, blocked,
                    duration, min(duration, ceil((last_encounter + 10.0) / self.dt) * self.dt),
                )
                if optimized is not None:
                    policy, scored = optimized
            self.policy, self.duration, self.elapsed = policy, duration, self.dt
            return scored
        if incumbent is not None:
            self.elapsed += self.dt
            return incumbent
        # A coarse chart/traffic pass is not an acceptance certificate. If
        # every such candidate failed the dense check, search actuator
        # sequences before declaring that this library has no maneuver.
        if traffic.preview_growth_seconds is not None and not recovery_attempted:
            recovery = self._recover_policy(observation, traffic, nominal, grid, blocked,
                                            nominal_course=nominal_course, prefer_starboard=prefer_starboard)
            if recovery is not None:
                policy, scored = recovery
                self.policy, self.duration, self.elapsed = policy, _RECOVERY_SECONDS, self.dt
                return scored
        self.policy = None
        return None

    def _predictive_escape(self, observation, grid, blocked, traffic, prefer_starboard):
        from .ais_predictive_fallback import search_escape

        self.policy = None
        step = _local_transition(self.dynamics.parameters, self.dt)
        if step is None:
            self.escape_seed = None
            return None
        initial, forcing = self._motion_inputs(observation)
        proposal = search_escape(
            step, self.dynamics, initial, forcing, observation, grid, blocked,
            traffic, self.lower, self.upper, self.slew, self.dt, self.escape_seed, prefer_starboard,
        )
        self.escape_seed = None if proposal is None else proposal.continuation
        if proposal is None:
            return None
        return AISBufferManeuver(
            proposal.action, observation.boat.heading,
            proposal.nominal_margin, proposal.preview_margin, predictive_fallback=True,
        )

    def _report_course_revised(self, traffic: AISPrediction) -> bool:
        previous = self.report_motion
        self.report_motion = {
            target.mmsi: (
                traffic.observed_at_ns - round(target.report_age_seconds * 1e9),
                target.course,
                target.speed,
            ) for target in traffic.targets
            if target.mmsi is not None and target.report_age_seconds is not None
            and target.course is not None and np.isfinite(target.course)
        }
        return any(
            mmsi in previous and epoch > previous[mmsi][0]
            and (abs(_signed_angle(course - previous[mmsi][1])) >= _REPORT_COURSE_REVISION_DEGREES
                 or speed is not None and previous[mmsi][2] is not None
                 and abs(speed - previous[mmsi][2]) >= _REPORT_SPEED_REVISION_MPS)
            for mmsi, (epoch, course, speed) in self.report_motion.items()
        )

    def _policies(self, observation: WorldObservation, course: float) -> list[_Policy]:
        assert observation.wind is not None
        wind = observation.wind.direction
        flap_min, flap_max = self.lower[1], self.upper[1]
        current_trim = flap_max if _signed_angle(observation.boat.heading - wind) >= 0 else flap_min
        headings = dict.fromkeys([course, observation.boat.heading, *(
            (wind + side * angle) % 360.0 for angle in
            (self.no_go_angle + 5.0, 65.0, 80.0, 90.0, 100.0, 115.0, 130.0, 145.0, 160.0, 170.0)
            for side in (1, -1)
        )])
        policies = []
        for heading in headings:
            wind_angle = _signed_angle(heading - wind)
            if abs(wind_angle) < self.no_go_angle:
                continue
            trim = flap_max if wind_angle >= 0 else flap_min
            for gain in (0.45, 0.65):
                policies.append(_Policy(heading, trim, gain))
                if trim != current_trim:
                    policies.append(_Policy(heading, trim, gain, current_trim))
        policies.append(_Policy(observation.boat.heading, 0.0, steering_gain=0.0))
        # Turning across the downwind axis can preserve steerage early while
        # opening a different passing lane before the next contact arrives.
        for side in (1, -1):
            initial, switched = (flap_max, flap_min) if side > 0 else (flap_min, flap_max)
            for near_downwind in (155.0, 160.0, 170.0):
                for after_turn in (130.0, 145.0, 160.0):
                    for duration in (35.0, 45.0, 55.0):
                        policies.append(_Policy(
                            (wind + side * near_downwind) % 360.0, initial,
                            next_heading=(wind - side * after_turn) % 360.0,
                            next_trim=switched, switch_after_seconds=duration,
                        ))
        return policies

    def _refine_downwind_turns(self, observation: WorldObservation) -> list[_Policy]:
        """Sample nearby bow references and switch times when contacts overlap."""
        assert observation.wind is not None
        wind = observation.wind.direction
        flap_min, flap_max = self.lower[1], self.upper[1]
        refinements = []
        for side, trim, switched in ((1, flap_max, flap_min), (-1, flap_min, flap_max)):
            for approach in range(155, 161):
                for departure in (approach - 5, approach, approach + 5):
                    for gain in (0.55, 0.60, 0.65):
                        for switch_after in (40.0, 45.0):
                            refinements.append(_Policy(
                                (wind + side * approach) % 360.0, trim, gain,
                                next_heading=(wind - side * departure) % 360.0,
                                next_trim=switched, switch_after_seconds=switch_after,
                            ))
        return refinements

    def _screen(
        self, rollout: _Rollout, policy: _Policy, elapsed: float,
        traffic: AISPrediction, nominal: AISPrediction, grid: LocalGrid, blocked: np.ndarray,
        refreshed: AISPrediction | None = None,
    ) -> AISBufferManeuver | None:
        full = traffic.assess_local_rollout(rollout.positions, rollout.times, compute_contact_times=False)
        fixed = nominal.assess_local_rollout(rollout.positions, rollout.times, compute_contact_times=False)
        # A declared report-age budget is an explicit full-clearance preview,
        # not permission to retain a nominal-only heading through its buffer.
        # Keep the historical uncapped-preview semantics for legacy callers.
        if (policy.recovery_controls or traffic.preview_growth_seconds is not None) and not full.safe:
            return None
        if not fixed.safe:
            return None
        refreshed_margin = None
        if refreshed is not None and not full.safe:
            refreshed_margin = refreshed.assess_local_rollout(
                rollout.positions, rollout.times, compute_contact_times=False,
            ).minimum_clearance_metres
            if refreshed_margin <= 0.0:
                return None
        if _blocked_segments(grid, blocked, rollout.positions):
            return None
        return AISBufferManeuver(rollout.first, policy.demand(elapsed, self.dt)[0],
                                 fixed.minimum_clearance_metres, full.minimum_clearance_metres,
                                 refreshed_margin, elapsed < len(policy.controls) * self.dt,
                                 policy.recovery_controls,
                                 max(0.0, len(policy.controls) * self.dt - elapsed) if policy.recovery_controls else 0.0,
                                 policy.recovered_departure_turn_deg, policy.colreg_soft_override,
                                 policy.recovery_backend)

    @staticmethod
    def _recovery_rank(rollout: _Rollout, margin: float, course: float,
                       prefer_starboard: bool) -> tuple[tuple[float, float, float], float, bool]:
        """Keep full-clearance priority, then the existing soft course bias.

        An optimized seed's heading is not its realized departure course,
        especially during native astern motion. Use the first 30 seconds of
        the verified ground displacement. This remains a soft preference,
        not a COLREG compliance certificate or a forbidden escape side.
        """
        index = min(int(np.searchsorted(rollout.times, 30.0)), len(rollout.times) - 1)
        delta = rollout.positions[index] - rollout.positions[0]
        departure = course if np.linalg.norm(delta) < 1e-9 else float(np.degrees(np.arctan2(*delta)))
        turn = _signed_angle(departure - course)
        override = bool(prefer_starboard and turn < 0.0)
        cost = abs(turn) + (15.0 if override else 0.0)
        direction = np.array((np.sin(radians(course)), np.cos(radians(course))))
        progress = float((rollout.positions[-1] - rollout.positions[0]) @ direction)
        return (min(margin, _MARGIN_RESERVE_METRES), progress - cost, margin), turn, override

    def _recovery_policies(self, observation: WorldObservation) -> list[_Policy]:
        """Generic forward/astern startup seeds, without imposing a velocity.

        Opposite first trims allow the native plant to build either sign of
        surge before it turns. All later commands and slews are optimized.
        The compass grid is independent of route, contact identity or truth.
        """
        assert observation.wind is not None
        wind = observation.wind.direction
        policies = []
        for heading in range(0, 360, 10):
            wind_angle = _signed_angle(heading - wind)
            if abs(wind_angle) < self.no_go_angle:
                continue
            trim = self.upper[1] if wind_angle >= 0 else self.lower[1]
            for gain in (0.35, 0.65, 1.0):
                for first_trim in (None, -trim):
                    policies.append(_Policy(float(heading), float(trim), gain, first_trim))
        return policies

    def _recover_policy(
        self, observation: WorldObservation, traffic: AISPrediction, nominal: AISPrediction,
        grid: LocalGrid, blocked: np.ndarray,
        *, nominal_course: float | None = None, prefer_starboard: bool = False,
    ) -> tuple[_Policy, AISBufferManeuver] | None:
        """Optimize rejected seeds against full preview uncertainty before fallback.

        Received AIS supplies every target center and velocity. The fixed
        radius is the maximum over this preview, so initial small report ages
        cannot make an early pass look artificially attractive. The planner
        retains a sequence only after dense native/chart/full-margin checks,
        and revalidates its remaining commands on every subsequent update.
        """
        from .ais_clearance_refinement import refine_clearance

        step = _local_transition(self.dynamics.parameters, self.dt)
        if step is None:
            return None
        duration = _RECOVERY_SECONDS
        course = observation.boat.heading if nominal_course is None else nominal_course
        count = round(duration / self.dt)
        state, forcing = self._motion_inputs(observation)
        from .ais_fast_recovery import search_feedback

        fast, profile = search_feedback(step, state, forcing, traffic, self.lower, self.upper,
                                        self.slew, self.dt, duration, local_evaluations=_FAST_LOCAL_EVALUATIONS)
        self.recovery_search_profile = profile
        radii = np.max(traffic.radii_at(np.arange(count * 4 + 1) * self.dt / 4), axis=0)
        polishing_started = perf_counter()
        polished = refine_clearance(
            step, forcing, fast.states, fast.commands, traffic.positions_metres, traffic.velocities_mps,
            radii, np.radians(self.lower), np.radians(self.upper), np.radians(self.slew), self.dt,
            max_iterations=_FAST_POLISH_ITERATIONS, cpu_seconds=None, keep_best_iterate=True,
        )
        profile["native_polish_seconds"] = perf_counter() - polishing_started
        profile["native_polish_iteration_limit"] = _FAST_POLISH_ITERATIONS
        best_fast, best_fast_rank = None, None
        for commands in ([fast.commands] if polished is None else [fast.commands, polished]):
            policy = _Policy(fast.family.heading, fast.family.flap,
                             controls=tuple(BoatAction(*row) for row in np.degrees(commands)),
                             recovery_controls=True, recovery_backend="native feedback + bounded polish")
            verified = self._rollout(observation, policy, duration, 0., dense=True)
            scored = self._screen(verified, policy, 0., traffic, nominal, grid, blocked)
            if scored is not None:
                rank, turn, override = self._recovery_rank(verified, scored.predicted_margin_metres,
                                                           course, prefer_starboard)
                if best_fast_rank is None or rank > best_fast_rank:
                    best_fast_rank = rank
                    best_fast = (replace(policy, recovered_departure_turn_deg=turn, colreg_soft_override=override),
                                 replace(scored, recovered_departure_turn_deg=turn, colreg_soft_override=override))
        # Six metres caps ranking preference, rather than adding a second
        # acceptance buffer on top of the configured full uncertainty/reserve.
        # With no long search following this branch, discarding an independently
        # screened positive-margin sequence would needlessly invoke last resort.
        if best_fast is not None:
            profile["slow_recovery_fallback"] = 0
            profile["recovery_search_outcome"] = "full-clearance sequence verified"
            return best_fast
        # A bounded report-age run must not hide an unbounded multi-solve
        # search behind a failed fast attempt. No sequence is certified here;
        # the caller proceeds to its separately screened one-step last resort.
        # Its applied uncertainty check is unchanged and may stop navigation.
        profile["slow_recovery_fallback"] = 0
        profile["recovery_search_outcome"] = "no full-clearance sequence found"
        return None

    def _refine_policy(
        self, observation: WorldObservation, seed: _Policy, rollout: _Rollout,
        traffic: AISPrediction, nominal: AISPrediction, aged: AISPrediction,
        grid: LocalGrid, blocked: np.ndarray, duration: float, optimization_duration: float,
    ) -> tuple[_Policy, AISBufferManeuver] | None:
        from .ais_clearance_refinement import refine_clearance

        step = _local_transition(self.dynamics.parameters, self.dt)
        if step is None:
            return None
        state, forcing = self._motion_inputs(observation)
        count = round(optimization_duration / self.dt)
        commands = np.radians([(a.rudder, a.flap) for a in rollout.controls[:count]])
        states = [state]
        for command in commands:
            states.append(np.asarray(step(states[-1], command, forcing)).ravel())
        seed_clearance = aged.assess_local_rollout(rollout.positions, rollout.times, compute_contact_times=False)
        radii = aged.clearance_radii_metres.copy()
        approaching = [(index, contact.minimum_clearance_time_seconds)
                       for index, contact in enumerate(seed_clearance.contacts)
                       if contact.minimum_clearance_metres < _TRIGGER_MARGIN_METRES]
        if approaching:
            first_contact, _ = min(approaching, key=lambda contact: contact[1])
            # The first pass has less time for new reports or a correction.
            # Budget three additional control intervals of possible motion.
            radii[first_contact] += 0.5 * traffic.acceleration_bound_mps2 * (3 * self.dt)**2
        refinement_options = ({"max_iterations": _FAST_POLISH_ITERATIONS, "cpu_seconds": None,
                               "keep_best_iterate": True}
                              if traffic.preview_growth_seconds is not None else {})
        refined = refine_clearance(
            step, forcing, np.asarray(states), commands, aged.positions_metres, aged.velocities_mps,
            radii, np.radians(self.lower), np.radians(self.upper), np.radians(self.slew), self.dt,
            **refinement_options,
        )
        if refined is None:
            return None
        policy = replace(seed, controls=tuple(BoatAction(*row) for row in np.degrees(refined)))
        verified = self._rollout(observation, policy, duration, 0.0, dense=True)
        scored = self._screen(verified, policy, 0.0, traffic, nominal, grid, blocked)
        if scored is None:
            return None
        reserved = replace(aged, clearance_radii_metres=radii)
        new_clearance = reserved.assess_local_rollout(verified.positions, verified.times, compute_contact_times=False)
        old_clearance = reserved.assess_local_rollout(rollout.positions, rollout.times, compute_contact_times=False)
        if new_clearance.minimum_clearance_metres <= old_clearance.minimum_clearance_metres:
            return None
        return policy, scored

    @staticmethod
    def _motion_inputs(observation: WorldObservation) -> tuple[np.ndarray, np.ndarray]:
        assert observation.wind is not None
        boat, current, wave, wind = observation.boat, observation.current, observation.wave, observation.wind
        forcing = np.array((wind.u, wind.v, 0.0 if current is None else current.u,
                            0.0 if current is None else current.v, 0.0 if wave is None else wave.height))
        state = np.array((0.0, 0.0, radians(boat.heading), boat.speed if boat.surge_speed is None else boat.surge_speed,
                          boat.sway_speed, boat.yaw_rate, radians(boat.flap), radians(boat.rudder), radians(boat.flap)))
        return state, forcing

    def _coarse_rollouts(
        self, observation: WorldObservation, policies: list[_Policy], duration: float,
    ) -> list[_Rollout]:
        """Batch the unchanged heading catalog with the same native transition.

        This only removes repeated Python/CasADi crossings. It preserves the
        scalar catalog's control demands, actual-flap slew reference, ordering,
        integration interval and output positions. Dense verification remains
        separate. Preset command sequences use the general scalar path.
        """
        if not policies:
            return []
        step = _local_transition(self.dynamics.parameters, self.dt)
        if step is None or any(policy.controls for policy in policies):
            return [self._rollout(observation, policy, duration, 0., dense=False) for policy in policies]
        initial, forcing = self._motion_inputs(observation)
        count = max(1, round(duration / self.dt))
        number = len(policies)
        nodes = np.broadcast_to(initial[:, None], (9, number)).copy()
        environment = np.broadcast_to(forcing[:, None], (5, number)).copy()
        transition = step.map(number)
        positions = [nodes[:2].T.copy()]
        commands = []
        gains = np.asarray([policy.steering_gain for policy in policies])
        for index in range(count):
            demand = np.asarray([policy.demand(index * self.dt, self.dt) for policy in policies])
            error = (np.degrees(nodes[2]) - demand[:, 0] + 180.) % 360. - 180.
            steering = gains * error + 2.5 * np.degrees(nodes[5])
            # Match the historical scalar catalog exactly. Its dense native
            # path and the applied controller use the observed actuator state.
            previous = np.degrees(nodes[[7, 6]].T)
            control = np.clip(np.column_stack((steering, demand[:, 1])),
                              np.maximum(self.lower, previous - self.slew),
                              np.minimum(self.upper, previous + self.slew))
            nodes = np.asarray(transition(nodes, np.radians(control).T, environment))
            positions.append(nodes[:2].T.copy())
            commands.append(control)
        positions, commands = np.asarray(positions), np.asarray(commands)
        times = np.arange(count + 1) * self.dt
        result = []
        for index in range(number):
            actions = tuple(BoatAction(*map(float, row)) for row in commands[:, index])
            result.append(_Rollout(actions[0], positions[:, index], times, actions))
        return result

    def _rollout(
        self, observation: WorldObservation, policy: _Policy, duration: float, elapsed: float, *, dense: bool,
    ) -> _Rollout:
        assert observation.wind is not None
        boat = observation.boat
        pieces = max(1, ceil(self.dt / _SWEEP_INTERVAL_SECONDS)) if dense else 1
        interval = self.dt / pieces
        step = _local_transition(self.dynamics.parameters, interval)
        node, forcing = self._motion_inputs(observation)
        heading, flap, rudder = boat.heading, boat.flap, boat.rudder
        yaw = boat.yaw_rate
        points = [(0.0, 0.0)]
        count = max(1, round(duration / self.dt))
        commands = []
        start_index = round(elapsed / self.dt)
        for index in range(count):
            control_index = start_index + index
            if control_index < len(policy.controls):
                planned = policy.controls[control_index]
                steering, trim = planned.rudder, planned.flap
            else:
                desired_heading, trim = policy.demand(elapsed + index * self.dt, self.dt)
                steering = policy.steering_gain * _signed_angle(heading - desired_heading) + 2.5 * np.degrees(yaw)
            old = np.array((rudder, flap))
            control = np.clip((steering, trim), np.maximum(self.lower, old - self.slew),
                              np.minimum(self.upper, old + self.slew))
            action = BoatAction(*map(float, control))
            commands.append(action)
            for _ in range(pieces):
                if step is None:
                    boat = self.dynamics.step(boat, action, observation, interval)
                    points.append(local_east_north_metres(boat.latitude, boat.longitude,
                                                          observation.boat.latitude, observation.boat.longitude))
                else:
                    node = np.asarray(step(node, np.radians(control), forcing)).ravel()
                    points.append(node[:2])
            if step is None:
                heading, yaw, flap, rudder = boat.heading, boat.yaw_rate, boat.flap, boat.rudder
            else:
                heading, flap, rudder = np.degrees(node[[2, 6, 7]])
                yaw = node[5]
        return _Rollout(commands[0], np.asarray(points), np.arange(len(points)) * interval, tuple(commands))
