"""Small nonlinear direct-shooting MPC for the saildrone's two controls."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, ceil, cos, degrees, floor, hypot, radians, sin
from typing import TYPE_CHECKING

import numpy as np

from config import HARMONIC_TACTICAL_PLANNER, SAILDRONE_MPC

from ._geo import local_east_north_metres
from .boat import BoatAction, BoatDynamics, BoatState
from .harmonic_planner import HarmonicPlan
from .occupancy import LocalGrid
from .world import WorldObservation

if TYPE_CHECKING:
    from .weather import Wind


@dataclass(frozen=True)
class MPCResult:
    """The first action and predicted trajectory selected by one MPC solve."""

    action: BoatAction
    cost: float
    predicted_states: tuple[BoatState, ...]
    candidates_evaluated: int


@dataclass(frozen=True)
class SaildroneMPC:
    """Evaluate bounded rudder/flap actions against a harmonic track reference.

    This is nonlinear direct shooting, not a linearised controller: every
    candidate is propagated with the supplied dynamics model.  The selected
    action is applied for one simulator step only and then replanned, which is
    the defining receding-horizon behaviour of MPC.
    """

    prediction_horizon_steps: int = int(SAILDRONE_MPC["prediction_horizon_steps"])
    prediction_dt_seconds: float = float(SAILDRONE_MPC["prediction_dt_seconds"])
    rudder_candidates_degrees: tuple[float, ...] = tuple(
        SAILDRONE_MPC["rudder_candidates_degrees"]
    )
    flap_candidates_degrees: tuple[float, ...] = tuple(
        SAILDRONE_MPC["flap_candidates_degrees"]
    )
    heading_error_weight: float = float(SAILDRONE_MPC["heading_error_weight"])
    progress_reward_per_metre: float = float(SAILDRONE_MPC["progress_reward_per_metre"])
    reverse_route_progress_penalty_per_metre: float = float(
        SAILDRONE_MPC["reverse_route_progress_penalty_per_metre"]
    )
    harmonic_potential_weight: float = float(SAILDRONE_MPC["harmonic_potential_weight"])
    obstacle_penalty: float = float(SAILDRONE_MPC["obstacle_penalty"])
    outside_grid_penalty: float = float(SAILDRONE_MPC["outside_grid_penalty"])
    no_go_penalty: float = float(SAILDRONE_MPC["no_go_penalty"])
    heel_soft_limit_degrees: float = float(SAILDRONE_MPC["heel_soft_limit_degrees"])
    heel_weight: float = float(SAILDRONE_MPC["heel_weight"])
    control_effort_weight: float = float(SAILDRONE_MPC["control_effort_weight"])
    control_change_weight: float = float(SAILDRONE_MPC["control_change_weight"])
    max_rudder_change_per_update_degrees: float = float(
        SAILDRONE_MPC["max_rudder_change_per_update_degrees"]
    )
    max_flap_change_per_update_degrees: float = float(
        SAILDRONE_MPC["max_flap_change_per_update_degrees"]
    )
    no_go_angle_degrees: float = float(HARMONIC_TACTICAL_PLANNER["no_go_angle_degrees"])
    close_hauled_rudder_step_degrees: float = float(
        SAILDRONE_MPC["close_hauled_rudder_step_degrees"]
    )

    def __post_init__(self) -> None:
        if self.prediction_horizon_steps < 1 or self.prediction_dt_seconds <= 0.0:
            raise ValueError("MPC prediction horizon and dt must be positive.")
        if not self.rudder_candidates_degrees or not self.flap_candidates_degrees:
            raise ValueError("MPC needs at least one rudder and flap candidate.")
        if any(not np.isfinite(value) for value in self.rudder_candidates_degrees):
            raise ValueError("MPC rudder candidates must be finite.")
        if any(not np.isfinite(value) for value in self.flap_candidates_degrees):
            raise ValueError("MPC flap candidates must be finite.")
        if not np.isfinite(self.close_hauled_rudder_step_degrees) or (
            self.close_hauled_rudder_step_degrees <= 0.0
        ):
            raise ValueError("Close-hauled rudder step must be finite and positive.")
        if self.no_go_angle_degrees <= 0.0:
            raise ValueError("MPC no-go angle must be positive.")
        if (
            self.max_rudder_change_per_update_degrees <= 0.0
            or self.max_flap_change_per_update_degrees <= 0.0
        ):
            raise ValueError("MPC actuator slew bounds must be positive.")
        if self.reverse_route_progress_penalty_per_metre < 0.0:
            raise ValueError("MPC reverse-route penalty must be non-negative.")
        if self.heel_soft_limit_degrees < 0.0 or self.heel_weight < 0.0:
            raise ValueError("MPC heel soft limit and weight must be non-negative.")

    def choose_action(
        self,
        dynamics: BoatDynamics,
        state: BoatState,
        observation: WorldObservation,
        desired_heading: float,
        *,
        local_grid: LocalGrid | None = None,
        harmonic_plan: HarmonicPlan | None = None,
    ) -> MPCResult:
        """Return the lowest-cost bounded action for one receding-horizon step."""

        best_action: BoatAction | None = None
        best_cost = np.inf
        best_states: tuple[BoatState, ...] = ()
        candidates_evaluated = 0
        for rudder in self._rudder_candidates(state, observation, desired_heading, harmonic_plan):
            for flap in self.flap_candidates_degrees:
                action = BoatAction(rudder=float(rudder), flap=float(flap))
                candidates_evaluated += 1
                if not self._within_actuator_slew(state, action):
                    continue
                predicted_states, cost = self._rollout_cost(
                    dynamics,
                    state,
                    observation,
                    action,
                    desired_heading,
                    local_grid=local_grid,
                    harmonic_plan=harmonic_plan,
                )
                if cost < best_cost:
                    best_action = action
                    best_cost = cost
                    best_states = predicted_states
        assert best_action is not None  # Candidate sets are validated above.
        return MPCResult(
            action=best_action,
            cost=float(best_cost),
            predicted_states=best_states,
            candidates_evaluated=candidates_evaluated,
        )

    def _rudder_candidates(
        self, state: BoatState, observation: WorldObservation, reference: float,
        plan: HarmonicPlan | None,
    ) -> tuple[float, ...]:
        wind = observation.wind
        if plan is not None and plan.desired_bow_heading is not None:
            reference = plan.desired_bow_heading
        if (
            plan is None or plan.navigation_target is None or wind is None
            or self._angular_error(state.heading, wind.direction) > self.no_go_angle_degrees + 20.0
            or self._angular_error(reference, wind.direction) >= 90.0
            or self._crossing_wind(state.heading, reference, wind.direction)
        ):
            return self.rudder_candidates_degrees
        step = self.close_hauled_rudder_step_degrees
        slew = self.max_rudder_change_per_update_degrees
        lower = max(min(self.rudder_candidates_degrees), state.rudder - slew)
        upper = min(max(self.rudder_candidates_degrees), state.rudder + slew)
        trim_candidates = (i * step for i in range(ceil(lower / step), floor(upper / step) + 1))
        # The wing's yaw moment at full trim can require a rudder angle
        # between the coarse commands used for regular manoeuvres.
        return tuple(sorted(set(self.rudder_candidates_degrees).union(trim_candidates)))

    def evaluate_action(
        self,
        dynamics: BoatDynamics,
        state: BoatState,
        observation: WorldObservation,
        action: BoatAction,
        desired_heading: float,
        *,
        local_grid: LocalGrid | None = None,
        harmonic_plan: HarmonicPlan | None = None,
    ) -> MPCResult:
        """Evaluate one externally proposed action with the MPC rollout cost."""

        if not self._within_actuator_slew(state, action):
            raise ValueError("Proposed action exceeds the configured actuator slew bounds.")
        predicted_states, cost = self._rollout_cost(
            dynamics,
            state,
            observation,
            action,
            desired_heading,
            local_grid=local_grid,
            harmonic_plan=harmonic_plan,
        )
        return MPCResult(
            action=action,
            cost=float(cost),
            predicted_states=predicted_states,
            candidates_evaluated=1,
        )

    def _rollout_cost(
        self,
        dynamics: BoatDynamics,
        state: BoatState,
        observation: WorldObservation,
        action: BoatAction,
        desired_heading: float,
        *,
        local_grid: LocalGrid | None,
        harmonic_plan: HarmonicPlan | None,
    ) -> tuple[tuple[BoatState, ...], float]:
        predicted: list[BoatState] = []
        current = state
        bow_reference = desired_heading
        if harmonic_plan is not None and harmonic_plan.desired_bow_heading is not None:
            bow_reference = harmonic_plan.desired_bow_heading
        crossing_wind = (
            harmonic_plan is not None
            and harmonic_plan.navigation_target is not None
            and observation.wind is not None
            and self._crossing_wind(state.heading, bow_reference, observation.wind.direction)
        )
        turning_to_course = (
            harmonic_plan is not None
            and harmonic_plan.navigation_target is not None
            and self._angular_error(state.heading, bow_reference) > 60.0
        )
        cost = self.control_effort_weight * (
            (action.rudder / 30.0) ** 2 + (action.flap / 15.0) ** 2
        )
        cost += self.control_change_weight * (
            ((action.rudder - state.rudder) / 30.0) ** 2
            + ((action.flap - state.flap) / 15.0) ** 2
        )
        for _ in range(self.prediction_horizon_steps):
            next_state = dynamics.step(
                current,
                action,
                observation,
                self.prediction_dt_seconds,
            )
            predicted.append(next_state)
            cost += self._state_cost(
                current,
                next_state,
                observation.wind,
                desired_heading,
                local_grid=local_grid,
                harmonic_plan=harmonic_plan,
                turning_to_course=turning_to_course,
                crossing_wind=crossing_wind,
            )
            current = next_state
        if crossing_wind or turning_to_course:
            heading_error = self._angular_error(current.heading, bow_reference)
            terminal_weight = 7.0 if crossing_wind else 4.0
            cost += terminal_weight * self.heading_error_weight * (heading_error / 90.0) ** 2
        return tuple(predicted), float(cost)

    def _crossing_wind(self, heading: float, reference: float, wind_from: float) -> bool:
        """Recognize a change of tack and keep it active until the bow exits the cone."""
        bow = (heading - wind_from + 180.0) % 360.0 - 180.0
        target = (reference - wind_from + 180.0) % 360.0 - 180.0
        return (
            self.no_go_angle_degrees <= abs(target) < 90.0
            and abs(bow) < 90.0
            and (bow * target < 0.0 or abs(bow) < self.no_go_angle_degrees)
        )

    def _state_cost(
        self,
        previous: BoatState,
        current: BoatState,
        wind: "Wind | None",
        desired_heading: float,
        *,
        local_grid: LocalGrid | None,
        harmonic_plan: HarmonicPlan | None,
        turning_to_course: bool = False,
        crossing_wind: bool = False,
    ) -> float:
        # The harmonic path is Earth-fixed and OpenCPN displays COG.  Track
        # the predicted ground course, not the hull heading: current and sway
        # otherwise leave a persistent course error even when heading appears
        # to match the waypoint bearing.
        course_over_ground = self._course_over_ground(previous, current)
        course_error = self._angular_error(course_over_ground, desired_heading)
        course_weight = 0.02 if crossing_wind else 1.0
        cost = course_weight * self.heading_error_weight * (course_error / 90.0) ** 2
        if harmonic_plan is not None and harmonic_plan.desired_bow_heading is not None and not crossing_wind:
            bow_error = self._angular_error(current.heading, harmonic_plan.desired_bow_heading)
            cost += self.heading_error_weight * (bow_error / 90.0) ** 2
        east, north = local_east_north_metres(
            current.latitude,
            current.longitude,
            previous.latitude,
            previous.longitude,
        )
        if harmonic_plan is not None and harmonic_plan.navigation_target is not None:
            target = harmonic_plan.navigation_target
            target_east, target_north = local_east_north_metres(
                target.latitude, target.longitude, previous.latitude, previous.longitude
            )
            along_track = hypot(target_east, target_north) - hypot(
                target_east - east, target_north - north
            )
        else:
            route_heading = self._route_heading(local_grid, harmonic_plan)
            progress_heading = desired_heading if route_heading is None else route_heading
            desired_radians = radians(progress_heading)
            along_track = float(east) * sin(desired_radians) + float(north) * cos(desired_radians)
        cost -= self.progress_reward_per_metre * along_track
        # A turn can begin with a short loss of progress. Leave the turn
        # available to MPC and judge its terminal heading in the rollout.
        if not (turning_to_course or crossing_wind):
            cost += self.reverse_route_progress_penalty_per_metre * max(0.0, -along_track)
        excess_heel = max(0.0, abs(current.heel) - self.heel_soft_limit_degrees)
        cost += self.heel_weight * excess_heel**2
        if wind is not None:
            upwind_error = self._angular_error(current.heading, wind.direction)
            if upwind_error < self.no_go_angle_degrees:
                # The bow must pass through the wind on an intentional tack.
                # During that turn, terminal alignment drives the manoeuvre.
                no_go_weight = 0.04 if crossing_wind else 1.0
                cost += no_go_weight * self.no_go_penalty * (
                    (self.no_go_angle_degrees - upwind_error) / self.no_go_angle_degrees
                ) ** 2
        if local_grid is not None and harmonic_plan is not None:
            row, column = self._cell_for_state(local_grid, current)
            if not 0 <= row < local_grid.cells or not 0 <= column < local_grid.cells:
                cost += self.outside_grid_penalty
            elif harmonic_plan.blocked[row, column]:
                cost += self.obstacle_penalty
            else:
                cost += self.harmonic_potential_weight * harmonic_plan.potential[row, column]
        return float(cost)

    @staticmethod
    def _cell_for_state(grid: LocalGrid, state: BoatState) -> tuple[int, int]:
        east, north = local_east_north_metres(
            state.latitude,
            state.longitude,
            grid.origin_latitude,
            grid.origin_longitude,
        )
        centre = (grid.cells - 1) / 2.0
        return (
            int(round(centre - float(north) / grid.resolution)),
            int(round(centre + float(east) / grid.resolution)),
        )

    @staticmethod
    def _course_over_ground(previous: BoatState, current: BoatState) -> float:
        """Derive true COG from consecutive Earth-fixed state positions."""

        east, north = local_east_north_metres(
            current.latitude,
            current.longitude,
            previous.latitude,
            previous.longitude,
        )
        if np.hypot(east, north) < 1e-6:
            return float(current.heading % 360.0)
        return float(degrees(atan2(float(east), float(north))) % 360.0)

    @staticmethod
    def _route_heading(
        grid: LocalGrid | None,
        plan: HarmonicPlan | None,
    ) -> float | None:
        """Return the geographic direction from the grid centre to its target."""

        if grid is None or plan is None:
            return None
        centre = (grid.cells - 1) / 2.0
        east = float(plan.target_cell[1] - centre)
        north = float(centre - plan.target_cell[0])
        if np.hypot(east, north) < 1e-12:
            return None
        return float(np.degrees(np.arctan2(east, north)) % 360.0)

    def _within_actuator_slew(self, state: BoatState, action: BoatAction) -> bool:
        """Limit successive commanded rudder and flap changes at one MPC update."""

        return (
            abs(action.rudder - state.rudder)
            <= self.max_rudder_change_per_update_degrees + 1e-12
            and abs(action.flap - state.flap)
            <= self.max_flap_change_per_update_degrees + 1e-12
        )

    @staticmethod
    def _angular_error(first: float, second: float) -> float:
        return abs((first - second + 180.0) % 360.0 - 180.0)
