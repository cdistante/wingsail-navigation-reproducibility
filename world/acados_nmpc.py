"""Optional acados NMPC for sailable-segment tracking with the Fossen plant."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, ceil, cos, degrees, hypot, pi, radians, sin, tan
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np

from ._geo import local_east_north_metres, offset_latitude_longitude
from .boat import BoatAction, BoatState
from .fossen import Fossen3DOFSaildroneDynamics
from .harmonic_planner import HarmonicPlan
from .legacy_planner_adapter import LegacyPlannerAdapter
from .mpc import MPCResult, SaildroneMPC
from .occupancy import LocalGrid
from .scenario import Waypoint
from .world import WorldObservation

if TYPE_CHECKING:
    from .ais_avoidance import AISPrediction


_NAVIGATION_PARAMETERS = 14
_AIS_PARAMETERS = 5
_AIS_BUFFER_WEIGHT = 200.0


@dataclass(frozen=True)
class AcadosMPCResult(MPCResult):
    solver_status: int
    solve_time_ms: float
    fallback_reason: str | None = None
    safety_intervention: str | None = None
    control_source: str = "acados"


class AISControlInfeasible(RuntimeError):
    """No native or grid-MPC action passed the one-step safety shield."""

    def __init__(
        self, *, solver_status: int, solve_time_ms: float,
        fallback_reason: str | None, safety_intervention: str,
    ) -> None:
        super().__init__(safety_intervention)
        self.solver_status = solver_status
        self.solve_time_ms = solve_time_ms
        self.fallback_reason = fallback_reason
        self.safety_intervention = safety_intervention


class AcadosNMPC:
    """Optimize a sequence of actuator commands along a planner-supplied segment.

    The numerical Fossen simulator remains the applied plant. Wind, current and
    wave height are frozen at the current observation over this short horizon.
    An optional AIS prediction supplies moving hull constraints at every future
    knot. The first action is screened against all contacts over its actual step.
    """

    def __init__(
        self,
        dynamics: Fossen3DOFSaildroneDynamics,
        mode: str = "direct",
        *,
        dt_seconds: float = 5.0,
        no_go_angle_degrees: float = 45.0,
        build_directory: Path | str | None = None,
        horizon_seconds: float = 90.0,
        ais_capacity: int = 0,
    ) -> None:
        if mode not in {"direct", "los"}:
            raise ValueError("acados mode must be 'direct' or 'los'.")
        if not np.all(np.isfinite([dt_seconds, horizon_seconds])) or dt_seconds <= 0 or horizon_seconds <= 0:
            raise ValueError("acados control interval and horizon must be positive.")
        if not isinstance(ais_capacity, int) or ais_capacity < 0:
            raise ValueError("acados AIS capacity must be a non-negative integer.")
        self.mode = mode
        self.dynamics = dynamics
        self.dt_seconds = float(dt_seconds)
        self.stages = max(2, ceil(horizon_seconds / dt_seconds))
        self.no_go_angle_degrees = no_go_angle_degrees
        self.ais_capacity = ais_capacity
        self.model_name = f"maritime_fossen_{mode}"
        if ais_capacity:
            self.model_name += f"_ais_{ais_capacity}"
        self.fallback = SaildroneMPC(no_go_angle_degrees=no_go_angle_degrees)
        self._warm_commands: np.ndarray | None = None
        self._warm_bow: float | None = None

        if build_directory is None:
            self._temporary_build = TemporaryDirectory(prefix="maritime_acados_")
            build_path = Path(self._temporary_build.name)
        else:
            build_name = f"{mode}_ais_{ais_capacity}" if ais_capacity else mode
            build_path = Path(build_directory).resolve() / build_name
            build_path.mkdir(parents=True, exist_ok=True)
        self.lower = np.radians([dynamics.parameters.rudder_min, dynamics.parameters.flap_min])
        self.upper = np.radians([dynamics.parameters.rudder_max, dynamics.parameters.flap_max])
        self.slew = np.radians([
            self.fallback.max_rudder_change_per_update_degrees,
            self.fallback.max_flap_change_per_update_degrees,
        ])
        self._solver, self._discrete_step = self._build_solver(build_path)

    def _build_solver(self, directory: Path):
        try:
            import casadi as ca
            from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
        except (ImportError, OSError) as error:
            raise RuntimeError(
                "acados NMPC requires CasADi and acados_template with built acados libraries."
            ) from error

        from .acados_fossen import discrete_fossen

        state = ca.SX.sym("x", 9)
        command = ca.SX.sym("u", 2)
        parameter_count = _NAVIGATION_PARAMETERS + self.ais_capacity * _AIS_PARAMETERS
        parameters = ca.SX.sym("p", parameter_count)
        model = AcadosModel()
        model.name = self.model_name
        model.x = state
        model.u = command
        model.p = parameters
        model.disc_dyn_expr = discrete_fossen(self.dynamics, state, command, parameters[:5], self.dt_seconds)

        position = state[:2]
        heading = state[2]
        goal = parameters[5:7]
        tangent = parameters[7:9]
        normal = ca.vertcat(tangent[1], -tangent[0])
        along = ca.dot(goal - position, tangent)
        cross = ca.dot(position - goal, normal)
        remaining = ca.fmax(along, 0.0)
        overshoot = ca.fmax(-along, 0.0)
        if self.mode == "los":
            lookahead = ca.fmin(ca.fmax(along, 10.0), 60.0)
            limit = lookahead * tan(radians(15.0))
            correction = ca.if_else(parameters[12] > 0.5, ca.fmin(ca.fmax(-cross, -limit), limit), -cross)
            preview = tangent * lookahead + normal * correction
            reference = preview / ca.sqrt(ca.sumsqr(preview) + 1e-6)
        else:
            reference = tangent

        ground = ca.vertcat(
            state[3] * ca.sin(heading) + state[4] * ca.cos(heading) + parameters[2],
            state[3] * ca.cos(heading) - state[4] * ca.sin(heading) + parameters[3],
        )
        along_speed = ca.dot(ground, reference)
        expected_speed = ca.fmin(0.65, ca.sqrt(ca.sumsqr(goal - position) + 1e-6) / 25.0)
        cross_track_scale = 25.0
        if self.mode == "direct":
            # A planner-supplied bow comes with a feasible ground track at
            # its own speed, including when reaching across a strong current.
            planned_bow = parameters[13] > 0.5
            expected_speed = ca.if_else(planned_bow, 0.0, expected_speed)
            cross_track_scale = ca.if_else(planned_bow, 8.0, cross_track_scale)
        forward_shortfall = ca.fmax(0.0, expected_speed - along_speed)
        lateral_velocity = ground[0] * reference[1] - ground[1] * reference[0]
        bow = parameters[9]
        bow_terms = parameters[10] * ca.vertcat(ca.sin(heading) - ca.sin(bow), ca.cos(heading) - ca.cos(bow))
        wind_cosine = -ca.dot(parameters[:2], ca.vertcat(ca.sin(heading), ca.cos(heading))) / ca.sqrt(ca.sumsqr(parameters[:2]) + 1e-6)
        no_go = parameters[11] * ca.fmax(0.0, wind_cosine - cos(radians(self.no_go_angle_degrees)))
        tracking = ca.if_else(parameters[11] < 0.5, 0.45, 1.0)
        negative_surge = ca.fmax(0.0, -state[3]) / 0.25
        slew = command - state[7:9]
        position_terms = ca.vertcat(cross / cross_track_scale, remaining / 90.0, overshoot / 35.0)
        model.cost_y_expr = ca.vertcat(
            tracking * position_terms, tracking * forward_shortfall / 0.35,
            tracking * lateral_velocity / 0.35, bow_terms, no_go, state[5] / 0.35,
            negative_surge, slew / ca.DM(self.slew),
            command / ca.DM(np.maximum(np.abs(self.lower), np.abs(self.upper))),
        )
        model.cost_y_expr_e = ca.vertcat(position_terms, bow_terms)
        model.con_h_expr = slew
        model.con_h_expr_0 = slew

        ocp = AcadosOcp()
        ocp.model = model
        ocp.parameter_values = np.zeros(parameter_count)
        ocp.constraints.x0 = np.zeros(9)
        ocp.constraints.idxbu = np.arange(2)
        ocp.constraints.lbu = self.lower
        ocp.constraints.ubu = self.upper
        ocp.constraints.lh = -self.slew
        ocp.constraints.uh = self.slew
        ocp.constraints.lh_0 = -self.slew
        ocp.constraints.uh_0 = self.slew
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        weights = [4.0, 0.6, 0.5, 2.5, 3.0, 0.8, 0.8, 15.0, 0.12, 3.5, 0.2, 0.12, 0.06, 0.03]
        terminal_weights = [10.0, 4.0, 4.0, 1.5, 1.5]
        if self.ais_capacity:
            hull, buffer = [], []
            for slot in range(self.ais_capacity):
                start = _NAVIGATION_PARAMETERS + _AIS_PARAMETERS * slot
                target = parameters[start : start + _AIS_PARAMETERS]
                separation_squared = ca.sumsqr(position - target[:2])
                active = target[4] > 0.5
                hull_ratio = separation_squared / ca.fmax(target[2], 1e-6) ** 2
                buffer_ratio = ca.sqrt(separation_squared + 1e-8) / ca.fmax(target[3], 1e-6)
                hull.append(ca.if_else(active, hull_ratio - 1.0, 1.0))
                buffer.append(ca.if_else(active, ca.fmax(0.0, 1.0 - buffer_ratio), 0.0))
            hull_constraints = ca.vertcat(*hull)
            buffer_residuals = ca.vertcat(*buffer)
            # The fixed initial state may already be inside a domain. Physical
            # hull separation applies at future knots, including the terminal one.
            model.con_h_expr = ca.vertcat(slew, hull_constraints)
            model.con_h_expr_e = hull_constraints
            ocp.constraints.lh = np.concatenate((-self.slew, np.zeros(self.ais_capacity)))
            ocp.constraints.uh = np.concatenate((self.slew, np.full(self.ais_capacity, 1e10)))
            ocp.constraints.lh_e = np.zeros(self.ais_capacity)
            ocp.constraints.uh_e = np.full(self.ais_capacity, 1e10)
            model.cost_y_expr = ca.vertcat(model.cost_y_expr, buffer_residuals)
            model.cost_y_expr_e = ca.vertcat(model.cost_y_expr_e, buffer_residuals)
            weights.extend([_AIS_BUFFER_WEIGHT] * self.ais_capacity)
            terminal_weights.extend([_AIS_BUFFER_WEIGHT] * self.ais_capacity)
        ocp.cost.W = np.diag(weights)
        ocp.cost.yref = np.zeros(model.cost_y_expr.numel())
        ocp.cost.W_e = np.diag(terminal_weights)
        ocp.cost.yref_e = np.zeros(model.cost_y_expr_e.numel())
        ocp.solver_options.N_horizon = self.stages
        ocp.solver_options.tf = self.stages * self.dt_seconds
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.qp_solver_cond_N = min(6, self.stages)
        ocp.solver_options.levenberg_marquardt = 1e-3
        ocp.solver_options.print_level = 0
        export = directory / "c_generated_code"
        if hasattr(ocp, "code_gen_options"):
            ocp.code_gen_options.code_export_directory = str(export)
            ocp.code_gen_options.json_file = "ocp.json"
        else:
            ocp.code_export_directory = str(export)
        try:
            if hasattr(ocp, "code_gen_options"):
                reusable = (export / "ocp.json").is_file() and any(
                    file.suffix in {".dylib", ".so", ".dll"}
                    for file in export.glob(f"*acados_ocp_solver_*{model.name}*")
                )
                solver = AcadosOcpSolver(ocp, build=not reusable, generate=not reusable, verbose=False)
            else:
                solver = AcadosOcpSolver(ocp, json_file=str(directory / "ocp.json"), verbose=False)
        except (AttributeError, OSError, RuntimeError, ValueError) as error:
            raise RuntimeError(f"Unable to build the acados {self.mode} solver: {error}") from error
        return solver, ca.Function("maritime_discrete_step", [state, command, parameters], [model.disc_dyn_expr])

    def _stage_parameters(
        self, state: BoatState, observation: WorldObservation, desired_heading: float,
        plan: HarmonicPlan | None, goal: Waypoint | None, leg_start: Waypoint | None,
    ) -> np.ndarray:
        if observation.wind is None:
            raise ValueError("acados NMPC requires a wind observation.")
        if goal is None and plan is not None:
            goal = plan.navigation_target
        heading = radians(desired_heading)
        direction = np.array([sin(heading), cos(heading)])
        endpoint = 115.0 * direction
        if goal is not None:
            endpoint = np.array(local_east_north_metres(goal.latitude, goal.longitude, state.latitude, state.longitude))
        if leg_start is not None:
            origin = np.array(local_east_north_metres(leg_start.latitude, leg_start.longitude, state.latitude, state.longitude))
            segment = endpoint - origin
            if np.linalg.norm(segment) > 1e-6:
                direction = segment / np.linalg.norm(segment)
        wind, current, wave = observation.wind, observation.current, observation.wave
        bow_heading = None if plan is None else plan.desired_bow_heading
        has_planned_bow = bow_heading is not None
        wind_angle = (desired_heading - wind.direction + 180.0) % 360.0 - 180.0
        if bow_heading is None and self.no_go_angle_degrees <= abs(wind_angle) < 90.0:
            tack_angle = min(self.no_go_angle_degrees + 5.0, 89.0)
            bow_heading = wind.direction + np.copysign(tack_angle, wind_angle)
        bow_reference = desired_heading if bow_heading is None else bow_heading
        close_hauled = bow_heading is not None and self.fallback._angular_error(bow_reference, wind.direction) < 90.0
        turning = self.fallback._crossing_wind(state.heading, bow_reference, wind.direction)
        if has_planned_bow and self._backing_away(state, bow_reference):
            turning = True
        return np.array([
            wind.u, wind.v, 0.0 if current is None else current.u, 0.0 if current is None else current.v,
            0.0 if wave is None else wave.height, *endpoint, *direction, radians(bow_reference),
            (3.5 if turning else 1.0) if bow_heading is not None else 0.0,
            0.15 if turning else 1.0, float(close_hauled), float(has_planned_bow),
        ])

    @staticmethod
    def _backing_away(state: BoatState, bow_degrees: float) -> bool:
        return (
            state.surge_speed is not None and state.surge_speed < -0.05
            and abs((state.heading - bow_degrees + 180.0) % 360.0 - 180.0) > 60.0
        )

    def _ais_stage_parameters(
        self, state: BoatState, parameters: np.ndarray, ais_context: AISPrediction | None,
    ) -> np.ndarray:
        """Pack the same ranked contact slots at each time knot, including N."""

        values = np.zeros((self.stages + 1, _NAVIGATION_PARAMETERS + self.ais_capacity * _AIS_PARAMETERS))
        values[:, :_NAVIGATION_PARAMETERS] = parameters
        if ais_context is None or len(ais_context.positions_metres) == 0:
            return values

        times = np.arange(self.stages + 1) * self.dt_seconds
        centres = ais_context.centres_at(times)
        clearance_radii = ais_context.radii_at(times)
        heading = radians(state.heading)
        surge = state.speed if state.surge_speed is None else state.surge_speed
        own_velocity = np.array([
            surge * sin(heading) + state.sway_speed * cos(heading) + parameters[2],
            surge * cos(heading) - state.sway_speed * sin(heading) + parameters[3],
        ])
        distances = np.linalg.norm(centres - times[:, None, None] * own_velocity, axis=-1)
        predicted_margin = np.min(distances - clearance_radii, axis=0)
        selected = np.argsort(predicted_margin, kind="stable")[: self.ais_capacity]
        contacts = values[:, _NAVIGATION_PARAMETERS:].reshape(self.stages + 1, self.ais_capacity, _AIS_PARAMETERS)
        slots = contacts[:, : len(selected)]
        slots[:, :, :2] = centres[:, selected]
        slots[:, :, 2] = ais_context.hull_radii_metres[selected]
        slots[:, :, 3] = clearance_radii[:, selected]
        slots[:, :, 4] = 1.0
        return values

    def _seed(
        self, state: BoatState, parameters: np.ndarray, stage_parameters: np.ndarray | None = None,
    ) -> None:
        start = np.array([
            0.0, 0.0, radians(state.heading), state.speed if state.surge_speed is None else state.surge_speed,
            state.sway_speed, state.yaw_rate, radians(state.flap), radians(state.rudder), radians(state.flap),
        ])
        solver = self._solver
        solver.constraints_set(0, "lbx", start)
        solver.constraints_set(0, "ubx", start)
        turning = parameters[11] < 0.5
        bow = parameters[9]
        bow_change = pi if self._warm_bow is None else abs((bow - self._warm_bow + pi) % (2.0 * pi) - pi)
        cold_start = self._warm_commands is None
        planned_bow = parameters[13] > 0.5
        # Refresh the maneuver after a substantial planned bow change or if
        # the previous solution is sailing backwards with the wrong bow.
        seed_tack = (turning and (cold_start or bow_change > pi / 3.0)) or (
            planned_bow and (cold_start or bow_change > radians(25.0) or self._backing_away(state, degrees(bow)))
        )
        if cold_start or seed_tack:
            commands = np.tile(start[7:9], (self.stages, 1))
        else:
            commands = np.vstack((self._warm_commands[1:], self._warm_commands[-1]))
        node = start
        for index in range(self.stages):
            command = commands[index]
            if seed_tack:
                wind_from = atan2(-parameters[0], -parameters[1])
                tack_side = (bow - wind_from + pi) % (2.0 * pi) - pi
                straightening = (node[2] - bow + pi) % (2.0 * pi) - pi
                command = np.array([0.9 * straightening + 2.5 * node[5],
                                    self.upper[1] if tack_side > 0.0 else self.lower[1]])
            command = np.clip(command, np.maximum(self.lower, node[7:9] - self.slew),
                              np.minimum(self.upper, node[7:9] + self.slew))
            solver.set(index, "x", node)
            solver.set(index, "u", command)
            stage = parameters if stage_parameters is None else stage_parameters[index]
            solver.set(index, "p", stage)
            node = np.asarray(self._discrete_step(node, command, stage)).ravel()
        solver.set(self.stages, "x", node)
        solver.set(self.stages, "p", parameters if stage_parameters is None else stage_parameters[-1])

    def choose_action(
        self, dynamics: Fossen3DOFSaildroneDynamics, state: BoatState,
        observation: WorldObservation, desired_heading: float, *,
        local_grid: LocalGrid | None = None, harmonic_plan: HarmonicPlan | None = None,
        goal: Waypoint | None = None, leg_start: Waypoint | None = None,
        ais_context: AISPrediction | None = None,
    ) -> AcadosMPCResult:
        """Apply the first admissible acados action, or explicitly report fallback."""

        if dynamics.parameters != self.dynamics.parameters:
            raise ValueError("acados prediction and the Fossen plant must use the same parameters.")
        if ais_context is not None and len(ais_context.positions_metres):
            if not self.ais_capacity:
                raise ValueError("acados requires positive AIS capacity for moving contacts.")
            if not np.allclose(
                (ais_context.origin_latitude, ais_context.origin_longitude),
                (state.latitude, state.longitude), rtol=0.0, atol=1e-9,
            ):
                raise ValueError("AIS prediction must share the controller's initial position.")
        params = self._stage_parameters(state, observation, desired_heading, harmonic_plan, goal, leg_start)
        stages = self._ais_stage_parameters(state, params, ais_context) if self.ais_capacity else None
        before = perf_counter()
        status = -1
        first = None
        try:
            self._seed(state, params, stages)
            status = int(self._solver.solve())
            reason = None if status == 0 else f"acados status {status}"
            if reason is None:
                commands = np.asarray([self._solver.get(i, "u") for i in range(self.stages)])
                cost = float(self._solver.get_cost())
                previous = np.radians([state.rudder, state.flap])
                lower, upper = np.maximum(self.lower, previous - self.slew), np.minimum(self.upper, previous + self.slew)
                if not np.all(np.isfinite(commands)) or not np.isfinite(cost):
                    reason = "non-finite acados solution"
                elif np.any(commands[0] < lower - 1e-5) or np.any(commands[0] > upper + 1e-5):
                    reason = "first action exceeds actuator or slew bounds"
                else:
                    first = BoatAction(*map(float, np.degrees(np.clip(commands[0], lower, upper))))
        except (RuntimeError, ValueError) as error:
            reason = f"acados solve failed: {error}"

        intervention = None
        if first is not None:
            veto, recovery = self._first_step_veto(
                dynamics, state, first, observation, local_grid, harmonic_plan, ais_context,
            )
            if veto is None:
                try:
                    prediction = self._predicted_states(state, params)
                except (RuntimeError, ValueError) as error:
                    reason = f"acados solve failed: {error}"
                else:
                    self._warm_commands = commands
                    self._warm_bow = params[9]
                    return AcadosMPCResult(
                        first, cost, prediction, 1, status, (perf_counter() - before) * 1000,
                        safety_intervention="moving AIS clearance recovery" if recovery else None,
                    )
            elif ais_context is None:
                reason = f"acados first step {veto}"
            else:
                intervention = f"acados first step {veto}"

        solve_time = (perf_counter() - before) * 1000
        self._warm_commands = None
        fallback = self.fallback.choose_action(dynamics, state, observation, desired_heading,
                                               local_grid=local_grid, harmonic_plan=harmonic_plan)
        veto, recovery = self._first_step_veto(
            dynamics, state, fallback.action, observation, local_grid, harmonic_plan, ais_context,
        )
        if veto is not None:
            if ais_context is None:
                raise RuntimeError("Neither acados nor the baseline MPC found a clear first step.")
            detail = f"grid MPC first step {veto}"
            raise AISControlInfeasible(
                solver_status=status, solve_time_ms=solve_time, fallback_reason=reason,
                safety_intervention=f"{intervention}, {detail}" if intervention else detail,
            )
        return AcadosMPCResult(fallback.action, fallback.cost, fallback.predicted_states,
                               fallback.candidates_evaluated, status, solve_time, reason,
                               intervention or ("moving AIS clearance recovery" if recovery else None), "grid-mpc")

    def _first_step_veto(
        self, dynamics: Fossen3DOFSaildroneDynamics, state: BoatState, action: BoatAction,
        observation: WorldObservation, grid: LocalGrid | None, plan: HarmonicPlan | None,
        ais_context: AISPrediction | None,
    ) -> tuple[str | None, bool]:
        if ais_context is None:
            clear = self._first_step_is_clear(dynamics, state, action, observation, grid, plan)
            return (None if clear else "intersects a blocked cell"), False

        from .ais_guidance import assess_action

        chart_clear, assessment = assess_action(
            dynamics, state, observation, action, self.dt_seconds,
            grid, None if plan is None else plan.blocked, ais_context,
        )
        if not chart_clear:
            return "intersects a blocked cell", False
        if not (assessment.safe or assessment.recovering):
            return "violates moving AIS clearance", False
        return None, assessment.recovering

    def _first_step_is_clear(
        self, dynamics: Fossen3DOFSaildroneDynamics, initial: BoatState, action: BoatAction,
        observation: WorldObservation, grid: LocalGrid | None, plan: HarmonicPlan | None,
    ) -> bool:
        if grid is None or plan is None:
            return True

        def grid_position(state: BoatState) -> tuple[float, float]:
            east, north = local_east_north_metres(state.latitude, state.longitude, grid.origin_latitude, grid.origin_longitude)
            centre = (grid.cells - 1) / 2.0
            return centre - float(north) / grid.resolution, centre + float(east) / grid.resolution

        pieces = max(1, ceil(self.dt_seconds))
        state, previous = initial, grid_position(initial)
        for _ in range(pieces):
            state = dynamics.step(state, action, observation, self.dt_seconds / pieces)
            position = grid_position(state)
            if (not all(-0.5 < value < grid.cells - 0.5 for value in position)
                    or not LegacyPlannerAdapter._segment_is_clear(plan.blocked, previous, position)):
                return False
            previous = position
        return True

    def _predicted_states(self, initial: BoatState, parameters: np.ndarray) -> tuple[BoatState, ...]:
        knots = []
        for index in range(1, self.stages + 1):
            east, north, heading, surge, sway, yaw, flap, rudder, _ = self._solver.get(index, "x")
            latitude, longitude = offset_latitude_longitude(initial.latitude, initial.longitude, east, north)
            ground_e = surge * sin(heading) + sway * cos(heading) + parameters[2]
            ground_n = surge * cos(heading) - sway * sin(heading) + parameters[3]
            knots.append(BoatState(float(latitude), float(longitude), degrees(heading) % 360.0,
                                   hypot(ground_e, ground_n), degrees(rudder), degrees(flap),
                                   surge_speed=float(surge), sway_speed=float(sway), yaw_rate=float(yaw)))
        return tuple(knots)
