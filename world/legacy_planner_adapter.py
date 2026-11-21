"""Adapter from the preserved legacy harmonic planners to Maritime World APIs."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, radians, sin
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.ndimage import distance_transform_edt

from .drift_guidance import DriftGuidance
from .forecast_tacks import ForecastTackPlanner, steady_ground_polar
from .fossen import Fossen3DOFSaildroneDynamics
from .harmonic_planner import HarmonicPlan, HarmonicTacticalPlanner
from .legacy_planners import GammaHarmonicPlanner, HarmonicPlanner, WindAwareHarmonicPlanner
from .local_map import LegacyLocalMap
from .scenario import Waypoint
from .transformer_direction_policy import TransformerDirectionPolicy
from .waypoint_laylines import layline_candidates
from .world import WorldObservation
from ._geo import local_east_north_metres, offset_latitude_longitude

if TYPE_CHECKING:
    from .forecast_eta_release import ForecastETARelease
    from .forecast_hold import ForecastHoldPlanner
    from .temporal_recovery import TemporalHoldDecision, TemporalRecoveryGuidance


LegacyPlannerMode = Literal["standard", "gamma", "diffusion", "transformer_diffusion"]


class _LegacyDiffusionGrid:
    """Minimal object expected by ``WindAwareHarmonicPlanner``."""

    def __init__(self, occupancy_grid: np.ndarray) -> None:
        self.occupancy_grid = np.asarray(occupancy_grid, dtype=np.int8)
        self.grid_size = int(self.occupancy_grid.shape[0])


@dataclass
class LegacyPlannerAdapter:
    """Run one of the old field solvers while returning current ``HarmonicPlan``.

    The legacy solver is responsible only for the tactical path/potential.
    AIS prediction and restriction masking are retained from the Maritime World
    safety layer, and actuator commands remain the responsibility of MPC.
    """

    mode: LegacyPlannerMode = "gamma"
    boundary: str = "mixed"
    no_go_angle_degrees: float = 45.0
    transformer_checkpoint: Path | None = None
    transformer_device: str | None = None
    forecast_tacks: ForecastTackPlanner | None = None
    fossen_dynamics: Fossen3DOFSaildroneDynamics | None = None
    arrival_radius: float = 50.0
    limit_drift: bool = False
    commit_recovery_tacks: bool = False
    temporal_recovery: TemporalRecoveryGuidance | None = None
    forecast_hold: ForecastHoldPlanner | None = None
    closest_waypoint_layline: bool = False
    prioritize_waypoint_layline: bool = False
    compare_waypoint_layline_eta: bool = False
    forecast_release: ForecastETARelease | None = None

    def __post_init__(self) -> None:
        mode = str(self.mode).lower()
        if mode not in {"standard", "gamma", "diffusion", "transformer_diffusion"}:
            raise ValueError(
                "mode must be 'standard', 'gamma', 'diffusion', or "
                "'transformer_diffusion'."
            )
        if not np.isfinite(self.arrival_radius) or self.arrival_radius <= 0:
            raise ValueError("arrival_radius must be finite and positive")
        if self.commit_recovery_tacks and not self.limit_drift:
            raise ValueError("Recovery tack commitment requires drift limiting.")
        if self.temporal_recovery is not None and not self.limit_drift:
            raise ValueError("Temporal recovery requires observation-based drift limiting.")
        if self.forecast_hold is not None and not self.limit_drift:
            raise ValueError("Forecast holding trims require drift limiting.")
        if self.forecast_release is not None and not self.limit_drift:
            raise ValueError("Forecast ETA hold release requires drift limiting.")
        if (self.limit_drift or self.closest_waypoint_layline
                or self.prioritize_waypoint_layline or self.compare_waypoint_layline_eta) and self.fossen_dynamics is None:
            raise ValueError("Drift limiting and closest waypoint laylines require Fossen dynamics.")
        if self.compare_waypoint_layline_eta and (self.forecast_tacks is None or self.prioritize_waypoint_layline):
            raise ValueError("Layline ETA comparison requires forecast tacking and its own selection mode.")
        self._drift_guidance = (
            DriftGuidance(self.fossen_dynamics, self.no_go_angle_degrees)
            if self.limit_drift and self.fossen_dynamics is not None else None
        )
        self.mode = mode  # type: ignore[assignment]
        if self.mode == "transformer_diffusion":
            if self.transformer_checkpoint is None:
                raise ValueError(
                    "transformer_checkpoint is required for transformer_diffusion."
                )
            self._transformer_policy = TransformerDirectionPolicy(
                self.transformer_checkpoint,
                device=self.transformer_device,
            )
        else:
            self._transformer_policy = None
        self.transformer_predictions = 0
        self.transformer_acceptances = 0
        self.transformer_fallbacks = 0
        self._tacking_goal: Waypoint | None = None
        self._tack_side = 1.0
        self._opposite_layline: float | None = None
        self._steady_polar: tuple[int, np.ndarray, np.ndarray, np.ndarray] | None = None
        self._recovery_tack: tuple[Waypoint, int, float] | None = None
        self._temporal_decision: TemporalHoldDecision | None = None
        self.temporal_recovery_telemetry: dict[str, str | float | None] = {}
        self.forecast_hold_telemetry: dict[str, str | float | bool | None] = {}
        self.forecast_release_telemetry: dict[str, str | float] = {}
        self.waypoint_layline_telemetry: dict[str, str | float | None] = {}
        # Reuse only the current obstacle-mask/AIS-prediction utility; the
        # potential solve/path extraction below is entirely legacy code.
        self._safety_mask = HarmonicTacticalPlanner(no_go_angle_degrees=self.no_go_angle_degrees)

    def suspend_open_water_recovery(self, reason: str = "AIS encounter") -> None:
        """Discard the live hold or tack and record why AIS took priority."""
        self._recovery_tack = None
        self._temporal_decision = None
        self.waypoint_layline_telemetry = {}
        self.forecast_release_telemetry = {}
        if self.forecast_tacks is not None:
            self.forecast_tacks.suspend(reason)
        if self._drift_guidance is not None:
            self._drift_guidance.suspend()
        if self.forecast_release is not None:
            self.forecast_release.invalidate_selection()
        if self.temporal_recovery is not None:
            self.temporal_recovery.reset_release_confirmation()
            self.temporal_recovery_telemetry.update(
                temporal_recovery_applied_action="planner",
                temporal_recovery_source="classical fallback",
                temporal_recovery_fallback_reason=reason,
            )
        if self.forecast_hold is not None:
            self.forecast_hold.invalidate_selection()
            window = self.forecast_hold_telemetry.get("forecast_hold_window_utc")
            self.forecast_hold_telemetry = {
                "forecast_hold_source": "classical fallback",
                "forecast_hold_fallback_reason": reason,
            }
            if window is not None:
                self.forecast_hold_telemetry["forecast_hold_window_utc"] = window

    def plan(
        self,
        local_map: LegacyLocalMap,
        observation: WorldObservation,
        goal: Waypoint,
        *,
        extra_blocked: np.ndarray | None = None,
        time_aware_ais: bool = False,
        suspend_recovery: bool | None = None,
    ) -> HarmonicPlan:
        """Build a reference; time-aware calls opt in to screened open water explicitly."""
        grid = local_map.tensor
        start = self._boat_cell(grid.cells)
        blocked = self._safety_mask.obstacle_mask(
            grid,
            observation,
            extra_blocked=local_map.static_blocked,
            include_ais=not time_aware_ais,
        )
        if extra_blocked is not None:
            if extra_blocked.shape != blocked.shape:
                raise ValueError("extra_blocked must match the local map shape.")
            blocked |= np.asarray(extra_blocked, dtype=bool)
        blocked[start] = False

        target, target_on_boundary = self._goal_cell(grid, goal)
        target = self._nearest_free(blocked, target, boundary_only=target_on_boundary)
        blocked[target] = False

        distance_to_obstacle = distance_transform_edt(~blocked)
        wind_from_deg = None if observation.wind is None else float(observation.wind.direction)
        wind_from_rad = None if wind_from_deg is None else radians(wind_from_deg)
        occupancy = blocked.astype(np.int8)
        east, north = local_east_north_metres(
            goal.latitude, goal.longitude, grid.origin_latitude, grid.origin_longitude
        )
        direct_heading = degrees(atan2(east, north)) % 360.0
        clear_approach = (
            not target_on_boundary
            and hypot(east, north) > 1e-6
            and self._segment_is_clear(
                blocked, start, (start[0] - north / grid.resolution, start[1] + east / grid.resolution)
            )
        )
        desired_heading = None
        desired_bow = None
        navigation_target = None
        forecast = None
        drift_mode = None
        drift_trim = None
        drift_range = None
        gate_vmg = None
        drift = self._drift_guidance
        temporal = self.temporal_recovery
        self.temporal_recovery_telemetry = {}
        self.forecast_hold_telemetry = {}
        self.forecast_release_telemetry = {}
        self.waypoint_layline_telemetry = {}
        screened_traffic = time_aware_ais and suspend_recovery is not None and not bool(suspend_recovery)
        if time_aware_ais:
            recovery_suspended = not screened_traffic
        else:
            recovery_suspended = bool(observation.ais_targets) or bool(suspend_recovery)
        if temporal is not None:
            temporal.observe(observation)
            bypass_reason = "no clear open-water approach"
            if observation.wind is None or observation.current is None:
                bypass_reason = "missing observed weather"
            self.temporal_recovery_telemetry = {
                "temporal_recovery_source": "classical fallback",
                "temporal_recovery_applied_action": "planner",
                "temporal_recovery_fallback_reason": bypass_reason,
            }
        if (drift is not None or self.forecast_tacks is not None or self.closest_waypoint_layline
                or self.prioritize_waypoint_layline) and recovery_suspended:
            self.suspend_open_water_recovery()
        if (self.prioritize_waypoint_layline and not recovery_suspended
                and not target_on_boundary and hypot(east, north) > 1e-6):
            layline = self._steady_open_water_reference(
                observation, np.array([east, north]), grid, blocked, start,
                recovery_goal=goal, allow_screened_contacts=screened_traffic, closest_layline=True,
            )
            if layline is not None:
                desired_heading, desired_bow, navigation_target = layline
                self.waypoint_layline_telemetry["waypoint_layline_policy"] = "priority"
                if drift is not None:
                    drift.suspend()
                if self.forecast_hold is not None:
                    self.forecast_hold.invalidate_selection()
        commit_tacks = self.commit_recovery_tacks and self.forecast_tacks is None
        if (
            desired_heading is None and drift is not None and not recovery_suspended
            and observation.wind is not None and observation.current is not None
            and hypot(east, north) > 1e-6
        ):
            vector = np.array([east, north])
            # Check the in-grid part of the actual ray even after a drift
            # excursion put the waypoint beyond the planning window.
            if target_on_boundary:
                scale = min(1.0, (grid.cells // 2 - 1) * grid.resolution / np.max(np.abs(vector)))
                end = start[0] - scale * north / grid.resolution, start[1] + scale * east / grid.resolution
                safe_corridor = self._segment_is_clear(blocked, start, end)
            else:
                safe_corridor = clear_approach
            if safe_corridor:
                if temporal is None:
                    drift.update(observation, goal, vector)
                else:
                    def choose_hold(was_holding: bool, classical_hold: bool) -> bool:
                        anchor = drift.anchor if was_holding else Waypoint(observation.boat.latitude, observation.boat.longitude)
                        hold_available = any(self._clear_ground_reference(trim.velocity, grid, blocked, start) is not None
                                             for trim in drift.ranked_modes(observation, anchor=anchor))
                        choice = temporal.decide(observation, vector, was_holding=was_holding,
                                                 classical_hold=classical_hold, gate_vmg=drift.gate_vmg,
                                                 hold_available=hold_available, goal=goal)
                        self._temporal_decision = choice
                        return choice.hold

                    drift.update(observation, goal, vector, choose_hold=choose_hold)
                    if self._temporal_decision is not None:
                        self.temporal_recovery_telemetry = self._temporal_decision.telemetry()
                gate_vmg = drift.gate_vmg
                drift_range = drift.distance_from_anchor(observation)
                if drift.mode == "holding":
                    self._recovery_tack = None
                    forecast_trim = None
                    if self.forecast_hold is not None:
                        forecast_trim = self.forecast_hold.reference(
                            observation, goal, vector, drift.ranked_modes(observation),
                            chart_allows=lambda mode: self._clear_ground_reference(mode.velocity, grid, blocked, start) is not None,
                        )
                        self.forecast_hold_telemetry = self.forecast_hold.telemetry.copy()
                    trim_candidates = (forecast_trim,) if forecast_trim is not None else drift.ranked_modes(observation)
                    for trim in trim_candidates:
                        endpoint = self._clear_ground_reference(trim.velocity, grid, blocked, start)
                        if endpoint is not None:
                            drift_trim = trim
                            drift_mode = "holding"
                            desired_heading, desired_bow, navigation_target = trim.course, trim.bow, endpoint
                            break
                    if self.forecast_release is not None and drift_trim is not None:
                        release = self.forecast_release.release_reference(
                            observation, goal, vector, drift_trim, drift.ranked_modes(observation),
                            chart_allows=lambda mode: self._clear_ground_reference(
                                mode.velocity, grid, blocked, start,
                            ) is not None,
                        )
                        if release is not None:
                            drift_mode, drift_trim = "forecast_release", release
                            desired_heading, desired_bow = release.course, release.bow
                            navigation_target = self._clear_ground_reference(release.velocity, grid, blocked, start)
                            self.forecast_release_telemetry = self.forecast_release.telemetry.copy()
                elif drift.mode == "recovering" or (commit_tacks and drift.released):
                    drift_mode = "recovering"
                    if target_on_boundary or commit_tacks:
                        steady = self._steady_open_water_reference(
                            observation, vector, grid, blocked, start,
                            recovery_vmg=drift.gate_vmg,
                            recovery_goal=goal if commit_tacks else None,
                            allow_screened_contacts=screened_traffic,
                        )
                        if steady is not None:
                            desired_heading, desired_bow, navigation_target = steady
                if temporal is not None and self._temporal_decision is not None and self._temporal_decision.hold and drift_trim is None:
                    self.temporal_recovery_telemetry.update(
                        temporal_recovery_source="classical fallback",
                        temporal_recovery_applied_action="planner",
                        temporal_recovery_fallback_reason="no chart-clear hold trim",
                    )
        if desired_heading is not None:
            self._tacking_goal = None
            self._opposite_layline = None
        elif clear_approach:
            forecast_turn_in_progress = False
            if self.forecast_tacks is not None and not recovery_suspended:
                if self.compare_waypoint_layline_eta:
                    forecast_turn_in_progress = self.forecast_tacks.turn_in_progress(observation, goal)
                forecast = self.forecast_tacks.reference(observation, goal)
            if forecast is not None:
                distance = min(grid.resolution, hypot(east, north))
                offset_east = distance * sin(radians(forecast.course))
                offset_north = distance * cos(radians(forecast.course))
                end = start[0] - offset_north / grid.resolution, start[1] + offset_east / grid.resolution
                if self._segment_is_clear(blocked, start, end):
                    latitude, longitude = offset_latitude_longitude(
                        grid.origin_latitude, grid.origin_longitude, offset_east, offset_north,
                    )
                    desired_heading = forecast.course
                    desired_bow = forecast.bow
                    navigation_target = Waypoint(float(latitude), float(longitude))
                    self._tacking_goal = None
                    self._opposite_layline = None
                else:
                    forecast = None
            if self.compare_waypoint_layline_eta and forecast is not None and not forecast_turn_in_progress:
                layline = self._steady_open_water_reference(
                    observation, np.array([east, north]), grid, blocked, start,
                    recovery_goal=goal, allow_screened_contacts=screened_traffic, closest_layline=True,
                )
                comparison = self.waypoint_layline_telemetry
                if layline is not None:
                    forecast_eta, layline_eta = self.forecast_tacks.compare_layline_arrival(
                        observation, goal, layline[1], comparison["waypoint_layline_second_bow_deg"],
                        comparison["waypoint_layline_first_sail_s"],
                    )
                    if (forecast_eta is not None and layline_eta is not None
                            and np.isfinite((forecast_eta, layline_eta)).all()
                            and forecast_eta - layline_eta > 1e-6):
                        comparison.update(
                            waypoint_layline_policy="eta",
                            waypoint_layline_steady_eta_s=comparison["waypoint_layline_eta_s"],
                            waypoint_layline_eta_s=layline_eta,
                            waypoint_layline_forecast_eta_s=forecast_eta,
                            waypoint_layline_eta_advantage_s=forecast_eta - layline_eta,
                            waypoint_layline_comparison_radius_m=self.arrival_radius,
                        )
                        desired_heading, desired_bow, navigation_target = layline
                        forecast = None
                    else:
                        comparison.clear()
                        self._recovery_tack = None
            forecast_protects_approach = self.closest_waypoint_layline and forecast is not None and (
                forecast.closest_metres <= self.arrival_radius
                or forecast.tack_in_minutes is not None
                or self.forecast_tacks.maneuver_telemetry.get("forecast_maneuver_commit_active")
            )
            if (self.closest_waypoint_layline and not self.prioritize_waypoint_layline
                    and not self.compare_waypoint_layline_eta
                    and not forecast_protects_approach and not recovery_suspended
                    and (drift is None or drift.mode != "holding")):
                layline = self._steady_open_water_reference(
                    observation, np.array([east, north]), grid, blocked, start,
                    recovery_goal=goal, allow_screened_contacts=screened_traffic, closest_layline=True,
                )
                if layline is not None:
                    desired_heading, desired_bow, navigation_target = layline
                    forecast = None
                    self._tacking_goal = None
                    self._opposite_layline = None
            if forecast is None and desired_heading is None:
                steady = None
                if not time_aware_ais or (drift is not None and screened_traffic):
                    steady = self._steady_open_water_reference(
                        observation, np.array([east, north]), grid, blocked, start,
                        allow_screened_contacts=screened_traffic,
                    )
                if steady is None:
                    desired_heading, navigation_target = self._open_water_reference(
                        observation, goal, direct_heading, hypot(east, north), grid, blocked, start
                    )
                else:
                    desired_heading, desired_bow, navigation_target = steady
                    self._tacking_goal = None
                    self._opposite_layline = None
        else:
            self._tacking_goal = None

        if self.forecast_tacks is not None and forecast is None:
            if recovery_suspended:
                reason = "AIS encounter"
            elif drift_trim is not None:
                reason = "holding trim selected"
            elif not clear_approach:
                reason = "no clear nominal approach"
            elif self.waypoint_layline_telemetry:
                reason = "closest waypoint layline selected"
            else:
                reason = "forecast reference unavailable or chart rejected"
            if self.waypoint_layline_telemetry.get("waypoint_layline_policy") == "eta":
                self.forecast_tacks.yield_to_layline()
            else:
                self.forecast_tacks.suspend(reason)

        if self.mode == "standard":
            solver = HarmonicPlanner(
                grid_size=grid.cells,
                boundary=self.boundary,
                no_go_angle=radians(self.no_go_angle_degrees),
            )
            path = solver.plan(
                occupancy,
                start,
                target,
                dist_to_obst=distance_to_obstacle,
                wind_direction=wind_from_rad,
                use_virtual_obstacle=wind_from_rad is not None,
                use_wind_awareness=wind_from_rad is not None,
                boat_speed=max(0.5, float(observation.boat.speed)),
            )
            potential = np.asarray(solver.potential, dtype=np.float64)
            iterations = int(solver.iterations)

        elif self.mode == "gamma":
            solver = GammaHarmonicPlanner(
                grid_width=grid.cells - 1,
                grid_height=grid.cells - 1,
                resolution=1.0,
                wind_direction_deg=wind_from_deg,
                no_go_angle_deg=self.no_go_angle_degrees,
            )
            path = solver.plan(occupancy, start, target, dist_to_obst=distance_to_obstacle)
            potential = np.asarray(
                solver.potential_rc if solver.potential_rc is not None else solver.potential,
                dtype=np.float64,
            )
            iterations = int(solver.iterations)

        else:
            diffusion_grid = _LegacyDiffusionGrid(occupancy)
            solver = WindAwareHarmonicPlanner(
                diffusion_grid,
                target,
                wind_from_rad,
                no_go_angle=radians(self.no_go_angle_degrees),
                boundary=self.boundary,
            )
            solver.solve_potential(dist_to_obst=distance_to_obstacle)
            path = solver.extract_diffusion_path(start)
            potential = np.asarray(solver.potential, dtype=np.float64)
            iterations = 1000

            if self._transformer_policy is not None and desired_heading is None:
                feature_field = self._finite_field(solver.diffusion_field)
                finite_potential = self._finite_field(potential)
                prediction = self._transformer_policy.predict_direction_with_diagnostics(
                    occupancy,
                    feature_field,
                    finite_potential,
                    start,
                    target,
                    0.0 if wind_from_deg is None else wind_from_deg,
                    self.no_go_angle_degrees,
                )
                self.transformer_predictions += 1
                predicted_path = self._accepted_transformer_path(
                    blocked,
                    start,
                    target,
                    prediction,
                    wind_from_degrees=wind_from_deg,
                    no_go_angle_degrees=self.no_go_angle_degrees,
                )
                if predicted_path is not None:
                    path = predicted_path
                    self.transformer_acceptances += 1
                else:
                    self.transformer_fallbacks += 1

        path_cells = tuple((int(round(p[0])), int(round(p[1]))) for p in (path or []))
        if navigation_target == goal:
            path_cells = self._safety_mask._straight_path_if_feasible(
                blocked, start, target, None
            ) or (start,)
        if len(path_cells) < 2 and navigation_target is None:
            fallback = self._fallback_neighbor(blocked, potential, start, target)
            if fallback is not None:
                path_cells = (start, fallback)
        if not path_cells:
            path_cells = (start,)
        elif path_cells[0] != start:
            path_cells = (start,) + path_cells
        if desired_heading is None:
            desired_heading = self._path_heading(path_cells)
        converged = bool(np.all(np.isfinite(potential)) and desired_heading is not None)
        return HarmonicPlan(
            potential=potential,
            blocked=blocked,
            path_cells=path_cells,
            desired_heading=desired_heading,
            target_cell=target,
            target_on_boundary=target_on_boundary,
            converged=converged,
            iterations=iterations,
            navigation_target=navigation_target,
            desired_bow_heading=desired_bow,
            forecast_closest_metres=None if forecast is None else forecast.closest_metres,
            forecast_tack_in_minutes=None if forecast is None else forecast.tack_in_minutes,
            drift_mode=drift_mode,
            steady_speed_mps=None if drift_trim is None else float(np.linalg.norm(drift_trim.velocity)),
            gate_vmg_mps=gate_vmg,
            drift_range_m=drift_range,
            steady_rudder_degrees=None if drift_trim is None else drift_trim.rudder,
            steady_flap_degrees=None if drift_trim is None else drift_trim.flap,
        )

    def _clear_ground_reference(self, velocity: np.ndarray, grid, blocked, start) -> Waypoint | None:
        speed = np.linalg.norm(velocity)
        offset_east, offset_north = velocity * (grid.resolution / max(speed, 1e-8))
        end = start[0] - offset_north / grid.resolution, start[1] + offset_east / grid.resolution
        if not self._segment_is_clear(blocked, start, end):
            return None
        return Waypoint(*offset_latitude_longitude(
            grid.origin_latitude, grid.origin_longitude, offset_east, offset_north,
        ))

    def _steady_open_water_reference(
        self, observation, target: np.ndarray, grid, blocked, start, *, recovery_vmg: float | None = None,
        recovery_goal: Waypoint | None = None, allow_screened_contacts: bool = False,
        closest_layline: bool = False,
    ) -> tuple[float, float, Waypoint] | None:
        """Find a ground layline from the observed forcing when drift matters."""
        wind, current = observation.wind, observation.current
        interval = ForecastTackPlanner.interval_seconds
        radius = self.arrival_radius / 2.0
        if (
            self.fossen_dynamics is None or wind is None or current is None
            or (observation.ais_targets and not allow_screened_contacts)
            or (recovery_goal is None and recovery_vmg is None and hypot(current.u, current.v) * interval < radius)
        ):
            return None

        window = int(observation.time.astype("datetime64[s]").astype(np.int64)) // interval
        if self._steady_polar is None or self._steady_polar[0] != window:
            angles = tuple(min(self.no_go_angle_degrees + extra, 179.0)
                           for extra in (0.5, 5, 15, 30, 50, 80, 115))
            offsets = np.array(sorted({-angle for angle in angles} | set(angles)))
            velocities, bows, drift = steady_ground_polar(observation, self.fossen_dynamics, offsets)
            # Interpolate adjacent equilibria on the same side of the wind.
            # Never interpolate across the upwind cone or an invalid mode.
            valid = np.isfinite(velocities).all(axis=1)
            neighbors = np.flatnonzero(valid[:-1] & valid[1:] & (offsets[:-1] * offsets[1:] > 0))
            fractions = np.array([0.25, 0.5, 0.75])
            between = (velocities[neighbors, None, :] * (1.0 - fractions[None, :, None])
                       + velocities[neighbors + 1, None, :] * fractions[None, :, None])
            between_bows = wind.direction + offsets[neighbors, None] + (
                offsets[neighbors + 1] - offsets[neighbors]
            )[:, None] * fractions[None, :]
            velocities = np.concatenate((velocities[valid], between.reshape(-1, 2)))
            bows = np.concatenate((bows[valid], between_bows.ravel() % 360))
            moving = np.sum(velocities**2, axis=1) > 0.02**2
            self._steady_polar = (window, velocities[moving], bows[moving], drift)
        _, velocities, bows, drift = self._steady_polar
        if not len(bows):
            return None

        turn_rate = 60.0 / ForecastTackPlanner.tack_seconds
        if closest_layline:
            assert recovery_goal is not None
            return self._closest_layline_reference(
                observation, recovery_goal, target, velocities, bows, drift, grid, blocked, start, turn_rate,
            )

        # Account for initial and intermediate turns while the boat drifts.
        # Use an inner circle to leave tracking tolerance at the true gate.
        horizon = interval * ForecastTackPlanner.horizon_intervals
        if recovery_vmg is not None or recovery_goal is not None:
            # A viable return from outside the grid can take longer than the
            # usual 48-minute horizon. Select its next chart-checked segment.
            minimum_vmg, cap = (0.02, 43200.0) if recovery_goal is not None else (0.03, 21600.0)
            travel_time = np.linalg.norm(target) / max(recovery_vmg or 0.0, minimum_vmg)
            horizon = max(horizon, min(cap, travel_time + 360.0))
        turn = np.abs((bows - observation.boat.heading + 180.0) % 360.0 - 180.0) / turn_rate
        coast_target = target - turn[:, None] * drift
        speed_squared = np.sum(velocities**2, axis=1)
        projection = np.sum(coast_target * velocities, axis=1) / speed_squared
        ray_time = np.clip(projection, 0.0, horizon - turn)
        cross_track = np.linalg.norm(coast_target - projection[:, None] * velocities, axis=1)
        entry = turn + np.maximum(
            projection - np.sqrt(np.maximum(radius**2 - cross_track**2, 0.0) / speed_squared), 0.0,
        )
        direct_arrival = np.where((cross_track <= radius) & (projection > 0) & (entry <= horizon), entry, np.inf)

        # Reach the target with two positive-duration ground tracks if no
        # suitable straight approach exists for a given first heading.
        between = np.abs((bows[:, None] - bows + 180.0) % 360.0 - 180.0) / turn_rate
        finish = coast_target[:, None, :] - between[:, :, None] * drift
        east, north = velocities.T
        determinant = east[:, None] * north - north[:, None] * east
        usable = np.abs(determinant) > 1e-6
        before = np.divide(finish[:, :, 0] * north - finish[:, :, 1] * east, determinant,
                           out=np.full_like(determinant, np.inf), where=usable)
        after = np.divide(east[:, None] * finish[:, :, 1] - north[:, None] * finish[:, :, 0], determinant,
                          out=np.full_like(determinant, np.inf), where=usable)
        duration = turn[:, None] + before + between + after
        routes = np.where(usable & (before >= 30.0) & (after >= 0.0) & (duration <= horizon), duration, np.inf)
        arrival = np.minimum(direct_arrival, routes.min(axis=1))
        reachable = np.isfinite(arrival)
        # When a ray misses, judge it after at least one interval of sailing.
        # A ray with its closest point behind the boat would otherwise favor
        # the current trim even when it sails away fastest.
        sustained_time = np.maximum(ray_time, float(interval))
        sustained_distance = np.linalg.norm(coast_target - sustained_time[:, None] * velocities, axis=1)
        score = np.where(reachable, arrival, sustained_distance)
        order = np.lexsort((turn, score, ~reachable))
        current_tack = self._recovery_tack
        if recovery_goal is not None:
            time = float(observation.time.astype("datetime64[ms]").astype(np.int64)) / 1000.0
            sides = np.where((bows - wind.direction + 180.0) % 360.0 - 180.0 >= 0.0, 1, -1)
            if current_tack is not None and current_tack[0] == recovery_goal:
                _, side, settle_until = current_tack
                same = sides == side
                direct = np.isfinite(direct_arrival)
                opposite_finish = ~same & direct
                # Sail to the layline before turning across the wind.
                if time < settle_until or np.any(same & direct):
                    order = np.r_[order[same[order]], order[~same[order]]]
                elif np.any(opposite_finish):
                    finish_order = np.argsort(direct_arrival)
                    order = np.r_[finish_order[opposite_finish[finish_order]], order[~opposite_finish[order]]]
                elif np.any(same & reachable) or not np.any(reachable):
                    order = np.r_[order[same[order]], order[~same[order]]]

        leg_distance = min(grid.resolution, np.linalg.norm(target))
        for candidate in order:
            velocity = velocities[candidate]
            offset_east, offset_north = velocity * (leg_distance / np.sqrt(speed_squared[candidate]))
            end = start[0] - offset_north / grid.resolution, start[1] + offset_east / grid.resolution
            if not self._segment_is_clear(blocked, start, end):
                continue
            latitude, longitude = offset_latitude_longitude(
                grid.origin_latitude, grid.origin_longitude, offset_east, offset_north,
            )
            course = degrees(atan2(offset_east, offset_north)) % 360.0
            if recovery_goal is not None:
                side = int(sides[candidate])
                if current_tack is None or current_tack[:2] != (recovery_goal, side):
                    self._recovery_tack = (recovery_goal, side, time + max(float(interval), turn[candidate] + 30.0))
            return course, float(bows[candidate]), Waypoint(float(latitude), float(longitude))
        return None

    def _closest_layline_reference(
        self, observation, goal, target, velocities, bows, drift, grid, blocked, start, turn_rate,
    ) -> tuple[float, float, Waypoint] | None:
        wind_from = observation.wind.direction
        candidates = layline_candidates(
            target, velocities, bows, wind_from, observation.boat.heading, drift,
            ForecastTackPlanner.interval_seconds * ForecastTackPlanner.horizon_intervals,
            self.arrival_radius / 2., turn_rate,
        )
        sides = np.sign((bows - wind_from + 180.) % 360. - 180.)
        time = float(observation.time.astype("datetime64[ms]").astype(np.int64)) / 1000.
        committed = self._recovery_tack
        if committed is not None and committed[0] == goal:
            _, side, until = committed
            if time < until:
                candidates = [candidate for candidate in candidates if sides[candidate.first] == side]
            else:
                candidates.sort(key=lambda candidate: (candidate.second is not None,
                                                         sides[candidate.first] != side))
        for candidate in candidates:
            east, north = candidate.vertices_en.T
            vertices = np.column_stack((start[0] - north / grid.resolution,
                                        start[1] + east / grid.resolution))
            if ((vertices < 0.).any() or (vertices > grid.cells - 1.).any()
                    or not all(self._segment_is_clear(blocked, first, second)
                               for first, second in zip(vertices[:-1], vertices[1:]))):
                continue
            first = candidate.first
            velocity = velocities[first]
            leg_distance = min(grid.resolution, np.linalg.norm(target), max(1., candidate.first_distance_m))
            offset_east, offset_north = velocity * (leg_distance / np.linalg.norm(velocity))
            endpoint = Waypoint(*offset_latitude_longitude(
                grid.origin_latitude, grid.origin_longitude, offset_east, offset_north,
            ))
            side = int(sides[first])
            if committed is None or committed[:2] != (goal, side):
                turn = abs((bows[first] - observation.boat.heading + 180.) % 360. - 180.) / turn_rate
                # Revalidate the path every update, but allow time to execute a turn.
                self._recovery_tack = (goal, side, time + turn + 30.)
            self.waypoint_layline_telemetry = {
                "waypoint_layline_source": "direct gate approach" if candidate.second is None else "closest reachable tack",
                "waypoint_layline_distance_m": candidate.first_distance_m,
                "waypoint_layline_eta_s": candidate.arrival_s,
                "waypoint_layline_second_bow_deg": None if candidate.second is None else float(bows[candidate.second]),
            }
            if self.compare_waypoint_layline_eta:
                self.waypoint_layline_telemetry["waypoint_layline_first_sail_s"] = (
                    candidate.first_distance_m / float(np.linalg.norm(velocity))
                )
            return degrees(atan2(velocity[0], velocity[1])) % 360., float(bows[first]), endpoint
        return None

    def _open_water_reference(
        self, observation, goal, bearing, distance, grid, blocked, start
    ) -> tuple[float | None, Waypoint | None]:
        """Use the true waypoint or a checked close-hauled leg up to its layline."""
        if observation.wind is None:
            self._tacking_goal = None
            return bearing, goal
        wind_from = observation.wind.direction
        tack_angle = min(self.no_go_angle_degrees + 10.0, 87.0)
        relative_bearing = (bearing - wind_from + 180.0) % 360.0 - 180.0
        layline = tack_angle
        if self._tacking_goal == goal and relative_bearing * self._tack_side < 0.0:
            layline = self._ground_layline(observation, tack_angle)
        if abs(relative_bearing) >= layline:
            self._tacking_goal = None
            self._opposite_layline = None
            return bearing, goal
        if self._tacking_goal != goal:
            relative_heading = (observation.boat.heading - wind_from + 180.0) % 360.0 - 180.0
            self._tack_side = 1.0 if relative_heading >= 0.0 else -1.0
            self._tacking_goal = goal
            self._opposite_layline = None

        # Sustain a tack while the goal is between the laylines. If traffic
        # blocks that track, test the other tack before using the grid solver.
        leg_length = min(max(distance, grid.resolution), min(start) * grid.resolution)
        for side in (self._tack_side, -self._tack_side):
            heading = (wind_from + side * tack_angle) % 360.0
            heading = self._ground_tack_course(observation, heading, side)
            east, north = leg_length * sin(radians(heading)), leg_length * cos(radians(heading))
            end = start[0] - north / grid.resolution, start[1] + east / grid.resolution
            if not self._segment_is_clear(blocked, start, end):
                continue
            if side != self._tack_side:
                self._opposite_layline = None
            self._tack_side = side
            latitude, longitude = offset_latitude_longitude(
                grid.origin_latitude, grid.origin_longitude, east, north
            )
            return heading, Waypoint(float(latitude), float(longitude))
        return None, None

    def _ground_tack_course(
        self, observation: WorldObservation, heading: float, side: float,
    ) -> float:
        """Translate a close-hauled through-water bearing to a ground-course reference."""
        boat, wind, current = observation.boat, observation.wind, observation.current
        if wind is None or current is None or boat.surge_speed is None:
            return heading
        water_speed = hypot(boat.surge_speed, boat.sway_speed)
        if water_speed < 0.1:
            return heading
        east = water_speed * sin(radians(heading)) + current.u
        north = water_speed * cos(radians(heading)) + current.v
        course = degrees(atan2(east, north)) % 360.0
        relative_course = side * ((course - wind.direction + 180.0) % 360.0 - 180.0)
        if self.no_go_angle_degrees <= relative_course < 90.0:
            return course
        return heading

    def _ground_layline(self, observation: WorldObservation, tack_angle: float) -> float:
        """Project the opposite tack over ground using the settled water-relative motion."""
        boat, wind, current = observation.boat, observation.wind, observation.current
        if wind is None or current is None or boat.surge_speed is None:
            return tack_angle
        bow = (boat.heading - wind.direction + 180.0) % 360.0 - 180.0
        if bow * self._tack_side >= self.no_go_angle_degrees - 5.0 and boat.surge_speed > 0.1:
            drift = np.array([current.u, current.v])
            wind_over_water = np.array([wind.u, wind.v]) - drift
            wind_speed = np.linalg.norm(wind_over_water)
            if wind_speed > 1e-6:
                axis = wind_over_water / wind_speed
                water_velocity = Fossen3DOFSaildroneDynamics._body_to_earth(boat.heading) @ (
                    boat.surge_speed, boat.sway_speed
                )
                # Reflect the through-water track in the wind axis, then add
                # current once to estimate the other tack's ground track.
                opposite = 2.0 * np.dot(water_velocity, axis) * axis - water_velocity + drift
                course = degrees(atan2(opposite[0], opposite[1]))
                angle = -self._tack_side * ((course - wind.direction + 180.0) % 360.0 - 180.0)
                if angle > 0.0:
                    estimate = float(np.clip(angle, tack_angle, 87.0))
                    previous = self._opposite_layline
                    self._opposite_layline = (
                        estimate if previous is None else 0.9 * previous + 0.1 * estimate
                    )
        return tack_angle if self._opposite_layline is None else self._opposite_layline

    @staticmethod
    def _segment_is_clear(
        blocked: np.ndarray, start: tuple[int, int], target: tuple[float, float]
    ) -> bool:
        """Check the continuous segment against the full footprint of blocked cells."""
        occupied = np.nonzero(blocked)
        enter = np.zeros(occupied[0].size)
        leave = np.ones(occupied[0].size)
        for origin, end, indices in zip(start, target, occupied):
            delta = end - origin
            if abs(delta) < 1e-12:
                leave[np.abs(indices - origin) > 0.5] = -1.0
            else:
                centre = (indices - origin) / delta
                radius = 0.5 / abs(delta)
                enter = np.maximum(enter, centre - radius)
                leave = np.minimum(leave, centre + radius)
        return not np.any(enter <= leave)

    @staticmethod
    def _finite_field(field: np.ndarray) -> np.ndarray:
        values = np.asarray(field, dtype=np.float64)
        values = np.nan_to_num(values, nan=1.0, posinf=1.0, neginf=0.0)
        return np.clip(values, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _accepted_transformer_path(
        blocked: np.ndarray,
        start: tuple[int, int],
        target: tuple[int, int],
        prediction: tuple[float, float, float, float],
        *,
        wind_from_degrees: float | None = None,
        no_go_angle_degrees: float = 45.0,
    ) -> list[tuple[int, int]] | None:
        direction_row, direction_column, magnitude, goal_alignment = prediction
        if (
            not np.all(np.isfinite(prediction))
            or magnitude < 0.05
            or goal_alignment < -0.15
        ):
            return None

        # A learned direction is rounded to grid cells before it becomes a
        # steering reference, so check both the proposal and the resulting leg.
        angle_to_wind = HarmonicTacticalPlanner._angular_difference_degrees
        if wind_from_degrees is not None:
            heading = degrees(atan2(direction_column, -direction_row))
            if angle_to_wind(heading, wind_from_degrees) < no_go_angle_degrees:
                return None

        safe_cells = [start]
        row, column = map(float, start)
        for _ in range(8):
            row += direction_row
            column += direction_column
            cell = (int(round(row)), int(round(column)))
            if (
                not (0 <= cell[0] < blocked.shape[0] and 0 <= cell[1] < blocked.shape[1])
                or blocked[cell]
            ):
                break
            if safe_cells[-1] == cell:
                continue
            if not LegacyPlannerAdapter._segment_is_clear(blocked, safe_cells[-1], cell):
                break
            safe_cells.append(cell)
        if len(safe_cells) < 4:
            return None
        first = safe_cells[1]
        if wind_from_degrees is not None:
            heading = LegacyPlannerAdapter._path_heading((start, first))
            assert heading is not None
            if angle_to_wind(heading, wind_from_degrees) < no_go_angle_degrees:
                return None
        if (
            (first[0] - target[0]) ** 2
            + (first[1] - target[1]) ** 2
            >= (start[0] - target[0]) ** 2 + (start[1] - target[1]) ** 2
            and goal_alignment < 0.0
        ):
            return None
        return safe_cells

    @staticmethod
    def _boat_cell(cells: int) -> tuple[int, int]:
        centre = int(round((cells - 1) / 2.0))
        return centre, centre

    @staticmethod
    def _goal_cell(grid, goal: Waypoint) -> tuple[tuple[int, int], bool]:
        east, north = local_east_north_metres(
            goal.latitude,
            goal.longitude,
            grid.origin_latitude,
            grid.origin_longitude,
        )
        centre = (grid.cells - 1) / 2.0
        row = centre - float(north) / grid.resolution
        column = centre + float(east) / grid.resolution
        last = grid.cells - 1
        inside = 0.0 <= row <= last and 0.0 <= column <= last
        if inside:
            return (int(round(row)), int(round(column))), False
        dr, dc = row - centre, column - centre
        scale = min(
            centre / abs(dr) if abs(dr) > 1e-12 else np.inf,
            centre / abs(dc) if abs(dc) > 1e-12 else np.inf,
        )
        return (
            int(np.clip(round(centre + scale * dr), 0, last)),
            int(np.clip(round(centre + scale * dc), 0, last)),
        ), True

    @staticmethod
    def _nearest_free(
        blocked: np.ndarray,
        cell: tuple[int, int],
        *,
        boundary_only: bool,
    ) -> tuple[int, int]:
        if not blocked[cell]:
            return cell
        rows, cols = np.nonzero(~blocked)
        if not rows.size:
            raise RuntimeError("No free cells exist in the local planner map.")
        if boundary_only:
            last = blocked.shape[0] - 1
            keep = (rows == 0) | (rows == last) | (cols == 0) | (cols == last)
            if np.any(keep):
                rows, cols = rows[keep], cols[keep]
        idx = int(np.argmin((rows - cell[0]) ** 2 + (cols - cell[1]) ** 2))
        return int(rows[idx]), int(cols[idx])

    @staticmethod
    def _path_heading(path: tuple[tuple[int, int], ...]) -> float | None:
        if len(path) < 2:
            return None
        dr = float(path[1][0] - path[0][0])
        dc = float(path[1][1] - path[0][1])
        if abs(dr) + abs(dc) < 1e-12:
            return None
        return degrees(atan2(dc, -dr)) % 360.0

    @staticmethod
    def _fallback_neighbor(
        blocked: np.ndarray,
        potential: np.ndarray,
        start: tuple[int, int],
        target: tuple[int, int],
    ) -> tuple[int, int] | None:
        """Choose a free neighboring cell when Gamma has no descending step."""

        rows, columns = blocked.shape
        candidates = []
        for row_delta, column_delta in (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ):
            row = start[0] + row_delta
            column = start[1] + column_delta
            if not (0 <= row < rows and 0 <= column < columns):
                continue
            if blocked[row, column] or not np.isfinite(potential[row, column]):
                continue
            target_distance = (row - target[0]) ** 2 + (column - target[1]) ** 2
            candidates.append((float(potential[row, column]), target_distance, (row, column)))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][2]
