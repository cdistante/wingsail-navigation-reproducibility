"""Run imported legacy planners with Maritime World + MPC + Fossen dynamics.

Example
-------
python -m world.run_legacy_fossen --planner gamma --steps 120 --plot-every 10

This runner intentionally does *not* import ``simulator_2``.  The data/control
path is:

GEBCO/global map -> local map -> legacy planner -> SaildroneMPC
-> Fossen3DOFSaildroneDynamics -> MaritimeWorld -> repeat.
"""

from __future__ import annotations

import argparse
import json
import time
import xml.etree.ElementTree as ElementTree
from collections import Counter
from dataclasses import replace
from math import cos, degrees, radians, sin
from pathlib import Path


import numpy as np

from world import (
    AIS,
    Bathymetry,
    BoatAction,
    BoatState,
    Coastline,
    Currents,
    Fossen3DOFSaildroneDynamics,
    MaritimeWorld,
    Route,
    RouteWaypoint,
    SaildroneMPC,
    Waves,
    Weather,
    Wind,
    Waypoint,
)
from world.colreg import ColregPolicy
from world.ais_avoidance import AISPrediction
from world.ais_buffer_maneuver import AISBufferPlanner
from world.ais_course_transformer import AISCourseReplayForecaster
from world.ais_guidance import AISGuidance, assess_action, chart_action_is_clear, fallback_action
from world._geo import (
    haversine_metres,
    local_east_north_metres,
    normalise_time,
    offset_latitude_longitude,
)
from world.global_map import GlobalOccupancyMap
from world.forecast_tacks import ForecastTackPlanner
from world.legacy_planner_adapter import LegacyPlannerAdapter
from world.local_map import LegacyLocalMapBuilder
from world.marine_controller import (
    MarineController,
    global_features as marine_global_features,
    map_tensor as marine_map_tensor,
    planner_feature_field,
)
from world.planner_visualization import (
    plot_global_trajectory,
    plot_planner_snapshot,
    plot_start_goal_map,
)
from world.nmea import (
    NMEA0183TCPClientOutput,
    NMEA0183TCPServerOutput,
    NMEA0183UDPOutput,
    aivdm_position_report,
    aivdm_static_data_sentences,
    hdt_sentence,
    rmc_sentence,
)
from world.live_ais import LiveAIS, LiveAISUDP
from world.synthetic_ais import SyntheticAIS, SyntheticAISReporting
from world.tack_plan_replay import TackPlanReplay


class ConstantWindWeather:
    """Minimal weather provider for deterministic planner/Fossen smoke runs."""

    def __init__(self, speed: float, direction: float) -> None:
        self.speed = float(speed)
        self.direction = float(direction) % 360.0
        self.loaded = False

    def load(self):
        self.loaded = True
        return self

    def sample(self, latitude: float, longitude: float, when):
        direction_rad = np.radians(self.direction)
        # Meteorological direction is FROM; u/v point TO.
        return Wind(
            u=float(-self.speed * np.sin(direction_rad)),
            v=float(-self.speed * np.cos(direction_rad)),
            speed=self.speed,
            direction=self.direction,
        )

    def close(self) -> None:
        self.loaded = False


def build_global_waypoints(
    start: BoatState,
    goal: Waypoint,
    spacing_metres: float,
) -> tuple[Waypoint, ...]:
    """Build route-progress waypoints between the vessel and final goal."""

    if spacing_metres <= 0.0:
        raise ValueError("global waypoint spacing must be positive.")
    total_distance = haversine_metres(
        start.latitude,
        start.longitude,
        goal.latitude,
        goal.longitude,
    )
    segments = max(1, int(np.ceil(total_distance / spacing_metres)))
    return tuple(
        Waypoint(
            start.latitude + (goal.latitude - start.latitude) * fraction,
            start.longitude + (goal.longitude - start.longitude) * fraction,
        )
        for fraction in (index / segments for index in range(1, segments + 1))
    )


def goal_reached(state: BoatState, goal: Waypoint, threshold_metres: float) -> bool:
    """Return whether the vessel is inside the final goal radius."""

    if threshold_metres <= 0.0:
        raise ValueError("goal threshold must be positive.")
    return (
        haversine_metres(
            state.latitude,
            state.longitude,
            goal.latitude,
            goal.longitude,
        )
        <= threshold_metres
    )


def mission_goal_reached(
    state: BoatState,
    goal: Waypoint,
    threshold_metres: float,
    *,
    waypoint_index: int,
    waypoint_count: int,
) -> bool:
    """Reach an endpoint only after every preceding waypoint is complete."""

    if waypoint_count < 1:
        raise ValueError("A mission must contain at least one active waypoint.")
    return waypoint_index == waypoint_count - 1 and goal_reached(
        state, goal, threshold_metres
    )


def legacy_kinematic_step(
    state: BoatState,
    plan,
    global_map: GlobalOccupancyMap,
    resolution_metres: float,
    velocity_cells: np.ndarray,
    dt: float,
    speed_cells_per_step: float,
) -> tuple[BoatState, np.ndarray]:
    """Advance using the old planner's smoothed grid-cell movement model."""

    centre = (plan.blocked.shape[0] - 1) / 2.0
    path = plan.path_cells
    target = path[min(2, len(path) - 1)] if path else (centre, centre)
    direction = np.array(
        [float(target[1]) - centre, centre - float(target[0])],
        dtype=float,
    )
    if len(path) < 2 and plan.navigation_target is not None:
        goal = plan.navigation_target
        direction = np.array(local_east_north_metres(
            goal.latitude, goal.longitude, state.latitude, state.longitude
        )) / resolution_metres
    magnitude = float(np.linalg.norm(direction))
    if magnitude <= 1e-9:
        return state, np.zeros(2, dtype=float)

    target_velocity = direction / magnitude * float(speed_cells_per_step)
    velocity_cells = velocity_cells * 0.7 + target_velocity * 0.3
    east_metres, north_metres = velocity_cells * float(resolution_metres)
    latitude, longitude = offset_latitude_longitude(
        state.latitude,
        state.longitude,
        east_metres,
        north_metres,
    )
    next_latitude = float(latitude)
    next_longitude = float(longitude)
    row, column = global_map.latitude_longitude_to_grid(next_latitude, next_longitude)
    row_index, column_index = int(round(row)), int(round(column))
    assert global_map.occupancy is not None
    inside = (
        0 <= row_index < global_map.occupancy.shape[0]
        and 0 <= column_index < global_map.occupancy.shape[1]
    )
    if not inside or global_map.occupancy[row_index, column_index]:
        return state, np.zeros(2, dtype=float)

    heading = float(np.degrees(np.arctan2(east_metres, north_metres)) % 360.0)
    speed = float(np.hypot(east_metres, north_metres) / max(dt, 1e-9))
    return (
        BoatState(
            latitude=next_latitude,
            longitude=next_longitude,
            heading=heading,
            speed=speed,
            rudder=state.rudder,
            flap=state.flap,
            heel=state.heel,
            surge_speed=speed,
            sway_speed=0.0,
            yaw_rate=0.0,
        ),
        velocity_cells,
    )


def write_gpx(path: Path, trajectory: list[BoatState], name: str) -> None:
    """Write the simulated geographic trajectory as an OpenCPN GPX track."""

    namespace = "http://www.topografix.com/GPX/1/1"
    ElementTree.register_namespace("", namespace)
    root = ElementTree.Element(
        f"{{{namespace}}}gpx",
        {
            "version": "1.1",
            "creator": "maritime_world",
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
        },
    )
    track = ElementTree.SubElement(root, f"{{{namespace}}}trk")
    ElementTree.SubElement(track, f"{{{namespace}}}name").text = name
    segment = ElementTree.SubElement(track, f"{{{namespace}}}trkseg")
    for state in trajectory:
        ElementTree.SubElement(
            segment,
            f"{{{namespace}}}trkpt",
            {"lat": f"{state.latitude:.9f}", "lon": f"{state.longitude:.9f}"},
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    ElementTree.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    print(f"Saved GPX track: {path}")


def save_outputs(
    arguments: argparse.Namespace,
    trajectory: list[BoatState],
    ais_trajectories: dict[int, list[tuple[float, float]]] | None = None,
) -> None:
    """Save GPX and full global plot for completed or partial runs."""

    route = (
        Route.load_gpx(arguments.route_gpx, route_name=arguments.route_name)
        if arguments.route_gpx is not None
        else None
    )
    departure = route.departure if route is not None else Waypoint(
        arguments.start_latitude, arguments.start_longitude
    )
    destination = route.destination if route is not None else Waypoint(
        arguments.goal_latitude, arguments.goal_longitude
    )
    final_goal = departure if arguments.return_to_start else destination
    turnaround = destination if arguments.return_to_start else None
    if arguments.output_gpx is not None:
        write_gpx(arguments.output_gpx, trajectory, f"{arguments.planner} Fossen trajectory")
    if arguments.global_output is not None and trajectory:
        global_map = GlobalOccupancyMap(
            path=arguments.global_map,
            minimum_depth_metres=arguments.minimum_depth,
        ).load()
        figure = plot_global_trajectory(
            global_map,
            [(state.latitude, state.longitude) for state in trajectory],
            trajectory[0],
            final_goal,
            mode=arguments.planner,
            turnaround=turnaround,
            output=arguments.global_output,
        )
        import matplotlib.pyplot as plt
        plt.close(figure)
        print(f"Saved global plot: {arguments.global_output}")
    if arguments.ais_global_output is not None and trajectory:
        global_map = GlobalOccupancyMap(
            path=arguments.global_map,
            minimum_depth_metres=arguments.minimum_depth,
        ).load()
        figure = plot_global_trajectory(
            global_map,
            [(state.latitude, state.longitude) for state in trajectory],
            trajectory[0],
            final_goal,
            mode=arguments.planner,
            ais_trajectories=ais_trajectories,
            turnaround=turnaround,
            output=arguments.ais_global_output,
        )
        import matplotlib.pyplot as plt
        plt.close(figure)
        print(f"Saved boat/AIS global plot: {arguments.ais_global_output}")


def trajectory_course(previous: BoatState | None, current: BoatState) -> float:
    """Derive course over ground for an NMEA RMC sentence."""

    if previous is None:
        return float(current.heading % 360.0)
    east, north = local_east_north_metres(
        current.latitude,
        current.longitude,
        previous.latitude,
        previous.longitude,
    )
    if float(np.hypot(east, north)) < 1e-6:
        return float(current.heading % 360.0)
    return float(np.degrees(np.arctan2(east, north)) % 360.0)


def combined_controller_cost_is_safe(
    candidate_cost: float,
    expert_cost: float,
    maximum_fractional_increase: float,
) -> bool:
    """Accept a learned action only when its rollout remains near expert MPC."""

    if not np.all(np.isfinite([candidate_cost, expert_cost])):
        return False
    tolerance = maximum_fractional_increase * max(1.0, abs(expert_cost))
    return candidate_cost <= expert_cost + tolerance


def combined_controller_action_is_safe(
    candidate: BoatAction,
    expert: BoatAction,
    maximum_rudder_error: float,
    maximum_flap_error: float,
) -> bool:
    """Require learned controls to remain close to the expert action."""

    return (
        abs(candidate.rudder - expert.rudder) <= maximum_rudder_error
        and abs(candidate.flap - expert.flap) <= maximum_flap_error
    )


def time_aware_ais_enabled(arguments: argparse.Namespace) -> bool:
    explicit = getattr(arguments, "time_aware_ais", None)
    has_ais = any(getattr(arguments, name, None)
                  for name in ("ais_scenario", "ais_replay", "live_ais_host"))
    return bool(explicit if explicit is not None else arguments.dynamics == "fossen" and has_ais)


def drift_trim_action(dynamics, mpc, state, plan) -> BoatAction | None:
    """Track the attainable trim selected by the drift limiter within actuator slew."""
    if (
        plan.drift_mode not in {"holding", "forecast_release"} or plan.desired_bow_heading is None
        or plan.steady_rudder_degrees is None or plan.steady_flap_degrees is None
    ):
        return None
    parameters = dynamics.parameters
    error = (state.heading - plan.desired_bow_heading + 180.0) % 360.0 - 180.0
    rudder = plan.steady_rudder_degrees + 0.9 * error + 2.5 * degrees(state.yaw_rate)
    return BoatAction(
        float(np.clip(rudder, max(parameters.rudder_min, state.rudder - mpc.max_rudder_change_per_update_degrees),
                      min(parameters.rudder_max, state.rudder + mpc.max_rudder_change_per_update_degrees))),
        float(np.clip(plan.steady_flap_degrees, max(parameters.flap_min, state.flap - mpc.max_flap_change_per_update_degrees),
                      min(parameters.flap_max, state.flap + mpc.max_flap_change_per_update_degrees))),
    )


def synthetic_ais_reporting(arguments: argparse.Namespace) -> SyntheticAISReporting | None:
    delay = getattr(arguments, "ais_report_delay", 0.0)
    windows = tuple(tuple(window) for window in (getattr(arguments, "ais_drop_reports", None) or ()))
    if delay == 0.0 and not windows:
        return None
    if arguments.ais_scenario is None:
        raise ValueError("AIS report delay and drop windows require --ais-scenario.")
    return SyntheticAISReporting(arguments.dt, delay, windows)


def navigation_method_lines(arguments: argparse.Namespace) -> tuple[str, ...]:
    """Describe how a recording's route and actuator commands were produced."""

    planner = {
        "standard": "Standard harmonic planner",
        "gamma": "Gamma harmonic planner",
        "diffusion": "Wind-aware diffusion planner",
        "transformer_diffusion": "Transformer Diffusion planner",
    }[arguments.planner]
    if arguments.dynamics == "legacy_kinematic":
        lines = ["Legacy kinematic model", planner]
    else:
        controller = {
            "grid-mpc": "Grid MPC",
            "acados-direct": "acados NMPC · Direct path following",
            "acados-los": "acados NMPC · Line-of-sight guidance (LOS)",
        }[arguments.controller]
        if arguments.combined_controller_checkpoint is not None:
            controller = "Marine Transformer with grid MPC fallback"
        lines = [controller, f"Fossen 3-DOF · {planner}"]
        if arguments.controller == "acados-direct":
            lines.append("Cross-track and progress")
    lines.append("Archived wind, waves and current" if arguments.environment in {"marine", "full"} else "Constant wind")
    lines.append(f"{arguments.arrival_radius:g} m waypoint arrival radius")
    if getattr(arguments, "native_hybrid_enabled", False):
        lines.append("Transformer weather + AIS · forecast physics veto")
    if getattr(arguments, "compare_waypoint_layline_eta", False):
        lines.append("Closest waypoint layline only on faster shared forecast ETA")
    elif getattr(arguments, "prioritize_waypoint_layline", False):
        lines.append("Closest reachable waypoint layline priority · after AIS screen")
    elif getattr(arguments, "closest_waypoint_layline", False):
        lines.append("Closest reachable waypoint layline · wind and current")
    if getattr(arguments, "limit_upwind_drift", False):
        lines.append("Upwind drift limiting · chart-checked trim hold")
        if getattr(arguments, "forecast_hold_tacks", False):
            lines.append("Archived look-ahead · tacks during hold")
        if getattr(arguments, "forecast_eta_hold_release", False):
            lines.append("Forecast ETA hold release · sail now vs one more holding window")
        if getattr(arguments, "regatta_time_to_mark", False):
            lines.append("Regatta guidance · forecast time to mark + refined tacks")
        if getattr(arguments, "temporal_recovery_model", None) is not None:
            lines.append("Temporal Transformer · hold and release guidance")
            if getattr(arguments, "confirm_temporal_release", False):
                lines.append("Two-window Fossen release confirmation")
        if getattr(arguments, "commit_recovery_tacks", False) and not arguments.forecast_tacks:
            lines.append("Post-hold ground laylines · observed forcing")
    if time_aware_ais_enabled(arguments):
        lines.append("Time-aware AIS · Fossen buffer maneuvers")
        if getattr(arguments, "ais_candidate_handoff", False):
            lines.append("AIS handoff · candidate tack screened when active avoidance clears")
        acceleration_bound = getattr(arguments, "ais_acceleration_bound", 0.0)
        if acceleration_bound:
            lines.append(f"AIS acceleration bound {acceleration_bound:g} m/s²")
        clearance_reserve = getattr(arguments, "ais_clearance_reserve", 0.0)
        if clearance_reserve:
            lines.append(f"AIS extra clearance reserve {clearance_reserve:g} m")
        preview_age = getattr(arguments, "ais_preview_age_budget", 0.0)
        if preview_age:
            lines.append(f"AIS planning report-age budget {preview_age:g} s")
        lines.append("COLREG course preference" if arguments.colreg else "COLREG preference off")
    elif any(getattr(arguments, name, None) for name in ("ais_scenario", "ais_replay", "live_ais_host")):
        lines.append("Swept AIS corridor" + (" · COLREG mask" if arguments.colreg else ""))
    if getattr(arguments, "ais_scenario", None) is not None:
        scenario = arguments.ais_scenario.replace("_", "-")
        lines.append(f"Synthetic AIS · {scenario}")
        launch_lead = getattr(arguments, "ais_launch_lead", 0.0)
        if launch_lead:
            lines.append(f"Saildrone launch {launch_lead:g} s before AIS encounter reference")
        if getattr(arguments, "ais_maneuvering_sharp", False):
            lines.append("Ships make sharper course and speed changes")
        elif getattr(arguments, "ais_maneuvering", False):
            lines.append("Ships change course and speed")
        reporting = synthetic_ais_reporting(arguments)
        if reporting is not None:
            lines.append(f"AIS emission every {reporting.interval_seconds:g} s · Delay {reporting.delay_seconds:g} s")
            for start, stop in reporting.drop_windows:
                lines.append(f"AIS emissions dropped from {start:g} to {stop:g} s")
    elif getattr(arguments, "ais_replay", None) is not None:
        lines.append("Recorded AIS replay")
    if arguments.tack_plan_replay is not None:
        path = arguments.tack_plan_replay
        source = path.parent.name if path.name == "navigation.jsonl.gz" else path.name
        lines.extend(("Recorded tack plan", source or path.name))
    elif arguments.forecast_tacks:
        lines.append("Forecast tack planning")
        lines.append("Refined close-hauled polar angles" if getattr(arguments, "forecast_tack_refinement", False)
                     else "Original coarse polar angles")
        if getattr(arguments, "forecast_tracking_feedback", False):
            lines.append("Observed NMPC tracking feedback · schedule prediction")
        lines.append(f"Forecast replanning every {getattr(arguments, 'forecast_tack_step', 120):g} s")
        if getattr(arguments, "forecast_refinement_phase", "all") == "return":
            lines.append("Refinement on return · outbound uses original 120 s scheduler")
        if getattr(arguments, "forecast_gate_aim_fraction", 0.0):
            lines.append(f"Next-waypoint gate aim · {arguments.forecast_gate_aim_fraction:g} of capture radius")
            lines.append(f"Gate aim for continuation within {arguments.forecast_gate_aim_upwind_angle:g}° of wind FROM")
        if getattr(arguments, "forecast_maneuver_commit_seconds", 0.0):
            lines.append(f"Forecast maneuver commitment · {arguments.forecast_maneuver_commit_seconds:g} s")
        gain_seconds = getattr(arguments, "forecast_tack_gain_seconds", 0.0)
        gain_metres = getattr(arguments, "forecast_tack_gain_metres", 0.0)
        if gain_seconds or gain_metres:
            lines.append(f"Tack gain threshold · {gain_seconds:g} s arrival · {gain_metres:g} m closest")
        if getattr(arguments, "learned_arrival_model", None) is not None:
            lines.append("Learned arrival time · Fossen tack transitions")
        elif getattr(arguments, "learned_tack_model", None) is not None:
            lines.append("Learned first tack · Fossen rollout progress")
    return tuple(lines)


def warn_tack_commitment_scale(forecast_tacks) -> None:
    """Report tack-stability settings whose scale makes them ineffective.

    Maneuver commitment is consulted at a replan boundary. A window no
    longer than the replan interval has usually expired by then, so it
    cannot prevent an opposite-tack reversal except when a mid-window
    replan starts it off the boundary. The gain thresholds are selected by
    outcome class, so leaving one at zero leaves that class unguarded.
    """
    if 0.0 < forecast_tacks.maneuver_commit_seconds <= forecast_tacks.interval_seconds:
        print(
            f"Warning: --forecast-maneuver-commit-seconds "
            f"{forecast_tacks.maneuver_commit_seconds:g} does not exceed the "
            f"{forecast_tacks.interval_seconds} s replan interval, so it rarely "
            "constrains a replan. Use a longer window, or a tack gain threshold.",
            flush=True,
        )
    if bool(forecast_tacks.tack_gain_seconds) != bool(forecast_tacks.tack_gain_metres):
        unguarded = ("schedules that only approach the mark, which is the usual "
                     "upwind case" if not forecast_tacks.tack_gain_metres
                     else "schedules that reach the mark inside the horizon")
        print(
            "Warning: only one tack gain threshold is set, so the requirement is "
            f"zero for {unguarded}. Set both --forecast-tack-gain-seconds and "
            "--forecast-tack-gain-metres to guard every outcome.",
            flush=True,
        )


def suspend_nominal_guidance(planner, navigation: dict, reason: str) -> None:
    """Invalidate a proposal when another controller supplies the applied action."""
    planner.suspend_open_water_recovery(reason)
    navigation.clear()
    navigation.update(planner.temporal_recovery_telemetry)
    navigation.update(planner.forecast_hold_telemetry)
    navigation["nominal_guidance_suspended"] = reason


def run(arguments: argparse.Namespace) -> tuple[list[BoatState], dict[int, list[tuple[float, float]]]]:
    """Execute the integrated receding-horizon planner/controller loop."""

    if not np.isfinite(arguments.arrival_radius) or arguments.arrival_radius <= 0.0:
        raise ValueError("Arrival radius must be finite and positive.")
    gate_aim_fraction = getattr(arguments, "forecast_gate_aim_fraction", 0.0)
    if not np.isfinite(gate_aim_fraction) or not 0.0 <= gate_aim_fraction <= 0.5:
        raise ValueError("Forecast gate aim fraction must lie between zero and 0.5.")
    gate_aim_upwind_angle = getattr(arguments, "forecast_gate_aim_upwind_angle", 180.0)
    maneuver_commit_seconds = getattr(arguments, "forecast_maneuver_commit_seconds", 0.0)
    if not np.isfinite(gate_aim_upwind_angle) or not 0.0 <= gate_aim_upwind_angle <= 180.0:
        raise ValueError("Forecast continuation wind angle must lie between zero and 180 degrees.")
    if not np.isfinite(maneuver_commit_seconds) or maneuver_commit_seconds < 0.0:
        raise ValueError("Forecast maneuver commitment must be finite and non-negative.")
    tack_gain_seconds = getattr(arguments, "forecast_tack_gain_seconds", 0.0)
    tack_gain_metres = getattr(arguments, "forecast_tack_gain_metres", 0.0)
    if not np.isfinite(tack_gain_seconds) or tack_gain_seconds < 0.0:
        raise ValueError("Forecast tack gain seconds must be finite and non-negative.")
    if not np.isfinite(tack_gain_metres) or tack_gain_metres < 0.0:
        raise ValueError("Forecast tack gain metres must be finite and non-negative.")
    if (tack_gain_seconds or tack_gain_metres) and not arguments.forecast_tacks:
        raise ValueError("Forecast tack gain thresholds require --forecast-tacks.")
    if arguments.forecast_tacks and (
        arguments.environment not in {"marine", "full"}
        or arguments.dynamics != "fossen"
        or arguments.planner == "transformer_diffusion"
        or arguments.combined_controller_checkpoint is not None
    ):
        raise ValueError("--forecast-tacks requires archived marine data, Fossen dynamics, and MPC alone.")
    learned_tack_model = getattr(arguments, "learned_tack_model", None)
    learned_arrival_model = getattr(arguments, "learned_arrival_model", None)
    if learned_tack_model is not None and not arguments.forecast_tacks:
        raise ValueError("--learned-tack-model requires --forecast-tacks.")
    if learned_arrival_model is not None and not arguments.forecast_tacks:
        raise ValueError("--learned-arrival-model requires --forecast-tacks.")
    use_acados = arguments.controller != "grid-mpc"
    time_aware_ais = time_aware_ais_enabled(arguments)
    limit_upwind_drift = getattr(arguments, "limit_upwind_drift", False)
    forecast_hold_tacks = getattr(arguments, "forecast_hold_tacks", False)
    forecast_eta_hold_release = getattr(arguments, "forecast_eta_hold_release", False)
    regatta_time_to_mark = getattr(arguments, "regatta_time_to_mark", False)
    commit_recovery_tacks = getattr(arguments, "commit_recovery_tacks", False)
    prioritize_waypoint_layline = getattr(arguments, "prioritize_waypoint_layline", False)
    compare_waypoint_layline_eta = getattr(arguments, "compare_waypoint_layline_eta", False)
    closest_waypoint_layline = (prioritize_waypoint_layline or compare_waypoint_layline_eta
                                or getattr(arguments, "closest_waypoint_layline", False))
    temporal_recovery_model = getattr(arguments, "temporal_recovery_model", None)
    confirm_temporal_release = getattr(arguments, "confirm_temporal_release", False)
    if commit_recovery_tacks and not limit_upwind_drift:
        raise ValueError("--commit-recovery-tacks requires --limit-upwind-drift.")
    if compare_waypoint_layline_eta and (not arguments.forecast_tacks or prioritize_waypoint_layline):
        raise ValueError("--compare-waypoint-layline-eta requires --forecast-tacks and its own selection mode.")
    if closest_waypoint_layline and (
        arguments.environment not in {"marine", "full"} or arguments.dynamics != "fossen"
        or arguments.tack_plan_replay is not None
    ):
        raise ValueError("Waypoint laylines require live Fossen guidance with marine forcing.")
    if forecast_hold_tacks and not limit_upwind_drift:
        raise ValueError("--forecast-hold-tacks requires --limit-upwind-drift.")
    if forecast_eta_hold_release and not limit_upwind_drift:
        raise ValueError("--forecast-eta-hold-release requires --limit-upwind-drift.")
    if regatta_time_to_mark and not (forecast_hold_tacks and arguments.forecast_tacks):
        raise ValueError("--regatta-time-to-mark requires forecast hold and forecast tack guidance.")
    if temporal_recovery_model is not None and not limit_upwind_drift:
        raise ValueError("--temporal-recovery-model requires --limit-upwind-drift.")
    if confirm_temporal_release and temporal_recovery_model is None:
        raise ValueError("--confirm-temporal-release requires --temporal-recovery-model.")
    if limit_upwind_drift and (
        arguments.environment not in {"marine", "full"} or arguments.dynamics != "fossen"
        or arguments.tack_plan_replay is not None
    ):
        raise ValueError("Upwind drift limiting requires archived Fossen guidance without recorded tacks.")
    ais_clearance_reserve = getattr(arguments, "ais_clearance_reserve", 0.0)
    ais_preview_age_budget = getattr(arguments, "ais_preview_age_budget", 0.0)
    ais_predictive_fallback = getattr(arguments, "ais_predictive_fallback", False)
    ais_candidate_handoff = getattr(arguments, "ais_candidate_handoff", False)
    if time_aware_ais and arguments.dynamics != "fossen":
        raise ValueError("Time-aware AIS avoidance requires Fossen dynamics.")
    if ais_clearance_reserve and not time_aware_ais:
        raise ValueError("--ais-clearance-reserve requires time-aware AIS avoidance.")
    if ais_preview_age_budget and not time_aware_ais:
        raise ValueError("--ais-preview-age-budget requires time-aware AIS avoidance.")
    if ais_predictive_fallback and (not time_aware_ais or not ais_preview_age_budget > 0.0):
        raise ValueError("--ais-predictive-fallback requires time-aware AIS and a positive preview age budget.")
    if ais_candidate_handoff and not (ais_predictive_fallback and limit_upwind_drift):
        raise ValueError("--ais-candidate-handoff requires --ais-predictive-fallback and --limit-upwind-drift.")
    if use_acados and (
        arguments.dynamics != "fossen"
        or arguments.combined_controller_checkpoint is not None
        or arguments.planner == "transformer_diffusion"
    ):
        raise ValueError("acados requires Fossen dynamics and a non-learned tactical planner.")
    if arguments.tack_plan_replay is not None and (not use_acados or arguments.route_gpx is None):
        raise ValueError("--tack-plan-replay requires an acados controller and a GPX route.")

    route = (
        Route.load_gpx(arguments.route_gpx, route_name=arguments.route_name)
        if arguments.route_gpx is not None
        else None
    )
    start_latitude = route.departure.latitude if route is not None else arguments.start_latitude
    start_longitude = route.departure.longitude if route is not None else arguments.start_longitude

    live_view = None
    if arguments.live_plot:
        import os

        import matplotlib

        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            if matplotlib.get_backend().lower().endswith("agg"):
                matplotlib.use("qtagg", force=True)
        import matplotlib.pyplot as plt

        from world.animate_episode import SaildroneAnimation

        plt.ion()

    if arguments.live_plot or arguments.record_replay is not None:
        from world.animate_episode import EpisodeFrame

    live_ais = None
    ais_course_forecaster = None
    reporting = synthetic_ais_reporting(arguments)
    launch_lead = getattr(arguments, "ais_launch_lead", 0.0)
    if launch_lead and arguments.ais_scenario is None:
        raise ValueError("--ais-launch-lead requires --ais-scenario.")
    if arguments.ais_scenario is not None:
        sharp_manoeuvres = getattr(arguments, "ais_maneuvering_sharp", False)
        live_ais = SyntheticAIS(
            arguments.ais_scenario,
            start_latitude,
            start_longitude,
            reporting=reporting,
            manoeuvring=getattr(arguments, "ais_maneuvering", False) or sharp_manoeuvres,
            sharp_manoeuvres=sharp_manoeuvres,
            launch_lead_seconds=launch_lead,
        ).load()
        print(f"Loaded synthetic AIS traffic scenario: {arguments.ais_scenario}.", flush=True)
    elif arguments.ais_replay is not None:
        live_ais = AIS(arguments.ais_replay, mmsi=arguments.ais_mmsi).load()
        details = live_ais.info()
        print(
            f"Loaded AIS replay: {details['vessels']} vessel(s), {details['reports']} reports, "
            f"time mode {details['time_mode']}.",
            flush=True,
        )
    elif arguments.live_ais_host is not None:
        live_ais_class = LiveAIS if arguments.live_ais_transport == "tcp" else LiveAISUDP
        live_ais = live_ais_class(arguments.live_ais_host, arguments.live_ais_port)
        deadline = time.monotonic() + arguments.live_ais_connect_timeout
        while True:
            try:
                live_ais.load()
                print(
                    f"Connected to live AIS {arguments.live_ais_transport.upper()} "
                    f"{arguments.live_ais_host}:"
                    f"{arguments.live_ais_port}.",
                    flush=True,
                )
                break
            except OSError as error:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Could not connect to live AIS {arguments.live_ais_transport.upper()} "
                        f"{arguments.live_ais_host}:{arguments.live_ais_port}. "
                        "Start the OpenCPN multiplexer TCP output first, or omit "
                        "--live-ais-host for a run without live AIS."
                    ) from error
                print(
                    f"Waiting for live AIS {arguments.live_ais_transport.upper()} "
                    f"{arguments.live_ais_host}:"
                    f"{arguments.live_ais_port}...",
                    flush=True,
                )
                time.sleep(arguments.live_ais_retry_seconds)
    if arguments.ais_course_checkpoint is not None:
        ais_course_forecaster = AISCourseReplayForecaster(
            arguments.ais_course_checkpoint,
            arguments.ais_replay,
            device=arguments.ais_course_device,
            max_interpolation_gap_seconds=arguments.ais_course_max_gap_seconds,
        )
        print(
            f"Loaded AIS course Transformer: {arguments.ais_course_checkpoint}.",
            flush=True,
        )
    if arguments.environment in {"marine", "full"}:
        world = MaritimeWorld(
            weather=Weather(),
            waves=Waves(),
            currents=Currents(),
            bathymetry=Bathymetry() if arguments.environment == "full" else None,
            coastline=Coastline() if arguments.environment == "full" else None,
            ais=live_ais,
        ).load()
    else:
        world = MaritimeWorld(
            weather=ConstantWindWeather(arguments.wind_speed, arguments.wind_direction),
            ais=live_ais,
        ).load()
    global_map = GlobalOccupancyMap(
        path=arguments.global_map,
        minimum_depth_metres=arguments.minimum_depth,
    ).load()
    local_builder = LegacyLocalMapBuilder(
        global_map,
        span_metres=arguments.local_span,
        resolution_metres=arguments.local_resolution,
        obstacle_buffer_metres=arguments.obstacle_buffer,
    )
    dynamics = Fossen3DOFSaildroneDynamics()
    forecast_tacks = None
    if arguments.forecast_tacks:
        learned_selector = None
        arrival_selector = None
        if learned_tack_model is not None:
            from world.learned_tacks import LearnedTackSelector

            learned_selector = LearnedTackSelector.load(learned_tack_model)
            print(f"Loaded learned tack selector: {learned_tack_model}", flush=True)
        if learned_arrival_model is not None:
            from world.learned_tack_arrival import LearnedTackArrivalSelector

            arrival_selector = LearnedTackArrivalSelector.load(learned_arrival_model)
            print(f"Loaded learned arrival-time selector: {learned_arrival_model}", flush=True)
        refine_outbound = arguments.forecast_refinement_phase == "all" or regatta_time_to_mark
        forecast_tacks = ForecastTackPlanner(
            world, dynamics, arguments.no_go_angle, arguments.arrival_radius,
            learned_selector=learned_selector, arrival_selector=arrival_selector,
            refine_modes=arguments.forecast_tack_refinement if refine_outbound else False,
            tracking_feedback=arguments.forecast_tracking_feedback if refine_outbound else False,
            interval_seconds=arguments.forecast_tack_step if refine_outbound else 120,
            gate_aim_fraction=gate_aim_fraction if refine_outbound else 0.0,
            gate_aim_upwind_angle=gate_aim_upwind_angle,
            maneuver_commit_seconds=maneuver_commit_seconds if refine_outbound else 0.0,
            # The tack-gain threshold guards both legs, because a replan can
            # alternate between symmetric close-hauled options at any stage.
            tack_gain_seconds=tack_gain_seconds, tack_gain_metres=tack_gain_metres,
        )
        print("Forecast tacks: 48-minute archived wind, wave and current look-ahead; "
              f"{'refined' if arguments.forecast_tack_refinement else 'original'} sailing modes.", flush=True)
        warn_tack_commitment_scale(forecast_tacks)
    temporal_recovery = None
    if temporal_recovery_model is not None:
        from world.temporal_recovery import TemporalRecoveryGuidance

        temporal_recovery = TemporalRecoveryGuidance(temporal_recovery_model, confirm_release=confirm_temporal_release)
        print(f"Loaded temporal hold/release guidance: {temporal_recovery_model}", flush=True)
    forecast_hold = None
    forecast_release = None
    if forecast_eta_hold_release:
        from world.forecast_eta_release import ForecastETARelease

        forecast_release = ForecastETARelease(world, dynamics, arguments.no_go_angle, arguments.arrival_radius)
    if forecast_hold_tacks:
        from world.forecast_hold import ForecastHoldPlanner

        if regatta_time_to_mark:
            from world.regatta_guidance import RegattaHoldPlanner

            forecast_hold = RegattaHoldPlanner(world, dynamics, arguments.no_go_angle, arguments.arrival_radius)
        else:
            forecast_hold = ForecastHoldPlanner(world, dynamics, arguments.no_go_angle)
        print("Forecast hold tacks: archived wind, waves and current look-ahead.", flush=True)
    planner = LegacyPlannerAdapter(
        mode=arguments.planner,
        no_go_angle_degrees=arguments.no_go_angle,
        transformer_checkpoint=arguments.transformer_diffusion_checkpoint,
        transformer_device=arguments.transformer_diffusion_device,
        forecast_tacks=forecast_tacks,
        fossen_dynamics=dynamics if (
            arguments.dynamics == "fossen" and arguments.environment in {"marine", "full"}
            and (limit_upwind_drift or closest_waypoint_layline or not time_aware_ais)
        ) else None,
        arrival_radius=arguments.arrival_radius,
        limit_drift=limit_upwind_drift,
        commit_recovery_tacks=commit_recovery_tacks,
        temporal_recovery=temporal_recovery,
        forecast_hold=forecast_hold,
        closest_waypoint_layline=closest_waypoint_layline,
        prioritize_waypoint_layline=prioritize_waypoint_layline,
        compare_waypoint_layline_eta=compare_waypoint_layline_eta,
        forecast_release=forecast_release,
    )
    if arguments.planner == "transformer_diffusion":
        print(
            "Loaded own-ship Diffusion Transformer: "
            f"{arguments.transformer_diffusion_checkpoint}.",
            flush=True,
        )
    mpc = (
        SaildroneMPC(no_go_angle_degrees=arguments.no_go_angle)
        if forecast_tacks is not None else SaildroneMPC()
    )
    acados_controller = None
    nmpc_outcomes: Counter[str] = Counter()
    ais_outcomes: Counter[str] = Counter()
    ais_guidance = AISGuidance(arguments.no_go_angle) if time_aware_ais else None
    buffer_planner = (
        AISBufferPlanner(dynamics, mpc, arguments.dt, arguments.no_go_angle,
                         predictive_fallback=ais_predictive_fallback,
                         reports_refreshed_each_step=(
                             isinstance(live_ais, SyntheticAIS) and live_ais.reports_refreshed_each_step
                         ))
        if time_aware_ais else None
    )
    termination_reason = None
    tack_sources: Counter[str] = Counter()
    if use_acados:
        from world.acados_nmpc import AcadosNMPC, AISControlInfeasible

        acados_controller = AcadosNMPC(
            dynamics=dynamics,
            mode=arguments.controller.removeprefix("acados-"),
            dt_seconds=arguments.dt,
            no_go_angle_degrees=arguments.no_go_angle,
            build_directory=arguments.acados_build_dir,
            **({"ais_capacity": 6} if time_aware_ais and live_ais is not None else {}),
        )
        print(f"Controller: {arguments.controller}", flush=True)
    combined_controller = (
        MarineController(
            arguments.combined_controller_checkpoint,
            device=arguments.combined_controller_device,
            confidence_threshold=arguments.combined_controller_min_confidence,
        )
        if arguments.combined_controller_checkpoint is not None
        else None
    )
    controller_accepted = 0
    controller_fallback = 0
    controller_rollout_rejected = 0
    controller_fallback_reasons: Counter[str] = Counter()
    if combined_controller is not None:
        print(f"Loaded combined marine controller: {arguments.combined_controller_checkpoint}.", flush=True)
    colreg_policy = ColregPolicy() if arguments.colreg else None
    legacy_velocity_cells = np.zeros(2, dtype=float)

    state = BoatState(
        latitude=start_latitude,
        longitude=start_longitude,
        heading=arguments.heading,
        speed=0.0,
        rudder=0.0,
        flap=arguments.flap,
    )
    start = Waypoint(state.latitude, state.longitude)
    goal = (
        Waypoint(route.destination.latitude, route.destination.longitude)
        if route is not None
        else Waypoint(arguments.goal_latitude, arguments.goal_longitude)
    )
    returning_to_start = False
    if route is not None:
        global_waypoints = tuple(
            Waypoint(waypoint.latitude, waypoint.longitude)
            for waypoint in route.waypoints[1:]
        )
        global_waypoint_names = tuple(waypoint.name for waypoint in route.waypoints[1:])
        print(
            f"Loaded route {route.name!r}: {len(route.waypoints)} waypoint(s), "
            f"{route.total_distance_metres / 1852.0:.2f} NM outbound.",
            flush=True,
        )
    else:
        global_waypoints = build_global_waypoints(
            state,
            goal,
            arguments.global_waypoint_spacing,
        )
        global_waypoint_names = tuple(
            f"{index + 1:03d}" for index in range(len(global_waypoints))
        )
    global_waypoint_index = 0
    display_route = route.waypoints if route is not None else (
        RouteWaypoint(start.latitude, start.longitude, name="Start"),
        *(
            RouteWaypoint(waypoint.latitude, waypoint.longitude, name=name)
            for waypoint, name in zip(global_waypoints, global_waypoint_names)
        ),
    )
    if arguments.global_map is not None:
        assert global_map.latitudes is not None and global_map.longitudes is not None
        map_latitude_min = min(float(global_map.latitudes[0]), float(global_map.latitudes[-1]))
        map_latitude_max = max(float(global_map.latitudes[0]), float(global_map.latitudes[-1]))
        map_longitude_min = min(float(global_map.longitudes[0]), float(global_map.longitudes[-1]))
        map_longitude_max = max(float(global_map.longitudes[0]), float(global_map.longitudes[-1]))
        route_points = (
            tuple((waypoint.latitude, waypoint.longitude) for waypoint in route.waypoints)
            if route is not None
            else ((state.latitude, state.longitude), (goal.latitude, goal.longitude))
        )
        outside = [
            (latitude, longitude)
            for latitude, longitude in route_points
            if not (
                map_latitude_min <= latitude <= map_latitude_max
                and map_longitude_min <= longitude <= map_longitude_max
            )
        ]
        if outside:
            raise ValueError(
                "Start/goal coordinates are outside the selected global map. "
                f"Map bounds: latitude {map_latitude_min:.5f}..{map_latitude_max:.5f}, "
                f"longitude {map_longitude_min:.5f}..{map_longitude_max:.5f}; "
                f"received: {route_points}."
            )
    current_time = normalise_time(arguments.time)
    tack_plan = None
    if arguments.tack_plan_replay is not None:
        assert route is not None
        tack_plan = TackPlanReplay(
            arguments.tack_plan_replay, route, current_time, arguments.dt, arguments.arrival_radius,
        )
        print(f"Tack plan replay: {arguments.tack_plan_replay}", flush=True)
    leg_reference = None
    leg_start = start
    leg_start_step = 0
    trajectory = [state]
    ais_trajectories: dict[int, list[tuple[float, float]]] = {}
    output_dir = Path(arguments.output_dir)
    active_local_map = None
    active_plan = None
    active_local_goal = None
    active_local_goal_latitude = None
    active_local_goal_longitude = None
    nmea_output = None
    previous_state = None
    announced_ais_mmsi: set[int] = set()
    latest_ais_forecasts = {}
    if arguments.nmea_host is not None:
        if arguments.nmea_transport == "udp":
            nmea_output = NMEA0183UDPOutput(arguments.nmea_host, arguments.nmea_port)
        elif arguments.nmea_transport == "tcp-server":
            nmea_output = NMEA0183TCPServerOutput(arguments.nmea_host, arguments.nmea_port)
            print(
                f"Waiting for OpenCPN TCP connection on "
                f"{arguments.nmea_host}:{arguments.nmea_port}.",
                flush=True,
            )
            nmea_output.wait_for_client()
        else:
            nmea_output = NMEA0183TCPClientOutput(arguments.nmea_host, arguments.nmea_port)
        print(
            f"Publishing live NMEA to {arguments.nmea_transport} "
            f"{arguments.nmea_host}:{arguments.nmea_port}."
        )

    waypoint_arrivals: list[dict[str, str | int | float]] = []
    mission_completed = False
    recorder = None
    initial_frame = None
    method_lines = navigation_method_lines(arguments)
    display_weather = world.weather if isinstance(world.weather, Weather) else None
    try:
        if arguments.record_replay is not None:
            from world.navigation_replay import NavigationRecorder

            recorder = NavigationRecorder(
                arguments.record_replay,
                route=display_route,
                route_name=route.name if route is not None else None,
                interval_ms=arguments.live_plot_interval_ms,
                show_environment=True,
                view_radius_metres=arguments.live_plot_radius,
                boat_scale=arguments.live_plot_boat_scale,
                local_frame=arguments.live_plot_frame == "boat-centered",
                fixed_local_view=arguments.live_plot_frame == "fixed",
                method_lines=method_lines,
            )
        step = 0
        while True:
            phase = "return" if returning_to_start else "outbound"
            distance = haversine_metres(
                state.latitude,
                state.longitude,
                goal.latitude,
                goal.longitude,
            )
            while global_waypoint_index < len(global_waypoints):
                active_waypoint = global_waypoints[global_waypoint_index]
                final_waypoint = global_waypoint_index == len(global_waypoints) - 1
                waypoint_distance = distance if final_waypoint else haversine_metres(
                    state.latitude,
                    state.longitude,
                    active_waypoint.latitude,
                    active_waypoint.longitude,
                )
                arrival_radius = (
                    arguments.arrival_radius
                    if route is not None or final_waypoint
                    else arguments.global_waypoint_radius
                )
                if not (waypoint_distance <= arrival_radius):
                    break
                if route is not None:
                    waypoint_name = global_waypoint_names[global_waypoint_index]
                    waypoint_arrivals.append(
                        {
                            "phase": phase,
                            "waypoint": waypoint_name,
                            "step": step,
                            "distance_metres": waypoint_distance,
                        }
                    )
                    print(f"Route waypoint {waypoint_name} reached at step {step}.", flush=True)
                global_waypoint_index += 1
                active_local_goal = None
                active_local_goal_latitude = None
                active_local_goal_longitude = None

            if global_waypoint_index == len(global_waypoints):
                if arguments.return_to_start and not returning_to_start:
                    print(
                        f"Turnaround reached at step {step}: {distance:.1f} m remaining. "
                        "Returning to start.",
                        flush=True,
                    )
                    goal = start
                    returning_to_start = True
                    if forecast_tacks is not None and arguments.forecast_refinement_phase == "return":
                        forecast_tacks = ForecastTackPlanner(
                            world, dynamics, arguments.no_go_angle, arguments.arrival_radius,
                            learned_selector=learned_selector, arrival_selector=arrival_selector,
                            refine_modes=arguments.forecast_tack_refinement,
                            tracking_feedback=arguments.forecast_tracking_feedback,
                            interval_seconds=arguments.forecast_tack_step,
                            gate_aim_fraction=gate_aim_fraction,
                            gate_aim_upwind_angle=gate_aim_upwind_angle,
                            maneuver_commit_seconds=maneuver_commit_seconds,
                            tack_gain_seconds=tack_gain_seconds,
                            tack_gain_metres=tack_gain_metres,
                        )
                        warn_tack_commitment_scale(forecast_tacks)
                        planner.forecast_tacks = forecast_tacks
                        print("Return-phase forecast refinement activated.", flush=True)
                    if route is not None:
                        return_waypoints = tuple(reversed(route.waypoints[:-1]))
                        global_waypoints = tuple(
                            Waypoint(waypoint.latitude, waypoint.longitude)
                            for waypoint in return_waypoints
                        )
                        global_waypoint_names = tuple(
                            waypoint.name for waypoint in return_waypoints
                        )
                    else:
                        global_waypoints = build_global_waypoints(
                            state,
                            goal,
                            arguments.global_waypoint_spacing,
                        )
                        global_waypoint_names = tuple(
                            f"{index + 1:03d}" for index in range(len(global_waypoints))
                        )
                    global_waypoint_index = 0
                    active_local_map = None
                    active_plan = None
                    active_local_goal = None
                    active_local_goal_latitude = None
                    active_local_goal_longitude = None
                    continue
                destination = "Start" if returning_to_start else "Goal"
                print(
                    f"{destination} reached at step {step}: {distance:.1f} m remaining.",
                    flush=True,
                )
                mission_completed = True
                break

            if arguments.steps is not None and step >= arguments.steps:
                break

            native_decision_started = (time.monotonic() if getattr(arguments, "native_control_timing", False)
                                       else None)
            local_goal_reached = (
                arguments.plot_mode != "local_goal"
                or arguments.live_ais_host is not None
                or arguments.ais_replay is not None
                or time_aware_ais
                or
                active_local_goal is None
                or (
                    active_local_goal_latitude is not None
                    and active_local_goal_longitude is not None
                    and haversine_metres(
                        state.latitude,
                        state.longitude,
                        active_local_goal_latitude,
                        active_local_goal_longitude,
                    ) <= arguments.local_goal_radius
                )
            )
            if local_goal_reached or forecast_tacks is not None or acados_controller is not None or prioritize_waypoint_layline:
                observation = world.observe(
                    state,
                    current_time,
                    ais_elapsed_seconds=step * arguments.dt,
                )
                for target in observation.ais_targets:
                    if target.mmsi is not None:
                        ais_trajectories.setdefault(int(target.mmsi), []).append(
                            (target.latitude, target.longitude)
                        )
                extra_blocked = None
                traffic = None
                traffic_preview = None
                suspend_recovery = None
                active_decisions = ()
                active_local_map = local_builder.build(observation, active_waypoint)
                if time_aware_ais:
                    traffic = AISPrediction.from_observation(
                        observation, acceleration_bound_mps2=getattr(arguments, "ais_acceleration_bound", 0.0),
                        clearance_reserve_metres=ais_clearance_reserve,
                    )
                    traffic_preview = (traffic.for_preview(ais_preview_age_budget, arguments.dt)
                                       if ais_preview_age_budget else traffic)
                if colreg_policy is not None:
                    assessment_observation = observation
                    if traffic is not None:
                        targets = tuple(
                            replace(target, latitude=lat, longitude=lon)
                            for target, point in zip(traffic.targets, traffic.positions_metres)
                            for lat, lon in (offset_latitude_longitude(
                                state.latitude, state.longitude, *point,
                            ),)
                        )
                        assessment_observation = replace(observation, ais_targets=targets)
                    decisions = colreg_policy.assess(state, assessment_observation)
                    active_decisions = tuple(decision for decision in decisions if decision.active)
                    if active_decisions:
                        print(
                            "COLREG: "
                            + " | ".join(decision.summary() for decision in active_decisions),
                            flush=True,
                        )
                        if not time_aware_ais:
                            extra_blocked = colreg_policy.extra_blocked_mask(
                                active_local_map.tensor,
                                state,
                                decisions,
                            )
                if (limit_upwind_drift or prioritize_waypoint_layline or compare_waypoint_layline_eta) and time_aware_ais:
                    assert buffer_planner is not None and traffic is not None and ais_guidance is not None
                    retained_detour = ais_candidate_handoff and ais_guidance.avoidance_heading is not None
                    suspend_recovery = buffer_planner.blocks_open_water_recovery(
                        active_local_map.tensor, traffic, bool(active_decisions) or retained_detour,
                        candidate_handoff=ais_candidate_handoff,
                    )
                if forecast_tacks is not None:
                    forecast_tacks.continuation_goal = (
                        global_waypoints[global_waypoint_index + 1]
                        if global_waypoint_index + 1 < len(global_waypoints) else None
                    )
                active_plan = planner.plan(
                    active_local_map,
                    observation,
                    active_waypoint,
                    extra_blocked=extra_blocked,
                    time_aware_ais=time_aware_ais,
                    suspend_recovery=suspend_recovery,
                )
                centre = (active_local_map.tensor.cells - 1) / 2.0
                target_row, target_column = active_plan.target_cell
                east = (float(target_column) - centre) * active_local_map.tensor.resolution
                north = (centre - float(target_row)) * active_local_map.tensor.resolution
                active_local_goal_latitude, active_local_goal_longitude = offset_latitude_longitude(
                    state.latitude,
                    state.longitude,
                    east,
                    north,
                )
                active_local_goal = active_plan.target_cell
                if (arguments.plot_mode == "local_goal" and local_goal_reached) or (
                    arguments.plot_mode == "periodic"
                    and arguments.plot_every > 0
                    and step % arguments.plot_every == 0
                ):
                    output = output_dir / f"{arguments.planner}_step_{step:04d}.png"
                    figure = plot_planner_snapshot(
                        global_map,
                        active_local_map,
                        active_plan,
                        state,
                        active_waypoint,
                        trajectory=[(s.latitude, s.longitude) for s in trajectory],
                        mode=arguments.planner,
                        output=output,
                    )
                    import matplotlib.pyplot as plt
                    plt.close(figure)

            local_map = active_local_map
            plan = active_plan
            assert local_map is not None and plan is not None
            ais_navigation = {}
            ais_override = False
            escape_heading = None
            maneuver = None
            if ais_guidance is not None:
                assert traffic_preview is not None and buffer_planner is not None
                previous_recovery_profile = getattr(buffer_planner, "recovery_search_profile", None)
                decision_timing = {
                    "step": step + 1,
                    "phase": phase, "waypoint": global_waypoint_names[global_waypoint_index],
                    "observation_utc": str(observation.time),
                    "nominal_course_deg": plan.desired_heading, "prefer_starboard": bool(active_decisions),
                }
                maneuver_search_started = time.monotonic()
                try:
                    maneuver = buffer_planner.choose(
                        observation, local_map.tensor, plan.blocked, traffic_preview,
                        plan.desired_heading, bool(active_decisions),
                    )
                finally:
                    maneuver_search_seconds = time.monotonic() - maneuver_search_started
                    # A safety stop can reject this update before it reaches
                    # replay recording. Keep every decision's measured cost.
                    print("AIS decision timing: " + json.dumps({
                        **decision_timing, "wall_seconds": maneuver_search_seconds,
                    }), flush=True)
                if maneuver is not None:
                    ais_override = True
                    escape_heading = maneuver.bow_heading
                    ais_navigation = {
                        "ais_mode": "buffer maneuver",
                        "ais_course_margin_m": maneuver.predicted_margin_metres,
                        "ais_25m_margin_m": maneuver.nominal_margin_metres,
                        "ais_refresh_margin_m": maneuver.refreshed_margin_metres,
                    }
                    if getattr(maneuver, "predictive_fallback", False):
                        ais_navigation.update(ais_mode="predictive fallback",
                                              ais_control="receding-horizon escape")
                    elif getattr(maneuver, "recovery_controls", False):
                        ais_navigation.update(
                            ais_control="sustained recovery",
                            ais_recovery_active=1,
                            ais_recovery_reserve_margin_m=maneuver.predicted_margin_metres,
                            ais_recovery_remaining_s=maneuver.recovery_remaining_seconds,
                            ais_recovery_departure_turn_deg=maneuver.recovered_departure_turn_deg,
                            ais_recovery_backend=getattr(maneuver, "recovery_backend", None) or "native full NLP",
                            ais_colreg_soft_override=int(bool(active_decisions) and maneuver.colreg_soft_override),
                        )
                    elif maneuver.refined_controls:
                        ais_navigation["ais_control"] = "refined maneuver"
                    else:
                        ais_navigation["ais_escape_bow_deg"] = maneuver.bow_heading
                else:
                    reference = ais_guidance.choose_reference(
                        observation, local_map.tensor, plan.blocked, traffic_preview,
                        plan.desired_heading, bool(active_decisions),
                    )
                    escape_heading = reference.escape_heading if reference.heading is None else reference.heading
                    ais_override = reference.waypoint is not None or reference.heading is None
                    if ais_override:
                        plan = replace(
                            plan, desired_heading=reference.heading, navigation_target=reference.waypoint,
                            desired_bow_heading=None, forecast_closest_metres=None,
                            forecast_tack_in_minutes=None,
                        )
                    ais_navigation = {
                        "ais_mode": "checking fallback" if reference.heading is None else (
                            "avoidance course" if ais_override else "route"
                        ),
                        "ais_course_margin_m": reference.predicted_margin_metres,
                    }
                ais_navigation["ais_maneuver_search_seconds"] = maneuver_search_seconds
                if ais_candidate_handoff:
                    ais_navigation["ais_weather_candidate_admitted"] = suspend_recovery is False
                recovery_profile = getattr(buffer_planner, "recovery_search_profile", None)
                if recovery_profile is not None and recovery_profile is not previous_recovery_profile:
                    ais_navigation["ais_recovery_search_profile"] = dict(recovery_profile)
                if colreg_policy is not None:
                    ais_navigation["ais_colreg_active_contacts"] = len(active_decisions)
                if ais_override:
                    planner.suspend_open_water_recovery("AIS steering override")
                    plan = replace(
                        plan, drift_mode=None, desired_bow_heading=None,
                        forecast_closest_metres=None, forecast_tack_in_minutes=None,
                    )
            nmpc_navigation = {}
            if acados_controller is not None:
                path_start = Waypoint(state.latitude, state.longitude)
                path_target = plan.navigation_target
                waypoint_name = global_waypoint_names[global_waypoint_index]
                leg = phase, global_waypoint_index
                if leg != leg_reference:
                    leg_reference, leg_start, leg_start_step = leg, path_start, step
                if path_target == active_waypoint:
                    path_start = leg_start
                if tack_plan is not None and not ais_override:
                    leg_step = step - leg_start_step
                    recorded = tack_plan.reference(phase, waypoint_name, leg_step)
                    tack_source = "unrecorded leg"
                    if recorded is not None:
                        centre = local_map.tensor.cells // 2
                        bearing = radians(recorded.course)
                        end = centre - cos(bearing), centre + sin(bearing)
                        if planner._segment_is_clear(plan.blocked, (centre, centre), end):
                            plan = replace(
                                plan, desired_heading=recorded.course,
                                desired_bow_heading=recorded.bow,
                                navigation_target=recorded.target,
                            )
                            path_start, path_target = recorded.start, recorded.target
                            tack_source = "recorded"
                        else:
                            tack_source = "blocked reference"
                    tack_sources[tack_source] += 1
                    nmpc_navigation["tack_plan"] = tack_source
                    nmpc_navigation["tack_plan_leg_step"] = leg_step
                if path_target is None and plan.desired_heading is not None:
                    angle = radians(plan.desired_heading)
                    distance = local_map.tensor.resolution
                    path_target = Waypoint(*offset_latitude_longitude(
                        state.latitude, state.longitude, distance * sin(angle), distance * cos(angle),
                    ))
            if plan.desired_heading is None and arguments.dynamics != "legacy_kinematic" and not time_aware_ais:
                raise RuntimeError(f"{arguments.planner} planner produced no feasible heading at step {step}.")
            if plan.desired_heading is None and not time_aware_ais:
                print(
                    f"step={step:04d}: {arguments.planner} planner found no free heading "
                    "(boxed in by traffic/obstacles); holding position.",
                    flush=True,
                )

            forecast_navigation = {}
            if plan.desired_bow_heading is not None:
                forecast_navigation = {
                    "bow_reference_deg": plan.desired_bow_heading,
                    "forecast_closest_m": plan.forecast_closest_metres,
                    "forecast_tack_in_min": plan.forecast_tack_in_minutes,
                }
                if forecast_tacks is not None and plan.forecast_closest_metres is not None:
                    forecast_navigation.update(forecast_tacks.tracking_telemetry)
                    forecast_navigation.update(forecast_tacks.gate_telemetry)
                    forecast_navigation.update(forecast_tacks.maneuver_telemetry)
                    forecast_navigation.update(forecast_tacks.tack_economy_telemetry)
                if ((learned_tack_model is not None or learned_arrival_model is not None)
                        and forecast_tacks is not None
                        and plan.forecast_closest_metres is not None
                        and nmpc_navigation.get("tack_plan") != "recorded"):
                    forecast_navigation.update(forecast_tacks.learned_selection)
                    forecast_navigation["learned_tack_selection_utc"] = forecast_tacks.learned_selection_time
            if limit_upwind_drift and plan.drift_mode is not None:
                forecast_navigation.update(
                    drift_mode=plan.drift_mode, steady_speed_mps=plan.steady_speed_mps,
                    gate_vmg_mps=plan.gate_vmg_mps, drift_range_m=plan.drift_range_m,
                )
            if temporal_recovery is not None:
                forecast_navigation.update(planner.temporal_recovery_telemetry)
            if forecast_hold is not None:
                forecast_navigation.update(planner.forecast_hold_telemetry)
            if forecast_release is not None:
                forecast_navigation.update(planner.forecast_release_telemetry)
            if closest_waypoint_layline:
                forecast_navigation.update(planner.waypoint_layline_telemetry)
            if step == 0 and (arguments.live_plot or recorder is not None):
                if reporting is not None and observation.ais_targets:
                    ais_navigation["ais_oldest_report_s"] = max(
                        target.report_age_seconds for target in observation.ais_targets
                    )
                initial_frame = EpisodeFrame(
                    state, observation,
                    navigation={
                        "step": 0,
                        "phase": phase,
                        "waypoint": global_waypoint_names[global_waypoint_index],
                        "waypoint_distance_m": waypoint_distance,
                        "heading_reference_deg": plan.desired_heading,
                        **forecast_navigation,
                        **nmpc_navigation,
                        **ais_navigation,
                    },
                )
                if recorder is not None:
                    initial_frame = recorder.record(
                        initial_frame, local_grid=local_map.tensor, blocked=plan.blocked,
                        weather=display_weather,
                    )
            previous_state = state
            selected_action = None if maneuver is None else maneuver.action
            if limit_upwind_drift and selected_action is None and not ais_override:
                hold_action = drift_trim_action(dynamics, mpc, state, plan)
                if hold_action is not None:
                    if chart_action_is_clear(
                        dynamics, state, observation, hold_action, arguments.dt, local_map.tensor, plan.blocked,
                    ):
                        selected_action = hold_action
                        forecast_navigation["control_source"] = (
                            "forecast release trim" if plan.drift_mode == "forecast_release" else "drift trim"
                        )
                    else:
                        if plan.drift_mode == "forecast_release":
                            for key in planner.forecast_release_telemetry:
                                forecast_navigation.pop(key, None)
                            forecast_navigation.pop("drift_mode", None)
                            forecast_release.invalidate_selection()
                        if temporal_recovery is not None:
                            forecast_navigation.update(
                                temporal_recovery_applied_action="planner",
                                temporal_recovery_source="classical fallback",
                                temporal_recovery_fallback_reason="hold trim rejected by swept chart check",
                            )
                        if forecast_navigation.get("forecast_hold_source") == "archive forecast":
                            forecast_navigation.update(
                                forecast_hold_source="planner fallback",
                                forecast_hold_fallback_reason="hold trim rejected by swept chart check",
                            )
            if arguments.dynamics == "legacy_kinematic":
                state, legacy_velocity_cells = legacy_kinematic_step(
                    state,
                    plan,
                    global_map,
                    local_map.tensor.resolution,
                    legacy_velocity_cells,
                    arguments.dt,
                    arguments.legacy_speed_cells,
                )
                mpc_cost = 0.0
                current_time = current_time + np.timedelta64(
                    int(round(arguments.dt * 1_000_000_000)), "ns"
                )
            else:
                mpc_cost = 0.0
                if combined_controller is not None and selected_action is None and plan.desired_heading is not None:
                    mpc_result = mpc.choose_action(
                        dynamics,
                        state,
                        observation,
                        plan.desired_heading,
                        local_grid=local_map.tensor,
                        harmonic_plan=plan,
                    )
                    centre = local_map.tensor.cells // 2
                    decision = combined_controller.decide(
                        marine_map_tensor(
                            plan.blocked,
                            planner_feature_field(plan),
                            plan.potential,
                            (centre, centre),
                            plan.target_cell,
                        ),
                        marine_global_features(observation, state, arguments.no_go_angle),
                        arguments.planner,
                        state,
                        max_rudder_slew=mpc.max_rudder_change_per_update_degrees,
                        max_flap_slew=mpc.max_flap_change_per_update_degrees,
                    )
                    if decision.accepted:
                        candidate = mpc.evaluate_action(
                            dynamics,
                            state,
                            observation,
                            decision.action,
                            plan.desired_heading,
                            local_grid=local_map.tensor,
                            harmonic_plan=plan,
                        )
                        cost_safe = combined_controller_cost_is_safe(
                            candidate.cost,
                            mpc_result.cost,
                            arguments.combined_controller_max_cost_increase,
                        )
                        action_safe = combined_controller_action_is_safe(
                            decision.action,
                            mpc_result.action,
                            arguments.combined_controller_max_rudder_error,
                            arguments.combined_controller_max_flap_error,
                        )
                        if cost_safe and action_safe:
                            selected_action = decision.action
                            mpc_cost = candidate.cost
                            controller_accepted += 1
                        else:
                            controller_rollout_rejected += 1
                            controller_fallback += 1
                            if not cost_safe:
                                controller_fallback_reasons["rollout_cost"] += 1
                            rudder_mismatch = (
                                abs(decision.action.rudder - mpc_result.action.rudder)
                                > arguments.combined_controller_max_rudder_error
                            )
                            flap_mismatch = (
                                abs(decision.action.flap - mpc_result.action.flap)
                                > arguments.combined_controller_max_flap_error
                            )
                            if rudder_mismatch:
                                controller_fallback_reasons["rudder_disagreement"] += 1
                            if flap_mismatch:
                                controller_fallback_reasons["flap_disagreement"] += 1
                    else:
                        controller_fallback += 1
                        controller_fallback_reasons[decision.reason] += 1
                if selected_action is None and plan.desired_heading is not None:
                    if acados_controller is not None:
                        try:
                            result = acados_controller.choose_action(
                                dynamics, state, observation, plan.desired_heading,
                                local_grid=local_map.tensor, harmonic_plan=plan,
                                goal=path_target, leg_start=path_start,
                                **({"ais_context": traffic_preview} if time_aware_ais else {}),
                            )
                        except AISControlInfeasible as exc:
                            if not time_aware_ais:
                                raise
                            result = None
                            ais_navigation["ais_intervention"] = exc.safety_intervention
                            nmpc_navigation.update({
                                "solver_status": exc.solver_status,
                                "solve_time_ms": exc.solve_time_ms,
                                "solver_fallback": exc.fallback_reason,
                            })
                            suspend_nominal_guidance(planner, forecast_navigation, "AIS controller infeasible")
                        if result is not None:
                            selected_action = result.action
                            mpc_cost = result.cost
                            nmpc_outcomes[result.fallback_reason or result.control_source] += 1
                            if time_aware_ais and result.safety_intervention is not None:
                                ais_navigation["ais_intervention"] = result.safety_intervention
                            nmpc_navigation.update({
                                "solver_status": result.solver_status,
                                "solve_time_ms": result.solve_time_ms,
                                "solver_fallback": result.fallback_reason,
                            })
                            if (result.safety_intervention is not None or result.fallback_reason is not None
                                    or result.control_source != "acados"):
                                suspend_nominal_guidance(planner, forecast_navigation, "Controller action override")
                        nmpc_navigation["controller"] = arguments.controller
                    else:
                        if combined_controller is None:
                            mpc_result = mpc.choose_action(
                                dynamics, state, observation, plan.desired_heading,
                                local_grid=local_map.tensor, harmonic_plan=plan,
                            )
                        selected_action = mpc_result.action
                        mpc_cost = mpc_result.cost
                if time_aware_ais:
                    assert traffic is not None
                    applied_ais_assessment = None
                    if reporting is not None and traffic.targets:
                        ais_navigation["ais_step_report_age_s"] = float(np.max(traffic.report_ages_seconds))
                    if selected_action is not None and traffic.targets:
                        chart_clear, assessed = assess_action(
                            dynamics, state, observation, selected_action, arguments.dt,
                            local_map.tensor, plan.blocked, traffic,
                        )
                        step_clear = chart_clear and (assessed.safe or (assessed.recovering and not ais_predictive_fallback))
                        if step_clear:
                            applied_ais_assessment = assessed
                            if use_acados and maneuver is not None:
                                nmpc_outcomes["AIS " + ais_navigation["ais_mode"]] += 1
                        else:
                            selected_action = None
                    if selected_action is None:
                        suspend_nominal_guidance(planner, forecast_navigation, "AIS action fallback")
                        fallback = fallback_action(
                            dynamics, mpc, state, observation, local_map.tensor, plan.blocked,
                            traffic, arguments.dt, escape_heading=escape_heading,
                            full_clearance_only=ais_predictive_fallback,
                        )
                        if ais_predictive_fallback and (fallback is None or not fallback.assessment.safe):
                            elapsed = time.monotonic() - maneuver_search_started
                            if native_decision_started is not None:
                                print("Native control decision timing: " + json.dumps({
                                    **decision_timing, "wall_seconds": time.monotonic() - native_decision_started,
                                    "applied": False,
                                }), flush=True)
                            print("AIS applied decision timing: " + json.dumps({
                                **decision_timing, "wall_seconds": elapsed, "applied": False,
                            }), flush=True)
                            termination_reason = "no_full_clearance_action"
                            print(f"AIS strict clearance stop before step {step + 1}: no tested full-clear action.",
                                  flush=True)
                            break
                        if fallback is None:
                            termination_reason = "no_chart_clear_action"
                            print(f"AIS fallback stopped before step {step + 1}: no chart-clear action.", flush=True)
                            break
                        if fallback.assessment.minimum_hull_clearance_metres <= 0.0:
                            termination_reason = "predicted_hull_contact"
                            print(f"AIS fallback stopped before step {step + 1}: all chart-clear actions predict hull contact.",
                                  flush=True)
                            break
                        selected_action = fallback.action
                        mpc_cost = 0.0
                        ais_navigation["ais_mode"] = f"fallback {fallback.mode}"
                        ais_navigation["ais_escape_heading_deg"] = escape_heading
                        applied_ais_assessment = fallback.assessment
                        if use_acados:
                            nmpc_outcomes[f"AIS fallback {fallback.mode}"] += 1
                    if traffic.targets and applied_ais_assessment is not None:
                        margin = applied_ais_assessment.minimum_clearance_metres
                        ais_navigation["ais_step_margin_m"] = margin + traffic.clearance_reserve_metres
                        if traffic.clearance_reserve_metres:
                            ais_navigation["ais_step_reserve_margin_m"] = margin
                    ais_outcomes[ais_navigation["ais_mode"]] += 1
                    if ais_predictive_fallback:
                        elapsed = time.monotonic() - maneuver_search_started
                        ais_navigation["ais_applied_decision_seconds"] = elapsed
                        print("AIS applied decision timing: " + json.dumps({
                            **decision_timing, "wall_seconds": elapsed, "applied": True,
                        }), flush=True)
                if use_acados and forecast_navigation.get("control_source") == "drift trim":
                    nmpc_outcomes["drift trim"] += 1
                assert selected_action is not None
                if native_decision_started is not None:
                    native_seconds = time.monotonic() - native_decision_started
                    nmpc_navigation["native_control_decision_seconds"] = native_seconds
                    print("Native control decision timing: " + json.dumps({
                        "step": step + 1, "phase": phase, "waypoint": global_waypoint_names[global_waypoint_index],
                        "observation_utc": str(observation.time), "wall_seconds": native_seconds, "applied": True,
                    }), flush=True)
                transition = world.step(
                    dynamics,
                    state,
                    selected_action,
                    current_time,
                    arguments.dt,
                    current_depth=0.0,
                    ais_elapsed_seconds=step * arguments.dt,
                )
                state = transition.next_observation.boat
                current_time = transition.next_observation.time
                observation = transition.next_observation
            if arguments.dynamics == "legacy_kinematic":
                observation = world.observe(
                    state,
                    current_time,
                    ais_elapsed_seconds=(step + 1) * arguments.dt,
                )
            trajectory.append(state)

            if arguments.live_plot or recorder is not None:
                if reporting is not None and observation.ais_targets:
                    ais_navigation["ais_oldest_report_s"] = max(
                        target.report_age_seconds for target in observation.ais_targets
                    )
                waypoint_distance_after = haversine_metres(
                    state.latitude, state.longitude, active_waypoint.latitude, active_waypoint.longitude,
                )
                if time_aware_ais and waypoint_distance_after <= arrival_radius:
                    ais_navigation["waypoint_event"] = f"{global_waypoint_names[global_waypoint_index]} reached"
                if (
                    ais_course_forecaster is not None
                    and step % arguments.ais_course_predict_every == 0
                ):
                    nearby_mmsi = []
                    for target in observation.ais_targets:
                        if target.mmsi is None:
                            continue
                        east, north = local_east_north_metres(
                            target.latitude,
                            target.longitude,
                            state.latitude,
                            state.longitude,
                        )
                        if float(np.hypot(east, north)) <= arguments.live_plot_radius:
                            nearby_mmsi.append(int(target.mmsi))
                    latest_ais_forecasts, forecast_stats = (
                        ais_course_forecaster.predict_targets(
                            nearby_mmsi,
                            current_time,
                        )
                    )
                    print(
                        "AIS course forecasts: "
                        f"{forecast_stats['forecasted']}/{forecast_stats['requested']} nearby "
                        f"(history={forecast_stats['insufficient_history']}, "
                        f"gaps={forecast_stats['reporting_gap']}, "
                        f"missing={forecast_stats['missing']}).",
                        flush=True,
                    )
                frame = EpisodeFrame(
                    state, observation, latest_ais_forecasts,
                    navigation={
                        "step": step + 1,
                        "phase": phase,
                        "waypoint": global_waypoint_names[global_waypoint_index],
                        "waypoint_distance_m": waypoint_distance_after,
                        "heading_reference_deg": plan.desired_heading,
                        **forecast_navigation,
                        **nmpc_navigation,
                        **ais_navigation,
                        "course_over_ground_deg": trajectory_course(previous_state, state),
                        "mpc_cost": mpc_cost,
                        "rudder_command_deg": (
                            selected_action.rudder if selected_action is not None else None
                        ),
                        "flap_command_deg": (
                            selected_action.flap if selected_action is not None else None
                        ),
                    },
                )
                if recorder is not None:
                    frame = recorder.record(
                        frame, local_grid=local_map.tensor, blocked=plan.blocked,
                        weather=display_weather,
                    )
            if arguments.live_plot:
                if live_view is None:
                    live_view = SaildroneAnimation(
                        [initial_frame, frame] if initial_frame is not None else [frame],
                        interval_ms=arguments.live_plot_interval_ms,
                        follow_boat=False,
                        show_environment=True,
                        weather=display_weather,
                        view_radius_metres=arguments.live_plot_radius,
                        boat_scale=arguments.live_plot_boat_scale,
                        local_frame=arguments.live_plot_frame == "boat-centered",
                        fixed_local_view=arguments.live_plot_frame == "fixed",
                        method_lines=method_lines,
                    )
                    live_view.set_route(display_route)
                    plt.show(block=False)
                else:
                    live_view.add_frame(frame)
                grid_centre_east, grid_centre_north = local_east_north_metres(
                    local_map.tensor.origin_latitude,
                    local_map.tensor.origin_longitude,
                    live_view.origin.latitude,
                    live_view.origin.longitude,
                )
                live_view.set_local_grid(
                    plan.blocked,
                    local_map.tensor.resolution,
                    grid_centre_east,
                    grid_centre_north,
                )
                if step % arguments.live_plot_every == 0:
                    live_view.render_live()

            if nmea_output is not None:
                ais_sentences: list[str] = []
                for target in observation.ais_targets:
                    if target.mmsi is not None and int(target.mmsi) not in announced_ais_mmsi:
                        ais_sentences.extend(aivdm_static_data_sentences(target))
                        announced_ais_mmsi.add(int(target.mmsi))
                    ais_sentences.append(aivdm_position_report(target))
                nmea_output.send(
                    rmc_sentence(
                        current_time,
                        state.latitude,
                        state.longitude,
                        state.speed,
                        trajectory_course(previous_state, state),
                    ),
                    hdt_sentence(state.heading),
                    *ais_sentences,
                )
                if arguments.nmea_realtime:
                    time.sleep(arguments.dt)
            previous_state = state

            if arguments.plot_mode == "local_goal" and (
                haversine_metres(
                    previous_state.latitude,
                    previous_state.longitude,
                    state.latitude,
                    state.longitude,
                ) < arguments.minimum_progress_metres
            ):
                active_local_goal = None
                active_local_goal_latitude = None
                active_local_goal_longitude = None

            hdg_ref = "  n/a" if plan.desired_heading is None else f"{plan.desired_heading:6.1f}"
            print(
                f"step={step:04d} dist={distance:8.1f}m "
                f"wp={global_waypoint_names[global_waypoint_index]}"
                f"({global_waypoint_index + 1}/{len(global_waypoints)}) "
                f"wp_dist={waypoint_distance:7.1f}m "
                f"hdg_ref={hdg_ref} "
                f"hdg={state.heading:6.1f} sog={state.speed:5.2f} "
                f"rudder={state.rudder:5.1f} flap={state.flap:5.1f} "
                f"mpc_cost={mpc_cost:8.2f}",
                flush=True,
            )
            if (
                arguments.output_gpx is not None
                and arguments.gpx_checkpoint_every > 0
                and step % arguments.gpx_checkpoint_every == 0
            ):
                write_gpx(arguments.output_gpx, trajectory, f"{arguments.planner} Fossen trajectory")
            step += 1
    except Exception:
        save_outputs(arguments, trajectory, ais_trajectories)
        raise
    finally:
        if recorder is not None:
            recorder.close()
        if nmea_output is not None:
            nmea_output.close()
        if live_view is not None:
            import matplotlib.pyplot as plt

            plt.ioff()
        world.close()
    outcome = {
        "status": "completed" if mission_completed else termination_reason or "step_limit",
        "steps": step,
        "phase": phase,
        "return_to_start": arguments.return_to_start,
        "route": route.name if route is not None else None,
        "departure_waypoint": route.departure.name if route is not None else None,
        "arrival_radius_metres": arguments.arrival_radius,
        "waypoint_arrivals": waypoint_arrivals,
        "active_waypoint": None if mission_completed else global_waypoint_names[global_waypoint_index],
        "waypoint_distance_metres": waypoint_distance if np.isfinite(waypoint_distance) else None,
    }
    print(f"Mission outcome: {json.dumps(outcome)}", flush=True)
    if time_aware_ais:
        print(f"AIS avoidance outcomes: {json.dumps(ais_outcomes, sort_keys=True)}", flush=True)
    if use_acados:
        print(f"NMPC outcomes: {json.dumps(nmpc_outcomes, sort_keys=True)}", flush=True)
        if tack_plan is not None:
            print(f"Tack reference sources: {json.dumps(tack_sources, sort_keys=True)}", flush=True)
    if recorder is not None:
        print(f"Saved navigation replay: {arguments.record_replay}", flush=True)
    if arguments.planner == "transformer_diffusion":
        print(
            "Own-ship Diffusion Transformer: "
            f"{planner.transformer_acceptances}/{planner.transformer_predictions} "
            f"predictions accepted, {planner.transformer_fallbacks} safety fallback(s).",
            flush=True,
        )
    if combined_controller is not None:
        print(
            "Combined marine controller: "
            f"{controller_accepted} accepted, {controller_fallback} explicit MPC fallback(s), "
            f"{controller_rollout_rejected} safety-gate rejection(s).",
            flush=True,
        )
        print(
            "Combined marine fallback reasons: "
            + ", ".join(
                f"{reason}={count}"
                for reason, count in sorted(controller_fallback_reasons.items())
            )
            if controller_fallback_reasons
            else "Combined marine fallback reasons: none.",
            flush=True,
        )
    return trajectory, ais_trajectories


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Legacy harmonic planners with Fossen dynamics.")
    p.add_argument(
        "--planner",
        choices=("standard", "gamma", "diffusion", "transformer_diffusion"),
        default="gamma",
    )
    p.add_argument(
        "--transformer-diffusion-checkpoint",
        type=Path,
        help="Own-ship Diffusion Transformer checkpoint.",
    )
    p.add_argument(
        "--transformer-diffusion-device",
        default=None,
        help="Torch device for the own-ship Diffusion Transformer (for example cuda or cpu).",
    )
    p.add_argument(
        "--combined-controller-checkpoint",
        type=Path,
        help="Planner-conditioned rudder/flap Transformer checkpoint (Fossen only).",
    )
    p.add_argument(
        "--combined-controller-device",
        choices=("cpu", "cuda"),
        default=None,
        help="Torch device for the combined actuator controller.",
    )
    p.add_argument(
        "--combined-controller-min-confidence",
        type=float,
        default=0.45,
        help="Below this confidence, use the explicit MPC fallback.",
    )
    p.add_argument(
        "--combined-controller-max-cost-increase",
        type=float,
        default=0.10,
        help=(
            "Maximum fractional MPC rollout-cost increase allowed for a learned "
            "action before explicit MPC fallback."
        ),
    )
    p.add_argument(
        "--combined-controller-max-rudder-error",
        type=float,
        default=3.0,
        help="Maximum learned-versus-MPC rudder difference in degrees.",
    )
    p.add_argument(
        "--combined-controller-max-flap-error",
        type=float,
        default=3.0,
        help="Maximum learned-versus-MPC flap difference in degrees.",
    )
    p.add_argument(
        "--dynamics",
        choices=("fossen", "legacy_kinematic"),
        default="fossen",
        help="fossen uses physical dynamics; legacy_kinematic reproduces old grid movement.",
    )
    p.add_argument(
        "--controller", choices=("grid-mpc", "acados-direct", "acados-los"), default="grid-mpc",
        help="Actuator controller. The acados modes require an optional native acados installation.",
    )
    p.add_argument("--acados-build-dir", type=Path, help="Directory for generated acados solver code.")
    p.add_argument(
        "--tack-plan-replay", type=Path,
        help="Use recorded tack references on each GPX leg for an acados comparison.",
    )
    p.add_argument(
        "--live-plot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show a live Matplotlib panel with the boat hull, sail/flap, rudder, trajectory, and AIS targets.",
    )
    p.add_argument(
        "--live-plot-every",
        type=int,
        default=1,
        help="Redraw the live panel every N steps (higher values trade smoothness for speed).",
    )
    p.add_argument(
        "--live-plot-interval-ms",
        type=int,
        default=100,
        help="Visible delay between live frames in milliseconds.",
    )
    p.add_argument(
        "--live-plot-radius",
        type=float,
        default=120.0,
        help="Follow-camera half-width in metres for --live-plot; smaller shows the boat larger (a few boat-lengths), larger shows more surrounding context.",
    )
    p.add_argument(
        "--live-plot-frame",
        choices=("fixed", "boat-centered"),
        default="fixed",
        help="fixed shows the boat moving from its start; boat-centered keeps it at the origin.",
    )
    p.add_argument(
        "--live-plot-boat-scale",
        type=float,
        default=1.0,
        help="Render the boat hull/sail/rudder this many times larger than true 18 m scale, for visibility (cosmetic only).",
    )
    p.add_argument(
        "--nmea-host",
        help="Enable live UDP NMEA output to OpenCPN at this host.",
    )
    p.add_argument("--nmea-port", type=int, default=10110)
    p.add_argument(
        "--nmea-transport",
        choices=("udp", "tcp-server", "tcp-client"),
        default="udp",
        help="NMEA transport; tcp-server matches the legacy simulator's OpenCPN connection.",
    )
    p.add_argument(
        "--ais-scenario",
        choices=SyntheticAIS.available_scenarios(),
        help="Simulate deterministic marine traffic instead of live AIS (e.g. crossing, head_on, overtaking, mixed).",
    )
    p.add_argument(
        "--ais-report-delay", type=float, default=0.0, metavar="SECONDS",
        help="Delay synthetic AIS delivery while preserving the report timestamp.",
    )
    p.add_argument(
        "--ais-launch-lead", type=float, default=0.0, metavar="SECONDS",
        help="Launch the saildrone this many seconds before the synthetic AIS encounter reference.",
    )
    ais_maneuvering = p.add_mutually_exclusive_group()
    ais_maneuvering.add_argument(
        "--ais-maneuvering", action="store_true",
        help="Have the ships in the synthetic mixed scenario change course and speed.",
    )
    ais_maneuvering.add_argument(
        "--ais-maneuvering-sharp", action="store_true",
        help="Use larger course and speed changes in the synthetic mixed scenario.",
    )
    p.add_argument(
        "--ais-acceleration-bound", type=float, default=0.0, metavar="M/S²",
        help="Assumed upper bound on AIS target ground acceleration for the uncertainty envelope.",
    )
    p.add_argument(
        "--ais-clearance-reserve", type=float, default=0.0, metavar="METRES",
        help="Add this steering reserve to time-aware AIS clearance after accounting for uncertainty.",
    )
    p.add_argument(
        "--ais-preview-age-budget", type=float, default=0.0, metavar="SECONDS",
        help="Assume reports renew by this age for planning. Applied-step checks retain full growth.",
    )
    p.add_argument(
        "--ais-predictive-fallback", action="store_true",
        help="Search predictive escape actions and stop before a command without full applied-step clearance.",
    )
    p.add_argument(
        "--ais-candidate-handoff", action="store_true",
        help="Let weather guidance propose tacks once active avoidance clears. Keep predictive checks on every proposal.",
    )
    p.add_argument(
        "--ais-drop-reports", action="append", nargs=2, type=float,
        metavar=("START", "END"),
        help="Drop synthetic AIS emissions in the elapsed seconds interval [START, END). Repeatable.",
    )
    p.add_argument(
        "--ais-replay",
        type=Path,
        help="Recorded AIS CSV replayed against the simulation clock.",
    )
    p.add_argument(
        "--ais-mmsi",
        type=int,
        help="Optional MMSI filter applied to --ais-replay.",
    )
    p.add_argument(
        "--ais-course-checkpoint",
        type=Path,
        help="Temporal AIS course checkpoint used for live visualization only.",
    )
    p.add_argument(
        "--ais-course-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    p.add_argument(
        "--ais-course-predict-every",
        type=int,
        default=60,
        help="Refresh nearby AIS Transformer forecasts every N simulator steps.",
    )
    p.add_argument(
        "--ais-course-max-gap-seconds",
        type=int,
        default=900,
        help="Maximum reporting gap allowed in a forecast history.",
    )
    p.add_argument(
        "--live-ais-host",
        help="OpenCPN AIS source host/address providing !AIVDM sentences.",
    )
    p.add_argument("--live-ais-port", type=int, default=3000)
    p.add_argument(
        "--live-ais-transport",
        choices=("tcp", "udp"),
        default="tcp",
        help="Transport used by the live AIS source; UDP binds locally to the host/port.",
    )
    p.add_argument(
        "--live-ais-connect-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for the OpenCPN AIS TCP output.",
    )
    p.add_argument(
        "--live-ais-retry-seconds",
        type=float,
        default=2.0,
        help="Delay between live AIS TCP connection attempts.",
    )
    p.add_argument(
        "--colreg",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply AIS-informed COLREG avoidance to local planning.",
    )
    p.add_argument(
        "--time-aware-ais", action=argparse.BooleanOptionalAction, default=None,
        help="Use timed AIS separation and bounded Fossen fallback. Default for Fossen with AIS; "
             "use --no-time-aware-ais to replay the older swept-mask method.",
    )
    p.add_argument(
        "--nmea-realtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait dt seconds between live NMEA updates.",
    )
    p.add_argument(
        "--legacy-speed-cells",
        type=float,
        default=5.0,
        help="Legacy kinematic target speed in local-grid cells per step.",
    )
    p.add_argument(
        "--environment",
        choices=("constant", "marine", "full"),
        default="constant",
        help=(
            "constant uses configured wind only; marine loads measured wind, waves, "
            "and currents; full also loads bathymetry and coastline."
        ),
    )
    p.add_argument("--wind-speed", type=float, default=8.0)
    p.add_argument("--wind-direction", type=float, default=180.0, help="Meteorological FROM direction in degrees.")
    p.add_argument("--start-latitude", type=float, default=39.90660803)
    p.add_argument("--start-longitude", type=float, default=18.391827091)
    p.add_argument("--goal-latitude", type=float, default=39.919176667)
    p.add_argument("--goal-longitude", type=float, default=18.394151667)
    p.add_argument(
        "--route-gpx",
        type=Path,
        help="OpenCPN GPX route; its departure and ordered waypoints replace start/goal.",
    )
    p.add_argument(
        "--route-name",
        help="Optional route name when the GPX contains multiple routes.",
    )
    p.add_argument(
        "--return-to-start",
        action="store_true",
        help="Use the goal coordinates as a turnaround, then navigate back to the start.",
    )
    p.add_argument("--time", default="2026-07-29T06:00:00")
    p.add_argument("--heading", type=float, default=0.0)
    p.add_argument("--flap", type=float, default=-10.0)
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Optional maximum simulation steps; omitted means run until the goal threshold.",
    )
    p.add_argument("--dt", type=float, default=1.0)
    p.add_argument("--arrival-radius", type=float, default=50.0)
    p.add_argument(
        "--global-waypoint-spacing",
        type=float,
        default=5_000.0,
        help="Distance between strategic route waypoints in metres.",
    )
    p.add_argument(
        "--global-waypoint-radius",
        type=float,
        default=250.0,
        help="Distance at which the next strategic waypoint is activated.",
    )
    p.add_argument("--local-span", type=float, default=4715.0, help="41 x 115 m legacy-style local map.")
    p.add_argument("--local-resolution", type=float, default=115.0)
    p.add_argument("--obstacle-buffer", type=float, default=115.0)
    p.add_argument("--minimum-depth", type=float, default=3.0)
    p.add_argument(
        "--global-map",
        type=Path,
        default=None,
        help="GEBCO NetCDF or simulator_harmonic GeoJSON occupancy map.",
    )
    p.add_argument("--no-go-angle", type=float, default=45.0)
    p.add_argument(
        "--forecast-tacks", action="store_true",
        help="Plan open-water tacks ahead using archived wind, waves, and current, with MPC for actuation.",
    )
    p.add_argument(
        "--native-control-timing", action="store_true",
        help="Measure observation through command acceptance, including planning and AIS screening.",
    )
    p.add_argument(
        "--forecast-tack-refinement", action=argparse.BooleanOptionalAction, default=False,
        help="Resolve close-hauled forecast polar angles more finely; disable to reproduce the original modes.",
    )
    p.add_argument(
        "--forecast-tracking-feedback", action=argparse.BooleanOptionalAction, default=False,
        help="Correct forecast schedule velocities from settled tracking observations without changing the command polar.",
    )
    p.add_argument(
        "--forecast-tack-step", type=int, choices=(60, 120), default=120,
        help="Forecast sailing/replanning interval in seconds; the look-ahead remains 48 minutes.",
    )
    p.add_argument(
        "--forecast-refinement-phase", choices=("all", "return"), default="all",
        help="Apply the configured refinement throughout, or preserve the original scheduler until the return phase.",
    )
    p.add_argument(
        "--forecast-gate-aim-fraction", type=float, default=0.0,
        help="Offset the forecast aim toward the next waypoint by up to half the capture radius; true GPX capture is unchanged.",
    )
    p.add_argument(
        "--forecast-gate-aim-upwind-angle", type=float, default=180.0,
        help="Apply gate look-ahead only when the continuation bearing lies within this angle of wind FROM.",
    )
    p.add_argument(
        "--forecast-maneuver-commit-seconds", type=float, default=0.0,
        help="Retain a selected tack side through this initial turn interval before permitting an opposite replan.",
    )
    p.add_argument(
        "--forecast-tack-gain-seconds", type=float, default=0.0,
        help="Require this predicted arrival saving before the forecast first leg crosses to the opposite tack.",
    )
    p.add_argument(
        "--forecast-tack-gain-metres", type=float, default=0.0,
        help="Require this predicted closest-approach gain before crossing tacks when no schedule reaches the mark.",
    )
    p.add_argument(
        "--limit-upwind-drift", action="store_true",
        help="Limit drift with chart-checked sailing trim until the upwind waypoint is reachable over ground.",
    )
    p.add_argument(
        "--commit-recovery-tacks", action="store_true",
        help="Hold a ground layline after drift limiting using observed conditions. Applies to non-forecast guidance.",
    )
    p.add_argument(
        "--closest-waypoint-layline", action="store_true",
        help="Prefer the closest chart-clear tack point that can reach the active waypoint over ground.",
    )
    p.add_argument(
        "--prioritize-waypoint-layline", action="store_true",
        help="Try a feasible closest waypoint layline before hold and forecast guidance once AIS admits route steering.",
    )
    p.add_argument(
        "--compare-waypoint-layline-eta", action="store_true",
        help="Use a feasible closest waypoint layline only if a common forecast predicts a faster waypoint capture.",
    )
    p.add_argument(
        "--forecast-hold-tacks", action="store_true",
        help="Plan tacks during drift limiting with archived wind, waves and current look-ahead.",
    )
    p.add_argument(
        "--forecast-eta-hold-release", action="store_true",
        help="Leave hold on an admissible trim when it reduces distance and forecast time to the active waypoint.",
    )
    p.add_argument(
        "--regatta-time-to-mark", action="store_true",
        help="Rank open-water trims by forecast mark entry and refine tack selection on both legs.",
    )
    p.add_argument(
        "--temporal-recovery-model", type=Path,
        help="Checkpoint for causal temporal hold/release guidance in --limit-upwind-drift.",
    )
    p.add_argument(
        "--confirm-temporal-release", action="store_true",
        help="Release a learned hold when two consecutive Fossen estimates meet the classical release threshold.",
    )
    learned_models = p.add_mutually_exclusive_group()
    learned_models.add_argument(
        "--learned-tack-model", type=Path,
        help="Rank comparable first legs in --forecast-tacks using a trained progress model.",
    )
    learned_models.add_argument(
        "--learned-arrival-model", type=Path,
        help="Rank comparable tack schedules in --forecast-tacks using learned waypoint arrival times.",
    )
    p.add_argument("--plot-every", type=int, default=10)
    p.add_argument(
        "--plot-mode",
        choices=("periodic", "local_goal"),
        default="periodic",
        help="periodic saves every N steps; local_goal saves only after each local-goal replan.",
    )
    p.add_argument(
        "--local-goal-radius",
        type=float,
        default=115.0,
        help="Distance in metres used to declare the cached local goal reached.",
    )
    p.add_argument(
        "--minimum-progress-metres",
        type=float,
        default=1.0,
        help="Replan when a step moves less than this distance, indicating a blocked plan.",
    )
    p.add_argument("--output-dir", default="data/results/legacy_fossen")
    p.add_argument(
        "--output-gpx",
        type=Path,
        help="Optional GPX track containing the simulated trajectory for OpenCPN.",
    )
    p.add_argument(
        "--record-replay",
        type=Path,
        help="Record the route and each displayed navigation state to .jsonl or .jsonl.gz.",
    )
    p.add_argument(
        "--gpx-checkpoint-every",
        type=int,
        default=0,
        help="With --output-gpx, rewrite the GPX file every N steps so it can be reloaded as an OpenCPN track layer mid-run (0 disables).",
    )
    p.add_argument(
        "--global-output",
        type=Path,
        help="Optional PNG showing the complete trajectory on the full global map.",
    )
    p.add_argument(
        "--ais-global-output",
        type=Path,
        help="Optional PNG showing the simulated boat and replayed AIS tracks on GEBCO.",
    )
    p.add_argument(
        "--plot-start-goal",
        type=Path,
        help="Write a GEBCO map with only the selected start and goal, without navigating.",
    )
    return p


def validate_combined_controller_args(arguments: argparse.Namespace) -> None:
    """Validate the actuator model combination without affecting direction mode."""

    if arguments.combined_controller_checkpoint is None:
        return
    if arguments.dynamics != "fossen":
        raise SystemExit("--combined-controller-checkpoint requires --dynamics fossen.")
    if arguments.planner not in {"standard", "gamma", "diffusion"}:
        raise SystemExit(
            "--combined-controller-checkpoint requires --planner standard, gamma, or diffusion."
        )
    if not 0.0 <= arguments.combined_controller_min_confidence <= 1.0:
        raise SystemExit("--combined-controller-min-confidence must be in [0, 1].")
    if arguments.combined_controller_max_cost_increase < 0.0:
        raise SystemExit("--combined-controller-max-cost-increase must be non-negative.")
    if (
        arguments.combined_controller_max_rudder_error < 0.0
        or arguments.combined_controller_max_flap_error < 0.0
    ):
        raise SystemExit("Combined-controller actuator error limits must be non-negative.")


def main() -> None:
    arguments = parser().parse_args()
    validate_combined_controller_args(arguments)
    if arguments.steps is not None and arguments.steps < 1:
        raise SystemExit("--steps must be >=1 when provided.")
    if arguments.dt <= 0.0:
        raise SystemExit("--dt must be >0.")
    if not 1 <= arguments.nmea_port <= 65_535:
        raise SystemExit("--nmea-port must be in [1, 65535].")
    if not 1 <= arguments.live_ais_port <= 65_535:
        raise SystemExit("--live-ais-port must be in [1, 65535].")
    ais_sources = sum(
        value is not None
        for value in (arguments.ais_scenario, arguments.ais_replay, arguments.live_ais_host)
    )
    if ais_sources > 1:
        raise SystemExit("Choose only one of --ais-scenario, --ais-replay, or --live-ais-host.")
    if not np.isfinite(arguments.ais_launch_lead) or arguments.ais_launch_lead < 0.0:
        raise SystemExit("--ais-launch-lead must be finite and non-negative.")
    if arguments.ais_launch_lead and arguments.ais_scenario is None:
        raise SystemExit("--ais-launch-lead requires --ais-scenario.")
    if (arguments.ais_maneuvering or arguments.ais_maneuvering_sharp) and arguments.ais_scenario != "mixed":
        flag = "--ais-maneuvering-sharp" if arguments.ais_maneuvering_sharp else "--ais-maneuvering"
        raise SystemExit(f"{flag} requires --ais-scenario mixed.")
    if not np.isfinite(arguments.ais_acceleration_bound) or arguments.ais_acceleration_bound < 0.0:
        raise SystemExit("--ais-acceleration-bound must be finite and non-negative.")
    if not np.isfinite(arguments.ais_clearance_reserve) or arguments.ais_clearance_reserve < 0.0:
        raise SystemExit("--ais-clearance-reserve must be finite and non-negative.")
    if not np.isfinite(arguments.ais_preview_age_budget) or arguments.ais_preview_age_budget < 0.0:
        raise SystemExit("--ais-preview-age-budget must be finite and non-negative.")
    try:
        synthetic_ais_reporting(arguments)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if arguments.ais_mmsi is not None and arguments.ais_replay is None:
        raise SystemExit("--ais-mmsi requires --ais-replay.")
    if arguments.ais_course_checkpoint is not None:
        if arguments.ais_replay is None:
            raise SystemExit("--ais-course-checkpoint requires --ais-replay.")
        if not arguments.live_plot and arguments.record_replay is None:
            raise SystemExit("--ais-course-checkpoint requires --live-plot or --record-replay.")
    if (
        arguments.planner == "transformer_diffusion"
        and arguments.transformer_diffusion_checkpoint is None
    ):
        raise SystemExit(
            "--planner transformer_diffusion requires "
            "--transformer-diffusion-checkpoint."
        )
    if (
        arguments.planner != "transformer_diffusion"
        and arguments.transformer_diffusion_checkpoint is not None
    ):
        raise SystemExit(
            "--transformer-diffusion-checkpoint requires "
            "--planner transformer_diffusion."
        )
    if arguments.ais_course_predict_every < 1:
        raise SystemExit("--ais-course-predict-every must be >=1.")
    if arguments.ais_course_max_gap_seconds < 1:
        raise SystemExit("--ais-course-max-gap-seconds must be >=1.")
    if arguments.route_name is not None and arguments.route_gpx is None:
        raise SystemExit("--route-name requires --route-gpx.")
    if arguments.gpx_checkpoint_every < 0:
        raise SystemExit("--gpx-checkpoint-every must be >=0.")
    if arguments.live_plot_every < 1:
        raise SystemExit("--live-plot-every must be >=1.")
    if arguments.live_plot_interval_ms < 1:
        raise SystemExit("--live-plot-interval-ms must be >=1.")
    if arguments.live_plot_radius <= 0.0:
        raise SystemExit("--live-plot-radius must be >0.")
    if arguments.live_plot_boat_scale <= 0.0:
        raise SystemExit("--live-plot-boat-scale must be >0.")
    if arguments.live_ais_connect_timeout <= 0.0 or arguments.live_ais_retry_seconds <= 0.0:
        raise SystemExit("Live AIS timeout and retry interval must be positive.")
    if arguments.plot_start_goal is not None:
        if arguments.global_map is None:
            raise SystemExit("--plot-start-goal requires --global-map.")
        route = (
            Route.load_gpx(arguments.route_gpx, route_name=arguments.route_name)
            if arguments.route_gpx is not None
            else None
        )
        start = BoatState(
            latitude=route.departure.latitude if route is not None else arguments.start_latitude,
            longitude=route.departure.longitude if route is not None else arguments.start_longitude,
            heading=arguments.heading,
            speed=0.0,
        )
        goal = (
            Waypoint(route.destination.latitude, route.destination.longitude)
            if route is not None
            else Waypoint(arguments.goal_latitude, arguments.goal_longitude)
        )
        global_map = GlobalOccupancyMap(
            path=arguments.global_map,
            minimum_depth_metres=arguments.minimum_depth,
        ).load()
        figure = plot_start_goal_map(
            global_map,
            start,
            goal,
            output=arguments.plot_start_goal,
        )
        import matplotlib.pyplot as plt
        plt.close(figure)
        print(f"Saved start/goal map: {arguments.plot_start_goal}")
        return
    trajectory, ais_trajectories = run(arguments)
    save_outputs(arguments, trajectory, ais_trajectories)
    if arguments.live_plot:
        import matplotlib.pyplot as plt

        print("Close the live plot window to exit.")
        plt.show()


if __name__ == "__main__":
    main()
