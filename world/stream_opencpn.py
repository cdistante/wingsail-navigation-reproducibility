"""Publish a live simulated GPS/heading feed for OpenCPN.

For a UDP input in OpenCPN, run:

    python -m world.stream_opencpn --steps 54000 --dt 1

For an OpenCPN TCP input at ``localhost:3000``, run:

    python -m world.stream_opencpn --transport tcp-server --port 3000

To use an OpenCPN route export as the simulator mission, add:

    --route-gpx current_route/rotta_corfu_nord.gpx

For harmonic tactical planning with nonlinear rudder/flap MPC, add:

    --guidance harmonic --controller mpc

To enable the conservative AIS-informed good-visibility encounter policy, add:

    --colreg

The simulator sends RMC (position, COG, SOG) and HDT (true heading) sentences.
When a synthetic AIS scenario is selected it also emits AIVDM dynamic and
static target reports that OpenCPN renders as normal AIS traffic.
OpenCPN remains solely responsible for rendering locally licensed charts.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from world import (
    BoatAction,
    BoatState,
    Bathymetry,
    Coastline,
    Currents,
    ColregPolicy,
    Fossen3DOFSaildroneDynamics,
    HarmonicTacticalPlanner,
    MaritimeWorld,
    OccupancyGrid,
    Route,
    RouteMission,
    SaildroneDynamics,
    SaildroneMPC,
    SaildroneParameters,
    SyntheticAIS,
    Waypoint,
    Waves,
    Weather,
)
from world._geo import local_east_north_metres, normalise_time
from world.guidance import WaypointGuidance
from world.nmea import (
    NMEA0183TCPClientOutput,
    NMEA0183TCPServerOutput,
    NMEA0183UDPOutput,
    aivdm_position_report,
    aivdm_static_data_sentences,
    hdt_sentence,
    rmc_sentence,
)


def course_over_ground(
    previous: BoatState | None,
    current: BoatState,
    fallback_heading: float,
) -> float:
    """Derive true COG from consecutive geographic positions."""

    if previous is None:
        return float(fallback_heading % 360.0)
    east, north = local_east_north_metres(
        current.latitude,
        current.longitude,
        previous.latitude,
        previous.longitude,
    )
    if float(np.hypot(east, north)) < 1e-4:
        return float(fallback_heading % 360.0)
    return float(np.degrees(np.arctan2(east, north)) % 360.0)


def nmea_output(transport: str, host: str, port: int):
    """Create the NMEA endpoint matching the configured OpenCPN connection."""

    if transport == "udp":
        return NMEA0183UDPOutput(host, port)
    if transport == "tcp-client":
        return NMEA0183TCPClientOutput(host, port)
    if transport == "tcp-server":
        return NMEA0183TCPServerOutput(host, port)
    raise ValueError(f"Unsupported NMEA transport: {transport!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish simulated position and heading to an OpenCPN NMEA connection."
    )
    parser.add_argument("--latitude", type=float, help="Initial latitude; overrides GPX departure.")
    parser.add_argument("--longitude", type=float, help="Initial longitude; overrides GPX departure.")
    parser.add_argument("--time", help="Initial UTC time; overrides GPX planned departure.")
    parser.add_argument("--heading", type=float, default=90.0)
    parser.add_argument("--rudder", type=float, default=3.0)
    parser.add_argument("--flap", type=float, default=-10.0)
    parser.add_argument("--steps", type=int, default=3_600, help="Number of transitions to publish.")
    parser.add_argument("--dt", type=float, default=1.0, help="Simulation interval in seconds.")
    parser.add_argument("--host", default="127.0.0.1", help="OpenCPN host/listen address.")
    parser.add_argument("--port", type=int, default=10110, help="NMEA port.")
    parser.add_argument("--route-gpx", help="OpenCPN GPX route used as the mission reference.")
    parser.add_argument("--route-name", help="Route name when the GPX file contains multiple routes.")
    parser.add_argument(
        "--ais-scenario",
        choices=("none", *SyntheticAIS.available_scenarios()),
        default="none",
        help="Synthetic traffic encounter exported to OpenCPN as AIS targets.",
    )
    parser.add_argument(
        "--ais-static-interval",
        type=float,
        default=300.0,
        help="Seconds between repeated AIS type-5 static reports (zero disables repeats).",
    )
    parser.add_argument(
        "--colreg",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply the conservative AIS-based good-visibility COLREG policy to "
            "harmonic replans. This is a simulation aid, not a certified "
            "collision-avoidance system."
        ),
    )
    parser.add_argument(
        "--arrival-radius",
        type=float,
        default=100.0,
        help="Waypoint acceptance radius in metres for route-mission progress.",
    )
    parser.add_argument(
        "--follow-route",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use low-order rudder guidance to the active GPX waypoint. Enabled by "
            "default when --route-gpx is supplied."
        ),
    )
    parser.add_argument(
        "--route-rudder-gain",
        type=float,
        default=0.25,
        help="Route-guidance rudder demand per degree of course error.",
    )
    parser.add_argument(
        "--guidance",
        choices=("waypoint", "harmonic"),
        default="waypoint",
        help=(
            "waypoint: steer directly to the active GPX point; harmonic: use the "
            "wind/AIS-aware harmonic planner as the rudder-heading reference."
        ),
    )
    parser.add_argument(
        "--harmonic-grid-span",
        type=float,
        default=400.0,
        help="Width of the tactical harmonic-planning grid in metres.",
    )
    parser.add_argument(
        "--harmonic-grid-resolution",
        type=float,
        default=5.0,
        help="Harmonic-planning grid cell size in metres.",
    )
    parser.add_argument(
        "--harmonic-replan-interval",
        type=float,
        default=5.0,
        help="Seconds between harmonic replans; use 1 for a replan every step.",
    )
    parser.add_argument(
        "--controller",
        choices=("heading", "mpc"),
        default="heading",
        help=(
            "heading: convert the harmonic reference into a proportional rudder "
            "command; mpc: jointly select bounded rudder/flap commands with a "
            "short nonlinear Fossen-model rollout."
        ),
    )
    parser.add_argument(
        "--transport",
        choices=("udp", "tcp-client", "tcp-server"),
        default="udp",
        help=(
            "udp: send to an OpenCPN UDP input; tcp-client: connect to an OpenCPN TCP "
            "output server; tcp-server: accept an OpenCPN TCP input connection."
        ),
    )
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait dt seconds between messages (use --no-realtime for a fast test).",
    )
    parser.add_argument(
        "--model",
        choices=("fossen_3dof", "quasi_steady"),
        default="fossen_3dof",
        help="Dynamics model; legacy mode overrides this selection.",
    )
    parser.add_argument("--legacy", action="store_true")
    arguments = parser.parse_args()

    if arguments.steps < 1:
        parser.error("--steps must be at least 1.")
    if arguments.dt <= 0.0:
        parser.error("--dt must be positive.")
    if not 1 <= arguments.port <= 65_535:
        parser.error("--port must be in [1, 65535].")
    if (arguments.latitude is None) != (arguments.longitude is None):
        parser.error("Specify both --latitude and --longitude, or neither.")
    if arguments.arrival_radius <= 0.0:
        parser.error("--arrival-radius must be positive.")
    if arguments.route_rudder_gain < 0.0:
        parser.error("--route-rudder-gain must be non-negative.")
    if arguments.ais_static_interval < 0.0:
        parser.error("--ais-static-interval must be non-negative.")
    if arguments.harmonic_grid_span <= 0.0 or arguments.harmonic_grid_resolution <= 0.0:
        parser.error("Harmonic grid span and resolution must be positive.")
    if arguments.harmonic_replan_interval <= 0.0:
        parser.error("--harmonic-replan-interval must be positive.")
    if arguments.legacy and arguments.steps > 10:
        parser.error("Legacy mode is accelerated; keep --steps at 10 or fewer.")

    route = (
        Route.load_gpx(arguments.route_gpx, route_name=arguments.route_name)
        if arguments.route_gpx is not None
        else None
    )
    if arguments.latitude is not None:
        latitude, longitude = arguments.latitude, arguments.longitude
    elif route is not None:
        latitude, longitude = route.departure.latitude, route.departure.longitude
    else:
        latitude, longitude = 39.88, 18.42
    if arguments.time is not None:
        start_time = arguments.time
    elif route is not None and route.planned_departure is not None:
        start_time = route.planned_departure
    else:
        start_time = "2026-07-26T21:00:00Z"
    mission = (
        RouteMission(route, arrival_radius_metres=arguments.arrival_radius)
        if route is not None
        else None
    )
    if arguments.guidance == "harmonic" and mission is None:
        parser.error("--guidance harmonic requires --route-gpx.")
    if arguments.guidance == "harmonic" and not arguments.follow_route:
        parser.error("--guidance harmonic requires route following; omit --no-follow-route.")
    if arguments.controller == "mpc" and arguments.guidance != "harmonic":
        parser.error("--controller mpc requires --guidance harmonic.")
    if arguments.colreg and arguments.guidance != "harmonic":
        parser.error("--colreg requires --guidance harmonic.")
    guidance = (
        WaypointGuidance(
            flap_degrees=arguments.flap,
            rudder_gain_per_degree=arguments.route_rudder_gain,
        )
        if mission is not None and arguments.follow_route
        else None
    )
    harmonic_planner = (
        HarmonicTacticalPlanner() if arguments.guidance == "harmonic" else None
    )
    harmonic_grid = (
        OccupancyGrid(
            span=arguments.harmonic_grid_span,
            resolution=arguments.harmonic_grid_resolution,
        )
        if harmonic_planner is not None
        else None
    )
    mpc = SaildroneMPC() if arguments.controller == "mpc" else None
    colreg_policy = ColregPolicy() if arguments.colreg else None
    synthetic_ais = (
        SyntheticAIS(arguments.ais_scenario, latitude, longitude)
        if arguments.ais_scenario != "none"
        else None
    )

    world = MaritimeWorld(
        weather=Weather(),
        waves=Waves(),
        currents=Currents(),
        bathymetry=Bathymetry() if harmonic_planner is not None else None,
        coastline=Coastline(),
        ais=synthetic_ais,
    ).load()
    if arguments.legacy:
        dynamics = SaildroneDynamics(SaildroneParameters.legacy_visualisation())
    elif arguments.model == "fossen_3dof":
        dynamics = Fossen3DOFSaildroneDynamics()
    else:
        dynamics = SaildroneDynamics()

    state = BoatState(
        latitude=latitude,
        longitude=longitude,
        heading=arguments.heading,
        speed=0.0,
        flap=arguments.flap,
    )
    action = BoatAction(rudder=arguments.rudder, flap=arguments.flap)
    current_time = normalise_time(start_time)
    previous_state: BoatState | None = None
    last_harmonic_plan_time = -np.inf
    last_harmonic_plan = None
    last_harmonic_grid = None

    # Fail before publishing a misleading stationary GPS point when an imported
    # route's planned departure falls outside the locally downloaded forecast.
    try:
        observation = world.observe(state, current_time, ais_elapsed_seconds=0.0)
    except ValueError as error:
        world.close()
        if route is not None and arguments.time is None and route.planned_departure is not None:
            raise RuntimeError(
                "The GPX planned departure is outside the locally available environmental "
                "forecast. Download weather/wave/current data covering that time, or pass "
                "--time with an available UTC simulation time."
            ) from error
        raise

    if route is not None:
        first_goal = mission.goal.name if mission is not None and mission.goal is not None else "complete"
        print(
            f"Loaded GPX route {route.name!r}: {len(route.waypoints)} points, "
            f"{route.total_distance_metres / 1000.0:.2f} km, active goal {first_goal!r}."
        )
        if guidance is not None:
            if harmonic_planner is None:
                print("Waypoint guidance is controlling the rudder towards the active route point.")
            else:
                if mpc is None:
                    print(
                        "Harmonic tactical guidance is controlling the rudder from the "
                        "wind/AIS-aware local occupancy grid."
                    )
                else:
                    print(
                        "Harmonic tactical guidance supplies a local path reference; "
                        "nonlinear MPC jointly selects rudder and flap commands."
                    )
    if synthetic_ais is not None:
        print(
            f"Synthetic AIS scenario {arguments.ais_scenario!r}: "
            f"{len(observation.ais_targets)} target(s) will be sent as AIVDM."
        )
    if colreg_policy is not None:
        print(
            "COLREG policy enabled: good-visibility AIS CPA assessment with a "
            "starboard give-way preference (simulation aid only)."
        )
    if arguments.transport == "tcp-server":
        print(f"Waiting for OpenCPN TCP input on {arguments.host}:{arguments.port}.")
    else:
        print(
            f"Publishing NMEA 0183 {arguments.transport} to "
            f"{arguments.host}:{arguments.port}."
        )
    try:
        with nmea_output(arguments.transport, arguments.host, arguments.port) as output:
            for step in range(arguments.steps + 1):
                waypoint_changed = (
                    mission is not None
                    and mission.update(Waypoint(state.latitude, state.longitude))
                )
                if waypoint_changed:
                    next_goal = mission.goal.name if mission.goal is not None else "complete"
                    print(f"Route waypoint reached; active goal is now {next_goal!r}.")
                course = course_over_ground(previous_state, state, state.heading)
                ais_sentences = [
                    aivdm_position_report(target) for target in observation.ais_targets
                ]
                static_interval_steps = (
                    None
                    if arguments.ais_static_interval == 0.0
                    else max(1, int(round(arguments.ais_static_interval / arguments.dt)))
                )
                if static_interval_steps is not None and step % static_interval_steps == 0:
                    for sequence_id, target in enumerate(observation.ais_targets, start=1):
                        ais_sentences.extend(
                            aivdm_static_data_sentences(target, sequence_id=str(sequence_id))
                        )
                output.send(
                    rmc_sentence(
                        current_time,
                        state.latitude,
                        state.longitude,
                        state.speed,
                        course,
                    ),
                    hdt_sentence(state.heading),
                    *ais_sentences,
                )
                if step == arguments.steps:
                    break
                if arguments.realtime:
                    time.sleep(arguments.dt)
                if guidance is None:
                    next_action = action
                elif harmonic_planner is None or harmonic_grid is None or mission is None:
                    next_action = guidance.action(state, mission.goal)
                elif mission.goal is None:
                    next_action = guidance.action(state, None)
                else:
                    elapsed_seconds = step * arguments.dt
                    should_replan = (
                        last_harmonic_plan is None
                        or waypoint_changed
                        or elapsed_seconds - last_harmonic_plan_time
                        >= arguments.harmonic_replan_interval
                    )
                    if should_replan:
                        last_harmonic_grid = harmonic_grid.build(
                            state,
                            observation,
                            goal=mission.goal,
                            restricted=bool(
                                observation.chart and observation.chart.is_restricted
                            ),
                        )
                        colreg_decisions = (
                            colreg_policy.assess(state, observation)
                            if colreg_policy is not None
                            else ()
                        )
                        colreg_blocked = (
                            colreg_policy.extra_blocked_mask(
                                last_harmonic_grid,
                                state,
                                colreg_decisions,
                            )
                            if colreg_policy is not None
                            else None
                        )
                        try:
                            last_harmonic_plan = harmonic_planner.plan(
                                last_harmonic_grid,
                                observation,
                                mission.goal,
                                extra_blocked=colreg_blocked,
                            )
                            last_harmonic_plan_time = elapsed_seconds
                            heading_text = (
                                f"{last_harmonic_plan.desired_heading:.1f}°"
                                if last_harmonic_plan.desired_heading is not None
                                else "none"
                            )
                            print(
                                "Harmonic replan: "
                                f"heading {heading_text}, "
                                f"path {len(last_harmonic_plan.path_cells)} cells, "
                                f"converged={last_harmonic_plan.converged}."
                            )
                            for decision in colreg_decisions:
                                if decision.risk_of_collision:
                                    print(f"COLREG: {decision.summary()}")
                        except RuntimeError as error:
                            last_harmonic_plan = None
                            last_harmonic_plan_time = elapsed_seconds
                            print(f"Harmonic planner unavailable ({error}); using waypoint guidance.")
                    if last_harmonic_plan is not None and last_harmonic_plan.has_path:
                        if mpc is not None and last_harmonic_grid is not None:
                            mpc_result = mpc.choose_action(
                                dynamics,
                                state,
                                observation,
                                last_harmonic_plan.desired_heading,
                                local_grid=last_harmonic_grid,
                                harmonic_plan=last_harmonic_plan,
                            )
                            next_action = mpc_result.action
                            if should_replan:
                                print(
                                    "MPC selected: "
                                    f"rudder {next_action.rudder:+.1f}°, "
                                    f"flap {next_action.flap:+.1f}°, "
                                    f"cost {mpc_result.cost:.2f} "
                                    f"({mpc_result.candidates_evaluated} candidates)."
                                )
                        else:
                            next_action = guidance.action_to_heading(
                                state,
                                last_harmonic_plan.desired_heading,
                            )
                    else:
                        # A fully blocked/no-descent local map must not make
                        # the vessel switch to a different, high-gain steering
                        # law.  Preserve the bounded two-control MPC while
                        # recovering directly toward the active route point.
                        if mpc is not None:
                            recovery_heading = guidance.desired_course(state, mission.goal)
                            mpc_result = mpc.choose_action(
                                dynamics,
                                state,
                                observation,
                                recovery_heading,
                            )
                            next_action = mpc_result.action
                            if should_replan:
                                print(
                                    "Harmonic recovery MPC: "
                                    f"heading {recovery_heading:.1f}°, "
                                    f"rudder {next_action.rudder:+.1f}°, "
                                    f"flap {next_action.flap:+.1f}°."
                                )
                        else:
                            next_action = guidance.action(state, mission.goal)
                transition = world.step(
                    dynamics,
                    state,
                    next_action,
                    current_time,
                    arguments.dt,
                    ais_elapsed_seconds=step * arguments.dt,
                )
                previous_state = state
                state = transition.next_observation.boat
                current_time = transition.next_observation.time
                observation = transition.next_observation
    except KeyboardInterrupt:
        print("OpenCPN stream stopped.")
    finally:
        world.close()


if __name__ == "__main__":
    main()
