"""Animate the legacy planner driving the Fossen saildrone model.

Run from the repository root, for example::

    python -m world.animate_legacy_fossen \
      --global-map data/occupancy/navigation_obstacles.geojson \
      --planner diffusion --steps 120
"""

from __future__ import annotations

import argparse
import os

import matplotlib

from world import (
    BoatState,
    Fossen3DOFSaildroneDynamics,
    MaritimeWorld,
    SaildroneMPC,
    Waypoint,
)
from world._geo import haversine_metres
from world.animate_episode import EpisodeFrame, SaildroneAnimation
from world.global_map import GlobalOccupancyMap
from world.legacy_planner_adapter import LegacyPlannerAdapter
from world.local_map import LegacyLocalMapBuilder
from world.run_legacy_fossen import ConstantWindWeather, build_global_waypoints, goal_reached


if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
    if matplotlib.get_backend().lower().endswith("agg"):
        matplotlib.use("qtagg", force=True)
import matplotlib.pyplot as plt


def simulate_planner_episode(arguments: argparse.Namespace) -> list[EpisodeFrame]:
    """Generate frames from legacy planning, MPC, and Fossen dynamics."""

    world = MaritimeWorld(
        weather=ConstantWindWeather(arguments.wind_speed, arguments.wind_direction)
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
    planner = LegacyPlannerAdapter(mode=arguments.planner)
    dynamics = Fossen3DOFSaildroneDynamics()
    mpc = SaildroneMPC()
    state = BoatState(
        latitude=arguments.start_latitude,
        longitude=arguments.start_longitude,
        heading=arguments.heading,
        speed=0.0,
        flap=arguments.flap,
    )
    goal = Waypoint(arguments.goal_latitude, arguments.goal_longitude)
    global_waypoints = build_global_waypoints(
        state,
        goal,
        arguments.global_waypoint_spacing,
    )
    global_waypoint_index = 0
    current_time = arguments.time
    frames: list[EpisodeFrame] = []

    try:
        step = 0
        while arguments.steps is None or step <= arguments.steps:
            observation = world.observe(
                state,
                current_time,
                ais_elapsed_seconds=step * arguments.dt,
            )
            frames.append(EpisodeFrame(state, observation))
            if goal_reached(state, goal, arguments.arrival_radius):
                break
            if arguments.steps is not None and step == arguments.steps:
                break

            while (
                global_waypoint_index < len(global_waypoints) - 1
                and haversine_metres(
                    state.latitude,
                    state.longitude,
                    global_waypoints[global_waypoint_index].latitude,
                    global_waypoints[global_waypoint_index].longitude,
                )
                <= arguments.global_waypoint_radius
            ):
                global_waypoint_index += 1

            active_waypoint = global_waypoints[global_waypoint_index]
            local_map = local_builder.build(observation, active_waypoint)
            plan = planner.plan(local_map, observation, active_waypoint)
            if plan.desired_heading is None:
                break
            action = mpc.choose_action(
                dynamics,
                state,
                observation,
                plan.desired_heading,
                local_grid=local_map.tensor,
                harmonic_plan=plan,
            ).action
            transition = world.step(
                dynamics,
                state,
                action,
                current_time,
                arguments.dt,
                current_depth=0.0,
                ais_elapsed_seconds=step * arguments.dt,
            )
            state = transition.next_observation.boat
            current_time = transition.next_observation.time
            step += 1
    finally:
        world.close()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Animate the legacy planner with Fossen dynamics."
    )
    parser.add_argument("--global-map", required=True)
    parser.add_argument("--planner", choices=("standard", "gamma", "diffusion"), default="diffusion")
    parser.add_argument("--start-latitude", type=float, default=70.977166356)
    parser.add_argument("--start-longitude", type=float, default=-155.681418082)
    parser.add_argument("--goal-latitude", type=float, default=71.013364051)
    parser.add_argument("--goal-longitude", type=float, default=-155.209698075)
    parser.add_argument("--time", default="2026-07-29T06:00:00")
    parser.add_argument("--heading", type=float, default=0.0)
    parser.add_argument("--flap", type=float, default=-10.0)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Optional maximum frames; omitted means animate until the goal threshold.",
    )
    parser.add_argument("--arrival-radius", type=float, default=50.0)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--interval", type=int, default=100)
    parser.add_argument("--global-waypoint-spacing", type=float, default=5000.0)
    parser.add_argument("--global-waypoint-radius", type=float, default=250.0)
    parser.add_argument("--wind-speed", type=float, default=8.0)
    parser.add_argument("--wind-direction", type=float, default=180.0)
    parser.add_argument("--local-span", type=float, default=4715.0)
    parser.add_argument("--local-resolution", type=float, default=115.0)
    parser.add_argument("--obstacle-buffer", type=float, default=115.0)
    parser.add_argument("--minimum-depth", type=float, default=3.0)
    arguments = parser.parse_args()
    if (
        (arguments.steps is not None and arguments.steps < 1)
        or arguments.dt <= 0.0
        or arguments.interval <= 0
        or arguments.arrival_radius <= 0.0
    ):
        parser.error("steps, dt, interval, and arrival-radius must be positive when provided.")

    frames = simulate_planner_episode(arguments)
    animation = SaildroneAnimation(frames, arguments.interval)
    animation_controller = animation.animate()
    plt.show()


if __name__ == "__main__":
    main()