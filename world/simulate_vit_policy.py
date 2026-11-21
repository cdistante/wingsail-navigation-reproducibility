"""Run an environmental saildrone rollout using a trained Vision Transformer policy.

The ViT predicts the desired Earth-frame heading from the local occupancy grid
and environmental/boat-state features. The existing bounded nonlinear MPC
still converts that heading to rudder and flap actions, preserving the
simulator's actuator constraints and Fossen dynamics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ._geo import haversine_metres, normalise_time
from .bathymetry import Bathymetry
from .boat import BoatState
from .coastline import Coastline
from .currents import Currents
from .fossen import Fossen3DOFSaildroneDynamics
from .harmonic_planner import HarmonicTacticalPlanner
from .mpc import SaildroneMPC
from .occupancy import OccupancyGrid
from .scenario import Route, RouteMission, Waypoint
from .vit_dataset import global_environment_features
from .vit_inference import EnvironmentalVisionTransformerPolicy
from .waves import Waves
from .weather import Weather
from .world import MaritimeWorld


def angular_difference(first: float, second: float) -> float:
    """Return the unsigned smallest angular difference in degrees."""

    return abs((first - second + 180.0) % 360.0 - 180.0)


def run_rollout(
    checkpoint: Path | str,
    route: Route,
    *,
    start_time: str,
    heading: float,
    flap: float,
    steps: int,
    dt: float,
    arrival_radius_metres: float = 100.0,
) -> dict[str, object]:
    """Run a non-realtime ViT-heading + MPC simulation and return its summary."""

    if steps < 1 or dt <= 0.0:
        raise ValueError("steps and dt must be positive.")
    policy = EnvironmentalVisionTransformerPolicy(checkpoint)
    grid_span = float(policy.metadata.get("grid_span_metres", 400.0))
    grid_resolution = float(policy.metadata.get("grid_resolution_metres", 10.0))
    grid_builder = OccupancyGrid(span=grid_span, resolution=grid_resolution)
    if grid_builder.cells != policy.grid_size:
        raise ValueError(
            "Checkpoint grid geometry does not match its ViT shape. Rebuild the dataset "
            "or use a checkpoint trained with matching grid settings."
        )
    expected_features = tuple(policy.metadata.get("global_feature_names", ()))
    if len(expected_features) != policy.global_feature_count:
        raise ValueError("Checkpoint global-feature metadata does not match the ViT configuration.")

    saved_bounds = policy.metadata.get("environmental_bounds", {})
    latitude_bounds = saved_bounds.get("latitude") if isinstance(saved_bounds, dict) else None
    longitude_bounds = saved_bounds.get("longitude") if isinstance(saved_bounds, dict) else None
    if latitude_bounds is not None:
        latitude_bounds = tuple(float(value) for value in latitude_bounds)
    if longitude_bounds is not None:
        longitude_bounds = tuple(float(value) for value in longitude_bounds)

    world = MaritimeWorld(
        # Replay precisely the environmental corridor represented in the
        # training archive.  Apart from ensuring train/runtime consistency,
        # subsetting before load avoids expanding full Copernicus wave maps
        # in memory for a local route rollout.
        weather=Weather(
            latitude_bounds=latitude_bounds,
            longitude_bounds=longitude_bounds,
        ),
        waves=Waves(
            latitude_bounds=latitude_bounds,
            longitude_bounds=longitude_bounds,
        ),
        currents=Currents(
            latitude_bounds=latitude_bounds,
            longitude_bounds=longitude_bounds,
        ),
        bathymetry=Bathymetry(),
        coastline=Coastline(),
    ).load()
    try:
        mission = RouteMission(route, arrival_radius_metres=arrival_radius_metres)
        dynamics = Fossen3DOFSaildroneDynamics()
        mpc = SaildroneMPC()
        teacher = HarmonicTacticalPlanner()
        state = BoatState(
            latitude=route.departure.latitude,
            longitude=route.departure.longitude,
            heading=float(heading),
            speed=0.0,
            flap=float(flap),
        )
        current_time = normalise_time(start_time)
        observation = world.observe(state, current_time)
        teacher_errors: list[float] = []
        heading_history: list[float] = []
        cost_history: list[float] = []
        executed_steps = 0
        for step in range(steps):
            mission.update(Waypoint(state.latitude, state.longitude))
            if mission.goal is None:
                break
            local_grid = grid_builder.build(
                state,
                observation,
                goal=mission.goal,
                restricted=bool(observation.chart and observation.chart.is_restricted),
            )
            features = global_environment_features(state, observation, mission.goal)
            predicted_heading = policy.heading(local_grid.data, features)
            # The teacher is diagnostic only. The predicted heading alone is
            # supplied to MPC, making this a genuine ViT-controlled rollout.
            try:
                teacher_plan = teacher.plan(local_grid, observation, mission.goal)
                if teacher_plan.desired_heading is not None:
                    teacher_errors.append(
                        angular_difference(predicted_heading, teacher_plan.desired_heading)
                    )
            except RuntimeError:
                pass
            result = mpc.choose_action(dynamics, state, observation, predicted_heading)
            transition = world.step(dynamics, state, result.action, current_time, dt)
            heading_history.append(predicted_heading)
            cost_history.append(result.cost)
            state = transition.next_observation.boat
            observation = transition.next_observation
            current_time = observation.time
            executed_steps = step + 1
        mission.update(Waypoint(state.latitude, state.longitude))
        final_goal = route.destination if mission.goal is None else mission.goal
        final_distance = haversine_metres(
            state.latitude,
            state.longitude,
            final_goal.latitude,
            final_goal.longitude,
        )
        return {
            "controller": "environmental_vision_transformer_heading_plus_fossen_mpc",
            "route": route.name,
            "checkpoint": str(Path(checkpoint)),
            "start_time": str(normalise_time(start_time)),
            "executed_steps": executed_steps,
            "dt_seconds": float(dt),
            "mission_complete": mission.complete,
            "final_latitude": state.latitude,
            "final_longitude": state.longitude,
            "final_speed_mps": state.speed,
            "final_heading_degrees": state.heading,
            "distance_to_active_goal_metres": final_distance,
            "mean_vit_teacher_heading_error_degrees": (
                None if not teacher_errors else float(np.mean(teacher_errors))
            ),
            "mean_mpc_cost": None if not cost_history else float(np.mean(cost_history)),
            "mean_vit_heading_degrees": (
                None if not heading_history else float(np.mean(heading_history))
            ),
        }
    finally:
        world.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a forecast-replayed Fossen/MPC simulation directed by an environmental ViT."
    )
    parser.add_argument("--checkpoint", required=True, help="Exported environmental ViT .npz checkpoint.")
    parser.add_argument("--route-gpx", required=True)
    parser.add_argument("--route-name")
    parser.add_argument("--time", required=True, help="UTC time inside the local forecast coverage.")
    parser.add_argument("--heading", type=float, default=90.0)
    parser.add_argument("--flap", type=float, default=-10.0)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--arrival-radius", type=float, default=100.0)
    parser.add_argument("--output", help="Optional JSON summary path.")
    arguments = parser.parse_args()
    route = Route.load_gpx(arguments.route_gpx, route_name=arguments.route_name)
    result = run_rollout(
        arguments.checkpoint,
        route,
        start_time=arguments.time,
        heading=arguments.heading,
        flap=arguments.flap,
        steps=arguments.steps,
        dt=arguments.dt,
        arrival_radius_metres=arguments.arrival_radius,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if arguments.output:
        output = Path(arguments.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")


if __name__ == "__main__":
    main()
