"""Run several saildrone transitions and plot the resulting trajectory.

Example:

    python -m world.plot_episode --steps 600 --dt 1 --heading 90 --flap -10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from world import (
    BoatAction,
    BoatState,
    Coastline,
    Currents,
    Fossen3DOFSaildroneDynamics,
    MaritimeWorld,
    SaildroneDynamics,
    SaildroneParameters,
    Waves,
    Weather,
)


def run_episode(
    world: MaritimeWorld,
    dynamics: SaildroneDynamics,
    state: BoatState,
    start_time: str,
    action: BoatAction,
    steps: int,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Advance a fixed-control episode and return trajectory time series."""

    timestamps = []
    latitudes = []
    longitudes = []
    speeds = []
    headings = []

    current_time = np.datetime64(start_time)
    current_state = state
    for _ in range(steps + 1):
        timestamps.append(current_time)
        latitudes.append(current_state.latitude)
        longitudes.append(current_state.longitude)
        speeds.append(current_state.speed)
        headings.append(current_state.heading)

        if len(timestamps) == steps + 1:
            break

        transition = world.step(dynamics, current_state, action, current_time, dt)
        current_state = transition.next_observation.boat
        current_time = transition.next_observation.time

    elapsed_seconds = np.arange(len(timestamps), dtype=float) * dt
    return (
        elapsed_seconds,
        np.asarray(latitudes),
        np.asarray(longitudes),
        np.asarray(speeds),
        np.asarray(headings),
    )


def plot_episode(
    elapsed_seconds: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    speeds: np.ndarray,
    headings: np.ndarray,
    *,
    legacy: bool,
) -> plt.Figure:
    """Create trajectory, speed, and heading plots for an episode."""

    figure, (trajectory_axis, speed_axis, heading_axis) = plt.subplots(
        3,
        1,
        figsize=(10, 11),
        constrained_layout=True,
    )

    trajectory_axis.plot(longitudes, latitudes, marker="o", markersize=2, linewidth=1.5)
    trajectory_axis.scatter(longitudes[0], latitudes[0], marker="o", label="start", zorder=3)
    trajectory_axis.scatter(longitudes[-1], latitudes[-1], marker="X", label="end", zorder=3)
    trajectory_axis.set_xlabel("Longitude [deg]")
    trajectory_axis.set_ylabel("Latitude [deg]")
    trajectory_axis.set_title("Saildrone trajectory")
    trajectory_axis.legend()
    trajectory_axis.grid(True, alpha=0.3)
    trajectory_axis.set_aspect(1.0 / np.cos(np.radians(np.mean(latitudes))))
    trajectory_axis.ticklabel_format(axis="both", style="plain", useOffset=False)

    speed_axis.plot(elapsed_seconds, speeds)
    speed_axis.set_xlabel("Simulation time [s]")
    speed_axis.set_ylabel("Speed [m/s]")
    speed_axis.set_title("Ground speed" if not legacy else "Legacy signed speed")
    speed_axis.grid(True, alpha=0.3)

    heading_axis.plot(elapsed_seconds, headings)
    heading_axis.set_xlabel("Simulation time [s]")
    heading_axis.set_ylabel("Heading [deg]")
    heading_axis.set_ylim(0.0, 360.0)
    heading_axis.set_title("Heading")
    heading_axis.grid(True, alpha=0.3)

    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a multi-step saildrone episode.")
    parser.add_argument("--latitude", type=float, default=39.88)
    parser.add_argument("--longitude", type=float, default=18.42)
    parser.add_argument("--time", default="2026-07-26T21:00:00Z")
    parser.add_argument("--heading", type=float, default=90.0)
    parser.add_argument("--rudder", type=float, default=0.0)
    parser.add_argument("--flap", type=float, default=-10.0)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument(
        "--model",
        choices=("fossen_3dof", "quasi_steady"),
        default="fossen_3dof",
        help="Dynamics model; legacy mode overrides this selection.",
    )
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional PNG output path instead of a window.")
    arguments = parser.parse_args()

    if arguments.steps < 1:
        parser.error("--steps must be at least 1.")
    if arguments.dt <= 0.0:
        parser.error("--dt must be positive.")
    if arguments.legacy and arguments.steps > 10:
        parser.error("Legacy mode is accelerated; keep --steps at 10 or fewer.")

    world = MaritimeWorld(
        weather=Weather(),
        waves=Waves(),
        currents=Currents(),
        coastline=Coastline(),
    ).load()
    if arguments.legacy:
        dynamics = SaildroneDynamics(SaildroneParameters.legacy_visualisation())
    elif arguments.model == "fossen_3dof":
        dynamics = Fossen3DOFSaildroneDynamics()
    else:
        dynamics = SaildroneDynamics()
    state = BoatState(
        latitude=arguments.latitude,
        longitude=arguments.longitude,
        heading=arguments.heading,
        speed=0.0,
        flap=arguments.flap,
    )
    action = BoatAction(rudder=arguments.rudder, flap=arguments.flap)

    try:
        result = run_episode(
            world,
            dynamics,
            state,
            arguments.time,
            action,
            arguments.steps,
            arguments.dt,
        )
        figure = plot_episode(*result, legacy=arguments.legacy)
        if arguments.output is not None:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(arguments.output, dpi=160)
            print(f"Saved plot: {arguments.output}")
        else:
            plt.show()
    finally:
        world.close()


if __name__ == "__main__":
    main()
