"""Animate the boat-centred occupancy tensor used by JEPA and MPC.

Run from PyCharm in module mode, or from the terminal:

    python -m world.animate_occupancy --steps 120 --dt 1 --rudder 3 --flap -10
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyArrowPatch
from matplotlib import patheffects

from world.animate_episode import EpisodeFrame, SaildroneAnimation

from world import (
    BoatAction,
    BoatState,
    Bathymetry,
    Coastline,
    Currents,
    Fossen3DOFSaildroneDynamics,
    LocalGrid,
    MaritimeWorld,
    OccupancyGrid,
    SaildroneDynamics,
    SaildroneParameters,
    SyntheticAIS,
    Waves,
    Weather,
    WorldObservation,
)
from world._geo import normalise_time
from world.scenario import Waypoint


@dataclass(frozen=True)
class OccupancyFrame:
    """A state, observation, and boat-centred grid at one simulation time."""

    state: BoatState
    observation: WorldObservation
    grid: LocalGrid


def simulate_occupancy_episode(
    world: MaritimeWorld,
    dynamics: SaildroneDynamics,
    grid_builder: OccupancyGrid,
    state: BoatState,
    start_time: str,
    action: BoatAction,
    steps: int,
    dt: float,
    *,
    goal: Waypoint | None = None,
) -> list[OccupancyFrame]:
    """Sample a new occupancy tensor after every boat transition."""

    current_state = state
    current_time = normalise_time(start_time)
    elapsed_seconds = 0.0
    observation = world.observe(
        current_state,
        current_time,
        ais_elapsed_seconds=elapsed_seconds,
    )
    frames = [
        OccupancyFrame(
            current_state,
            observation,
            grid_builder.build(
                current_state,
                observation,
                goal=goal,
                restricted=bool(observation.chart and observation.chart.is_restricted),
            ),
        )
    ]

    for _ in range(steps):
        transition = world.step(
            dynamics,
            current_state,
            action,
            current_time,
            dt,
            ais_elapsed_seconds=elapsed_seconds,
        )
        current_state = transition.next_observation.boat
        observation = transition.next_observation
        current_time = observation.time
        elapsed_seconds += dt
        frames.append(
            OccupancyFrame(
                current_state,
                observation,
                grid_builder.build(
                    current_state,
                    observation,
                    goal=goal,
                    restricted=bool(observation.chart and observation.chart.is_restricted),
                ),
            )
        )

    return frames


class OccupancyAnimation:
    """Matplotlib artists for a compact multi-channel local-grid animation."""

    PANEL_NAMES = (
        "navigation",
        "depth",
        "coast_distance",
        "wind_speed",
        "wave_height",
        "current_speed",
    )

    def __init__(self, frames: list[OccupancyFrame], interval_ms: int) -> None:
        if not frames:
            raise ValueError("At least one occupancy frame is required.")

        self.frames = frames
        self.interval_ms = interval_ms
        self.grid = frames[0].grid
        half_span = self.grid.span / 2.0
        self.extent = (-half_span, half_span, -half_span, half_span)
        self.figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
        self.axes = dict(zip(self.PANEL_NAMES, axes.flat))
        self.title = self.figure.suptitle("Boat-centred occupancy grid")

        self.navigation_cmap = ListedColormap(["#f7f7f7", "#d73027", "#1a9850", "#756bb1"])
        self.navigation_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], self.navigation_cmap.N)
        self.images = {}
        self._create_images()

        navigation_axis = self.axes["navigation"]
        self.boat_marker, = navigation_axis.plot(
            [0.0], [0.0], marker="o", markersize=7, color="tab:blue", zorder=7
        )
        self.heading_arrow = self._outlined_arrow("tab:blue", 17)
        navigation_axis.add_patch(self.heading_arrow)

        self.wind_arrow = self._outlined_arrow("tab:purple", 17)
        self.axes["wind_speed"].add_patch(self.wind_arrow)
        self.current_arrow = self._outlined_arrow("tab:green", 17)
        self.axes["current_speed"].add_patch(self.current_arrow)

        self.update(0)

    def animate(self) -> FuncAnimation:
        """Return a non-repeating animation that advances one occupancy frame."""

        animation = FuncAnimation(
            self.figure,
            self.update,
            frames=len(self.frames),
            interval=self.interval_ms,
            repeat=False,
            blit=False,
        )
        self.figure._occupancy_animation = animation
        return animation

    def update(self, index: int):
        """Render one complete local tensor and its environmental vectors."""

        frame = self.frames[index]
        for name, image in self.images.items():
            image.set_data(self._panel_data(name, frame))
        self._set_panel_titles(frame)

        heading = self._bearing_vector(frame.state.heading)
        heading_length = self.grid.span * 0.16
        self.heading_arrow.set_positions((0.0, 0.0), tuple(heading_length * heading))
        self._set_environment_arrows(frame.observation)
        self.title.set_text(
            "  |  ".join(
                [
                    "Boat-centred occupancy grid",
                    f"step {index:04d}",
                    np.datetime_as_string(frame.observation.time, unit="s"),
                    f"speed {frame.state.speed:.2f} m/s",
                    f"heading {frame.state.heading:.1f}°",
                ]
            )
        )
        return tuple(self.images.values()) + (
            self.boat_marker,
            self.heading_arrow,
            self.wind_arrow,
            self.current_arrow,
            self.title,
        )

    def _create_images(self) -> None:
        first = self.frames[0]
        limits = {name: self._limits(name) for name in self.PANEL_NAMES if name != "navigation"}
        for name, axis in self.axes.items():
            if name == "navigation":
                image = axis.imshow(
                    self._panel_data(name, first),
                    extent=self.extent,
                    origin="upper",
                    interpolation="nearest",
                    cmap=self.navigation_cmap,
                    norm=self.navigation_norm,
                )
            else:
                lower, upper = limits[name]
                image = axis.imshow(
                    self._panel_data(name, first),
                    extent=self.extent,
                    origin="upper",
                    interpolation="nearest",
                    cmap="viridis",
                    vmin=lower,
                    vmax=upper,
                )
                self.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("East [m]")
            axis.set_ylabel("North [m]")
            self.images[name] = image

    def _limits(self, name: str) -> tuple[float, float]:
        values = np.concatenate(
            [self._panel_data(name, frame).ravel() for frame in self.frames]
        )
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            return 0.0, 1.0
        upper = float(np.max(finite_values))
        # Environmental channels are almost uniform within this 400 m view,
        # because their source grids have kilometre-scale resolution.  Fixed
        # meaningful lower bounds prevent colourbars from amplifying tiny
        # interpolation changes into misleading visual gradients.
        minimum_ranges = {
            "depth": 100.0,
            "coast_distance": 1_000.0,
            "wind_speed": 15.0,
            "wave_height": 3.0,
            "current_speed": 1.0,
        }
        return 0.0, max(minimum_ranges[name], upper * 1.1)

    def _panel_data(self, name: str, frame: OccupancyFrame) -> np.ndarray:
        if name == "navigation":
            navigation = np.zeros_like(frame.grid.channel("ais_occupied"))
            navigation[frame.grid.channel("ais_occupied") > 0.0] = 1.0
            navigation[frame.grid.channel("goal") > 0.0] = 2.0
            navigation[frame.grid.channel("restricted") > 0.0] = 3.0
            return navigation
        if name == "depth":
            return (
                frame.grid.channel("depth")
                if frame.observation.seabed is not None
                else np.full_like(frame.grid.channel("depth"), np.nan)
            )
        if name == "coast_distance":
            return (
                frame.grid.channel("coast_distance")
                if frame.observation.coast is not None
                else np.full_like(frame.grid.channel("coast_distance"), np.nan)
            )
        if name == "wind_speed":
            return (
                np.hypot(frame.grid.channel("wind_u"), frame.grid.channel("wind_v"))
                if frame.observation.wind is not None
                else np.full_like(frame.grid.channel("wind_u"), np.nan)
            )
        if name == "wave_height":
            return (
                frame.grid.channel("wave_height")
                if frame.observation.wave is not None
                else np.full_like(frame.grid.channel("wave_height"), np.nan)
            )
        if name == "current_speed":
            return (
                np.hypot(frame.grid.channel("current_u"), frame.grid.channel("current_v"))
                if frame.observation.current is not None
                else np.full_like(frame.grid.channel("current_u"), np.nan)
            )
        raise KeyError(f"Unknown display panel: {name}.")

    def _set_panel_titles(self, frame: OccupancyFrame) -> None:
        observation = frame.observation
        self.axes["navigation"].set_title("AIS / goal / restricted\nblue: boat, arrow: heading")

        depth = observation.seabed.depth if observation.seabed is not None else None
        self.axes["depth"].set_title(
            "Depth [m]" if depth is not None else "Depth [m] — no bathymetry data"
        )
        coast = observation.coast.distance if observation.coast is not None else None
        self.axes["coast_distance"].set_title(
            f"Coast distance: {coast:.0f} m" if coast is not None else "Coast distance — unavailable"
        )
        wind = observation.wind
        self.axes["wind_speed"].set_title(
            f"Wind: {wind.speed:.1f} m/s FROM {wind.direction:.0f}°"
            if wind is not None
            else "Wind — unavailable"
        )
        wave = observation.wave
        self.axes["wave_height"].set_title(
            f"Wave height: {wave.height:.1f} m" if wave is not None else "Waves — unavailable"
        )
        current = observation.current
        self.axes["current_speed"].set_title(
            f"Current: {current.speed:.2f} m/s TO {current.direction:.0f}°"
            if current is not None
            else "Current — unavailable"
        )

    def _set_environment_arrows(self, observation: WorldObservation) -> None:
        arrow_length = self.grid.span * 0.24
        if observation.wind is not None:
            # Wind channels contain velocity towards u/v; visual convention is
            # meteorological FROM direction, hence the negation.
            wind_from = -np.array([observation.wind.u, observation.wind.v])
            self.wind_arrow.set_positions((0.0, 0.0), tuple(self._scaled_vector(wind_from, arrow_length)))
        else:
            self.wind_arrow.set_positions((0.0, 0.0), (0.0, 0.0))

        if observation.current is not None:
            current_to = np.array([observation.current.u, observation.current.v])
            self.current_arrow.set_positions(
                (0.0, 0.0), tuple(self._scaled_vector(current_to, arrow_length))
            )
        else:
            self.current_arrow.set_positions((0.0, 0.0), (0.0, 0.0))

    @staticmethod
    def _outlined_arrow(color: str, mutation_scale: float) -> FancyArrowPatch:
        arrow = FancyArrowPatch((0.0, 0.0), (0.0, 0.0), color=color, mutation_scale=mutation_scale)
        arrow.set_path_effects(
            [patheffects.Stroke(linewidth=4.0, foreground="white"), patheffects.Normal()]
        )
        return arrow

    @staticmethod
    def _bearing_vector(direction: float) -> np.ndarray:
        radians = np.radians(direction)
        return np.array([np.sin(radians), np.cos(radians)])

    @staticmethod
    def _scaled_vector(vector: np.ndarray, length: float) -> np.ndarray:
        magnitude = float(np.hypot(*vector))
        return np.zeros(2) if magnitude < 1e-9 else vector / magnitude * length


def main() -> None:
    parser = argparse.ArgumentParser(description="Animate boat-centred occupancy-grid channels.")
    parser.add_argument("--latitude", type=float, default=39.88)
    parser.add_argument("--longitude", type=float, default=18.42)
    parser.add_argument("--time", default="2026-07-26T21:00:00Z")
    parser.add_argument("--heading", type=float, default=90.0)
    parser.add_argument("--rudder", type=float, default=3.0)
    parser.add_argument("--flap", type=float, default=-10.0)
    parser.add_argument(
        "--ais-scenario",
        choices=("none", *SyntheticAIS.available_scenarios()),
        default="none",
        help="Synthetic AIS traffic shown in the grid and environmental animation.",
    )
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--interval", type=int, default=100, help="Display interval in milliseconds.")
    parser.add_argument("--span", type=float, default=400.0, help="Grid width in metres.")
    parser.add_argument("--resolution", type=float, default=5.0, help="Display-grid cell size in metres.")
    parser.add_argument(
        "--model",
        choices=("fossen_3dof", "quasi_steady"),
        default="fossen_3dof",
        help="Dynamics model; legacy mode overrides this selection.",
    )
    parser.add_argument("--goal-latitude", type=float)
    parser.add_argument("--goal-longitude", type=float)
    parser.add_argument(
        "--occupancy-only",
        action="store_true",
        help="Show only the occupancy-grid figure, not the environmental saildrone figure.",
    )
    parser.add_argument("--legacy", action="store_true")
    arguments = parser.parse_args()

    if arguments.steps < 1:
        parser.error("--steps must be at least 1.")
    if arguments.dt <= 0.0 or arguments.interval <= 0:
        parser.error("--dt and --interval must be positive.")
    if arguments.span <= 0.0 or arguments.resolution <= 0.0:
        parser.error("--span and --resolution must be positive.")
    if (arguments.goal_latitude is None) != (arguments.goal_longitude is None):
        parser.error("Specify both --goal-latitude and --goal-longitude, or neither.")
    if arguments.legacy and arguments.steps > 10:
        parser.error("Legacy mode is accelerated; keep --steps at 10 or fewer.")

    goal = (
        Waypoint(arguments.goal_latitude, arguments.goal_longitude)
        if arguments.goal_latitude is not None
        else None
    )
    world = MaritimeWorld(
        weather=Weather(),
        waves=Waves(),
        currents=Currents(),
        bathymetry=Bathymetry(),
        coastline=Coastline(),
        ais=(
            SyntheticAIS(arguments.ais_scenario, arguments.latitude, arguments.longitude)
            if arguments.ais_scenario != "none"
            else None
        ),
    ).load()
    try:
        if arguments.legacy:
            dynamics = SaildroneDynamics(SaildroneParameters.legacy_visualisation())
        elif arguments.model == "fossen_3dof":
            dynamics = Fossen3DOFSaildroneDynamics()
        else:
            dynamics = SaildroneDynamics()
        frames = simulate_occupancy_episode(
            world,
            dynamics,
            OccupancyGrid(span=arguments.span, resolution=arguments.resolution),
            BoatState(
                latitude=arguments.latitude,
                longitude=arguments.longitude,
                heading=arguments.heading,
                speed=0.0,
                flap=arguments.flap,
            ),
            arguments.time,
            BoatAction(rudder=arguments.rudder, flap=arguments.flap),
            arguments.steps,
            arguments.dt,
            goal=goal,
        )
        occupancy_visualisation = OccupancyAnimation(frames, arguments.interval)
        occupancy_visualisation.animate()
        if not arguments.occupancy_only:
            environment_frames = [
                EpisodeFrame(frame.state, frame.observation) for frame in frames
            ]
            environment_visualisation = SaildroneAnimation(
                environment_frames,
                arguments.interval,
                weather=world.weather,
            )
            environment_visualisation.animate()
        plt.show()
    finally:
        world.close()


if __name__ == "__main__":
    main()
