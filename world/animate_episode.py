"""Animate the saildrone, its controls, and environmental forces.

Run from PyCharm in module mode, or from the terminal:

    python -m world.animate_episode --steps 120 --dt 1 --rudder 3 --flap -10
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch, Polygon

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
    WorldObservation,
)
from world._geo import local_east_north_metres, normalise_time, offset_latitude_longitude
from world.ais_avoidance import AISPrediction

if TYPE_CHECKING:
    from world.ais_course_transformer import AISCourseForecast


@dataclass(frozen=True)
class EpisodeFrame:
    """One visual frame from a sampled saildrone episode."""

    state: BoatState
    observation: WorldObservation
    ais_course_forecasts: dict[int, "AISCourseForecast"] = field(default_factory=dict)
    navigation: dict[str, str | float | int | bool | None] = field(default_factory=dict)
    wind_grid: tuple[np.ndarray, np.ndarray] | None = None


def simulate_episode(
    world: MaritimeWorld,
    dynamics: SaildroneDynamics,
    state: BoatState,
    start_time: str,
    action: BoatAction,
    steps: int,
    dt: float,
) -> list[EpisodeFrame]:
    """Generate a deterministic sequence of environment-aware boat states."""

    current_state = state
    current_time = normalise_time(start_time)
    frames = [EpisodeFrame(current_state, world.observe(current_state, current_time))]

    for _ in range(steps):
        transition = world.step(dynamics, current_state, action, current_time, dt)
        current_state = transition.next_observation.boat
        current_time = transition.next_observation.time
        frames.append(EpisodeFrame(current_state, transition.next_observation))

    return frames


class SaildroneAnimation:
    """Matplotlib artists that render state, actuators, and local environment."""

    BOAT_LENGTH = 18.0
    BOAT_WIDTH = 5.0
    BOW_TAPER_FRACTION = 4.0 / BOAT_LENGTH
    RUDDER_LENGTH = 5.0
    SAIL_LENGTH = 14.0
    FLAP_LENGTH = 7.0
    WIND_GRID_SIDE = 5
    WIND_GRID_SPACING = 40.0

    def __init__(
        self,
        frames: list[EpisodeFrame],
        interval_ms: int,
        weather: Weather | None = None,
        *,
        follow_boat: bool = False,
        show_environment: bool = True,
        view_radius_metres: float = 250.0,
        boat_scale: float = 1.0,
        local_frame: bool = False,
        fixed_local_view: bool = False,
        constrained_layout: bool = True,
        method_lines: tuple[str, ...] = (),
    ) -> None:
        if not frames:
            raise ValueError("At least one episode frame is required.")

        self.frames = frames
        self.origin = frames[0].state
        self.interval_ms = interval_ms
        self.weather = weather
        self.follow_boat = follow_boat
        self.show_environment = show_environment
        self.view_radius_metres = view_radius_metres
        self.local_frame = local_frame
        self.fixed_local_view = fixed_local_view
        self._method_readout = "\n".join(method_lines) + "\n\n" if method_lines else ""
        display_scale = float(boat_scale)
        if local_frame or fixed_local_view:
            display_scale = max(display_scale, view_radius_metres * 0.02 / self.BOAT_LENGTH)
        self.boat_length = self.BOAT_LENGTH * display_scale
        self.boat_width = self.BOAT_WIDTH * display_scale
        self.rudder_length = self.RUDDER_LENGTH * display_scale
        self.sail_length = self.SAIL_LENGTH * display_scale
        self.flap_length = self.FLAP_LENGTH * display_scale
        self.positions = np.asarray(
            [
                local_east_north_metres(
                    frame.state.latitude,
                    frame.state.longitude,
                    self.origin.latitude,
                    self.origin.longitude,
                )
                for frame in frames
            ],
            dtype=float,
        )
        self.ais_positions = [self._ais_plot_positions(frame) for frame in frames]

        self.figure, (self.axis, info_axis) = plt.subplots(
            ncols=2, figsize=(13.5, 8),
            gridspec_kw={"width_ratios": (1.55, 1)},
            constrained_layout=constrained_layout,
        )
        info_axis.set_axis_off()
        if self.figure.canvas.manager is not None:
            self.figure.canvas.manager.set_window_title("Saildrone live local panel")
        self.axis.set_xlabel("East displacement [m]")
        self.axis.set_ylabel("North displacement [m]")
        self.axis.set_title(
            "Boat-centred local navigation (north up)"
            if local_frame
            else "Fixed local navigation from start (north up)"
            if fixed_local_view
            else "Saildrone environmental simulation"
        )
        self.axis.set_aspect("equal", adjustable="box")
        self.axis.grid(True, alpha=0.25)
        self._set_bounds()

        # Boat-centred local occupancy grid (obstacles/AIS corridor), shown
        # under the boat when the caller supplies one via set_local_grid().
        # Row 0 of the grid is the most-northern cell, matching this axis'
        # north-up east/north convention, so origin="upper" aligns correctly.
        self.local_grid_image = self.axis.imshow(
            np.zeros((2, 2)),
            origin="upper",
            cmap="Greys",
            vmin=0.0,
            vmax=1.0,
            alpha=0.55,
            extent=(0.0, 1.0, 0.0, 1.0),
            zorder=0.5,
        )
        self.local_grid_image.set_visible(False)

        self.path, = self.axis.plot([], [], color="0.4", linewidth=1.4, label="trajectory")
        self.route_positions = np.zeros((0, 2), dtype=float)
        self.route_labels = {}
        self.route_line, = self.axis.plot(
            [],
            [],
            color="tab:blue",
            linestyle="--",
            linewidth=1.8,
            zorder=3,
            label="mission route",
        )
        self.route_markers = self.axis.scatter(
            [],
            [],
            color="tab:blue",
            edgecolors="white",
            marker="D",
            s=45,
            zorder=4,
        )
        self.start_marker = self.axis.scatter(
            [0.0],
            [0.0],
            color="tab:green",
            marker="o",
            zorder=4,
            label="start",
        )
        self.start_marker.set_visible(not local_frame)
        self.ais_targets = self._create_ais_quiver(np.zeros((0, 2)), (), label=True)
        self.ais_forecast_lines = {}
        for horizon, color in ((5, "#00bcd4"), (10, "#9c27b0"), (15, "#ffc107")):
            collection = LineCollection(
                [],
                colors=color,
                linewidths=1.8,
                linestyles="dashed",
                alpha=0.9,
                zorder=4.5,
                label=f"AIS predicted {horizon} min",
            )
            self.axis.add_collection(collection)
            self.ais_forecast_lines[horizon] = collection
        self.hull = Polygon(
            np.zeros((5, 2)),
            closed=True,
            facecolor="#00c2d7",
            edgecolor="black",
            linewidth=1.5,
            alpha=1.0,
            zorder=8,
        )
        self.axis.add_patch(self.hull)
        self.bow_marker, = self.axis.plot(
            [], [], marker="o", markersize=5, markerfacecolor="#ffc43d", markeredgecolor="black",
            markeredgewidth=0.8, linestyle="None", zorder=10,
        )
        self.ground_course_arrow = FancyArrowPatch(
            (0, 0), (0, 0), color="#793ca3", linewidth=1.6, linestyle="--", arrowstyle="-|>",
            shrinkA=0, shrinkB=0, mutation_scale=13, zorder=10, visible=False,
        )
        self.axis.add_patch(self.ground_course_arrow)
        self.sail, = self.axis.plot(
            [], [], color="black", linewidth=2.0, zorder=9, label="wing sail"
        )
        self.flap, = self.axis.plot(
            [], [], color="tab:red", linewidth=3.0, zorder=9, label="flap"
        )
        self.rudder, = self.axis.plot(
            [], [], color="0.15", linewidth=2.5, zorder=9, label="rudder"
        )
        # A meteorological wind-barb field mirrors the compact symbols used on
        # weather charts.  Each shaft points toward the *FROM* bearing and its
        # feathers encode speed in knots.  Two layers make it readable over a
        # future map/colour overlay as well as this plain animation view.
        self.wind_grid_halo = None
        self.wind_grid = None
        self.wind_dots, = self.axis.plot(
            [],
            [],
            marker="o",
            markersize=4.5,
            markerfacecolor="0.15",
            markeredgecolor="white",
            markeredgewidth=1.6,
            linestyle="None",
            zorder=7,
        )
        self.wind_grid_offsets = self._wind_grid_offsets()
        drawn_wind_cells = np.zeros((self.WIND_GRID_SIDE, self.WIND_GRID_SIDE), dtype=bool)
        drawn_wind_cells[::2, ::2] = True
        wind_clearance = max(
            self.sail_length + self.flap_length,
            self.boat_length / 2.0 + self.rudder_length,
        ) + self.WIND_GRID_SPACING / 2.0
        self.drawn_wind_cells = drawn_wind_cells.ravel() & (
            np.linalg.norm(self.wind_grid_offsets, axis=1) > wind_clearance
        )
        self.current_arrow = FancyArrowPatch(
            (0, 0), (0, 0), color="tab:green", arrowstyle="-|>", mutation_scale=14, zorder=7.5
        )
        self.axis.add_patch(self.current_arrow)
        self.readout = info_axis.text(
            0.02,
            0.98,
            "",
            transform=info_axis.transAxes,
            va="top",
            family="monospace",
            fontsize=9,
            parse_math=False,
        )
        handles, labels = self.axis.get_legend_handles_labels()
        info_axis.legend(handles, labels, loc="lower left", ncol=2, fontsize=8.5)

    def animate(self) -> FuncAnimation:
        """Return a non-repeating animation that advances one simulation frame at a time."""

        animation = FuncAnimation(
            self.figure,
            self.update,
            frames=len(self.frames),
            interval=self.interval_ms,
            repeat=False,
            blit=False,
        )
        # Retain it for the life of the figure; otherwise Matplotlib can
        # garbage-collect the animation before the GUI event loop starts.
        self.figure._saildrone_animation = animation
        return animation

    def add_frame(self, frame: EpisodeFrame) -> None:
        """Append one more simulated step for incremental/live rendering.

        Unlike the constructor, which requires the full episode upfront for a
        post-hoc :meth:`animate` run, this lets a caller stream frames one at
        a time as a live simulation progresses; call :meth:`render_live` after
        each addition (or every few additions) to redraw.
        """

        east, north = local_east_north_metres(
            frame.state.latitude,
            frame.state.longitude,
            self.origin.latitude,
            self.origin.longitude,
        )
        self.frames.append(frame)
        self.positions = np.vstack([self.positions, [[east, north]]])
        self.ais_positions.append(self._ais_plot_positions(frame))

    def _ais_plot_positions(self, frame: EpisodeFrame) -> np.ndarray:
        targets = frame.observation.ais_targets
        if not targets:
            return np.empty((0, 2))
        if "ais_mode" in frame.navigation and any(target.report_age_seconds for target in targets):
            traffic = AISPrediction.from_observation(frame.observation)
            latitudes, longitudes = offset_latitude_longitude(
                traffic.origin_latitude, traffic.origin_longitude, *traffic.positions_metres.T,
            )
        else:
            latitudes, longitudes = zip(*((target.latitude, target.longitude) for target in targets))
        return np.column_stack(local_east_north_metres(
            latitudes, longitudes, self.origin.latitude, self.origin.longitude,
        ))

    def set_local_grid(
        self,
        blocked: np.ndarray,
        resolution: float,
        centre_east: float,
        centre_north: float,
        *,
        frame_index: int = -1,
    ) -> None:
        """Overlay the current boat-centred local occupancy grid as a background.

        ``blocked`` uses the same row/column convention as ``HarmonicPlan``:
        row 0 is the most-northern cell, column 0 the most-western.
        """

        cells = blocked.shape[0]
        half_span = (cells * resolution) / 2.0
        self.local_grid_image.set_data(np.asarray(blocked, dtype=float))
        if self.local_frame:
            centre_east -= float(self.positions[frame_index, 0])
            centre_north -= float(self.positions[frame_index, 1])
        self.local_grid_image.set_extent(
            (
                centre_east - half_span,
                centre_east + half_span,
                centre_north - half_span,
                centre_north + half_span,
            )
        )
        self.local_grid_image.set_visible(True)

    def set_route(self, waypoints) -> None:
        """Overlay mission waypoints in the animation's local metre frame."""

        waypoints = tuple(waypoints)
        self.route_positions = np.asarray(
            [
                local_east_north_metres(
                    waypoint.latitude,
                    waypoint.longitude,
                    self.origin.latitude,
                    self.origin.longitude,
                )
                for waypoint in waypoints
            ],
            dtype=float,
        ).reshape(-1, 2)
        for label in self.route_labels.values():
            label.remove()
        self.route_labels = {
            index: self.axis.annotate(
                waypoint.name, xy=(0.0, 0.0), xytext=(5, 6), textcoords="offset points",
                color="tab:blue", fontsize=8, annotation_clip=True, parse_math=False, zorder=4,
            )
            for index, waypoint in enumerate(waypoints)
            if getattr(waypoint, "name", "")
        }
        self._set_route_artists(len(self.frames) - 1)

    def render_live(self, *, autoscale: bool = True) -> None:
        """Redraw the most recently added frame outside of ``FuncAnimation``.

        Used for a live view driven by an external simulation loop rather than
        Matplotlib's own timer.
        """

        if autoscale:
            self._set_bounds()
        self.update(len(self.frames) - 1)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        self.figure.canvas.start_event_loop(max(0.001, self.interval_ms / 1_000.0))

    def update(self, index: int):
        """Render one sampled state and expose the current forces/readout."""

        frame = self.frames[index]
        east, north = self.positions[index]
        state = frame.state
        observation = frame.observation

        if self.local_frame:
            frame_origin = self.positions[index]
            rendered_positions = self.positions[: index + 1] - frame_origin
            rendered_east, rendered_north = 0.0, 0.0
            rendered_ais = self.ais_positions[index] - frame_origin
        else:
            rendered_positions = self.positions[: index + 1]
            rendered_east, rendered_north = east, north
            rendered_ais = self.ais_positions[index]

        self.path.set_data(rendered_positions[:, 0], rendered_positions[:, 1])
        self._set_route_artists(index)
        hull_points = self._boat_polygon(rendered_east, rendered_north, state.heading)
        self.hull.set_xy(hull_points)
        bow_east, bow_north = hull_points[2]
        self.bow_marker.set_data([bow_east], [bow_north])

        mast = np.array([rendered_east, rendered_north])
        ground_course = self._ground_course(frame)
        self.ground_course_arrow.set_visible(ground_course is not None)
        if ground_course is not None:
            along_ground = self.boat_length * self._bearing_vector(ground_course)
            self.ground_course_arrow.set_positions(mast + 0.85 * along_ground, mast + 1.55 * along_ground)
        apparent_wind_from = self._apparent_wind_from(state, observation)
        # The physical model treats the main wing as a passive weather-vane;
        # its neutral chord aligns with the apparent-wind axis.  The red flap
        # is the commanded deflection that creates aerodynamic lift.
        sail_end = mast + self.sail_length * self._bearing_vector(apparent_wind_from)
        flap_end = sail_end + self.flap_length * self._bearing_vector(
            apparent_wind_from + state.flap
        )
        stern = mast - self.boat_length / 2.0 * self._bearing_vector(state.heading)
        rudder_end = stern + self.rudder_length * self._bearing_vector(
            state.heading + 180.0 + state.rudder
        )

        self.sail.set_data([mast[0], sail_end[0]], [mast[1], sail_end[1]])
        self.flap.set_data([sail_end[0], flap_end[0]], [sail_end[1], flap_end[1]])
        self.rudder.set_data([stern[0], rudder_end[0]], [stern[1], rudder_end[1]])
        self._set_environment_indicators(mast, frame)
        rendered_ais, rendered_targets = self._visible_ais(rendered_ais, observation.ais_targets)
        self._set_ais_forecast_lines(
            rendered_ais,
            rendered_targets,
            frame.ais_course_forecasts,
        )
        if len(rendered_targets) != self.ais_targets.N:
            self.ais_targets.remove()
            self.ais_targets = self._create_ais_quiver(rendered_ais, rendered_targets)
        else:
            ais_vectors = self._ais_arrow_vectors(rendered_targets)
            self.ais_targets.set_offsets(rendered_ais)
            self.ais_targets.set_UVC(
                ais_vectors[:, 0] if ais_vectors.size else np.asarray([]),
                ais_vectors[:, 1] if ais_vectors.size else np.asarray([]),
            )
            self.ais_targets.set_color(self._ais_colours(rendered_targets))
        self.readout.set_text(self._method_readout + self._readout(frame, index, east, north))

        return (
            self.hull,
            self.bow_marker,
            self.ground_course_arrow,
            self.sail,
            self.flap,
            self.rudder,
            self.path,
            self.ais_targets,
            *self.ais_forecast_lines.values(),
            self.readout,
        )

    def _set_ais_forecast_lines(
        self,
        positions: np.ndarray,
        targets: tuple,
        forecasts: dict[int, "AISCourseForecast"],
    ) -> None:
        segments = {horizon: [] for horizon in self.ais_forecast_lines}
        for position, target in zip(positions, targets):
            if target.mmsi is None or target.speed is None:
                continue
            forecast = forecasts.get(int(target.mmsi))
            if forecast is None:
                continue
            for index, horizon in enumerate(forecast.horizons_minutes):
                if horizon not in segments:
                    continue
                displacement = (
                    np.asarray(forecast.directions[index], dtype=float)
                    * max(0.0, float(target.speed))
                    * float(horizon)
                    * 60.0
                )
                segments[horizon].append(
                    np.stack((position, position + displacement))
                )
        for horizon, collection in self.ais_forecast_lines.items():
            collection.set_segments(segments[horizon])

    def _set_route_artists(self, index: int) -> None:
        route_positions = self.route_positions
        if self.local_frame and route_positions.size:
            route_positions = route_positions - self.positions[index]
        self.route_line.set_data(route_positions[:, 0], route_positions[:, 1])
        self.route_markers.set_offsets(route_positions)
        for waypoint_index, label in self.route_labels.items():
            label.xy = tuple(route_positions[waypoint_index])

    def _set_bounds(self) -> None:
        if self.local_frame or self.fixed_local_view:
            radius = self.view_radius_metres
            self.axis.set_xlim(-radius, radius)
            self.axis.set_ylim(-radius, radius)
            return
        if self.follow_boat:
            east, north = self.positions[-1]
            radius = self.view_radius_metres
            traffic = self.ais_positions[-1] if self.ais_positions else np.zeros((0, 2))
            if traffic.size:
                distances = np.hypot(traffic[:, 0] - east, traffic[:, 1] - north)
                radius = max(radius, float(distances.max()) * 1.15)
            self.axis.set_xlim(east - radius, east + radius)
            self.axis.set_ylim(north - radius, north + radius)
            return
        traffic = [positions for positions in self.ais_positions if positions.size]
        all_positions = np.vstack((self.positions, *traffic)) if traffic else self.positions
        minimum = all_positions.min(axis=0)
        maximum = all_positions.max(axis=0)
        margin = max(100.0, 0.15 * float(np.max(maximum - minimum)))
        self.axis.set_xlim(minimum[0] - margin, maximum[0] + margin)
        self.axis.set_ylim(minimum[1] - margin, maximum[1] + margin)

    def _boat_polygon(self, east: float, north: float, heading: float) -> np.ndarray:
        forward = self._bearing_vector(heading)
        right = np.array([forward[1], -forward[0]])
        centre = np.array([east, north])
        stern = centre - self.boat_length / 2.0 * forward
        shoulder = centre + (0.5 - self.BOW_TAPER_FRACTION) * self.boat_length * forward
        bow = centre + self.boat_length / 2.0 * forward
        return np.asarray(
            [
                stern - self.boat_width / 2.0 * right,
                shoulder - self.boat_width / 2.0 * right,
                bow,
                shoulder + self.boat_width / 2.0 * right,
                stern + self.boat_width / 2.0 * right,
            ]
        )

    def _visible_ais(
        self,
        positions: np.ndarray,
        targets: tuple,
    ) -> tuple[np.ndarray, tuple]:
        if not (self.local_frame or self.fixed_local_view) or not positions.size:
            return positions, targets
        visible = np.hypot(positions[:, 0], positions[:, 1]) <= self.view_radius_metres
        return positions[visible], tuple(
            target for target, is_visible in zip(targets, visible) if is_visible
        )

    def _create_ais_quiver(
        self,
        positions: np.ndarray,
        targets: tuple,
        *,
        label: bool = False,
    ):
        vectors = self._ais_arrow_vectors(targets)
        return self.axis.quiver(
            positions[:, 0] if positions.size else np.asarray([]),
            positions[:, 1] if positions.size else np.asarray([]),
            vectors[:, 0] if vectors.size else np.asarray([]),
            vectors[:, 1] if vectors.size else np.asarray([]),
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=self._ais_colours(targets),
            edgecolor="white",
            linewidth=0.8,
            width=0.006,
            headwidth=4.5,
            headlength=6.0,
            headaxislength=5.0,
            pivot="middle",
            zorder=5,
            label="AIS course/speed" if label else None,
        )

    @staticmethod
    def _ais_colours(targets: tuple) -> list[str]:
        return ["#ff8c00" if target.mmsi is None else "#ff4d4d" for target in targets]

    def _ais_arrow_vectors(self, targets: tuple) -> np.ndarray:
        """Return visible local vectors following each target's reported course."""

        vectors = []
        maximum_length = self.view_radius_metres * 0.12
        minimum_length = min(40.0, maximum_length)
        for target in targets:
            bearing = target.course if target.course is not None else target.heading
            if bearing is None:
                vectors.append((0.0, 0.0))
                continue
            speed = max(0.0, target.speed or 0.0)
            length = min(maximum_length, minimum_length + speed * 12.0)
            vectors.append(length * self._bearing_vector(bearing))
        return np.asarray(vectors, dtype=float).reshape(-1, 2)

    def _set_environment_indicators(
        self,
        mast: np.ndarray,
        frame: EpisodeFrame,
    ) -> None:
        if not self.show_environment:
            self._clear_wind_grid()
            self.current_arrow.set_visible(False)
            return
        observation = frame.observation
        if observation.wind is not None and self.drawn_wind_cells.any():
            positions = mast + self.wind_grid_offsets[self.drawn_wind_cells]
            wind_u, wind_v = frame.wind_grid if frame.wind_grid is not None else (
                self.sample_wind_grid(frame.state, observation, self.weather)
            )
            self._draw_wind_grid(
                positions, wind_u[self.drawn_wind_cells], wind_v[self.drawn_wind_cells]
            )
        else:
            self._clear_wind_grid()

        current = observation.current
        if current is not None and current.speed > 0.0:
            current_length = min(40.0, 15.0 + 100.0 * current.speed)
            current_to = self._bearing_vector(current.direction)
            self.current_arrow.set_positions(mast, mast + current_length * current_to)
            self.current_arrow.set_visible(True)
        else:
            self.current_arrow.set_visible(False)

    @classmethod
    def sample_wind_grid(
        cls,
        state: BoatState,
        observation: WorldObservation,
        weather: Weather | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the local wind velocity components in knots for Matplotlib barbs."""

        fallback = observation.wind
        assert fallback is not None
        east, north = cls._wind_grid_offsets().T
        latitudes, longitudes = offset_latitude_longitude(
            state.latitude,
            state.longitude,
            east,
            north,
        )
        samples = []
        for latitude, longitude in zip(latitudes, longitudes):
            try:
                sample = (
                    weather.sample(float(latitude), float(longitude), observation.time)
                    if weather is not None
                    else fallback
                )
            except ValueError:
                # The boat sample is valid.  A near-edge grid cell may fall just
                # outside the forecast raster, so retain a meaningful local cue.
                sample = fallback
            samples.append(sample)

        # Matplotlib uses flow components to draw the feathered shaft toward
        # the meteorological source.
        scale = 1.94384449  # metres per second -> knots
        wind_u = np.asarray([sample.u * scale for sample in samples])
        wind_v = np.asarray([sample.v * scale for sample in samples])
        return wind_u, wind_v

    def _draw_wind_grid(
        self,
        positions: np.ndarray,
        wind_u: np.ndarray,
        wind_v: np.ndarray,
    ) -> None:
        """Draw a white-outlined, black weather-chart wind-barb field."""

        self._clear_wind_grid()
        common = {
            "length": 6,
            "pivot": "middle",
            "barb_increments": {"half": 5, "full": 10, "flag": 50},
        }
        self.wind_grid_halo = self.axis.barbs(
            positions[:, 0],
            positions[:, 1],
            wind_u,
            wind_v,
            barbcolor="white",
            flagcolor="white",
            linewidth=3.5,
            zorder=5,
            **common,
        )
        self.wind_grid = self.axis.barbs(
            positions[:, 0],
            positions[:, 1],
            wind_u,
            wind_v,
            barbcolor="0.1",
            flagcolor="0.1",
            linewidth=1.2,
            zorder=6,
            **common,
        )
        self.wind_dots.set_data(positions[:, 0], positions[:, 1])

    def _clear_wind_grid(self) -> None:
        for wind_barb in (self.wind_grid_halo, self.wind_grid):
            if wind_barb is not None:
                wind_barb.remove()
        self.wind_grid_halo = None
        self.wind_grid = None
        self.wind_dots.set_data([], [])

    @classmethod
    def _wind_grid_offsets(cls) -> np.ndarray:
        """Place an evenly spaced square field around the vessel."""

        coordinates = (
            np.arange(cls.WIND_GRID_SIDE, dtype=float) - (cls.WIND_GRID_SIDE - 1) / 2.0
        ) * cls.WIND_GRID_SPACING
        east, north = np.meshgrid(coordinates, coordinates)
        return np.column_stack((east.ravel(), north.ravel()))

    @staticmethod
    def _bearing_vector(direction: float) -> np.ndarray:
        radians = np.radians(direction)
        return np.array([np.sin(radians), np.cos(radians)])

    @staticmethod
    def _ground_course(frame: EpisodeFrame) -> float | None:
        course = frame.navigation.get("course_over_ground_deg")
        return float(course) if isinstance(course, (int, float, np.floating)) and np.isfinite(course) else None

    @staticmethod
    def _apparent_wind_from(state: BoatState, observation: WorldObservation) -> float:
        """Return the apparent-wind FROM bearing used by the wing-sail drawing."""

        if observation.wind is None:
            return state.heading
        forward = SaildroneAnimation._bearing_vector(state.heading)
        starboard = np.array([forward[1], -forward[0]])
        if state.surge_speed is None:
            ground_velocity = state.speed * forward
        else:
            ground_velocity = state.surge_speed * forward + state.sway_speed * starboard
            if observation.current is not None:
                ground_velocity += np.array([observation.current.u, observation.current.v])
        apparent_velocity = np.array([observation.wind.u, observation.wind.v]) - ground_velocity
        if np.hypot(*apparent_velocity) < 1e-6:
            return observation.wind.direction
        return float(np.degrees(np.arctan2(-apparent_velocity[0], -apparent_velocity[1])) % 360.0)

    @staticmethod
    def _readout(
        frame: EpisodeFrame,
        index: int,
        east_from_start: float = 0.0,
        north_from_start: float = 0.0,
    ) -> str:
        observation = frame.observation
        wind = observation.wind
        wave = observation.wave
        current = observation.current
        wind_line = "Wind: unavailable"
        if wind is not None:
            wind_line = f"Wind: {wind.speed:4.1f} m/s FROM {wind.direction:6.1f}°"
            apparent_from = SaildroneAnimation._apparent_wind_from(frame.state, observation)
            apparent_line = f"Apparent wind: FROM {apparent_from:6.1f}°"
        else:
            apparent_line = "Apparent wind: unavailable"
        wave_line = "Waves: unavailable"
        if wave is not None:
            wave_line = f"Waves: {wave.height:3.1f} m, period {wave.peak_period:3.1f} s"
        current_line = "Current: unavailable"
        if current is not None:
            current_line = f"Current: {current.speed:4.2f} m/s TO   {current.direction:6.1f}°"
        labels = {
            "course_over_ground_deg": "Actual COG deg",
            "ais_25m_margin_m": "AIS nominal margin m",
            "ais_refresh_margin_m": "AIS refreshed forecast m",
            "ais_step_margin_m": "AIS swept step margin m",
            "ais_oldest_report_s": "AIS display age s",
            "ais_step_report_age_s": "AIS age at step start s",
            "ais_colreg_active_contacts": "Active COLREG contacts",
        }
        if frame.navigation.get("ais_refresh_margin_m") is not None:
            labels["ais_course_margin_m"] = "AIS one-report forecast m"
        navigation = [
            f"{labels.get(key, key.replace('_', ' ').capitalize())} {value:.2f}"
            if isinstance(value, (float, np.floating))
            else f"{labels.get(key, key.replace('_', ' ').capitalize())} {value}"
            for key, value in frame.navigation.items()
            if key != "step" and value is not None
        ]
        step = frame.navigation.get("step", index)
        aged_ais = "ais_mode" in frame.navigation and any(
            target.report_age_seconds for target in observation.ais_targets
        )
        bow_and_course = "Yellow dot: bow"
        if SaildroneAnimation._ground_course(frame) is not None:
            bow_and_course += "     Purple arrow: COG"
        return "\n".join(
            [
                f"Step: {str(step).zfill(4)}   "
                f"Time: {np.datetime_as_string(observation.time, unit='s')}",
                f"SOG: {frame.state.speed:4.2f} m/s   Bow heading {frame.state.heading:6.1f}°",
                (
                    f"From start: E {east_from_start:+7.1f} m   "
                    f"N {north_from_start:+7.1f} m"
                ),
                f"Controls: rudder {frame.state.rudder:+5.1f}°   flap {frame.state.flap:+5.1f}°",
                wind_line,
                apparent_line,
                wave_line,
                current_line,
                *("   ".join(navigation[i : i + 2]) for i in range(0, len(navigation), 2)),
                *(["AIS arrows: age-adjusted COG/SOG estimates"] if aged_ais else []),
                bow_and_course,
                "Wind barbs: FROM     Green arrow: current TO",
            ]
        )



def main() -> None:
    parser = argparse.ArgumentParser(description="Animate a multi-step saildrone episode.")
    parser.add_argument("--latitude", type=float, default=39.88)
    parser.add_argument("--longitude", type=float, default=18.42)
    parser.add_argument("--time", default="2026-07-26T21:00:00Z")
    parser.add_argument("--heading", type=float, default=90.0)
    parser.add_argument("--rudder", type=float, default=3.0)
    parser.add_argument("--flap", type=float, default=-10.0)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--interval", type=int, default=100, help="Display interval in milliseconds.")
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
    if arguments.dt <= 0.0 or arguments.interval <= 0:
        parser.error("--dt and --interval must be positive.")
    if arguments.legacy and arguments.steps > 10:
        parser.error("Legacy mode is accelerated; keep --steps at 10 or fewer.")

    world = MaritimeWorld(
        weather=Weather(),
        waves=Waves(),
        currents=Currents(),
        coastline=Coastline(),
    ).load()
    try:
        if arguments.legacy:
            dynamics = SaildroneDynamics(SaildroneParameters.legacy_visualisation())
        elif arguments.model == "fossen_3dof":
            dynamics = Fossen3DOFSaildroneDynamics()
        else:
            dynamics = SaildroneDynamics()
        frames = simulate_episode(
            world,
            dynamics,
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
        )
        visualisation = SaildroneAnimation(frames, arguments.interval, weather=world.weather)
        visualisation.animate()
        plt.show()
    finally:
        world.close()


if __name__ == "__main__":
    main()
