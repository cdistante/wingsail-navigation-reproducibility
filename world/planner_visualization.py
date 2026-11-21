"""Visualization for the legacy-planner + Maritime World integration.

This replaces the old ``plot_2.py``/Tkinter simulator dependency with figures
that consume the new georeferenced global map, local planner map, Fossen state,
and ``HarmonicPlan`` directly.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .boat import BoatState
from .global_map import GlobalOccupancyMap
from .harmonic_planner import HarmonicPlan
from .local_map import LegacyLocalMap
from .scenario import Waypoint


def plot_planner_snapshot(
    global_map: GlobalOccupancyMap,
    local_map: LegacyLocalMap,
    plan: HarmonicPlan,
    state: BoatState,
    goal: Waypoint,
    *,
    trajectory: list[tuple[float, float]] | tuple[tuple[float, float], ...] = (),
    mode: str = "legacy",
    output: Path | str | None = None,
) -> plt.Figure:
    """Plot global context, local occupancy, potential, and control state."""

    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    ax_global, ax_local, ax_potential, ax_state = axes.ravel()

    _plot_global(ax_global, global_map, state, goal, trajectory)
    _plot_local(ax_local, local_map, plan)
    _plot_potential(ax_potential, plan)
    _plot_state(ax_state, state, plan, mode)

    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150)
    return figure


def plot_global_trajectory(
    global_map: GlobalOccupancyMap,
    trajectory: list[tuple[float, float]],
    start: BoatState,
    goal: Waypoint,
    *,
    mode: str = "legacy",
    ais_trajectories: dict[int, list[tuple[float, float]]] | None = None,
    turnaround: Waypoint | None = None,
    output: Path | str | None = None,
) -> plt.Figure:
    """Plot the geographic trajectory cropped to the route's bounding box.

    A short start-to-goal hop would otherwise be a few invisible pixels on
    the full GEBCO tile, so the crop always includes a minimum margin instead
    of showing the whole loaded map extent.
    """

    global_map._require_loaded()
    assert global_map.occupancy is not None
    assert global_map.latitudes is not None and global_map.longitudes is not None

    figure, axis = plt.subplots(figsize=(12, 10), constrained_layout=True)
    points = [(start.latitude, start.longitude), (goal.latitude, goal.longitude)]
    if turnaround is not None:
        points.append((turnaround.latitude, turnaround.longitude))
    points.extend(trajectory)
    for ais_trajectory in (ais_trajectories or {}).values():
        points.extend(ais_trajectory)
    latitudes_all = np.array([point[0] for point in points], dtype=float)
    longitudes_all = np.array([point[1] for point in points], dtype=float)
    margin_lat = max(0.02, float(np.ptp(latitudes_all)) * 0.25)
    margin_lon = max(0.02, float(np.ptp(longitudes_all)) * 0.25)
    lat_min, lat_max = latitudes_all.min() - margin_lat, latitudes_all.max() + margin_lat
    lon_min, lon_max = longitudes_all.min() - margin_lon, longitudes_all.max() + margin_lon

    row_start = max(0, int(np.searchsorted(global_map.latitudes, lat_min) - 1))
    row_end = min(len(global_map.latitudes), int(np.searchsorted(global_map.latitudes, lat_max) + 2))
    column_start = max(0, int(np.searchsorted(global_map.longitudes, lon_min) - 1))
    column_end = min(len(global_map.longitudes), int(np.searchsorted(global_map.longitudes, lon_max) + 2))
    axis.imshow(
        global_map.occupancy[row_start:row_end, column_start:column_end],
        origin="lower",
        cmap="Greys",
        alpha=0.7,
        extent=(
            global_map.longitudes[column_start],
            global_map.longitudes[column_end - 1],
            global_map.latitudes[row_start],
            global_map.latitudes[row_end - 1],
        ),
        aspect="auto",
    )
    latitudes = [point[0] for point in trajectory]
    longitudes = [point[1] for point in trajectory]
    axis.plot(longitudes, latitudes, color="tab:blue", linewidth=2.2, label="trajectory", zorder=4)
    for mmsi, ais_trajectory in (ais_trajectories or {}).items():
        if not ais_trajectory:
            continue
        ais_latitudes = [point[0] for point in ais_trajectory]
        ais_longitudes = [point[1] for point in ais_trajectory]
        axis.plot(
            ais_longitudes,
            ais_latitudes,
            color="tab:red",
            linewidth=2.0,
            label=f"AIS {mmsi}",
            zorder=4,
        )
        axis.scatter(
            [ais_longitudes[-1]],
            [ais_latitudes[-1]],
            color="tab:red",
            edgecolors="black",
            marker="^",
            s=70,
            label=f"AIS {mmsi} final",
            zorder=6,
        )
    axis.scatter(
        [start.longitude],
        [start.latitude],
        color="limegreen",
        marker="x",
        s=120,
        label="start",
        zorder=5,
    )
    axis.scatter(
        [goal.longitude],
        [goal.latitude],
        color="red",
        edgecolors="black",
        marker="*",
        s=180,
        label="goal",
        zorder=5,
    )
    if turnaround is not None:
        axis.scatter(
            [turnaround.longitude],
            [turnaround.latitude],
            color="darkviolet",
            edgecolors="black",
            marker="D",
            s=110,
            label="turnaround",
            zorder=6,
        )
    if trajectory:
        axis.scatter(
            [trajectory[-1][1]],
            [trajectory[-1][0]],
            color="orange",
            edgecolors="black",
            marker="o",
            s=70,
            label="final boat position",
            zorder=6,
        )
    axis.set_xlabel("Longitude [deg]")
    axis.set_ylabel("Latitude [deg]")
    axis.set_title(f"Global trajectory: {mode.upper()} ({len(trajectory)} points)")
    axis.legend(loc="best")
    axis.grid(True, alpha=0.25)

    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=170)
    return figure


def plot_start_goal_map(
    global_map: GlobalOccupancyMap,
    start: BoatState,
    goal: Waypoint,
    *,
    output: Path | str,
) -> plt.Figure:
    """Plot GEBCO occupancy cropped to the start/goal bounding box."""

    global_map._require_loaded()
    assert global_map.occupancy is not None
    assert global_map.latitudes is not None and global_map.longitudes is not None

    figure, axis = plt.subplots(figsize=(12, 10), constrained_layout=True)
    latitudes_all = np.array([start.latitude, goal.latitude], dtype=float)
    longitudes_all = np.array([start.longitude, goal.longitude], dtype=float)
    margin_lat = max(0.02, float(np.ptp(latitudes_all)) * 0.25)
    margin_lon = max(0.02, float(np.ptp(longitudes_all)) * 0.25)
    lat_min, lat_max = latitudes_all.min() - margin_lat, latitudes_all.max() + margin_lat
    lon_min, lon_max = longitudes_all.min() - margin_lon, longitudes_all.max() + margin_lon

    row_start = max(0, int(np.searchsorted(global_map.latitudes, lat_min) - 1))
    row_end = min(len(global_map.latitudes), int(np.searchsorted(global_map.latitudes, lat_max) + 2))
    column_start = max(0, int(np.searchsorted(global_map.longitudes, lon_min) - 1))
    column_end = min(len(global_map.longitudes), int(np.searchsorted(global_map.longitudes, lon_max) + 2))
    axis.imshow(
        global_map.occupancy[row_start:row_end, column_start:column_end],
        origin="lower",
        cmap="Greys",
        alpha=0.7,
        extent=(
            global_map.longitudes[column_start],
            global_map.longitudes[column_end - 1],
            global_map.latitudes[row_start],
            global_map.latitudes[row_end - 1],
        ),
        aspect="auto",
    )
    axis.plot(
        [start.longitude, goal.longitude],
        [start.latitude, goal.latitude],
        color="tab:blue",
        linestyle="--",
        linewidth=1.5,
        label="start-goal line",
        zorder=4,
    )
    axis.scatter(
        [start.longitude],
        [start.latitude],
        color="limegreen",
        marker="x",
        s=140,
        label="start",
        zorder=5,
    )
    axis.scatter(
        [goal.longitude],
        [goal.latitude],
        color="red",
        edgecolors="black",
        marker="*",
        s=190,
        label="goal",
        zorder=5,
    )
    axis.set_xlabel("Longitude [deg]")
    axis.set_ylabel("Latitude [deg]")
    axis.set_title("GEBCO start / goal check")
    axis.legend(loc="best")
    axis.grid(True, alpha=0.25)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    return figure


def _plot_global(axis, global_map, state, goal, trajectory) -> None:
    global_map._require_loaded()
    assert global_map.occupancy is not None
    assert global_map.latitudes is not None and global_map.longitudes is not None

    points = [(state.latitude, state.longitude), (goal.latitude, goal.longitude)]
    points.extend((lat, lon) for lat, lon in trajectory)
    lats = np.array([p[0] for p in points], dtype=float)
    lons = np.array([p[1] for p in points], dtype=float)
    margin_lat = max(0.02, float(np.ptp(lats)) * 0.25)
    margin_lon = max(0.02, float(np.ptp(lons)) * 0.25)
    lat_min, lat_max = lats.min() - margin_lat, lats.max() + margin_lat
    lon_min, lon_max = lons.min() - margin_lon, lons.max() + margin_lon

    r0 = max(0, int(np.searchsorted(global_map.latitudes, lat_min) - 1))
    r1 = min(len(global_map.latitudes), int(np.searchsorted(global_map.latitudes, lat_max) + 2))
    c0 = max(0, int(np.searchsorted(global_map.longitudes, lon_min) - 1))
    c1 = min(len(global_map.longitudes), int(np.searchsorted(global_map.longitudes, lon_max) + 2))
    if r1 > r0 and c1 > c0:
        axis.imshow(
            global_map.occupancy[r0:r1, c0:c1],
            origin="lower",
            cmap="Greys",
            alpha=0.45,
            extent=(
                global_map.longitudes[c0],
                global_map.longitudes[c1 - 1],
                global_map.latitudes[r0],
                global_map.latitudes[r1 - 1],
            ),
            aspect="auto",
        )
    if trajectory:
        tlat = [p[0] for p in trajectory]
        tlon = [p[1] for p in trajectory]
        axis.plot(tlon, tlat, linewidth=1.5, label="Fossen trajectory")
    axis.scatter([state.longitude], [state.latitude], marker="o", s=60, label="boat")
    axis.scatter([goal.longitude], [goal.latitude], marker="*", s=140, label="goal")
    axis.set_xlabel("Longitude [deg]")
    axis.set_ylabel("Latitude [deg]")
    axis.set_title("Global occupancy / mission context")
    axis.legend(loc="best")
    axis.grid(True, alpha=0.25)


def _plot_local(axis, local_map: LegacyLocalMap, plan: HarmonicPlan) -> None:
    blocked = np.asarray(plan.blocked, dtype=float)
    axis.imshow(blocked, origin="upper", cmap="Greys", vmin=0.0, vmax=1.0)
    if plan.path_cells:
        rows = [cell[0] for cell in plan.path_cells]
        cols = [cell[1] for cell in plan.path_cells]
        axis.plot(cols, rows, linewidth=2.0, label="legacy path")
        axis.scatter([cols[0]], [rows[0]], marker="o", s=55, label="boat")
    axis.scatter([plan.target_cell[1]], [plan.target_cell[0]], marker="*", s=130, label="local target")
    axis.set_title("Local binary map + planner path")
    axis.set_xlabel("column (east +)")
    axis.set_ylabel("row (south +)")
    axis.legend(loc="best", fontsize=8)


def _plot_potential(axis, plan: HarmonicPlan) -> None:
    image = axis.imshow(plan.potential, origin="upper", cmap="viridis")
    if plan.path_cells:
        rows = [cell[0] for cell in plan.path_cells]
        cols = [cell[1] for cell in plan.path_cells]
        axis.plot(cols, rows, linewidth=1.8)
    axis.set_title("Legacy harmonic potential")
    axis.set_xlabel("column")
    axis.set_ylabel("row")
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="potential")


def _plot_state(axis, state: BoatState, plan: HarmonicPlan, mode: str) -> None:
    axis.axis("off")
    lines = [
        f"Planner: {mode.upper()}",
        f"Desired heading: {plan.desired_heading if plan.desired_heading is not None else float('nan'):.1f} deg",
        f"Fossen heading: {state.heading:.1f} deg",
        f"Speed over ground: {state.speed:.2f} m/s",
        f"Rudder: {state.rudder:.1f} deg",
        f"Flap: {state.flap:.1f} deg",
        f"Heel: {state.heel:.1f} deg",
        f"Yaw rate: {state.yaw_rate:.4f} rad/s",
        f"Path cells: {len(plan.path_cells)}",
        f"Target on boundary: {plan.target_on_boundary}",
        f"Solver converged/path available: {plan.converged}",
    ]
    axis.text(0.03, 0.97, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=11)
    axis.set_title("Planner → MPC → Fossen state")
