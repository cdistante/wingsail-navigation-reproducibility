"""Wind-aware harmonic tactical planning over a boat-centred occupancy grid.

The planner is intentionally an intermediate layer: it turns a local grid and
a waypoint into a short safe track reference.  It does not command rudder or
flap, and it is therefore suitable as a transparent reference/fallback for a
future JEPA/MPC controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, radians, sin

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import bicgstab

from config import HARMONIC_TACTICAL_PLANNER

from ._geo import local_east_north_metres
from .ais import AISTarget
from .occupancy import LocalGrid
from .scenario import Waypoint
from .world import WorldObservation


@dataclass(frozen=True)
class HarmonicPlan:
    """Result of one local harmonic-planner solve.

    ``path_cells`` uses the same row/column convention as :class:`LocalGrid`.
    ``desired_heading`` is true degrees, or ``None`` when no feasible local
    movement reference exists. For a clear geographic approach,
    ``navigation_target`` retains the steering endpoint at sub-cell precision.
    """

    potential: np.ndarray
    blocked: np.ndarray
    path_cells: tuple[tuple[int, int], ...]
    desired_heading: float | None
    target_cell: tuple[int, int]
    target_on_boundary: bool
    converged: bool
    iterations: int
    navigation_target: Waypoint | None = None
    desired_bow_heading: float | None = None
    forecast_closest_metres: float | None = None
    forecast_tack_in_minutes: float | None = None
    drift_mode: str | None = None
    steady_speed_mps: float | None = None
    gate_vmg_mps: float | None = None
    drift_range_m: float | None = None
    steady_rudder_degrees: float | None = None
    steady_flap_degrees: float | None = None

    @property
    def has_path(self) -> bool:
        """Whether the plan contains a non-zero local movement reference."""

        return self.desired_heading is not None and (
            self.navigation_target is not None or len(self.path_cells) > 1
        )


@dataclass(frozen=True)
class HarmonicTacticalPlanner:
    """Generate a wind-feasible, collision-aware local track reference.

    Obstacles are fixed at potential one and the active local goal is a zero
    potential sink.  The solver applies lower directional diffusion into the
    upwind no-go cone; path extraction then treats that cone as hard heading
    infeasibility, so a direct upwind gradient is never issued to the MPC.
    """

    no_go_angle_degrees: float = float(HARMONIC_TACTICAL_PLANNER["no_go_angle_degrees"])
    minimum_diffusion: float = float(HARMONIC_TACTICAL_PLANNER["minimum_diffusion"])
    transition_band_multiplier: float = float(
        HARMONIC_TACTICAL_PLANNER["transition_band_multiplier"]
    )
    minimum_goal_progress_cosine: float = float(
        HARMONIC_TACTICAL_PLANNER["minimum_goal_progress_cosine"]
    )
    maximum_iterations: int = int(HARMONIC_TACTICAL_PLANNER["maximum_iterations"])
    convergence_tolerance: float = float(
        HARMONIC_TACTICAL_PLANNER["convergence_tolerance"]
    )
    maximum_path_steps: int = int(HARMONIC_TACTICAL_PLANNER["maximum_path_steps"])
    ais_prediction_horizon_seconds: float = float(
        HARMONIC_TACTICAL_PLANNER["ais_prediction_horizon_seconds"]
    )
    ais_prediction_interval_seconds: float = float(
        HARMONIC_TACTICAL_PLANNER["ais_prediction_interval_seconds"]
    )
    ais_safety_radius_metres: float = float(
        HARMONIC_TACTICAL_PLANNER["ais_safety_radius_metres"]
    )
    minimum_depth_metres: float = float(HARMONIC_TACTICAL_PLANNER["minimum_depth_metres"])

    _OFFSETS = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    )
    # The harmonic stencil uses cardinal neighbours.  Diagonal cells remain
    # available during path extraction so the tactical reference can select
    # port/starboard tacks.
    _SOLVER_OFFSETS = _OFFSETS[:4]

    def __post_init__(self) -> None:
        if not 0.0 < self.no_go_angle_degrees < 90.0:
            raise ValueError("no_go_angle_degrees must be in (0, 90).")
        if not 0.0 < self.minimum_diffusion <= 1.0:
            raise ValueError("minimum_diffusion must be in (0, 1].")
        if self.transition_band_multiplier < 1.0:
            raise ValueError("transition_band_multiplier must be at least one.")
        if not 0.0 <= self.minimum_goal_progress_cosine < 1.0:
            raise ValueError("minimum_goal_progress_cosine must be in [0, 1).")
        if self.maximum_iterations < 1 or self.maximum_path_steps < 1:
            raise ValueError("maximum iteration and path-step counts must be positive.")
        if self.convergence_tolerance <= 0.0:
            raise ValueError("convergence_tolerance must be positive.")
        if self.ais_prediction_horizon_seconds < 0.0:
            raise ValueError("ais_prediction_horizon_seconds must be non-negative.")
        if self.ais_prediction_interval_seconds <= 0.0:
            raise ValueError("ais_prediction_interval_seconds must be positive.")
        if self.ais_safety_radius_metres <= 0.0 or self.minimum_depth_metres < 0.0:
            raise ValueError("planner safety distances must be non-negative and non-zero.")

    def plan(
        self,
        grid: LocalGrid,
        observation: WorldObservation,
        goal: Waypoint,
        *,
        extra_blocked: np.ndarray | None = None,
    ) -> HarmonicPlan:
        """Solve a local plan from the boat-centred grid towards ``goal``.

        A destination outside the local grid is converted into a boundary exit
        target in its geographic direction.  This gives the MPC a short
        horizon reference rather than incorrectly clipping the global waypoint
        to an arbitrary grid corner.
        """

        if grid.cells < 3:
            raise ValueError("Harmonic planning requires at least a 3x3 local grid.")
        start_cell = self._boat_cell(grid)
        # A swept AIS corridor can cross the boat's *future* position and
        # therefore overlap this current cell.  Keep the departure cell free;
        # the remainder of that corridor remains blocked to force an evasive
        # route.  A present AIS/restricted overlap is still a true invalid
        # starting condition.
        if (
            grid.channel("ais_occupied")[start_cell] > 0.0
            or grid.channel("restricted")[start_cell] > 0.0
        ):
            raise RuntimeError("The saildrone is inside a local planner obstacle.")
        blocked = self.obstacle_mask(grid, observation, extra_blocked=extra_blocked)
        blocked[start_cell] = False
        target_cell, target_on_boundary = self._goal_cell(grid, goal)
        target_cell = (
            self._nearest_free_boundary_cell(blocked, target_cell)
            if target_on_boundary
            else self._nearest_free_cell(blocked, target_cell)
        )

        wind_from = None if observation.wind is None else observation.wind.direction
        potential, converged, iterations = self._solve(blocked, target_cell, wind_from)
        # In open water, do not turn a simple beam/downwind course into an
        # unnecessary harmonic detour.  As soon as a projected AIS corridor
        # no longer intersects the route, this restores direct waypoint
        # tracking.  The ordinary harmonic descent remains the avoidance
        # solution whenever that direct line is blocked or lies in the no-go
        # cone.
        path_cells = self._straight_path_if_feasible(
            blocked,
            start_cell,
            target_cell,
            wind_from,
        )
        if path_cells is None:
            path_cells = self._extract_path(
                potential,
                blocked,
                start_cell,
                target_cell,
                wind_from,
            )
        desired_heading = self._path_heading(path_cells)
        return HarmonicPlan(
            potential=potential,
            blocked=blocked,
            path_cells=path_cells,
            desired_heading=desired_heading,
            target_cell=target_cell,
            target_on_boundary=target_on_boundary,
            converged=converged,
            iterations=iterations,
        )

    def obstacle_mask(
        self,
        grid: LocalGrid,
        observation: WorldObservation,
        *,
        extra_blocked: np.ndarray | None = None,
        include_ais: bool = True,
    ) -> np.ndarray:
        """Build a binary obstacle map from named grid channels and AIS tracks."""

        blocked = grid.channel("restricted") > 0.0
        if include_ais:
            blocked |= grid.channel("ais_occupied") > 0.0
        if observation.seabed is not None:
            blocked |= grid.channel("depth") < self.minimum_depth_metres
        if extra_blocked is not None:
            if extra_blocked.shape != blocked.shape:
                raise ValueError("extra_blocked must have the local grid's row/column shape.")
            blocked |= np.asarray(extra_blocked, dtype=bool)
        if include_ais:
            for target in observation.ais_targets:
                self._paint_predicted_ais(blocked, grid, target)
        return blocked

    def _paint_predicted_ais(
        self,
        blocked: np.ndarray,
        grid: LocalGrid,
        target: AISTarget,
    ) -> None:
        """Paint the target's constant-COG/SOG swept safety corridor."""

        east, north = local_east_north_metres(
            target.latitude,
            target.longitude,
            grid.origin_latitude,
            grid.origin_longitude,
        )
        speed = 0.0 if target.speed is None else max(0.0, float(target.speed))
        course = 0.0 if target.course is None else float(target.course)
        course_radians = radians(course)
        east_rate = speed * sin(course_radians)
        north_rate = speed * cos(course_radians)
        vessel_radius = 0.0
        if target.length_metres is not None and target.beam_metres is not None:
            vessel_radius = 0.5 * hypot(target.length_metres, target.beam_metres)
        radius_cells = max(
            1,
            int(np.ceil((self.ais_safety_radius_metres + vessel_radius) / grid.resolution)),
        )
        sample_times = np.arange(
            0.0,
            self.ais_prediction_horizon_seconds + self.ais_prediction_interval_seconds,
            self.ais_prediction_interval_seconds,
        )
        for elapsed in sample_times:
            row, column = self._east_north_to_cell(
                grid,
                float(east + east_rate * elapsed),
                float(north + north_rate * elapsed),
            )
            self._paint_disc(blocked, row, column, radius_cells)

    def _solve(
        self,
        blocked: np.ndarray,
        target_cell: tuple[int, int],
        wind_from_degrees: float | None,
    ) -> tuple[np.ndarray, bool, int]:
        """Solve a directionally weighted discrete harmonic potential field."""

        potential = np.full(blocked.shape, 0.5, dtype=np.float64)
        potential[blocked] = 1.0
        fixed = blocked.copy()
        fixed[0, :] = fixed[-1, :] = fixed[:, 0] = fixed[:, -1] = True
        target_row, target_column = target_cell
        fixed[target_row, target_column] = True
        potential[fixed] = 1.0
        potential[target_row, target_column] = 0.0

        weights = np.asarray(
            [
                self._diffusion_weight(self._heading_from_offset(offset), wind_from_degrees)
                for offset in self._SOLVER_OFFSETS
            ],
            dtype=float,
        )
        denominator = float(weights.sum())

        # Build A phi = b only for non-fixed cells. Fixed obstacle/boundary
        # values are moved to b. Directional weights make A non-symmetric, so
        # BiCGSTAB is used rather than conjugate gradient.
        free_rows, free_columns = np.nonzero(~fixed)
        unknown_count = len(free_rows)
        if unknown_count == 0:
            return potential, True, 0
        node_index = np.full(blocked.shape, -1, dtype=int)
        node_index[free_rows, free_columns] = np.arange(unknown_count)
        matrix_rows: list[int] = []
        matrix_columns: list[int] = []
        matrix_values: list[float] = []
        right_hand_side = np.zeros(unknown_count, dtype=float)

        for node, (row, column) in enumerate(zip(free_rows, free_columns)):
            matrix_rows.append(node)
            matrix_columns.append(node)
            matrix_values.append(denominator)
            for weight, (row_offset, column_offset) in zip(weights, self._SOLVER_OFFSETS):
                neighbour_row, neighbour_column = row + row_offset, column + column_offset
                neighbour_node = node_index[neighbour_row, neighbour_column]
                if neighbour_node >= 0:
                    matrix_rows.append(node)
                    matrix_columns.append(int(neighbour_node))
                    matrix_values.append(-float(weight))
                else:
                    right_hand_side[node] += float(weight) * potential[
                        neighbour_row, neighbour_column
                    ]

        matrix = coo_matrix(
            (matrix_values, (matrix_rows, matrix_columns)),
            shape=(unknown_count, unknown_count),
        ).tocsr()
        iterations = 0

        def count_iteration(_: np.ndarray) -> None:
            nonlocal iterations
            iterations += 1

        solution, status = bicgstab(
            matrix,
            right_hand_side,
            x0=np.full(unknown_count, 0.5, dtype=float),
            rtol=self.convergence_tolerance,
            atol=0.0,
            maxiter=self.maximum_iterations,
            callback=count_iteration,
        )
        potential[free_rows, free_columns] = solution
        potential[fixed] = 1.0
        potential[target_row, target_column] = 0.0
        return potential, status == 0, iterations if iterations else max(0, status)

    def _extract_path(
        self,
        potential: np.ndarray,
        blocked: np.ndarray,
        start_cell: tuple[int, int],
        target_cell: tuple[int, int],
        wind_from_degrees: float | None,
    ) -> tuple[tuple[int, int], ...]:
        """Extract a no-go-feasible descending path from the solved field."""

        path = [start_cell]
        visited = {start_cell}
        current = start_cell
        for _ in range(self.maximum_path_steps):
            if current == target_cell:
                break
            candidates: list[tuple[int, int]] = []
            for row_offset, column_offset in self._OFFSETS:
                candidate = current[0] + row_offset, current[1] + column_offset
                if not self._cell_is_free(blocked, candidate) or candidate in visited:
                    continue
                heading = self._heading_from_offset((row_offset, column_offset))
                if self._is_no_go(heading, wind_from_degrees):
                    continue
                if not self._makes_goal_progress(
                    current,
                    candidate,
                    target_cell,
                ):
                    continue
                candidates.append(candidate)
            if not candidates:
                break
            next_cell = min(candidates, key=lambda cell: potential[cell])
            # The no-go and forward-progress gates can make a short,
            # initially non-descending tack the only sailable choice.  The
            # potential still ranks alternatives, while ``visited`` and the
            # positive route projection prevent cycles or a route U-turn.
            path.append(next_cell)
            visited.add(next_cell)
            current = next_cell
        return tuple(path)

    def _straight_path_if_feasible(
        self,
        blocked: np.ndarray,
        start_cell: tuple[int, int],
        target_cell: tuple[int, int],
        wind_from_degrees: float | None,
    ) -> tuple[tuple[int, int], ...] | None:
        """Return a collision-free direct raster path, or ``None`` if blocked.

        The path is sampled at one cell per step, so it uses exactly the same
        obstacle, wind no-go, and forward-progress tests as an extracted
        harmonic avoidance path.
        """

        row_delta = target_cell[0] - start_cell[0]
        column_delta = target_cell[1] - start_cell[1]
        steps = max(abs(row_delta), abs(column_delta))
        if steps == 0:
            return (start_cell,)
        path = [start_cell]
        current = start_cell
        for step in range(1, steps + 1):
            candidate = (
                int(round(start_cell[0] + row_delta * step / steps)),
                int(round(start_cell[1] + column_delta * step / steps)),
            )
            if candidate == current:
                continue
            if not self._cell_is_free(blocked, candidate):
                return None
            offset = candidate[0] - current[0], candidate[1] - current[1]
            heading = self._heading_from_offset(offset)
            if self._is_no_go(heading, wind_from_degrees) or not self._makes_goal_progress(
                current,
                candidate,
                target_cell,
            ):
                return None
            path.append(candidate)
            current = candidate
        return tuple(path) if current == target_cell else None

    def _goal_cell(self, grid: LocalGrid, goal: Waypoint) -> tuple[tuple[int, int], bool]:
        east, north = local_east_north_metres(
            goal.latitude,
            goal.longitude,
            grid.origin_latitude,
            grid.origin_longitude,
        )
        row_float, column_float = self._east_north_to_cell(grid, float(east), float(north))
        maximum = grid.cells - 1
        inside = 0.0 <= row_float <= maximum and 0.0 <= column_float <= maximum
        if inside:
            return (int(round(row_float)), int(round(column_float))), False

        centre = (grid.cells - 1) / 2.0
        row_delta, column_delta = row_float - centre, column_float - centre
        scale = min(
            centre / abs(row_delta) if abs(row_delta) > 1e-12 else np.inf,
            centre / abs(column_delta) if abs(column_delta) > 1e-12 else np.inf,
        )
        row = int(np.clip(round(centre + scale * row_delta), 0, maximum))
        column = int(np.clip(round(centre + scale * column_delta), 0, maximum))
        return (row, column), True

    @staticmethod
    def _boat_cell(grid: LocalGrid) -> tuple[int, int]:
        centre = int(round((grid.cells - 1) / 2.0))
        return centre, centre

    @staticmethod
    def _east_north_to_cell(
        grid: LocalGrid,
        east_metres: float,
        north_metres: float,
    ) -> tuple[float, float]:
        centre = (grid.cells - 1) / 2.0
        return (
            centre - north_metres / grid.resolution,
            centre + east_metres / grid.resolution,
        )

    @staticmethod
    def _paint_disc(mask: np.ndarray, row: float, column: float, radius: int) -> None:
        cells = mask.shape[0]
        row_centre, column_centre = int(round(row)), int(round(column))
        row_start, row_end = max(0, row_centre - radius), min(cells, row_centre + radius + 1)
        column_start, column_end = max(0, column_centre - radius), min(cells, column_centre + radius + 1)
        if row_start >= row_end or column_start >= column_end:
            return
        rows, columns = np.ogrid[row_start:row_end, column_start:column_end]
        mask[row_start:row_end, column_start:column_end] |= (
            (rows - row_centre) ** 2 + (columns - column_centre) ** 2 <= radius**2
        )

    @staticmethod
    def _cell_is_free(blocked: np.ndarray, cell: tuple[int, int]) -> bool:
        row, column = cell
        return 0 <= row < blocked.shape[0] and 0 <= column < blocked.shape[1] and not blocked[cell]

    @staticmethod
    def _nearest_free_cell(blocked: np.ndarray, cell: tuple[int, int]) -> tuple[int, int]:
        if not blocked[cell]:
            return cell
        rows, columns = np.nonzero(~blocked)
        if not rows.size:
            raise RuntimeError("The local planner has no free cells.")
        index = int(np.argmin((rows - cell[0]) ** 2 + (columns - cell[1]) ** 2))
        return int(rows[index]), int(columns[index])

    @staticmethod
    def _nearest_free_boundary_cell(
        blocked: np.ndarray,
        cell: tuple[int, int],
    ) -> tuple[int, int]:
        """Keep a distant waypoint's temporary exit target on the grid edge.

        Selecting an arbitrary free *interior* cell when an AIS corridor
        occupies the desired edge can rotate an eastbound mission into a
        northbound local target.  The closest free boundary cell preserves the
        outward mission direction while leaving the harmonic field to select a
        safe approach to that exit.
        """

        rows, columns = np.nonzero(~blocked)
        if not rows.size:
            raise RuntimeError("The local planner has no free cells.")
        last = blocked.shape[0] - 1
        boundary = (rows == 0) | (rows == last) | (columns == 0) | (columns == last)
        if not np.any(boundary):
            return HarmonicTacticalPlanner._nearest_free_cell(blocked, cell)
        boundary_rows, boundary_columns = rows[boundary], columns[boundary]
        index = int(
            np.argmin(
                (boundary_rows - cell[0]) ** 2 + (boundary_columns - cell[1]) ** 2
            )
        )
        return int(boundary_rows[index]), int(boundary_columns[index])

    def _diffusion_weight(self, heading: float, wind_from_degrees: float | None) -> float:
        if wind_from_degrees is None:
            return 1.0
        difference = self._angular_difference_degrees(heading, wind_from_degrees)
        if difference < self.no_go_angle_degrees:
            return self.minimum_diffusion
        transition_limit = self.no_go_angle_degrees * self.transition_band_multiplier
        if difference >= transition_limit:
            return 1.0
        blend = (difference - self.no_go_angle_degrees) / (
            transition_limit - self.no_go_angle_degrees
        )
        return self.minimum_diffusion + (1.0 - self.minimum_diffusion) * blend

    def _is_no_go(self, heading: float, wind_from_degrees: float | None) -> bool:
        return (
            wind_from_degrees is not None
            and self._angular_difference_degrees(heading, wind_from_degrees)
            < self.no_go_angle_degrees
        )

    def _makes_goal_progress(
        self,
        current: tuple[int, int],
        candidate: tuple[int, int],
        target: tuple[int, int],
    ) -> bool:
        """Keep each extracted segment non-negative along the route direction.

        The harmonic potential handles obstacles, but a strongly anisotropic
        field can otherwise make a lower-potential neighbour lie behind the
        boat.  The goal-direction projection allows a lateral avoidance leg
        while rejecting the backwards/U-turn command visible to the operator.
        """

        goal_east = float(target[1] - current[1])
        goal_north = float(current[0] - target[0])
        goal_norm = hypot(goal_east, goal_north)
        if goal_norm < 1e-12:
            return True
        move_east = float(candidate[1] - current[1])
        move_north = float(current[0] - candidate[0])
        move_norm = hypot(move_east, move_north)
        if move_norm < 1e-12:
            return False
        cosine = (move_east * goal_east + move_north * goal_north) / (
            move_norm * goal_norm
        )
        return cosine >= self.minimum_goal_progress_cosine

    @staticmethod
    def _heading_from_offset(offset: tuple[int, int]) -> float:
        row_offset, column_offset = offset
        return degrees(atan2(column_offset, -row_offset)) % 360.0

    @staticmethod
    def _angular_difference_degrees(first: float, second: float) -> float:
        return abs((first - second + 180.0) % 360.0 - 180.0)

    def _path_heading(self, path_cells: tuple[tuple[int, int], ...]) -> float | None:
        if len(path_cells) < 2:
            return None
        row_offset = path_cells[1][0] - path_cells[0][0]
        column_offset = path_cells[1][1] - path_cells[0][1]
        return self._heading_from_offset((row_offset, column_offset))
