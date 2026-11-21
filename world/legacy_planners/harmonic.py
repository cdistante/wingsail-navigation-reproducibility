"""
harmonic.py  —  Harmonic Potential Field local path planner
============================================================
Solves the Laplace equation (∇²φ = 0) on the local occupancy grid
using Gauss-Seidel relaxation.  Two boundary modes are supported:

  DIRICHLET (boundary='dirichlet')  — fixed values at edges
    φ = 0   at the goal cell          (attractor)
    φ = 1   at every obstacle cell    (repulsor)
    φ = 1   at grid boundaries        (no-pass walls — goal is unique sink)

  NEUMANN   (boundary='neumann')    — zero-flux / zero normal-derivative
    φ = 0   at the goal cell          (Dirichlet — still fixed)
    φ = 1   at every obstacle cell    (Dirichlet — still fixed)
    ∂φ/∂n = 0 at grid edges          (field reflects smoothly through edge)
    Implemented by copying the first interior row/col to the boundary
    after every Gauss-Seidel sweep.

  MIXED     (boundary='mixed')      — Neumann on the goal edge, Dirichlet=1
    on the three remaining edges.  Best of both: open water toward the
    goal, walls on the sides.

Because the solution is a harmonic function it has NO local minima
in the free space — the negative gradient always leads to the goal.

Region-Based Zones (mirroring qlearning.py):
  dist_to_obst < 2  →  DANGER  (cells raised toward 1 before solving)
  dist_to_obst < 3  →  CAUTION (cells raised slightly)

The planner returns a list of (row, col) tuples by following the
steepest descent of the potential from start to goal.

API matches AStarPlanner / QLearningPlanner / AcoAgent so it plugs
straight into main_3.py without changes to calling code.
"""

import math
import numpy as np
import torch
import torch.nn.functional as _F

from .constants import LOCAL_GRID_SIZE, OBSTACLE_CELL

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Region-based zone constants (same thresholds as qlearning.py)
# ---------------------------------------------------------------------------
_ZONE_DANGER   = 2     # inner ring: dist_to_obst < 2  → raised toward 1
_ZONE_CAUTION  = 5     # outer ring: dist_to_obst < 5 (and ≥ 2) → raised slightly
_RAISE_DANGER  = 0.85    # potential floor for danger zone
_RAISE_CAUTION = 0.55     # potential floor for caution zone

# 8-connected neighbourhood for path extraction
_NEIGHBOURS = [
    (-1,  0), (-1,  1), ( 0,  1), ( 1,  1),
    ( 1,  0), ( 1, -1), ( 0, -1), (-1, -1),
]


# ---------------------------------------------------------------------------
class VirtualObstacleNoGoZone:
    """
    Virtual obstacle method for no-go zone avoidance.

    The no-go zone is represented as a cone extending upwind from the boat.
    The resulting field is added to the harmonic potential before solving so
    the planner naturally prefers headings outside the cone.
    """

    def __init__(
        self,
        wind_direction_deg,
        no_go_angle_deg=45.0,
        cone_length_ratio=2.0,
        repulsion_strength=0.5,
    ):
        self.wind_dir = wind_direction_deg
        self.no_go_angle = no_go_angle_deg
        self.cone_length_ratio = cone_length_ratio
        self.repulsion_strength = repulsion_strength

    def update_wind(self, wind_direction_deg):
        """Update wind direction (for example, from a sensor reading)."""
        self.wind_dir = wind_direction_deg

    def get_virtual_obstacle_potential(self, boat_x, boat_y, boat_speed, grid_res, grid_shape):
        """
        Return a 2-D potential field that penalizes cells inside the no-go cone.

        The output shape matches the local harmonic grid and is intended to be
        added to the initial potential before relaxation.
        """
        nx, ny = grid_shape
        potential = np.zeros((nx, ny), dtype=np.float64)

        if self.wind_dir is None:
            return potential

        cone_length = max(boat_speed * self.cone_length_ratio, grid_res)

        rows = np.arange(nx, dtype=np.float64)[:, None]
        cols = np.arange(ny, dtype=np.float64)[None, :]

        # Compass convention used everywhere else in the planner:
        # 0° = north, 90° = east.  Array rows increase downward, so north is
        # the negative row direction and east is the positive column direction.
        north = (boat_x - rows) * grid_res
        east  = (cols - boat_y) * grid_res

        dist = np.hypot(north, east)
        # Angle of each cell relative to the boat, measured clockwise from north.
        angle_to_point = (np.degrees(np.arctan2(east, north)) % 360.0)
        angle_diff = np.abs((angle_to_point - self.wind_dir + 180.0) % 360.0 - 180.0)

        in_cone = (angle_diff < self.no_go_angle) & (dist < cone_length) & (dist > 0.1)
        if not np.any(in_cone):
            return potential

        angle_factor = np.clip(1.0 - (angle_diff / (self.no_go_angle + 1e-6)) ** 2, 0.0, 1.0)
        distance_factor = np.clip(1.0 - (dist / (cone_length + 1e-6)) ** 2, 0.0, 1.0)
        potential[in_cone] = self.repulsion_strength * angle_factor[in_cone] * distance_factor[in_cone]
        return potential


# ---------------------------------------------------------------------------
class HarmonicPlanner:
    """
    Harmonic Potential Field planner for a 2-D occupancy grid.

    Parameters
    ----------
    grid_size  : int   — side length of the square local grid (default 41)
    iterations : int   — relaxation iterations (more → more accurate)
    relax      : float — weighted-Jacobi factor in (0, 1].
                         Use <= 1 for stability with the vectorised update.
    """

    def __init__(
        self,
        grid_size: int   = LOCAL_GRID_SIZE,
        iterations: int  = None,
        relax: float     = None,
        boundary: str    = 'mixed',
        no_go_angle: float = math.radians(45.0),
        minimum_goal_progress_cosine: float = 0.25,
    ):
        self.grid_size       = grid_size
        # The solver below is Jacobi-style (simultaneous interior updates),
        # so relaxation must stay <= 1 to avoid checkerboard odd/even oscillations.
        default_relax = 1.0
        self.relax           = relax if relax is not None else default_relax
        self.iterations      = iterations if iterations is not None else max(1000, 25 * grid_size)
        self.boundary        = boundary
        self.no_go_angle     = no_go_angle
        self.minimum_goal_progress_cosine = minimum_goal_progress_cosine
        self.potential       = None
        self.wind_direction  = None
        self._wind_direction = None
        self.wind_bias       = None
        self.virtual_bias    = None

    def world_to_grid(self, x: float, y: float):
        """Identity mapping: world coords are already cell indices (row, col)."""
        return int(round(x)), int(round(y))

    # -----------------------------------------------------------------------

    def _get_diffusion_coefficient(self, heading_rad: float,
                                   wind_direction: float,
                                   cell: tuple = None,
                                   goal: tuple = None,
                                   relax_radius: float = 7.0) -> float:
        """
        Return an anisotropic diffusion weight in [0.01, 1.0] for a cell
        movement in compass direction `heading_rad` given wind coming FROM
        `wind_direction` (both in compass radians: 0=N, π/2=E).

        If cell and goal are provided, relax the upwind penalty within a radius of the goal.
        """
        # Distance-based relaxation near the goal
        relax_factor = 1.0
        if cell is not None and goal is not None:
            dist = math.hypot(cell[0] - goal[0], cell[1] - goal[1])
            if dist < relax_radius:
                # Linear relaxation: penalty drops off steadily near the goal
                t = dist / relax_radius
                relax_factor = 0.2 + 0.8 * t  # Never relax above 0.2 in no-go zone

        diff = abs(math.remainder(heading_rad - wind_direction, math.tau))
        if diff < self.no_go_angle:
            return 0.01 * relax_factor
        elif diff < self.no_go_angle * 1.5:
            t = (diff - self.no_go_angle) / (self.no_go_angle * 0.5)
            return (0.01 + 0.99 * t) * relax_factor
        return 1.0

    # -----------------------------------------------------------------------
    def plan(
        self,
        grid: np.ndarray,
        start: tuple,
        goal: tuple,
        dist_to_obst: np.ndarray = None,
        wind_direction: float    = None,   # radians — biases potential away from no-go cone
        use_virtual_obstacle: bool = False,
        **kwargs,
    ):
        """
        Returns a list of (row, col) tuples from start → goal,
        or an empty list if planning fails.
        """
        gs = self.grid_size

        # ---- Build initial potential field --------------------------------
        pot = np.zeros((gs, gs), dtype=np.float64)

        # All non-free cells (obstacles + wind obstacles) → 1
        pot[grid != 0] = 1.0

        # Region-based safety zones: only applied to free cells (grid == 0)
        if dist_to_obst is not None:
            danger_mask  = (dist_to_obst < _ZONE_DANGER)  & (grid == 0)
            caution_mask = (dist_to_obst < _ZONE_CAUTION) & (grid == 0) & ~danger_mask
            pot[danger_mask]  = np.maximum(pot[danger_mask],  _RAISE_DANGER)
            pot[caution_mask] = np.maximum(pot[caution_mask], _RAISE_CAUTION)

        # Goal → 0 (must override any zone raise)
        gr, gc = int(np.clip(goal[0], 0, gs - 1)), int(np.clip(goal[1], 0, gs - 1))
        pot[gr, gc] = 0.0

        # ---- Initial boundary setup (Dirichlet modes) --------------------
        # Neumann edges are handled dynamically inside the loop.
        # For Dirichlet / Mixed we pin walls to 1 now.
        if self.boundary == 'dirichlet':
            # All four edges = 1 (walls).  Goal re-pinned to 0 after.
            pot[0,  :] = 1.0
            pot[-1, :] = 1.0
            pot[:,  0] = 1.0
            pot[:, -1] = 1.0
            pot[gr, gc] = 0.0
        elif self.boundary == 'mixed':
            # Determine which edge the goal sits on and leave that edge free
            # (Neumann); the other three get Dirichlet=1.
            on_top    = (gr == 0)
            on_bottom = (gr == gs - 1)
            on_left   = (gc == 0)
            on_right  = (gc == gs - 1)
            if not on_top:    pot[0,  :] = 1.0
            if not on_bottom: pot[-1, :] = 1.0
            if not on_left:   pot[:,  0] = 1.0
            if not on_right:  pot[:, -1] = 1.0
            pot[gr, gc] = 0.0
        # 'neumann': no fixed boundary values — updated each iteration below

        # ---- Wind handling -------------------------------------------------
        # Default behaviour: use wind only as a virtual obstacle bias.
        # Set use_wind_awareness=True to re-enable anisotropic direction
        # weighting / heading projection.
        use_wind_awareness = bool(kwargs.get("use_wind_awareness", False))
        self.wind_direction = wind_direction if use_wind_awareness else None
        self._wind_direction = self.wind_direction

        boat_speed = float(kwargs.get("boat_speed", 1.0))
        wind_bias = None
        if use_wind_awareness and wind_direction is not None:
            wind_layer = VirtualObstacleNoGoZone(
                wind_direction_deg=math.degrees(wind_direction),
                no_go_angle_deg=math.degrees(self.no_go_angle),
                repulsion_strength=float(kwargs.get("wind_repulsion_strength", 0.25)),
            )
            wind_bias = wind_layer.get_virtual_obstacle_potential(
                start[0],
                start[1],
                boat_speed,
                1.0,
                (gs, gs),
            )

        virtual_bias = None
        if use_virtual_obstacle and wind_direction is not None:
            virtual_obstacle = VirtualObstacleNoGoZone(
                wind_direction_deg=math.degrees(wind_direction),
                no_go_angle_deg=math.degrees(self.no_go_angle),
                repulsion_strength=float(kwargs.get("virtual_repulsion_strength", 0.5)),
            )
            virtual_bias = virtual_obstacle.get_virtual_obstacle_potential(
                start[0],
                start[1],
                boat_speed,
                1.0,
                (gs, gs),
            )

        # ---- GPU-vectorized Jacobi/SOR solver ----------------------------
        obstacle_mask = (grid != 0)

        # Precompute per-cell direction weights once (outside iteration loop).
        # Each weight map is (gs, gs): scalar base * per-cell relax factor near goal.
        _rows = np.arange(gs, dtype=np.float32)[:, None]
        _cols = np.arange(gs, dtype=np.float32)[None, :]
        _dg   = np.hypot(_rows - gr, _cols - gc)
        _near = (np.abs(_rows - gr) <= 1) & (np.abs(_cols - gc) <= 1)
        _rf   = np.clip(0.2 + 0.8 * _dg / 7.0, 0.2, 1.0)
        _rf[_near] = 1.0

        def _wmap(h):
            if (not use_wind_awareness) or (wind_direction is None):
                return np.ones((gs, gs), np.float32)
            d = abs(math.remainder(h - wind_direction, math.tau))
            if d < self.no_go_angle:
                return (0.01 * _rf).astype(np.float32)
            if d < self.no_go_angle * 1.5:
                t = (d - self.no_go_angle) / (self.no_go_angle * 0.5)
                return ((0.01 + 0.99 * t) * _rf).astype(np.float32)
            return np.ones((gs, gs), np.float32)

        wn_np = _wmap(0.0)            # north  (row - 1)
        ws_np = _wmap(math.pi)        # south  (row + 1)
        we_np = _wmap(math.pi / 2)    # east   (col + 1)
        ww_np = _wmap(-math.pi / 2)   # west   (col - 1)

        pot_t = torch.from_numpy(pot.astype(np.float32)).to(_DEVICE)
        self.wind_bias = wind_bias
        self.virtual_bias = virtual_bias
        if wind_bias is not None:
            pot_t = torch.clamp(pot_t + torch.from_numpy(wind_bias.astype(np.float32)).to(_DEVICE), 0.0, 1.0)
        if virtual_bias is not None:
            pot_t = torch.clamp(pot_t + torch.from_numpy(virtual_bias.astype(np.float32)).to(_DEVICE), 0.0, 1.0)
        obs_t = torch.from_numpy(obstacle_mask).to(_DEVICE)
        _sor  = float(np.clip(self.relax, 1e-6, 1.0))

        if self.boundary == 'neumann':
            wn_t = torch.from_numpy(wn_np).to(_DEVICE)
            ws_t = torch.from_numpy(ws_np).to(_DEVICE)
            we_t = torch.from_numpy(we_np).to(_DEVICE)
            ww_t = torch.from_numpy(ww_np).to(_DEVICE)
            tw_t = (wn_t + ws_t + we_t + ww_t).clamp(min=1e-9)
            if dist_to_obst is not None:
                dan_t = torch.from_numpy(danger_mask).to(_DEVICE)
                cau_t = torch.from_numpy(caution_mask).to(_DEVICE)
            for _ in range(self.iterations):
                # replicate-pad implements zero-flux (Neumann) boundary
                p = _F.pad(pot_t.unsqueeze(0).unsqueeze(0),
                           (1, 1, 1, 1), mode='replicate')[0, 0]
                avg = (wn_t * p[:-2, 1:-1] + ws_t * p[2:, 1:-1] +
                       we_t * p[1:-1, 2:] + ww_t * p[1:-1, :-2]) / tw_t
                new_pot = _sor * avg + (1.0 - _sor) * pot_t
                if dist_to_obst is not None:
                    new_pot = torch.where(dan_t, new_pot.clamp(min=_RAISE_DANGER),  new_pot)
                    new_pot = torch.where(cau_t, new_pot.clamp(min=_RAISE_CAUTION), new_pot)
                pot_t = new_pot
                pot_t[obs_t] = 1.0
                pot_t[gr, gc] = 0.0
        else:
            # Inner-cell weights only; boundary rows/cols are fixed.
            wn_t = torch.from_numpy(wn_np[1:-1, 1:-1]).to(_DEVICE)
            ws_t = torch.from_numpy(ws_np[1:-1, 1:-1]).to(_DEVICE)
            we_t = torch.from_numpy(we_np[1:-1, 1:-1]).to(_DEVICE)
            ww_t = torch.from_numpy(ww_np[1:-1, 1:-1]).to(_DEVICE)
            tw_t = (wn_t + ws_t + we_t + ww_t).clamp(min=1e-9)
            if dist_to_obst is not None:
                dan_t = torch.from_numpy(danger_mask[1:-1, 1:-1]).to(_DEVICE)
                cau_t = torch.from_numpy(caution_mask[1:-1, 1:-1]).to(_DEVICE)
            for _ in range(self.iterations):
                avg = (wn_t * pot_t[:-2, 1:-1] + ws_t * pot_t[2:, 1:-1] +
                       we_t * pot_t[1:-1, 2:] + ww_t * pot_t[1:-1, :-2]) / tw_t
                new_inn = _sor * avg + (1.0 - _sor) * pot_t[1:-1, 1:-1]
                if dist_to_obst is not None:
                    new_inn = torch.where(dan_t, new_inn.clamp(min=_RAISE_DANGER),  new_inn)
                    new_inn = torch.where(cau_t, new_inn.clamp(min=_RAISE_CAUTION), new_inn)
                pot_t[1:-1, 1:-1] = new_inn
                pot_t[obs_t] = 1.0
                pot_t[gr, gc] = 0.0
                if self.boundary == 'mixed':
                    if on_top:    pot_t[0,  :] = pot_t[1,  :]
                    if on_bottom: pot_t[-1, :] = pot_t[-2, :]
                    if on_left:   pot_t[:,  0] = pot_t[:,  1]
                    if on_right:  pot_t[:, -1] = pot_t[:, -2]
                    pot_t[gr, gc] = 0.0

        pot = pot_t.cpu().numpy().astype(np.float64)
        self.potential = pot
        straight_path = self._straight_path_if_feasible(grid, start, (gr, gc))
        if straight_path is not None:
            return straight_path
        return self._extract_path(pot, grid, start, goal)

    # -----------------------------------------------------------------------
    def _is_no_go_heading(self, heading_rad: float) -> bool:
        """Return whether ``heading_rad`` falls inside the upwind no-go cone."""

        if self.wind_direction is None:
            return False
        diff = abs(math.remainder(heading_rad - self.wind_direction, math.tau))
        return diff < self.no_go_angle

    def _makes_goal_progress(self, current: tuple, candidate: tuple, goal: tuple) -> bool:
        """Reject a step with a negative projection on the route direction.

        Mirrors ``HarmonicTacticalPlanner._makes_goal_progress`` so a strongly
        anisotropic field cannot descend into a backwards/U-turn command.
        """

        goal_east = float(goal[1] - current[1])
        goal_north = float(current[0] - goal[0])
        goal_norm = math.hypot(goal_east, goal_north)
        if goal_norm < 1e-12:
            return True
        move_east = float(candidate[1] - current[1])
        move_north = float(current[0] - candidate[0])
        move_norm = math.hypot(move_east, move_north)
        if move_norm < 1e-12:
            return False
        cosine = (move_east * goal_east + move_north * goal_north) / (move_norm * goal_norm)
        return cosine >= self.minimum_goal_progress_cosine

    def _straight_path_if_feasible(self, grid: np.ndarray, start: tuple, goal: tuple):
        """Return a direct raster path when it is obstacle- and no-go-free.

        Skips the harmonic descent for a simple beam/downwind course, mirroring
        ``HarmonicTacticalPlanner._straight_path_if_feasible``.
        """

        row_delta = goal[0] - start[0]
        column_delta = goal[1] - start[1]
        steps = max(abs(row_delta), abs(column_delta))
        if steps == 0:
            return [start]
        path = [start]
        current = start
        for step in range(1, steps + 1):
            candidate = (
                int(round(start[0] + row_delta * step / steps)),
                int(round(start[1] + column_delta * step / steps)),
            )
            if candidate == current:
                continue
            if not (0 <= candidate[0] < grid.shape[0] and 0 <= candidate[1] < grid.shape[1]):
                return None
            if grid[candidate] != 0:
                return None
            row_offset = candidate[0] - current[0]
            column_offset = candidate[1] - current[1]
            heading = math.atan2(column_offset, -row_offset)
            if self._is_no_go_heading(heading) or not self._makes_goal_progress(current, candidate, goal):
                return None
            path.append(candidate)
            current = candidate
        return path if current == goal else None

    def _extract_path(
        self, pot: np.ndarray, grid: np.ndarray, start: tuple, goal: tuple
    ):
        """
        Greedy steepest-descent on the potential field.
        At each step move to the 8-connected neighbour with the lowest φ.
        Only the immediately previous cell is excluded (mirrors DIFFUSION),
        so the path can navigate around obstacles without getting trapped
        by a full visited set in wind-distorted fields.  Candidates that
        enter the upwind no-go cone or make no net progress toward the goal
        are hard-excluded, matching ``HarmonicTacticalPlanner``.
        """
        gs   = self.grid_size
        path = [start]
        r, c = start
        prev = None   # block only direct backtrack, not full history

        for _ in range(gs * gs):      # upper bound prevents infinite loops
            if (r, c) == (goal[0], goal[1]):
                break

            best_val = float('inf')
            best_nb  = None

            for dr, dc in _NEIGHBOURS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < gs and 0 <= nc < gs) or grid[nr, nc] != 0 or (nr, nc) == prev:
                    continue
                heading = math.atan2(dc, -dr)
                if self._is_no_go_heading(heading):
                    continue
                if not self._makes_goal_progress((r, c), (nr, nc), goal):
                    continue
                if pot[nr, nc] < best_val:
                    best_val = pot[nr, nc]
                    best_nb  = (nr, nc)

            if best_nb is None:
                break   # trapped — no candidate is free, sailable, and goal-progressing

            # Stop if every reachable cell is higher potential (field not descending)
            if pot[best_nb[0], best_nb[1]] > pot[r, c] + 1e-4:
                break

            prev = (r, c)
            r, c = best_nb
            path.append((r, c))

        # DISABLED: return self._smooth_path(path, grid)  # Preserves intermediate waypoints for curvature analysis
        return path

    # -----------------------------------------------------------------------
    def _smooth_path(self, path: list, grid: np.ndarray) -> list:
        """
        Line-of-sight shortcutting: walk the path and skip intermediate
        waypoints whenever a straight line between two points is fully
        obstacle-free.  Removes the zigzag from greedy 8-connected descent.
        """
        if len(path) < 3:
            return path

        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            # Try to reach as far ahead as possible in a straight line
            j = len(path) - 1
            while j > i + 1:
                if self._line_of_sight(path[i], path[j], grid):
                    break
                j -= 1
            smoothed.append(path[j])
            i = j
        return smoothed

    def _line_of_sight(self, p0: tuple, p1: tuple, grid: np.ndarray) -> bool:
        """
        Bresenham line check: returns True if every cell on the straight line
        from p0 to p1 is obstacle-free.
        """
        r0, c0 = int(round(p0[0])), int(round(p0[1]))
        r1, c1 = int(round(p1[0])), int(round(p1[1]))
        dr, dc  = abs(r1 - r0), abs(c1 - c0)
        sr, sc  = (1 if r1 > r0 else -1), (1 if c1 > c0 else -1)
        err     = dr - dc
        r, c    = r0, c0
        gs      = self.grid_size
        while True:
            if not (0 <= r < gs and 0 <= c < gs):
                return False
            if grid[r, c] != 0:
                return False
            if r == r1 and c == c1:
                return True
            e2 = 2 * err
            if e2 > -dc:
                err -= dc; r += sr
            if e2 <  dr:
                err += dr; c += sc

    # -----------------------------------------------------------------------
    def get_direction(self, pos: tuple) -> tuple:
        """
        Compute the steepest-descent direction at `pos` using finite differences
        on the solved potential field.  Returns (delta_row, delta_col) as a unit
        vector.  If wind_direction was passed to plan(), the result is projected
        out of the no-go cone so the boat never tries to head directly upwind.

        Call plan() first so self.potential is populated.
        Returns (0, 0) if gradient is flat (already at goal).
        """
        if self.potential is None:
            raise ValueError("Call plan() before get_direction().")

        gs  = self.grid_size
        pot = self.potential
        r   = int(np.clip(pos[0], 1, gs - 2))
        c   = int(np.clip(pos[1], 1, gs - 2))

        # Central finite difference: ∇φ ≈ (φ(r+1)-φ(r-1))/2, (φ(c+1)-φ(c-1))/2
        dphi_dr = (pot[r + 1, c] - pot[r - 1, c]) / 2.0
        dphi_dc = (pot[r, c + 1] - pot[r, c - 1]) / 2.0

        dr, dc = -dphi_dr, -dphi_dc
        norm = math.hypot(dr, dc)
        if norm < 1e-6:
            return 0.0, 0.0
        dr /= norm
        dc /= norm

        # ---- No-go zone projection ----------------------------------------
        # Convert (dr, dc) in row-col space to compass heading, then check
        # whether the heading falls inside the upwind no-go cone.
        # Compass convention: north = 0, east = π/2.
        # row-1 = north → atan2(dc=0, -dr=1) = 0 ✓
        # col+1 = east  → atan2(dc=1, -dr=0) = π/2 ✓
        if self._wind_direction is not None:
            heading = math.atan2(dc, -dr)
            err = math.remainder(heading - self._wind_direction, math.tau)
            if abs(err) < self.no_go_angle:
                # Project to the nearest edge of the no-go cone.
                # Determine which side (port / starboard) is closer.
                new_heading = self._wind_direction + math.copysign(self.no_go_angle, err)
                dr = -math.cos(new_heading)
                dc =  math.sin(new_heading)

        return dr, dc


# ---------------------------------------------------------------------------
class WindAwareWaypointPlanner:
    """
    Decomposes an upwind goal into tacking waypoints (global coords).
    Use HarmonicPlanner to navigate between each waypoint locally.
    """

    def __init__(self, wind_direction: float,
                 no_go_angle: float  = math.pi / 4,
                 tack_angle: float   = math.radians(50.0),
                 tack_angle_stbd: float  = math.radians(50.0),
                 tack_angle_port: float  = math.radians(50.0)):
        self.wind_direction  = wind_direction
        self.no_go_angle     = no_go_angle
        # Starboard tack: wind on right → boat heads LEFT of wind source (wind_direction - angle)
        # Port tack:      wind on left  → boat heads RIGHT of wind source (wind_direction + angle)
        self.tack_angle_stbd = tack_angle_stbd
        self.tack_angle_port = tack_angle_port
        self.tack_angle      = tack_angle  # kept for external callers

    def is_upwind(self, heading: float) -> bool:
        # wind_direction = direction wind comes FROM; heading toward it = no-go zone
        # If no wind, nothing is upwind
        if self.wind_direction is None:
            return False
        diff = abs(math.remainder(heading - self.wind_direction, math.tau))
        return diff < self.no_go_angle

    def generate_tacking_waypoints(self, start_pos: tuple, goal_pos: tuple,
                                   step_size: float = 30.0,
                                   map_width: float = None,
                                   map_height: float = None,
                                   global_grid: np.ndarray = None) -> list:
        """
        Return a list of (x, y) global waypoints from start → goal.
        If goal is reachable directly, returns [goal_pos].
        Otherwise zig-zags on port/starboard tacks until close enough.

        Coordinate convention (matches main_harmonic.py):
          heading = atan2(x_delta, y_delta)  →  0=north, π/2=east
          x component of movement = sin(heading) * step
          y component of movement = cos(heading) * step

        global_grid : (rows, cols) ndarray or None.  When provided, each
          candidate waypoint is checked against the grid and snapped to the
          nearest free cell if it lands inside an obstacle.
        """
        dx = goal_pos[0] - start_pos[0]
        dy = goal_pos[1] - start_pos[1]
        if math.hypot(dx, dy) < 1e-3:
            return [goal_pos]

        # Compass convention: atan2(x_delta, y_delta) — 0=north, π/2=east
        goal_heading = math.atan2(dx, dy)
        if not self.is_upwind(goal_heading):
            return [goal_pos]

        # Starboard tack: head LEFT of wind source (wind on right side of boat)
        # Port tack:      head RIGHT of wind source (wind on left side of boat)
        tack_stbd = self.wind_direction - self.tack_angle_stbd
        tack_port = self.wind_direction + self.tack_angle_port

        # Start on the tack whose heading is closer to the goal direction
        dl   = abs((goal_heading - tack_port + math.pi) % (2 * math.pi) - math.pi)
        dr_a = abs((goal_heading - tack_stbd + math.pi) % (2 * math.pi) - math.pi)
        on_port = dl < dr_a

        rows = global_grid.shape[0] if global_grid is not None else None
        cols = global_grid.shape[1] if global_grid is not None else None

        def _snap_to_free(cx, cy):
            """Return the nearest free cell to (cx, cy) in a spiral search."""
            if global_grid is None:
                return cx, cy
            for r in range(15):
                for dcol in range(-r, r + 1):
                    for drow in range(-r, r + 1):
                        if abs(dcol) != r and abs(drow) != r:
                            continue
                        col = int(round(cx)) + dcol
                        row = int(round(cy)) + drow
                        if 0 <= row < rows and 0 <= col < cols:
                            if global_grid[row, col] == 0:
                                return float(col), float(row)
            return cx, cy   # fallback: no free cell found nearby

        # Along/cross-track unit vectors for the start→goal axis.
        total_dist = math.hypot(dx, dy)
        ax = dx / total_dist   # along-track unit vector (x)
        ay = dy / total_dist   # along-track unit vector (y)
        px = -ay               # cross-track unit vector (x, 90° left of along)
        py =  ax               # cross-track unit vector (y)

        # Maximum allowed lateral deviation: 30 % of total distance, capped at 45 cells.
        # Keeps waypoints inside the start→goal corridor and away from flanking land.
        max_cross = min(total_dist * 0.30, 45.0)

        waypoints = []
        pos = list(start_pos)

        for _ in range(60):
            rem = math.hypot(goal_pos[0] - pos[0], goal_pos[1] - pos[1])
            if rem < step_size:
                break
            heading = tack_port if on_port else tack_stbd
            # Compass heading decomposition: east=sin, north=cos
            nx = pos[0] + math.sin(heading) * step_size
            ny = pos[1] + math.cos(heading) * step_size

            # ── Corridor enforcement ──────────────────────────────────────
            # Decompose candidate point into along-track and cross-track components
            # relative to the start→goal line, then clamp both.
            rel_x = nx - start_pos[0]
            rel_y = ny - start_pos[1]
            along = ax * rel_x + ay * rel_y          # signed progress toward goal
            cross = px * rel_x + py * rel_y          # signed lateral offset

            # Never go behind start (along < 0) or past goal (along > total_dist)
            along = max(0.0, min(total_dist, along))
            # Keep inside the lateral corridor
            cross = max(-max_cross, min(max_cross, cross))

            nx = start_pos[0] + along * ax + cross * px
            ny = start_pos[1] + along * ay + cross * py
            # ─────────────────────────────────────────────────────────────

            # Clamp to global map bounds
            if map_width  is not None: nx = max(0.0, min(map_width  - 1, nx))
            if map_height is not None: ny = max(0.0, min(map_height - 1, ny))
            # Snap out of obstacle
            if global_grid is not None:
                col, row = int(round(nx)), int(round(ny))
                if 0 <= row < rows and 0 <= col < cols and global_grid[row, col] != 0:
                    nx, ny = _snap_to_free(nx, ny)
            pos = [nx, ny]
            waypoints.append((pos[0], pos[1]))
            on_port = not on_port   # switch tack

        waypoints.append(goal_pos)
        return waypoints


# ---------------------------------------------------------------------------
# Module-level convenience function  (matches other planners' API style)
# ---------------------------------------------------------------------------
def harmonic_plan(grid, start, goal, dist_to_obst=None, wind_direction=None,
            boundary='mixed', use_virtual_obstacle=True, **kwargs):
    """Convenience wrapper — instantiates and calls HarmonicPlanner.plan()."""
    planner = HarmonicPlanner(boundary=boundary, **kwargs)
    return planner.plan(grid, start, goal,
                        dist_to_obst=dist_to_obst,
                wind_direction=wind_direction,
                use_virtual_obstacle=use_virtual_obstacle)
