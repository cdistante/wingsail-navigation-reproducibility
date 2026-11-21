"""
harmonic_gamma.py — Gamma-Harmonic Potential Field planner
===========================================================
Solves the anisotropic Laplace equation  ∇·(γ(x)∇φ) = 0
where γ is low (≈ γ_min) inside the upwind no-go cone and
γ_max = 1 everywhere else.

The low-conductance no-go zone discourages the potential gradient
from pointing upwind, so the boat naturally finds tacking-like paths
without explicit waypoints.

API is compatible with HarmonicPlanner.plan():
  plan(grid, start, goal, dist_to_obst=None)
  → list of (row, col) tuples
"""

import math
import math

import numpy as np
import torch
import torch.nn.functional as _F
from .constants import LOCAL_GRID_SIZE

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_ZONE_DANGER   = 3    # inner ring: dist < 2 → raised to 1
_ZONE_CAUTION  = 6     # outer ring: 2 ≤ dist < 5 → raised slightly
_RAISE_DANGER  = 0.95
_RAISE_CAUTION = 0.65

_NEIGHBOURS = [
    (-1,  0), (-1,  1), ( 0,  1), ( 1,  1),
    ( 1,  0), ( 1, -1), ( 0, -1), (-1, -1),
]


class GammaHarmonicPlanner:
    """
    Gamma-Harmonic planner for wind-aware navigation.
    Solves ∇·(γ(x)∇φ) = 0 where γ is low in the no-go zone.

    Parameters
    ----------
    grid_width, grid_height : float  — span of the local grid (in resolution units)
    resolution              : float  — cell size (use 1 for grid-index units)
    wind_direction_deg      : float  — direction wind comes FROM, in degrees
    no_go_angle_deg         : float  — half-angle of no-go cone (default 45°)
    gamma_min               : float  — conductance inside no-go (default 0.01)
    gamma_max               : float  — conductance outside no-go (default 1.0)
    transition_width_deg    : float  — smooth ramp at cone boundary (default 15°)
    iterations              : int    — relaxation sweeps
    relax                   : float  — SOR over-relaxation factor
    """

    def __init__(self, grid_width=LOCAL_GRID_SIZE - 1,
                 grid_height=LOCAL_GRID_SIZE - 1,
                 resolution=115.0,
                 wind_direction_deg=None,
                 no_go_angle_deg=45.0,
                 gamma_min=0.01,
                 gamma_max=1.0,
                 transition_width_deg=15.0,
                 iterations=500,
                 relax=1.0,
                 minimum_goal_progress_cosine=0.25):
        self.width             = float(grid_width)
        self.height            = float(grid_height)
        self.res               = float(resolution)
        self.nx                = int(grid_width  / resolution) + 1
        self.ny                = int(grid_height / resolution) + 1
        self.wind_dir          = wind_direction_deg
        self.no_go_angle       = no_go_angle_deg
        self.gamma_min         = gamma_min
        self.gamma_max         = gamma_max
        self.transition_width  = transition_width_deg
        self.iterations        = iterations
        self.relax             = relax
        self.minimum_goal_progress_cosine = minimum_goal_progress_cosine

        # Populated by plan() / solve_potential()
        self.potential       = None   # (nx, ny) = (cols, rows) internal convention
        self.potential_rc    = None   # (rows, cols) — for display / compatibility
        self.gamma_field     = None
        self.gamma_field_rc  = None   # (rows, cols) — for display
        self._boat_x         = None
        self._boat_y         = None

    # ------------------------------------------------------------------
    def compute_gamma_field(self, grid_center_x, grid_center_y):
        """
        Vectorised computation of γ over the full (nx, ny) grid.
        γ = gamma_min inside no-go cone, gamma_max outside,
        with a smooth linear ramp in the transition band.
        When wind_dir is None, γ = gamma_max everywhere (isotropic).
        """
        # If no wind, use full conductance everywhere
        if self.wind_dir is None:
            return np.full((self.nx, self.ny), self.gamma_max, dtype=np.float64)
        
        # Cell centres in world coordinates — shape (nx, ny)
        i_idx = np.arange(self.nx, dtype=np.float64)
        j_idx = np.arange(self.ny, dtype=np.float64)
        ii, jj = np.meshgrid(i_idx, j_idx, indexing='ij')

        cell_x = grid_center_x - self.width  / 2.0 + (ii + 0.5) * self.res
        cell_y = grid_center_y - self.height / 2.0 + (jj + 0.5) * self.res

        dx = cell_x - grid_center_x
        dy = cell_y - grid_center_y

        # Compass convention (0=north, 90=east) so it matches wind_direction_deg
        angle_to_cell = np.degrees(np.arctan2(dx, dy)) % 360.0

        angle_diff = np.abs(angle_to_cell - self.wind_dir)
        angle_diff = np.minimum(angle_diff, 360.0 - angle_diff)

        gamma = np.full((self.nx, self.ny), self.gamma_max, dtype=np.float64)

        in_nogo  = angle_diff < self.no_go_angle
        in_trans = (~in_nogo) & (angle_diff < self.no_go_angle + self.transition_width)

        gamma[in_nogo] = self.gamma_min
        t = (angle_diff - self.no_go_angle) / (self.transition_width + 1e-12)
        gamma[in_trans] = self.gamma_min + t[in_trans] * (self.gamma_max - self.gamma_min)

        return gamma

    # ------------------------------------------------------------------
    def solve_potential(self, obs_xy, goal_x, goal_y, boat_x, boat_y,
                        dist_to_obst_xy=None):
        """
        Solve ∇·(γ∇φ)=0 using vectorised Jacobi-SOR.

        Parameters
        ----------
        obs_xy          : (nx, ny) ndarray — occupancy grid, 0=free />0=obstacle
        goal_x, goal_y  : float — goal position in world coords
        boat_x, boat_y  : float — boat (grid centre) in world coords
        dist_to_obst_xy : (nx, ny) ndarray or None — distance-to-obstacle map
        """
        self._boat_x = boat_x
        self._boat_y = boat_y

        self.gamma_field = self.compute_gamma_field(boat_x, boat_y)
        # Obstacle cells must always have full conductance.
        # If an obstacle sits in the no-go direction its gamma would be 0.3,
        # weakening the harmonic-mean edge conductance and letting the potential
        # drop near that wall — the boat then sees it as safe and collides.
        self.gamma_field[obs_xy > 0.5] = self.gamma_max
        # Also restore full conductance for free cells near obstacles.
        # Without this, the wind-bias suppresses the repulsion gradient around
        # upwind obstacles — the potential field doesn't build up properly there
        # and the boat sails into them.
        if dist_to_obst_xy is not None:
            near_obst = (dist_to_obst_xy < _ZONE_CAUTION) & (obs_xy <= 0.5)
            self.gamma_field[near_obst] = self.gamma_max
        g = self.gamma_field

        # Harmonic-mean edge conductances (shape nx × ny each)
        g_e = np.empty_like(g);  g_e[:-1, :] = 2*g[:-1,:]*g[1:,:]  / (g[:-1,:]+g[1:,:]  + 1e-12);  g_e[-1,  :] = g[-1, :]
        g_w = np.empty_like(g);  g_w[1:,  :] = 2*g[1:, :]*g[:-1,:] / (g[1:, :]+g[:-1,:] + 1e-12);  g_w[0,   :] = g[0,  :]
        g_n = np.empty_like(g);  g_n[:, :-1] = 2*g[:,:-1]*g[:,1:]   / (g[:,:-1]+g[:,1:]  + 1e-12);  g_n[:,  -1] = g[:, -1]
        g_s = np.empty_like(g);  g_s[:,  1:] = 2*g[:, 1:]*g[:,:-1]  / (g[:, 1:]+g[:,:-1] + 1e-12);  g_s[:,   0] = g[:,  0]

        denom = g_e + g_w + g_n + g_s  # (nx, ny) — always > 0

        # Initial potential
        pot = np.zeros((self.nx, self.ny), dtype=np.float64)
        pot[obs_xy > 0.5] = 1.0
        pot[0,  :] = 1.0
        pot[-1, :] = 1.0
        pot[:,  0] = 1.0
        pot[:, -1] = 1.0

        # Goal cell (fixed at 0)
        ig = int(np.clip((goal_x - (boat_x - self.width  / 2.0)) / self.res, 0, self.nx - 1))
        jg = int(np.clip((goal_y - (boat_y - self.height / 2.0)) / self.res, 0, self.ny - 1))
        pot[ig, jg] = 0.0

        # Fixed-cell mask (obstacles only; goal is pinned explicitly each sweep)
        obstacle_mask = obs_xy > 0.5
        boundary_mask = np.zeros_like(obstacle_mask, dtype=bool)
        boundary_mask[0,  :] = True
        boundary_mask[-1, :] = True
        boundary_mask[:,  0] = True
        boundary_mask[:, -1] = True
        fixed = obstacle_mask | boundary_mask

        # Zone floors (precomputed, applied inside each half-sweep)
        if dist_to_obst_xy is not None:
            zone_floor = np.where(
                (dist_to_obst_xy < _ZONE_DANGER)  & (obs_xy == 0), _RAISE_DANGER,
                np.where(
                    (dist_to_obst_xy < _ZONE_CAUTION) & (obs_xy == 0), _RAISE_CAUTION,
                    0.0))
        else:
            zone_floor = None

        # Red-Black index masks for Gauss-Seidel SOR (stable for relax < 2)
        ii_idx = np.arange(self.nx)
        jj_idx = np.arange(self.ny)
        II, JJ = np.meshgrid(ii_idx, jj_idx, indexing='ij')
        rb_masks = [
            ((II + JJ) % 2 == 0) & ~fixed,
            ((II + JJ) % 2 == 1) & ~fixed,
        ]

        relax = self.relax

        # Move everything to GPU for vectorized Jacobi-SOR
        pot_t   = torch.from_numpy(pot.astype(np.float32)).to(_DEVICE)
        ge_t    = torch.from_numpy(g_e.astype(np.float32)).to(_DEVICE)
        gw_t    = torch.from_numpy(g_w.astype(np.float32)).to(_DEVICE)
        gn_t    = torch.from_numpy(g_n.astype(np.float32)).to(_DEVICE)
        gs_t    = torch.from_numpy(g_s.astype(np.float32)).to(_DEVICE)
        den_t   = torch.from_numpy((denom + 1e-12).astype(np.float32)).to(_DEVICE)
        fixed_t = torch.from_numpy(fixed).to(_DEVICE)
        zf_t    = (torch.from_numpy(zone_floor.astype(np.float32)).to(_DEVICE)
                   if zone_floor is not None else None)

        for _ in range(self.iterations):
            # Replicate-pad for Neumann-like boundary (zero-flux)
            p = _F.pad(pot_t.unsqueeze(0).unsqueeze(0),
                       (1, 1, 1, 1), mode='replicate')[0, 0]
            # p has shape (nx+2, ny+2); pot_t sits at p[1:-1, 1:-1]
            new_val = (ge_t * p[2:,   1:-1] + gw_t * p[:-2,  1:-1] +
                       gn_t * p[1:-1, 2:]   + gs_t * p[1:-1, :-2]) / den_t
            if zf_t is not None:
                new_val = torch.maximum(new_val, zf_t)
            pot_t = pot_t + relax * (new_val - pot_t)
            pot_t = pot_t.clamp(0.0, 1.0)
            pot_t[fixed_t] = 1.0
            pot_t[ig, jg]  = 0.0

        pot = pot_t.cpu().numpy().astype(np.float64)
        self.potential = pot
        return pot

    # ------------------------------------------------------------------
    def plan(self, grid, start, goal, dist_to_obst=None):
        """
        Parameters
        ----------
        grid                  : (rows, cols) ndarray — occupancy grid, 0=free
        start, goal           : (row, col) tuples
        dist_to_obst          : (rows, cols) ndarray or None
        Returns
        -------
        list of (row, col) tuples — smoothed gradient-descent path
        """
        # Internal convention: (nx, ny) = (cols, rows) = (x, y)
        # Binarize occupancy: treat any non-zero cell as obstacle to avoid
        # missing special cell values (wind-obstacle, vessels, etc.).
        obs_xy = np.asarray(grid, dtype=np.float64).T   # (cols, rows)
        obs_xy = (obs_xy != 0).astype(np.float64)

        boat_cx = self.width  / 2.0
        boat_cy = self.height / 2.0
        goal_x  = float(goal[1])   # col → x
        goal_y  = float(goal[0])   # row → y

        dist_xy = dist_to_obst.T if dist_to_obst is not None else None

        self.solve_potential(obs_xy, goal_x, goal_y, boat_cx, boat_cy,
                             dist_to_obst_xy=dist_xy)

        # Expose all fields in (rows, cols) for display
        self.potential_rc    = self.potential.T
        self.gamma_field_rc  = self.gamma_field.T  if self.gamma_field  is not None else None

        straight = self._straight_path_if_feasible(grid, start, goal)
        if straight is not None:
            return straight
        path = self._extract_path(obs_xy, grid, start, goal)
        # DISABLED: return self._smooth_path(path, grid)  # Preserves intermediate waypoints for curvature analysis
        return path

    # ------------------------------------------------------------------
    def _is_no_go_heading(self, heading_deg: float) -> bool:
        """Return whether ``heading_deg`` falls inside the upwind no-go cone."""

        if self.wind_dir is None:
            return False
        diff = abs(heading_deg - self.wind_dir)
        diff = min(diff, 360.0 - diff)
        return diff < self.no_go_angle

    def _makes_goal_progress(self, current: tuple, candidate: tuple, goal: tuple) -> bool:
        """Reject a step with a negative projection on the route direction."""

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

    def _straight_path_if_feasible(self, grid_rc: np.ndarray, start: tuple, goal: tuple):
        """Return a direct raster path when it is obstacle- and no-go-free."""

        rows, cols = grid_rc.shape
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
            if not (0 <= candidate[0] < rows and 0 <= candidate[1] < cols):
                return None
            if grid_rc[candidate] != 0:
                return None
            row_offset = candidate[0] - current[0]
            column_offset = candidate[1] - current[1]
            heading = math.degrees(math.atan2(column_offset, -row_offset)) % 360.0
            if self._is_no_go_heading(heading) or not self._makes_goal_progress(current, candidate, goal):
                return None
            path.append(candidate)
            current = candidate
        return path if current == goal else None

    # ------------------------------------------------------------------
    def _extract_path(self, obs_xy, grid_rc, start, goal):
        """Greedy steepest-descent on self.potential (nx×ny convention).
        Only blocks the immediately previous cell to allow navigation around
        obstacles in wind-distorted fields without getting trapped.  Candidates
        that enter the upwind no-go cone or make no net progress toward the
        goal are hard-excluded, matching ``HarmonicTacticalPlanner``."""
        rows, cols = grid_rc.shape
        path   = [start]
        r, c   = start
        gr, gc = goal
        prev   = None   # block only direct backtrack, not full history

        for _ in range(rows * cols):
            if r == gr and c == gc:
                break
            best_val = float('inf')
            best_nb  = None
            for dr, dc in _NEIGHBOURS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols) or obs_xy[nc, nr] != 0 or (nr, nc) == prev:
                    continue
                heading = math.degrees(math.atan2(dc, -dr)) % 360.0
                if self._is_no_go_heading(heading):
                    continue
                if not self._makes_goal_progress((r, c), (nr, nc), goal):
                    continue
                v = self.potential[nc, nr]
                if v < best_val:
                    best_val = v
                    best_nb  = (nr, nc)
            if best_nb is None:
                break
            # Stop if field is not descending (potential rising)
            if self.potential[best_nb[1], best_nb[0]] > self.potential[c, r] + 1e-4:
                break
            prev = (r, c)
            r, c = best_nb
            path.append((r, c))

        return path

    # ------------------------------------------------------------------
    def _smooth_path(self, path, grid):
        """Line-of-sight shortcutting (same as HarmonicPlanner)."""
        if len(path) < 3:
            return path
        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self._line_of_sight(path[i], path[j], grid):
                    break
                j -= 1
            smoothed.append(path[j])
            i = j
        return smoothed

    def _line_of_sight(self, p0, p1, grid):
        """Bresenham line — True if entirely obstacle-free."""
        rows, cols = grid.shape
        r0, c0 = int(round(p0[0])), int(round(p0[1]))
        r1, c1 = int(round(p1[0])), int(round(p1[1]))
        dr, dc  = abs(r1 - r0), abs(c1 - c0)
        sr, sc  = (1 if r1 > r0 else -1), (1 if c1 > c0 else -1)
        err     = dr - dc
        r, c    = r0, c0
        while True:
            if not (0 <= r < rows and 0 <= c < cols):
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

    # ------------------------------------------------------------------
    def get_direction(self, boat_x, boat_y):
        """
        Negative gradient direction at (boat_x, boat_y) in world coords.
        Returns (dir_x, dir_y) normalised to unit length.
        When wind_dir is None, returns raw gradient (no no-go projection).
        Call plan() or solve_potential() first.
        """
        if self.potential is None:
            raise ValueError("Call plan() before get_direction().")
        if self._boat_x is None:
            raise ValueError("boat position not set — call solve_potential() first.")

        ix = int((boat_x - (self._boat_x - self.width  / 2.0)) / self.res)
        iy = int((boat_y - (self._boat_y - self.height / 2.0)) / self.res)
        ix = int(np.clip(ix, 1, self.nx - 2))
        iy = int(np.clip(iy, 1, self.ny - 2))

        dphi_dx = (self.potential[ix + 1, iy] - self.potential[ix - 1, iy]) / (2.0 * self.res)
        dphi_dy = (self.potential[ix, iy + 1] - self.potential[ix, iy - 1]) / (2.0 * self.res)

        dir_x, dir_y = -dphi_dx, -dphi_dy
        n = math.hypot(dir_x, dir_y)
        if n > 1e-6:
            dir_x /= n; dir_y /= n

        # No-go zone projection (only if wind exists)
        if self.wind_dir is not None:
            heading = math.degrees(math.atan2(dir_x, dir_y)) % 360.0
            upwind = self.wind_dir
            diff = abs(heading - upwind)
            diff = min(diff, 360.0 - diff)
            if diff < self.no_go_angle:
                err = (heading - upwind + 180.0) % 360.0 - 180.0
                new_heading = upwind + (self.no_go_angle if err > 0 else -self.no_go_angle)
                new_heading_rad = math.radians(new_heading)
                dir_x = math.sin(new_heading_rad)
                dir_y = math.cos(new_heading_rad)

        return dir_x, dir_y


