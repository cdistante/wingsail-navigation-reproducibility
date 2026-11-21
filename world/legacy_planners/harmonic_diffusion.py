import math
import os
import numpy as np
import matplotlib.pyplot as plt
from .constants import OBSTACLE_CELL


class WindAwareHarmonicPlanner:
        
    

    def __init__(self, grid, goal_world, wind_direction, no_go_angle=np.pi/4,
                 outside_goal_bias=0.1, boundary='mixed',
                 minimum_goal_progress_cosine=0.25):
        self.grid = grid                  # LocalGrid instance
        self.goal_world = goal_world      # (row, col) in local grid indices
        self.wind_direction = wind_direction
        self.no_go_angle = no_go_angle
        self.outside_goal_bias = outside_goal_bias
        self.minimum_goal_progress_cosine = minimum_goal_progress_cosine
        self.potential = None
        self.goal_inside = None
        self.boundary = boundary
        self.diffusion_field = None
    def get_direction(self, pos):
            """
            Compute the steepest-descent direction at `pos` using finite differences
            on the solved potential field. Returns (delta_row, delta_col) as a unit vector.
            If wind_direction is set, the result is projected out of the no-go cone so the boat never tries to head directly upwind.
            """
            if self.potential is None:
                raise ValueError("Call solve_potential() before get_direction().")

            n = self.grid.grid_size
            pot = self.potential
            r = int(np.clip(pos[0], 1, n - 2))
            c = int(np.clip(pos[1], 1, n - 2))

            # Central finite difference: ∇φ ≈ (φ(r+1)-φ(r-1))/2, (φ(c+1)-φ(c-1))/2
            dphi_dr = (pot[r + 1, c] - pot[r - 1, c]) / 2.0
            dphi_dc = (pot[r, c + 1] - pot[r, c - 1]) / 2.0

            dr, dc = -dphi_dr, -dphi_dc
            norm = np.hypot(dr, dc)
            if norm < 1e-6:
                return 0.0, 0.0
            dr /= norm
            dc /= norm

            # No-go zone projection (same as harmonic.py)
            if self.wind_direction is not None:
                heading = np.arctan2(dc, -dr)
                upwind = self.wind_direction
                err = math.remainder(heading - upwind, math.tau)
                if abs(err) < self.no_go_angle:
                    new_heading = upwind + np.copysign(self.no_go_angle, err)
                    dr = -np.cos(new_heading)
                    dc =  np.sin(new_heading)

            return dr, dc

    def _get_diffusion_coefficient(self, heading_angle, cell=None, goal=None, relax_radius=15.0):
        """
        Returns diffusion coefficient in [MIN_COEFF, 1.0].
        Directions in the no-go zone get MIN_COEFF; near the goal the coefficient
        relaxes toward 1.0 so the goal is always reachable from any direction.
        """
        if self.wind_direction is None:
            return 1.0
        MIN_COEFF = 0.5
        delta = abs(math.remainder(heading_angle - self.wind_direction, math.tau))

        if delta < self.no_go_angle:
            base = MIN_COEFF
        elif delta < self.no_go_angle * 1.5:
            t    = (delta - self.no_go_angle) / (self.no_go_angle * 0.5)
            base = MIN_COEFF + (1.0 - MIN_COEFF) * t
        else:
            return 1.0

        if cell is not None and goal is not None:
            dist = np.hypot(cell[0] - goal[0], cell[1] - goal[1])
            if dist < relax_radius:
                goal_prox = max(0.0, 1.0 - dist / relax_radius)
                base = base + (1.0 - base) * goal_prox  # blend toward 1.0 at goal
        return base

    def _find_boundary_cell_toward_goal(self, drone_i, drone_j):
        """Grid boundary cell in the direction from drone to goal (grid indices)."""
        gi, gj = self.goal_world
        di, dj = gi - drone_i, gj - drone_j
        norm = np.hypot(di, dj)
        n = self.grid.grid_size
        if norm < 1e-6:
            return n // 2, n // 2
        di /= norm
        dj /= norm

        ts = []
        for t in [
            -drone_i / di if abs(di) > 1e-6 else np.inf,
            (n - 1 - drone_i) / di if abs(di) > 1e-6 else np.inf,
            -drone_j / dj if abs(dj) > 1e-6 else np.inf,
            (n - 1 - drone_j) / dj if abs(dj) > 1e-6 else np.inf,
        ]:
            if t > 1e-6:
                ts.append(t)

        best_t = min(ts) if ts else 0.0
        bi = int(np.clip(round(drone_i + best_t * di), 0, n - 1))
        bj = int(np.clip(round(drone_j + best_t * dj), 0, n - 1))
        return bi, bj

    def solve_potential(self, iterations=1000, relax=1.0, dist_to_obst=None):  # dist_to_obst kept for API compat
        """
        Solve ∇·(D∇φ) = 0 via red-black Gauss-Seidel SOR.
        Obstacles fixed at 1.0; goal fixed at 0.0.
        """
        occ = self.grid.occupancy_grid
        n   = self.grid.grid_size
        pot = np.zeros((n, n), dtype=np.float64)

        obstacle_mask = (occ != 0)
        free_mask     = ~obstacle_mask
        pot[obstacle_mask] = 1.0

        # Safety zone floors are intentionally disabled: hard potential floors
        # create artificial local minima that break gradient-descent path extraction.
        # Obstacle avoidance is handled naturally by pot=1 fixed at obstacle cells.
        danger_mask = caution_mask = None

        ig = int(np.clip(self.goal_world[0], 0, n - 1))
        jg = int(np.clip(self.goal_world[1], 0, n - 1))
        
        # Dirichlet/Mixed Boundary Conditions
        if self.boundary == 'dirichlet':
            pot[0, :] = pot[-1, :] = pot[:, 0] = pot[:, -1] = 1.0
        elif self.boundary == 'mixed':
            on_top, on_bottom = (ig == 0), (ig == n - 1)
            on_left, on_right = (jg == 0), (jg == n - 1)
            if not on_top:    pot[0,  :] = 1.0
            if not on_bottom: pot[-1, :] = 1.0
            if not on_left:   pot[:,  0] = 1.0
            if not on_right:  pot[:, -1] = 1.0
        
        pot[ig, jg] = 0.0  # Goal is the absolute sink

        # --- CRITICAL FIX: Synchronize Offsets and Headings ---
        # We use (dc, -dr) because row indices grow DOWN, but Y-axis grows UP.
        offsets = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
        headings = [np.arctan2(dc, -dr) for dr, dc in offsets]

        rr, cc = np.mgrid[0:n, 0:n]
        relax_radius = 15.0
        dist_goal    = np.hypot(rr - ig, cc - jg)
        goal_prox    = np.clip(1.0 - dist_goal / relax_radius, 0.0, 1.0)
        MIN_COEFF    = 0.3

        # Compute Weights
        W = []
        for heading in headings:
            if self.wind_direction is None:
                W.append(np.ones((n, n)))
            else:
                delta = abs(math.remainder(heading - self.wind_direction, math.tau))
                if delta < self.no_go_angle:
                    base = MIN_COEFF
                elif delta < self.no_go_angle * 1.5:
                    t = (delta - self.no_go_angle) / (self.no_go_angle * 0.5)
                    base = MIN_COEFF + (1.0 - MIN_COEFF) * t
                else:
                    base = 1.0
                W.append(base + (1.0 - base) * goal_prox)

            # Scalar proxy map for transformer input:
            # average directional diffusivity at each cell (lower = stronger no-go penalty).
            self.diffusion_field = np.mean(np.stack(W, axis=0), axis=0).astype(np.float32)

        # Optimization Setup
        interior = np.zeros((n, n), dtype=bool)
        interior[1:n-1, 1:n-1] = True
        upd_base = free_mask & interior
        upd_base[ig, jg] = False 
        rb = ((rr + cc) % 2 == 0)

        def _half_sweep(color_mask):
            num = np.zeros((n, n))
            den = np.zeros((n, n))
            for k, (dr, dc) in enumerate(offsets):
                # Correct slicing for neighbors
                r_in = slice(max(0, -dr), n + min(0, -dr))
                c_in = slice(max(0, -dc), n + min(0, -dc))
                r_nb = slice(max(0, dr), n + min(0, dr))
                c_nb = slice(max(0, dc), n + min(0, dc))
                
                wsl = W[k][r_in, c_in]
                num[r_in, c_in] += wsl * pot[r_nb, c_nb]
                den[r_in, c_in] += wsl
            
            avg = num / np.maximum(den, 1e-9)
            upd = upd_base & color_mask
            pot[upd] = (1.0 - relax) * pot[upd] + relax * avg[upd]
            pot[obstacle_mask] = 1.0
            pot[ig, jg] = 0.0

        for _ in range(iterations):
            _half_sweep(rb)
            _half_sweep(~rb)
            # Neumann (zero-flux) only on the edge where the goal sits;
            # all other edges stay at their Dirichlet=1.0 initialisation.
            if self.boundary == 'mixed':
                if on_top:    pot[0,  :] = pot[1,  :]
                if on_bottom: pot[-1, :] = pot[-2, :]
                if on_left:   pot[:,  0] = pot[:,  1]
                if on_right:  pot[:, -1] = pot[:, -2]
                pot[ig, jg] = 0.0

        self.potential = pot
        return pot
    def plot_potential(self, step, goal=None, save_dir="results/harmonic_plots"):
        """
        Two-panel diagnostic saved as PNG.
        Panel 1: potential field heatmap with obstacles and goal.
        Panel 2: wind diffusion coefficient (polar).
        Called every 5th step via _save_plot_async in main_harmonic.py.
        """
        if self.potential is None:
            return

        os.makedirs(save_dir, exist_ok=True)
        occ = self.grid.occupancy_grid
        n   = self.grid.grid_size
        pot = self.potential

        fig = plt.figure(figsize=(12, 5))

        # --- Panel 1: Potential Field Heatmap ---
        ax1 = fig.add_subplot(1, 2, 1)
        im = ax1.imshow(pot, origin='upper', cmap='RdYlGn_r', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax1, label='Potential (0=Goal, 1=Obstacle)')
        ax1.contour(occ, levels=[OBSTACLE_CELL - 0.5], colors='black')
        if goal is not None:
            ax1.scatter(goal[1], goal[0], color='blue', marker='*',
                        s=200, label='Goal', zorder=5)
        # TO wind velocity in grid coordinates, where rows increase southward.
        if self.wind_direction is not None:
            wd = self.wind_direction
            ax1.quiver(n * 0.1, n * 0.15, -np.sin(wd), np.cos(wd),
                       angles='xy', color='blue', scale=10, label='Wind flow (TO)')
        ax1.set_title(f"Potential Field — step {step}")
        ax1.legend(fontsize=8)

        # --- Panel 2: Wind Diffusion Coefficient (Polar) ---
        ax2 = fig.add_subplot(1, 2, 2, projection='polar')
        ax2.set_theta_zero_location('N')
        ax2.set_theta_direction(-1)
        angles = np.linspace(0, 2 * np.pi, 360)
        coeffs = [self._get_diffusion_coefficient(a) for a in angles]
        ax2.plot(angles, coeffs, color='green', linewidth=2)
        ax2.fill(angles, coeffs, color='green', alpha=0.2)
        ax2.set_title("Wind Diffusion Coefficient\n(No-Go Zone)", pad=15)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"harmonic_diffusion_step_{step:04d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _line_of_sight(self, p0, p1, grid=None):
        """Bresenham check: True iff every cell on the line p0→p1 is obstacle-free."""
        occ = self.grid.occupancy_grid if grid is None else grid
        n   = self.grid.grid_size
        r0, c0 = int(round(p0[0])), int(round(p0[1]))
        r1, c1 = int(round(p1[0])), int(round(p1[1]))
        dr = abs(r1 - r0); dc = abs(c1 - c0)
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        err = dr - dc
        r, c = r0, c0
        while True:
            if not (0 <= r < n and 0 <= c < n):
                return False
            if occ[r, c] != 0:
                return False
            if r == r1 and c == c1:
                return True
            e2 = 2 * err
            if e2 > -dc:
                err -= dc; r += sr
            if e2 < dr:
                err += dr; c += sc
    def _is_no_go_heading(self, heading_rad: float) -> bool:
        """Return whether ``heading_rad`` falls inside the upwind no-go cone."""

        if self.wind_direction is None:
            return False
        diff = abs(math.remainder(heading_rad - self.wind_direction, math.tau))
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

    def _straight_path_if_feasible(self, start: tuple, goal: tuple):
        """Return a direct raster path when it is obstacle- and no-go-free."""

        occ = self.grid.occupancy_grid
        rows, cols = occ.shape
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
            if occ[candidate] != 0:
                return None
            row_offset = candidate[0] - current[0]
            column_offset = candidate[1] - current[1]
            heading = math.atan2(column_offset, -row_offset)
            if self._is_no_go_heading(heading) or not self._makes_goal_progress(current, candidate, goal):
                return None
            path.append(candidate)
            current = candidate
        return path if current == goal else None
    def _smooth_path(self, path, grid):
        """Line-of-sight shortcutting to remove unnecessary zigzag steps."""
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

    def extract_diffusion_path(self, start_local, max_steps=1500, epsilon=0.05,
                               previous=None):
        """
        Gradient descent on the solved potential field.
        8-connected. Blocks only the immediate previous cell to prevent
        oscillation without trapping the path in dead-ends.  Candidates that
        enter the upwind no-go cone or make no net progress toward the goal
        are hard-excluded, matching ``HarmonicTacticalPlanner``.
        """
        occ = self.grid.occupancy_grid
        pot = self.potential
        n   = self.grid.grid_size
        ig  = int(np.clip(self.goal_world[0], 0, n - 1))
        jg  = int(np.clip(self.goal_world[1], 0, n - 1))

        if previous is None:
            straight = self._straight_path_if_feasible(start_local, (ig, jg))
            if straight is not None:
                return straight

        path = [start_local]
        ix, iy = start_local
        # Preserve anti-backtracking state when a controller replans at every
        # step. A standalone full-path call can leave previous as None.
        prev   = previous

        for _ in range(max_steps):
            if not (0 <= ix < n and 0 <= iy < n):
                break

            candidates = [
                (ix - 1, iy), (ix + 1, iy),
                (ix, iy - 1), (ix, iy + 1),
                (ix - 1, iy - 1), (ix - 1, iy + 1),
                (ix + 1, iy - 1), (ix + 1, iy + 1),
            ]

            def _sailable(cx, cy, allow_prev):
                if not (0 <= cx < n and 0 <= cy < n) or occ[cx, cy] != 0:
                    return False
                if not allow_prev and (cx, cy) == prev:
                    return False
                heading = math.atan2(cy - iy, -(cx - ix))
                if self._is_no_go_heading(heading):
                    return False
                return self._makes_goal_progress((ix, iy), (cx, cy), (ig, jg))

            valid = [(cx, cy) for cx, cy in candidates if _sailable(cx, cy, allow_prev=False)]
            if not valid:
                # all sailable non-obstacle neighbours are the prev cell — allow it
                valid = [(cx, cy) for cx, cy in candidates if _sailable(cx, cy, allow_prev=True)]
            if not valid:
                break

            best = min(valid, key=lambda c: pot[c[0], c[1]])

            # stop only if strictly climbing past the plateau tolerance
            if pot[best[0], best[1]] > pot[ix, iy] + epsilon:
                break

            prev   = (ix, iy)
            ix, iy = best
            # After line 356, before appending:
            if len(path) > 100:  # Only after path gets long
                pot_drop = pot[ix, iy] - pot[best[0], best[1]]
                if pot_drop < epsilon / 2:
                    print(f"Stall at ({ix},{iy}): pot={pot[ix,iy]:.6f}, "
                        f"neighbors={[pot[c] for c in valid]}, "
                        f"drop={pot_drop:.6f}")
            path.append((ix, iy))

            if (ix, iy) == (ig, jg):
                break

        # DISABLED: return self._smooth_path(path, self.grid.occupancy_grid)  # Preserves intermediate waypoints for curvature analysis
        return path


# ---------------------------------------------------------------------------
# Module-level convenience function  (matches other planners' API style)
# ---------------------------------------------------------------------------
def harmonic_diffusion(grid, start, goal, wind_direction=None,
                       boundary='mixed', **kwargs):
    """Convenience wrapper — instantiates and calls WindAwareHarmonicPlanner."""
    planner = WindAwareHarmonicPlanner(
        grid,
        goal,
        wind_direction,
        boundary=boundary,
        **kwargs
    )
    planner.solve_potential()
    # Use extract_diffusion_path to get the path (API match to harmonic_plan)
    path = planner.extract_diffusion_path(start)
    return path
