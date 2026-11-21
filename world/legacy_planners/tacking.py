"""Gamma-specific waypoint generation.

This module keeps gamma's tacking geometry separate from the shared
wind-aware waypoint planner so the gamma mode can use a stricter no-go
clearance without changing the standard path logic.
"""

import math

import numpy as np

from .harmonic import WindAwareWaypointPlanner


class GammaTackingPlanner(WindAwareWaypointPlanner):
    """Waypoint planner that keeps gamma tacks safely outside the no-go cone."""

    def __init__(self, wind_direction: float,
                 no_go_angle: float = math.pi / 4,
                 tack_angle: float = math.radians(65.0),
                 tack_angle_stbd: float = math.radians(65.0),
                 tack_angle_port: float = math.radians(65.0),
                 safety_margin: float = math.radians(5.0)):
        super().__init__(wind_direction=wind_direction,
                         no_go_angle=no_go_angle,
                         tack_angle=tack_angle,
                         tack_angle_stbd=tack_angle_stbd,
                         tack_angle_port=tack_angle_port)
        self.safety_margin = safety_margin

    def _safe_tack_angles(self):
        min_safe = self.no_go_angle + self.safety_margin
        tack_stbd = max(self.tack_angle_stbd, min_safe)
        tack_port = max(self.tack_angle_port, min_safe)
        return (self.wind_direction - tack_stbd, self.wind_direction + tack_port)

    def generate_tacking_waypoints(self, start_pos: tuple, goal_pos: tuple,
                                   step_size: float = 30.0,
                                   map_width: float = None,
                                   map_height: float = None,
                                   global_grid: np.ndarray = None) -> list:
        """Generate gamma waypoints with headings kept outside the no-go cone."""
        dx = goal_pos[0] - start_pos[0]
        dy = goal_pos[1] - start_pos[1]
        if math.hypot(dx, dy) < 1e-3:
            return [goal_pos]

        goal_heading = math.atan2(dx, dy)
        if not self.is_upwind(goal_heading):
            return [goal_pos]

        tack_stbd, tack_port = self._safe_tack_angles()
        dl = abs((goal_heading - tack_port + math.pi) % (2 * math.pi) - math.pi)
        dr = abs((goal_heading - tack_stbd + math.pi) % (2 * math.pi) - math.pi)
        on_port = dl < dr

        rows = global_grid.shape[0] if global_grid is not None else None
        cols = global_grid.shape[1] if global_grid is not None else None

        def _snap_to_free(cx, cy):
            if global_grid is None:
                return cx, cy
            for radius in range(15):
                for dcol in range(-radius, radius + 1):
                    for drow in range(-radius, radius + 1):
                        if abs(dcol) != radius and abs(drow) != radius:
                            continue
                        col = int(round(cx)) + dcol
                        row = int(round(cy)) + drow
                        if 0 <= row < rows and 0 <= col < cols and global_grid[row, col] == 0:
                            return float(col), float(row)
            return cx, cy

        total_dist = math.hypot(dx, dy)
        ax = dx / total_dist
        ay = dy / total_dist
        px = -ay
        py = ax

        # Gamma uses a tighter corridor so tacks do not skim the no-go wedge.
        max_cross = min(total_dist * 0.25, 40.0)

        waypoints = []
        pos = list(start_pos)

        for _ in range(60):
            rem = math.hypot(goal_pos[0] - pos[0], goal_pos[1] - pos[1])
            if rem < step_size:
                break

            heading = tack_port if on_port else tack_stbd
            nx = pos[0] + math.sin(heading) * step_size
            ny = pos[1] + math.cos(heading) * step_size

            rel_x = nx - start_pos[0]
            rel_y = ny - start_pos[1]
            along = ax * rel_x + ay * rel_y
            cross = px * rel_x + py * rel_y

            along = max(0.0, min(total_dist, along))
            cross = max(-max_cross, min(max_cross, cross))

            nx = start_pos[0] + along * ax + cross * px
            ny = start_pos[1] + along * ay + cross * py

            if map_width is not None:
                nx = max(0.0, min(map_width - 1, nx))
            if map_height is not None:
                ny = max(0.0, min(map_height - 1, ny))

            if global_grid is not None:
                col, row = int(round(nx)), int(round(ny))
                if 0 <= row < rows and 0 <= col < cols and global_grid[row, col] != 0:
                    nx, ny = _snap_to_free(nx, ny)

            # Re-check the tack heading after snapping; keep gamma outside the cone.
            if self.is_upwind(math.atan2(nx - pos[0], ny - pos[1])):
                nx = pos[0] + math.sin(heading) * step_size
                ny = pos[1] + math.cos(heading) * step_size

            pos = [nx, ny]
            waypoints.append((pos[0], pos[1]))
            on_port = not on_port

        waypoints.append(goal_pos)
        return waypoints