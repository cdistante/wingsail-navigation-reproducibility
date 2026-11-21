"""Boat-centred local map adapter for the imported legacy planners.

The Maritime World 10-channel ``LocalGrid`` remains the environmental tensor
contract used by MPC/ViT.  This module adds the binary static obstacle plane
that the legacy harmonic planners expect, without changing those 10 channels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation

from .global_map import GlobalOccupancyMap
from .occupancy import LocalGrid, OccupancyGrid
from .scenario import Waypoint
from .world import WorldObservation


@dataclass(frozen=True)
class LegacyLocalMap:
    """Integrated local representation used by legacy planners and current MPC."""

    tensor: LocalGrid
    static_blocked: np.ndarray

    @property
    def cells(self) -> int:
        return self.tensor.cells


@dataclass
class LegacyLocalMapBuilder:
    """Combine current environmental channels with a legacy-style binary map."""

    global_map: GlobalOccupancyMap
    span_metres: float = 4_715.0
    resolution_metres: float = 115.0
    obstacle_buffer_metres: float = 115.0

    def __post_init__(self) -> None:
        self.tensor_builder = OccupancyGrid(
            span=float(self.span_metres),
            resolution=float(self.resolution_metres),
        )

    def build(self, observation: WorldObservation, goal: Waypoint) -> LegacyLocalMap:
        """Build the 10-channel tensor plus inflated static land/shallow mask."""

        tensor = self.tensor_builder.build(
            observation.boat,
            observation,
            goal=goal,
            restricted=bool(observation.chart and observation.chart.is_restricted),
        )
        blocked = self.global_map.local_obstacle_mask(
            observation.boat.latitude,
            observation.boat.longitude,
            cells=tensor.cells,
            resolution_metres=tensor.resolution,
        )
        radius_cells = int(np.ceil(max(0.0, self.obstacle_buffer_metres) / tensor.resolution))
        if radius_cells > 0:
            yy, xx = np.ogrid[-radius_cells : radius_cells + 1, -radius_cells : radius_cells + 1]
            structure = xx * xx + yy * yy <= radius_cells * radius_cells
            blocked = binary_dilation(blocked, structure=structure)
        return LegacyLocalMap(tensor=tensor, static_blocked=np.asarray(blocked, dtype=bool))
