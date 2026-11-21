"""Local multi-channel occupancy grids for JEPA and MPC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, TYPE_CHECKING

import numpy as np

from ._geo import local_east_north_metres

if TYPE_CHECKING:
    from .ais import AISTarget
    from .boat import BoatState
    from .scenario import Waypoint
    from .world import WorldObservation


@dataclass(frozen=True)
class LocalGrid:
    """A boat-centred East/North local map with named tensor channels."""

    data: np.ndarray
    channels: tuple[str, ...]
    resolution: float
    span: float
    origin_latitude: float
    origin_longitude: float

    @property
    def cells(self) -> int:
        """Return the square grid side length in cells."""

        return int(self.data.shape[1])

    def channel(self, name: str) -> np.ndarray:
        """Return one named grid plane."""

        try:
            return self.data[self.channels.index(name)]
        except ValueError as error:
            raise KeyError(f"Unknown occupancy channel: {name}.") from error


class OccupancyGrid:
    """Build a local, boat-centred tensor suitable for a JEPA encoder.

    Coordinates use a fixed East/North frame: positive ``x`` is east and
    positive ``y`` is north.  Heading stays in the boat-state vector, avoiding
    an implicit map rotation that would make global planning less transparent.
    Environmental values are spatially constant over this local grid because
    the current data have kilometre-scale resolution; AIS and goal channels
    retain their exact local positions.
    """

    CHANNELS = (
        "ais_occupied",
        "depth",
        "coast_distance",
        "wind_u",
        "wind_v",
        "wave_height",
        "current_u",
        "current_v",
        "goal",
        "restricted",
    )

    def __init__(self, span: float = 400.0, resolution: float = 1.0) -> None:
        if span <= 0.0 or resolution <= 0.0:
            raise ValueError("span and resolution must be positive.")
        cells = int(round(span / resolution))
        if cells < 2:
            raise ValueError("span/resolution must create at least a 2x2 grid.")
        self.span = float(cells * resolution)
        self.resolution = float(resolution)
        self.cells = cells

    def build(
        self,
        boat: "BoatState",
        observation: "WorldObservation",
        *,
        goal: "Waypoint | None" = None,
        ais_targets: Iterable["AISTarget"] | None = None,
        restricted: bool = False,
    ) -> LocalGrid:
        """Build a tensor from an already sampled :class:`WorldObservation`."""

        grid = np.zeros((len(self.CHANNELS), self.cells, self.cells), dtype=np.float32)
        index = {name: position for position, name in enumerate(self.CHANNELS)}

        if observation.seabed is not None:
            grid[index["depth"]].fill(observation.seabed.depth)
        if observation.coast is not None:
            grid[index["coast_distance"]].fill(observation.coast.distance)
        if observation.wind is not None:
            grid[index["wind_u"]].fill(observation.wind.u)
            grid[index["wind_v"]].fill(observation.wind.v)
        if observation.wave is not None:
            grid[index["wave_height"]].fill(observation.wave.height)
        if observation.current is not None:
            grid[index["current_u"]].fill(observation.current.u)
            grid[index["current_v"]].fill(observation.current.v)
        if restricted:
            grid[index["restricted"]].fill(1.0)

        for target in ais_targets if ais_targets is not None else observation.ais_targets:
            east, north = local_east_north_metres(
                target.latitude,
                target.longitude,
                boat.latitude,
                boat.longitude,
            )
            self._paint_disc(grid[index["ais_occupied"]], float(east), float(north), 10.0)

        if goal is not None:
            east, north = local_east_north_metres(
                goal.latitude,
                goal.longitude,
                boat.latitude,
                boat.longitude,
            )
            self._paint_disc(grid[index["goal"]], float(east), float(north), 5.0)

        return LocalGrid(
            data=grid,
            channels=self.CHANNELS,
            resolution=self.resolution,
            span=self.span,
            origin_latitude=boat.latitude,
            origin_longitude=boat.longitude,
        )

    def _paint_disc(self, plane: np.ndarray, east: float, north: float, radius: float) -> None:
        centre = (self.cells - 1) / 2.0
        column = int(round(centre + east / self.resolution))
        row = int(round(centre - north / self.resolution))
        radius_cells = max(1, int(np.ceil(radius / self.resolution)))

        row_start = max(0, row - radius_cells)
        row_end = min(self.cells, row + radius_cells + 1)
        column_start = max(0, column - radius_cells)
        column_end = min(self.cells, column + radius_cells + 1)
        if row_start >= row_end or column_start >= column_end:
            return

        rows, columns = np.ogrid[row_start:row_end, column_start:column_end]
        mask = (rows - row) ** 2 + (columns - column) ** 2 <= radius_cells**2
        plane[row_start:row_end, column_start:column_end][mask] = 1.0
