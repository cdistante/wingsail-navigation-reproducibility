"""Global occupancy map used by the integrated legacy planner stack.

The legacy project built its global grid from ``navigation_obstacles.geojson``
and a cached NumPy array.  Those two files are corrupted in the supplied
legacy archive, so this module preserves the *role* and coordinate interface
of that global grid while sourcing occupancy from the Maritime World GEBCO
bathymetry dataset.

Cells with water depth shallower than ``minimum_depth_metres`` (including
land) are treated as occupied.  The full source grid remains georeferenced in
latitude/longitude and can be sampled into a boat-centred local planner grid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from math import cos, pi

import numpy as np
import xarray as xr

from config import BATHYMETRY

_EARTH_RADIUS_M = 6_371_000.0


@dataclass
class GlobalOccupancyMap:
    """Georeferenced global binary occupancy grid from GEBCO or GeoJSON."""

    minimum_depth_metres: float = 2.0
    path: Path | str | None = None
    resolution_metres: float = 115.0
    geojson_padding_metres: float = 1_000.0
    geojson_closing_radius: int = 4

    def __post_init__(self) -> None:
        self.path = Path(self.path) if self.path is not None else self._default_bathymetry_path()
        self.latitudes: np.ndarray | None = None
        self.longitudes: np.ndarray | None = None
        self.elevation: np.ndarray | None = None
        self.occupancy: np.ndarray | None = None
        self.loaded = False

    @staticmethod
    def _default_bathymetry_path() -> Path:
        candidates = sorted(Path(BATHYMETRY).glob("gebco_*.nc"))
        candidates = [path for path in candidates if "sub_ice" not in path.name] or candidates
        if not candidates:
            raise RuntimeError(f"No GEBCO bathymetry NetCDF found in {BATHYMETRY}.")
        return candidates[0]

    def load(self) -> "GlobalOccupancyMap":
        """Load the selected source and build a compact uint8 occupancy raster."""

        if not Path(self.path).exists():
            raise RuntimeError(f"Global-map bathymetry file not found: {self.path}.")
        if Path(self.path).suffix.lower() in {".geojson", ".json"}:
            return self._load_geojson(Path(self.path))
        with xr.open_dataset(self.path) as dataset:
            if "elevation" not in dataset:
                raise ValueError(f"Expected GEBCO 'elevation' variable in {self.path}.")
            self.latitudes = np.asarray(dataset["lat"].values, dtype=np.float64)
            self.longitudes = np.asarray(dataset["lon"].values, dtype=np.float64)
            self.elevation = np.asarray(dataset["elevation"].values, dtype=np.float32)
        # GEBCO: elevation <= 0 is below sea level.  Water depth is -elevation.
        # Occupied means land OR water shallower than the required clearance.
        self.occupancy = (self.elevation >= -float(self.minimum_depth_metres)).astype(np.uint8)
        self.loaded = True
        return self

    def _load_geojson(self, path: Path) -> "GlobalOccupancyMap":
        """Rasterize the legacy simulator_harmonic obstacle GeoJSON."""

        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        features = data.get("features", [])
        coordinates: list[tuple[float, float]] = []
        for feature in features:
            coordinates.extend(self._flatten_coordinates(feature.get("geometry")))
        if not coordinates:
            raise ValueError(f"No coordinates found in {path}.")

        reference_latitude = min(latitude for _, latitude in coordinates)
        reference_longitude = min(longitude for longitude, _ in coordinates)
        projected = [
            self._longitude_latitude_to_metres(
                longitude, latitude, reference_latitude, reference_longitude
            )
            for longitude, latitude in coordinates
        ]
        east_values = [east for east, _ in projected]
        north_values = [north for _, north in projected]
        origin_east = min(east_values) - float(self.geojson_padding_metres)
        origin_north = min(north_values) - float(self.geojson_padding_metres)
        width = int(
            np.ceil(
                (max(east_values) - min(east_values) + 2.0 * self.geojson_padding_metres)
                / self.resolution_metres
            )
        )
        height = int(
            np.ceil(
                (max(north_values) - min(north_values) + 2.0 * self.geojson_padding_metres)
                / self.resolution_metres
            )
        )
        occupancy = np.zeros((height, width), dtype=np.uint8)

        try:
            from matplotlib.path import Path as PolygonPath
            from scipy.ndimage import binary_closing
        except ImportError as exc:
            raise RuntimeError("GeoJSON map loading requires matplotlib and scipy.") from exc

        for feature in features:
            geometry = feature.get("geometry") or {}
            properties = feature.get("properties") or {}
            if geometry.get("type") == "Polygon" and properties.get("layer") == "LNDARE":
                for ring in self._polygon_rings(geometry):
                    self._fill_polygon(
                        occupancy,
                        ring,
                        reference_latitude,
                        reference_longitude,
                        origin_east,
                        origin_north,
                        PolygonPath,
                    )
            else:
                for longitude, latitude in self._flatten_coordinates(geometry):
                    east, north = self._longitude_latitude_to_metres(
                        longitude, latitude, reference_latitude, reference_longitude
                    )
                    column = int(round((east - origin_east) / self.resolution_metres))
                    row = int(round((north - origin_north) / self.resolution_metres))
                    if 0 <= row < height and 0 <= column < width:
                        occupancy[row, column] = 1

        radius = max(0, int(self.geojson_closing_radius))
        if radius:
            structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
            occupancy = binary_closing(occupancy, structure=structure).astype(np.uint8)

        self.latitudes = reference_latitude + (
            (origin_north + np.arange(height) * self.resolution_metres) / _EARTH_RADIUS_M
        ) * (180.0 / pi)
        self.longitudes = reference_longitude + (
            (origin_east + np.arange(width) * self.resolution_metres)
            / (_EARTH_RADIUS_M * max(abs(cos(reference_latitude * pi / 180.0)), 1e-9))
        ) * (180.0 / pi)
        self.elevation = None
        self.occupancy = occupancy
        self.loaded = True
        return self

    @staticmethod
    def _flatten_coordinates(geometry: dict | None) -> list[tuple[float, float]]:
        if not geometry:
            return []
        coordinates = geometry.get("coordinates", [])
        if geometry.get("type") == "Point":
            return [tuple(coordinates)]
        if coordinates and isinstance(coordinates[0], (int, float)):
            return [tuple(coordinates)]
        flattened: list[tuple[float, float]] = []
        for child in coordinates:
            flattened.extend(GlobalOccupancyMap._flatten_coordinates({"coordinates": child}))
        return flattened

    @staticmethod
    def _polygon_rings(geometry: dict) -> list[list[tuple[float, float]]]:
        if geometry.get("type") != "Polygon":
            return []
        return [[tuple(point) for point in ring] for ring in geometry.get("coordinates", [])]

    @staticmethod
    def _longitude_latitude_to_metres(
        longitude: float,
        latitude: float,
        reference_latitude: float,
        reference_longitude: float,
    ) -> tuple[float, float]:
        latitude_scale = _EARTH_RADIUS_M * pi / 180.0
        east = (longitude - reference_longitude) * latitude_scale * cos(reference_latitude * pi / 180.0)
        north = (latitude - reference_latitude) * latitude_scale
        return east, north

    def _fill_polygon(
        self,
        occupancy: np.ndarray,
        ring: list[tuple[float, float]],
        reference_latitude: float,
        reference_longitude: float,
        origin_east: float,
        origin_north: float,
        polygon_path,
    ) -> None:
        if len(ring) < 3:
            return
        points = np.asarray(
            [
                self._longitude_latitude_to_metres(
                    longitude, latitude, reference_latitude, reference_longitude
                )
                for longitude, latitude in ring
            ],
            dtype=float,
        )
        points[:, 0] = (points[:, 0] - origin_east) / self.resolution_metres
        points[:, 1] = (points[:, 1] - origin_north) / self.resolution_metres
        min_column = max(int(np.floor(points[:, 0].min())), 0)
        max_column = min(int(np.ceil(points[:, 0].max())), occupancy.shape[1] - 1)
        min_row = max(int(np.floor(points[:, 1].min())), 0)
        max_row = min(int(np.ceil(points[:, 1].max())), occupancy.shape[0] - 1)
        if min_column > max_column or min_row > max_row:
            return
        columns, rows = np.meshgrid(
            np.arange(min_column, max_column + 1),
            np.arange(min_row, max_row + 1),
        )
        inside = polygon_path(points).contains_points(
            np.column_stack((columns.ravel(), rows.ravel()))
        ).reshape(rows.shape)
        occupancy[min_row : max_row + 1, min_column : max_column + 1][inside] = 1

    @property
    def shape(self) -> tuple[int, int]:
        self._require_loaded()
        assert self.occupancy is not None
        return self.occupancy.shape

    def latitude_longitude_to_grid(self, latitude: float, longitude: float) -> tuple[float, float]:
        """Return floating ``(row, column)`` coordinates in the global raster."""

        self._require_loaded()
        assert self.latitudes is not None and self.longitudes is not None
        dlat = float(np.median(np.diff(self.latitudes)))
        dlon = float(np.median(np.diff(self.longitudes)))
        row = (float(latitude) - float(self.latitudes[0])) / dlat
        column = (float(longitude) - float(self.longitudes[0])) / dlon
        return row, column

    def grid_to_latitude_longitude(self, row: float, column: float) -> tuple[float, float]:
        """Return the geographic position of a floating global-grid coordinate."""

        self._require_loaded()
        assert self.latitudes is not None and self.longitudes is not None
        dlat = float(np.median(np.diff(self.latitudes)))
        dlon = float(np.median(np.diff(self.longitudes)))
        return (
            float(self.latitudes[0] + row * dlat),
            float(self.longitudes[0] + column * dlon),
        )

    def local_obstacle_mask(
        self,
        latitude: float,
        longitude: float,
        *,
        cells: int,
        resolution_metres: float,
    ) -> np.ndarray:
        """Sample the global raster into a boat-centred East/North local mask.

        The returned array follows the planner convention: rows increase south,
        columns increase east, and the boat is at the centre cell.
        Locations outside the GEBCO domain are conservatively marked occupied.
        """

        self._require_loaded()
        if cells < 3 or resolution_metres <= 0.0:
            raise ValueError("Local-map cells must be >=3 and resolution must be positive.")
        assert self.latitudes is not None and self.longitudes is not None and self.occupancy is not None

        centre = (cells - 1) / 2.0
        rows, columns = np.indices((cells, cells), dtype=np.float64)
        north_m = (centre - rows) * float(resolution_metres)
        east_m = (columns - centre) * float(resolution_metres)

        latitude_grid = float(latitude) + (north_m / _EARTH_RADIUS_M) * (180.0 / pi)
        longitude_scale = _EARTH_RADIUS_M * max(abs(cos(float(latitude) * pi / 180.0)), 1e-9)
        longitude_grid = float(longitude) + (east_m / longitude_scale) * (180.0 / pi)

        dlat = float(np.median(np.diff(self.latitudes)))
        dlon = float(np.median(np.diff(self.longitudes)))
        src_rows = np.rint((latitude_grid - self.latitudes[0]) / dlat).astype(np.int64)
        src_columns = np.rint((longitude_grid - self.longitudes[0]) / dlon).astype(np.int64)

        inside = (
            (src_rows >= 0)
            & (src_rows < self.occupancy.shape[0])
            & (src_columns >= 0)
            & (src_columns < self.occupancy.shape[1])
        )
        mask = np.ones((cells, cells), dtype=bool)
        mask[inside] = self.occupancy[src_rows[inside], src_columns[inside]] > 0
        return mask

    def _require_loaded(self) -> None:
        if not self.loaded:
            raise RuntimeError("GlobalOccupancyMap is not loaded. Call load() first.")
