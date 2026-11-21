"""Static coastline layer for the Maritime World Simulator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import COASTLINE

from ._geo import haversine_metres


@dataclass(frozen=True)
class Coast:
    """Distance from a location to the nearest mapped coastline."""

    distance: float
    is_near: bool


class Coastline:
    """Load Natural Earth coastline geometry and provide proximity queries.

    The downloaded Natural Earth source is a collection of coast *lines*, not
    land polygons.  Consequently this layer intentionally exposes shoreline
    distance; land/water classification should come from bathymetry or ENC
    polygons where it matters for safety logic.
    """

    def __init__(self, path: Path | str | None = None, safety_margin: float = 50.0) -> None:
        if safety_margin < 0.0:
            raise ValueError("safety_margin must be non-negative.")
        self.path = Path(path) if path is not None else COASTLINE / "otranto_coastline.geojson"
        self.safety_margin = float(safety_margin)
        self._geometry = None
        self._feature_count = 0

    @property
    def loaded(self) -> bool:
        return self._geometry is not None

    def load(self) -> "Coastline":
        """Load the cropped coastline GeoJSON in WGS-84 coordinates."""

        import geopandas as gpd
        import shapely

        if not self.path.exists():
            raise RuntimeError(f"Coastline file not found: {self.path}.")

        coastline = gpd.read_file(self.path)
        if coastline.empty:
            raise ValueError(f"Coastline file contains no features: {self.path}.")
        if coastline.crs is None:
            coastline = coastline.set_crs("EPSG:4326")
        else:
            coastline = coastline.to_crs("EPSG:4326")

        self._geometry = shapely.union_all(coastline.geometry.to_numpy())
        self._feature_count = len(coastline)
        return self

    def sample(
        self,
        latitude: float,
        longitude: float,
        safety_margin: float | None = None,
    ) -> Coast:
        """Return shoreline distance and whether the location is within margin."""

        from shapely.geometry import Point
        from shapely.ops import nearest_points

        geometry = self._require_geometry()
        margin = self.safety_margin if safety_margin is None else float(safety_margin)
        if margin < 0.0:
            raise ValueError("safety_margin must be non-negative.")

        point = Point(float(longitude), float(latitude))
        nearest, _ = nearest_points(geometry, point)
        distance = haversine_metres(latitude, longitude, nearest.y, nearest.x)
        return Coast(distance=distance, is_near=distance <= margin)

    def distance(self, latitude: float, longitude: float) -> float:
        """Return distance to the nearest coastline in metres."""

        return self.sample(latitude, longitude).distance

    def info(self) -> dict[str, object] | None:
        """Return lightweight source metadata."""

        if self._geometry is None:
            return None
        bounds = self._geometry.bounds
        return {
            "path": str(self.path),
            "feature_count": self._feature_count,
            "bounds": (bounds[1], bounds[0], bounds[3], bounds[2]),
            "safety_margin": self.safety_margin,
        }

    def close(self) -> None:
        """Release loaded geometry."""

        self._geometry = None
        self._feature_count = 0

    def _require_geometry(self):
        if self._geometry is None:
            raise RuntimeError("Coastline is not loaded. Call load() first.")
        return self._geometry
