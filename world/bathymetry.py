"""Static GEBCO bathymetry layer for the Maritime World Simulator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import BATHYMETRY


@dataclass(frozen=True)
class Seabed:
    """Elevation and navigable water depth at a geographic location.

    GEBCO stores elevation relative to sea level.  ``depth`` is derived as a
    positive-down value for water, while ``elevation`` retains the source
    convention and is positive over land.
    """

    elevation: float
    depth: float
    is_land: bool


class Bathymetry:
    """Read a cropped GEBCO GeoTIFF and query seabed elevation/depth."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._dataset = None

    @property
    def loaded(self) -> bool:
        return self._dataset is not None

    def load(self) -> "Bathymetry":
        """Open the configured GeoTIFF, finding one in ``data/bathymetry`` if needed."""

        import rasterio

        self.close()
        path = self.path or self._find_default_file()
        if not path.exists():
            raise RuntimeError(f"Bathymetry GeoTIFF not found: {path}.")

        self.path = path
        self._dataset = rasterio.open(path)
        return self

    def sample(self, latitude: float, longitude: float) -> Seabed:
        """Return GEBCO elevation and positive-down water depth at a position."""

        dataset = self._require_dataset()
        x, y = self._to_dataset_coordinates(longitude, latitude)
        if not (dataset.bounds.left <= x <= dataset.bounds.right and dataset.bounds.bottom <= y <= dataset.bounds.top):
            raise ValueError("Position is outside bathymetry coverage.")

        value = next(dataset.sample([(x, y)], indexes=1, masked=True))[0]
        if np.ma.is_masked(value) or not np.isfinite(float(value)):
            raise ValueError("Bathymetry has no data at the requested position.")

        elevation = float(value)
        return Seabed(
            elevation=elevation,
            depth=max(-elevation, 0.0),
            is_land=elevation >= 0.0,
        )

    def depth(self, latitude: float, longitude: float) -> float:
        """Return positive-down water depth in metres."""

        return self.sample(latitude, longitude).depth

    def info(self) -> dict[str, object] | None:
        """Return source metadata without exposing the raster handle."""

        if self._dataset is None or self.path is None:
            return None
        bounds = self._dataset.bounds
        return {
            "path": str(self.path),
            "crs": str(self._dataset.crs),
            "width": self._dataset.width,
            "height": self._dataset.height,
            "bounds": (bounds.left, bounds.bottom, bounds.right, bounds.top),
            "nodata": self._dataset.nodata,
        }

    def close(self) -> None:
        """Close the GeoTIFF handle."""

        if self._dataset is not None:
            self._dataset.close()
        self._dataset = None

    def _find_default_file(self) -> Path:
        candidates = sorted(BATHYMETRY.glob("*.tif")) + sorted(BATHYMETRY.glob("*.tiff"))
        if not candidates:
            raise RuntimeError(
                f"No GeoTIFF found in {BATHYMETRY}. Run download/download_bathymetry.py first."
            )
        return candidates[0]

    def _to_dataset_coordinates(self, longitude: float, latitude: float) -> tuple[float, float]:
        dataset = self._require_dataset()
        if dataset.crs is None or dataset.crs.to_epsg() == 4326:
            return float(longitude), float(latitude)

        from rasterio.warp import transform

        x, y = transform("EPSG:4326", dataset.crs, [longitude], [latitude])
        return float(x[0]), float(y[0])

    def _require_dataset(self):
        if self._dataset is None:
            raise RuntimeError("Bathymetry is not loaded. Call load() first.")
        return self._dataset
