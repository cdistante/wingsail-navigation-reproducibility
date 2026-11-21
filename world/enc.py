"""Electronic Navigational Chart (S-57) semantic layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import ENC


@dataclass(frozen=True)
class ChartSample:
    """Safety-relevant ENC semantics at one geographic point."""

    is_land: bool
    is_restricted: bool


class ENCChart:
    """Load selected S-57 object classes for chart-aware planning.

    The layer deliberately extracts only durable semantic features.  ENC files
    are optional at this stage; the simulator remains usable with coastline
    and bathymetry until official chart cells are supplied in ``data/enc``.
    """

    _LAND_LAYERS = {"LNDARE", "LNDRGN", "SLCONS"}
    _RESTRICTED_LAYERS = {"RESARE", "MIPARE", "CTNARE", "FAIRWY"}

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._land = None
        self._restricted = None
        self._layers: tuple[str, ...] = ()

    @property
    def loaded(self) -> bool:
        return self._land is not None and self._restricted is not None

    def load(self, path: Path | str | None = None) -> "ENCChart":
        """Read an S-57 ``.000`` cell and extract land/restricted polygons."""

        import fiona
        import shapely
        from shapely.geometry import shape

        self.close()
        if path is not None:
            self.path = Path(path)
        source = self.path or self._find_default_file()
        if not source.exists():
            raise RuntimeError(f"ENC cell not found: {source}.")

        land = []
        restricted = []
        layers = fiona.listlayers(source)
        for layer in layers:
            upper = layer.upper()
            if upper not in self._LAND_LAYERS and upper not in self._RESTRICTED_LAYERS:
                continue
            with fiona.open(source, layer=layer) as features:
                geometries = [shape(feature["geometry"]) for feature in features if feature["geometry"]]
            if upper in self._LAND_LAYERS:
                land.extend(geometries)
            else:
                restricted.extend(geometries)

        self.path = source
        self._land = shapely.union_all(land) if land else shapely.GeometryCollection()
        self._restricted = (
            shapely.union_all(restricted) if restricted else shapely.GeometryCollection()
        )
        self._layers = tuple(layers)
        return self

    def sample(self, latitude: float, longitude: float) -> ChartSample:
        """Return land/restricted-area membership at a WGS-84 point."""

        from shapely.geometry import Point

        if not self.loaded:
            raise RuntimeError("ENC chart is not loaded. Call load() first.")
        point = Point(float(longitude), float(latitude))
        return ChartSample(
            is_land=bool(self._land.covers(point)),
            is_restricted=bool(self._restricted.covers(point)),
        )

    def info(self) -> dict[str, object] | None:
        """Return lightweight chart metadata."""

        if not self.loaded or self.path is None:
            return None
        return {
            "path": str(self.path),
            "layers": self._layers,
            "land_bounds": self._land.bounds if not self._land.is_empty else None,
            "restricted_bounds": (
                self._restricted.bounds if not self._restricted.is_empty else None
            ),
        }

    def close(self) -> None:
        """Release extracted geometry."""

        self._land = None
        self._restricted = None
        self._layers = ()

    @staticmethod
    def _find_default_file() -> Path:
        files = sorted(ENC.glob("*.000"))
        if not files:
            raise RuntimeError(
                f"No ENC cell found in {ENC}. Add an S-57 .000 file before loading ENCChart."
            )
        return files[0]
