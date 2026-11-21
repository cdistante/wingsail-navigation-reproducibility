"""Surface-current layer for the Maritime World Simulator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import xarray as xr

from config import CURRENTS, CURRENT_SAMPLING

from ._geo import TimeLike, local_east_north_metres, normalise_longitude, normalise_time
from ._netcdf import coverage_summary, ensure_in_coverage, load_temporal_netcdf


@dataclass(frozen=True)
class Current:
    """Water velocity in the East-North-Up navigation frame.

    ``direction`` is the direction *towards* which water flows, clockwise from
    true north.  Components and speed are in metres per second.
    """

    u: float
    v: float
    speed: float
    direction: float
    depth: float


class Currents:
    """Load and interpolate HYCOM current fields."""

    _VARIABLES = ("water_u", "water_v")

    def __init__(
        self,
        data_dir: Path | str | None = None,
        *,
        persist_single_snapshot: bool = True,
        coastal_fallback_max_distance_metres: float = float(
            CURRENT_SAMPLING["coastal_fallback_max_distance_metres"]
        ),
        latitude_bounds: tuple[float, float] | None = None,
        longitude_bounds: tuple[float, float] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else Path(CURRENTS)
        self.persist_single_snapshot = persist_single_snapshot
        if coastal_fallback_max_distance_metres < 0.0:
            raise ValueError("coastal_fallback_max_distance_metres must be non-negative.")
        self.coastal_fallback_max_distance_metres = float(
            coastal_fallback_max_distance_metres
        )
        self.latitude_bounds = latitude_bounds
        self.longitude_bounds = longitude_bounds
        self._dataset: xr.Dataset | None = None
        self._reported_coastal_fallback_cells: set[tuple[float, float]] = set()

    @property
    def loaded(self) -> bool:
        return self._dataset is not None

    def load(self) -> "Currents":
        """Load all HYCOM NetCDF fields from the configured directory."""

        self.close()
        self._dataset = load_temporal_netcdf(
            self.data_dir,
            self._VARIABLES,
            latitude_bounds=self.latitude_bounds,
            longitude_bounds=self.longitude_bounds,
        )
        return self

    def current(
        self,
        latitude: float,
        longitude: float,
        when: TimeLike,
        depth: float = 0.0,
    ) -> Current:
        """Interpolate the water velocity at the requested depth in metres."""

        dataset = self._require_dataset()
        longitude = float(normalise_longitude(longitude))
        if self.persist_single_snapshot and dataset.time.size == 1:
            # The current downloader requests HYCOM's present surface field.
            # Use it as a frozen disturbance for an episode rather than
            # invalidating every world step after that one timestamp.
            normalise_time(when)
            time = ensure_in_coverage(
                dataset,
                latitude,
                longitude,
                dataset.time.values[0],
                depth=depth if "depth" in dataset.coords else None,
            )
        else:
            time = ensure_in_coverage(
                dataset,
                latitude,
                longitude,
                when,
                depth=depth if "depth" in dataset.coords else None,
            )
        coordinates: dict[str, float | np.datetime64] = {
            "latitude": float(latitude),
            "longitude": longitude,
            "time": time,
        }
        if "depth" in dataset.coords:
            coordinates["depth"] = float(depth)

        # scipy interpolation is undefined on singleton dimensions. HYCOM
        # subsets often contain exactly one forecast time and surface level,
        # so select those axes exactly and interpolate only where there are at
        # least two grid points. Keep the time/depth-sampled spatial field for
        # a bounded coastal fallback when bilinear interpolation touches a
        # land-masked HYCOM cell.
        field = dataset
        for name in ("time", "depth"):
            if name not in coordinates:
                continue
            value = coordinates[name]
            if field.coords[name].size == 1:
                field = field.sel({name: value})
            else:
                field = field.interp({name: value})

        spatial_field = field
        spatial_coordinates: dict[str, float] = {}
        for name in ("latitude", "longitude"):
            value = float(coordinates[name])
            if field.coords[name].size == 1:
                field = field.sel({name: value})
            else:
                spatial_coordinates[name] = value
        if spatial_coordinates:
            field = field.interp(spatial_coordinates)
        u = float(field["water_u"].item())
        v = float(field["water_v"].item())
        if not np.all(np.isfinite([u, v])):
            u, v = self._nearest_valid_coastal_current(
                spatial_field,
                latitude=float(latitude),
                longitude=longitude,
            )

        return Current(
            u=u,
            v=v,
            speed=float(np.hypot(u, v)),
            direction=float(np.degrees(np.arctan2(u, v)) % 360.0),
            depth=float(depth),
        )

    def sample(
        self,
        latitude: float,
        longitude: float,
        when: TimeLike,
        depth: float = 0.0,
    ) -> Current:
        """Alias for :meth:`current` to match the world-layer interface."""

        return self.current(latitude, longitude, when, depth)

    def info(self) -> dict[str, object] | None:
        """Return coverage metadata, including available depth levels."""

        if self._dataset is None:
            return None
        summary = coverage_summary(self._dataset)
        if "depth" in self._dataset.coords:
            summary["depth_range"] = (
                float(self._dataset.depth.values.min()),
                float(self._dataset.depth.values.max()),
            )
        summary["persistent_single_snapshot"] = self.persist_single_snapshot
        summary["coastal_fallback_max_distance_metres"] = (
            self.coastal_fallback_max_distance_metres
        )
        return summary

    def close(self) -> None:
        """Release loaded data."""

        if self._dataset is not None:
            self._dataset.close()
        self._dataset = None
        self._reported_coastal_fallback_cells.clear()

    def _require_dataset(self) -> xr.Dataset:
        if self._dataset is None:
            raise RuntimeError("Currents are not loaded. Call load() first.")
        return self._dataset

    def _nearest_valid_coastal_current(
        self,
        spatial_field: xr.Dataset,
        *,
        latitude: float,
        longitude: float,
    ) -> tuple[float, float]:
        """Return a nearby finite sea value after a masked coastal interpolation.

        This method is deliberately limited to spatial fields with ordinary
        1-D latitude/longitude axes, which is the format normalised by the
        HYCOM downloader.  It does not replace a missing field with zero and
        it refuses a fallback farther than the configured coastal tolerance.
        """

        if self.coastal_fallback_max_distance_metres <= 0.0:
            raise ValueError("Current interpolation returned missing data.")
        try:
            u = spatial_field["water_u"].transpose("latitude", "longitude").values
            v = spatial_field["water_v"].transpose("latitude", "longitude").values
        except ValueError as error:
            raise ValueError("Current interpolation returned missing data.") from error

        valid = np.isfinite(u) & np.isfinite(v)
        if not np.any(valid):
            raise ValueError("Current interpolation returned missing data: no valid sea cells.")

        latitudes = np.asarray(spatial_field.latitude.values, dtype=float)
        longitudes = np.asarray(spatial_field.longitude.values, dtype=float)
        latitude_grid, longitude_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
        east, north = local_east_north_metres(
            latitude_grid,
            longitude_grid,
            latitude,
            longitude,
        )
        distance = np.hypot(east, north)
        distance = np.where(valid, distance, np.inf)
        row, column = np.unravel_index(int(np.argmin(distance)), distance.shape)
        nearest_distance = float(distance[row, column])
        if not np.isfinite(nearest_distance):
            raise ValueError("Current interpolation returned missing data: no valid sea cells.")
        if nearest_distance > self.coastal_fallback_max_distance_metres:
            raise ValueError(
                "Current interpolation returned missing coastal data; nearest valid sea cell "
                f"is {nearest_distance:.0f} m away, beyond the "
                f"{self.coastal_fallback_max_distance_metres:.0f} m fallback limit."
            )

        nearest_latitude = float(latitudes[row])
        nearest_longitude = float(longitudes[column])
        fallback_cell = (nearest_latitude, nearest_longitude)
        if fallback_cell not in self._reported_coastal_fallback_cells:
            warnings.warn(
                "HYCOM interpolation touched a masked coastal cell; using nearest valid sea "
                f"current {nearest_distance:.0f} m away at "
                f"({nearest_latitude:.5f}, {nearest_longitude:.5f}).",
                RuntimeWarning,
                stacklevel=3,
            )
            self._reported_coastal_fallback_cells.add(fallback_cell)
        return float(u[row, column]), float(v[row, column])
