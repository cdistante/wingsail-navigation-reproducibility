"""Wave forecast layer for the Maritime World Simulator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from config import WAVES

from ._geo import TimeLike, normalise_longitude
from ._netcdf import coverage_summary, ensure_in_coverage, load_temporal_netcdf


@dataclass(frozen=True)
class Wave:
    """Sea-state values at one position and time.

    Direction is the meteorological/oceanographic *from* direction clockwise
    from true north, matching Copernicus Marine's ``VMDR`` convention.
    """

    height: float
    direction: float
    peak_period: float


class Waves:
    """Load and interpolate Copernicus significant-wave forecasts."""

    _VARIABLES = ("VHM0", "VMDR", "VTPK")
    DEFAULT_MAX_NEAREST_VALID_DISTANCE_METRES = 20_000.0

    def __init__(
        self,
        data_dir: Path | str | None = None,
        *,
        max_nearest_valid_distance_metres: float = DEFAULT_MAX_NEAREST_VALID_DISTANCE_METRES,
        latitude_bounds: tuple[float, float] | None = None,
        longitude_bounds: tuple[float, float] | None = None,
    ) -> None:
        if max_nearest_valid_distance_metres <= 0.0:
            raise ValueError("max_nearest_valid_distance_metres must be positive.")
        self.data_dir = Path(data_dir) if data_dir is not None else Path(WAVES)
        self.max_nearest_valid_distance_metres = float(max_nearest_valid_distance_metres)
        self.latitude_bounds = latitude_bounds
        self.longitude_bounds = longitude_bounds
        self._dataset: xr.Dataset | None = None

    @property
    def loaded(self) -> bool:
        return self._dataset is not None

    def load(self) -> "Waves":
        """Load all wave NetCDF files in the configured data directory."""

        self.close()
        self._dataset = load_temporal_netcdf(
            self.data_dir,
            self._VARIABLES,
            latitude_bounds=self.latitude_bounds,
            longitude_bounds=self.longitude_bounds,
        )
        return self

    def wave(self, latitude: float, longitude: float, when: TimeLike) -> Wave:
        """Interpolate wave height, direction, and peak period."""

        dataset = self._require_dataset()
        longitude = float(normalise_longitude(longitude))
        time = ensure_in_coverage(dataset, latitude, longitude, when)
        field = dataset.interp(latitude=float(latitude), longitude=longitude, time=time)

        values = [float(field[variable].item()) for variable in self._VARIABLES]
        if not np.all(np.isfinite(values)):
            values = self._nearest_valid_values(dataset, latitude, longitude, time)

        height, direction, peak_period = values
        direction %= 360.0
        if not np.all(np.isfinite([height, direction, peak_period])):
            raise ValueError("Wave interpolation returned missing data.")

        return Wave(height=height, direction=direction, peak_period=peak_period)

    def sample(self, latitude: float, longitude: float, when: TimeLike) -> Wave:
        """Alias for :meth:`wave` to match the world-layer interface."""

        return self.wave(latitude, longitude, when)

    def info(self) -> dict[str, object] | None:
        """Return coverage metadata, or ``None`` before loading."""

        return None if self._dataset is None else coverage_summary(self._dataset)

    def close(self) -> None:
        """Release loaded data."""

        if self._dataset is not None:
            self._dataset.close()
        self._dataset = None

    def _require_dataset(self) -> xr.Dataset:
        if self._dataset is None:
            raise RuntimeError("Waves are not loaded. Call load() first.")
        return self._dataset

    def _nearest_valid_values(
        self,
        dataset: xr.Dataset,
        latitude: float,
        longitude: float,
        time: np.datetime64,
    ) -> list[float]:
        """Use the nearest complete sea cell when coastal interpolation is masked.

        Copernicus fields are masked over land.  Linear interpolation at an
        offshore point can consequently become NaN when any surrounding corner
        is land.  This bounded fallback retains the closest complete water-cell
        observation instead of treating the route point as missing sea state.
        """

        field = (
            dataset.interp(time=time)
            if dataset.sizes.get("time", 0) > 1
            else dataset.sel(time=time, method="nearest")
        )
        values = np.stack(
            [np.asarray(field[variable].values, dtype=float) for variable in self._VARIABLES],
            axis=0,
        )
        valid = np.all(np.isfinite(values), axis=0)
        if not np.any(valid):
            raise ValueError("Wave forecast contains no valid sea cells at the requested time.")

        latitudes = np.asarray(field["latitude"].values, dtype=float)
        longitudes = np.asarray(field["longitude"].values, dtype=float)
        latitude_grid, longitude_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
        north = (latitude_grid - latitude) * 111_320.0
        east = normalise_longitude(longitude_grid - longitude) * 111_320.0 * np.cos(
            np.radians(latitude)
        )
        distance = np.hypot(east, north)
        distance[~valid] = np.inf
        row, column = np.unravel_index(np.argmin(distance), distance.shape)
        nearest_distance = float(distance[row, column])
        if nearest_distance > self.max_nearest_valid_distance_metres:
            raise ValueError(
                "Wave interpolation is missing and the nearest valid sea cell is "
                f"{nearest_distance:.0f} m away, beyond the configured "
                f"{self.max_nearest_valid_distance_metres:.0f} m fallback limit."
            )
        return [float(values[index, row, column]) for index in range(len(self._VARIABLES))]
