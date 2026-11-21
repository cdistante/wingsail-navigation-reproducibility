"""Wind forecast layer for the maritime-world simulator.

The downloaded GFS files contain the 10 m wind components for individual
forecast steps.  :class:`Weather` turns those files into one regular,
time-indexed wind field that a simulator, JEPA model, or MPC controller can
query at arbitrary positions and times.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Union

import numpy as np
import xarray as xr

from config import WEATHER


TimeLike = Union[datetime, np.datetime64, str]


@dataclass(frozen=True)
class Wind:
    """Local 10 m wind in the East-North-Up navigation frame.

    ``u`` is positive towards the east and ``v`` is positive towards the
    north, both in metres per second.  ``direction`` is the meteorological
    direction *from* which the wind blows, measured clockwise from true north.
    """

    u: float
    v: float
    speed: float
    direction: float


class Weather:
    """Load and interpolate the GFS 10 m wind forecast files.

    Parameters
    ----------
    data_dir:
        Directory containing the ``*.nc`` files produced by
        ``download/download_weather.py``.  By default this is
        ``data/weather`` from :mod:`config`.

    Notes
    -----
    Interpolation is linear in latitude, longitude, and valid forecast time.
    Requests outside the forecast domain raise :class:`ValueError` rather
    than silently extrapolating environmental conditions.
    """

    _WIND_VARIABLES = ("u10", "v10")

    def __init__(
        self,
        data_dir: Path | str | None = None,
        *,
        latitude_bounds: tuple[float, float] | None = None,
        longitude_bounds: tuple[float, float] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else Path(WEATHER)
        self.latitude_bounds = latitude_bounds
        self.longitude_bounds = longitude_bounds
        self.dataset: xr.Dataset | None = None
        self.loaded = False

    def load(self) -> "Weather":
        """Load every GFS forecast file and return this weather provider.

        Each downloaded file represents one forecast valid time.  The source
        coordinate can be either a scalar ``valid_time`` or a ``time`` plus
        ``step`` pair; both formats are normalised to a ``valid_time`` axis.
        """

        files = sorted(self.data_dir.glob("*.nc"))
        if not files:
            raise RuntimeError(f"No weather files found in {self.data_dir}.")

        self.close()
        forecasts = [
            (self._forecast_run_time(path), self._load_forecast(path)) for path in files
        ]
        forecasts = self._latest_forecast_per_valid_time(forecasts)
        dataset = xr.concat(
            forecasts,
            dim="valid_time",
            data_vars="minimal",
            coords="minimal",
            compat="override",
        ).sortby("valid_time")

        self.dataset = dataset.load()
        self.loaded = True
        return self

    def wind(self, latitude: float, longitude: float, when: TimeLike) -> Wind:
        """Return the interpolated 10 m wind at a position and time.

        ``latitude`` and ``longitude`` are geographic degrees.  Longitudes
        are accepted in either the ``[-180, 180)`` or ``[0, 360)`` convention.
        Naive datetimes are interpreted as UTC; timezone-aware datetimes are
        converted to UTC before interpolation.
        """

        dataset = self._require_dataset()
        longitude = self._normalise_longitude(longitude)
        valid_time = self._normalise_time(when)

        self._ensure_in_bounds(dataset, latitude, longitude, valid_time)

        sample = dataset.interp(
            latitude=float(latitude),
            longitude=longitude,
            valid_time=valid_time,
        )

        u = float(sample["u10"].item())
        v = float(sample["v10"].item())

        if not np.isfinite(u) or not np.isfinite(v):
            raise ValueError("Wind interpolation returned missing data.")

        speed = float(np.hypot(u, v))
        direction = float((np.degrees(np.arctan2(u, v)) + 180.0) % 360.0)

        return Wind(u=u, v=v, speed=speed, direction=direction)

    def sample(self, latitude: float, longitude: float, when: TimeLike) -> Wind:
        """Alias for :meth:`wind`, useful for a generic world sampler."""

        return self.wind(latitude, longitude, when)

    def info(self) -> xr.Dataset | None:
        """Return the loaded dataset, or ``None`` before :meth:`load`."""

        return self.dataset

    def plot(self, when: TimeLike, stride: int = 1) -> None:
        """Plot the interpolated 10 m wind field for visual inspection."""

        if stride < 1:
            raise ValueError("stride must be at least 1.")

        import matplotlib.pyplot as plt

        dataset = self._require_dataset()
        valid_time = self._normalise_time(when)
        self._ensure_time_in_bounds(dataset, valid_time)

        field = dataset.interp(valid_time=valid_time)
        field = field.isel(latitude=slice(None, None, stride), longitude=slice(None, None, stride))

        plt.figure(figsize=(10, 8))
        plt.quiver(field.longitude, field.latitude, field.u10, field.v10)
        plt.xlabel("Longitude [deg]")
        plt.ylabel("Latitude [deg]")
        plt.title(f"10 m wind flow (TO) — {np.datetime_as_string(valid_time, unit='m')}")
        plt.tight_layout()
        plt.show()

    def close(self) -> None:
        """Release any loaded forecast data."""

        if self.dataset is not None:
            self.dataset.close()

        self.dataset = None
        self.loaded = False

    def _load_forecast(self, path: Path) -> xr.Dataset:
        """Open one file and convert it to a single-valid-time wind field."""

        with xr.open_dataset(path) as source:
            missing = set(self._WIND_VARIABLES).difference(source.data_vars)
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(f"{path.name} is missing required variable(s): {names}.")

            valid_time = self._extract_valid_time(source, path)
            forecast = source[list(self._WIND_VARIABLES)].squeeze(drop=True).load()

        if "latitude" not in forecast.coords or "longitude" not in forecast.coords:
            raise ValueError(f"{path.name} must provide latitude and longitude coordinates.")

        forecast = forecast.assign_coords(
            longitude=self._normalise_longitude(forecast.longitude)
        ).sortby("longitude").sortby("latitude")
        if self.latitude_bounds is not None:
            south, north = sorted(float(value) for value in self.latitude_bounds)
            forecast = forecast.sel(latitude=slice(south, north))
        if self.longitude_bounds is not None:
            west, east = sorted(
                float(self._normalise_longitude(value)) for value in self.longitude_bounds
            )
            forecast = forecast.sel(longitude=slice(west, east))
        if forecast.sizes.get("latitude", 0) < 1 or forecast.sizes.get("longitude", 0) < 1:
            raise ValueError(f"{path.name} has no cells inside the requested geographic bounds.")

        # The individual files have scalar time metadata.  A dedicated axis is
        # necessary so xarray can interpolate across forecast steps later on.
        return forecast.expand_dims(valid_time=[valid_time])

    @staticmethod
    def _forecast_run_time(path: Path) -> np.datetime64:
        """Extract the GFS production time from its local filename.

        Several GFS runs can predict the same valid time.  Filenames produced
        by :mod:`download.download_weather` contain the run time, so retain
        the newest forecast rather than accidentally using an older one just
        because it sorts first alphabetically.
        """

        match = re.fullmatch(r"gfs_med_(\d{8}_\d{2})_f\d+\.nc", path.name)
        if match is not None:
            run = datetime.strptime(match.group(1), "%Y%m%d_%H")
            return np.datetime64(run, "ns")
        return np.datetime64(
            datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc), "ns"
        )

    @staticmethod
    def _latest_forecast_per_valid_time(
        forecasts: list[tuple[np.datetime64, xr.Dataset]],
    ) -> list[xr.Dataset]:
        """Keep the newest GFS run for each forecast valid time."""

        selected: dict[np.datetime64, tuple[np.datetime64, xr.Dataset]] = {}
        for run_time, forecast in forecasts:
            valid_time = np.asarray(forecast.valid_time.values).reshape(-1)[0].astype(
                "datetime64[ns]"
            )
            previous = selected.get(valid_time)
            if previous is None or run_time >= previous[0]:
                selected[valid_time] = (run_time, forecast)
        return [selected[time][1] for time in sorted(selected)]

    @staticmethod
    def _extract_valid_time(dataset: xr.Dataset, path: Path) -> np.datetime64:
        """Read the one valid forecast time represented by a source file."""

        if "valid_time" in dataset.coords:
            value = dataset.coords["valid_time"].values
        elif "time" in dataset.coords and "step" in dataset.coords:
            value = (dataset.coords["time"] + dataset.coords["step"]).values
        elif "time" in dataset.coords:
            value = dataset.coords["time"].values
        else:
            raise ValueError(f"Cannot determine the valid forecast time in {path.name}.")

        values = np.asarray(value).reshape(-1)
        if values.size != 1:
            raise ValueError(
                f"{path.name} contains {values.size} valid times; expected one per forecast file."
            )

        valid_time = values[0].astype("datetime64[ns]")
        if np.isnat(valid_time):
            raise ValueError(f"{path.name} has an invalid valid forecast time.")

        return valid_time

    def _require_dataset(self) -> xr.Dataset:
        if not self.loaded or self.dataset is None:
            raise RuntimeError("Weather is not loaded. Call load() first.")
        return self.dataset

    @staticmethod
    def _normalise_longitude(longitude):
        """Map geographic longitudes to the ``[-180, 180)`` convention."""

        return ((longitude + 180.0) % 360.0) - 180.0

    @staticmethod
    def _normalise_time(when: TimeLike) -> np.datetime64:
        """Convert supported time inputs to a UTC ``datetime64[ns]`` value."""

        if isinstance(when, datetime):
            if when.tzinfo is not None:
                when = when.astimezone(timezone.utc).replace(tzinfo=None)
            value = np.datetime64(when, "ns")
        elif isinstance(when, np.datetime64):
            value = when.astype("datetime64[ns]")
        elif isinstance(when, str):
            text = when.strip()
            if text.endswith("Z"):
                text = f"{text[:-1]}+00:00"
            try:
                parsed = datetime.fromisoformat(text)
            except ValueError:
                value = np.datetime64(when, "ns")
            else:
                value = Weather._normalise_time(parsed)
        else:
            raise TypeError("when must be a datetime, numpy.datetime64, or ISO-8601 string.")

        if np.isnat(value):
            raise ValueError("when must be a valid time.")

        return value

    def _ensure_in_bounds(
        self,
        dataset: xr.Dataset,
        latitude: float,
        longitude: float,
        valid_time: np.datetime64,
    ) -> None:
        self._ensure_coordinate_in_bounds(dataset, "latitude", float(latitude))
        self._ensure_coordinate_in_bounds(dataset, "longitude", longitude)
        self._ensure_time_in_bounds(dataset, valid_time)

    @staticmethod
    def _ensure_coordinate_in_bounds(dataset: xr.Dataset, name: str, value: float) -> None:
        coordinate = np.asarray(dataset.coords[name].values, dtype=float)
        lower, upper = float(np.nanmin(coordinate)), float(np.nanmax(coordinate))
        if not np.isfinite(value) or value < lower or value > upper:
            raise ValueError(f"{name}={value} is outside forecast coverage [{lower}, {upper}].")

    @staticmethod
    def _ensure_time_in_bounds(dataset: xr.Dataset, value: np.datetime64) -> None:
        coordinate = np.asarray(dataset.valid_time.values).astype("datetime64[ns]")
        lower, upper = coordinate.min(), coordinate.max()
        if value < lower or value > upper:
            start = np.datetime_as_string(lower, unit="m")
            end = np.datetime_as_string(upper, unit="m")
            requested = np.datetime_as_string(value, unit="m")
            raise ValueError(
                f"valid_time={requested} is outside forecast coverage [{start}, {end}]."
            )
