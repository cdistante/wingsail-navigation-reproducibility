"""Reusable NetCDF loading and validation helpers for environmental layers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr

from ._geo import normalise_longitude, normalise_time


def load_temporal_netcdf(
    directory: Path | str,
    variables: Iterable[str],
    *,
    latitude_bounds: tuple[float, float] | None = None,
    longitude_bounds: tuple[float, float] | None = None,
) -> xr.Dataset:
    """Load NetCDF files as a sorted ``time, latitude, longitude`` dataset.

    The project downloaders use both ``lat/lon`` and ``latitude/longitude``;
    this helper normalises them once at the world-layer boundary.  Optional
    geographic bounds are applied *before* each source is loaded into memory;
    this is useful for long, high-resolution wave archives when a simulation
    only uses one regional route.
    """

    directory = Path(directory)
    files = sorted(directory.glob("*.nc"))
    if not files:
        raise RuntimeError(f"No NetCDF files found in {directory}.")

    requested = tuple(variables)
    fields: list[tuple[float, xr.Dataset]] = []

    for path in files:
        with xr.open_dataset(path) as source:
            missing = set(requested).difference(source.data_vars)
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(f"{path.name} is missing required variable(s): {names}.")

            field = source[list(requested)]
            field = _standardise_coordinates(field, path)
            field = _subset_geographic(
                field,
                latitude_bounds=latitude_bounds,
                longitude_bounds=longitude_bounds,
                path=path,
            )
            # A repeated retrieval can overlap older forecast files.  File
            # modification time is the available source-version indicator for
            # these Copernicus/HYCOM subsets; later files win per time slice.
            fields.append((path.stat().st_mtime, field.load()))

    dataset = xr.concat(
        _latest_time_slices(fields),
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="override",
    ).sortby("time")

    return dataset.load()


def _subset_geographic(
    dataset: xr.Dataset,
    *,
    latitude_bounds: tuple[float, float] | None,
    longitude_bounds: tuple[float, float] | None,
    path: Path,
) -> xr.Dataset:
    """Restrict a normalised dataset to an inclusive regional bounding box."""

    if latitude_bounds is not None:
        south, north = sorted(float(value) for value in latitude_bounds)
        dataset = dataset.sel(latitude=slice(south, north))
    if longitude_bounds is not None:
        west, east = sorted(float(normalise_longitude(value)) for value in longitude_bounds)
        dataset = dataset.sel(longitude=slice(west, east))
    if dataset.sizes.get("latitude", 0) < 1 or dataset.sizes.get("longitude", 0) < 1:
        raise ValueError(f"{path.name} has no cells inside the requested geographic bounds.")
    return dataset


def _latest_time_slices(fields: list[tuple[float, xr.Dataset]]) -> list[xr.Dataset]:
    """Keep the most recently retrieved slice for every valid timestamp."""

    selected: dict[np.datetime64, tuple[float, xr.Dataset]] = {}
    for source_time, field in fields:
        times = np.asarray(field.time.values).astype("datetime64[ns]")
        for index, valid_time in enumerate(times):
            previous = selected.get(valid_time)
            if previous is None or source_time >= previous[0]:
                selected[valid_time] = (source_time, field.isel(time=slice(index, index + 1)))
    return [selected[time][1] for time in sorted(selected)]


def ensure_in_coverage(
    dataset: xr.Dataset,
    latitude: float,
    longitude: float,
    when,
    *,
    depth: float | None = None,
) -> np.datetime64:
    """Validate a query and return its normalised timestamp."""

    longitude = float(normalise_longitude(longitude))
    valid_time = normalise_time(when)
    _ensure_coordinate(dataset, "latitude", float(latitude))
    _ensure_coordinate(dataset, "longitude", longitude)
    _ensure_time(dataset, valid_time)
    if depth is not None:
        _ensure_coordinate(dataset, "depth", float(depth))
    return valid_time


def coverage_summary(dataset: xr.Dataset) -> dict[str, object]:
    """Return stable metadata without leaking the backing xarray object."""

    return {
        "time_start": str(np.datetime_as_string(dataset.time.values.min(), unit="m")),
        "time_end": str(np.datetime_as_string(dataset.time.values.max(), unit="m")),
        "latitude_range": (
            float(dataset.latitude.values.min()),
            float(dataset.latitude.values.max()),
        ),
        "longitude_range": (
            float(dataset.longitude.values.min()),
            float(dataset.longitude.values.max()),
        ),
        "variables": tuple(dataset.data_vars),
    }


def _standardise_coordinates(dataset: xr.Dataset, path: Path) -> xr.Dataset:
    renames: dict[str, str] = {}
    if "latitude" not in dataset.coords and "lat" in dataset.coords:
        renames["lat"] = "latitude"
    if "longitude" not in dataset.coords and "lon" in dataset.coords:
        renames["lon"] = "longitude"
    if "time" not in dataset.coords and "valid_time" in dataset.coords:
        renames["valid_time"] = "time"
    if renames:
        dataset = dataset.rename(renames)

    required = {"latitude", "longitude", "time"}
    missing = required.difference(dataset.coords)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{path.name} is missing required coordinate(s): {names}.")

    time_values = np.asarray(dataset.time.values).reshape(-1)
    if "time" not in dataset.dims:
        if time_values.size != 1:
            raise ValueError(f"{path.name} has a non-dimensional time coordinate.")
        dataset = dataset.expand_dims(time=[time_values[0]])

    return (
        dataset.assign_coords(
            longitude=normalise_longitude(dataset.longitude),
            time=np.asarray(dataset.time.values).astype("datetime64[ns]"),
        )
        .sortby("longitude")
        .sortby("latitude")
    )


def _ensure_coordinate(dataset: xr.Dataset, name: str, value: float) -> None:
    coordinate = np.asarray(dataset.coords[name].values, dtype=float)
    lower, upper = float(np.nanmin(coordinate)), float(np.nanmax(coordinate))
    if not np.isfinite(value) or value < lower or value > upper:
        raise ValueError(f"{name}={value} is outside coverage [{lower}, {upper}].")


def _ensure_time(dataset: xr.Dataset, value: np.datetime64) -> None:
    times = np.asarray(dataset.time.values).astype("datetime64[ns]")
    start, end = times.min(), times.max()
    if value < start or value > end:
        requested = np.datetime_as_string(value, unit="m")
        lower = np.datetime_as_string(start, unit="m")
        upper = np.datetime_as_string(end, unit="m")
        raise ValueError(f"time={requested} is outside coverage [{lower}, {upper}].")
