"""Retrieve registered seasonal wind, current, and wave fields from public archives."""

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from itertools import product
from pathlib import Path
from threading import Lock
import json
import math
import time

import numpy as np
import requests
import xarray as xr


STUDY = Path(__file__).resolve().parents[1]
SEASONS = {
    "winter": date(2026, 1, 12),
    "spring": date(2026, 4, 13),
    "summer": date(2026, 7, 13),
    "autumn": date(2026, 10, 12),
}
BOUNDS = ((39.85, 40.85), (17.65, 19.15))
MARINE = "https://marine-api.open-meteo.com/v1/marine"
SOURCES = {
    "wind": {
        "url": "https://archive-api.open-meteo.com/v1/archive",
        "model": "era5",
        "provider": "ECMWF/Copernicus Climate Change Service ERA5 single-level reanalysis",
        "variables": ("wind_speed_10m", "wind_direction_10m"),
        "units": ("m/s", "°"),
        "spacing": 0.25,
        "offset": 0.,
        "coordinate_mapping": "native regular grid",
        "citation": "https://doi.org/10.24381/cds.adbb2d47",
    },
    "current": {
        "url": MARINE,
        "model": "meteofrance_currents",
        "provider": "Météo-France SMOC total surface current (Eulerian, wave and tide components)",
        "variables": ("ocean_current_velocity", "ocean_current_direction"),
        "units": ("m/s", "°"),
        "spacing": 1. / 12.,
        "offset": 0.5,
        "coordinate_mapping": "native regular cell centres",
        "citation": "https://doi.org/10.48670/moi-00016",
    },
    "wave": {
        "url": MARINE,
        "model": "ecmwf_wam",
        "provider": "ECMWF WAM archived forecast (approximately 9 km)",
        "variables": ("wave_height", "wave_direction", "wave_peak_period"),
        "units": ("m", "°", "s"),
        "spacing": 0.08,
        "offset": 0.,
        "coordinate_mapping": "nearest native reduced-grid cell on a 0.08 degree regular target grid",
        "citation": "https://www.ecmwf.int/en/forecasts/datasets/set-ii",
    },
}
CONVENTIONS = {
    "u10": ("m s-1", "eastward_wind"),
    "v10": ("m s-1", "northward_wind"),
    "water_u": ("m s-1", "eastward_sea_water_velocity"),
    "water_v": ("m s-1", "northward_sea_water_velocity"),
    "VHM0": ("m", "sea_surface_wave_significant_height"),
    "VMDR": ("degree", "sea_surface_wave_from_direction"),
    "VTPK": ("s", "sea_surface_wave_period_at_variance_spectral_density_maximum"),
}
POINTS_PER_REQUEST = 36
SECONDS_PER_LOCATION = 0.16
RATE_LOCK = Lock()
NEXT_REQUEST = 0.


def checksum(path):
    return sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    staged = path.with_suffix(path.suffix + ".part")
    staged.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    staged.replace(path)


def axis(bounds, spacing, offset):
    low, high = (bound / spacing - offset for bound in bounds)
    return (np.arange(math.floor(low), math.ceil(high) + 1) + offset) * spacing


def hourly_range(start, end):
    return np.arange(np.datetime64(start, "h"), np.datetime64(end + timedelta(days=1), "h"))


def query_parameters(points, source, start, end):
    return {
        "latitude": ",".join(f"{lat:.6f}" for lat, _ in points),
        "longitude": ",".join(f"{lon:.6f}" for _, lon in points),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": "UTC",
        "cell_selection": "nearest",
        "wind_speed_unit": "ms",
        "hourly": ",".join(source["variables"]),
        "models": source["model"],
    }


def validated_rows(data, variables, units, expected_time, size):
    rows = data if isinstance(data, list) else [data]
    if len(rows) != size:
        raise ValueError(f"Expected {size} archive points, including mask cells, got {len(rows)}.")
    for row in rows:
        hourly = row.get("hourly", {})
        if row.get("utc_offset_seconds") != 0 or not np.array_equal(
            np.asarray(hourly.get("time", ()), dtype="datetime64[h]"), expected_time
        ):
            raise ValueError("The archive response did not provide every requested UTC hour.")
        for variable, unit in zip(variables, units, strict=True):
            if row.get("hourly_units", {}).get(variable) != unit or len(hourly.get(variable, ())) != len(expected_time):
                raise ValueError(f"Archive variable {variable} has unexpected dimensions or units.")
    return rows


def pace_request(locations, *, cooldown=None):
    global NEXT_REQUEST
    with RATE_LOCK:
        if cooldown is not None:
            NEXT_REQUEST = max(NEXT_REQUEST, time.monotonic() + cooldown)
            return
        while time.monotonic() < NEXT_REQUEST:
            time.sleep(min(10., NEXT_REQUEST - time.monotonic()))
        NEXT_REQUEST = time.monotonic() + SECONDS_PER_LOCATION * locations


def absent_row(source, time_count):
    return {
        "latitude": None, "longitude": None,
        "hourly": {name: [None] * time_count for name in source["variables"]},
    }


def fetch_task(task):
    path, source, points, start, end = task
    expected_time = hourly_range(start, end)
    params = query_parameters(points, source, start, end)
    if not path.exists():
        for attempt in range(4):
            try:
                pace_request(len(points))
                with requests.get(source["url"], params=params, timeout=(10, 35)) as response:
                    if response.status_code == 429:
                        retry_after = response.headers.get("Retry-After")
                        delay = max(62., float(retry_after)) if retry_after and retry_after.isdigit() else 62.
                        pace_request(len(points), cooldown=delay)
                        print(f"Public archive rate limit. Cooling down {delay:.0f}s before retry.", flush=True)
                    if response.status_code == 400:
                        body = response.json()
                        if body.get("reason") != "No data is available for this location":
                            raise ValueError(f"Rejected archive request: {body.get('reason', body)}")
                    else:
                        response.raise_for_status()
                        validated_rows(response.json(), source["variables"], source["units"], expected_time, len(points))
                    staged = path.with_suffix(".json.part")
                    staged.write_bytes(response.content)
                    staged.replace(path)
                break
            except (requests.RequestException, json.JSONDecodeError):
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    data = json.loads(path.read_bytes())
    receipt = {
        "response": str(path.relative_to(STUDY)),
        "sha256": checksum(path),
        "request_url": requests.Request("GET", source["url"], params=params).prepare().url,
        "points": len(points),
    }
    if isinstance(data, dict) and data.get("reason") == "No data is available for this location":
        receipt["response_status"] = "no data for at least one requested location"
        if len(points) == 1:
            return [absent_row(source, len(expected_time))], [receipt]
        # A land/out-of-domain cell can reject a whole request. Isolate the
        # provider mask on the original grid and retain each raw response.
        mid = len(points) // 2
        groups = (points[:mid], points[mid:])
        responses = [
            fetch_task((path.with_stem(f"{path.stem}_{index}"), source, group, start, end))
            for index, group in enumerate(groups)
        ]
        return [row for rows, _ in responses for row in rows], [receipt] + [
            child for _, children in responses for child in children
        ]
    return validated_rows(data, source["variables"], source["units"], expected_time, len(points)), [receipt]


def vectors(speed, direction, *, wind_from):
    angle = np.radians(direction)
    sign = -1. if wind_from else 1.
    return sign * speed * np.sin(angle), sign * speed * np.cos(angle)


def normalize(kind, source, rows, dates, latitudes, longitudes):
    shape = (len(dates), len(latitudes), len(longitudes))
    values = {
        variable: np.asarray(
            [row["hourly"][variable] for row in rows], dtype=np.float64
        ).T.reshape(shape)
        for variable in source["variables"]
    }
    target_lat, target_lon = np.meshgrid(latitudes, longitudes, indexing="ij")
    source_lat = np.asarray([row["latitude"] for row in rows], dtype=float).reshape(target_lat.shape)
    source_lon = np.asarray([row["longitude"] for row in rows], dtype=float).reshape(target_lon.shape)
    dy = 111_320. * (source_lat - target_lat)
    dx = 111_320. * np.cos(np.radians(target_lat)) * (source_lon - target_lon)
    distance = np.hypot(dx, dy)
    maximum_distance = 10_000 if kind == "wave" else 15
    if np.any(distance > maximum_distance):
        raise ValueError(f"{kind} source point is too far from its target cell ({np.nanmax(distance):.1f} m).")

    columns = tuple(values[name] for name in source["variables"])
    if kind == "wave":
        height, direction, period = columns
        fields = {"VHM0": height, "VMDR": direction % 360., "VTPK": period}
        valid = np.all(np.isfinite(columns), axis=0) & (height >= 0.) & (period > 0.)
    else:
        speed, direction = columns
        east, north = vectors(speed, direction, wind_from=(kind == "wind"))
        names = ("u10", "v10") if kind == "wind" else ("water_u", "water_v")
        fields = dict(zip(names, (east, north), strict=True))
        valid = np.all(np.isfinite(columns), axis=0) & (speed >= 0.)
    complete = np.all(valid, axis=0)
    if not np.any(complete) or (kind == "wind" and not np.all(valid)):
        raise ValueError(f"{kind} is missing required spatial and temporal coverage.")
    dims = ("time", "latitude", "longitude")
    ds = xr.Dataset(
        {
            name: (dims, np.where(valid, field, np.nan).astype("float32"),
                   dict(zip(("units", "standard_name"), CONVENTIONS[name], strict=True)))
            for name, field in fields.items()
        },
        coords={
            "time": dates.astype("datetime64[ns]"),
            "latitude": ("latitude", latitudes, {"units": "degrees_north"}),
            "longitude": ("longitude", longitudes, {"units": "degrees_east"}),
            "source_latitude": (("latitude", "longitude"), source_lat, {"units": "degrees_north"}),
            "source_longitude": (("latitude", "longitude"), source_lon, {"units": "degrees_east"}),
        },
        attrs={
            "provider": source["provider"], "model": source["model"],
            "source_endpoint": source["url"], "citation": source["citation"],
            "coordinate_mapping": source["coordinate_mapping"],
            "source_time": "hourly valid time, UTC; archive series, forecast issue times are unavailable",
            "missing": "missing marine cells/times preserved as NaN; no imputation in acquisition",
        },
    )
    return ds, {
        "grid_shape": list(shape),
        "latitude_range": [float(latitudes.min()), float(latitudes.max())],
        "longitude_range": [float(longitudes.min()), float(longitudes.max())],
        "max_source_target_distance_m": float(np.nanmax(distance)),
        "provider_rejected_grid_cells": int(np.isnan(distance).sum()),
        "complete_grid_cells": int(complete.sum()),
        "ever_valid_grid_cells": int(valid.any(axis=0).sum()),
        "valid_fraction": float(valid.mean()),
        "hours_with_a_valid_cell": int(valid.any(axis=(1, 2)).sum()),
    }


def seasonal_block(season, workers):
    first = SEASONS[season]
    through = first + timedelta(days=7)
    start = first - timedelta(days=1)
    end = through + timedelta(days=1)
    times = hourly_range(start, end)
    target = STUDY / "forcing" / season
    raw = STUDY / "forcing_raw" / season
    raw.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    manifest = {
        "season": season, "core_start_utc": f"{first}T00:00:00Z", "core_end_exclusive_utc": f"{through}T00:00:00Z",
        "coverage_first_utc": f"{times[0]}:00:00Z", "coverage_last_utc": f"{times[-1]}:00:00Z",
        "requested_bounds": {"latitude": BOUNDS[0], "longitude": BOUNDS[1]},
        "source_utc_step_hours": 1, "data": {},
    }
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for kind, source in SOURCES.items():
            latitudes, longitudes = (axis(bounds, source["spacing"], source["offset"]) for bounds in BOUNDS)
            points = list(product(latitudes, longitudes))
            mask = np.zeros(len(points), dtype=bool)
            reused_mask = None
            if kind == "wave" and season != "winter":
                reference_wave = STUDY / "forcing" / "winter" / "wave.nc"
                with xr.open_dataset(reference_wave) as reference:
                    if not (np.array_equal(reference.latitude, latitudes)
                            and np.array_equal(reference.longitude, longitudes)
                            and reference.attrs["model"] == source["model"]):
                        raise ValueError("The prior winter location mask uses a different grid/provider.")
                    mask = np.isnan(reference.source_latitude.values).reshape(-1)
                if np.any(mask):
                    reused_mask = {
                        "method": "same provider, same grid; reuse explicit winter no-data-location mask",
                        "reference_file": str(reference_wave.relative_to(STUDY)),
                        "reference_sha256": checksum(reference_wave),
                        "skipped_grid_cells": int(mask.sum()),
                    }
            active = [point for point, rejected in zip(points, mask, strict=True) if not rejected]
            tasks = [
                (raw / f"{kind}_{offset//POINTS_PER_REQUEST:03d}.json", source,
                 active[offset:offset+POINTS_PER_REQUEST], start, end)
                for offset in range(0, len(active), POINTS_PER_REQUEST)
            ]
            responses = list(pool.map(fetch_task, tasks))
            returned = iter(row for response, _ in responses for row in response)
            rows = [absent_row(source, len(times)) if rejected else next(returned) for rejected in mask]
            ds, stats = normalize(kind, source, rows, times, latitudes, longitudes)
            path = target / f"{kind}.nc"
            staged = path.with_suffix(".nc.part")
            ds.to_netcdf(staged, engine="netcdf4", encoding={
                name: {"zlib": True, "complevel": 3} for name in ds.data_vars
            })
            staged.replace(path)
            manifest["data"][kind] = {
                **source, **stats,
                "path": str(path.relative_to(STUDY)), "sha256": checksum(path),
                "reused_provider_location_mask": reused_mask,
                "responses": [meta for _, receipt in responses for meta in receipt],
            }
            print(f"{season} {kind}: {len(times)} h, {stats['complete_grid_cells']}/{len(points)} complete cells", flush=True)
    manifest["confirmed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(target / "acquisition.json", manifest)
    return manifest


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--season", choices=tuple(SEASONS), action="append")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    for season in args.season or SEASONS:
        seasonal_block(season, args.workers)
    manifests = {}
    for season in SEASONS:
        path = STUDY / "forcing" / season / "acquisition.json"
        if path.exists():
            manifests[season] = {"path": str(path.relative_to(STUDY)), "sha256": checksum(path)}
    write_json(STUDY / "forcing" / "acquisition_index.json", {
        "archive_interfaces": ["https://open-meteo.com/en/docs/historical-weather-api",
                               "https://open-meteo.com/en/docs/marine-weather-api"],
        "seasons": manifests,
    })


if __name__ == "__main__":
    main()
