"""Verify seasonal inputs, routes and sampled wet drift margins."""

from argparse import ArgumentParser
from hashlib import file_digest
from math import cos, radians, sin
from pathlib import Path
import json

import numpy as np
import xarray as xr

from seasonal_environment import SeasonalArchive
from world._geo import offset_latitude_longitude
from world.bathymetry import Bathymetry


STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[2]
SEASONS = ("winter", "spring", "summer", "autumn")
VARIABLES = {
    "wind": ("u10", "v10"),
    "current": ("water_u", "water_v"),
    "wave": ("VHM0", "VMDR", "VTPK"),
}


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def validate_sources(season, registered):
    folder = STUDY / "forcing" / season
    manifest = json.loads((folder / "acquisition.json").read_text())
    start = np.datetime64(registered["requested_archive_start_utc"].removesuffix("Z"), "h")
    end = np.datetime64(registered["requested_archive_end_utc_exclusive"].removesuffix("Z"), "h")
    calendar = np.arange(start, end)
    results = {}
    for kind, names in VARIABLES.items():
        receipt = manifest["data"][kind]
        path = STUDY / receipt["path"]
        if digest(path) != receipt["sha256"]:
            raise ValueError(f"{path} differs from its acquisition record.")
        mask = receipt.get("reused_provider_location_mask")
        if mask and digest(STUDY / mask["reference_file"]) != mask["reference_sha256"]:
            raise ValueError(f"{season} wave location mask differs from its winter reference.")
        for row in receipt["responses"]:
            if digest(STUDY / row["response"]) != row["sha256"]:
                raise ValueError(f"Raw response {row['response']} differs from its acquisition record.")
        with xr.open_dataset(path) as data:
            times = data.time.values.astype("datetime64[h]")
            if not np.array_equal(times, calendar):
                raise ValueError(f"{path} misses one or more registered UTC hours.")
            values = np.stack([data[name].transpose("time", "latitude", "longitude").values for name in names], axis=-1)
            valid = np.isfinite(values).all(axis=-1)
            if not valid.any(axis=(1, 2)).all():
                raise ValueError(f"{path} has hours without any complete grid cell.")
            if kind == "wave" and (np.any(values[..., 0][valid] < 0) or np.any(values[..., 2][valid] <= 0)):
                raise ValueError(f"{path} contains invalid wave height or period.")
            locations = np.stack((data.source_latitude.values, data.source_longitude.values), axis=-1)
            unique = np.unique(locations[valid.all(axis=0)], axis=0)
            results[kind] = {
                "hours": len(times), "complete_cells": int(valid.all(axis=0).sum()),
                "cells_ever_valid": int(valid.any(axis=0).sum()),
                "fully_valid_distinct_native_locations": len(unique),
                "variables": {name: data[name].attrs["units"] for name in names},
                "raw_response_receipts": len(receipt["responses"]),
                "source_sha256": receipt["sha256"],
            }
    return calendar, results


def route_points(registered):
    families, points = {}, set()
    for route in registered.values():
        vertices = [(row["latitude"], row["longitude"]) for row in route["waypoints"]]
        families.setdefault(route["family"], set()).update(vertices)
        points.update(vertices)
        for start, end in zip(vertices, vertices[1:]):
            points.update(tuple((1 - fraction) * left + fraction * right
                                for left, right in zip(start, end, strict=True))
                          for fraction in (.25, .5, .75))
    return families, points


def drift_points(families):
    path = ROOT / "data/bathymetry/gebco_2026_n44.0_s37.0_w14.0_e22.0_geotiff.tif"
    bathymetry = Bathymetry(path).load()
    wet, dry = set(), 0
    try:
        for group in families.values():
            centre = np.mean(sorted(group), axis=0)
            for distance in (12_500., 25_000.):
                for bearing in np.arange(0., 360., 30.):
                    angle = radians(bearing)
                    pos = offset_latitude_longitude(*centre, distance * sin(angle), distance * cos(angle))
                    if bathymetry.sample(*pos).is_land:
                        dry += 1
                    else:
                        wet.add(pos)
    finally:
        bathymetry.close()
    return wet, dry


def verify(season, registered, sample_positions):
    calendar, sources = validate_sources(season, registered)
    environment = SeasonalArchive(STUDY / "forcing" / season)
    times = calendar.astype("datetime64[m]")
    queries = np.sort(np.r_[times, times[:-1] + np.timedelta64(30, "m")])
    minimum = np.full(4, np.inf)
    maximum = np.full(4, -np.inf)
    for point in sorted(sample_positions):
        for now in queries:
            w = environment.at(*point, now)
            values = np.array((w.wind.speed, w.current.speed, w.wave.height, w.wave.peak_period))
            if not np.isfinite(values).all():
                raise ValueError(f"Missing forcing in {season} on {now} at {point}.")
            minimum = np.minimum(minimum, values)
            maximum = np.maximum(maximum, values)
    return {
        "field_checks": sources, "query_times": len(queries),
        "sample_positions": len(sample_positions), "interpolation_checks": len(queries) * len(sample_positions),
        "range_sampled": {name: [float(low), float(high)] for name, low, high in zip(
            ("wind_speed_mps", "current_speed_mps", "significant_wave_height_m", "wave_peak_period_s"),
            minimum, maximum, strict=True,
        )},
        "nearest_wet_fallbacks": environment.coastal_fallbacks(),
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--season", choices=SEASONS, action="append")
    args = parser.parse_args()
    registered = json.loads((STUDY / "manifest.json").read_text())
    families, route = route_points(registered["routes"])
    drift, dry = drift_points(families)
    requested = args.season or SEASONS
    result = {}
    for season in requested:
        result[season] = verify(season, registered["archive_blocks"][season], route | drift)
        print(season, result[season]["interpolation_checks"], "valid route/drift samples", flush=True)
    suffix = "" if tuple(requested) == SEASONS else "_" + "_".join(requested)
    receipt = STUDY / f"forcing_validation{suffix}.json"
    receipt.write_text(json.dumps({
        "route_points": len(route), "wet_drift_ring_points": len(drift), "dry_ring_points": dry,
        "drift_ring_metres": [12_500, 25_000], "seasons": result,
    }, indent=2, allow_nan=False) + "\n")
