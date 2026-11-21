"""Build grouped external mark-entry cases from every saved GPX leg and local archive forcing."""
from __future__ import annotations

import csv
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import runpy
import warnings

import numpy as np

from environment import Forcing, INTERVAL_SECONDS, Scenario, save_scenarios
from world._geo import haversine_metres, local_east_north_metres
from world.boat import BoatState
from world.currents import Currents
from world.drift_guidance import DriftGuidance
from world.fossen import Fossen3DOFSaildroneDynamics
from world.learned_mark_policy import MAX_SECONDS, PolarSnapshot
from world.scenario import Route
from world.waves import Waves
from world.weather import Weather, Wind
from world.world import WorldObservation


STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[2]
PRIOR = STUDY.with_name("wind_direction_ablation_20261118")
JULY_BLOCK = "2026-07-29T06:00:00Z"
WIND_FROM = (None, 355, 5, 85, 95, 175, 185, 265, 275)
CHANGE_AT = 3 * 3600
SYNTHETIC_RANGE_METRES = (200.0, 2450.0)
ORIGINAL_HEADING = 45.0
TIMES = (("wind", "valid_time", 3), ("current", "time", 3), ("wave", "time", 1))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def route_point(point):
    return point.latitude, point.longitude


def utc(when):
    return np.datetime_as_string(when, unit="s") + "Z"


def bracket(axis, when, max_gap_hours):
    index = np.searchsorted(axis, when)
    if index < len(axis) and axis[index] == when:
        return [utc(when), utc(when)]
    if index == 0 or index == len(axis):
        raise ValueError(f"{utc(when)} outside archive coverage")
    before, after = axis[index - 1], axis[index]
    if after - before > np.timedelta64(max_gap_hours, "h"):
        raise ValueError(f"{utc(when)} lies in the archive gap {utc(before)} to {utc(after)}")
    return [utc(before), utc(after)]


def force_from_observation(sampler, position, when, sampled, wind_direction):
    native, current, wave = sampled
    wind = native
    if wind_direction is not None:
        angle = np.radians(wind_direction)
        wind = Wind(float(-native.speed*np.sin(angle)), float(-native.speed*np.cos(angle)),
                    native.speed, float(wind_direction))
    observation = WorldObservation(
        when, BoatState(*position, ORIGINAL_HEADING, 0.0), wind, wave, current,
        None, None, None, (),
    )
    # Match the reduced synthetic environment's compression of the steady Fossen
    # catalogue while feeding the actual archive wave direction and peak period.
    by_bow = {}
    for mode in sampler._steady_modes(observation):
        by_bow.setdefault(mode.bow, []).append(mode)
    flow = np.array((current.u, current.v))
    selected = [mode for bow in sorted(by_bow) for mode in sorted(
        by_bow[bow], key=lambda entry: -np.linalg.norm(entry.velocity - flow))[:2]]
    if not selected:
        raise ValueError(f"No steady Fossen trim at wind {wind.speed:.3f} m/s, {utc(when)}")
    result = Forcing(PolarSnapshot.from_modes(selected, flow, wind.direction), wind.speed, wave.height)
    return result, len(selected)


def load_routes():
    known = runpy.run_path(str(PRIOR / "reproduce/run_study.py"))["ROUTES"]
    expected_hashes = json.loads((PRIOR / "input_sha256.json").read_text())
    cases = {case.file: case for case in known.values()}
    files = sorted((ROOT / "current_route").glob("*.gpx"))
    if {path.name for path in files} != set(cases):
        raise ValueError("The actual GPX folder differs from the historical nine-route matrix.")
    routes, errors = {}, []
    for path in files:
        key = path.relative_to(ROOT).as_posix()
        checksum = digest(path)
        if checksum != expected_hashes[key]:
            raise ValueError(f"The historical route has changed: {key}")
        try:
            route = Route.load_gpx(path)
        except (OSError, ValueError) as error:
            errors.append({"gpx": path.name, "sha256": checksum, "error": str(error)})
        else:
            routes[path.name] = (route, cases[path.name], checksum)
    return routes, errors


def chart_captures():
    path = PRIOR / "route_chart_preflight.csv"
    with path.open(newline="") as stream:
        return {
            (row["route_file"], row["waypoint"]): {
                "cell_blocked": row["cell_blocked"] == "True",
                "complete_capture_disk_in_blocked_cell": row["complete_capture_disk_in_blocked_cell"] == "True",
                "cell_elevation_m": float(row["chart_elevation_m"]),
            }
            for row in csv.DictReader(stream)
        }


def archive_source_arrays(providers):
    results = {}
    for name, provider in providers.items():
        dataset = provider._require_dataset()
        arrays = {}
        for key in sorted(dataset.variables):
            item = dataset[key]
            arr = np.asarray(item.values)
            if arr.dtype.kind == "O":
                arr = arr.astype(str)
            packed = np.ascontiguousarray(arr)
            arrays[key] = {
                "dims": list(item.dims), "dtype": arr.dtype.str, "shape": list(arr.shape),
                "sha256": hashlib.sha256(memoryview(packed.view(np.uint8)).cast("B")).hexdigest(),
            }
        results[name] = {
            "arrays": arrays, "sha256": hashlib.sha256(json.dumps(arrays, sort_keys=True).encode()).hexdigest(),
        }
    return results


def run():
    routes, errors = load_routes()
    charts = chart_captures()
    all_positions = np.array([route_point(point) for route, _, _ in routes.values() for point in route.waypoints])
    south, west = all_positions.min(axis=0) - 0.5
    north, east = all_positions.max(axis=0) + 0.5
    bounds = {"latitude_bounds": (float(south), float(north)), "longitude_bounds": (float(west), float(east))}
    providers = {name: factory(**bounds).load() for name, factory in
                 (("wind", Weather), ("current", Currents), ("wave", Waves))}
    axes = {name: providers[name]._require_dataset().coords[key].values.astype("datetime64[ns]")
            for name, key, _ in TIMES}
    source_arrays = archive_source_arrays(providers)
    sampler = DriftGuidance(Fossen3DOFSaildroneDynamics(), 45.0)
    offsets = np.array([45.5, 52.0, 67.0, 85.0, 105.0, 128.0, 152.0, 175.0])
    sampler.offsets = np.r_[-offsets, offsets]
    sampler.flaps = sampler.flaps[[0, 3, 6]]

    @lru_cache(maxsize=None)
    def archived_sample(position, when):
        brackets = {name: bracket(axes[name], when, max_gap) for name, _, max_gap in TIMES}
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            values = tuple(providers[name].sample(*position, when) for name, _, _ in TIMES)
        scope = {"sample_utc": utc(when), "brackets_utc": brackets,
                 **{key: asdict(value) for (key, _, _), value in zip(TIMES, values, strict=True)},
                 "provider_warnings": [str(warning.message) for warning in records]}
        return values, scope

    @lru_cache(maxsize=None)
    def polar(position, when, direction):
        values, _ = archived_sample(position, when)
        return force_from_observation(sampler, position, when, values, direction)

    groups, layers, records, scenarios = {}, {}, {}, []
    for name, (route, declaration, checksum) in sorted(routes.items()):
        for block, departure_string in (("original", declaration.departure), ("july_overlap", JULY_BLOCK)):
            stamp = np.datetime64(departure_string.removesuffix("Z"), "ns")
            future_stamp = stamp + np.timedelta64(CHANGE_AT, "s")
            group = slug(Path(name).stem) + "__" + block
            groups[group] = {"gpx": name, "gpx_sha256": checksum, "archive_block": block,
                             "departure_utc": departure_string, "declared_radius_m": declaration.radius,
                             "original_native_cap_s": declaration.cap_s,
                             "target_legs": len(route.waypoints)-1}
            for index, (origin, mark) in enumerate(zip(route.waypoints[:-1], route.waypoints[1:], strict=True), start=1):
                position, destination = route_point(origin), route_point(mark)
                leg = f"{group}__leg{index:02d}"
                exact_range = haversine_metres(*position, *destination)
                projected = np.array(local_east_north_metres(*destination, *position))
                if np.linalg.norm(projected) > 0:
                    target = projected * (exact_range / np.linalg.norm(projected))
                else:
                    target = projected
                chart = charts[(name, mark.name)]
                layers[leg] = {"group": group, "leg_index": index, "origin_waypoint": origin.name,
                               "target_waypoint": mark.name, "origin_lat_lon": list(position),
                               "target_lat_lon": list(destination), "initial_leg_distance_m": exact_range,
                               "target_vector_east_north_m": target.tolist(), "target_chart_preflight": chart,
                               "range_outside_synthetic_support": not SYNTHETIC_RANGE_METRES[0] <= exact_range <= SYNTHETIC_RANGE_METRES[1],
                               "radius_outside_synthetic_support": declaration.radius != 50}
                try:
                    _, first_scope = archived_sample(position, stamp)
                    _, future_scope = archived_sample(position, future_stamp)
                except ValueError as error:
                    layers[leg]["missing_archive"] = str(error)
                else:
                    layers[leg].update(first_forcing=first_scope, future_forcing=future_scope)
                for direction in WIND_FROM:
                    variant = "native" if direction is None else f"{direction:03d}"
                    identifier = f"{leg}__wind_{variant}"
                    info = {"leg": leg, "group": group, "wind_variant": variant, "status": "generated"}
                    records[identifier] = info
                    if "missing_archive" in layers[leg]:
                        info.update(status="unavailable_archive", detail=layers[leg]["missing_archive"])
                        continue
                    try:
                        first, first_count = polar(position, stamp, direction)
                        future, future_count = polar(position, future_stamp, direction)
                    except ValueError as error:
                        info.update(status="unavailable_steady_trim", detail=str(error))
                        continue
                    info.update(first_trim_count=first_count, future_trim_count=future_count,
                                first_wind_from_deg=first.polar.wind_from, future_wind_from_deg=future.polar.wind_from,
                                low_trim_support=min(first_count, future_count) < 4)
                    scenarios.append(Scenario(identifier, "archive_route", "external", target.copy(),
                                              ORIGINAL_HEADING, float(declaration.radius), first, future, float(CHANGE_AT)))
    for provider in providers.values():
        provider.close()
    output = STUDY / "archive_scenarios.json.gz"
    temporary = output.with_suffix(".tmp")
    save_scenarios(temporary, scenarios)
    temporary.replace(output)
    status_counts = {status: sum(row["status"] == status for row in records.values())
                     for status in sorted({row["status"] for row in records.values()})}
    manifest = {
        "evaluation_scope": "Grouped external transfer on independent saved GPX legs. No route chaining or use in fitting.",
        "archive_scope": "Wind, current and sea state at each leg origin at the declared start and three hours later. "
                         "The reduced plant holds the first snapshot for 3 h and the future snapshot for the rest of the "
                         "6 h cap. Each subsequent GPX leg starts at its own preceding point on the same block clock.",
        "chart_scope": "GEBCO waypoint-cell flags are retained for stratification. "
                       "Raw scenarios use only the reduced dynamics; the chart is not integrated here.",
        "temporal_selection": "July 29 06:00 UTC and 09:00 UTC are shared exact valid times for GFS, HYCOM and "
                              "Copernicus archives. Original-route blocks keep their documented departure and +3 h. "
                              "Interpolation across a larger gap than 3 h wind/current or 1 h waves is rejected.",
        "traffic_scope": "No contacts in these base cases. Matched modeled traffic treatments may be paired separately by ID.",
        "reduced_control_interval_s": INTERVAL_SECONDS, "change_at_s": CHANGE_AT, "reduced_cap_s": MAX_SECONDS,
        "wind_bearings_from_deg": WIND_FROM, "reduced_training_initial_leg_range_m": SYNTHETIC_RANGE_METRES,
        "archive_crop": bounds, "archive_array_fingerprints": source_arrays,
        "route_parse_errors": errors, "routes_declared": len(routes) + len(errors),
        "groups": groups, "leg_blocks": layers, "rows": records,
        "status_counts": status_counts, "scenario_file": output.name, "scenario_sha256": digest(output),
        "source_sha256": digest(Path(__file__).resolve()),
        "reference_protocol_source_sha256": digest(PRIOR / "reproduce/run_study.py"),
        "reference_chart_preflight_sha256": digest(PRIOR / "route_chart_preflight.csv"),
    }
    path = STUDY / "archive_scenario_manifest.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(f"{len(groups)} route blocks, {len(layers)} leg blocks, {len(records)} scenario rows: {status_counts}.", flush=True)


if __name__ == "__main__":
    run()
