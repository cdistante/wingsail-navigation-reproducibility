"""Verify seasonal archive inputs and locked episode assignments independently."""
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from hashlib import file_digest, sha256
import json
from math import dist
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import xarray as xr

from scenario_environment import CHART_MINIMUM_DEPTH_M, CHART_PATH, ScenarioEnvironment
from seasonal_environment import SeasonalArchive
from world.boat import BoatState


STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[2]
SEASONS = ("winter", "spring", "summer", "autumn")
MODELS = {"wind": "era5", "current": "meteofrance_currents", "wave": "ecmwf_wam"}
FIELDS = {
    "wind": {"u10": ("m s-1", "eastward_wind"), "v10": ("m s-1", "northward_wind")},
    "current": {"water_u": ("m s-1", "eastward_sea_water_velocity"),
                "water_v": ("m s-1", "northward_sea_water_velocity")},
    "wave": {"VHM0": ("m", "sea_surface_wave_significant_height"),
             "VMDR": ("degree", "sea_surface_wave_from_direction"),
             "VTPK": ("s", "sea_surface_wave_period_at_variance_spectral_density_maximum")},
}


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def moment(value):
    return np.datetime64(str(value).removesuffix("Z"), "ns")


def timestamp(value):
    return np.datetime_as_string(value, unit="s") + "Z"


def route_marks(path):
    points = ET.parse(path).findall(".//{*}rtept")
    return [{"waypoint": entry.findtext("{*}name"), "latitude": float(entry.attrib["lat"]),
             "longitude": float(entry.attrib["lon"])} for entry in points]


def verify_manifests():
    names = ("manifest.json", "manifest_expanded.json")
    manifests = [read_json(STUDY / name) for name in names]
    pilot, expanded = [{row["id"]: row for row in manifest["scenarios"]} for manifest in manifests]
    require(all(expanded.get(key) == value for key, value in pilot.items()), "Expanded roster changed a pilot case.")
    source = digest(STUDY / "reproduce/make_manifest.py")
    coverage = {}
    for name, manifest in zip(names, manifests, strict=True):
        require(manifest["manifest_source_sha256"] == source, f"{name}: generator hash changed.")
        routes, blocks, profiles = [manifest[key] for key in ("routes", "archive_blocks", "traffic_profiles")]
        require(set(blocks) == set(SEASONS), "Calendar requires all four seasons.")
        roles = {season: block["split"] for season, block in blocks.items()}
        require(roles == {"winter": "train", "spring": "validation", "summer": "train", "autumn": "test"},
                f"{name}: seasonal roles were modified.")
        ordered = sorted(blocks.values(), key=lambda block: block["requested_archive_start_utc"])
        require(all(moment(left["requested_archive_end_utc_exclusive"]) <= moment(right["requested_archive_start_utc"])
                    for left, right in zip(ordered, ordered[1:])), "Season buffers overlap.")
        for route in routes.values():
            path = ROOT / route["route_gpx"]
            require(digest(path) == route["sha256"] and route_marks(path) == route["waypoints"],
                    f"GPX bytes or exact ordered mark list changed: {path.name}.")
        require({r["route_gpx"] for r in routes.values() if r["family"] == "brindisi"} == {
            "current_route/small route.gpx", "current_route/opencpn_001_003_test.gpx",
            "current_route/opencpn_001_004_test.gpx"}, "Brindisi holdout family is incomplete.")
        require(len({row["id"] for row in manifest["scenarios"]}) == len(manifest["scenarios"]),
                "Scenario IDs overlap.")
        counts = Counter(row["split"] for row in manifest["scenarios"])
        require(counts == manifest["scenario_counts"], "Role counters disagree.")
        expected_counts = {"train": 16, "validation": 6, "test": 12} if name.endswith("_expanded.json") else {
            "train": 8, "validation": 3, "test": 8}
        require(counts == expected_counts, "Registered roster count changed.")
        families = {role: sorted({row["route_family"] for row in manifest["scenarios"] if row["split"] == role})
                    for role in counts}
        require(families == {"train": ["otranto"], "validation": ["otranto"], "test": ["brindisi"]},
                "Held-out route family overlaps model development.")
        signatures, roles_by_traffic, test_groups = {}, {}, Counter()
        for row in manifest["scenarios"]:
            route = routes[row["route_id"]]
            block = blocks[row["archive_block"]]
            recipe = profiles[row["traffic_profile"]]
            role = row["split"]
            require(role == block["split"] == recipe["split"] and row["route_family"] == route["family"]
                    and row["route_gpx"] == route["route_gpx"], "A scenario crosses a locked split.")
            require(moment(block["start_utc"]) <= moment(row["departure_utc"]) <= moment(row["weather_coverage_through_utc"])
                    < moment(block["end_utc_exclusive"]), "An episode escapes its core weather block.")
            require(row["expected_captures"] == [{"phase": "outbound", "waypoint": x["waypoint"]}
                                                 for x in route["waypoints"][1:]], "Incorrect ordered GPX targets.")
            require(row["departure"] == route["waypoints"][0] and row["capture_radius_m"] == 50.
                    and row["control_dt_s"] == 5 and row["max_steps"] * row["control_dt_s"] == 21600
                    and row["return_to_start"] is False and row["wind_from_override_deg"] is None,
                    "Native conditions or capture specification changed.")
            shapes = [{key: value for key, value in target.items() if key != "name"} for target in recipe["targets"]]
            signature = sha256(json.dumps((shapes, recipe.get("velocity_ramps", ())), sort_keys=True).encode()).hexdigest()
            roles_by_traffic.setdefault(signature, set()).add(role)
            signatures[row["traffic_profile"]] = signature
            if role == "test":
                test_groups[row["route_id"], row["departure_utc"]] += 1
                require(row["pairing"] == ["physics", "seasonal_learner"], "An external test cell is unpaired.")
        require(all(len(roles) == 1 for roles in roles_by_traffic.values()), "An exact traffic geometry crosses roles.")
        require(all(value == 2 for value in test_groups.values()), "Test route/departure groups require two profiles.")
        coverage[name] = {"sha256": digest(STUDY / name), "scenario_counts": dict(counts), "route_families": families,
                          "traffic_geometry_sha256": signatures, "test_route_time_groups": len(test_groups)}
    return manifests, coverage


def emitted_rows(receipt, calendar):
    stations = {}
    for source in receipt["responses"]:
        path = STUDY / source["response"]
        require(digest(path) == source["sha256"], f"Raw archive response changed: {path.name}.")
        payload = read_json(path)
        rows = payload if isinstance(payload, list) else [payload]
        for row in rows:
            if "hourly" not in row:
                require(source.get("response_status"), f"Unknown raw response status for {path.name}.")
                continue
            stamp = np.asarray(row["hourly"]["time"], dtype="datetime64[ns]")
            require(row["utc_offset_seconds"] == 0 and np.array_equal(stamp, calendar),
                    f"Non-UTC or incomplete raw sample {path.name}.")
            for variable, unit in zip(receipt["variables"], receipt["units"], strict=True):
                require(row["hourly_units"].get(variable) == unit, f"Raw provider unit mismatch: {variable}.")
            key = float(row["latitude"]), float(row["longitude"])
            values = tuple(np.asarray(row["hourly"][name], dtype=float) for name in receipt["variables"])
            if key in stations:
                require(all(np.array_equal(x, y, equal_nan=True) for x, y in zip(stations[key], values, strict=True)),
                        f"Provider returned inconsistent duplicate station values at {key}.")
            stations[key] = values
    return stations


def expected_fields(kind, raw):
    if kind == "wave":
        height, direction, period = raw
        good = np.isfinite(height + direction + period) & (height >= 0) & (period > 0)
        values = height, direction % 360., period
    else:
        speed, direction = raw
        good = np.isfinite(speed + direction) & (speed >= 0)
        angle = np.radians(direction)
        sign = -1. if kind == "wind" else 1.
        values = sign * speed * np.sin(angle), sign * speed * np.cos(angle)
    return [np.where(good, value, np.nan) for value in values]


def verify_block(season, registered, bounds):
    folder = STUDY / "forcing" / season
    source_manifest = folder / "acquisition.json"
    receipt = read_json(source_manifest)
    start, end = [moment(registered[name]) for name in (
        "requested_archive_start_utc", "requested_archive_end_utc_exclusive")]
    core_start, core_end = [moment(registered[name]) for name in ("start_utc", "end_utc_exclusive")]
    time = np.arange(start, end, np.timedelta64(1, "h"), dtype="datetime64[ns]")
    require(len(time) == 240 and sum((time >= core_start) & (time < core_end)) == 168
            and sum(time < core_start) == 24 and sum(time >= core_end) == 48, "Unexpected archive hours.")
    require(receipt["season"] == season and moment(receipt["core_start_utc"]) == core_start
            and moment(receipt["core_end_exclusive_utc"]) == core_end
            and moment(receipt["coverage_first_utc"]) == time[0] and moment(receipt["coverage_last_utc"]) == time[-1]
            and receipt["source_utc_step_hours"] == 1, f"{season}: acquisition dates disagree.")
    require(receipt["requested_bounds"] == {
        "latitude": [bounds["min_latitude"], bounds["max_latitude"]],
        "longitude": [bounds["min_longitude"], bounds["max_longitude"]],
    }, f"{season}: provider crop disagrees with the registration.")
    result = {}
    for kind, members in FIELDS.items():
        path = STUDY / registered["files"][kind]
        entry = receipt["data"][kind]
        require(entry["path"] == registered["files"][kind] and entry["model"] == MODELS[kind]
                and entry["sha256"] == digest(path), f"{season}/{kind}: source provenance changed.")
        mask = entry.get("reused_provider_location_mask")
        if mask is not None:
            require(digest(STUDY / mask["reference_file"]) == mask["reference_sha256"],
                    f"{season}/{kind}: reused provider mask source changed.")
        station = emitted_rows(entry, time)
        with xr.open_dataset(path) as dataset:
            source_times = dataset["time"].values.astype("datetime64[ns]")
            require(np.array_equal(time, source_times), f"{season}/{kind}: incomplete buffered hourly coverage.")
            require(dataset.attrs["model"] == MODELS[kind] and dataset.attrs["provider"] == entry["provider"],
                    f"{season}/{kind}: inconsistent archived provider attribution.")
            for axis in ("latitude", "longitude"):
                coordinate = dataset[axis].values
                require(np.all(np.diff(coordinate) > 0.) and
                        coordinate[0] <= bounds[f"min_{axis}"] <= bounds[f"max_{axis}"] <= coordinate[-1],
                        f"{season}/{kind}: missing or unordered registered {axis} extent.")
            for name, (units, standard_name) in members.items():
                field = dataset[name]
                require(field.dims == ("time", "latitude", "longitude")
                        and field.attrs["units"] == units and field.attrs["standard_name"] == standard_name,
                        f"{season}/{kind}: source convention mismatch for {name}.")
            values = np.stack([dataset[name].values for name in members], axis=-1)
            valid = np.isfinite(values).all(axis=-1)
            require(valid.any(axis=(1, 2)).all(), f"{season}/{kind}: time has no wet/valid grid cell.")
            require(entry["grid_shape"] == list(values.shape[:-1])
                    and entry["complete_grid_cells"] == int(valid.all(axis=0).sum())
                    and entry["hours_with_a_valid_cell"] == int(valid.any(axis=(1, 2)).sum()),
                    f"{season}/{kind}: cell/hour counters disagree.")
            inspected = 0
            for j, i in zip(*np.nonzero(valid.any(axis=0)), strict=True):
                key = float(dataset.source_latitude.values[j, i]), float(dataset.source_longitude.values[j, i])
                require(key in station, f"{season}/{kind}: normalized grid cell is missing its raw station {key}.")
                physical = np.column_stack(expected_fields(kind, station[key]))
                require(np.allclose(values[:, j, i], physical, atol=1e-6, rtol=1e-6, equal_nan=True),
                        f"{season}/{kind}: normalized physical values/compass direction differ at {key}.")
                inspected += len(time)
            result[kind] = {"normalized_sha256": entry["sha256"], "provider": entry["provider"], "model": MODELS[kind],
                            "grid_shape": list(values.shape[:-1]), "buffered_hours": len(time),
                            "core_hours": 168, "raw_response_files": len(entry["responses"]),
                            "provider_to_grid_cell_hours_verified": inspected, "units_standard_names": members}
    return {"coverage_first_utc": timestamp(time[0]), "coverage_last_utc": timestamp(time[-1]),
            "acquisition_sha256": digest(source_manifest), "sources": result}


def independent_receipt_epoch(profile, elapsed):
    available = elapsed - profile["delay_s"]
    cadence = profile["report_interval_s"]
    epoch = np.floor(available / cadence) * cadence
    while any(start <= epoch < stop for start, stop in profile["drop_windows_s"]):
        epoch -= cadence
    return float(epoch)


def verify_scene(manifest, archives, scenario):
    route = manifest["routes"][scenario["route_id"]]
    block = scenario["archive_block"]
    archive = archives[block]
    environment = ScenarioEnvironment.from_manifest(archive, manifest, scenario)
    positions = [(point["latitude"], point["longitude"]) for point in route["waypoints"]]
    midpoints = [tuple((a + b) / 2. for a, b in zip(left, right, strict=True))
                 for left, right in zip(positions, positions[1:])]
    chart_path = np.concatenate([np.linspace(left, right, 101) for left, right in zip(positions, positions[1:])])
    require(environment.chart_path_is_clear(chart_path),
            f"GEBCO blocks a planned GPX leg in {scenario['id']}.")
    sampled = []
    for minute in (0, 30, 180, 360):
        instant = environment.start_utc + np.timedelta64(minute, "m")
        for origin in (*positions, *midpoints):
            boat = BoatState(*origin, scenario["initial_heading_deg"], 0.)
            obs = environment.observe(boat, instant)
            w, v, h = obs.wind, obs.current, obs.wave
            values = (w.speed, v.speed, h.height, h.peak_period)
            require(all(np.isfinite(values)) and min(values[:3]) >= 0 and values[3] > 0,
                    f"Unavailable physical forcing at {scenario['id']} {timestamp(instant)} {origin}")
            require(abs(w.speed - dist((0, 0), (w.u, w.v))) < 1e-10 and
                    abs(v.speed - dist((0, 0), (v.u, v.v))) < 1e-10, "Wind/current scalar-vector magnitude mismatch.")
            require(np.allclose((-w.speed*np.sin(np.radians(w.direction)), -w.speed*np.cos(np.radians(w.direction))),
                                (w.u, w.v), atol=1e-10), "Wind must be meteorological FROM.")
            require(np.allclose((v.speed*np.sin(np.radians(v.direction)), v.speed*np.cos(np.radians(v.direction))),
                                (v.u, v.v), atol=1e-10), "Current course must point TO.")
            sampled.append(values)
    history = environment.weather_history(*environment.origin, environment.start_utc)
    require(history.shape == (12, 5) and np.isfinite(history).all()
            and np.allclose(history[-1], archive.at(*environment.origin, environment.start_utc).vector()),
            f"Incorrect causal weather history at {scenario['id']}.")
    profile = manifest["traffic_profiles"][scenario["traffic_profile"]]
    for elapsed in (-5, 0, 17, 110, 155, 175, 215, 250):
        epoch = independent_receipt_epoch(profile, elapsed)
        obs_time = environment.start_utc + np.timedelta64(elapsed, "s")
        report, truth = environment.reports_at(obs_time), environment.truth(epoch)
        require(len(report) == len(truth) == len(profile["targets"]), "An AIS contact was lost from its fixture.")
        require(all(a.report_age_seconds == elapsed - epoch and
                    (a.latitude, a.longitude, a.speed, a.course, a.mmsi) ==
                    (b.latitude, b.longitude, b.speed, b.course, b.mmsi)
                    for a, b in zip(report, truth, strict=True)), "The receiver anticipates a future or dropped AIS packet.")
    require(environment.max_target_acceleration_mps2 <= .026, "Fixture acceleration exceeds the planned AIS bound.")
    array = np.asarray(sampled)
    fields = ("wind_mps", "current_mps", "wave_height_m", "wave_peak_period_s")
    return {"forcing_samples": len(sampled), "ais_delivery_checks": 8, "geographic_chart_checks": len(chart_path),
            "range": {name: [float(low), float(high)] for name, low, high in
                      zip(fields, array.min(axis=0), array.max(axis=0), strict=True)}}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--season", choices=SEASONS, action="append",
                        help="Check available seasons and print a partial receipt without writing the full artifact.")
    args = parser.parse_args()
    manifests, coverage = verify_manifests()
    expanded = manifests[1]
    requested = tuple(dict.fromkeys(args.season or SEASONS))
    blocks, archives = {}, {}
    for season in requested:
        blocks[season] = verify_block(season, expanded["archive_blocks"][season], expanded["archive_bounds_request"])
        archives[season] = SeasonalArchive(STUDY / "forcing" / season)
        print(f"{season}: 168 core hours, 72 buffered hours, all three sources and raw direction conventions verified.", flush=True)
    scenes = {row["id"]: verify_scene(expanded, archives, row) for row in expanded["scenarios"] if row["archive_block"] in archives}
    result = {"status": "PASS", "archives_verified": len(blocks), "physical_files_verified": 3 * len(blocks),
              "manifests": coverage, "blocks": blocks, "expanded_cases_sampled": len(scenes),
              "planned_route_and_midpoint_forcing_samples": sum(item["forcing_samples"] for item in scenes.values()),
              "sampling_scope": "Independent declared departures plus 0.5, 3, and 6 h, over stored GPX vertices and leg midpoints.",
              "scenarios": scenes, "verifier_sha256": digest(Path(__file__).resolve()),
              "scenario_environment_sha256": digest(STUDY / "reproduce/scenario_environment.py"),
              "gebco_chart": {"path": str(CHART_PATH.relative_to(ROOT)), "sha256": digest(CHART_PATH),
                              "minimum_depth_m": CHART_MINIMUM_DEPTH_M,
                              "station_checks": sum(item["geographic_chart_checks"] for item in scenes.values())}}
    if args.season is None:
        destination = STUDY / "independent_seasonal_inputs.json"
        destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "archives_verified", "physical_files_verified",
                                                    "expanded_cases_sampled", "planned_route_and_midpoint_forcing_samples")}))


if __name__ == "__main__":
    main()
