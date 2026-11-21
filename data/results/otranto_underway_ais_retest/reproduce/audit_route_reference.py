"""Verify saved weak-wind Otranto routes independently of mission handoff.

The audit accepts a completed mission or a correctly recorded cap outcome.
It checks all saved timestamps and samples every environmental provider at
every saved position.  Wind speed comes from the archive; only its bearing
is replaced by the declared experimental FROM bearing.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import gzip
from hashlib import file_digest
import json
import math
from pathlib import Path
import sys
from xml.etree import ElementTree

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
sys.path.insert(0, str(ROOT))

from world import Currents, Waves, Weather
from world._geo import haversine_metres


DEPARTURE = datetime(2026, 9, 30, 23, 50, tzinfo=timezone.utc)
ENCOUNTER = DEPARTURE + timedelta(seconds=40)
STEP_SECONDS = 5
RADIUS = 50.0
MAXIMUM_STEPS = 17280
ROUTE_PATH = ROOT / "current_route/otranto_short_test_loop.gpx"
OTRANTO_BOUNDS = {"latitude_bounds": (39.5, 41.0), "longitude_bounds": (18.0, 19.5)}
ENVIRONMENT = {"wind": Weather, "wave": Waves, "current": Currents}


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def utc(value):
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def position(record):
    return record["latitude"], record["longitude"]


def gpx_position(node):
    return float(node.attrib["lat"]), float(node.attrib["lon"])


def one_json(lines, prefix):
    entry, = [line.removeprefix(prefix) for line in lines if line.startswith(prefix)]
    return json.loads(entry)


def option(args, flag, default=None):
    if flag in args:
        return args[args.index(flag) + 1]
    return next((token.split("=", 1)[1] for token in args if token.startswith(flag + "=")), default)


def distance_sum(coordinates, start, finish):
    return math.fsum(haversine_metres(*coordinates[i], *coordinates[i + 1]) for i in range(start, finish))


def archive_values(provider, layer, coordinates, times, wind_from=None):
    """Batch the same linear archive interpolation, with native coastal fallback.

    Paired DataArray coordinates evaluate one sample per saved fix rather
    than constructing a Cartesian position/time product. Scalar checks in
    audit_run verify agreement at each five-minute boundary and capture.
    """
    dataset = provider._require_dataset()
    time_axis = "valid_time" if layer == "wind" else "time"
    variables = {"wind": ("u10", "v10"), "current": ("water_u", "water_v"),
                 "wave": ("VHM0", "VMDR", "VTPK")}[layer]
    query = {"latitude": coordinates[:, 0], "longitude": (coordinates[:, 1] + 180) % 360 - 180,
             time_axis: np.asarray([np.datetime64(utc(time).replace(tzinfo=None), "ns") for time in times])}
    if "depth" in dataset.coords:
        query["depth"] = np.zeros(len(coordinates))
    interpolated = dataset[list(variables)]
    remaining = {}
    for axis, values in query.items():
        grid = dataset.coords[axis].values
        persistent = layer == "current" and axis == "time" and grid.size == 1 and provider.persist_single_snapshot
        if grid.size == 1:
            assert persistent or np.all(values == grid[0]), (layer, axis, "Singleton coordinate mismatch")
            interpolated = interpolated.isel({axis: 0}, drop=True)
        else:
            assert np.all(values >= grid.min()) and np.all(values <= grid.max()), (layer, axis, "Archive bounds")
            remaining[axis] = xr.DataArray(values, dims="audit_frame")
    sampled = interpolated.interp(remaining)
    values = {name: np.asarray(sampled[name].values, dtype=float) for name in variables}
    if layer in {"wind", "current"}:
        u, v = [values[name] for name in variables]
        speed = np.hypot(u, v)
        direction = np.degrees(np.arctan2(u, v)) % 360
        if layer == "wind":
            direction = (direction + 180) % 360
            if wind_from is not None:
                angle = math.radians(wind_from)
                u, v = -speed * math.sin(angle), -speed * math.cos(angle)
                direction = np.full(len(coordinates), float(wind_from))
        result = {"u": u, "v": v, "speed": speed, "direction": direction}
        if layer == "current":
            result["depth"] = np.zeros(len(coordinates))
    else:
        result = {"height": values["VHM0"], "direction": values["VMDR"] % 360,
                  "peak_period": values["VTPK"]}
    missing = np.flatnonzero(~np.all(np.isfinite(np.column_stack(list(result.values()))), axis=1))
    for index in missing:
        scalar = asdict(provider.sample(*coordinates[index], times[index]))
        if layer == "wind" and wind_from is not None:
            angle = math.radians(wind_from)
            scalar.update(u=-scalar["speed"] * math.sin(angle), v=-scalar["speed"] * math.cos(angle),
                          direction=float(wind_from))
        for key, value in scalar.items():
            result[key][index] = value
    return result, len(missing)


def audit_run(arm, directory, route, expected, providers, wind_from):
    log = (directory / "run.log").read_text().splitlines()
    gate = json.loads((directory / "departure_gate.json").read_text())
    actual_departure = utc(gate["actual_departure_utc"])
    configuration = json.loads((STUDY / "underway_config.json").read_text())
    assert utc(gate["monitoring_start_utc"]) == DEPARTURE and utc(gate["traffic_epoch_utc"]) == ENCOUNTER
    assert actual_departure == DEPARTURE + timedelta(seconds=gate["gate_wait_seconds"])
    args = one_json(log, "Runner arguments: ")
    crop = one_json(log, "Marine archive source grids: ")
    assert crop == {name: list(bounds) for name, bounds in OTRANTO_BOUNDS.items()}, arm
    outcome = one_json(log, "Mission outcome: ")
    arg = lambda flag, default=None: option(args, flag, default)
    assert arg("--planner") == "gamma" and arg("--controller") == "acados-direct", arm
    assert arg("--environment") == "marine" and arg("--dynamics") == "fossen", arm
    assert (ROOT / arg("--route-gpx")).resolve() == ROUTE_PATH, arm
    assert "--return-to-start" in args and outcome["return_to_start"], arm
    assert utc(arg("--time")) == actual_departure and float(arg("--dt")) == STEP_SECONDS, arm
    assert int(arg("--steps")) == MAXIMUM_STEPS, arm
    assert float(arg("--arrival-radius")) == outcome["arrival_radius_metres"] == RADIUS, arm
    assert float(arg("--heading")) == 45 and float(arg("--flap")) == -10, arm
    assert float(arg("--minimum-depth", "3")) == 3 and arg("--global-map") is None, arm
    assert float(arg("--local-span", "4715")) == 4715 and float(arg("--local-resolution", "115")) == 115, arm
    assert float(arg("--obstacle-buffer", "115")) == 115, arm
    assert not any(flag in args for flag in (
        "--tack-plan-replay", "--learned-tack-model", "--learned-arrival-model",
        "--combined-controller-checkpoint", "--transformer-diffusion-checkpoint", "--ais-replay",
        "--ais-course-checkpoint", "--wind-speed", "--wind-direction",
    )), arm

    replay_path = directory / "navigation.jsonl.gz"
    with gzip.open(replay_path, "rt") as stream:
        header = json.loads(next(stream))
        replay = [json.loads(line) for line in stream]
    assert [waypoint["name"] for waypoint in header["route"]] == list(route), arm
    assert all(haversine_metres(*position(w), *route[w["name"]]) < 0.001 for w in header["route"]), arm
    assert len(replay) == outcome["steps"] + 1 and outcome["steps"] <= MAXIMUM_STEPS, arm
    start = replay[0]["state"]
    assert start == gate["certificate"]["own_state"], arm
    assert replay[0]["observation"]["ais_targets"] == gate["certificate"]["received_reports"], arm
    assert start["heading"] == 45 and start["speed"] == 0 and start["flap"] == -10, arm
    assert haversine_metres(*position(start), *route["001"]) < 0.001, arm

    coordinates = [position(frame["state"]) for frame in replay]
    gpx_path = directory / "track.gpx"
    gpx_nodes = ElementTree.parse(gpx_path).findall(".//{*}trkpt")
    gpx_coordinates = [gpx_position(node) for node in gpx_nodes]
    gpx_max_error_m = max(haversine_metres(*a, *b) for a, b in zip(coordinates, gpx_coordinates, strict=True))
    assert gpx_max_error_m < 0.001, arm
    for step, (frame, node) in enumerate(zip(replay, gpx_nodes, strict=True)):
        obs = frame["observation"]
        assert step == frame["navigation"]["step"] and frame["state"] == obs["boat"], (arm, step)
        expected_time = actual_departure + timedelta(seconds=step * STEP_SECONDS)
        assert utc(obs["time"]) == expected_time, (arm, step)
        gpx_time = node.findtext("{*}time")
        if gpx_time is not None:
            assert utc(gpx_time) == expected_time, (arm, step, "GPX time")
        encounter_seconds = (expected_time - ENCOUNTER).total_seconds()
        report_count = 3 + (encounter_seconds >= configuration["underway_first_delivery_s"])
        assert len(obs["ais_targets"]) == report_count and all(obs[field] is not None for field in ENVIRONMENT), (arm, step)

    arrivals = outcome["waypoint_arrivals"]
    assert [(e["phase"], e["waypoint"]) for e in arrivals] == expected[:len(arrivals)], arm
    assert (outcome["status"] == "completed") == (len(arrivals) == len(expected)), arm
    if outcome["status"] == "completed":
        assert arrivals[-1]["step"] == outcome["steps"], arm
    elif outcome["steps"] < MAXIMUM_STEPS:
        if outcome["status"] == "no_full_clearance_action":
            stopped = f"AIS strict clearance stop before step {outcome['steps'] + 1}:"
        else:
            stopped = "AIS fallback stopped before step "
        assert any(line.startswith(stopped) for line in log), (arm, "Undocumented early stop", outcome)
    validated_captures, start_step = [], 0
    for event in arrivals:
        finish = event["step"]
        target = route[event["waypoint"]]
        distances = [haversine_metres(*point, *target) for point in coordinates[start_step:finish + 1]]
        distance = distances[-1]
        assert finish > start_step and distance <= RADIUS and min(distances[:-1]) > RADIUS, (arm, event)
        assert abs(distance - event["distance_metres"]) < 1e-6, (arm, event)
        for frame in replay[start_step + (start_step != 0):finish + 1]:
            nav = frame["navigation"]
            assert (nav["phase"], nav["waypoint"]) == (event["phase"], event["waypoint"]), (arm, nav["step"])
        validated_captures.append({
            **event, "elapsed_s": finish * STEP_SECONDS,
            "from_monitoring_start_s": gate["gate_wait_seconds"] + finish * STEP_SECONDS,
            "segment_s": (finish - start_step) * STEP_SECONDS,
            "segment_distance_m": distance_sum(coordinates, start_step, finish),
            "utc": utc(replay[finish]["observation"]["time"]).isoformat(),
            "latitude": coordinates[finish][0], "longitude": coordinates[finish][1],
        })
        start_step = finish
    unfinished = None
    if len(arrivals) < len(expected):
        phase, name = expected[len(arrivals)]
        distances = [haversine_metres(*point, *route[name]) for point in coordinates[start_step:]]
        assert min(distances) > RADIUS, (arm, "Unrecorded capture", phase, name)
        assert all((f["navigation"]["phase"], f["navigation"]["waypoint"]) == (phase, name)
                   for f in replay[start_step + (start_step != 0):]), arm
        unfinished = {"phase": phase, "waypoint": name, "from_step": start_step,
                      "elapsed_s": outcome["steps"] * STEP_SECONDS,
                      "segment_s": (outcome["steps"] - start_step) * STEP_SECONDS,
                      "segment_distance_m": distance_sum(coordinates, start_step, outcome["steps"]),
                      "minimum_range_m": min(distances), "terminal_range_m": distances[-1]}

    archive_max_error, scalar_max_error = 0.0, 0.0
    scalar_steps = sorted({0, len(replay) - 1, *range(0, len(replay), 60), *(event["step"] for event in arrivals)})
    archive_fallback_counts = {}
    angle = math.radians(wind_from)
    times = [frame["observation"]["time"] for frame in replay]
    for field, provider in providers.items():
        values, archive_fallback_counts[field] = archive_values(provider, field, np.asarray(coordinates), times, wind_from)
        actual = {name: np.asarray([frame["observation"][field][name] for frame in replay]) for name in values}
        assert values.keys() == replay[0]["observation"][field].keys(), (arm, field)
        for name, expected_array in values.items():
            error = np.abs(expected_array - actual[name])
            archive_max_error = max(archive_max_error, float(error.max()))
            assert np.all(np.isfinite(error)) and np.all(error <= 1e-8), (arm, field, name, int(np.argmax(error)), float(error.max()))
        for step in scalar_steps:
            scalar = asdict(provider.sample(*coordinates[step], times[step]))
            if field == "wind":
                scalar.update(u=-scalar["speed"] * math.sin(angle), v=-scalar["speed"] * math.cos(angle),
                              direction=float(wind_from))
            for name, value in scalar.items():
                difference = abs(value - values[name][step])
                scalar_max_error = max(scalar_max_error, difference)
                assert difference <= 1e-8, (arm, step, field, name, "Batch/scalar disagreement")
        print(f"{arm}: {field} archive checked at all {len(replay)} fixes", flush=True)

    return {
        "directory": str(directory.relative_to(ROOT)), "planner": "gamma", "controller": "acados-direct",
        "wind_from_degrees": wind_from, "status": outcome["status"], "steps": outcome["steps"],
        "elapsed_s": outcome["steps"] * STEP_SECONDS, "captures": validated_captures,
        "monitoring_start_utc": gate["monitoring_start_utc"],
        "actual_departure_utc": gate["actual_departure_utc"],
        "gate_wait_seconds": gate["gate_wait_seconds"],
        "total_elapsed_s": gate["gate_wait_seconds"] + outcome["steps"] * STEP_SECONDS,
        "gate_attempts": gate["attempts"], "maximum_gate_decision_s": gate["maximum_gate_decision_s"],
        "launch_gate_uncapped_margin_m": gate["certificate"]["min_uncapped_startup_margin_m"],
        "launch_gate_preview_margin_m": gate["certificate"]["min_preview_margin_m"],
        "unfinished_leg": unfinished, "track_m": distance_sum(coordinates, 0, len(coordinates) - 1),
        "forecast_updates": sum(f["navigation"].get("forecast_closest_m") is not None for f in replay[1:]),
        "controller_outcomes": one_json(log, "NMPC outcomes: "),
        "maximum_gpx_position_difference_m": gpx_max_error_m,
        "gpx_timestamp_count": sum(node.findtext("{*}time") is not None for node in gpx_nodes),
        "archive_checks_per_provider": len(replay), "maximum_archive_value_difference": archive_max_error,
        "archive_scalar_crosschecks_per_provider": len(scalar_steps),
        "maximum_archive_batch_scalar_difference": scalar_max_error,
        "archive_native_coastal_fallback_counts": archive_fallback_counts,
        "departure_environment": {name: replay[0]["observation"][name] for name in ENVIRONMENT},
        "sha256": {path.name: digest(path) for path in (
            replay_path, gpx_path, directory / "run.log", directory / "departure_gate.json",
            directory / "departure_gate.jsonl",
        )},
    }, replay


def compare_trajectories(first, second):
    common = list(zip(first, second))
    same_length = len(first) == len(second)
    first_difference = lambda field: next((i for i, (a, b) in enumerate(common) if a[field] != b[field]), None)
    controls = ("rudder_command_deg", "flap_command_deg")
    references = ("heading_reference_deg", "bow_reference_deg", "forecast_closest_m", "forecast_tack_in_min")
    first_command = next((i for i, (a, b) in enumerate(common)
                          if any(a["navigation"].get(k) != b["navigation"].get(k) for k in controls)), None)
    return {
        "paired_frames": len(common), "frame_counts_equal": same_length,
        "states_identical": same_length and first_difference("state") is None,
        "observations_identical": same_length and first_difference("observation") is None,
        "first_state_difference_at_step": first_difference("state"),
        "first_observation_difference_at_step": first_difference("observation"),
        "first_applied_command_difference_at_step": first_command,
        "first_applied_command_difference_start_s": None if first_command is None else (first_command - 1) * STEP_SECONDS,
        "applied_commands_identical": same_length and first_command is None,
        "nominal_references_identical": same_length and all(
            all(a["navigation"].get(k) == b["navigation"].get(k) for k in references) for a, b in common),
        "maximum_coincident_position_difference_m": max(
            haversine_metres(*position(a["state"]), *position(b["state"])) for a, b in common),
    }
