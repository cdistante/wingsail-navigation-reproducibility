"""Audit navigation replays and export matched wind-direction ablation tables."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import product, zip_longest
import json
from math import asin, atan2, cos, degrees, hypot, isfinite, radians, sin, sqrt
from pathlib import Path
import re
from statistics import mean, median
import sys
from xml.etree import ElementTree

from chart_preflight import ChartPointSampler, write_chart_context
from run_study import CHART, METHODS as BENCHMARK_METHODS, ROUTES, OFFSETS, WINDS


WORKSPACE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WORKSPACE))
from world._geo import EARTH_RADIUS_M
from world.scenario import Route
sys.path.pop(0)


CONTRASTS = (
    ("standard_grid", "gamma_grid"),
    ("diffusion_grid", "gamma_grid"),
    ("transformer_grid", "diffusion_grid"),
    ("gamma_los", "gamma_grid"),
    ("gamma_direct", "gamma_grid"),
    ("gamma_direct", "gamma_los"),
    ("forecast", "gamma_direct"),
    ("progress", "forecast"),
    ("arrival", "forecast"),
    ("arrival", "progress"),
    ("actuator", "gamma_grid"),
)
METHODS = (
    "standard_grid", "gamma_grid", "diffusion_grid", "transformer_grid", "gamma_los",
    "gamma_direct", "forecast", "progress", "arrival", "actuator",
)
METHOD_LABELS = dict(zip(METHODS, (
    "Standard + grid MPC", "Gamma + grid MPC", "Diffusion + grid MPC",
    "Diffusion Transformer + grid MPC", "Gamma + acados LOS", "Gamma + acados Direct",
    "Forecast tacks", "Forecast + 120 s progress", "Forecast + arrival time",
    "Learned actuator + grid MPC fallback",
)))
CELL_FIELDS = ("scope", "route", "base_wind", "offset")
MATCH_FIELDS = ("wind_from", "cap_s", "radius_m", "dt_s", "route_file", "departure",
                "wind_speed_mps", "archive_id", "return_to_start", "chart", "reference_kernels")
RUN_METADATA = ("run_id", "route", "method", "base_wind", "offset", "wind_from", "cap_s",
                "radius_m", "dt_s", "route_file", "departure", "scope")
METRES_TOLERANCE = 0.001
WIND_TOLERANCE = 1e-6
METRES_PER_SECOND_TO_KNOTS = 1.94384449


def angle_diff(first, second):
    return (first - second + 180.0) % 360.0 - 180.0


def timestamp(value):
    when = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def position(state):
    return float(state["latitude"]), float(state["longitude"])


def distance(first, second):
    lat0, lon0, lat1, lon1 = map(radians, (*first, *second))
    value = sin((lat1 - lat0) / 2.0)**2 + cos(lat0) * cos(lat1) * sin((lon1 - lon0) / 2.0)**2
    return 2.0 * EARTH_RADIUS_M * asin(sqrt(min(1.0, max(0.0, value))))


def track_points(path):
    for _, element in ElementTree.iterparse(path, events=("end",)):
        if element.tag.rsplit("}", 1)[-1] == "trkpt":
            yield float(element.attrib["lat"]), float(element.attrib["lon"])
            element.clear()


def log_payloads(path, return_code, original, route, replay_path, gpx_path):
    mission = None
    counters = {}
    arrivals = []
    phase = "outbound"
    last_update = -1
    final_line = ""
    for line in path.open(encoding="utf-8"):
        final_line = line.strip() or final_line
        if line.startswith("Mission outcome: "):
            if mission is not None:
                raise ValueError("multiple mission outcomes in run log")
            mission = json.loads(line.removeprefix("Mission outcome: "))
        elif line.startswith("NMPC outcomes: "):
            counters.update({"nmpc_" + key: count for key, count in
                             json.loads(line.removeprefix("NMPC outcomes: ")).items()})
        elif match := re.match(r"Own-ship Diffusion Transformer: (\d+)/(\d+) predictions accepted, (\d+) safety fallback", line):
            counters.update(zip(("transformer_acceptances", "transformer_predictions", "transformer_safety_fallbacks"),
                                map(int, match.groups())))
        elif match := re.match(r"Combined marine controller: (\d+) accepted, (\d+) explicit MPC fallback\(s\), (\d+) safety-gate rejection", line):
            counters.update(zip(("actuator_acceptances", "actuator_mpc_fallbacks", "actuator_safety_rejections"),
                                map(int, match.groups())))
        elif match := re.fullmatch(r"Route waypoint (.+) reached at step (\d+)\.\s*", line):
            arrivals.append({"phase": phase, "waypoint": match[1], "step": int(match[2])})
        elif line.startswith("Turnaround reached at step "):
            phase = "return"
        elif line.startswith("step="):
            last_update = int(line.split()[0].removeprefix("step=").removesuffix(":"))
    if return_code == 0 and mission is not None:
        return mission, counters
    stop = re.fullmatch(r"RuntimeError: (\w+) planner produced no feasible heading at step (\d+)\.", final_line)
    strategy = BENCHMARK_METHODS.get(original["method"])
    expected_planner = strategy.planner if strategy is not None else "gamma"
    if return_code != 1 or mission is not None or stop is None or stop[1] != expected_planner:
        raise ValueError(f"missing native mission outcome or unrecognized nonzero exit {return_code}")
    steps = int(stop[2])
    if last_update != steps - 1 or not gpx_path.is_file():
        raise ValueError("planner-stop step does not match the final logged update or saved track")
    returning = original.get("return_to_start", False)
    targets = mission_targets(route, returning)
    actual_prefix = [(event["phase"], event["waypoint"]) for event in arrivals]
    if actual_prefix != [(phase, target.name) for phase, target in targets[:len(arrivals)]] or len(arrivals) >= len(targets):
        raise ValueError("planner-stop waypoint prefix differs from the route")
    observed_steps = {steps, *(item["step"] for item in arrivals)}
    points = {}
    with gzip.open(replay_path, "rt", encoding="utf-8") as stream:
        next(stream)
        end_step = -1
        for end_step, line in enumerate(stream):
            if end_step in observed_steps:
                frame = json.loads(line)
                if frame["navigation"]["step"] != end_step:
                    raise ValueError("planner-stop replay step order disagrees with the log")
                points[end_step] = position(frame["state"])
    if end_step != steps or set(points) != observed_steps:
        raise ValueError("planner-stop replay lacks the final update or a waypoint capture frame")
    for event, (_, target) in zip(arrivals, targets):
        capture_range = distance(points[event["step"]], (target.latitude, target.longitude))
        if capture_range > float(original["radius_m"]) + METRES_TOLERANCE:
            raise ValueError("planner-stop replay does not substantiate a logged waypoint capture")
        event["distance_metres"] = capture_range
    current_phase, current_target = targets[len(arrivals)]
    mission = {
        "status": "no_feasible_heading", "steps": steps, "phase": current_phase, "return_to_start": returning,
        "route": route.name, "departure_waypoint": route.departure.name, "arrival_radius_metres": float(original["radius_m"]),
        "waypoint_arrivals": arrivals, "active_waypoint": current_target.name,
        "waypoint_distance_metres": distance(points[steps], (current_target.latitude, current_target.longitude)),
        "termination_planner": stop[1], "terminal_source": "runner traceback and replay",
    }
    return mission, counters


def wind_sample(observation, expected_direction, expected_speed):
    wind = observation["wind"]
    u, v, speed, declared = (float(wind[key]) for key in ("u", "v", "speed", "direction"))
    if not all(isfinite(value) for value in (u, v, speed, declared)) or speed <= 0:
        raise ValueError("missing, nonfinite, or calm wind sample")
    from_bearing = (degrees(atan2(u, v)) + 180.0) % 360.0
    magnitude_error = abs(hypot(u, v) - speed)
    direction_error = max(abs(angle_diff(from_bearing, expected_direction)),
                          abs(angle_diff(declared, from_bearing)))
    if expected_speed is not None:
        magnitude_error = max(magnitude_error, abs(speed - expected_speed))
    if direction_error > WIND_TOLERANCE or magnitude_error > WIND_TOLERANCE:
        raise ValueError(f"expected {expected_direction}° FROM and {expected_speed} m/s, got {from_bearing}° and {speed}")
    return speed, direction_error


def current_wave_sample(observation):
    current, wave = observation["current"], observation["wave"]
    u, v, speed, declared = (float(current[key]) for key in ("u", "v", "speed", "direction"))
    if (not all(isfinite(value) for value in (u, v, speed, declared))
            or abs(hypot(u, v) - speed) > WIND_TOLERANCE
            or (speed > 0 and abs(angle_diff(declared, degrees(atan2(u, v)))) > WIND_TOLERANCE)):
        raise ValueError("recorded current TO bearing or vector magnitude disagrees")
    if not all(isfinite(wave[key]) for key in ("height", "direction", "peak_period")):
        raise ValueError("nonfinite wave forcing")


def barb_error(frame):
    wind = frame["observation"]["wind"]
    east, north = frame["wind_grid"]
    if not east or len(east) != len(north):
        raise ValueError("wind barb velocity grid is malformed")
    errors = []
    for values, component in ((east, wind["u"]), (north, wind["v"])):
        knots = component * METRES_PER_SECOND_TO_KNOTS
        errors.extend((abs(min(values) - knots), abs(max(values) - knots)))
    error = max(errors)
    if not isfinite(error) or error > WIND_TOLERANCE:
        raise ValueError("wind barbs differ from observed ownship wind")
    return error


def file_fingerprint(path, study):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.relative_to(study)), "bytes": path.stat().st_size, "sha256": digest}


def resolve_route(meta, workspace):
    path = Path(meta["route_file"])
    if not path.is_absolute() and not (workspace / path).is_file():
        path = Path("current_route") / path
    return workspace / path


def mission_targets(route, returning):
    targets = [("outbound", point) for point in route.waypoints[1:]]
    if returning:
        targets.extend(("return", point) for point in reversed(route.waypoints[:-1]))
    return targets


def run_inputs(folder, workspace):
    original = json.loads((folder / "experiment.json").read_text())
    receipt = json.loads((folder / "execution.json").read_text())
    if any(receipt.get(key) != original[key] for key in RUN_METADATA):
        raise ValueError("run execution and experiment metadata differ")
    paths = tuple(folder / original.get(key, filename) for key, filename in
                  (("replay_file", "navigation.jsonl.gz"), ("track_file", "track.gpx"), ("log_file", "run.log")))
    replay_path, gpx_path, log_path = paths
    route = Route.load_gpx(resolve_route(original, workspace), route_name=original.get("route_name"))
    mission, counters = log_payloads(log_path, receipt["return_code"], original, route, replay_path, gpx_path)
    return original, receipt["return_code"], paths, route, mission, counters


def load_mission(folder, workspace=WORKSPACE):
    """Return the native outcome or reconstruct a recognized planner-stop outcome."""
    return run_inputs(Path(folder), workspace)[4]


def audited_run(metadata_path, study, workspace):
    folder = metadata_path.parent
    receipt_path = folder / "execution.json"
    original, return_code, paths, route, mission, counters = run_inputs(folder, workspace)
    meta = {key: original[key] for key in RUN_METADATA}
    meta.update((key, original[key]) for key in MATCH_FIELDS if key in original)
    for key in ("base_wind", "offset", "wind_from", "cap_s", "radius_m", "dt_s"):
        meta[key] = float(meta[key])
    base, offset, direction = (meta[key] for key in ("base_wind", "offset", "wind_from"))
    if base not in (0, 90, 180, 270) or offset not in (-5, 5) or abs(angle_diff(base + offset, direction)) > WIND_TOLERANCE:
        raise ValueError("wind setup differs from the four-direction, ±5° design")
    if meta["method"] not in METHODS or meta["scope"] not in ("full_route", "bounded_progress"):
        raise ValueError("unknown method or scope")
    dt, cap_s, radius = (meta[key] for key in ("dt_s", "cap_s", "radius_m"))
    if not all(isfinite(value) and value > 0 for value in (dt, cap_s, radius)):
        raise ValueError("invalid cap, capture radius, or time step")
    wind_speed = original.get("wind_speed_mps", original.get("wind_speed"))
    replay_path, gpx_path, log_path = paths
    if "return_to_start" in original and original["return_to_start"] != mission["return_to_start"]:
        raise ValueError("recorded return configuration differs from the requested mission")
    meta["return_to_start"] = mission["return_to_start"]
    chart = ChartPointSampler(workspace / original.get("chart", CHART))
    expected = mission_targets(route, mission["return_to_start"])
    planned_targets = [(phase, point.name) for phase, point in expected]
    arrivals = mission["waypoint_arrivals"]
    arrival_targets = [(item["phase"], item["waypoint"]) for item in arrivals]
    completed = mission["status"] == "completed"
    if (arrival_targets != planned_targets[:len(arrivals)]
            or mission["route"] != route.name or mission["departure_waypoint"] != route.departure.name
            or mission["arrival_radius_metres"] != radius or completed != (len(arrivals) == len(expected))):
        raise ValueError("mission status, capture configuration or waypoint order disagrees with the GPX")
    arrival_steps = [item["step"] for item in arrivals]
    if arrival_steps != sorted(arrival_steps):
        raise ValueError("capture updates are out of order")
    by_step = defaultdict(list)
    for item in arrivals:
        by_step[item["step"]].append(item)
    frames, track_length, max_gpx_error, max_wind_error, max_barb_error = 0, 0.0, 0.0, 0.0, 0.0
    first_range = None
    previous, active, last_side, through_wind, tack_count = None, 0, 0, False, 0
    first_speed = None
    wind_min, wind_max = float("inf"), -float("inf")
    learned_windows = {}
    trajectory = hashlib.sha256()
    chart_blocked_samples, chart_outside_coverage_samples = 0, 0
    first_blocked_step, first_outside_step = None, None
    legs = [
        {**meta, "leg": ix, "phase": phase, "waypoint": point.name, "active_steps": 0,
         "closest_m": None, "closest_elapsed_s": None, "arrival_elapsed_s": None, "arrival_distance_m": None}
        for ix, (phase, point) in enumerate(expected, 1)
    ]
    begin = timestamp(meta["departure"])
    with gzip.open(replay_path, "rt", encoding="utf-8") as stream:
        header = json.loads(next(stream))
        if (header["format"] != "maritime_world.navigation_replay" or header["route_name"] != route.name
                or header["route"] != [{"latitude": point.latitude, "longitude": point.longitude, "name": point.name}
                                       for point in route.waypoints]):
            raise ValueError("replay header does not match the source GPX route")
        for step, (line, gpx_point) in enumerate(zip_longest(stream, track_points(gpx_path))):
            if line is None or gpx_point is None:
                raise ValueError("GPX track and replay have different frame counts")
            frame = json.loads(line)
            state, observation, nav = (frame[key] for key in ("state", "observation", "navigation"))
            trajectory.update(json.dumps((state, nav.get("rudder_command_deg"), nav.get("flap_command_deg")),
                                         sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
            trajectory.update(b"\n")
            point = position(state)
            if (nav["step"] != step or (timestamp(observation["time"]) - begin).total_seconds() != step * dt
                    or not all(isfinite(value) for value in (*point, state["heading"]))
                    or position(observation["boat"]) != point):
                raise ValueError(f"replay clock, step or observed position disagrees at step {step}")
            chart_status = chart.blocked(*point)
            if chart_status is None:
                chart_outside_coverage_samples += 1
                if first_outside_step is None:
                    first_outside_step = step
            elif chart_status:
                chart_blocked_samples += 1
                if first_blocked_step is None:
                    first_blocked_step = step
            gpx_error = distance(point, gpx_point)
            if gpx_error > METRES_TOLERANCE:
                raise ValueError(f"replay and GPX position disagree at frame {step}")
            speed, direction_error = wind_sample(observation, direction, wind_speed)
            current_wave_sample(observation)
            max_barb_error = max(max_barb_error, barb_error(frame))
            wind_min, wind_max = min(wind_min, speed), max(wind_max, speed)
            max_wind_error = max(max_wind_error, direction_error)
            max_gpx_error = max(max_gpx_error, gpx_error)
            phase, target = expected[active]
            leg = legs[active]
            target_range = distance(point, (target.latitude, target.longitude))
            if (nav["waypoint"] != target.name or nav["phase"] != phase
                    or abs(nav["waypoint_distance_m"] - target_range) > METRES_TOLERANCE):
                raise ValueError(f"replay target does not match ordered GPX target at frame {step}")
            if step == 0:
                first_speed = speed
                first_range = target_range
                first_forcing = {key: observation[key] for key in ("wind", "current", "wave")}
                if distance(point, (route.departure.latitude, route.departure.longitude)) > METRES_TOLERANCE:
                    raise ValueError("replay launch position differs from the GPX departure")
            else:
                track_length += distance(previous, point)
                leg["active_steps"] += 1
            if leg["closest_m"] is None or target_range < leg["closest_m"]:
                leg.update(closest_m=target_range, closest_elapsed_s=step * dt)
            relative = angle_diff(state["heading"], observation["wind"]["direction"])
            if abs(relative) < 20:
                through_wind = True
            else:
                side = 1 if relative > 0 else -1
                tack_count += bool(last_side and side != last_side and through_wind)
                last_side, through_wind = side, False
            if decision_time := nav.get("learned_tack_selection_utc"):
                key = active, decision_time
                status = nav["learned_tack_selection"]
                if learned_windows.setdefault(key, status) != status:
                    raise ValueError("learned decision status changed within a forecast window")
            for event in by_step.get(step, ()):
                phase, target = expected[active]
                leg = legs[active]
                capture_range = distance(point, (target.latitude, target.longitude))
                if (step == 0 or (event["phase"], event["waypoint"]) != (phase, target.name)
                        or capture_range > radius + METRES_TOLERANCE
                        or abs(capture_range - event["distance_metres"]) > METRES_TOLERANCE):
                    raise ValueError(f"capture event disagrees with replay at step {step}")
                if leg["closest_m"] is None or capture_range < leg["closest_m"]:
                    leg.update(closest_m=capture_range, closest_elapsed_s=step * dt)
                leg.update(arrival_elapsed_s=step * dt, arrival_distance_m=capture_range)
                if active < len(legs) - 1:
                    active += 1
            previous = point
            frames += 1
    steps = frames - 1
    if steps != mission["steps"] or steps < 0 or steps * dt > cap_s or (completed and arrival_steps[-1] != steps):
        raise ValueError("recorded steps differ from the mission log or requested cap")
    if mission["status"] == "step_limit" and steps * dt != cap_s:
        raise ValueError("step-limited run finished at a different cap")
    if len(arrivals) != sum(row["arrival_elapsed_s"] is not None for row in legs):
        raise ValueError("some logged arrival events were not observed in the replay")
    if mission["active_waypoint"] != (None if completed else expected[active][1].name):
        raise ValueError("active waypoint at termination differs from the verified ordered prefix")
    final_target = expected[active][1]
    final_range = distance(previous, (final_target.latitude, final_target.longitude))
    reported_range = mission["waypoint_distance_metres"]
    if reported_range is not None and abs(reported_range - final_range) > METRES_TOLERANCE:
        raise ValueError("recorded final range differs from the mission outcome")
    decision_counts = Counter(learned_windows.values())
    result = {**meta, **counters, "wind_from": direction, "mission_status": mission["status"],
              "runner_return_code": return_code, "termination_planner": mission.get("termination_planner"),
              "steps": steps, "elapsed_s": steps * dt, "mission_completed": completed,
              "waypoint_order_verified": True, "waypoints_reached": len(arrivals), "waypoints_total": len(legs),
              "arrival_order": " | ".join(f"{phase} {name}" for phase, name in arrival_targets),
              "first_target_range_initial_m": first_range, "first_target_closest_m": legs[0]["closest_m"],
              "first_target_best_progress_pct": 100.0 * (1.0 - legs[0]["closest_m"] / first_range),
              "last_active_waypoint": final_target.name, "last_active_final_range_m": final_range,
              "last_active_closest_m": legs[active]["closest_m"], "track_length_m": track_length,
              "trajectory_and_applied_controls_sha256": trajectory.hexdigest(),
              "upwind_heading_crossings": tack_count, "learned_windows": len(learned_windows),
              "learned_overrides": decision_counts["learned first leg"],
              "learned_outside_support": decision_counts["classical input outside training support"],
              "wind_speed_first_mps": first_speed, "wind_speed_min_mps": wind_min, "wind_speed_max_mps": wind_max,
              "wind_from_max_error_deg": max_wind_error, "wind_barb_max_error_kn": max_barb_error,
              "gpx_max_error_m": max_gpx_error,
              "chart_sampled_frames": frames, "chart_blocked_samples": chart_blocked_samples,
              "chart_outside_coverage_samples": chart_outside_coverage_samples,
              "chart_first_blocked_step": first_blocked_step, "chart_first_outside_coverage_step": first_outside_step,
              "initial_forcing": json.dumps(first_forcing, sort_keys=True),
              "replay": str(replay_path.relative_to(study))}
    manifest = {**meta, "frames": frames, "mission_status": mission["status"], "runner_return_code": return_code,
                **{label: file_fingerprint(path, study) for label, path in
                   (("replay", replay_path), ("track", gpx_path), ("log", log_path),
                    ("metadata", metadata_path), ("execution", receipt_path))}}
    return result, legs, manifest


def save_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(runs, group_fields):
    groups = defaultdict(list)
    for row in runs:
        groups[tuple(row[field] for field in group_fields)].append(row)
    for group, rows in groups.items():
        times = [row["elapsed_s"] for row in rows if row["mission_completed"]]
        yield {
            **dict(zip(group_fields, group)), "runs": len(rows), "completed": len(times),
            "completion_pct": 100.0 * len(times) / len(rows),
            "waypoints_reached": sum(row["waypoints_reached"] for row in rows),
            "waypoints_total": sum(row["waypoints_total"] for row in rows),
            "median_completed_s": median(times) if times else None,
            "mean_best_first_target_progress_pct": mean(row["first_target_best_progress_pct"] for row in rows),
            "median_track_m": median(row["track_length_m"] for row in rows),
            "median_upwind_crossings": median(row["upwind_heading_crossings"] for row in rows),
            "learned_overrides": sum(row["learned_overrides"] for row in rows),
        }


def matched_ablation(runs):
    cells = defaultdict(dict)
    for row in runs:
        key = tuple(row[field] for field in CELL_FIELDS)
        if row["method"] in cells[key]:
            raise ValueError(f"duplicate method {row['method']} in {key}")
        cells[key][row["method"]] = row
    for cell, methods in cells.items():
        departures = {(*(row.get(key) for key in MATCH_FIELDS), row["initial_forcing"]) for row in methods.values()}
        if len(departures) != 1:
            raise ValueError(f"run settings or initial environment vary between methods in {cell}")
        for variant, reference in CONTRASTS:
            if not {variant, reference}.issubset(methods):
                continue
            test, control = methods[variant], methods[reference]
            both = test["mission_completed"] and control["mission_completed"]
            yield {
                **dict(zip(CELL_FIELDS, cell)), "variant": variant, "reference": reference,
                "variant_run": test["run_id"], "reference_run": control["run_id"],
                "variant_completed": test["mission_completed"], "reference_completed": control["mission_completed"],
                "identical_state_and_control_trace": test["trajectory_and_applied_controls_sha256"]
                                                     == control["trajectory_and_applied_controls_sha256"],
                "waypoint_delta": test["waypoints_reached"] - control["waypoints_reached"],
                "completion_time_delta_s": test["elapsed_s"] - control["elapsed_s"] if both else None,
                "first_target_progress_delta_pp": test["first_target_best_progress_pct"] - control["first_target_best_progress_pct"],
            }


def summarize_pairs(pairs):
    groups = defaultdict(list)
    for pair in pairs:
        groups[pair["scope"], pair["variant"], pair["reference"]].append(pair)
    for (scope, variant, reference), rows in groups.items():
        times = [row["completion_time_delta_s"] for row in rows if row["completion_time_delta_s"] is not None]
        improved = sum(row["variant_completed"] and not row["reference_completed"] for row in rows)
        worsened = sum(row["reference_completed"] and not row["variant_completed"] for row in rows)
        yield {
            "scope": scope, "variant": variant, "reference": reference, "paired_cells": len(rows),
            "variant_only_completed": improved, "reference_only_completed": worsened,
            "completion_delta_pp": 100.0 * (improved - worsened) / len(rows),
            "identical_state_and_control_traces": sum(row["identical_state_and_control_trace"] for row in rows),
            "total_waypoint_delta": sum(row["waypoint_delta"] for row in rows),
            "variant_more_waypoints": sum(row["waypoint_delta"] > 0 for row in rows),
            "variant_fewer_waypoints": sum(row["waypoint_delta"] < 0 for row in rows),
            "both_completed": len(times), "median_completion_time_delta_s": median(times) if times else None,
            "variant_faster_pairs": sum(value < 0 for value in times),
            "variant_slower_pairs": sum(value > 0 for value in times),
            "equal_time_pairs": sum(value == 0 for value in times),
            "mean_first_target_progress_delta_pp": mean(row["first_target_progress_delta_pp"] for row in rows),
        }


def perturbation_pairs(runs):
    cells = defaultdict(dict)
    for row in runs:
        cells[row["scope"], row["route"], row["method"], row["base_wind"]][row["offset"]] = row
    for (scope, route, method, base), offsets in cells.items():
        if set(offsets) != set(OFFSETS):
            continue
        negative, positive = (offsets[offset] for offset in sorted(OFFSETS))
        def archive_at_start(row):
            forcing = json.loads(row["initial_forcing"])
            return forcing["wind"]["speed"], forcing["current"], forcing["wave"]

        if (archive_at_start(negative) != archive_at_start(positive)
                or any(negative.get(key) != positive.get(key) for key in MATCH_FIELDS if key != "wind_from")):
            raise ValueError(f"offset pair has unmatched settings or archived launch forcing for {scope}/{route}/{method}/{base}")
        both = negative["mission_completed"] and positive["mission_completed"]
        yield {
            "scope": scope, "route": route, "method": method, "base_wind": base,
            "negative_offset_run": negative["run_id"], "positive_offset_run": positive["run_id"],
            "negative_offset_completed": negative["mission_completed"],
            "positive_offset_completed": positive["mission_completed"],
            "completion_changed": negative["mission_completed"] != positive["mission_completed"],
            "waypoint_delta": positive["waypoints_reached"] - negative["waypoints_reached"],
            "completion_time_delta_s": positive["elapsed_s"] - negative["elapsed_s"] if both else None,
            "first_target_progress_delta_pp": positive["first_target_best_progress_pct"] - negative["first_target_best_progress_pct"],
        }


def summarize_perturbations(pairs):
    groups = defaultdict(list)
    for row in pairs:
        groups[row["scope"], row["method"]].append(row)
    for (scope, method), rows in groups.items():
        times = [row["completion_time_delta_s"] for row in rows if row["completion_time_delta_s"] is not None]
        yield {
            "scope": scope, "method": method, "matched_route_base_pairs": len(rows),
            "completion_flips": sum(row["completion_changed"] for row in rows),
            "positive_offset_only_completed": sum(row["positive_offset_completed"] and row["completion_changed"] for row in rows),
            "negative_offset_only_completed": sum(row["negative_offset_completed"] and row["completion_changed"] for row in rows),
            "both_completed": len(times), "median_signed_time_delta_s": median(times) if times else None,
            "median_absolute_time_delta_s": median(map(abs, times)) if times else None,
            "mean_absolute_first_target_progress_delta_pp": mean(abs(row["first_target_progress_delta_pp"]) for row in rows),
        }


def bounded_route_summary(runs):
    methods = defaultdict(lambda: defaultdict(list))
    for row in runs:
        if row["scope"] == "bounded_progress":
            methods[row["method"]][row["route"]].append(row)
    for method, cases in methods.items():
        details = {"method": method}
        for route in ("naples", "corfu"):
            rows = cases.get(route, [])
            details[f"{route}_runs"] = len(rows)
            details[f"{route}_first_waypoint_captures"] = sum(row["waypoints_reached"] > 0 for row in rows)
            details[f"{route}_median_best_range_reduction_km"] = median(
                (row["first_target_range_initial_m"] - row["first_target_closest_m"]) / 1000.0 for row in rows
            ) if rows else None
        yield details


def summarize_chart_samples(runs):
    groups = defaultdict(list)
    for row in runs:
        groups[row["scope"], row["method"]].append(row)
    for (scope, method), rows in groups.items():
        yield {
            "scope": scope, "method": method, "runs": len(rows),
            "sampled_replay_frames": sum(row["chart_sampled_frames"] for row in rows),
            "runs_with_blocked_samples": sum(row["chart_blocked_samples"] > 0 for row in rows),
            "blocked_samples": sum(row["chart_blocked_samples"] for row in rows),
            "runs_with_out_of_coverage_samples": sum(row["chart_outside_coverage_samples"] > 0 for row in rows),
            "out_of_coverage_samples": sum(row["chart_outside_coverage_samples"] for row in rows),
            "completed_with_no_blocked_or_out_of_coverage_samples": sum(
                row["mission_completed"] and row["chart_blocked_samples"] == row["chart_outside_coverage_samples"] == 0
                for row in rows
            ),
        }


def make_tables(study, method, wind, ablations, perturbations, long_routes, chart_samples):
    full = [row for row in method if row["scope"] == "full_route"]
    bounded = [row for row in method if row["scope"] == "bounded_progress"]

    def fmt(value):
        return "--" if value is None else f"{value:.1f}"

    def table(name, caption, label, columns, headings, cells):
        lines = [
            r"\begin{table*}[t]", r"\centering", r"\small",
            rf"\caption{{{caption}}}", rf"\label{{{label}}}", rf"\begin{{tabular}}{{{columns}}}",
            r"\toprule", " & ".join(headings) + r" \\", r"\midrule",
            *(" & ".join(row) + r" \\" for row in cells),
            r"\bottomrule", r"\end{tabular}", r"\end{table*}", "",
        ]
        (study / name).write_text("\n".join(lines), encoding="utf-8")

    table("paper_comparison.tex",
          r"Full-route runs across four nominal wind-from directions with paired $-5^\circ$ and $+5^\circ$ offsets. "
          "Time uses finished runs. WP counts pool ordered captures and eligible waypoints. "
          "The three Otranto route inputs overlap, so aggregated rows are descriptive.",
          "tab:wind-comparison", "lrrrr", ("Method", "Completed", r"Completion (\%)", "WPs", "Median time (s)"),
          ((METHOD_LABELS[row["method"]], f"{row['completed']}/{row['runs']}", fmt(row["completion_pct"]),
            f"{row['waypoints_reached']}/{row['waypoints_total']}", fmt(row["median_completed_s"])) for row in full))
    wind_lookup = {(row["method"], row["base_wind"]): f"{row['completed']}/{row['runs']}"
                   for row in wind if row["scope"] == "full_route"}
    table("paper_wind.tex",
          r"Full-route completions by nominal wind-from direction. Each cell pools the $-5^\circ$ "
          r"and $+5^\circ$ offset runs.",
          "tab:wind-direction", "lrrrr", ("Method", *(rf"\({base}^\circ\)" for base in WINDS)),
          ((METHOD_LABELS[row["method"]], *(wind_lookup.get((row["method"], base), "--") for base in WINDS))
           for row in full))
    table("paper_ablation.tex",
          "Matched full-route ablations. Positive completion change favors the variant. Time changes are reported "
          "only for route and wind pairs completed by both methods and are measured as variant minus reference.",
          "tab:wind-ablation", r"p{.22\textwidth}p{.20\textwidth}rrrr",
          ("Variant", "Reference", "Pairs", r"\shortstack{$\Delta$ completion\\(pp)}", r"$\Delta$ WPs",
           r"\shortstack{Median $\Delta$ time\\(s)}"),
          ((METHOD_LABELS[row["variant"]], METHOD_LABELS[row["reference"]], str(row["paired_cells"]),
            fmt(row["completion_delta_pp"]), str(row["total_waypoint_delta"]), fmt(row["median_completion_time_delta_s"]))
           for row in ablations if row["scope"] == "full_route"))
    table("paper_perturbation.tex",
          r"Full-route sensitivity to moving from the $-5^\circ$ offset to the $+5^\circ$ offset. "
          "Pairs share route, nominal bearing and method. Time change is reported when both runs completed.",
          "tab:wind-perturbation", "lrrrrr",
          ("Method", "Pairs", r"\shortstack{Completion\\flips}", r"\shortstack{Both\\complete}",
           r"\shortstack{Median $\Delta$ time\\(s)}", r"\shortstack{Median $|\Delta|$ time\\(s)}"),
          ((METHOD_LABELS[row["method"]], str(row["matched_route_base_pairs"]), str(row["completion_flips"]),
            str(row["both_completed"]), fmt(row["median_signed_time_delta_s"]), fmt(row["median_absolute_time_delta_s"]))
           for row in perturbations if row["scope"] == "full_route"))
    table("paper_bounded_progress.tex",
          "First-waypoint captures and median best reduction in initial range on the long routes. "
          "Each run has a three-hour cap and 100 m capture radius.",
          "tab:wind-bounded-progress", "lrrrr",
          ("Method", "Naples WP 002", r"\shortstack{Naples range\\reduced (km)}", "Corfu WP 002",
           r"\shortstack{Corfu range\\reduced (km)}"),
          ((METHOD_LABELS[row["method"]],
            f"{row['naples_first_waypoint_captures']}/{row['naples_runs']}",
            "--" if row["naples_median_best_range_reduction_km"] is None else f"{row['naples_median_best_range_reduction_km']:.3f}",
            f"{row['corfu_first_waypoint_captures']}/{row['corfu_runs']}",
            "--" if row["corfu_median_best_range_reduction_km"] is None else f"{row['corfu_median_best_range_reduction_km']:.3f}")
           for row in long_routes))
    table("paper_chart_sampling.tex",
          "Replay positions sampled on the raw nearest-cell GEBCO chart at each recorded five-second update "
          "and launch. Counts report runs with at least one blocked or out-of-coverage sample. "
          "The diagnostic measures discrete chart samples and omits swept clearance and local buffer effects.",
          "tab:wind-chart-samples", "lrrrr",
          ("Method", r"\shortstack{Compact\\blocked}", r"\shortstack{Compact\\out of coverage}",
           r"\shortstack{Long\\blocked}", r"\shortstack{Long\\out of coverage}"),
          chart_sample_table_rows(chart_samples))
    save_csv(study / "paper_comparison.csv", full)
    save_csv(study / "wind_summary.csv", wind)
    save_csv(study / "bounded_progress.csv", bounded)
    save_csv(study / "paper_bounded_progress.csv", long_routes)
    save_csv(study / "ablation_summary.csv", ablations)
    save_csv(study / "chart_sampling_summary.csv", chart_samples)


def chart_sample_table_rows(rows):
    counts = {(row["method"], row["scope"]): row for row in rows}
    for method in METHODS:
        result = []
        for scope in ("full_route", "bounded_progress"):
            row = counts.get((method, scope))
            if row is None:
                result.extend(("--", "--"))
            else:
                for field in ("runs_with_blocked_samples", "runs_with_out_of_coverage_samples"):
                    result.append(f"{row[field]}/{row['runs']}")
        if any((method, scope) in counts for scope in ("full_route", "bounded_progress")):
            yield METHOD_LABELS[method], *result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--allow-partial", action="store_true", help="Report available runs and list audit errors.")
    args = parser.parse_args()
    study, workspace = args.study.resolve(), args.workspace.resolve()
    runs, legs, manifest, errors = [], [], [], []
    for path in sorted((study / "runs").rglob("experiment.json")):
        try:
            row, waypoints, files = audited_run(path, study, workspace)
        except (KeyError, ValueError, RuntimeError, OSError, EOFError, ElementTree.ParseError) as error:
            errors.append({"file": str(path.relative_to(study)), "error": str(error)})
            if not args.allow_partial:
                (study / "audit_errors.json").write_text(json.dumps(errors, indent=2) + "\n")
                raise RuntimeError(f"{path}: {error}") from error
            continue
        runs.append(row)
        legs.extend(waypoints)
        manifest.append(files)
    if not runs:
        raise ValueError(f"no audited runs in {study / 'runs'}")
    expected = set(product(ROUTES, METHODS, WINDS, OFFSETS))
    observed = {(row["route"], row["method"], row["base_wind"], row["offset"]) for row in runs}
    absent = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if unexpected or not args.allow_partial and absent:
        raise ValueError(f"wind matrix differs from design: {len(absent)} missing cells, {len(unexpected)} unexpected")
    pairs = list(matched_ablation(runs))
    method_index = {method: ix for ix, method in enumerate(METHODS)}
    runs.sort(key=lambda row: (*(row[field] for field in CELL_FIELDS), method_index[row["method"]]))
    save_csv(study / "run_metrics.csv", runs)
    save_csv(study / "waypoint_metrics.csv", legs)
    save_csv(study / "ablation_pairs.csv", pairs)
    offsets = list(perturbation_pairs(runs))
    sensitivity = list(summarize_perturbations(offsets))
    save_csv(study / "perturbation_pairs.csv", offsets)
    save_csv(study / "perturbation_summary.csv", sensitivity)
    save_csv(study / "route_method_summary.csv", list(summarize(runs, ("scope", "route", "method"))))
    manifest_data = {"audited_runs": len(runs), "total_frames": sum(row["frames"] for row in manifest),
                     "expected_runs": len(expected), "missing_cells": absent,
                     "raw_telemetry_format": "gzip JSONL, replay header followed by every recorded frame",
                     "runs": manifest, "audit_errors": errors}
    (study / "telemetry_manifest.json").write_text(json.dumps(manifest_data, indent=2, allow_nan=False) + "\n")
    method = sorted(summarize(runs, ("scope", "method")), key=lambda row: (row["scope"], method_index[row["method"]]))
    wind = sorted(summarize(runs, ("scope", "method", "base_wind")),
                  key=lambda row: (row["scope"], method_index[row["method"]], row["base_wind"]))
    make_tables(study, method, wind, list(summarize_pairs(pairs)), sensitivity,
                sorted(bounded_route_summary(runs), key=lambda row: method_index[row["method"]]),
                list(summarize_chart_samples(runs)))
    chart_context = write_chart_context(study, workspace)
    blocked_routes = {row["route"] for row in chart_context if row["complete_capture_disk_in_blocked_cell"]}
    chart_sensitivity = list(summarize([row for row in runs if row["scope"] == "full_route"
                                        and row["route"] not in blocked_routes], ("scope", "method")))
    for row in chart_sensitivity:
        row["routes_excluded_for_entire_capture_disk_in_blocked_cell"] = " | ".join(sorted(blocked_routes))
    save_csv(study / "chart_feasible_sensitivity.csv", chart_sensitivity)
    print(f"{len(runs)} audited runs, {manifest_data['total_frames']} recorded samples, "
          f"{len(pairs)} matched ablation pairs, {len(errors)} audit errors")


if __name__ == "__main__":
    main()
