"""Verify independent rollups and kinematics of archived joint-policy results."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import gzip
from hashlib import sha256
import json
from math import dist, hypot, isclose
from pathlib import Path
from statistics import mean, median


STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[2]
ARCHIVE = STUDY.with_name("learned_safe_mark_policy") / "archive_scenarios.json.gz"
STATUSES = ("captured", "timeout", "ais_blocked", "chart_preflight_blocked")
POSITION_TOLERANCE_M = .0002
TIME_TOLERANCE_S = 1e-6


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def close(a, b, tolerance=1e-8):
    if a is None or b is None:
        return a is b
    return isclose(a, b, rel_tol=1e-11, abs_tol=tolerance)


def signed_angle(heading, reference):
    return (heading - reference + 180.) % 360. - 180.


def number(row, name):
    return None if row[name] == "" else float(row[name])


def captured(row):
    return row["status"] == "captured"


def point(row, prefix):
    return float(row[prefix + "_east"]), float(row[prefix + "_north"])


def nearest_distance(a, b, target):
    delta = tuple(v - u for u, v in zip(a, b, strict=True))
    length = sum(value**2 for value in delta)
    fraction = max(0., min(1., sum((x - u)*v for x, u, v in zip(target, a, delta, strict=True))/length)) if length else 0.
    return dist(target, tuple(u + fraction*v for u, v in zip(a, delta, strict=True)))


def digest(path):
    with path.open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def key(row, profile="traffic"):
    return row["scenario"], row[profile], row["method"]


def summarize(rows):
    times = [float(row["duration_s"]) for row in rows if captured(row)]
    margins = {name: [number(row, name) for row in rows if row[name]]
               for name in ("min_report_margin_m", "min_truth_margin_m")}
    return {
        "episodes": len(rows), **{status: sum(row["status"] == status for row in rows) for status in STATUSES},
        "median_capture_time_s": median(times) if times else None,
        "mean_capped_time_s": mean(float(row["capped_time_s"]) for row in rows),
        "min_applied_truth_margin_m": min(margins["min_truth_margin_m"], default=None),
        "min_applied_report_margin_m": min(margins["min_report_margin_m"], default=None),
        "applied_episodes_clearance_unconfirmed": sum(row["clearance_confirmed"] == "False" for row in rows),
        "applied_episodes_audited": len(margins["min_truth_margin_m"]),
        "captured_with_clearance_confirmed": sum(captured(row) and row["clearance_confirmed"] == "True" for row in rows),
        "max_report_age_s": max(float(row["max_report_age_s"]) for row in rows),
        "episodes_observing_loss": sum(int(row["max_dropped_epochs"]) > 0 for row in rows),
        "episodes_scored_or_eligible": sum(int(row["eligible_decisions"]) > 0 for row in rows),
        "changed_decisions": sum(int(row["changed_decisions"]) for row in rows),
    }


def paired(cells, method):
    differences = [float(values[method]["duration_s"]) - float(values["physics"]["duration_s"])
                   for values in cells if captured(values[method]) and captured(values["physics"])]
    return {
        "joint_captures": len(differences), "median_delta_s": median(differences) if differences else None,
        "faster": sum(value < -1e-6 for value in differences),
        "slower": sum(value > 1e-6 for value in differences),
        "tied": sum(abs(value) <= 1e-6 for value in differences),
        "learned_only": sum(captured(values[method]) and not captured(values["physics"]) for values in cells),
        "physics_only": sum(captured(values["physics"]) and not captured(values[method]) for values in cells),
    }


def check_path(episode, trace, scenario):
    target = tuple(scenario["target"])
    radius = float(episode["radius_m"])
    is_blocked = episode["status"] == "ais_blocked"
    preflight = episode["status"] == "chart_preflight_blocked"
    require(close(radius, scenario["radius"]), "Wrong radius")
    require(close(float(episode["leg_length_m"]), hypot(*target)), "Wrong leg length")
    require(len(trace) + is_blocked == int(episode["decisions"]), "Applied and attempted decision counts differ")
    require(bool(trace) or is_blocked or preflight, "Missing applied trajectory")
    require(not preflight or not trace, "Chart blocked path should have zero applied steps")
    require(close(float(episode["capped_time_s"]), float(episode["duration_s"]) if captured(episode) else 43200.),
            "Capped cost differs")
    elapsed, position, bow, sailed = 0., (0., 0.), scenario["bow"], 0.
    path_error = 0.
    for index, step in enumerate(trace):
        starts_at, dt = float(step["elapsed_s"]), float(step["applied_s"])
        start, end = point(step, "start"), point(step, "end")
        initial = float(step["initial_bow"])
        require(close(elapsed, starts_at, TIME_TOLERANCE_S) and dist(position, start) <= POSITION_TOLERANCE_M,
                f"Discontinuous time/position at {starts_at}")
        require(abs(signed_angle(initial, bow)) < 1e-7, f"Discontinuous bow at {starts_at}")
        require(0. < dt <= 300., f"Invalid step duration {dt}")
        snapshot = scenario["future"] if scenario["future"] and starts_at >= scenario["change_at"] else scenario["first"]
        index_action = int(step["selected_action"])
        action_bow = snapshot["bow"][index_action]
        velocity = snapshot["velocity"][index_action]
        current = snapshot["flow"]
        wind_from = snapshot["wind_from"]
        turn = abs(signed_angle(action_bow, initial)) + (30. if (
            (signed_angle(initial, wind_from) < 0) != (signed_angle(action_bow, wind_from) < 0)) else 0.)
        turn_seconds = min(dt, turn)
        sail_seconds = dt - turn_seconds
        after_turn = tuple(u + turn_seconds*c for u, c in zip(start, current, strict=True))
        expected_end = tuple(u + sail_seconds*v for u, v in zip(after_turn, velocity, strict=True))
        error = dist(expected_end, end)
        path_error = max(path_error, error)
        require(error <= POSITION_TOLERANCE_M, f"Applied position does not match selected trim at {starts_at}: {error}")
        sailed += turn_seconds*hypot(*current) + sail_seconds*hypot(*velocity)
        if not (index + 1 == len(trace) and captured(episode)):
            separation = min(nearest_distance(start, after_turn, target), nearest_distance(after_turn, end, target))
            require(separation >= radius - POSITION_TOLERANCE_M, f"Non-terminal step entered capture circle at {starts_at}")
        elapsed, position, bow = starts_at + dt, end, float(step["nominal_end_bow"])
    require(close(elapsed, float(episode["duration_s"]), TIME_TOLERANCE_S), "Trace duration differs")
    require(close(sailed, float(episode["distance_sailed_m"]), POSITION_TOLERANCE_M), "Sailed distance differs")
    capture_error = None
    if captured(episode):
        actual = dist(position, target)
        capture_error = abs(actual - radius)
        require(capture_error <= POSITION_TOLERANCE_M, "Terminal position misses capture circle")
        require(close(actual, float(episode["capture_distance_m"]), POSITION_TOLERANCE_M),
                "Trace-derived capture range differs")
    else:
        require(episode["capture_distance_m"] == "", "Uncaptured episode recorded a capture distance")
        if episode["status"] == "timeout":
            require(close(elapsed, 21600., TIME_TOLERANCE_S), "Invalid timeout duration")
    if trace:
        truth = [float(step["truth_margin_m"]) for step in trace]
        report = [float(step["report_margin_m"]) for step in trace]
        require(close(min(truth), number(episode, "min_truth_margin_m")), "Swept truth margin does not match trace")
        require(close(min(report), number(episode, "min_report_margin_m")), "Swept report margin does not match trace")
        minimum_index = truth.index(min(truth))
        require(close(float(trace[minimum_index]["elapsed_s"]), number(episode, "min_truth_margin_step_s")),
                "Minimum truth-margin timestamp differs")
        require((min(truth) > 0 and min(report) > 0) == (episode["clearance_confirmed"] == "True"),
                "Clearance flag does not match applied margins")
        ages = [float(step["age_s"]) for step in trace]
        reported_age = float(episode["max_report_age_s"])
        require(max(ages) <= reported_age and (is_blocked or close(max(ages), reported_age)),
                "Maximum applied report age differs")
        require(sum(age > 10. for age in ages) + is_blocked >= int(episode["stale_steps"]) >= sum(age > 10. for age in ages),
                "Applied stochastic-report step count differs")
    else:
        require(not episode["min_truth_margin_m"] and not episode["min_report_margin_m"] and not episode["clearance_confirmed"],
                "Unapplied episode carries measured clearance")
    changed = sum(step["selected_action"] != step["physics_action_at_same_state"] for step in trace)
    require(changed == int(episode["changed_decisions"]), "Changed decision count differs")
    require(changed == 0 or episode["method"] != "physics", "Physics baseline changed its own action")
    return path_error, capture_error


def same_statistics(calculated, stored, label):
    for metric, value in calculated.items():
        require(close(value, stored[metric]), f"{label} {metric}: reconstructed {value!r}, reported {stored[metric]!r}")


def main():
    paths = {name: STUDY / name for name in (
        "splits.json", "training_receipt.json", "test_summary.json", "test_episodes.csv", "test_leg_trajectories.csv.gz"
    )}
    split, summary, receipt = (json.loads(paths[name].read_text()) for name in (
        "splits.json", "test_summary.json", "training_receipt.json"))
    fingerprints = {name: digest(path) for name, path in paths.items()}
    for filename, expected in split["files_sha256"].items():
        require(digest(ROOT / filename) == expected, f"Frozen split input changed {filename}")
    for name, name_in_summary in (("test_episodes.csv", "test_csv_sha256"), ("test_leg_trajectories.csv.gz", "test_trajectories_sha256"),
                                  ("training_receipt.json", "training_receipt_sha256")):
        require(summary[name_in_summary] == fingerprints[name], f"Summary provenance failed: {name}")
    require(receipt["split_sha256"] == fingerprints["splits.json"], "Training/test split provenance")
    with ARCHIVE.open("rb") as stream:
        require(sha256(stream.read()).hexdigest() == split["files_sha256"][str(ARCHIVE.relative_to(ROOT))],
                "Archive trajectories not tied to raw scenario data")
    methods = {str(name): factor for name, _, factor in summary["method_strengths"]}
    expected = {(case, traffic, method) for case in split["test"]["archive_ids"] for traffic in split["test"]["traffic_profiles"]
                for method in methods}
    with paths["test_episodes.csv"].open(newline="", encoding="utf-8") as stream:
        row_index = {}
        for row in csv.DictReader(stream):
            triplet = key(row)
            require(triplet in expected and triplet not in row_index, f"Unexpected or duplicate row {triplet}")
            row_index[triplet] = row
    require(expected == row_index.keys(), "Held-out matrix coverage differs from locked test split")
    traces = defaultdict(list)
    with gzip.open(paths["test_leg_trajectories.csv.gz"], "rt", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            triplet = key(row, "profile")
            require(triplet in expected, f"Unexpected trajectory {triplet}")
            traces[triplet].append(row)
    with gzip.open(ARCHIVE, "rt", encoding="utf-8") as stream:
        scenarios = {row["identifier"]: row for row in json.load(stream)}
    cells = defaultdict(dict)
    maximum_path_error, maximum_capture_error = 0., 0.
    minima = {name: {"margin_m": None} for name in methods}
    for triplet, row in row_index.items():
        identifier, traffic, method = triplet
        metadata = split["archive_cases"][identifier]
        require(row["route"] == metadata["gpx"] and row["weather_block"] == metadata["archive_block"],
                f"Test metadata disagrees with manifest: {triplet}")
        require(row["status"] in STATUSES, f"Unknown status: {triplet}")
        require((row["status"] == "chart_preflight_blocked") == bool(metadata["entire_chart_disk_blocked"]),
                f"Preflight disagrees with manifest: {triplet}")
        require(close(float(row["strength"]), methods[method]), f"Score strength mismatch: {triplet}")
        require(float(row["radius_m"]) == metadata["radius_m"], f"Radius disagrees with manifest: {triplet}")
        try:
            path_error, capture_error = check_path(row, traces.get(triplet, []), scenarios[identifier])
        except ValueError as exc:
            raise ValueError(f"{triplet}: {exc}") from exc
        maximum_path_error = max(maximum_path_error, path_error)
        if capture_error is not None:
            maximum_capture_error = max(maximum_capture_error, capture_error)
        cells[identifier, traffic][method] = row
        margin = number(row, "min_truth_margin_m")
        if margin is not None and (minima[method]["margin_m"] is None or margin < minima[method]["margin_m"]):
            minima[method] = {"scenario": identifier, "traffic": traffic, "margin_m": margin,
                              "elapsed_s": number(row, "min_truth_margin_step_s"), "status": row["status"]}
    strata = {
        "all": list(cells.values()),
        **{name: [rows for (_, profile), rows in cells.items() if profile == name]
           for name in split["test"]["traffic_profiles"]},
        **{name: [rows for rows in cells.values() if rows["physics"]["route"] == name]
           for name in sorted({row["route"] for row in row_index.values()})}
    }
    strata["joint_eligible"] = [rows for rows in cells.values() if int(rows["joint"]["eligible_decisions"]) > 0]
    output_statistics = {}
    for name, selected in strata.items():
        margins = {method: summarize([cell[method] for cell in selected]) for method in methods}
        pairs = {method: paired(selected, method) for method in methods if method != "physics"}
        if name in summary["strata"]:
            for method, result in margins.items():
                same_statistics(result, summary["strata"][name][method], f"{name}/{method}")
        if name == "all":
            for method, result in pairs.items():
                same_statistics(result, summary["paired_all"][method], f"paired {method}")
        output_statistics[name] = {"metrics": margins, "paired_vs_physics": pairs}
    excluded_cases = Counter(row["status"] for row in row_index.values() if not int(row["eligible_decisions"]))
    report = {
        "status": "PASS",
        "scope": "Independent standard-library aggregation and reconstruction from saved traces and archived polar arrays. "
                 "Clearance values are audited lower bounds from the production trace, recomputed here as individual minima; "
                 "this verification does not run the AIS predictor or rerun native Fossen dynamics.",
        "heldout_episode_cells": len(cells),
        "episode_rows": len(row_index),
        "applied_trace_rows": sum(map(len, traces.values())),
        "maximum_reconstructed_endpoint_error_m": maximum_path_error,
        "maximum_capture_radius_error_m": maximum_capture_error,
        "test_methods": sorted(methods),
        "minimum_truth_margin_cells": minima,
        "inactive_decision_outcomes": dict(excluded_cases),
        "strata": output_statistics,
        "sha256": {**fingerprints, "reproduce/verify_joint.py": digest(Path(__file__)), ARCHIVE.name: digest(ARCHIVE)},
    }
    (STUDY / "independent_metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: report[name] for name in (
        "status", "heldout_episode_cells", "episode_rows", "applied_trace_rows",
        "maximum_reconstructed_endpoint_error_m", "maximum_capture_radius_error_m"
    )}))


if __name__ == "__main__":
    main()
