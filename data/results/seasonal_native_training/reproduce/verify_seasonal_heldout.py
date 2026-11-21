"""Reconcile expanded held-out results against the locked native episode traces."""
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter, defaultdict
import csv
import gzip
from hashlib import file_digest
import json
from math import asin, cos, isclose, isfinite, radians, sin, sqrt
from pathlib import Path
from statistics import mean, median

import numpy as np


STUDY = Path(__file__).resolve().parents[1]
V2 = STUDY.parents[2]
ENCODER_STUDY = STUDY.with_name("learned_weather_ais_policy")
METHODS = ("physics", "hybrid")
EPS = 1e-6


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def close(a, b, tol=1e-7):
    if a is None or b is None:
        return a is b
    return isclose(a, b, rel_tol=1e-11, abs_tol=tol)


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def load_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def number(row, key):
    return None if row[key] == "" else float(row[key])


def completed(row):
    return row["completed"] == "True"


def compare(actual, recorded, context):
    for key, value in actual.items():
        old = recorded[key]
        equal = value == old if isinstance(value, dict) else close(value, old)
        require(equal, f"{context}: {key} reconstructed {value!r}, recorded {old!r}.")


def summarize(rows):
    captures = [number(row, "duration_s") for row in rows if completed(row)]
    margins = [min(number(row, "min_report_margin_m"), number(row, "min_truth_margin_m"))
               for row in rows if row["min_report_margin_m"] and row["min_truth_margin_m"]]
    times = [number(row, "duration_s") if completed(row) else 2 * number(row, "cap_s") for row in rows]
    return {"cases": len(rows), "completed": len(captures), "status": dict(Counter(r["status"] for r in rows)),
            "fully_audited_cases": len(margins), "nonpositive_clearance_cases": sum(value <= 0. for value in margins),
            "minimum_full_margin_m": min(margins, default=None), "mean_penalized_s": mean(times),
            "median_completed_s": median(captures) if captures else None,
            "learned_changes": sum(int(row["learned_changes"]) for row in rows),
            "maximum_decision_s": max(number(row, "maximum_decision_s") for row in rows),
            "overruns_5s": sum(int(row["overruns_5s"]) for row in rows)}


def grouped(rows, keys, methods):
    cells = defaultdict(dict)
    for row in rows:
        case, policy = row["scenario"], row["method"]
        require(case in keys and policy in methods and policy not in cells[case], f"Unexpected or duplicate cell {case}/{policy}.")
        cells[case][policy] = row
    require(cells.keys() == keys and all(group.keys() == methods for group in cells.values()), "Incomplete paired cells.")
    return cells


def paired(cells):
    a_and_b = [(cell["physics"], cell["hybrid"]) for cell in cells.values()]
    delta = [number(b, "duration_s") - number(a, "duration_s") for a, b in a_and_b if completed(a) and completed(b)]
    return {"joint_captures": len(delta), "median_hybrid_minus_physics_s": median(delta) if delta else None,
            "faster": sum(x < -EPS for x in delta), "slower": sum(x > EPS for x in delta),
            "ties": sum(abs(x) <= EPS for x in delta),
            "hybrid_only": sum(completed(b) and not completed(a) for a, b in a_and_b),
            "physics_only": sum(completed(a) and not completed(b) for a, b in a_and_b)}


def training_partition(manifest_path, rows, report):
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    test = [row for row in rows if row["split"] == "test"]
    allowed_train = {row["id"]: row for row in train}
    allowed_validation = {row["id"]: row for row in validation}
    withheld = {row["id"] for row in test}
    receipt_path = STUDY / "training_receipt.json"
    selection_path = STUDY / "validation_selection.json"
    labelled_path = STUDY / "native_tack_examples.jsonl.gz"
    receipt, selection = load_json(receipt_path), load_json(selection_path)
    expected_model_sha = digest(STUDY / "seasonal_hybrid.npz")
    require(receipt["split_manifest_sha256"] == digest(manifest_path) and
            receipt["model_sha256"] == expected_model_sha and receipt["labels_sha256"] == digest(labelled_path),
            "Frozen manifest, labels, or learned weights differ from training receipt.")
    require(receipt["frozen_encoder_sha256"] == digest(ENCODER_STUDY / "fresh_weather_encoder.pt"),
        "Frozen weather history encoder changed after training.")
    chart = V2 / "data/bathymetry/gebco_2026_n44.0_s37.0_w14.0_e22.0.nc"
    require(receipt["bathymetry_sha256"] == digest(chart), "The GEBCO chart changed after fitting.")
    require(selection["training_receipt_sha256"] == digest(receipt_path) and
            report["validation_selection_sha256"] == digest(selection_path), "Validation or test receipt belongs to another model.")
    for file, expected in receipt["source_sha256"].items():
        if file == "weather_embedding.py":
            source = ENCODER_STUDY / "reproduce" / file
        elif file.startswith("world/"):
            source = V2 / file
        else:
            source = STUDY / "reproduce" / file
        require(digest(source) == expected, f"Pinned source changed: {file}.")
    for season, fields in receipt["forcing_sha256"].items():
        for kind, expected in fields.items():
            require(digest(STUDY / "forcing" / season / f"{kind}.nc") == expected,
                    f"Pinned forcing file changed: {season}/{kind}.")
    by_case = {row["scenario"]: row for row in receipt["training_cases"]}
    require(by_case.keys() == allowed_train.keys() and not withheld.intersection(by_case),
            "Training labeled a held-out or unexpected case.")
    state_counts, branch_counts = Counter(), Counter()
    state_ids = set()
    with gzip.open(labelled_path, "rt") as stream:
        for line in stream:
            label = json.loads(line)
            case = label["scenario"]
            require(case in allowed_train and label["state_id"] not in state_ids, f"Unexpected or duplicate label {case}.")
            scenario = allowed_train[case]
            require(label["block"] == scenario["archive_block"] and str(label["state_id"]).startswith(case + "/"),
                    f"Label belongs to another weather block or case: {case}.")
            launch = np.datetime64(scenario["departure_utc"].removesuffix("Z"), "ns")
            actual = np.datetime64(label["decision_utc"].removesuffix("Z"), "ns")
            require(close((actual - launch) / np.timedelta64(1, "s"), label["clock_s"], 1e-5),
                    f"Label clock differs from frozen training episode: {case}.")
            state_counts[case] += 1
            branch_counts[case] += len(label["branches"])
            state_ids.add(label["state_id"])
    require(sum(branch_counts.values()) == receipt["training_examples"], "Native branch totals differ from training receipt.")
    for case, old in by_case.items():
        require(old["decision_states"] == state_counts[case] and old["native_branches"] == branch_counts[case],
                f"Native label count differs for {case}.")
    validation_rows = read_rows(STUDY / "validation_episodes.csv")
    strength_to_method = {number(row, "strength"): row["method"] for row in validation_rows}
    grouped(validation_rows, allowed_validation.keys(), set(strength_to_method.values()))
    for row in validation_rows:
        check_episode_metadata(row, allowed_validation[row["scenario"]])
        require(row["method"] == strength_to_method[number(row, "strength")], "Validation method and strength mismatch.")
    require(len(allowed_validation) == selection["validation_cases"] and not withheld.intersection(allowed_validation),
            "Validation tested an invalid partition.")
    stats = {key: summarize([row for row in validation_rows if number(row, "strength") == key])
             for key in sorted(strength_to_method)}
    for strength, actual in stats.items():
        compare(actual, selection["metrics"][str(strength)], f"validation strength {strength:g}")
    rank = lambda x: (stats[x]["nonpositive_clearance_cases"], -stats[x]["completed"], stats[x]["mean_penalized_s"], x)
    selected = min(stats, key=rank)
    best_active = min((mix for mix in stats if mix > 0.), key=rank)
    require(selected == selection["selected_strength"] == report["validation_preferred_strength"]
            and best_active == selection["best_nonzero_strength"] == report["hybrid_strength_tested"],
            "Held-out policy strength did not match the frozen validation criterion.")
    return {"training_cases": len(allowed_train), "validation_cases": len(allowed_validation),
            "training_label_states": len(state_ids), "training_native_branches": sum(branch_counts.values()),
            "validation_strengths": sorted(stats), "selected_strength": selected, "tested_nonzero_strength": best_active,
            "training_receipt_sha256": digest(receipt_path), "validation_selection_sha256": digest(selection_path)}


def separation(a, b):
    lat, lon, target_lat, target_lon = map(radians, (*a, *b))
    x = sin((target_lat - lat) / 2)**2 + cos(lat)*cos(target_lat)*sin((target_lon - lon) / 2)**2
    return 2 * 6_371_008.8 * asin(min(1., sqrt(x)))


def check_episode_metadata(row, spec):
    require(row["route"] == spec["route_gpx"] and row["family"] == spec["route_family"]
            and row["weather_block"] == spec["archive_block"] and row["traffic"] == spec["traffic_profile"]
            and row["departure_utc"] == spec["departure_utc"], "Episode metadata disagrees with locked held-out cell.")
    duration = number(row, "duration_s")
    cap = number(row, "cap_s")
    require(0. <= duration <= cap + EPS and cap == spec["max_steps"] * spec["control_dt_s"], "Episode time outside cap.")
    captures = json.loads(row["captures"])
    require(len(captures) == int(row["capture_count"]) <= len(spec["expected_captures"]) == int(row["expected_captures"]),
            "Capture count differs from GPX declaration.")
    require(completed(row) == (row["status"] in {"captured", "completed"} and len(captures) == int(row["expected_captures"])),
            "Completion does not match captured waypoint count.")
    for capture, expected in zip(captures, spec["expected_captures"][:len(captures)], strict=True):
        require(all(capture[key] == value for key, value in expected.items()), "Waypoint phase or order changed.")
        require(0 <= capture["distance_m"] <= spec["capture_radius_m"] + 1e-4, "Capture is outside the 50 m gate.")
    return captures


def seconds_at(utc, start):
    return float((np.datetime64(utc.removesuffix("Z"), "ns") - start) / np.timedelta64(1, "s"))


def delivered_age(profile, elapsed_ns):
    # Integer clock division reconciles delayed delivery without floating-point
    # rounding at a report boundary.
    interval = profile["report_interval_s"]
    ns = 1_000_000_000
    current = ((elapsed_ns - round(profile["delay_s"] * ns)) // round(interval * ns)) * interval
    epoch = current
    while any(left <= epoch < right for left, right in profile["drop_windows_s"]):
        epoch -= interval
    return elapsed_ns / ns - epoch, round((current - epoch) / interval)


def verify_trace(row, spec, route, profile, payload, captures):
    states, controls, decisions = (payload[name] for name in ("states", "controls", "decisions"))
    attempts = payload["control_attempts"]
    require(payload["captures"] == [{key: value for key, value in capture.items() if key not in {"phase", "waypoint"}}
                                    for capture in captures], "Trace and episode captures differ.")
    require(len(states) == len(controls), "Applied control and plant endpoint counts differ.")
    require([{key: value for key, value in item.items() if key != "applied"}
             for item in attempts if item["applied"]] == controls, "Control attempts and applied controls differ.")
    unapplied = [item for item in attempts if not item["applied"]]
    require(len(unapplied) <= 1 and (not unapplied or unapplied[0] == attempts[-1]),
            "An unapplied native stop is missing or misplaced.")
    start = np.datetime64(spec["departure_utc"].removesuffix("Z"), "ns")
    goals = route["waypoints"][1:]
    last = 0.
    capture_index = 0
    time_errors = []
    ranks = []
    lost = []
    margins = {"report": [], "truth": []}
    radius_errors = []
    for index, (point, goal) in enumerate(zip(captures, goals[:len(captures)], strict=True)):
        require(point["name"] == goal["waypoint"] and point["index"] == index, "Native capture index diverged from GPX.")
        distance = separation((point["latitude"], point["longitude"]), (goal["latitude"], goal["longitude"]))
        error = abs(distance - point["distance_m"])
        radius_errors.append(error)
        require(error <= 1e-5 and distance <= spec["capture_radius_m"] + 1e-4,
                "Independent great-circle position does not match the capture.")
        require(close(seconds_at(point["utc"], start), point["elapsed_s"], EPS), "Capture UTC/elapsed clock mismatch.")
        if index:
            require(captures[index-1]["elapsed_s"] <= point["elapsed_s"], "Capture elapsed times run backwards.")
        knot = point.get("integrator_elapsed_s", point["elapsed_s"])
        require(0. <= knot - point["elapsed_s"] <= .05 + EPS, "Capture falls outside the native integrator substep.")
        if knot > 0.:
            candidates = [fix for fix in states if close(fix["elapsed_s"], knot, EPS)]
            require(len(candidates) == 1, "Capture has no matching applied integrator knot.")
            fix = candidates[0]
            require(separation((point["latitude"], point["longitude"]), (fix["latitude"], fix["longitude"]))
                    <= .05 * max(1., abs(fix["speed"])) + .25, "Capture position is inconsistent with applied endpoint.")
    for action, fix in zip(controls, states, strict=True):
        step = action["duration_s"]
        require(close(action["elapsed_s"], last, EPS) and 0. < step <= spec["control_dt_s"] + EPS,
                "Applied interval is missing or discontinuous.")
        require(close(action["native_substeps"] * .05, step, EPS) and action["native_substeps"] > 0,
                "Applied interval is not made up of native 0.05-second substeps.")
        while capture_index < len(captures):
            point = captures[capture_index]
            if point.get("integrator_elapsed_s", point["elapsed_s"]) > last + EPS:
                break
            capture_index += 1
        require(capture_index < len(goals) and action["goal"] == goals[capture_index]["waypoint"],
                "Applied action does not target the next original GPX waypoint.")
        action_utc = np.datetime64(action["utc"].removesuffix("Z"), "ns")
        expected_age, expected_lost = delivered_age(profile, int((action_utc - start).astype(np.int64)))
        recorded_age = action["max_report_age_s"]
        require(close(recorded_age, expected_age, EPS) and action["dropped_report_epochs"] == expected_lost,
                "Applied receiver hold age or epoch loss disagrees with manifest.")
        ages = (seconds_at(action["utc"], start) - last,
                seconds_at(fix["utc"], start) - fix["elapsed_s"],
                fix["elapsed_s"] - last - step)
        time_errors.extend(map(abs, ages))
        require(max(map(abs, ages)) < EPS, "Applied control UTC/elapsed clocks do not agree.")
        require(isfinite(fix["latitude"]) and isfinite(fix["longitude"]), "Nonfinite native plant endpoint.")
        require(close(fix["rudder"], action["rudder_deg"], 1e-5), "Plant rudder differs from the actual applied command.")
        for label in margins:
            value = action[f"{label}_minimum_m"]
            require(isfinite(value), "Recorded swept clearance is nonfinite.")
            margins[label].append(value)
        ranks.append(recorded_age)
        lost.append(expected_lost)
        last = fix["elapsed_s"]
    duration = captures[-1]["elapsed_s"] if completed(row) else last
    require(close(duration, number(row, "duration_s"), EPS), "Episode duration differs from first waypoint entry or stop.")
    if unapplied:
        stop = unapplied[0]
        require(stop["duration_s"] == 0. and stop["reason"] == row["status"]
                and close(stop["elapsed_s"], last, EPS) and close(seconds_at(stop["utc"], start), last, EPS),
                "Terminal rejected control differs from the saved stop.")
    for label in margins:
        require(close(min(margins[label], default=None), number(row, f"min_{label}_margin_m")),
                f"Episodic {label} clearance differs from swept controls.")
    require(close(max(ranks, default=0.), number(row, "maximum_report_age_s"), EPS), "Reported applied AIS age differs.")
    require(max(lost, default=0) == int(row["max_lost_epochs"]), "Reported cumulative AIS drops differ from fixture.")
    for current, previous in zip(decisions[1:], decisions[:-1]):
        require(current["elapsed_s"] >= previous["elapsed_s"], "Control decisions run backwards.")
    require(all(close(seconds_at(decision["utc"], start), decision["elapsed_s"], EPS) for decision in decisions),
            "Tack decision timestamps are not anchored to the manifest.")
    timers = [step["complete_control_seconds"] for step in attempts]
    require(all(isfinite(value) and value >= 0. for value in timers), "Invalid complete-control runtime.")
    require(close(max(timers, default=0.), number(row, "maximum_decision_s"))
            and sum(value >= 5. for value in timers) == int(row["overruns_5s"]), "Complete-control timing summary differs.")
    proposal_changes = sum(item["requested_index"] != item["physics_index"] for item in decisions)
    require(0 <= int(row["learned_changes"]) <= proposal_changes and
            (row["method"] != "physics" or int(row["learned_changes"]) == 0),
            "Recorded learned changes exceed the nonincumbent proposals or appear in physics.")
    return {"applied_intervals": len(controls), "forecast_decisions": len(decisions), "captures": len(captures),
            "maximum_clock_error_s": max(time_errors, default=0.), "maximum_capture_distance_error_m": max(radius_errors, default=0.),
            "maximum_recorded_capture_distance_m": max((c["distance_m"] for c in captures), default=None),
            "nonincumbent_proposals_including_native_scheduled_switches": proposal_changes,
            "authenticated_complete_control_attempts": len(timers), "terminal_unapplied_attempts": len(unapplied),
            "trace_clearance_minimum_m": {
                label: min(values, default=None) for label, values in margins.items()}}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=STUDY / "manifest_expanded.json")
    args = parser.parse_args()
    manifest = load_json(args.manifest)
    require(manifest["roster"] == "expanded", "This audit expects the locked expanded study.")
    indexed = {row["id"]: row for row in manifest["scenarios"] if row["split"] == "test"}
    require(len(indexed) == 12, "The planned expanded test includes exactly 12 cases.")
    episodes = STUDY / "heldout_episodes.csv"
    results = STUDY / "heldout_results.json"
    report = load_json(results)
    require(report["episodes_sha256"] == digest(episodes), "Held-out summary differs from episode CSV.")
    receipt = training_partition(args.manifest, manifest["scenarios"], report)
    rows = read_rows(episodes)
    cells = grouped(rows, indexed.keys(), set(METHODS))
    require(set(report["trace_sha256es"]) == {row["trajectory"] for row in rows}
            and len(report["trace_sha256es"]) == len(rows), "Trace coverage is incomplete or repeated.")
    trace_checks = {}
    for row in rows:
        case = row["scenario"]
        spec = indexed[case]
        route = manifest["routes"][spec["route_id"]]
        key = case, row["method"]
        active = 0. if row["method"] == "physics" else report["hybrid_strength_tested"]
        require(number(row, "strength") == active, "Incorrect held-out policy strength.")
        captures = check_episode_metadata(row, spec)
        path = STUDY / row["trajectory"]
        require(path.resolve().is_relative_to(STUDY / "telemetry")
                and report["trace_sha256es"][row["trajectory"]] == digest(path), f"Trace changed: {key}.")
        try:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                frames = json.load(stream)
            trace_checks[row["trajectory"]] = verify_trace(
                row, spec, route, manifest["traffic_profiles"][spec["traffic_profile"]], frames, captures)
        except ValueError as exc:
            raise ValueError(f"{key}: {exc}") from exc
    stats = {method: summarize([cell[method] for cell in cells.values()]) for method in METHODS}
    for method, derived in stats.items():
        compare(derived, report["summary"][method], method)
    relative = paired(cells)
    compare(relative, report["paired"], "matched policy pairs")
    by_traffic = {}
    for profile in sorted({row["traffic"] for row in rows}):
        members = {key: value for key, value in cells.items() if indexed[key]["traffic_profile"] == profile}
        by_traffic[profile] = {"summary": {method: summarize([cell[method] for cell in members.values()]) for method in METHODS},
                               "paired": paired(members)}
    artifact = {"status": "PASS", "training_split": receipt, "matched_cells": len(cells), "episode_rows": len(rows),
                "scope": "Read-only audit of locked sources, split, capture geometry, traffic receiver ages, native swept clearance logs and complete-control timing logs.",
                "summary": stats, "paired": relative, "by_traffic": by_traffic, "traces": trace_checks,
                "manifest_sha256": digest(args.manifest), "results_sha256": digest(results),
                "episodes_sha256": digest(episodes), "verifier_sha256": digest(Path(__file__).resolve())}
    (STUDY / "independent_seasonal_heldout.json").write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": artifact["status"], "matched_cells": len(cells), "episode_rows": len(rows),
                      "paired": relative}), flush=True)


if __name__ == "__main__":
    main()
