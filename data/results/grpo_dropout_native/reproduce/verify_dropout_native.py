"""Verify the matched Fossen runs and compare GRPODropout with progress GRPO."""

from __future__ import annotations

from collections import Counter
import gzip
import json
from pathlib import Path
from statistics import mean, median

from run_dropout_native import (
    BASELINE, MANIFEST, METHODS, STRENGTH, STUDY, V2, checkpoint_for, locked_protocol, seasonal, stale,
)
from verify_native_comparison import EPS, clear_completion, safety_summary, validate_telemetry, verify_training
from verify_seasonal_heldout import (
    check_episode_metadata, close, compare, completed, digest, grouped, load_json, number,
    read_rows, require, summarize, verify_trace,
)

NATIVE = STUDY / "native"


def read_arm(method, roster, manifest, protocol, previous, checkpoint, training_path, trained):
    directory = NATIVE / method
    inputs = load_json(directory / "inputs.json")
    run = load_json(directory / "run.json")
    require(
        inputs["method"] == run["method"] == method and not inputs["probe"] and inputs["scenarios"] == list(roster),
        f"Wrong method or case roster for {method}.",
    )
    require(all(inputs[key] == value for key, value in protocol.items()), f"Inputs changed in {method}.")
    require(run["inputs_sha256"] == digest(directory / "inputs.json")
            and run["episodes_sha256"] == digest(directory / "episodes.csv"), f"Receipts changed in {method}.")
    require(inputs["package_versions"] == previous["package_versions"],
            f"Simulator packages differ from the previous run in {method}.")
    require(
        inputs["checkpoint"] == str(checkpoint.relative_to(V2))
        and inputs["checkpoint_sha256"] == digest(checkpoint)
        and inputs["training_receipt_sha256"] == digest(training_path)
        and inputs["selected_seed"] == trained["selected_seed"]
        and inputs["selected_update"] == trained["selected_update"]
        and inputs["adapter_sha256"] == digest(stale.ADAPTER)
        and inputs["legacy_support_sha256"] == digest(stale.SUPPORT)
        and inputs["residual_strength"] == STRENGTH,
        f"Policy inputs differ from the frozen checkpoint and native adapter in {method}.",
    )
    records = read_rows(directory / "episodes.csv")
    grouped(records, roster.keys(), {method})
    compare(summarize(records), run["summary"], method)
    require(set(run["trace_sha256es"]) == {row["trajectory"] for row in records}, f"Missing traces in {method}.")
    checks = {}
    for row in records:
        case = row["scenario"]
        spec = roster[case]
        reference = f"telemetry/{case}__{method}.json.gz"
        path = directory / reference
        require(reference == row["trajectory"] and digest(path) == run["trace_sha256es"][reference],
                f"Trace differs from the recorded receipt in {case}/{method}.")
        captures = check_episode_metadata(row, spec)
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        events = payload["anticipation_events"]
        require(isinstance(events, list) and all(isinstance(event, dict) for event in events)
                and len(events) == int(row["anticipation_events"]),
                f"Anticipation counts differ from the trace in {case}/{method}.")
        checks[f"{case}/{method}"] = {
            **verify_trace(row, spec, manifest["routes"][spec["route_id"]],
                           manifest["traffic_profiles"][spec["traffic_profile"]], payload, captures),
            **validate_telemetry(row, payload),
            "anticipation_events": len(events),
            "anticipation_reasons": dict(Counter(event.get("reason", "unspecified") for event in events)),
        }
    return records, checks


def finish_statistics(cells):
    matched = [tuple(cell[method] for method in METHODS) for cell in cells.values()]
    deltas = [number(dropout, "duration_s") - number(progress, "duration_s")
              for progress, dropout in matched if completed(progress) and completed(dropout)]
    clear_deltas = [number(dropout, "duration_s") - number(progress, "duration_s")
                    for progress, dropout in matched if clear_completion(progress) and clear_completion(dropout)]
    return {
        "joint_completions": len(deltas),
        "joint_clear_completions": len(clear_deltas),
        "dropout_only_finishes": sum(completed(b) and not completed(a) for a, b in matched),
        "progress_only_finishes": sum(completed(a) and not completed(b) for a, b in matched),
        "dropout_faster": sum(delta < -EPS for delta in deltas),
        "dropout_slower": sum(delta > EPS for delta in deltas),
        "ties": sum(abs(delta) <= EPS for delta in deltas),
        "median_dropout_minus_progress_s": median(deltas) if deltas else None,
        "mean_dropout_minus_progress_s": mean(deltas) if deltas else None,
        "median_dropout_minus_progress_s_both_clear": median(clear_deltas) if clear_deltas else None,
    }


def paired_row(cell):
    progress, dropout = (cell[method] for method in METHODS)
    result = {key: progress[key] for key in ("scenario", "route", "departure_utc", "weather_block", "traffic")}
    result["expected_captures"] = int(progress["expected_captures"])
    for method in METHODS:
        row = cell[method]
        result.update({
            f"{method}_status": row["status"],
            f"{method}_completed": completed(row),
            f"{method}_ordered_captures": " ".join(mark["name"] for mark in json.loads(row["captures"])),
            f"{method}_elapsed_s": number(row, "duration_s"),
            f"{method}_full_truth_margin_m": number(row, "min_truth_margin_m"),
            f"{method}_full_report_margin_m": number(row, "min_report_margin_m"),
            f"{method}_max_report_age_s": number(row, "maximum_report_age_s"),
            f"{method}_learned_changes": int(row["learned_changes"]),
            f"{method}_anticipation_events": int(row["anticipation_events"]),
        })
    result["dropout_minus_progress_s_on_shared_finish"] = (
        number(dropout, "duration_s") - number(progress, "duration_s")
        if completed(dropout) and completed(progress) else None
    )
    for center in ("truth", "report"):
        a, b = (number(row, f"min_{center}_margin_m") for row in (progress, dropout))
        result[f"dropout_minus_progress_full_{center}_margin_m"] = b - a if a is not None and b is not None else None
    return result


def baseline_parity(cells):
    previous = {row["scenario"]: row for row in read_rows(BASELINE / "episodes.csv")}
    require(previous.keys() == cells.keys(), "The earlier GRPO baseline roster differs.")
    identical = 0
    deviations = {}
    for case, group in cells.items():
        old, new = previous[case], group[METHODS[0]]
        changed = [
            field for field in ("status", "captures", "capture_count", "learned_changes", "anticipation_events")
            if old[field] != new[field]
        ]
        for field in ("duration_s", "min_truth_margin_m", "min_report_margin_m"):
            if not close(number(old, field), number(new, field), EPS):
                changed.append(field)
        if changed:
            deviations[case] = changed
        else:
            identical += 1
    return {"same_outcome_cases": identical, "changed_fields": deviations}


def main():
    scenarios, protocol, previous = locked_protocol()
    manifest = load_json(MANIFEST)
    roster = {spec["id"]: spec for spec in scenarios}
    verify_training(roster)
    records = []
    traces = {}
    training = {}
    for method in METHODS:
        checkpoint, training_path, trained = checkpoint_for(method)
        rows, checks = read_arm(method, roster, manifest, protocol, previous, checkpoint, training_path, trained)
        records.extend(rows)
        traces.update(checks)
        training[method] = {
            "checkpoint_sha256": digest(checkpoint),
            "training_receipt_sha256": digest(training_path),
            "selected_seed": trained["selected_seed"], "selected_update": trained["selected_update"],
            "selected_validation": trained["selected_validation"],
            "seeds": [row["seed"] for row in trained["runs"]],
            "groups_per_seed": [row["sampled_groups"] for row in trained["runs"]],
            "rollouts_per_group": trained["group_size"],
        }
    cells = grouped(records, roster.keys(), set(METHODS))
    ordered = {case: cells[case] for case in roster}
    summary = {method: safety_summary([cell[method] for cell in ordered.values()]) for method in METHODS}
    paired = finish_statistics(ordered)
    parity = baseline_parity(ordered)
    pair_path = NATIVE / "paired_episodes.csv"
    seasonal.save_csv(pair_path, [paired_row(cell) for cell in ordered.values()])
    seasonal.save_json(NATIVE / "independent_comparison.json", {
        "status": "PASS", "matched_cases": len(ordered), "episode_rows": len(records),
        "manifest_sha256": digest(MANIFEST),
        "training": training, "audited_capture_order": True, "guard": protocol["guard"],
        "full_margin_scope": "Original full report and truth envelopes around native applied motion, with hulls, "
                             "25 m gap, report uncertainty and 3 m reserve.",
        "summary": summary, "paired": paired, "progress_rerun_matches_earlier": parity, "trajectories": traces,
        "audit_sha256": digest(Path(__file__)), "paired_episodes_sha256": digest(pair_path),
    })
    print(f"Verified {len(records)} native traces from {len(ordered)} locked pairs.")
    for method, outcome in summary.items():
        print(method, outcome)
    print("Matched finish differences", paired)
    print("GRPO rerun parity", parity)


if __name__ == "__main__":
    main()
