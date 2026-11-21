"""Verify the fixed-case efficient AIS comparison against both earlier guards."""

from __future__ import annotations

from collections import Counter
import gzip
import json
from pathlib import Path
from statistics import mean, median

from run_efficient_retest import (
    CHECKPOINT, MANIFEST, METHODS, STRENGTH, STUDY, baseline_inputs, locked_protocol, prior_dir, seasonal, stale,
)
from verify_native_comparison import pair_stats, paired_row, safety_summary, validate_telemetry, verify_training
from verify_seasonal_heldout import (
    check_episode_metadata, compare, completed, digest, grouped, load_json, number,
    read_rows, require, summarize, verify_trace,
)

NATIVE = STUDY / "native"
COMPARISONS = ("earlier", "robust_only")


def read_arm(method, roster, manifest, provenance):
    directory = NATIVE / method
    inputs = load_json(directory / "inputs.json")
    run = load_json(directory / "run.json")
    require(inputs["method"] == run["method"] == method and not inputs["probe"]
            and inputs["scenarios"] == list(roster), f"Wrong arm or test roster in {method}.")
    require(all(inputs[key] == value for key, value in provenance.items()),
            f"Sources, guard configuration or registered data changed after running {method}.")
    require(run["inputs_sha256"] == digest(directory / "inputs.json")
            and run["episodes_sha256"] == digest(directory / "episodes.csv"), f"Changed {method} receipt.")
    require(inputs["package_versions"] == baseline_inputs(method)["package_versions"],
            f"The {method} retest used different simulator dependency versions.")
    if method == "grpo_progress":
        require(inputs["checkpoint_sha256"] == digest(CHECKPOINT) and inputs["residual_strength"] == STRENGTH
                and inputs["adapter_sha256"] == digest(stale.ADAPTER)
                and inputs["legacy_support_sha256"] == digest(stale.SUPPORT),
                "The efficient AIS GRPO arm differs from the frozen learned policy.")
    else:
        require(inputs["checkpoint"] is None and inputs["residual_strength"] == 0.,
                "The physics arm unexpectedly uses learned steering.")

    records = read_rows(directory / "episodes.csv")
    grouped(records, roster.keys(), {method})
    compare(summarize(records), run["summary"], method)
    require(set(run["trace_sha256es"]) == {row["trajectory"] for row in records}, f"Missing traces: {method}.")
    checks = {}
    for row in records:
        case = row["scenario"]
        spec = roster[case]
        relative = f"telemetry/{case}__{method}.json.gz"
        require(row["trajectory"] == relative, f"Unexpected telemetry path: {row['trajectory']}.")
        path = directory / relative
        require(run["trace_sha256es"][relative] == digest(path), f"Changed trace: {path}.")
        captures = check_episode_metadata(row, spec)
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        events = payload["anticipation_events"]
        require(isinstance(events, list) and all(isinstance(event, dict) for event in events)
                and len(events) == int(row["anticipation_events"]),
                f"Anticipatory event counts differ from the native trace in {case}/{method}.")
        checks[f"{case}/{method}"] = {
            **verify_trace(row, spec, manifest["routes"][spec["route_id"]],
                           manifest["traffic_profiles"][spec["traffic_profile"]], payload, captures),
            **validate_telemetry(row, payload),
            "anticipation_events": len(events),
            "anticipation_reasons": dict(Counter(event.get("reason", "unspecified") for event in events)),
        }
    return records, checks


def compare_baseline(method, fresh, comparison):
    prior = prior_dir(method, comparison) / "episodes.csv"
    before = {row["scenario"]: row for row in read_rows(prior)}
    after = {row["scenario"]: row for row in fresh}
    require(before.keys() == after.keys(), f"The {method} retest differs from the {comparison} scenarios.")
    by_case = {}
    finish_deltas = []
    for case, row in after.items():
        old = before[case]
        for field in ("route", "weather_block", "departure_utc", "traffic", "expected_captures", "cap_s"):
            require(row[field] == old[field], f"Changed scenario condition {case}/{field}.")
        delta = None
        if completed(old) and completed(row):
            delta = number(row, "duration_s") - number(old, "duration_s")
            finish_deltas.append(delta)
        by_case[case] = {
            "before_status": old["status"], "after_status": row["status"],
            "before_ordered_captures": int(old["capture_count"]),
            "after_ordered_captures": int(row["capture_count"]),
            "before_elapsed_s": number(old, "duration_s"), "after_elapsed_s": number(row, "duration_s"),
            "after_minus_before_s_on_shared_finish": delta,
            "before_full_truth_margin_m": number(old, "min_truth_margin_m"),
            "after_full_truth_margin_m": number(row, "min_truth_margin_m"),
            "before_full_report_margin_m": number(old, "min_report_margin_m"),
            "after_full_report_margin_m": number(row, "min_report_margin_m"),
        }
    return {
        "prior_episodes_sha256": digest(prior),
        "before": safety_summary(list(before.values())), "after": safety_summary(fresh),
        "shared_finishes": len(finish_deltas),
        "median_after_minus_before_s_on_shared_finishes": median(finish_deltas) if finish_deltas else None,
        "mean_after_minus_before_s_on_shared_finishes": mean(finish_deltas) if finish_deltas else None,
        "recovered_routes": [case for case in after if not completed(before[case]) and completed(after[case])],
        "lost_routes": [case for case in after if completed(before[case]) and not completed(after[case])],
        "cases": by_case,
    }


def main():
    scenarios, provenance = locked_protocol()
    manifest = load_json(MANIFEST)
    roster = {row["id"]: row for row in scenarios}
    training = verify_training(roster)
    rows = []
    traces = {}
    historical = {name: {} for name in COMPARISONS}
    for method in METHODS:
        records, checks = read_arm(method, roster, manifest, provenance)
        rows.extend(records)
        traces.update(checks)
        for name in COMPARISONS:
            historical[name][method] = compare_baseline(method, records, name)
    cells = grouped(rows, roster.keys(), set(METHODS))
    ordered = {case: cells[case] for case in roster}
    pair_results = []
    for case, cell in ordered.items():
        row = paired_row(cell)
        for method in METHODS:
            row[f"{method}_anticipation_events"] = int(cell[method]["anticipation_events"])
            for name in COMPARISONS:
                history = historical[name][method]["cases"][case]
                row.update({f"{method}_{name}_{key.removeprefix('before_')}": value
                            for key, value in history.items() if key.startswith("before_")})
                row[f"{method}_efficient_minus_{name}_s_on_shared_finish"] = (
                    history["after_minus_before_s_on_shared_finish"])
        pair_results.append(row)
    pairs_path = NATIVE / "paired_episodes.csv"
    seasonal.save_csv(pairs_path, pair_results)
    totals = {method: history["after"] for method, history in historical["earlier"].items()}
    paired = pair_stats(ordered)
    seasonal.save_json(NATIVE / "independent_comparison.json", {
        "status": "PASS",
        "matched_cases": len(ordered), "episode_rows": len(rows),
        "manifest_sha256": digest(MANIFEST), "checkpoint_sha256": digest(CHECKPOINT),
        "training_partition": training,
        "audited_capture_order": True,
        "full_margin_scope": "Original report and truth envelope over applied native motion, with hulls, 25 m gap, "
                             "one report uncertainty radius and 3 m reserve.",
        "guard": provenance["guard"],
        "summary": totals, "paired": paired, "before_after_efficient_ais": historical,
        "trajectories": traces,
        "applied_grpo_changes": sum(result["applied_changes"] for result in traces.values()),
        "anticipation_events_by_method": {
            method: sum(int(cell[method]["anticipation_events"]) for cell in ordered.values())
            for method in METHODS
        },
        "audit_sha256": digest(Path(__file__)), "paired_episodes_sha256": digest(pairs_path),
    })
    print(f"Verified {len(rows)} native traces from {len(ordered)} locked pairs.")
    for method, part in totals.items():
        print(method, part)
        for name in COMPARISONS:
            summary = historical[name][method]
            print(f"{name}: recovered={summary['recovered_routes']} lost={summary['lost_routes']} "
                  f"median finish change={summary['median_after_minus_before_s_on_shared_finishes']}")
    print("Matched finish differences", paired)


if __name__ == "__main__":
    main()
