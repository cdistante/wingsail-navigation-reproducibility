"""Verify the paired stale-AIS retest receipts and compare against the prior guard."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from statistics import median

from run_stale_retest import (
    ADAPTER, CHECKPOINT, MANIFEST, METHODS, REFERENCE, STRENGTH, STUDY, SUPPORT,
    load_reference_inputs, locked_protocol, seasonal,
)
from verify_native_comparison import (
    pair_stats, paired_row, safety_summary, validate_telemetry, verify_training,
)
from verify_seasonal_heldout import (
    check_episode_metadata, compare, completed, digest, grouped, load_json, number,
    read_rows, require, summarize, verify_trace,
)


NATIVE = STUDY / "native"


def read_arm(method, roster, manifest, provenance):
    directory = NATIVE / method
    inputs = load_json(directory / "inputs.json")
    run = load_json(directory / "run.json")
    require(inputs["method"] == run["method"] == method and inputs["scenarios"] == list(roster),
            f"Wrong method or roster in {method}.")
    require(all(inputs[key] == value for key, value in provenance.items()),
            f"Sources, guard configuration or registered data changed after running {method}.")
    require(run["inputs_sha256"] == digest(directory / "inputs.json")
            and run["episodes_sha256"] == digest(directory / "episodes.csv"), f"Changed {method} receipt.")
    require(inputs["package_versions"] == load_reference_inputs(method)["package_versions"],
            f"The {method} retest used different simulator dependency versions.")
    if method == "grpo_progress":
        require(inputs["checkpoint_sha256"] == digest(CHECKPOINT) and inputs["residual_strength"] == STRENGTH
                and inputs["adapter_sha256"] == digest(ADAPTER) and inputs["legacy_support_sha256"] == digest(SUPPORT),
                "The held-out GRPO arm differs from the locked learned policy.")
    else:
        require(inputs["checkpoint"] is None and inputs["residual_strength"] == 0.,
                "The physics arm unexpectedly uses learned steering.")
    records = read_rows(directory / "episodes.csv")
    grouped(records, roster.keys(), {method})
    compare(summarize(records), run["summary"], method)
    require(set(run["trace_sha256es"]) == {row["trajectory"] for row in records}, f"Missing traces: {method}.")
    checks = {}
    for row in records:
        spec = roster[row["scenario"]]
        relative = f"telemetry/{row['scenario']}__{method}.json.gz"
        require(row["trajectory"] == relative, f"Unexpected telemetry path: {row['trajectory']}.")
        path = directory / relative
        require(run["trace_sha256es"][relative] == digest(path), f"Changed trace: {path}.")
        captures = check_episode_metadata(row, spec)
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        checks[f"{row['scenario']}/{method}"] = {
            **verify_trace(row, spec, manifest["routes"][spec["route_id"]],
                           manifest["traffic_profiles"][spec["traffic_profile"]], payload, captures),
            **validate_telemetry(row, payload),
        }
    return records, checks


def before_after(method, after):
    prior = REFERENCE / "native" / method / "episodes.csv"
    before = {row["scenario"]: row for row in read_rows(prior)}
    require(before.keys() == after.keys(), f"The {method} retest differs from the prior scenarios.")
    by_case = {}
    finish_deltas = []
    for case, fresh in after.items():
        old = before[case]
        for field in ("route", "weather_block", "departure_utc", "traffic", "expected_captures", "cap_s"):
            require(fresh[field] == old[field], f"Changed scenario condition {case}/{field}.")
        timed_delta = None
        if completed(old) and completed(fresh):
            timed_delta = number(fresh, "duration_s") - number(old, "duration_s")
            finish_deltas.append(timed_delta)
        by_case[case] = {
            "before_status": old["status"], "after_status": fresh["status"],
            "before_ordered_captures": int(old["capture_count"]),
            "after_ordered_captures": int(fresh["capture_count"]),
            "before_elapsed_s": number(old, "duration_s"), "after_elapsed_s": number(fresh, "duration_s"),
            "after_minus_before_s_on_shared_finish": timed_delta,
            "before_full_truth_margin_m": number(old, "min_truth_margin_m"),
            "after_full_truth_margin_m": number(fresh, "min_truth_margin_m"),
            "before_full_report_margin_m": number(old, "min_report_margin_m"),
            "after_full_report_margin_m": number(fresh, "min_report_margin_m"),
        }
    return {
        "prior_episodes_sha256": digest(prior),
        "before": safety_summary(list(before.values())),
        "after": safety_summary(list(after.values())),
        "shared_finishes": len(finish_deltas),
        "median_after_minus_before_s_on_shared_finishes": median(finish_deltas) if finish_deltas else None,
        "cases": by_case,
    }


def main():
    scenarios, provenance = locked_protocol()
    manifest = load_json(MANIFEST)
    roster = {row["id"]: row for row in scenarios}
    training = verify_training(roster)
    rows = []
    traces = {}
    for method in METHODS:
        records, checks = read_arm(method, roster, manifest, provenance)
        rows.extend(records)
        traces.update(checks)
    cells = grouped(rows, roster.keys(), set(METHODS))
    ordered = {case: cells[case] for case in roster}
    totals = {method: safety_summary([cell[method] for cell in ordered.values()]) for method in METHODS}
    paired = pair_stats(ordered)
    historical = {method: before_after(method, {case: cell[method] for case, cell in ordered.items()})
                  for method in METHODS}
    pairs_path = NATIVE / "paired_episodes.csv"
    seasonal.save_csv(pairs_path, [paired_row(cell) for cell in ordered.values()])
    seasonal.save_json(NATIVE / "independent_comparison.json", {
        "status": "PASS",
        "matched_cases": len(ordered), "episode_rows": len(rows),
        "manifest_sha256": digest(MANIFEST), "checkpoint_sha256": digest(CHECKPOINT),
        "training_partition": training,
        "audited_capture_order": True,
        "full_margin_scope": "Original report and truth envelope over applied native motion, with hulls, 25 m gap, "
                             "one report uncertainty radius and 3 m reserve.",
        "guard": provenance["guard"],
        "summary": totals, "paired": paired, "before_after_guard": historical,
        "trajectories": traces,
        "applied_grpo_changes": sum(result["applied_changes"] for result in traces.values()),
        "audit_sha256": digest(Path(__file__)), "paired_episodes_sha256": digest(pairs_path),
    })
    print(f"Verified {len(rows)} native traces from {len(ordered)} locked pairs.")
    for method, part in totals.items():
        print(method, part)
    print("Matched finish differences", paired)


if __name__ == "__main__":
    main()
