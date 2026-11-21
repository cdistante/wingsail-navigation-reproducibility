"""Verify the matched group-size reruns and compare them with the six-rollout arms."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median

from run_group12_native import BASELINE, MANIFEST, METHODS, STUDY, checkpoint_for, locked_protocol, seasonal

import verify_dropout_native as group6
from verify_native_comparison import EPS, safety_summary
from verify_seasonal_heldout import completed, digest, grouped, load_json, number, read_rows, require

NATIVE = STUDY / "native"


def paired_group_change(past, current):
    """Pair shared finishes for one method across the two training budgets."""
    deltas = [number(current[case], "duration_s") - number(old, "duration_s")
              for case, old in past.items() if completed(old) and completed(current[case])]
    return {
        "joint_completions": len(deltas),
        "new_finishes": [case for case, old in past.items() if not completed(old) and completed(current[case])],
        "lost_finishes": [case for case, old in past.items() if completed(old) and not completed(current[case])],
        "group12_faster": sum(delta < -EPS for delta in deltas),
        "group12_slower": sum(delta > EPS for delta in deltas),
        "ties": sum(abs(delta) <= EPS for delta in deltas),
        "median_group12_minus_group6_s": median(deltas) if deltas else None,
        "mean_group12_minus_group6_s": mean(deltas) if deltas else None,
    }


def paired_row(cell, past):
    row = group6.paired_row(cell)
    for method in METHODS:
        old = past[method][row["scenario"]]
        new = cell[method]
        prefix = f"{method}_group6"
        row.update({
            f"{prefix}_status": old["status"],
            f"{prefix}_completed": completed(old),
            f"{prefix}_ordered_captures": " ".join(mark["name"] for mark in json.loads(old["captures"])),
            f"{prefix}_elapsed_s": number(old, "duration_s"),
            f"{prefix}_full_truth_margin_m": number(old, "min_truth_margin_m"),
            f"{prefix}_full_report_margin_m": number(old, "min_report_margin_m"),
            f"{prefix}_learned_changes": int(old["learned_changes"]),
            f"{method}_group12_minus_group6_s_on_shared_finish": (
                number(new, "duration_s") - number(old, "duration_s")
                if completed(new) and completed(old) else None
            ),
        })
        for center in ("truth", "report"):
            before, after = (number(r, f"min_{center}_margin_m") for r in (old, new))
            row[f"{method}_group12_minus_group6_full_{center}_margin_m"] = (
                after - before if before is not None and after is not None else None
            )
    return row


def main():
    scenarios, protocol, previous = locked_protocol()
    manifest = load_json(MANIFEST)
    roster = {spec["id"]: spec for spec in scenarios}
    past = {}
    records = []
    traces = {}
    training = {}

    # The existing waypoint, time and trace verifier uses the same on-disk
    # schema. Point its input root at this study's 12-rollout run directory.
    group6.NATIVE = NATIVE
    for method in METHODS:
        old_rows = read_rows(BASELINE / "native" / method / "episodes.csv")
        grouped(old_rows, roster.keys(), {method})
        past[method] = {row["scenario"]: row for row in old_rows}
        checkpoint, training_path, trained = checkpoint_for(method)
        arm, checks = group6.read_arm(method, roster, manifest, protocol, previous[method],
                                      checkpoint, training_path, trained)
        records.extend(arm)
        traces.update(checks)
        training[method] = {
            "checkpoint_sha256": digest(checkpoint),
            "training_receipt_sha256": digest(training_path),
            "selected_seed": trained["selected_seed"], "selected_update": trained["selected_update"],
            "selected_validation": trained["selected_validation"],
            "seeds": [row["seed"] for row in trained["runs"]],
            "groups_per_seed": [row["sampled_groups"] for row in trained["runs"]],
            "rollouts_per_seed": [row["sampled_rollouts"] for row in trained["runs"]],
            "rollouts_per_group": trained["group_size"],
        }

    cells = grouped(records, roster.keys(), set(METHODS))
    ordered = {case: cells[case] for case in roster}
    summary = {method: safety_summary([cell[method] for cell in ordered.values()]) for method in METHODS}
    past_summary = {method: safety_summary(list(past[method].values())) for method in METHODS}
    paired = group6.finish_statistics(ordered)
    paired_vs_group6 = {
        method: paired_group_change(past[method], {case: cell[method] for case, cell in ordered.items()})
        for method in METHODS
    }
    pair_path = NATIVE / "paired_episodes.csv"
    seasonal.save_csv(pair_path, [paired_row(cell, past) for cell in ordered.values()])
    seasonal.save_json(NATIVE / "independent_comparison.json", {
        "status": "PASS", "matched_cases": len(ordered), "episode_rows": len(records),
        "manifest_sha256": digest(MANIFEST),
        "training": training, "training_budget": protocol["training_budget"], "audited_capture_order": True,
        "guard": protocol["guard"],
        "full_margin_scope": "Original full report and truth envelopes around native applied motion, including "
                             "both hulls, the 25 m gap, report uncertainty and 3 m reserve.",
        "summary": summary, "paired": paired,
        "group6_summary": past_summary, "paired_vs_group6": paired_vs_group6,
        "trajectories": traces, "audit_sha256": digest(Path(__file__)),
        "trace_verifier_sha256": digest(Path(group6.__file__)),
        "paired_episodes_sha256": digest(pair_path),
    })
    print(f"Verified {len(records)} traces from {len(ordered)} locked Fossen pairs.")
    for method, outcome in summary.items():
        print(method, outcome)
    print("GRPODropout versus GRPO paired finishes", paired)
    print("Paired difference from six-rollout training", paired_vs_group6)


if __name__ == "__main__":
    main()
