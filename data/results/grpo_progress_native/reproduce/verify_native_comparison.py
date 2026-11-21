"""Audit the paired native GRPO transfer benchmark and make per-case comparisons."""

from __future__ import annotations

from collections import Counter
import gzip
import json
from pathlib import Path
from statistics import mean, median

from run_native_comparison import MANIFEST, METHODS, SEASONAL, STUDY, STRENGTH, V2, seasonal
from verify_seasonal_heldout import (
    check_episode_metadata, close, compare, completed, digest, grouped, load_json, number,
    read_rows, require, summarize, verify_trace,
)


NATIVE = STUDY / "native"
CHECKPOINT = STUDY / "training" / "grpo_progress_selected.npz"
LEGACY_REDUCED = STUDY.with_name("learned_safe_mark_policy")
EPS = 1e-6


def row_margin(row):
    truth = number(row, "min_truth_margin_m")
    report = number(row, "min_report_margin_m")
    return min(truth, report) if truth is not None and report is not None else None


def clear_completion(row):
    margin = row_margin(row)
    return completed(row) and margin is not None and margin > 0.


def validate_telemetry(row, payload):
    events = payload["strategy_evaluations"]
    require(len(events) == int(row["strategy_calls"]), "Number of strategy calls differs from trace.")
    if row["method"] == "physics":
        require(not events, "A learned strategy evaluated a physics-arm episode.")
    decisions = {(item["elapsed_s"], item["utc"], item["goal"]): item for item in payload["decisions"]}
    actions = {(item["elapsed_s"], item["utc"], item["goal"]): item for item in payload["controls"]}
    changes = applied = 0
    for event in events:
        key = (event["elapsed_s"], event["utc"], event["goal"])
        require(key in decisions, f"Strategy evaluation is missing its native decision at {key}.")
        decision = decisions[key]
        require(event["requested_index"] == decision["requested_index"]
                and event["physics_index"] == decision["physics_index"], f"Strategy choice differs at {key}.")
        changed = event["requested_index"] != event["physics_index"]
        require(changed == (event["status"] == "selected"), f"Unexpected changed/fallback state at {key}.")
        if changed:
            require(event["eligible_candidates"] >= 2 and event["native_eta_regret_s"] <= 150. + EPS
                    and event["corrected_gain_s"] > 25.,
                    f"GRPO exceeded its frozen forecast envelope at {key}.")
            changes += 1
            actual = actions.get(key)
            applied += int(actual is not None and actual["choice_id"] == decision["choice_id"]
                           and decision["selected_index"] == event["requested_index"])
    require(changes == int(row["learned_changes"]), "Learned-change tally differs from stored strategy events.")
    return {"strategy_status": dict(Counter(event["status"] for event in events)),
            "requested_changes": changes, "applied_changes": applied}


def safety_summary(rows):
    verified = summarize(rows)
    verified.update({
        "ordered_captures": sum(int(row["capture_count"]) for row in rows),
        "required_captures": sum(int(row["expected_captures"]) for row in rows),
        "clear_completed_routes": sum(clear_completion(row) for row in rows),
        "minimum_truth_margin_m": min((float(row["min_truth_margin_m"]) for row in rows
                                       if row["min_truth_margin_m"]), default=None),
        "minimum_report_margin_m": min((float(row["min_report_margin_m"]) for row in rows
                                        if row["min_report_margin_m"]), default=None),
        "maximum_report_age_s": max(number(row, "maximum_report_age_s") for row in rows),
        "cases_exposing_drops": sum(int(row["max_lost_epochs"]) > 0 for row in rows),
    })
    return verified


def pair_stats(cells):
    common = [(cell["physics"], cell["grpo_progress"]) for cell in cells.values()]
    deltas = [number(b, "duration_s") - number(a, "duration_s")
              for a, b in common if completed(a) and completed(b)]
    safe_deltas = [number(b, "duration_s") - number(a, "duration_s")
                   for a, b in common if clear_completion(a) and clear_completion(b)]
    return {
        "joint_completions": len(deltas), "joint_clear_completions": len(safe_deltas),
        "grpo_only_completions": sum(completed(b) and not completed(a) for a, b in common),
        "physics_only_completions": sum(completed(a) and not completed(b) for a, b in common),
        "grpo_faster": sum(delta < -EPS for delta in deltas),
        "grpo_slower": sum(delta > EPS for delta in deltas),
        "ties": sum(abs(delta) <= EPS for delta in deltas),
        "median_grpo_minus_physics_s": median(deltas) if deltas else None,
        "mean_grpo_minus_physics_s": mean(deltas) if deltas else None,
        "median_grpo_minus_physics_s_both_clear": median(safe_deltas) if safe_deltas else None,
    }


def paired_row(cell):
    physics, grpo = (cell[method] for method in METHODS)
    row = {key: physics[key] for key in ("scenario", "route", "departure_utc", "weather_block", "traffic")}
    row["expected_captures"] = int(physics["expected_captures"])
    for arm, record in zip(METHODS, (physics, grpo), strict=True):
        row.update({
            f"{arm}_status": record["status"], f"{arm}_completed": completed(record),
            f"{arm}_ordered_captures": " ".join(capture["name"] for capture in json.loads(record["captures"])),
            f"{arm}_elapsed_s": number(record, "duration_s"),
            f"{arm}_full_truth_margin_m": number(record, "min_truth_margin_m"),
            f"{arm}_full_report_margin_m": number(record, "min_report_margin_m"),
            f"{arm}_max_report_age_s": number(record, "maximum_report_age_s"),
        })
    row["grpo_minus_physics_s_on_shared_finish"] = (
        number(grpo, "duration_s") - number(physics, "duration_s")
        if completed(physics) and completed(grpo) else None
    )
    row["grpo_learned_changes_requested"] = int(grpo["learned_changes"])
    return row


def verify_training(roster):
    receipt_path = CHECKPOINT.with_name("training_results.json")
    receipt = load_json(receipt_path)
    partition = load_json(LEGACY_REDUCED / "splits.json")["scenarios"]
    train = set(receipt["training_ids"])
    validation = set(receipt["validation_ids"])
    require(train == set(partition["train"]) and validation == set(partition["validation"])
            and train.isdisjoint(validation) and set(roster).isdisjoint(train | validation)
            and len(train) == receipt["training_cases"] and len(validation) == receipt["validation_cases"],
            "Unexpected data in training or checkpoint validation.")
    require(receipt["training_scenario_sha256"] == digest(LEGACY_REDUCED / "scenarios.json.gz")
            and receipt["training_source_sha256"] == digest(Path(__file__).with_name("train_progress_grpo.py"))
            and receipt["reduced_environment_sha256"] == digest(LEGACY_REDUCED / "reproduce" / "environment.py")
            and receipt["actor_source_sha256"] == digest(LEGACY_REDUCED / "reproduce" / "train_grpo.py")
            and receipt["selected_checkpoint"] == CHECKPOINT.name,
            "The training sources or checkpoint name changed.")
    selected = min(receipt["runs"], key=lambda row: (-row["selected_trained_validation"]["captured"],
                    row["selected_trained_validation"]["mean_capped_objective_s"], row["seed"]))
    require(receipt["selected_seed"] == selected["seed"]
            and receipt["selected_update"] == selected["chosen_trained_update"]
            and receipt["selected_validation"] == selected["selected_trained_validation"]
            and receipt["selected_checkpoint_sha256"] == digest(CHECKPOINT)
            and digest(CHECKPOINT.with_name(selected["checkpoint"])) == digest(CHECKPOINT),
            "Native GRPO did not use the checkpoint chosen on synthetic validation.")
    return {"training_receipt_sha256": digest(receipt_path), "training_scenarios": len(train),
            "validation_scenarios": len(validation), "selected_seed": receipt["selected_seed"],
            "selected_update": receipt["selected_update"]}


def compare_previous_physics(rows):
    previous = {row["scenario"]: row for row in read_rows(SEASONAL / "heldout_episodes.csv")
                if row["method"] == "physics"}
    require(previous.keys() == {row["scenario"] for row in rows}, "Historical physics roster differs.")
    checks = {}
    for row in rows:
        old = previous[row["scenario"]]
        same = row["status"] == old["status"] and json.loads(row["captures"]) == json.loads(old["captures"])
        deviations = {}
        for field in ("duration_s", "min_truth_margin_m", "min_report_margin_m"):
            new_value, old_value = number(row, field), number(old, field)
            deviations[field] = (new_value - old_value) if new_value is not None and old_value is not None else None
            same &= close(new_value, old_value, EPS)
        checks[row["scenario"]] = {"recorded_outcomes_match": same, "numeric_change": deviations}
    return checks


def main():
    manifest, forcing = seasonal.source_inputs(MANIFEST)
    roster = {row["id"]: row for row in manifest["scenarios"] if row["split"] == "test"}
    require(len(roster) == 12, "Incomplete or altered seasonal test roster.")
    training = verify_training(roster)
    provenance = seasonal.provenance(MANIFEST, forcing)
    provenance["source_sha256"]["grpo_progress_native/reproduce/run_native_comparison.py"] = digest(
        Path(__file__).with_name("run_native_comparison.py"))
    provenance["source_sha256"]["world/learned_mark_policy.py"] = digest(V2 / "world/learned_mark_policy.py")

    rows = []
    checks = {}
    input_receipts = {}
    for method in METHODS:
        directory = NATIVE / method
        inputs = load_json(directory / "inputs.json")
        run = load_json(directory / "run.json")
        require(inputs["method"] == method and run["method"] == method
                and inputs["scenarios"] == list(roster), f"Wrong method or roster in {method}.")
        require(all(inputs[key] == value for key, value in provenance.items()),
                f"Sources, forcing, chart or route manifest changed after running {method}.")
        require(run["inputs_sha256"] == digest(directory / "inputs.json")
                and run["episodes_sha256"] == digest(directory / "episodes.csv"), f"Changed {method} receipt.")
        if method == "grpo_progress":
            require(inputs["checkpoint_sha256"] == digest(CHECKPOINT)
                    and inputs["residual_strength"] == STRENGTH
                    and inputs["adapter_sha256"] == digest(Path(__file__).with_name("native_grpo.py"))
                    and inputs["legacy_support_sha256"] == digest(STUDY.with_name("learned_safe_mark_policy")
                                                                  / "learned_mark_policy.json"),
                    "Native run differs from the frozen model, adapter or guard.")
        else:
            require(inputs["checkpoint"] is None and inputs["residual_strength"] == 0.,
                    "Physics was initialized with a learned policy.")
        input_receipts[method] = inputs
        records = read_rows(directory / "episodes.csv")
        require(all(row["method"] == method for row in records), f"An episode is mislabeled in {method}.")
        grouped(records, roster.keys(), {method})
        compare(summarize(records), run["summary"], method)
        require(set(run["trace_sha256es"]) == {row["trajectory"] for row in records}, f"Missing traces: {method}.")
        for row in records:
            spec = roster[row["scenario"]]
            path = directory / row["trajectory"]
            require(path.is_relative_to(directory / "telemetry") and run["trace_sha256es"][row["trajectory"]] == digest(path),
                    f"Changed trace: {path}.")
            captures = check_episode_metadata(row, spec)
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                payload = json.load(stream)
            checks[f'{row["scenario"]}/{method}'] = {
                **verify_trace(row, spec, manifest["routes"][spec["route_id"]],
                               manifest["traffic_profiles"][spec["traffic_profile"]], payload, captures),
                **validate_telemetry(row, payload),
            }
        rows.extend(records)

    require(input_receipts["physics"]["package_versions"] == input_receipts["grpo_progress"]["package_versions"],
            "The paired arms used different simulator package versions.")
    cells = grouped(rows, roster.keys(), set(METHODS))
    ordered = {case: cells[case] for case in roster}
    totals = {method: safety_summary([cell[method] for cell in ordered.values()]) for method in METHODS}
    pairs = pair_stats(ordered)
    by_traffic = {}
    for profile in sorted({spec["traffic_profile"] for spec in roster.values()}):
        group = {case: cell for case, cell in ordered.items() if roster[case]["traffic_profile"] == profile}
        by_traffic[profile] = {"summary": {method: safety_summary([cell[method] for cell in group.values()])
                                           for method in METHODS}, "paired": pair_stats(group)}
    historical_parity = compare_previous_physics([cell["physics"] for cell in ordered.values()])
    seasonal.save_csv(NATIVE / "paired_episodes.csv", [paired_row(cell) for cell in ordered.values()])
    seasonal.save_json(
        NATIVE / "independent_comparison.json",
        {"status": "PASS", "matched_cases": len(cells), "episode_rows": len(rows),
         "manifest_sha256": digest(MANIFEST), "checkpoint_sha256": digest(CHECKPOINT),
         "training_partition": training,
         "route_departure_groups": len({(spec["route_id"], spec["departure_utc"]) for spec in roster.values()}),
         "audited_capture_order": True, "full_margin_scope": "Native swept bounds with hulls, 25 m, report uncertainty and 3 m reserve.",
         "summary": totals, "paired": pairs, "by_traffic": by_traffic, "trajectories": checks,
         "historical_physics_parity": historical_parity,
         "applied_grpo_changes": sum(result["applied_changes"] for result in checks.values()),
         "audit_sha256": digest(Path(__file__)),
         "paired_episodes_sha256": digest(NATIVE / "paired_episodes.csv")},
    )
    print(f"Verified {len(rows)} native traces from {len(cells)} pairs.")
    for method, part in totals.items():
        print(method, part)
    print("Matched finish differences", pairs)


if __name__ == "__main__":
    main()
