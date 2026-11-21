"""Independently replay the matched G=12 native comparison."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing
from pathlib import Path
import sys


STUDY = Path(__file__).resolve().parents[1]
V2 = STUDY.parents[2]
GROUP6 = STUDY.with_name("grpo_dropout_native")
PROGRESS = STUDY.with_name("grpo_progress_native")
EFFICIENT = STUDY.with_name("grpo_progress_efficient_ais")
sys.path.insert(0, str(GROUP6 / "reproduce"))

from audit_dropout_clearance import (
    audit_efficient_episode, clearance_flag, efficient, historical_rejection, require, summarize,
)


METHODS = ("grpo_progress", "grpo_dropout")
TRAINER = Path(__file__).with_name("train_group12_grpo.py")
GROUP6_TRAINING = {
    "grpo_progress": PROGRESS / "training/training_results.json",
    "grpo_dropout": GROUP6 / "training/training_results.json",
}
GROUP6_UPDATERS = {
    "grpo_progress": PROGRESS / "reproduce/train_progress_grpo.py",
    "grpo_dropout": GROUP6 / "reproduce/train_grpo_dropout.py",
}


def check_training(method, inputs):
    """Verify group size, per-seed sample count, and the validation-only freeze."""
    digest = efficient.seasonal.digest
    directory = STUDY / "training" / method
    checkpoint = directory / f"{method}_selected.npz"
    training_path = directory / "training_results.json"
    training = json.loads(training_path.read_text())
    former_path = GROUP6_TRAINING[method]
    former = json.loads(former_path.read_text())
    require(training["method"] == method
            and (training["group_size"], training["updates"], training["groups_per_update"],
                 training["validation_interval"]) == (12, 32, 3, 8),
            f"{method}: wrong group or update budget.")
    require(former["group_size"] == 6 and former["updates"] == 64,
            f"{method}: group-six comparison receipt changed.")
    for key in ("reward", "feature_names", "training_cases", "validation_cases", "training_ids",
                "validation_ids", "training_scenario_sha256", "reduced_environment_sha256",
                "actor_source_sha256", "validation_objective", "native_comparison_checkpoint_rule",
                "numpy_version", "torch_version"):
        require(training[key] == former[key], f"{method}: {key} differs from the group-six run.")
    require(len(training["feature_names"]) == 26
            and training["reference_training_receipt_sha256"] == digest(former_path)
            and training["update_source_sha256"] == digest(GROUP6_UPDATERS[method])
            and training["progress_sampler_source_sha256"] == digest(GROUP6_UPDATERS["grpo_progress"])
            and training["training_source_sha256"] == digest(TRAINER),
            f"{method}: trained checkpoint sources differ from the recorded protocol.")

    records = training["runs"]
    require([row["seed"] for row in records] == [row["seed"] for row in former["runs"]]
            == [7309, 8779, 10921], f"{method}: seed set changed.")
    for row, previous in zip(records, former["runs"], strict=True):
        require(row["sampled_groups"] == 96 and row["sampled_rollouts"] == 1152
                and previous["sampled_groups"] * former["group_size"] == row["sampled_rollouts"],
                f"{method}: total rollout count differs for seed {row['seed']}.")
        curve = row["validation_curve"]
        require(set(curve) == {"0", "8", "16", "24", "32"}
                and curve["0"] == previous["validation_curve"]["0"],
                f"{method}: validation schedule or starting policy changed for seed {row['seed']}.")
        best = min((8, 16, 24, 32), key=lambda update: (
            -curve[str(update)]["captured"], curve[str(update)]["mean_capped_objective_s"], update,
        ))
        require(row["chosen_trained_update"] == best
                and row["selected_trained_validation"] == curve[str(best)],
                f"{method}: incorrect validation selection for seed {row['seed']}.")

    chosen = min(records, key=lambda row: (
        -row["selected_trained_validation"]["captured"],
        row["selected_trained_validation"]["mean_capped_objective_s"], row["seed"],
    ))
    checkpoint_digest, training_digest = digest(checkpoint), digest(training_path)
    require((V2 / inputs["checkpoint"]).resolve() == checkpoint.resolve()
            and training["selected_checkpoint"] == checkpoint.name
            and inputs["checkpoint_sha256"] == training["selected_checkpoint_sha256"]
            == digest(directory / chosen["checkpoint"]) == checkpoint_digest
            and inputs["training_receipt_sha256"] == training_digest
            and inputs["selected_seed"] == training["selected_seed"] == chosen["seed"]
            and inputs["selected_update"] == training["selected_update"] == chosen["chosen_trained_update"]
            and training["selected_validation"] == chosen["selected_trained_validation"],
            f"{method}: native checkpoint differs from the validation selection.")
    return {
        "checkpoint": str(checkpoint.relative_to(V2)), "checkpoint_sha256": checkpoint_digest,
        "training_receipt_sha256": training_digest, "group6_training_receipt_sha256": digest(former_path),
        "selected_seed": chosen["seed"], "selected_update": chosen["chosen_trained_update"],
        "group_size": training["group_size"], "groups_per_seed": 96, "rollouts_per_seed": 1152,
    }


def locked_jobs(native):
    """Authenticate the new native receipts and the unchanged shared environment."""
    digest = efficient.seasonal.digest
    roster, protocol = efficient.locked_protocol()
    scenarios = {spec["id"]: spec for spec in roster}
    require(len(scenarios) == len(roster) == 12, "Expected all twelve locked cases.")

    jobs, sources = [], {}
    for method in METHODS:
        directory = native / method
        inputs_path, episodes_path, receipt_path = (
            directory / name for name in ("inputs.json", "episodes.csv", "run.json")
        )
        inputs, receipt = (json.loads(path.read_text()) for path in (inputs_path, receipt_path))
        historical = GROUP6 / "native" / method
        past_inputs_path = historical / "inputs.json"
        past_inputs = json.loads(past_inputs_path.read_text())
        past_receipt = json.loads((historical / "run.json").read_text())
        inputs_digest, episodes_digest = digest(inputs_path), digest(episodes_path)
        require(inputs["method"] == receipt["method"] == past_inputs["method"] == past_receipt["method"] == method
                and receipt["inputs_sha256"] == inputs_digest
                and receipt["episodes_sha256"] == episodes_digest
                and past_receipt["inputs_sha256"] == digest(past_inputs_path)
                and past_receipt["episodes_sha256"] == digest(historical / "episodes.csv")
                and not inputs["probe"] and not past_inputs["probe"],
                f"{method}: invalid canonical or historical native receipt.")
        require(inputs["scenarios"] == past_inputs["scenarios"] == list(scenarios),
                f"{method}: native case list changed.")
        for key in ("guard", "split_manifest_sha256", "forcing_sha256", "bathymetry_sha256"):
            require(inputs[key] == past_inputs[key] == protocol[key],
                    f"{method}: {key} differs from efficient AIS.")
        for key in ("package_versions", "adapter_sha256", "legacy_support_sha256", "residual_strength"):
            require(inputs[key] == past_inputs[key], f"{method}: native {key} differs from group six.")
        for path, checksum in protocol["source_sha256"].items():
            require(inputs["source_sha256"][path] == checksum == past_inputs["source_sha256"][path],
                    f"{method}: locked simulator, forcing reader, or guard source changed: {path}.")
        source = check_training(method, inputs)

        with episodes_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        require([row["scenario"] for row in rows] == list(scenarios),
                f"{method}: episode order differs from the roster.")
        trace_digests = receipt["trace_sha256es"]
        require(len(trace_digests) == len(rows), f"{method}: wrong number of trace digests.")
        for row in rows:
            scenario = row["scenario"]
            name = f"telemetry/{scenario}__{method}.json.gz"
            require(row["method"] == method and row["trajectory"] == name,
                    f"{method}/{scenario}: noncanonical trace.")
            jobs.append((scenarios[scenario], row, directory / name, trace_digests[name]))
        sources[method] = {
            **source, "inputs_sha256": inputs_digest, "episodes_sha256": episodes_digest,
            "run_receipt_sha256": digest(receipt_path), "group6_inputs_sha256": digest(past_inputs_path),
        }
    return scenarios, jobs, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, default=STUDY / "native")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, default=STUDY / "independent_clearance.json")
    args = parser.parse_args()
    require(args.workers > 0, "At least one audit worker is required.")
    scenarios, jobs, sources = locked_jobs(args.native)
    seasonal, manifest = efficient.seasonal, efficient.MANIFEST
    seasonal.init(manifest, None, False)
    old_rejection = historical_rejection(scenarios)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=seasonal.init, initargs=(manifest, None, False)) as pool:
        results = list(pool.map(audit_efficient_episode, jobs, chunksize=1))
    indexed = {(case["method"], case["scenario"]): case for case in results}
    unsafe = [{key: case[key] for key in ("scenario", "method", "nonpositive_control_intervals")}
              for case in results if any(case["nonpositive_control_intervals"].values())]
    seasonal.save_json(args.output, {
        "status": "failed_clearance" if unsafe else "verified",
        "scope": "Independent held-forcing Fossen re-integration at 0.05 s, analytical contact trajectories, "
                 "delivered report timing, both original full envelopes and the expanded applied guard. "
                 "Every first logged preview is replayed without target truth or later observations.",
        "manifest_sha256": seasonal.digest(manifest), "auditor_sha256": seasonal.digest(__file__),
        "efficient_auditor_sha256": seasonal.digest(EFFICIENT / "reproduce/audit_efficient_clearance.py"),
        "guard_sha256": seasonal.digest(efficient.GUARD_SOURCE), "run_sources": sources,
        "cases_per_arm": len(scenarios),
        "applied_controls": sum(case["applied_controls"] for case in results),
        "integration_substeps": sum(case["integration_substeps"] for case in results),
        "first_forecast_decisions_replayed": sum(case["first_forecast_causal_replay"] is not None
                                                 for case in results),
        "historical_breach_replay": old_rejection,
        "summary": {method: summarize(results, method) for method in METHODS},
        "paired_clearance_flags": {
            scenario: {method: clearance_flag(indexed[method, scenario]) for method in METHODS}
            for scenario in scenarios
        },
        "unsafe_applied_cases": unsafe, "cases": results,
    })
    require(not unsafe, f"Clearance or guard violations found. See {args.output}.")
    print(f"Verified {len(results)} G=12 swept-clearance and causal traces. {args.output}", flush=True)


if __name__ == "__main__":
    main()
