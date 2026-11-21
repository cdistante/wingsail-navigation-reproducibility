"""Rerun the two frozen 12-rollout tack policies on the locked Fossen cases."""

from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
import json
import multiprocessing
from pathlib import Path
import platform
import sys

STUDY = Path(__file__).resolve().parents[1]
V2 = STUDY.parents[2]
BASELINE = STUDY.with_name("grpo_dropout_native")
sys.path.insert(0, str(BASELINE / "reproduce"))

import run_dropout_native as group6

METHODS = group6.METHODS
MANIFEST = group6.MANIFEST
STRENGTH = group6.STRENGTH
seasonal = group6.seasonal
stale = group6.stale
efficient = group6.efficient
PACKAGES = group6.PACKAGES
TRAINER = Path(__file__).with_name("train_group12_grpo.py")
GROUP_SIZE = 12
UPDATES = 32
GROUPS_PER_UPDATE = 3
VALIDATION_INTERVAL = 8


def locked_protocol():
    """Keep the same fixture, guard, sources and simulator across both arms."""
    scenarios, previous_protocol, package_reference = group6.locked_protocol()
    case_ids = [spec["id"] for spec in scenarios]
    prior_runs = {}
    receipt_hashes = {}
    for method in METHODS:
        directory = BASELINE / "native" / method
        inputs = directory / "inputs.json"
        run_path = directory / "run.json"
        episodes = directory / "episodes.csv"
        received = json.loads(inputs.read_text())
        run = json.loads(run_path.read_text())
        checkpoint, training_receipt, trained = group6.checkpoint_for(method)
        if (
            received["method"] != run["method"] or run["method"] != method
            or received["scenarios"] != case_ids or received["probe"]
            or any(received[key] != value for key, value in previous_protocol.items())
            or received["package_versions"] != package_reference["package_versions"]
            or received["checkpoint"] != str(checkpoint.relative_to(V2))
            or received["checkpoint_sha256"] != seasonal.digest(checkpoint)
            or received["training_receipt_sha256"] != seasonal.digest(training_receipt)
            or received["selected_seed"] != trained["selected_seed"]
            or received["selected_update"] != trained["selected_update"]
            or received["adapter_sha256"] != seasonal.digest(stale.ADAPTER)
            or received["legacy_support_sha256"] != seasonal.digest(stale.SUPPORT)
            or received["residual_strength"] != STRENGTH
            or run["inputs_sha256"] != seasonal.digest(inputs)
            or run["episodes_sha256"] != seasonal.digest(episodes)
        ):
            raise ValueError(f"The frozen six-rollout comparison changed for {method}.")
        prior_runs[method] = received
        receipt_hashes[method] = {path.name: seasonal.digest(path) for path in (inputs, run_path, episodes)}
    protocol = {
        **previous_protocol,
        "source_sha256": {**previous_protocol["source_sha256"],
                          str(Path(__file__).relative_to(V2)): seasonal.digest(__file__)},
        "prior_group6_receipts_sha256": receipt_hashes,
        "training_budget": {
            "rollouts_per_group": GROUP_SIZE, "updates_per_seed": UPDATES,
            "groups_per_update": GROUPS_PER_UPDATE, "validation_interval_updates": VALIDATION_INTERVAL,
            "groups_per_seed": UPDATES * GROUPS_PER_UPDATE,
            "rollouts_per_seed": GROUP_SIZE * UPDATES * GROUPS_PER_UPDATE,
        },
    }
    return scenarios, protocol, prior_runs


def checkpoint_for(method):
    """Bind the export to a budget-matched training and checkpoint selection."""
    _, previous_path, original = group6.checkpoint_for(method)
    directory = STUDY / "training" / method
    receipt_path = directory / "training_results.json"
    checkpoint = directory / f"{method}_selected.npz"
    receipt = json.loads(receipt_path.read_text())
    inherited = (
        "reward", "feature_names", "training_scenario_sha256", "training_ids", "validation_ids",
        "training_cases", "validation_cases", "reduced_environment_sha256", "actor_source_sha256",
        "validation_objective", "native_comparison_checkpoint_rule", "numpy_version", "torch_version",
        "physics_validation",
    )
    if (any(receipt[key] != original[key] for key in inherited) or len(receipt["feature_names"]) != 26
            or receipt["method"] != method or receipt["group_size"] != GROUP_SIZE
            or receipt["updates"] != UPDATES or receipt["groups_per_update"] != GROUPS_PER_UPDATE
            or receipt["validation_interval"] != VALIDATION_INTERVAL):
        raise ValueError(f"{method} training differs from the declared rollout-group experiment.")

    expected = [(record["seed"], record["sampled_groups"] * original["group_size"]) for record in original["runs"]]
    actual = [(record["seed"], record["sampled_rollouts"]) for record in receipt["runs"]]
    if actual != expected or any(
        record["sampled_groups"] != UPDATES * GROUPS_PER_UPDATE
        or record["sampled_rollouts"] != record["sampled_groups"] * GROUP_SIZE
        or set(map(int, record["validation_curve"])) != set(range(0, UPDATES + 1, VALIDATION_INTERVAL))
        for record in receipt["runs"]
    ):
        raise ValueError(f"{method} training seeds, rollout counts or validation cadence changed.")
    update_source = group6.PROGRESS_TRAINER if method == METHODS[0] else group6.TRAINER
    if (
        receipt["reference_training_receipt_sha256"] != seasonal.digest(previous_path)
        or receipt["update_source_sha256"] != seasonal.digest(update_source)
        or receipt["progress_sampler_source_sha256"] != seasonal.digest(group6.PROGRESS_TRAINER)
        or receipt["training_source_sha256"] != seasonal.digest(TRAINER)
    ):
        raise ValueError(f"{method} training implementation differs from its receipt.")

    def score(metrics):
        return -metrics["captured"], metrics["mean_capped_objective_s"]

    for record in receipt["runs"]:
        curve = record["validation_curve"]
        best_update = min(range(VALIDATION_INTERVAL, UPDATES + 1, VALIDATION_INTERVAL),
                          key=lambda step: (score(curve[str(step)]), step))
        if (record["chosen_trained_update"] != best_update
                or record["selected_trained_validation"] != curve[str(best_update)]):
            raise ValueError(f"{method} seed {record['seed']} differs from its validation trace.")
    selected = min(receipt["runs"], key=lambda row: (*score(row["selected_trained_validation"]), row["seed"]))
    if (
        receipt["selected_seed"] != selected["seed"]
        or receipt["selected_update"] != selected["chosen_trained_update"]
        or receipt["selected_validation"] != selected["selected_trained_validation"]
        or receipt["selected_checkpoint"] != checkpoint.name
        or receipt["selected_checkpoint_sha256"] != seasonal.digest(checkpoint)
        or receipt["selected_checkpoint_sha256"] != seasonal.digest(checkpoint.with_name(selected["checkpoint"]))
    ):
        raise ValueError(f"The {method} checkpoint differs from the budget-matched validation selection.")
    return checkpoint, receipt_path, receipt


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scenario", help="Run one locked case in a separate probe directory.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    roster, protocol, previous = locked_protocol()
    scenarios = [spec for spec in roster if args.scenario in (None, spec["id"])]
    if not scenarios:
        parser.error(f"Unknown locked case {args.scenario}")
    default = STUDY / "native" / args.method
    if args.scenario:
        default = STUDY / "probe" / args.scenario / args.method
    output = (args.output or default).resolve()
    if args.scenario and output.is_relative_to(STUDY / "native"):
        parser.error("Single-case probes require a separate output directory.")
    packages = {package: version(package) for package in PACKAGES}
    if packages != previous[args.method]["package_versions"]:
        raise ValueError("Simulator dependencies differ from the six-rollout comparison.")
    checkpoint, training_receipt, trained = checkpoint_for(args.method)
    output.mkdir(parents=True, exist_ok=False)
    (output / "telemetry").mkdir()
    receipt = {
        **protocol, "method": args.method, "scenarios": [spec["id"] for spec in scenarios],
        "probe": args.scenario is not None, "workers": min(args.workers, len(scenarios)),
        "platform": platform.platform(), "python": platform.python_version(), "package_versions": packages,
        "checkpoint": str(checkpoint.relative_to(V2)), "checkpoint_sha256": seasonal.digest(checkpoint),
        "training_receipt_sha256": seasonal.digest(training_receipt),
        "selected_seed": trained["selected_seed"], "selected_update": trained["selected_update"],
        "adapter_sha256": seasonal.digest(stale.ADAPTER), "legacy_support_sha256": seasonal.digest(stale.SUPPORT),
        "residual_strength": STRENGTH,
    }
    seasonal.save_json(output / "inputs.json", receipt)
    init_args = (str(output), args.method, str(checkpoint))
    with ProcessPoolExecutor(max_workers=receipt["workers"], mp_context=multiprocessing.get_context("spawn"),
                             initializer=efficient.init_worker, initargs=init_args) as pool:
        rows = []
        for row in pool.map(efficient.earlier.execute, scenarios):
            rows.append(row)
            print(f"{args.method:<13} {row['scenario']} {row['capture_count']}/{row['expected_captures']} "
                  f"{row['status']} {row['duration_s']:.2f}s changes={row['learned_changes']} "
                  f"anticipation={row['anticipation_events']}", flush=True)
    episodes = output / "episodes.csv"
    seasonal.save_csv(episodes, rows)
    seasonal.save_json(output / "run.json", {
        "method": args.method, "inputs_sha256": seasonal.digest(output / "inputs.json"),
        "episodes_sha256": seasonal.digest(episodes), "summary": seasonal.summary(rows),
        "trace_sha256es": {row["trajectory"]: seasonal.digest(output / row["trajectory"]) for row in rows},
    })


if __name__ == "__main__":
    main()
