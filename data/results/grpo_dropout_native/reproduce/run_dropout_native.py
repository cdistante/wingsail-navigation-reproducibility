"""Rerun the matched progress-GRPO and GRPODropout policies in native Fossen."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
import json
import multiprocessing
from pathlib import Path
import platform
import sys

STUDY = Path(__file__).resolve().parents[1]
V2 = STUDY.parents[2]
REFERENCE = STUDY.with_name("grpo_progress_efficient_ais")
sys.path.insert(0, str(REFERENCE / "reproduce"))

import run_efficient_retest as efficient

METHODS = ("grpo_progress", "grpo_dropout")
MANIFEST = efficient.MANIFEST
STRENGTH = efficient.STRENGTH
seasonal = efficient.seasonal
stale = efficient.stale
DROPOUT_CHECKPOINT = STUDY / "training" / "grpo_dropout_selected.npz"
TRAINER = Path(__file__).with_name("train_grpo_dropout.py")
PROGRESS_TRAINER = efficient.CHECKPOINT.parents[1] / "reproduce" / "train_progress_grpo.py"
BASELINE = REFERENCE / "native" / METHODS[0]
PACKAGES = ("numpy", "scipy", "xarray", "casadi", "netCDF4", "pyyaml")


def locked_protocol():
    scenarios, protocol = efficient.locked_protocol()
    inputs = BASELINE / "inputs.json"
    previous = json.loads(inputs.read_text())
    run_path = BASELINE / "run.json"
    run = json.loads(run_path.read_text())
    if (
        previous["scenarios"] != [row["id"] for row in scenarios]
        or previous["method"] != run["method"] or run["method"] != METHODS[0]
        or previous["probe"]
        or any(previous[key] != value for key, value in protocol.items())
        or run["inputs_sha256"] != seasonal.digest(inputs)
        or run["episodes_sha256"] != seasonal.digest(BASELINE / "episodes.csv")
    ):
        raise ValueError("The prior efficient AIS GRPO run differs from the locked comparison.")
    protocol = dict(protocol)
    protocol["progress_training_receipt_sha256"] = protocol.pop("training_receipt_sha256")
    protocol["progress_model_sha256"] = protocol.pop("model_sha256")
    protocol["prior_efficient_receipts_sha256"] = {
        path.name: seasonal.digest(path) for path in (inputs, run_path, BASELINE / "episodes.csv")
    }
    protocol["source_sha256"][str(Path(__file__).relative_to(V2))] = seasonal.digest(__file__)
    return scenarios, protocol, previous


def checkpoint_for(method):
    checkpoint = efficient.CHECKPOINT if method == METHODS[0] else DROPOUT_CHECKPOINT
    receipt_path = checkpoint.with_name("training_results.json")
    receipt = json.loads(receipt_path.read_text())
    original_path = efficient.CHECKPOINT.with_name("training_results.json")
    original = json.loads(original_path.read_text())
    matched_fields = (
        "reward", "updates", "group_size", "feature_names", "training_scenario_sha256",
        "training_ids", "validation_ids", "training_cases", "validation_cases",
        "reduced_environment_sha256", "actor_source_sha256", "validation_objective",
        "native_comparison_checkpoint_rule", "numpy_version", "torch_version",
    )
    if any(receipt[key] != original[key] for key in matched_fields):
        raise ValueError("Training or validation settings differ between policies.")
    budgets = [(record["seed"], record["sampled_groups"]) for record in receipt["runs"]]
    if budgets != [(record["seed"], record["sampled_groups"]) for record in original["runs"]]:
        raise ValueError("Training seeds or rollout budgets differ between policies.")
    if method == METHODS[1] and (
        receipt["original_training_receipt_sha256"] != seasonal.digest(original_path)
        or receipt["progress_sampler_source_sha256"] != seasonal.digest(PROGRESS_TRAINER)
        or receipt["training_source_sha256"] != seasonal.digest(TRAINER)
    ):
        raise ValueError("The dropout training sources differ from the frozen comparison.")
    selected = min(
        receipt["runs"],
        key=lambda row: (-row["selected_trained_validation"]["captured"],
                         row["selected_trained_validation"]["mean_capped_objective_s"], row["seed"]),
    )
    if (
        receipt["selected_seed"] != selected["seed"]
        or receipt["selected_update"] != selected["chosen_trained_update"]
        or receipt["selected_validation"] != selected["selected_trained_validation"]
        or receipt["selected_checkpoint"] != checkpoint.name
        or receipt["selected_checkpoint_sha256"] != seasonal.digest(checkpoint)
        or receipt["selected_checkpoint_sha256"] != seasonal.digest(checkpoint.with_name(selected["checkpoint"]))
    ):
        raise ValueError(f"The {method} checkpoint differs from the validation selection.")
    return checkpoint, receipt_path, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scenario", help="Rerun one locked case in a separate probe directory.")
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
    if packages != previous["package_versions"]:
        raise ValueError("Simulator dependencies differ from the prior efficient AIS run.")
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
