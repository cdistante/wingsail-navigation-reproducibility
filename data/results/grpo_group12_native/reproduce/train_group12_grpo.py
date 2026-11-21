"""Train matched G=12 policies at the archived total rollout budget."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import torch

STUDY = Path(__file__).resolve().parents[1]
PROGRESS = STUDY.with_name("grpo_progress_native")
DROPOUT = STUDY.with_name("grpo_dropout_native")
sys.path[:0] = [str(PROGRESS / "reproduce"), str(DROPOUT / "reproduce")]

import train_grpo_dropout as dropout
import train_progress_grpo as progress

GROUP_SIZE = 12
GROUPS_PER_UPDATE = 3
ROLLOUTS_PER_SEED = 1152
UPDATES = ROLLOUTS_PER_SEED // (GROUP_SIZE * GROUPS_PER_UPDATE)
VALIDATION_INTERVAL = 8
METHODS = {"grpo_progress": (progress, PROGRESS), "grpo_dropout": (dropout, DROPOUT)}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument("method", choices=METHODS)
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    source, reference = METHODS[args.method]
    old_path = reference / "training" / "training_results.json"
    original = json.loads(old_path.read_text())
    seeds = tuple(row["seed"] for row in original["runs"])
    scenario_file = progress.PREVIOUS_STUDY / "scenarios.json.gz"
    environment_file = progress.PREVIOUS_STUDY / "reproduce/environment.py"
    actor_file = progress.PREVIOUS_STUDY / "reproduce/train_grpo.py"
    progress_sha = digest(progress.__file__)
    source_sha = digest(source.__file__)
    if (seeds != progress.SEEDS or original["group_size"] != 6 or original["updates"] != 64
            or progress.GROUPS_PER_UPDATE != GROUPS_PER_UPDATE or progress.VALIDATION_INTERVAL != 16
            or any(row["sampled_groups"] * original["group_size"] != ROLLOUTS_PER_SEED for row in original["runs"])):
        cli.error("The prior receipt disagrees with the matched rollout budget.")
    if (digest(scenario_file) != original["training_scenario_sha256"]
            or digest(environment_file) != original["reduced_environment_sha256"]
            or digest(actor_file) != original["actor_source_sha256"]
            or source_sha != original["training_source_sha256"]
            or progress_sha != original.get("progress_sampler_source_sha256", source_sha)
            or list(progress.FEATURE_NAMES) != original["feature_names"]):
        cli.error("Training scenarios, feature names, or prior sources differ from the archive.")
    if (np.__version__, torch.__version__) != (original["numpy_version"], original["torch_version"]):
        cli.error("NumPy or PyTorch differs from the archived training environment.")

    output = args.output or STUDY / "training" / args.method
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "training_results.json"
    destination = output / f"{args.method}_selected.npz"
    if receipt_path.exists() or destination.exists() or any(output.glob(f"{args.method}_seed_*.npz")):
        cli.error("Training artifacts already exist. Choose a fresh output directory.")

    torch.set_num_threads(1)
    cases = progress.load_scenarios(scenario_file)
    training = [case for case in cases if case.split == "train"]
    validation = [case for case in cases if case.split == "validation"]
    train_ids = [case.identifier for case in training]
    val_ids = [case.identifier for case in validation]
    if train_ids != original["training_ids"] or val_ids != original["validation_ids"]:
        cli.error("Training and validation splits differ from the archive.")
    physics_metrics = progress.metrics(progress.outcomes(validation, None))
    if physics_metrics != original["physics_validation"]:
        cli.error("Physics validation differs from the archive.")

    # Both original seed trainers read this cadence from the progress module.
    # Halve it alongside the update count to score at identical rollout budgets.
    progress.VALIDATION_INTERVAL = VALIDATION_INTERVAL
    runs = [source.train_seed(seed, training, validation, UPDATES, GROUP_SIZE, output) for seed in seeds]
    for run in runs:
        run["sampled_rollouts"] = run["sampled_groups"] * GROUP_SIZE
        if (run["sampled_rollouts"] != ROLLOUTS_PER_SEED
                or set(run["validation_curve"]) != {"0", "8", "16", "24", "32"}):
            raise RuntimeError("A completed seed has an unexpected budget or validation cadence.")
    selected = min(runs, key=lambda row: (*progress.selection_key(row["selected_trained_validation"]), row["seed"]))
    checkpoint = output / selected["checkpoint"]
    receipt = {
        "method": args.method,
        "training_cases": len(training), "validation_cases": len(validation),
        "training_ids": train_ids, "validation_ids": val_ids,
        "physics_validation": physics_metrics, "reward": original["reward"],
        "validation_objective": original["validation_objective"],
        "native_comparison_checkpoint_rule": original["native_comparison_checkpoint_rule"],
        "updates": UPDATES, "group_size": GROUP_SIZE, "groups_per_update": GROUPS_PER_UPDATE,
        "validation_interval": VALIDATION_INTERVAL, "rollouts_per_seed": ROLLOUTS_PER_SEED,
        "feature_names": list(progress.FEATURE_NAMES),
        "training_scenario_sha256": digest(scenario_file),
        "training_source_sha256": digest(__file__),
        "update_source_sha256": source_sha,
        "progress_sampler_source_sha256": progress_sha,
        "reference_training_receipt_sha256": digest(old_path),
        "reduced_environment_sha256": digest(environment_file),
        "actor_source_sha256": digest(actor_file),
        "numpy_version": np.__version__, "torch_version": torch.__version__,
        "runs": runs, "selected_seed": selected["seed"], "selected_update": selected["chosen_trained_update"],
        "selected_checkpoint_sha256": digest(checkpoint), "selected_checkpoint": destination.name,
        "selected_validation": selected["selected_trained_validation"],
        "selected_beats_physics_on_validation": (
            progress.selection_key(selected["selected_trained_validation"]) < progress.selection_key(physics_metrics)
        ),
    }
    if args.method == "grpo_dropout":
        receipt.update(optimizer=original["optimizer"], rollout_filter=original["rollout_filter"])
    destination.write_bytes(checkpoint.read_bytes())
    staged = receipt_path.with_suffix(".tmp")
    staged.write_text(json.dumps(receipt, indent=2) + "\n")
    staged.replace(receipt_path)
    print(f"{args.method} selected seed {selected['seed']} update {selected['chosen_trained_update']}: "
          f"{selected['selected_trained_validation']}", flush=True)


if __name__ == "__main__":
    main()
