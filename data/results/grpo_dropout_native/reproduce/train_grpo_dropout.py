"""Train the existing progress-reward tack policy with GRPODropout."""
from __future__ import annotations

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn


STUDY = Path(__file__).resolve().parents[1]
REFERENCE = STUDY.with_name("grpo_progress_native")
sys.path.insert(0, str(REFERENCE / "reproduce"))

import train_progress_grpo as baseline

PAPER = "https://arxiv.org/html/2610.11854v1"
EPSILON = 1e-8


def select_rollouts(advantages, log_probability):
    """Apply the published greedy covariance proxy, restoration and recentering."""
    surprisal = -log_probability
    contribution = advantages * (surprisal - surprisal.mean())
    retained = np.ones(len(advantages), dtype=bool)
    estimate = float(contribution.sum())
    for index in np.argsort(contribution, kind="stable"):
        if estimate >= -EPSILON or contribution[index] >= 0.:
            break
        retained[index] = False
        estimate -= contribution[index]
    retained |= advantages <= 0.
    weights = np.exp(log_probability[retained] - log_probability[retained].max())
    center = np.average(advantages[retained], weights=weights)
    centered = np.zeros_like(advantages)
    centered[retained] = advantages[retained] - center
    return retained, centered


def update(actor, optimizer, scenarios, group_size):
    """Use retained rollouts in the same per-decision GRPO objective."""
    groups = [baseline.sample_group(actor, scenario, group_size, i) for i, scenario in enumerate(scenarios)]
    returns = np.stack([group.returns for group in groups])
    terminal = np.stack([group.terminal_returns for group in groups])
    spread = returns.std(axis=1, keepdims=True)
    advantages = (returns - returns.mean(axis=1, keepdims=True)) / np.maximum(spread, EPSILON)
    lengths = np.concatenate([group.lengths for group in groups])
    decisions = [row for group in groups for row in group.decisions]
    log_probability_sum = np.zeros(len(lengths))
    for _, _, _, old_logp, members in decisions:
        np.add.at(log_probability_sum, members, old_logp)
    # Only genuinely drawn multi-choice actions enter the length normalizer.
    log_probability = log_probability_sum / np.maximum(lengths, 1)
    filtered = [select_rollouts(group_advantage, group_logp) for group_advantage, group_logp 
                in zip(advantages, log_probability.reshape(advantages.shape), strict=True)]
    keep = np.stack([mask for mask, _ in filtered])
    centered = np.stack([values for _, values in filtered])
    learnable = lengths.reshape(keep.shape) > 0
    has_signal = np.any((np.abs(centered) > EPSILON) & learnable & keep, axis=1)
    originally_informative = spread[:, 0] > EPSILON
    diagnostics = {
        "sampled_groups": len(groups),
        "sampled_rollouts": int(keep.size),
        "terminal_informative_groups": int(np.count_nonzero(np.ptp(terminal, axis=1) > EPSILON)),
        "shaped_informative_groups": int(np.count_nonzero(originally_informative)),
        "filtered_groups": int(np.count_nonzero(~keep.all(axis=1))),
        "retained_rollouts": int(keep.sum()),
        "removed_rollouts": int((~keep).sum()),
        "retained_reward_signal_groups": int(has_signal.sum()),
        "informative_groups_losing_reward_signal": int(np.count_nonzero(originally_informative & ~has_signal)),
        "sampled_decisions": int(lengths.sum()),
        "retained_decisions": int(lengths[keep.ravel()].sum()),
        "sampling_surprisal_sum": float(-log_probability_sum.sum()),
    }
    if not decisions or not diagnostics["shaped_informative_groups"] or not diagnostics["retained_decisions"]:
        return diagnostics

    count = diagnostics["sampled_decisions"]
    width = max(mask.shape[1] for _, mask, *_ in decisions)
    features = np.zeros((count, width, len(baseline.FEATURE_NAMES)), dtype=np.float32)
    masks = np.zeros((count, width), dtype=bool)
    start = 0
    for values, allowed, *_ in decisions:
        end = start + len(values)
        features[start:end, :values.shape[1]] = values
        masks[start:end, :allowed.shape[1]] = allowed
        start = end
    actions, previous, members = (np.concatenate(column) for column in zip(*(row[2:] for row in decisions)))
    retained = keep.ravel()[members]
    members = members[retained]
    x, mask = torch.from_numpy(features[retained]), torch.from_numpy(masks[retained])
    action = torch.from_numpy(actions[retained, None])
    old_logp = torch.as_tensor(previous[retained], dtype=torch.float32)
    advantage = torch.as_tensor(centered.ravel()[members], dtype=torch.float32)
    weight = torch.as_tensor(1.0 / lengths[members], dtype=torch.float32)
    with torch.no_grad():
        reference = torch.log_softmax(actor.reference_logits(x).masked_fill(~mask, -1e9), dim=1)
        policy = torch.log_softmax(actor(x).masked_fill(~mask, -1e9), dim=1)
        diagnostics["sampling_categorical_entropy_retained_sum"] = float((-(policy.exp() * policy).sum(dim=1)).sum())
        diagnostics["entropy_measured_decisions"] = diagnostics["retained_decisions"]
    for _ in range(2):
        prob = torch.log_softmax(actor(x).masked_fill(~mask, -1e9), dim=1)
        ratio = torch.exp((prob.gather(1, action)[:, 0] - old_logp).clamp(-20, 20))
        surrogate = torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage)
        kl = (prob.exp() * (prob - reference)).sum(dim=1)
        loss = ((-surrogate + .015 * kl) * weight).sum() / weight.sum()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        optimizer.step()
    return diagnostics


def train_seed(seed, training, validation, updates, group_size, destination):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    actor = baseline.ResidualActor()
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    initial = baseline.metrics(baseline.outcomes(validation, actor.costs))
    validation_curve = {"0": initial}
    training_curve = {}
    best_trained = None
    totals = {}
    for step in range(1, updates + 1):
        chosen = [training[int(index)] for index in rng.integers(len(training), size=baseline.GROUPS_PER_UPDATE)]
        scenarios = [baseline.with_chart_obstacle(case) if rng.random() < .5 else case for case in chosen]
        for name, count in update(actor, optimizer, scenarios, group_size).items():
            totals[name] = totals.get(name, 0) + count
        if step % baseline.VALIDATION_INTERVAL and step != updates:
            continue
        validation_result = baseline.metrics(baseline.outcomes(validation, actor.costs))
        validation_curve[str(step)] = validation_result
        training_curve[str(step)] = dict(totals)
        if best_trained is None or baseline.selection_key(validation_result) < baseline.selection_key(best_trained):
            best_trained = validation_result
            chosen_step = step
            weights = {name: value.detach().numpy().copy() for name, value in actor.state_dict().items()}
        print(json.dumps({"seed": seed, "update": step, "validation": validation_result, **totals}), flush=True)

    checkpoint = destination / f"grpo_dropout_seed_{seed}.npz"
    np.savez_compressed(checkpoint, **weights)
    return {
        "seed": seed, "chosen_trained_update": chosen_step, "selected_trained_validation": best_trained,
        "physics_initialization_validation": initial,
        "trained_beats_initialization_on_validation": baseline.selection_key(best_trained) < baseline.selection_key(initial),
        "validation_curve": validation_curve, "training_curve": training_curve, "checkpoint": checkpoint.name, **totals,
    }


def file_sha(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STUDY / "training")
    args = parser.parse_args()
    original_path = REFERENCE / "training" / "training_results.json"
    original = json.loads(original_path.read_text())
    if file_sha(Path(baseline.__file__)) != original["training_source_sha256"]:
        parser.error("The archived progress sampler source no longer matches its receipt.")
    updates = original["updates"]
    group_size = original["group_size"]
    seeds = [row["seed"] for row in original["runs"]]
    if (updates, group_size, tuple(seeds), baseline.GROUPS_PER_UPDATE) != (64, 6, baseline.SEEDS, 3):
        parser.error("The archived training budget disagrees with the matched protocol.")

    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "grpo_dropout_selected.npz"
    receipts = args.output / "training_results.json"
    if destination.exists() or receipts.exists() or any(
        (args.output / f"grpo_dropout_seed_{seed}.npz").exists() for seed in seeds
    ):
        parser.error("Training receipts already exist. Choose a fresh output directory.")

    torch.set_num_threads(1)
    scenario_file = baseline.PREVIOUS_STUDY / "scenarios.json.gz"
    if file_sha(scenario_file) != original["training_scenario_sha256"]:
        parser.error("The scenario bank no longer matches the archived GRPO run.")
    cases = baseline.load_scenarios(scenario_file)
    training = [case for case in cases if case.split == "train"]
    validation = [case for case in cases if case.split == "validation"]
    training_ids = [case.identifier for case in training]
    validation_ids = [case.identifier for case in validation]
    if training_ids != original["training_ids"] or validation_ids != original["validation_ids"]:
        parser.error("The ordered splits differ from the archived GRPO run.")
    physics_validation = baseline.metrics(baseline.outcomes(validation, None))
    if physics_validation != original["physics_validation"]:
        parser.error("Physics validation differs from the archived GRPO run.")
    if (np.__version__, torch.__version__) != (original["numpy_version"], original["torch_version"]):
        parser.error("NumPy or PyTorch versions differ from the archived GRPO run.")
    reduced_environment = baseline.PREVIOUS_STUDY / "reproduce/environment.py"
    actor_source = baseline.PREVIOUS_STUDY / "reproduce/train_grpo.py"
    if file_sha(reduced_environment) != original["reduced_environment_sha256"] or file_sha(actor_source) != original["actor_source_sha256"]:
        parser.error("The reduced environment or actor source differs from the archived GRPO run.")

    results = [train_seed(seed, training, validation, updates, group_size, args.output) for seed in seeds]
    selected = min(results, key=lambda row: (*baseline.selection_key(row["selected_trained_validation"]), row["seed"]))
    checkpoint = args.output / selected["checkpoint"]
    checkpoint_hash = file_sha(checkpoint)
    details = {
        "training_cases": len(training), "validation_cases": len(validation),
        "training_ids": training_ids, "validation_ids": validation_ids,
        "physics_validation": physics_validation, "reward": original["reward"],
        "validation_objective": original["validation_objective"],
        "native_comparison_checkpoint_rule": original["native_comparison_checkpoint_rule"],
        "updates": updates, "group_size": group_size, "groups_per_update": baseline.GROUPS_PER_UPDATE,
        "feature_names": list(baseline.FEATURE_NAMES),
        "optimizer": {
            "algorithm": "Adam", "learning_rate": 3e-4, "gradient_steps_per_update": 2,
            "clip_ratio": [.8, 1.2], "KL_weight_to_physics": .015, "gradient_norm_clip": 1.0,
            "loss_reduction": "mean over retained learnable rollouts, each averaging genuinely sampled decisions",
        },
        "rollout_filter": {
            "source": PAPER, "sections": "3.2–3.4", "method": "adaptive rollout-level entropy proxy",
            "length_normalizer": "mean old log probability over multi-choice decisions, zero for deterministic rollout",
            "candidate_deletion": "in increasing order of original-group A times centered surprisal until proxy is nonnegative",
            "restoration": "every rollout with nonpositive original advantage",
            "recentering": "subtract old-policy-probability-weighted retained mean, without rescaling",
            "ratio_scope": "same per-decision ratios as GRPO",
        },
        "training_scenario_sha256": file_sha(scenario_file),
        "training_source_sha256": file_sha(Path(__file__)),
        "progress_sampler_source_sha256": file_sha(Path(baseline.__file__)),
        "original_training_receipt_sha256": file_sha(original_path),
        "reduced_environment_sha256": file_sha(reduced_environment),
        "actor_source_sha256": file_sha(actor_source),
        "numpy_version": np.__version__, "torch_version": torch.__version__,
        "runs": results, "selected_seed": selected["seed"], "selected_update": selected["chosen_trained_update"],
        "selected_checkpoint_sha256": checkpoint_hash, "selected_checkpoint": destination.name,
        "selected_validation": selected["selected_trained_validation"],
        "selected_beats_physics_on_validation": (
            baseline.selection_key(selected["selected_trained_validation"]) < baseline.selection_key(physics_validation)
        ),
    }
    destination.write_bytes(checkpoint.read_bytes())
    receipts.write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps({"checkpoint": str(destination), "seed": selected["seed"],
                      "update": selected["chosen_trained_update"], "validation": details["selected_validation"],
                      "physics_validation": physics_validation}), flush=True)


if __name__ == "__main__":
    main()
