"""Train a physics-initialized tack ranker with signed waypoint progress."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn


STUDY = Path(__file__).resolve().parents[1]
PREVIOUS_STUDY = STUDY.parent / "learned_safe_mark_policy"
REPOSITORY = STUDY.parents[2]
sys.path.insert(0, str(REPOSITORY))
sys.path.insert(0, str(PREVIOUS_STUDY / "reproduce"))

from environment import INTERVAL_SECONDS, evaluate, load_scenarios, with_chart_obstacle
from train_grpo import ResidualActor, metrics, outcomes
from world.learned_mark_policy import FEATURE_NAMES, MAX_SECONDS, TIMEOUT_COST_SECONDS


PROGRESS_SECONDS = 3600.0
PROGRESS_DISCOUNT_PER_WINDOW = .985
BLOCK_COST_SECONDS = 7800.0
GROUPS_PER_UPDATE = 3
VALIDATION_INTERVAL = 16
SEEDS = (7309, 8779, 10921)


@dataclass(frozen=True)
class Group:
    returns: np.ndarray
    terminal_returns: np.ndarray
    lengths: np.ndarray
    decisions: list


@torch.inference_mode()
def sample_group(actor, scenario, group_size, group_index):
    """Simultaneous reduced rollouts from the same bow, mark and forcing."""
    target = np.repeat(scenario.target[None, :], group_size, axis=0)
    bow = np.full(group_size, scenario.bow)
    active = np.arange(group_size)
    timeout = -(MAX_SECONDS + TIMEOUT_COST_SECONDS)
    terminal = np.full(group_size, timeout)
    blocked = np.zeros(group_size, dtype=bool)
    progress = np.zeros(group_size)
    potential = np.zeros(group_size)
    lengths = np.zeros(group_size, dtype=int)
    initial_gap = max(float(np.linalg.norm(scenario.target)) - scenario.radius, 1.0)
    elapsed = 0.0
    decisions = []
    while len(active) and elapsed < MAX_SECONDS:
        batch, dt, allowed_batch = evaluate(scenario, elapsed, target[active], bow[active])
        can_move = allowed_batch.any(axis=1)
        blocked[active[~can_move]] = True
        rows = np.flatnonzero(can_move)
        if not len(rows):
            break
        active = active[rows]
        features = batch.features[rows]
        allowed = allowed_batch[rows]
        hits = np.where(allowed, batch.captured_after_s[rows], np.inf)
        finish = np.isfinite(hits).any(axis=1)
        selected = hits.argmin(axis=1)
        sampling = np.flatnonzero(~finish)
        if len(sampling):
            x = torch.from_numpy(features[sampling])
            mask = torch.from_numpy(allowed[sampling])
            dist = torch.distributions.Categorical(logits=actor(x).masked_fill(~mask, -torch.inf))
            sampled = dist.sample()
            selected[sampling] = sampled.numpy()
            learn = allowed[sampling].sum(axis=1) > 1
            if learn.any():
                members = active[sampling[learn]]
                lengths[members] += 1
                decisions.append((
                    features[sampling[learn]], allowed[sampling[learn]], sampled.numpy()[learn],
                    dist.log_prob(sampled).numpy()[learn], members + group_index * group_size,
                ))

        travel = rows[~finish], selected[~finish]
        next_target = batch.remaining_target[travel]
        gap = np.maximum(np.linalg.norm(next_target, axis=1) - scenario.radius, 0.0)
        next_potential = np.ones(len(active))
        next_potential[~finish] = np.clip(1.0 - gap / initial_gap, -1.0, 1.0)
        discount = PROGRESS_DISCOUNT_PER_WINDOW ** (elapsed / INTERVAL_SECONDS)
        progress[active] += discount * (next_potential - potential[active])
        potential[active] = next_potential
        terminal[active[finish]] = -(elapsed + hits[np.arange(len(rows)), selected])[finish]
        target[active[~finish]] = next_target
        bow[active[~finish]] = batch.end_bow[travel]
        active = active[~finish]
        elapsed += dt
    reward = terminal + PROGRESS_SECONDS * progress - BLOCK_COST_SECONDS * blocked
    return Group(reward, terminal, lengths, decisions)


def update(actor, optimizer, scenarios, group_size):
    """Group-normalized advantage over the clipped same-start policy loss."""
    groups = [sample_group(actor, scenario, group_size, i) for i, scenario in enumerate(scenarios)]
    returns = np.stack([group.returns for group in groups])
    terminal = np.stack([group.terminal_returns for group in groups])
    spread = returns.std(axis=1, keepdims=True)
    advantages = ((returns - returns.mean(axis=1, keepdims=True)) / np.maximum(spread, 1e-8)).ravel()
    lengths = np.concatenate([group.lengths for group in groups])
    decisions = [row for group in groups for row in group.decisions]
    diagnostics = {
        "sampled_groups": len(groups),
        "terminal_informative_groups": int(np.count_nonzero(np.ptp(terminal, axis=1) > 1e-8)),
        "shaped_informative_groups": int(np.count_nonzero(spread > 1e-8)),
        "sampled_decisions": int(lengths.sum()),
    }
    if not decisions or not diagnostics["shaped_informative_groups"]:
        return diagnostics

    count = diagnostics["sampled_decisions"]
    width = max(mask.shape[1] for _, mask, *_ in decisions)
    features = np.zeros((count, width, len(FEATURE_NAMES)), dtype=np.float32)
    masks = np.zeros((count, width), dtype=bool)
    start = 0
    for values, allowed, *_ in decisions:
        end = start + len(values)
        features[start:end, :values.shape[1]] = values
        masks[start:end, :allowed.shape[1]] = allowed
        start = end
    actions, previous, members = (np.concatenate(column) for column in zip(*(row[2:] for row in decisions)))
    x, mask = torch.from_numpy(features), torch.from_numpy(masks)
    action = torch.from_numpy(actions[:, None])
    old_logp = torch.as_tensor(previous, dtype=torch.float32)
    advantage = torch.as_tensor(advantages[members], dtype=torch.float32)
    weight = torch.as_tensor(1.0 / lengths[members], dtype=torch.float32)
    with torch.no_grad():
        reference = torch.log_softmax(actor.reference_logits(x).masked_fill(~mask, -1e9), dim=1)
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


def selection_key(validation_result):
    """Choose by captures first and ordinary capped time second."""
    return -validation_result["captured"], validation_result["mean_capped_objective_s"]


def train_seed(seed, training, validation, updates, group_size, destination):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    actor = ResidualActor()
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    baseline = metrics(outcomes(validation, actor.costs))
    validation_curve = {"0": baseline}
    best_trained = None
    diagnostics = {"sampled_groups": 0, "terminal_informative_groups": 0,
                   "shaped_informative_groups": 0, "sampled_decisions": 0}
    for step in range(1, updates + 1):
        chosen = [training[int(index)] for index in rng.integers(len(training), size=GROUPS_PER_UPDATE)]
        scenarios = [with_chart_obstacle(case) if rng.random() < .5 else case for case in chosen]
        for name, count in update(actor, optimizer, scenarios, group_size).items():
            diagnostics[name] += count
        if step % VALIDATION_INTERVAL and step != updates:
            continue
        validation_result = metrics(outcomes(validation, actor.costs))
        validation_curve[str(step)] = validation_result
        if best_trained is None or selection_key(validation_result) < selection_key(best_trained):
            best_trained = validation_result
            chosen_step = step
            weights = {name: value.detach().numpy().copy() for name, value in actor.state_dict().items()}
        print(json.dumps({"seed": seed, "update": step, "validation": validation_result, **diagnostics}),
              flush=True)

    checkpoint = destination / f"grpo_progress_seed_{seed}.npz"
    np.savez_compressed(checkpoint, **weights)
    return {
        "seed": seed, "chosen_trained_update": chosen_step, "selected_trained_validation": best_trained,
        "physics_initialization_validation": baseline,
        "trained_beats_initialization_on_validation": selection_key(best_trained) < selection_key(baseline),
        "validation_curve": validation_curve, "checkpoint": checkpoint.name, **diagnostics,
    }


def file_sha(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--output", type=Path, default=STUDY / "training")
    args = parser.parse_args()
    if args.updates < 1 or args.group_size < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use positive updates, at least two members per group and unique seeds.")
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "grpo_progress_selected.npz"
    receipts = args.output / "training_results.json"
    if destination.exists() or receipts.exists() or any(
        (args.output / f"grpo_progress_seed_{seed}.npz").exists() for seed in args.seeds
    ):
        parser.error("Training receipts already exist. Choose a fresh output directory.")

    torch.set_num_threads(1)
    scenario_file = PREVIOUS_STUDY / "scenarios.json.gz"
    cases = load_scenarios(scenario_file)
    training = [case for case in cases if case.split == "train"]
    validation = [case for case in cases if case.split == "validation"]
    assert {case.identifier for case in training}.isdisjoint(case.identifier for case in validation)
    physics_validation = metrics(outcomes(validation, None))
    results = [train_seed(seed, training, validation, args.updates, args.group_size, args.output)
               for seed in args.seeds]
    selected = min(results, key=lambda row: (*selection_key(row["selected_trained_validation"]), row["seed"]))
    checkpoint = args.output / selected["checkpoint"]
    checkpoint_hash = file_sha(checkpoint)
    details = {
        "training_cases": len(training), "validation_cases": len(validation),
        "training_ids": [case.identifier for case in training],
        "validation_ids": [case.identifier for case in validation],
        "physics_validation": physics_validation, "reward": {
            "capture": "negative elapsed seconds at first geometric gate entry",
            "timeout": -(MAX_SECONDS + TIMEOUT_COST_SECONDS),
            "blocked_extra_penalty_s": BLOCK_COST_SECONDS,
            "progress_seconds": PROGRESS_SECONDS,
            "discount_per_300s": PROGRESS_DISCOUNT_PER_WINDOW,
            "potential": "clip(1 - remaining gap to gate / initial gap to gate, -1, 1)",
            "shaping": "signed potential changes on applied intervals, with zero remaining gap at capture",
        },
        "validation_objective": "more captures, then lower mean capped time, then earliest update",
        "native_comparison_checkpoint_rule": "best post-update validation result, lowest seed on exact tie",
        "updates": args.updates, "group_size": args.group_size, "feature_names": list(FEATURE_NAMES),
        "training_scenario_sha256": file_sha(scenario_file),
        "training_source_sha256": file_sha(Path(__file__)),
        "reduced_environment_sha256": file_sha(PREVIOUS_STUDY / "reproduce/environment.py"),
        "actor_source_sha256": file_sha(PREVIOUS_STUDY / "reproduce/train_grpo.py"),
        "numpy_version": np.__version__, "torch_version": torch.__version__,
        "runs": results, "selected_seed": selected["seed"], "selected_update": selected["chosen_trained_update"],
        "selected_checkpoint_sha256": checkpoint_hash, "selected_checkpoint": destination.name,
        "selected_validation": selected["selected_trained_validation"],
        "selected_beats_physics_on_validation": (
            selection_key(selected["selected_trained_validation"]) < selection_key(physics_validation)
        ),
    }
    destination.write_bytes(checkpoint.read_bytes())
    receipts.write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps({"checkpoint": str(destination), "seed": selected["seed"],
                      "update": selected["chosen_trained_update"], "validation": details["selected_validation"],
                      "physics_validation": physics_validation}), flush=True)


if __name__ == "__main__":
    main()
