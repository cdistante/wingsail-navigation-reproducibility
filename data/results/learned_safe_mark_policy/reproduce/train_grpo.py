"""Same-scenario grouped PPO-style updates for a simulated tack ranker."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from environment import MAX_SECONDS, TIMEOUT_COST_SECONDS, evaluate, load_scenarios, rollout, with_chart_obstacle
from world.learned_mark_policy import FEATURE_NAMES


STUDY = Path(__file__).resolve().parents[1]
PHYSICS_FEATURE = FEATURE_NAMES.index("physics_cost_h")
TEMPERATURE_SECONDS = 600.0


class ResidualActor(nn.Module):
    """Share weights across available trims and warm start from the physics rank."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(len(FEATURE_NAMES), 48), nn.Tanh(), nn.Linear(48, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features):
        residual = 2.0 * torch.tanh(self.network(features)[..., 0])
        return self.reference_logits(features) + residual

    @staticmethod
    def reference_logits(features):
        return -features[..., PHYSICS_FEATURE] * (3600.0 / TEMPERATURE_SECONDS)

    @torch.inference_mode()
    def costs(self, features):
        return -TEMPERATURE_SECONDS * self(torch.as_tensor(features, dtype=torch.float32)).numpy()


@torch.inference_mode()
def same_start_group(actor, scenario, size, group):
    target = np.repeat(scenario.target[None, :], size, axis=0)
    bow = np.full(size, scenario.bow)
    active = np.arange(size)
    returns = np.full(size, -(MAX_SECONDS + TIMEOUT_COST_SECONDS))
    lengths = np.zeros(size, dtype=int)
    elapsed = 0.0
    decisions = []
    while len(active) and elapsed < MAX_SECONDS:
        batch, dt, chart_allowed = evaluate(scenario, elapsed, target[active], bow[active])
        rows = np.flatnonzero(chart_allowed.any(axis=1))
        if not len(rows):
            break
        active = active[rows]
        features = batch.features[rows]
        allowed = chart_allowed[rows]
        hits = np.where(allowed, batch.captured_after_s[rows], np.inf)
        finishes = np.isfinite(hits).any(axis=1)
        # Exact safe hits terminate in the current action interval.
        selected = np.argmin(hits, axis=1)
        sampling = np.flatnonzero(~finishes)
        if len(sampling):
            x = torch.from_numpy(features[sampling])
            mask = torch.from_numpy(allowed[sampling])
            logits = actor(x).masked_fill(~mask, -torch.inf)
            distribution = torch.distributions.Categorical(logits=logits)
            sampled = distribution.sample()
            selected[sampling] = sampled.numpy()
            learn = allowed[sampling].sum(axis=1) > 1
            if learn.any():
                members = active[sampling[learn]]
                lengths[members] += 1
                decisions.append((features[sampling[learn]], allowed[sampling[learn]],
                                  sampled.numpy()[learn], distribution.log_prob(sampled).numpy()[learn],
                                  members + group * size))
        returns[active[finishes]] = -(elapsed + hits[np.arange(len(rows)), selected])[finishes]
        travel = rows[~finishes], selected[~finishes]
        target[active[~finishes]] = batch.remaining_target[travel]
        bow[active[~finishes]] = batch.end_bow[travel]
        active = active[~finishes]
        elapsed += dt
    return returns, lengths, decisions


def learn_groups(actor, optimizer, scenarios, size):
    groups = [same_start_group(actor, scenario, size, i) for i, scenario in enumerate(scenarios)]
    rewards = np.stack([group[0] for group in groups])
    spread = rewards.std(axis=1, keepdims=True)
    advantages = ((rewards - rewards.mean(axis=1, keepdims=True)) / np.maximum(spread, 1e-8)).ravel()
    lengths = np.concatenate([group[1] for group in groups])
    rows = [row for group in groups for row in group[2]]
    informative = int(np.count_nonzero(spread))
    if not rows or not informative:
        return informative, int(lengths.sum())
    count = int(lengths.sum())
    width = max(len(mask[0]) for _, mask, *_ in rows)
    features = np.zeros((count, width, len(FEATURE_NAMES)), dtype=np.float32)
    masks = np.zeros((count, width), dtype=bool)
    start = 0
    for values, allowed, *_ in rows:
        end = start + len(values)
        features[start:end, :values.shape[1]] = values
        masks[start:end, :values.shape[1]] = allowed
        start = end
    actions, previous, members = (np.concatenate(column) for column in zip(*(row[2:] for row in rows)))
    x, mask, action = torch.from_numpy(features), torch.from_numpy(masks), torch.from_numpy(actions[:, None])
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
    return informative, count


def outcomes(scenarios, score):
    return [{"scenario": case.identifier, "family": case.family, "chart": chart,
             **rollout(with_chart_obstacle(case) if chart else case, score)}
            for case in scenarios for chart in (False, True)]


def metrics(rows):
    captured = [row["duration_s"] for row in rows if row["status"] == "captured"]
    return {
        "scenarios": len(rows), "captured": len(captured),
        "timeout": sum(row["status"] == "timeout" for row in rows),
        "blocked": sum(row["status"] == "blocked" for row in rows),
        "median_capture_s": float(np.median(captured)) if captured else None,
        "mean_capped_objective_s": float(np.mean([row["objective_s"] for row in rows])),
        "chart_mask_interventions": sum(row["chart_masks"] for row in rows),
    }


def train(seed, training, validation, test, steps, size):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    actor = ResidualActor()
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    best = metrics(outcomes(validation, actor.costs))["mean_capped_objective_s"]
    chosen_step = 0
    snapshot = {key: value.clone() for key, value in actor.state_dict().items()}
    best_trained = float("inf")
    trained_step = 0
    trained_snapshot = None
    validation_curve = {"0": best}
    informative = sampled_decisions = 0
    for step in range(1, steps + 1):
        chosen = [training[int(i)] for i in rng.integers(len(training), size=3)]
        scenarios = [with_chart_obstacle(case) if rng.random() < .5 else case for case in chosen]
        nonzero, count = learn_groups(actor, optimizer, scenarios, size)
        informative += nonzero
        sampled_decisions += count
        if step % 16 == 0 or step == steps:
            validation_metric = metrics(outcomes(validation, actor.costs))["mean_capped_objective_s"]
            validation_curve[str(step)] = validation_metric
            if validation_metric < best_trained:
                best_trained = validation_metric
                trained_step = step
                trained_snapshot = {key: value.clone() for key, value in actor.state_dict().items()}
            if validation_metric < best:
                best = validation_metric
                chosen_step = step
                snapshot = {key: value.clone() for key, value in actor.state_dict().items()}
    actor.load_state_dict(snapshot)
    rows = outcomes(test, actor.costs)
    torch.save(snapshot, STUDY / f"grpo_seed_{seed}.pt")
    actor.load_state_dict(trained_snapshot)
    trained_rows = outcomes(test, actor.costs)
    np.savez_compressed(STUDY / f"grpo_seed_{seed}.npz",
                        **{key: value.numpy() for key, value in trained_snapshot.items()})
    summary = {"seed": seed, "validation_selected_update": chosen_step,
               "validation_mean_capped_objective_s": best, "sampled_groups": steps * 3,
               "informative_groups": informative, "sampled_decisions": sampled_decisions,
               "test": metrics(rows), "validation_curve": validation_curve,
               "trained_only_update": trained_step, "trained_only_validation_objective_s": best_trained,
               "trained_only_test": metrics(trained_rows)}
    return summary, rows, trained_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7309, 8779, 10921])
    args = parser.parse_args()
    assert args.updates > 0 and args.group_size >= 2
    torch.set_num_threads(1)
    splits = load_scenarios(STUDY / "scenarios.json.gz")
    training, validation, test = ([case for case in splits if case.split == split]
                                  for split in ("train", "validation", "test"))
    baseline = outcomes(test, None)
    warm = outcomes(test, ResidualActor().costs)
    results = {"physics": metrics(baseline), "physics_initialized_actor": metrics(warm),
               "GRPO": [], "group_size": args.group_size,
               "reward": "Negative first-entry time, with fixed twice-cap penalty for timeout or blocked.",
               "environment": "Reduced Fossen equilibrium one-shift weather with optional swept chart disk.",
               "selection": "Physics-warm actor and trained checkpoints selected on validation capped time only."}
    rows = [{"method": method, **row} for method, episodes in
            (("physics", baseline), ("physics_initialized_actor", warm)) for row in episodes]
    for seed in args.seeds:
        summary, details, trained_details = train(seed, training, validation, test, args.updates, args.group_size)
        results["GRPO"].append(summary)
        rows.extend({"method": f"grpo_{seed}", **row} for row in details)
        rows.extend({"method": f"grpo_trained_{seed}", **row} for row in trained_details)
        print(json.dumps(summary), flush=True)
    (STUDY / "grpo_results.json").write_text(json.dumps(results, indent=2) + "\n")
    with (STUDY / "grpo_episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
