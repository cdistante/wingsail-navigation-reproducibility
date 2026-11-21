"""Train a fitted-Q candidate scorer and evaluate it on blocked scenarios.

Run from the V2 repository root with NumPy, SciPy and PyTorch available.
Only train-split polar environments contribute transitions or normalizers.
Validation chooses an iteration. Test episodes are scored once at the end.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from environment import (
    FAMILIES, MAX_SECONDS, SCENARIO_SEED, TIMEOUT_COST_SECONDS, enforce_first_entry, evaluate,
    generate_scenarios, load_scenarios, rollout, save_scenarios, with_chart_obstacle,
)
from world.learned_mark_policy import FEATURE_NAMES, MODEL_VERSION, LearnedMarkPolicy


STUDY = Path(__file__).resolve().parents[1]
TRAIN_SEED = 5612
DOMAIN = {
    "range_m": [55.0, 5500.0],
    "radius_m": [50.0, 50.0],
    "wind_mps": [2.4, 9.0],
    "current_mps": [0.0, .85],
    "wave_m": [.05, 2.0],
    "decision_s": [60.0, 300.0],
    "forecast_s": [0.0, 15_000.0],
    "ground_mps": [0.0, 2.8],
}


@dataclass
class ReplayDataset:
    features: np.ndarray
    warm_cost: np.ndarray
    immediate_cost: np.ndarray
    future_features: np.ndarray
    future_entries: np.ndarray
    future_mask: np.ndarray
    remaining: np.ndarray


def collect(scenarios) -> ReplayDataset:
    rng = np.random.default_rng(TRAIN_SEED)
    width = max(len(stage.polar.bow) for scenario in scenarios
                for stage in (scenario.first, scenario.future) if stage is not None)
    collected = [[] for _ in range(7)]
    for scenario in scenarios:
        for trajectory in range(2):
            target, bow = scenario.target.copy(), scenario.bow
            if trajectory:
                target += rng.uniform(-160.0, 160.0, size=2)
                bow = (bow + rng.uniform(-75.0, 75.0)) % 360.0
            elapsed = float(rng.integers(0, 9) * 30) if trajectory else 0.0
            for step in range(72):
                batch, dt, allowed = evaluate(scenario, elapsed, target, bow)
                chosen = np.flatnonzero(allowed[0])
                assert len(chosen)  # The training set is open water.
                hits = batch.captured_after_s[0]
                greedy = int(np.argmin(enforce_first_entry(batch.physics_cost_s, batch, dt, allowed)[0]))
                if step < 8 or (step - trajectory) % 3 == 0:
                    count = len(chosen)
                    next_features = np.zeros((count, width, len(FEATURE_NAMES)), dtype=np.float32)
                    next_entries = np.full((count, width), np.inf, dtype=np.float32)
                    next_mask = np.zeros((count, width), dtype=bool)
                    remaining = MAX_SECONDS - elapsed
                    continuation = ~np.isfinite(hits[chosen]) & (remaining > dt)
                    immediate = np.where(np.isfinite(hits[chosen]), hits[chosen], dt)
                    immediate += (remaining <= dt) * ~np.isfinite(hits[chosen]) * TIMEOUT_COST_SECONDS
                    if continuation.any():
                        valid = chosen[continuation]
                        following, _, safe = evaluate(scenario, elapsed + dt,
                                                      batch.remaining_target[0, valid], batch.end_bow[0, valid])
                        size = len(following.features[0])
                        next_features[continuation, :size] = following.features
                        next_entries[continuation, :size] = following.captured_after_s
                        next_mask[continuation, :size] = safe
                    samples = (batch.features[0, chosen],
                               np.minimum(remaining + TIMEOUT_COST_SECONDS, batch.physics_cost_s[0, chosen]),
                               immediate, next_features, next_entries, next_mask, np.full(count, remaining))
                    for sink, values in zip(collected, samples, strict=True):
                        sink.append(values)
                action = greedy
                if trajectory and rng.random() < .35:
                    action = int(rng.choice(chosen))
                if np.isfinite(hits[action]) or elapsed + dt >= MAX_SECONDS:
                    break
                target, bow = batch.remaining_target[0, action], float(batch.end_bow[0, action])
                elapsed += dt
    return ReplayDataset(*(np.concatenate(values) for values in collected))


def new_network():
    width = len(FEATURE_NAMES)
    return nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, 48), nn.ReLU(), nn.Linear(48, 1))


def train_batch(network, optimizer, x, target, epochs, shuffle):
    network.train()
    target = torch.as_tensor(target / 3600, dtype=torch.float32)
    criterion = nn.SmoothL1Loss(beta=.18)
    total = 0.0
    for _ in range(epochs):
        batches = torch.randperm(len(x), generator=shuffle).split(768)
        total = 0.0
        for ids in batches:
            prediction = network(x[ids]).flatten()
            loss = criterion(prediction, target[ids])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(ids)
    return total / len(x)


def bellman_targets(data, network, means, scales):
    """Regress current cost plus the minimum available next-state Q value."""
    network.eval()
    target = data.immediate_cost.copy()
    live = np.flatnonzero(np.any(data.future_mask, axis=1))
    with torch.no_grad():
        for ids in np.array_split(live, max(1, (len(live) + 2047) // 2048)):
            if not len(ids):
                continue
            features = (data.future_features[ids] - means) / scales
            continuation = network(torch.as_tensor(features)).numpy()[..., 0] * 3600.0
            continuation = np.maximum(continuation, 300.0)
            entries = data.future_entries[ids]
            continuation = np.where(np.isfinite(entries), entries, continuation)
            continuation = np.where(data.future_mask[ids], continuation, np.inf)
            target[ids] += np.min(continuation, axis=1)
    return np.minimum(target, data.remaining + TIMEOUT_COST_SECONDS)


def learned(network, means, scales):
    layers = tuple((layer.weight.detach().numpy().copy(), layer.bias.detach().numpy().copy())
                   for layer in network if isinstance(layer, nn.Linear))
    return LearnedMarkPolicy(means, scales, layers, DOMAIN)


def export(path, policy):
    data = {"version": MODEL_VERSION, "feature_names": list(FEATURE_NAMES),
            "means": policy.means.tolist(), "scales": policy.scales.tolist(),
            "layers": [{"weights": w.tolist(), "bias": b.tolist()} for w, b in policy.layers],
            "domain": policy.domain}
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")


def evaluation(scenarios, score):
    return [rollout(scenario, score) for scenario in scenarios]


def summary(rows):
    captured = [r["duration_s"] for r in rows if r["status"] == "captured"]
    return {"scenarios": len(rows), "captured": len(captured),
            "timeouts": sum(r["status"] == "timeout" for r in rows),
            "blocked": sum(r["status"] == "blocked" for r in rows),
            "mean_penalized_objective_s": float(np.mean([r["objective_s"] for r in rows])),
            "median_capture_s": float(np.median(captured)) if captured else None,
            "mean_bow_changes_30deg": float(np.mean([r["tacks"] for r in rows])),
            "chart_mask_interventions": sum(r["chart_masks"] for r in rows)}


def compare(baseline, candidate):
    paired = np.array([b["duration_s"] - a["duration_s"] for a, b in zip(baseline, candidate, strict=True)
                       if a["status"] == b["status"] == "captured"])
    return {"joint_captures": len(paired), "joint_faster": int(np.count_nonzero(paired < -1e-6)),
            "joint_slower": int(np.count_nonzero(paired > 1e-6)),
            "mean_joint_difference_s": float(paired.mean()) if len(paired) else None,
            "median_joint_difference_s": float(np.median(paired)) if len(paired) else None,
            "worst_joint_slowdown_s": float(paired.max()) if len(paired) else None,
            "baseline_only_capture": sum(a["status"] == "captured" and b["status"] != "captured"
                                         for a, b in zip(baseline, candidate, strict=True)),
            "learner_only_capture": sum(b["status"] == "captured" and a["status"] != "captured"
                                        for a, b in zip(baseline, candidate, strict=True))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--iterations", type=int, default=8)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("Run at least one Bellman fitted-Q iteration.")
    torch.set_num_threads(2)
    torch.manual_seed(TRAIN_SEED)
    shuffle = torch.Generator().manual_seed(TRAIN_SEED)
    scenarios_path = args.study / "scenarios.json.gz"
    if scenarios_path.exists():
        scenarios = load_scenarios(scenarios_path)
    else:
        scenarios = list(generate_scenarios())
        save_scenarios(scenarios_path, scenarios)
    grouped = {name: [s for s in scenarios if s.split == name] for name in ("train", "validation", "test")}
    split_ids = {k: [s.identifier for s in value] for k, value in grouped.items()}
    assert len(scenarios) == len({s.identifier for s in scenarios})
    assert all({s.family for s in group} == set(FAMILIES) for group in grouped.values())
    (args.study / "splits.json").write_text(json.dumps({"generator_seed": SCENARIO_SEED,
                                                       "scenarios": split_ids}, indent=2) + "\n")
    data = collect(grouped["train"])
    print(f"Training on {len(data.features)} counterfactual transitions from {len(grouped['train'])} train scenarios.",
          flush=True)
    means = data.features.mean(axis=0)
    scales = np.maximum(data.features.std(axis=0), .04)
    training = torch.as_tensor((data.features - means) / scales)
    network = new_network()
    optimizer = torch.optim.AdamW(network.parameters(), lr=.0012)
    warm_loss = train_batch(network, optimizer, training, data.warm_cost, 13, shuffle)
    warm = learned(network, means, scales)
    export(args.study / "supervised_mark_policy.json", warm)
    validation = grouped["validation"]
    physics_validation = summary(evaluation(validation, None))
    warm_validation = summary(evaluation(validation, warm.costs))
    iterations, best, best_mean = [], None, np.inf
    for step in range(1, args.iterations + 1):
        started = perf_counter()
        target = bellman_targets(data, network, means, scales)
        loss = train_batch(network, optimizer, training, target, 4, shuffle)
        policy = learned(network, means, scales)
        metrics = summary(evaluation(validation, policy.costs))
        iterations.append({"iteration": step, "bellman_regression_loss_hour_units": loss, **metrics})
        print(f"FQI iteration {step}: validation {metrics['captured']}/{metrics['scenarios']}; "
              f"cost {metrics['mean_penalized_objective_s']:.1f}s, {perf_counter() - started:.1f}s wall", flush=True)
        if metrics["mean_penalized_objective_s"] < best_mean:
            best, best_mean, selected = policy, metrics["mean_penalized_objective_s"], step
    model_path = args.study / "learned_mark_policy.json"
    export(model_path, best)
    policy = LearnedMarkPolicy.load(model_path)
    test = grouped["test"]
    subset = [with_chart_obstacle(scenario) for scenario in test[::2]]
    methods = {"physics": None, "supervised": warm.costs, "fitted_q": policy.costs}
    measurements = {}
    with (args.study / "test_episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("suite", "scenario", "family", "method", "status",
                                                   "duration_s", "objective_s", "chart_masks", "tacks"))
        writer.writeheader()
        for name, suite in (("open_water", test), ("chart_screen", subset)):
            outcomes = {method: evaluation(suite, score) for method, score in methods.items()}
            measurements[name] = {method: summary(rows) for method, rows in outcomes.items()}
            measurements[name]["fitted_q_vs_physics"] = compare(outcomes["physics"], outcomes["fitted_q"])
            measurements[name]["fitted_q_vs_supervised"] = compare(outcomes["supervised"], outcomes["fitted_q"])
            for method, rows in outcomes.items():
                writer.writerows({"suite":name, "scenario":scenario.identifier, "family":scenario.family,
                                  "method":method, **row} for scenario, row in zip(suite, rows, strict=True))
    metrics = {
        "algorithm": "Fitted Q by Bellman cost improvement from a physics-score supervised warm start.",
        "motion": "Fossen-equilibrium ground polars, drift-only turn and one scheduled change in forcing.",
        "objective": "Exact first 50 m entry, with 6 h cap and additional 6 h uncaptured penalty.",
        "guard": "No synthetic chart obstacles in training. Test chart suite masks both traveled segments "
                 "against a deterministic disk. Native AIS and chart checks are outside this experiment.",
        "training_transitions": len(data.features),
        "training_completed_in_first_action_transitions": int(np.count_nonzero(
            (data.immediate_cost < 300.0) & ~data.future_mask.any(axis=1))),
        "generator_seed": SCENARIO_SEED, "train_seed": TRAIN_SEED,
        "scenarios_sha256": hashlib.sha256(scenarios_path.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "warm_start_loss_hour_units": warm_loss,
        "validation": {"physics": physics_validation, "supervised": warm_validation, "fitted_q_iterations": iterations},
        "selected_iteration": selected, "test": measurements,
    }
    (args.study / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(json.dumps(measurements, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
