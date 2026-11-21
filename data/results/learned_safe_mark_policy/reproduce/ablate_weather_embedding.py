"""Matched fitted-Q comparison of observed-weather history inputs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from environment import active_forcing, enforce_first_entry, evaluate, load_scenarios
from train_fitted_q import TRAIN_SEED, collect, compare, summary, train_batch
from world.boat import BoatState
from world.currents import Current
from world.learned_mark_policy import FEATURE_NAMES, MAX_SECONDS, TIMEOUT_COST_SECONDS
from world.mark_weather_embedding import EMBEDDING_DIMENSION, FrozenMarkWeatherEncoder
from world.temporal_recovery import HISTORY_STEPS, WEATHER_INTERVAL_SECONDS, encode_hold_history
from world.weather import Wind
from world.waves import Wave
from world.world import WorldObservation


STUDY = Path(__file__).resolve().parents[1]
OUTPUT = STUDY / "weather_embedding_ablation"
CHECKPOINT = STUDY.with_name("otranto_temporal_hold_20261118") / "models/temporal_hold.pt"
ARMS = ("live_only", "repeated_current", "causal_history")
DIMENSIONS = len(FEATURE_NAMES) + EMBEDDING_DIMENSION


def observed_vectors(forcing):
    bearing = np.radians(forcing.polar.wind_from)
    wind = -forcing.wind_speed * np.array((np.sin(bearing), np.cos(bearing)))
    return np.r_[wind, forcing.polar.flow, forcing.wave_height]


def observed_history(scenario, elapsed, *, current_only=False):
    elapsed = np.asarray(elapsed)
    now = np.floor_divide(elapsed, WEATHER_INTERVAL_SECONDS)
    windows = np.maximum(0, now[..., None] - np.arange(HISTORY_STEPS - 1, -1, -1))
    if current_only:
        windows = np.broadcast_to(now[..., None], windows.shape)
    old = observed_vectors(scenario.first)
    if scenario.future is None:
        return np.broadcast_to(old, (*windows.shape, len(old)))
    seen_future = windows * WEATHER_INTERVAL_SECONDS >= scenario.change_at
    return np.where(seen_future[..., None], observed_vectors(scenario.future), old)


class BatchedEncoder:
    """Batch the same frozen forward pass as FrozenMarkWeatherEncoder.encode."""

    def __init__(self):
        self.frozen = FrozenMarkWeatherEncoder(CHECKPOINT)

    @torch.inference_mode()
    def __call__(self, history, targets):
        guidance = self.frozen._guidance
        projection, _ = encode_hold_history(history, targets, False)
        rows = torch.as_tensor(projection)
        scaling = guidance.scaling
        model = guidance.model
        result = []
        for batch in rows.split(512):
            values = (batch - scaling["history_mean"]) / scaling["history_scale"]
            result.append(model.encoder(model.projection(values) + model.position)[:, -1].numpy())
        return np.concatenate(result)


def verify_encoder_parity(scenario, encoder):
    """Rebuild one chronological history via the public runtime interface."""
    end = int(scenario.change_at + WEATHER_INTERVAL_SECONDS * 3)
    final = None
    for elapsed in range(0, end + 1, WEATHER_INTERVAL_SECONDS):
        forcing, _, _ = active_forcing(scenario, elapsed)
        wind_u, wind_v, current_u, current_v, wave_height = observed_vectors(forcing)
        current_speed = np.hypot(current_u, current_v)
        current_to = np.degrees(np.arctan2(current_u, current_v)) % 360
        wind_from = forcing.polar.wind_from
        final = WorldObservation(
            np.datetime64("2000-01-01") + np.timedelta64(elapsed, "s"),
            BoatState(0.0, 0.0, scenario.bow, 0.0),
            Wind(wind_u, wind_v, forcing.wind_speed, wind_from),
            Wave(wave_height, wind_from, 5.0),
            Current(current_u, current_v, current_speed, current_to, 0.0), None, None, None, (),
        )
        encoder.frozen.observe(final)
    answer = encoder.frozen.encode(final, scenario.target)
    batch = encoder(observed_history(scenario, np.array([end])), scenario.target[None])[0]
    error = float(np.max(np.abs(answer.values - batch)))
    assert max(answer.source_windows) <= max(answer.history_windows) and error < 2e-6
    return error


def training_context(scenarios, data, encoder):
    """Match the original counterfactual rows and add current/next histories."""
    rng = np.random.default_rng(TRAIN_SEED)
    embeddings = {arm: [[], []] for arm in ARMS[1:]}
    cursor = 0
    distinct = 0
    for scenario in scenarios:
        positions = [[], []]
        clocks = [[], []]
        usable_next = []
        for trajectory in range(2):
            target, bow = scenario.target.copy(), scenario.bow
            if trajectory:
                target += rng.uniform(-160.0, 160.0, size=2)
                bow = (bow + rng.uniform(-75.0, 75.0)) % 360
            elapsed = float(rng.integers(0, 9) * 30) if trajectory else 0.0
            for step in range(72):
                batch, dt, allowed = evaluate(scenario, elapsed, target, bow)
                chosen = np.flatnonzero(allowed[0])
                hits = batch.captured_after_s[0]
                greedy = int(np.argmin(enforce_first_entry(batch.physics_cost_s, batch, dt, allowed)[0]))
                if step < 8 or (step - trajectory) % 3 == 0:
                    count = len(chosen)
                    assert np.array_equal(data.features[cursor:cursor + count], batch.features[0, chosen])
                    cursor += count
                    positions[0].append(np.broadcast_to(target, (count, 2)))
                    positions[1].append(batch.remaining_target[0, chosen])
                    clocks[0].append(np.full(count, elapsed))
                    clocks[1].append(np.full(count, elapsed + dt))
                    usable_next.append(~np.isfinite(hits[chosen]) & (elapsed + dt < MAX_SECONDS))
                action = greedy
                if trajectory and rng.random() < .35:
                    action = int(rng.choice(chosen))
                if np.isfinite(hits[action]) or elapsed + dt >= MAX_SECONDS:
                    break
                target, bow = batch.remaining_target[0, action], float(batch.end_bow[0, action])
                elapsed += dt
        usable_next = np.concatenate(usable_next)
        for side in (0, 1):
            targets = np.concatenate(positions[side])
            elapsed = np.concatenate(clocks[side])
            valid = np.ones(len(targets), dtype=bool) if side == 0 else usable_next
            assert np.all(np.linalg.norm(targets[valid], axis=-1) > 1e-8)
            actual = observed_history(scenario, elapsed[valid])
            repeated = observed_history(scenario, elapsed[valid], current_only=True)
            if side == 0:
                distinct += int(np.any(actual != repeated, axis=(1, 2)).sum())
            for arm, vectors in (("causal_history", actual), ("repeated_current", repeated)):
                output = np.zeros((len(targets), EMBEDDING_DIMENSION), dtype=np.float32)
                if valid.any():
                    output[valid] = encoder(vectors, targets[valid])
                embeddings[arm][side].append(output)
    assert cursor == len(data.features)
    result = {arm: tuple(np.concatenate(arrays) for arrays in sides) for arm, sides in embeddings.items()}
    result["live_only"] = tuple(np.zeros_like(array) for array in result["causal_history"])
    return result, distinct


def bellman_targets(data, network, means, scales, next_embedding):
    network.eval()
    target = data.immediate_cost.copy()
    available = np.flatnonzero(np.any(data.future_mask, axis=1))
    with torch.inference_mode():
        for ids in np.array_split(available, max(1, (len(available) + 2047) // 2048)):
            if not len(ids):
                continue
            physical = data.future_features[ids]
            context = np.broadcast_to(next_embedding[ids, None, :], (*physical.shape[:2], EMBEDDING_DIMENSION))
            normalized = (np.concatenate((physical, context), axis=-1) - means) / scales
            continuation = np.maximum(network(torch.as_tensor(normalized)).numpy()[..., 0] * 3600.0, 300.0)
            continuation = np.where(np.isfinite(data.future_entries[ids]), data.future_entries[ids], continuation)
            continuation = np.where(data.future_mask[ids], continuation, np.inf)
            target[ids] += np.min(continuation, axis=1)
    return np.minimum(target, data.remaining + TIMEOUT_COST_SECONDS)


@torch.inference_mode()
def rollout(scenario, network, means, scales, arm, encoder):
    target, bow, elapsed, tacks, interventions = scenario.target.copy(), scenario.bow, 0.0, 0, 0
    while elapsed < MAX_SECONDS:
        batch, dt, allowed = evaluate(scenario, elapsed, target, bow)
        if network is None:
            scores = batch.physics_cost_s
        else:
            if arm == "live_only":
                embedding = np.zeros((1, EMBEDDING_DIMENSION), dtype=np.float32)
            else:
                history = observed_history(scenario, np.array([elapsed]), current_only=arm == "repeated_current")
                embedding = encoder(history, target[None])
            features = np.concatenate((batch.features, np.broadcast_to(
                embedding[:, None, :], (*batch.features.shape[:2], EMBEDDING_DIMENSION))), axis=-1)
            scores = 3600.0 * network(torch.as_tensor((features - means) / scales)).numpy()[..., 0]
        angle = enforce_first_entry(scores, batch, dt, allowed)
        choice = int(np.argmin(angle[0]))
        if not allowed.any():
            return dict(status="blocked", duration_s=elapsed, chart_masks=interventions, tacks=tacks,
                        objective_s=MAX_SECONDS + TIMEOUT_COST_SECONDS)
        interventions += choice != int(np.argmin(enforce_first_entry(scores, batch, dt, np.ones_like(allowed))[0]))
        entered = batch.captured_after_s[0, choice]
        if np.isfinite(entered):
            final = elapsed + float(entered)
            return dict(status="captured", duration_s=final, chart_masks=interventions, tacks=tacks, objective_s=final)
        updated = float(batch.end_bow[0, choice])
        tacks += abs((updated - bow + 180) % 360 - 180) >= 30
        target, bow, elapsed = batch.remaining_target[0, choice], updated, elapsed + dt
    return dict(status="timeout", duration_s=elapsed, chart_masks=interventions, tacks=tacks,
                objective_s=MAX_SECONDS + TIMEOUT_COST_SECONDS)


def train_arm(arm, data, embeddings, validation, encoder):
    torch.manual_seed(TRAIN_SEED)
    shuffle = torch.Generator().manual_seed(TRAIN_SEED)
    model = nn.Sequential(nn.Linear(DIMENSIONS, 64), nn.ReLU(), nn.Linear(64, 48), nn.ReLU(), nn.Linear(48, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0012)
    first, second = embeddings[arm]
    features = np.concatenate((data.features, first), axis=-1)
    means = features.mean(axis=0)
    scales = np.maximum(features.std(axis=0), .04)
    training = torch.as_tensor((features - means) / scales)
    train_batch(model, optimizer, training, data.warm_cost, 13, shuffle)
    history, best, best_score = [], None, np.inf
    for step in range(1, 9):
        target = bellman_targets(data, model, means, scales, second)
        loss = train_batch(model, optimizer, training, target, 4, shuffle)
        model.eval()
        measured = summary([rollout(s, model, means, scales, arm, encoder) for s in validation])
        measured.update(iteration=step, bellman_loss=loss)
        history.append(measured)
        score = measured["mean_penalized_objective_s"]
        if score < best_score:
            best = {key: value.clone() for key, value in model.state_dict().items()}
            best_score = score
            selected = step
    model.load_state_dict(best)
    torch.save({"state_dict": best, "means": torch.from_numpy(means), "scales": torch.from_numpy(scales),
                "arm": arm, "selected_iteration": selected, "feature_names": FEATURE_NAMES},
               OUTPUT / f"{arm}.pt")
    print(f"{arm}: selected iteration {selected}; validation cost {best_score:.2f} s", flush=True)
    return model, means, scales, {"curves": history, "selected_iteration": selected}


def main():
    OUTPUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    scenarios_path = STUDY / "scenarios.json.gz"
    scenarios = load_scenarios(scenarios_path)
    grouped = {name: [s for s in scenarios if s.split == name] for name in ("train", "validation", "test")}
    encoder = BatchedEncoder()
    parity = verify_encoder_parity(grouped["validation"][0], encoder)
    data = collect(grouped["train"])
    embeddings, differential_rows = training_context(grouped["train"], data, encoder)
    model_inputs = {"physics": (None, None, None)}
    histories = {}
    for arm in ARMS:
        network, means, scales, history = train_arm(arm, data, embeddings, grouped["validation"], encoder)
        model_inputs[arm] = network, means, scales
        histories[arm] = history
    held_out = grouped["test"]
    outcomes = {arm: [rollout(s, *args, arm, encoder) for s in held_out] for arm, args in model_inputs.items()}
    with (OUTPUT / "episodes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("arm", "scenario", "family", "status", "duration_s",
                                                   "objective_s", "chart_masks", "tacks"))
        writer.writeheader()
        for arm, results in outcomes.items():
            writer.writerows({"arm": arm, "scenario": scenario.identifier, "family": scenario.family,
                              **result} for scenario, result in zip(held_out, results, strict=True))
    report = {
        "scope": "Matched reduced-model synthetic open-water ablation. Absolute histories are observed "
                 "through the action timestamp. Future forcing available to baseline live features is identical in all arms.",
        "arms": {"live_only": "26 explicit features and 48 constant zero inputs",
                 "repeated_current": "26 explicit features plus frozen encoding of current observation repeated 12 times",
                 "causal_history": "26 explicit features plus frozen encoding of 12 causal five-minute observations"},
        "equal_trainable_parameter_count": sum(parameter.numel() for parameter in model_inputs["live_only"][0].parameters()),
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "scenario_file_sha256": hashlib.sha256(scenarios_path.read_bytes()).hexdigest(),
        "rows": len(data.features),
        "rows_with_different_causal_vs_repeated_history": differential_rows,
        "scenario_counts": {key: len(value) for key, value in grouped.items()},
        "maximum_checked_scalar_batch_difference": parity,
        "validation": histories,
        "test": {arm: summary(results) for arm, results in outcomes.items()},
        "history_vs_live_only": compare(outcomes["live_only"], outcomes["causal_history"]),
        "history_vs_repeated_current": compare(outcomes["repeated_current"], outcomes["causal_history"]),
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("test", "history_vs_live_only", "history_vs_repeated_current")},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
