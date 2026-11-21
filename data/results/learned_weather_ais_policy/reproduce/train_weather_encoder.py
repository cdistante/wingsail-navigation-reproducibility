"""Pretrain a weather-history encoder on synthetic training scenarios only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

from weather_embedding import DEFAULT_CHECKPOINT, STUDY, TRAINING_KIND, causal_history, encode_model, encoder_weights
from world.learned_mark_policy import MAX_SECONDS
from world.temporal_recovery import HISTORY_STEPS, WEATHER_INTERVAL_SECONDS, RecoveryHoldModel, encode_hold_history


SYNTHETIC_STUDY = STUDY.with_name("learned_safe_mark_policy")
SOURCE = SYNTHETIC_STUDY / "scenarios.json.gz"
sys.path.insert(0, str(SYNTHETIC_STUDY / "reproduce"))
from environment import load_scenarios


SEED = 30711


def examples(scenarios, seed):
    rng = np.random.default_rng(seed)
    snapshots = []
    axes = []
    cap = int(MAX_SECONDS // WEATHER_INTERVAL_SECONDS) - 1
    for case in scenarios:
        windows = [0, 1, HISTORY_STEPS - 1, cap, *rng.integers(0, cap + 1, size=8)]
        if case.future is not None:
            switch = int(np.ceil(case.change_at / WEATHER_INTERVAL_SECONDS))
            windows.extend(np.clip(switch + np.array([-1, 0, 1, 3, 6, 9, 11]), 0, cap))
        for window in sorted(set(windows)):
            angles = rng.uniform(0, 2 * np.pi, size=3)
            directions = np.column_stack((np.sin(angles), np.cos(angles)))
            snapshots.extend([causal_history(case, window * WEATHER_INTERVAL_SECONDS)] * 4)
            axes.extend([case.target, *directions])
    projected, _ = encode_hold_history(np.asarray(snapshots), np.asarray(axes), False)
    return torch.from_numpy(projected)


def targets(history):
    """Reconstruct current weather, observed change and recent variability."""
    return torch.cat((
        history[:, -1],
        history[:, -3:].mean(dim=1) - history[:, :3].mean(dim=1),
        history.std(dim=1, unbiased=False),
    ), dim=1)


def pretrain(source: Path, output: Path, epochs: int, seed: int):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    eligible = [case for case in load_scenarios(source) if case.split == "train"]
    if len(eligible) < 20:
        raise ValueError("Weather pretraining needs the original synthetic training partition.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(eligible))
    valid_count = len(eligible) // 8
    validation = [eligible[i] for i in order[:valid_count]]
    training = [eligible[i] for i in order[valid_count:]]
    train = examples(training, seed + 1)
    val = examples(validation, seed + 2)
    mean = train.mean(dim=(0, 1))
    scale = train.std(dim=(0, 1), unbiased=False).clamp_min(1e-4)
    train, val = (train - mean) / scale, (val - mean) / scale
    train_target, val_target = targets(train), targets(val)

    encoder = RecoveryHoldModel()
    encoder.head.requires_grad_(False)
    decoder = nn.Sequential(nn.Linear(48, 48), nn.GELU(), nn.Linear(48, 15))
    optimizer = torch.optim.AdamW([parameter for parameter in list(encoder.parameters()) +
                                   list(decoder.parameters()) if parameter.requires_grad], lr=0.001)
    loss_function = nn.MSELoss()
    trace = []
    best_loss, best_epoch, best_weights = float("inf"), 0, None
    for epoch in range(1, epochs + 1):
        encoder.train()
        decoder.train()
        squared_loss = 0.0
        for indices in torch.randperm(len(train)).split(128):
            optimizer.zero_grad(set_to_none=True)
            clean = train[indices]
            noisy = clean + 0.10 * torch.randn_like(clean)
            output_values = decoder(encode_model(encoder, noisy))
            loss = loss_function(output_values, train_target[indices])
            loss.backward()
            optimizer.step()
            squared_loss += float(loss.detach()) * len(indices)
        encoder.eval()
        decoder.eval()
        with torch.inference_mode():
            predicted = torch.cat([decoder(encode_model(encoder, batch)) for batch in val.split(256)])
            validation_loss = float(loss_function(predicted, val_target))
        trace.append({"epoch": epoch, "train_mse": squared_loss / len(train), "validation_mse": validation_loss})
        if validation_loss < best_loss:
            best_loss, best_epoch, best_weights = validation_loss, epoch, encoder_weights(encoder)
        print(f"epoch {epoch:02d} train {squared_loss / len(train):.5f} validation {validation_loss:.5f}", flush=True)
    if best_weights is None:
        raise ValueError("Weather encoder training did not produce finite validation loss.")

    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    training_ids, validation_ids = sorted(case.identifier for case in training), sorted(case.identifier for case in validation)
    checkpoint = {
        "kind": TRAINING_KIND,
        "encoder_state": best_weights,
        "history_mean": mean,
        "history_scale": scale,
        "source_sha256": source_digest,
        "seed": seed,
        "best_epoch": best_epoch,
        "training_scenario_ids": training_ids,
        "validation_scenario_ids": validation_ids,
    }
    temporary = output.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(output)
    metrics = {
        "checkpoint": output.name,
        "checkpoint_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "source_sha256": source_digest,
        "source_partition": "synthetic train only",
        "seed": seed,
        "training_scenario_ids": training_ids,
        "validation_scenario_ids": validation_ids,
        "training_examples": len(train),
        "validation_examples": len(val),
        "best_epoch": best_epoch,
        "best_validation_mse": best_loss,
        "constant_training_mean_validation_mse": float(loss_function(
            train_target.mean(dim=0).expand_as(val_target), val_target)),
        "epoch_metrics": trace,
    }
    output.with_name("weather_encoder_training.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    pretrain(args.source, args.output, args.epochs, args.seed)
