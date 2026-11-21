"""Train and evaluate the temporal AIS course Transformer."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .ais_course_transformer import (
    AISCourseTransformer,
    AISSequenceDataset,
    FEATURE_NAMES,
    angular_errors_degrees,
    build_ais_sequence_dataset,
    direction_loss,
)


def loader_for(
    dataset: AISSequenceDataset,
    partition: str,
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    selected = dataset.split == partition
    features = (dataset.features[selected] - mean) / standard_deviation
    tensors = TensorDataset(
        torch.from_numpy(features.astype(np.float32)),
        torch.from_numpy(dataset.targets[selected].astype(np.float32)),
    )
    return DataLoader(
        tensors,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(
    model: AISCourseTransformer,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, object]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    angle_sum = None
    samples = 0
    for features, targets in loader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            prediction = model(features)
            loss, _ = direction_loss(prediction, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        batch_size = features.shape[0]
        angles = angular_errors_degrees(prediction.detach(), targets).sum(dim=0)
        loss_sum += float(loss.item()) * batch_size
        angle_sum = angles if angle_sum is None else angle_sum + angles
        samples += batch_size
    if samples == 0 or angle_sum is None:
        raise RuntimeError("AIS partition contains no samples.")
    return {
        "loss": loss_sum / samples,
        "angle_degrees": (angle_sum / samples).cpu().tolist(),
        "mean_angle_degrees": float((angle_sum / samples).mean().item()),
        "samples": samples,
    }


def persistence_baseline(dataset: AISSequenceDataset, partition: str) -> dict[str, object]:
    """Evaluate the last reported COG as a constant-course forecast."""

    selected = dataset.split == partition
    current_direction = dataset.features[selected, -1, 3:5]
    prediction = np.repeat(
        current_direction[:, np.newaxis, :],
        len(dataset.horizons_minutes),
        axis=1,
    )
    target = dataset.targets[selected]
    cosine = np.clip(np.sum(prediction * target, axis=-1), -1.0, 1.0)
    angles = np.degrees(np.arccos(cosine))
    return {
        "angle_degrees": angles.mean(axis=0).tolist(),
        "mean_angle_degrees": float(angles.mean()),
        "samples": int(selected.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a temporal AIS course Transformer.")
    parser.add_argument("--ais-csv", type=Path, required=True)
    parser.add_argument("--dataset-cache", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/models/ais_course_transformer"))
    parser.add_argument("--history-minutes", type=int, default=30)
    parser.add_argument("--horizons-minutes", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--sample-interval-seconds", type=int, default=60)
    parser.add_argument("--max-interpolation-gap-seconds", type=int, default=900)
    parser.add_argument("--minimum-future-displacement-metres", type=float, default=25.0)
    parser.add_argument("--window-stride-minutes", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--ffn", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    arguments = parser.parse_args()

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    if arguments.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device(
        "cuda"
        if arguments.device == "cuda"
        or (arguments.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )

    if arguments.dataset_cache is not None and arguments.dataset_cache.exists():
        dataset = AISSequenceDataset.load(arguments.dataset_cache)
    else:
        dataset = build_ais_sequence_dataset(
            arguments.ais_csv,
            history_minutes=arguments.history_minutes,
            horizons_minutes=tuple(arguments.horizons_minutes),
            sample_interval_seconds=arguments.sample_interval_seconds,
            max_interpolation_gap_seconds=arguments.max_interpolation_gap_seconds,
            minimum_future_displacement_metres=arguments.minimum_future_displacement_metres,
            window_stride_minutes=arguments.window_stride_minutes,
            seed=arguments.seed,
        )
        if arguments.dataset_cache is not None:
            dataset.save(arguments.dataset_cache)

    train_features = dataset.features[dataset.split == "train"]
    feature_mean = train_features.mean(axis=(0, 1), keepdims=True)
    feature_standard_deviation = train_features.std(axis=(0, 1), keepdims=True)
    feature_standard_deviation = np.maximum(feature_standard_deviation, 1e-6)
    loaders = {
        partition: loader_for(
            dataset,
            partition,
            feature_mean,
            feature_standard_deviation,
            batch_size=arguments.batch_size,
            shuffle=partition == "train",
        )
        for partition in ("train", "val", "test")
    }
    model_config = {
        "input_features": len(FEATURE_NAMES),
        "horizons": len(dataset.horizons_minutes),
        "d_model": arguments.d_model,
        "nhead": arguments.nhead,
        "num_layers": arguments.layers,
        "dim_feedforward": arguments.ffn,
        "dropout": arguments.dropout,
        "maximum_sequence_length": dataset.features.shape[1],
    }
    model = AISCourseTransformer(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = arguments.output_dir / "history.csv"
    with history_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["epoch", "train_loss", "train_angle", "val_loss", "val_angle"]
            + [f"val_angle_{minutes}m" for minutes in dataset.horizons_minutes]
        )

    best_validation = float("inf")
    best_path = arguments.output_dir / "best.pt"
    for epoch in range(1, arguments.epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], device, optimizer)
        validation_metrics = run_epoch(model, loaders["val"], device, None)
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.5f} "
            f"train_angle={train_metrics['mean_angle_degrees']:.2f} "
            f"val_loss={validation_metrics['loss']:.5f} "
            f"val_angle={validation_metrics['mean_angle_degrees']:.2f}",
            flush=True,
        )
        with history_path.open("a", newline="") as stream:
            csv.writer(stream).writerow(
                [
                    epoch,
                    train_metrics["loss"],
                    train_metrics["mean_angle_degrees"],
                    validation_metrics["loss"],
                    validation_metrics["mean_angle_degrees"],
                    *validation_metrics["angle_degrees"],
                ]
            )
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": model_config,
            "horizons_minutes": dataset.horizons_minutes,
            "history_minutes": dataset.history_minutes,
            "sample_interval_seconds": dataset.sample_interval_seconds,
            "feature_names": FEATURE_NAMES,
            "feature_mean": feature_mean,
            "feature_standard_deviation": feature_standard_deviation,
            "validation_metrics": validation_metrics,
        }
        torch.save(checkpoint, arguments.output_dir / "last.pt")
        if float(validation_metrics["loss"]) < best_validation:
            best_validation = float(validation_metrics["loss"])
            torch.save(checkpoint, best_path)

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = run_epoch(model, loaders["test"], device, None)
    baseline_metrics = persistence_baseline(dataset, "test")
    summary = {
        "device": str(device),
        "samples": {
            partition: int(np.sum(dataset.split == partition))
            for partition in ("train", "val", "test")
        },
        "vessels": {
            partition: int(np.unique(dataset.mmsi[dataset.split == partition]).size)
            for partition in ("train", "val", "test")
        },
        "best_epoch": int(checkpoint["epoch"]),
        "validation_metrics": checkpoint["validation_metrics"],
        "test_metrics": test_metrics,
        "constant_course_test_baseline": baseline_metrics,
        "mean_test_improvement_degrees": (
            baseline_metrics["mean_angle_degrees"]
            - test_metrics["mean_angle_degrees"]
        ),
        "horizons_minutes": dataset.horizons_minutes,
    }
    (arguments.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
