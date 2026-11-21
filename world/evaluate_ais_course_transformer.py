"""Evaluate and plot temporal AIS course predictions on held-out vessels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .ais_course_transformer import (
    AISCourseTransformer,
    AISSequenceDataset,
    direction_to_course_degrees,
)


def angular_errors(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return per-sample, per-horizon circular errors in degrees."""

    prediction = np.asarray(prediction, dtype=float)
    target = np.asarray(target, dtype=float)
    prediction /= np.maximum(np.linalg.norm(prediction, axis=-1, keepdims=True), 1e-8)
    target /= np.maximum(np.linalg.norm(target, axis=-1, keepdims=True), 1e-8)
    cosine = np.clip(np.sum(prediction * target, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def horizon_metrics(
    errors: np.ndarray,
    baseline_errors: np.ndarray,
    horizons_minutes: tuple[int, ...],
) -> list[dict[str, float | int]]:
    """Summarize model and persistence errors at each horizon."""

    rows = []
    for index, horizon in enumerate(horizons_minutes):
        values = errors[:, index]
        baseline = baseline_errors[:, index]
        rows.append(
            {
                "horizon_minutes": horizon,
                "mean_error_degrees": float(values.mean()),
                "median_error_degrees": float(np.median(values)),
                "p90_error_degrees": float(np.percentile(values, 90)),
                "within_10_degrees": float(np.mean(values <= 10.0)),
                "within_20_degrees": float(np.mean(values <= 20.0)),
                "within_30_degrees": float(np.mean(values <= 30.0)),
                "baseline_mean_error_degrees": float(baseline.mean()),
                "mean_improvement_degrees": float(baseline.mean() - values.mean()),
            }
        )
    return rows


def predict_partition(
    dataset: AISSequenceDataset,
    checkpoint_path: Path,
    partition: str,
    *,
    batch_size: int,
    device_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return raw features, targets, model predictions, and baseline predictions."""

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device(
        "cuda"
        if device_name == "cuda"
        or (device_name == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = AISCourseTransformer(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    selected = dataset.split == partition
    if not np.any(selected):
        raise ValueError(f"AIS dataset partition {partition!r} is empty.")
    raw_features = dataset.features[selected]
    targets = dataset.targets[selected]
    mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
    standard_deviation = np.asarray(
        checkpoint["feature_standard_deviation"], dtype=np.float32
    )
    normalized = (raw_features - mean) / standard_deviation
    loader = DataLoader(
        TensorDataset(torch.from_numpy(normalized.astype(np.float32))),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    predictions = []
    with torch.no_grad():
        for (features,) in loader:
            output = F.normalize(model(features.to(device, non_blocking=True)), dim=-1)
            predictions.append(output.cpu().numpy())
    prediction = np.concatenate(predictions)
    current_course = raw_features[:, -1, 3:5]
    baseline = np.repeat(
        current_course[:, np.newaxis, :],
        len(dataset.horizons_minutes),
        axis=1,
    )
    return raw_features, targets, prediction, baseline


def save_evaluation_plots(
    output_dir: Path,
    dataset: AISSequenceDataset,
    partition_mask: np.ndarray,
    features: np.ndarray,
    targets: np.ndarray,
    predictions: np.ndarray,
    errors: np.ndarray,
    baseline_errors: np.ndarray,
    rows: list[dict[str, float | int]],
    *,
    history_path: Path | None,
    example_count: int = 6,
) -> None:
    """Write quantitative and qualitative evaluation figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    horizons = np.asarray(dataset.horizons_minutes)
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.plot(
        horizons,
        [row["mean_error_degrees"] for row in rows],
        marker="o",
        linewidth=2,
        label="AIS Transformer",
    )
    axis.plot(
        horizons,
        [row["baseline_mean_error_degrees"] for row in rows],
        marker="s",
        linewidth=2,
        label="constant reported COG",
    )
    axis.set_xlabel("Prediction horizon [minutes]")
    axis.set_ylabel("Mean angular error [degrees]")
    axis.set_title("Future AIS course prediction")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.savefig(output_dir / "horizon_error_comparison.png", dpi=170)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for index, horizon in enumerate(horizons):
        sorted_errors = np.sort(errors[:, index])
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        axis.plot(sorted_errors, cumulative, label=f"{horizon} min")
    axis.set_xlim(0.0, min(180.0, float(np.percentile(errors, 99.5)) + 5.0))
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Angular error [degrees]")
    axis.set_ylabel("Cumulative fraction")
    axis.set_title("AIS Transformer error CDF")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.savefig(output_dir / "error_cdf.png", dpi=170)
    plt.close(figure)

    if history_path is not None and history_path.exists():
        history = pd.read_csv(history_path)
        figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        axis.plot(history["epoch"], history["train_angle"], label="train")
        axis.plot(history["epoch"], history["val_angle"], label="validation")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Mean angular error [degrees]")
        axis.set_title("AIS Transformer learning curve")
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.savefig(output_dir / "learning_curve.png", dpi=170)
        plt.close(figure)

    test_indices = np.flatnonzero(partition_mask)
    selected_examples = np.linspace(
        0,
        len(features) - 1,
        min(example_count, len(features)),
        dtype=int,
    )
    columns = 3
    rows_count = int(np.ceil(len(selected_examples) / columns))
    figure, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(15, 4.5 * rows_count),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, local_index in zip(axes.ravel(), selected_examples):
        history = features[local_index, :, :2]
        axis.plot(history[:, 0], history[:, 1], color="0.35", linewidth=1.5)
        axis.scatter(history[-1, 0], history[-1, 1], color="black", s=35, zorder=4)
        for horizon_index, horizon in enumerate(horizons):
            length = 0.25 + 0.035 * float(horizon)
            actual = targets[local_index, horizon_index] * length
            predicted = predictions[local_index, horizon_index] * length
            axis.arrow(
                0.0,
                0.0,
                actual[0],
                actual[1],
                color=f"C{horizon_index}",
                width=0.008,
                length_includes_head=True,
                label=f"actual {horizon}m",
            )
            axis.arrow(
                0.0,
                0.0,
                predicted[0],
                predicted[1],
                color=f"C{horizon_index}",
                linestyle="--",
                alpha=0.65,
                width=0.004,
                length_includes_head=True,
            )
        global_index = test_indices[local_index]
        axis.set_title(
            f"MMSI {int(dataset.mmsi[global_index])}\n"
            f"mean error {errors[local_index].mean():.1f}°"
        )
        axis.set_xlabel("East from present [km]")
        axis.set_ylabel("North from present [km]")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(True, alpha=0.25)
    for axis in axes.ravel()[len(selected_examples):]:
        axis.set_visible(False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=len(horizons))
    figure.savefig(output_dir / "prediction_examples.png", dpi=170)
    plt.close(figure)

    predicted_courses = direction_to_course_degrees(predictions)
    actual_courses = direction_to_course_degrees(targets)
    np.savez_compressed(
        output_dir / "predictions.npz",
        predictions=predictions.astype(np.float32),
        targets=targets.astype(np.float32),
        predicted_courses_degrees=predicted_courses.astype(np.float32),
        actual_courses_degrees=actual_courses.astype(np.float32),
        errors_degrees=errors.astype(np.float32),
        baseline_errors_degrees=baseline_errors.astype(np.float32),
        mmsi=dataset.mmsi[partition_mask],
        timestamps_ns=dataset.timestamps_ns[partition_mask],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate temporal AIS course predictions.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partition", choices=("train", "val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--history", type=Path)
    arguments = parser.parse_args()

    dataset = AISSequenceDataset.load(arguments.dataset)
    features, targets, predictions, baseline = predict_partition(
        dataset,
        arguments.checkpoint,
        arguments.partition,
        batch_size=arguments.batch_size,
        device_name=arguments.device,
    )
    errors = angular_errors(predictions, targets)
    baseline_errors = angular_errors(baseline, targets)
    rows = horizon_metrics(errors, baseline_errors, dataset.horizons_minutes)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    with (arguments.output_dir / "per_horizon_metrics.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "partition": arguments.partition,
        "samples": len(features),
        "vessels": int(
            np.unique(dataset.mmsi[dataset.split == arguments.partition]).size
        ),
        "mean_error_degrees": float(errors.mean()),
        "median_error_degrees": float(np.median(errors)),
        "p90_error_degrees": float(np.percentile(errors, 90)),
        "baseline_mean_error_degrees": float(baseline_errors.mean()),
        "mean_improvement_degrees": float(baseline_errors.mean() - errors.mean()),
        "per_horizon": rows,
    }
    (arguments.output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    history_path = arguments.history or arguments.checkpoint.parent / "history.csv"
    partition_mask = dataset.split == arguments.partition
    save_evaluation_plots(
        arguments.output_dir,
        dataset,
        partition_mask,
        features,
        targets,
        predictions,
        errors,
        baseline_errors,
        rows,
        history_path=history_path,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
