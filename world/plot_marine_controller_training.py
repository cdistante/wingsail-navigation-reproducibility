"""Plot training and held-out actuator errors for the marine controller."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .marine_controller import PLANNER_MODES, PlannerConditionedMarineTransformer
from .train_marine_controller import MarineShardDataset, grouped_split


def _history(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Training history is empty: {path}")
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=float)
        for key in rows[0]
    }


class _ValidationDataset(Dataset):
    def __init__(
        self,
        dataset: MarineShardDataset,
        indices: list[int],
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        self.dataset = dataset
        self.indices = indices
        self.mean = torch.from_numpy(mean.astype(np.float32))
        self.std = torch.from_numpy(std.astype(np.float32))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        value = self.dataset[self.indices[index]]
        value["global_state"] = (value["global_state"] - self.mean) / self.std
        return value


def evaluate(
    dataset_dirs: list[Path],
    checkpoint: Path,
    *,
    batch_size: int,
    validation_ratio: float,
    seed: int,
    device: str,
) -> dict[str, np.ndarray]:
    dataset = MarineShardDataset(dataset_dirs)
    _, validation_indices = grouped_split(dataset, validation_ratio, seed)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = PlannerConditionedMarineTransformer(**payload["model_config"])
    model.load_state_dict(payload["model_state"])
    target_device = torch.device(device)
    model.to(target_device).eval()
    normalization = payload["normalization"]
    mean = np.asarray(normalization["global_mean"], dtype=np.float32)
    std = np.maximum(
        np.asarray(normalization["global_std"], dtype=np.float32), 1e-6
    )
    loader = DataLoader(
        _ValidationDataset(dataset, validation_indices, mean, std),
        batch_size=batch_size,
    )
    predictions = []
    targets = []
    planner_ids = []
    confidence_targets = []
    with torch.inference_mode():
        for batch in loader:
            output = model(
                batch["maps"].to(target_device),
                batch["global_state"].to(target_device),
                batch["planner_id"].to(target_device),
            )
            predictions.append(output.cpu().numpy())
            targets.append(batch["actions"].numpy())
            planner_ids.append(batch["planner_id"].numpy())
            confidence_targets.append(batch["confidence_target"].numpy())
    return {
        "prediction": np.concatenate(predictions),
        "target": np.concatenate(targets),
        "planner_id": np.concatenate(planner_ids),
        "confidence_target": np.concatenate(confidence_targets),
    }


def plot_all(
    evaluation: dict[str, np.ndarray],
    history: dict[str, np.ndarray],
    output_dir: Path,
    *,
    baseline_history: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction = evaluation["prediction"]
    target = evaluation["target"]
    planner_ids = evaluation["planner_id"].astype(int)
    errors = np.abs(prediction[:, :2] - target)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(history["epoch"], history["train_loss"], label="DAgger train loss")
    axes[1].plot(
        history["epoch"],
        history["validation_scaled_mae"],
        label="DAgger validation scaled MAE",
    )
    if baseline_history is not None:
        axes[0].plot(
            baseline_history["epoch"],
            baseline_history["train_loss"],
            linestyle="--",
            label="Expert-only train loss",
        )
        axes[1].plot(
            baseline_history["epoch"],
            baseline_history["validation_scaled_mae"],
            linestyle="--",
            label="Expert-only validation scaled MAE",
        )
    best_index = int(np.argmin(history["validation_scaled_mae"]))
    axes[1].scatter(
        history["epoch"][best_index],
        history["validation_scaled_mae"][best_index],
        color="red",
        zorder=3,
        label=f"Best epoch {int(history['epoch'][best_index])}",
    )
    axes[0].set(title="Training objective", xlabel="Epoch", ylabel="Loss")
    axes[1].set(
        title="Held-out validation objective",
        xlabel="Epoch",
        ylabel="Scaled actuator MAE",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(output_dir / "learning_curves.png", dpi=180)
    plt.close(figure)

    planner_mae = np.zeros((len(PLANNER_MODES), 2), dtype=float)
    planner_counts = np.zeros(len(PLANNER_MODES), dtype=int)
    for planner_id in range(len(PLANNER_MODES)):
        selected = errors[planner_ids == planner_id]
        planner_counts[planner_id] = len(selected)
        planner_mae[planner_id] = selected.mean(axis=0) if len(selected) else np.nan
    x = np.arange(len(PLANNER_MODES))
    figure, axis = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    width = 0.36
    axis.bar(x - width / 2, planner_mae[:, 0], width, label="Rudder MAE")
    axis.bar(x + width / 2, planner_mae[:, 1], width, label="Flap MAE")
    axis.set(
        title="Held-out actuator angular error by planner",
        xlabel="Conditioned planner mode",
        ylabel="Mean absolute error [degrees]",
        xticks=x,
        xticklabels=[mode.title() for mode in PLANNER_MODES],
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.savefig(output_dir / "angular_mae_by_planner.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    limits = ((-30.0, 30.0), (-15.0, 15.0))
    names = ("Rudder", "Flap")
    for index, axis in enumerate(axes):
        for planner_id, mode in enumerate(PLANNER_MODES):
            selected = planner_ids == planner_id
            axis.scatter(
                target[selected, index],
                prediction[selected, index],
                s=9,
                alpha=0.35,
                label=mode.title(),
            )
        axis.plot(limits[index], limits[index], color="black", linestyle="--")
        axis.set(
            title=f"{names[index]} prediction versus MPC expert",
            xlabel="Expert angle [degrees]",
            ylabel="Predicted angle [degrees]",
            xlim=limits[index],
            ylim=limits[index],
        )
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(output_dir / "prediction_vs_expert.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for index, axis in enumerate(axes):
        for planner_id, mode in enumerate(PLANNER_MODES):
            axis.hist(
                errors[planner_ids == planner_id, index],
                bins=30,
                density=True,
                histtype="step",
                linewidth=1.8,
                label=mode.title(),
            )
        axis.set(
            title=f"{names[index]} absolute angular error",
            xlabel="Absolute error [degrees]",
            ylabel="Density",
        )
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(output_dir / "angular_error_distributions.png", dpi=180)
    plt.close(figure)

    confidence = prediction[:, 2]
    agreement = (
        (errors[:, 0] <= 3.0) & (errors[:, 1] <= 3.0)
    ).astype(float)
    bins = np.linspace(0.0, 1.0, 11)
    bin_index = np.clip(np.digitize(confidence, bins) - 1, 0, len(bins) - 2)
    centres = (bins[:-1] + bins[1:]) / 2.0
    observed = np.full(len(centres), np.nan)
    counts = np.zeros(len(centres), dtype=int)
    for index in range(len(centres)):
        selected = bin_index == index
        counts[index] = int(selected.sum())
        if counts[index]:
            observed[index] = float(agreement[selected].mean())
    figure, axis = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    valid = counts > 0
    axis.plot([0, 1], [0, 1], color="black", linestyle="--", label="Ideal")
    axis.plot(centres[valid], observed[valid], marker="o", label="Observed")
    for x_value, y_value, count in zip(
        centres[valid], observed[valid], counts[valid]
    ):
        axis.annotate(str(count), (x_value, y_value), xytext=(0, 6), textcoords="offset points", ha="center")
    axis.set(
        title="Confidence calibration\n(numbers are validation samples)",
        xlabel="Predicted confidence",
        ylabel="Fraction within 3° on both actuators",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output_dir / "confidence_calibration.png", dpi=180)
    plt.close(figure)

    summary = {
        "samples": int(len(target)),
        "best_epoch": int(history["epoch"][best_index]),
        "best_validation_scaled_mae": float(
            history["validation_scaled_mae"][best_index]
        ),
        "overall": {
            "rudder_mae_degrees": float(errors[:, 0].mean()),
            "flap_mae_degrees": float(errors[:, 1].mean()),
            "both_within_3_degrees": float(agreement.mean()),
        },
        "per_planner": {
            mode: {
                "samples": int(planner_counts[index]),
                "rudder_mae_degrees": float(planner_mae[index, 0]),
                "flap_mae_degrees": float(planner_mae[index, 1]),
            }
            for index, mode in enumerate(PLANNER_MODES)
        },
    }
    (output_dir / "plot_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset-dir", type=Path, nargs="+", required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--history", type=Path, required=True)
    result.add_argument("--baseline-history", type=Path)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--validation-ratio", type=float, default=0.2)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--device", default="cuda")
    return result


def main() -> None:
    arguments = parser().parse_args()
    evaluation = evaluate(
        arguments.dataset_dir,
        arguments.checkpoint,
        batch_size=arguments.batch_size,
        validation_ratio=arguments.validation_ratio,
        seed=arguments.seed,
        device=arguments.device,
    )
    summary = plot_all(
        evaluation,
        _history(arguments.history),
        arguments.output_dir,
        baseline_history=(
            None
            if arguments.baseline_history is None
            else _history(arguments.baseline_history)
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
