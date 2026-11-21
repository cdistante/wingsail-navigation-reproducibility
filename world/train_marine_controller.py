"""Train the combined planner-conditioned marine actuator Transformer."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .marine_controller import PLANNER_MODES, PlannerConditionedMarineTransformer, checkpoint_payload


class MarineShardDataset(Dataset):
    """Indexed sharded dataset with episode metadata for leakage-free splits."""

    def __init__(self, dataset_dir: Path | list[Path] | tuple[Path, ...]) -> None:
        directories = (
            (dataset_dir,)
            if isinstance(dataset_dir, Path)
            else tuple(Path(path) for path in dataset_dir)
        )
        self.samples: list[tuple[Path, int, tuple[int, int]]] = []
        for directory_index, directory in enumerate(directories):
            with (directory / "index.csv").open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    path = directory / row["shard"]
                    with np.load(path) as shard:
                        self.samples.extend(
                            (
                                path,
                                index,
                                (directory_index, int(shard["episode_id"][index])),
                            )
                            for index in range(len(shard["maps"]))
                        )
        if not self.samples:
            raise ValueError("Dataset contains no samples.")
        self._cache_path: Path | None = None
        self._cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _shard(self, path: Path) -> dict[str, np.ndarray]:
        if path != self._cache_path:
            with np.load(path) as data:
                self._cache = {key: data[key] for key in data.files}
            self._cache_path = path
        return self._cache

    def numpy_item(self, index: int) -> dict[str, np.ndarray]:
        path, local_index, _ = self.samples[index]
        return {key: value[local_index] for key, value in self._shard(path).items()}

    def episode_group(self, index: int) -> tuple[int, int]:
        return self.samples[index][2]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.numpy_item(index)
        return {
            "maps": torch.from_numpy(np.asarray(item["maps"], dtype=np.float32)),
            "global_state": torch.from_numpy(np.asarray(item["global_state"], dtype=np.float32)),
            "planner_id": torch.tensor(int(item["planner_id"]), dtype=torch.long),
            "actions": torch.from_numpy(np.asarray(item["actions"], dtype=np.float32)),
            "confidence_target": torch.tensor(
                float(item.get("confidence_target", 1.0)), dtype=torch.float32
            ),
        }


def grouped_split(dataset: MarineShardDataset, validation_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    """Split whole environmental episodes, never individual samples."""

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be in (0, 1).")
    groups: dict[tuple[int, int], list[int]] = {}
    for index in range(len(dataset)):
        episode = dataset.episode_group(index)
        groups.setdefault(episode, []).append(index)
    if len(groups) < 2:
        # Smoke datasets may have one episode; a deterministic sample split is explicit.
        indices = np.arange(len(dataset))
        return indices[1:].tolist(), indices[:1].tolist()
    episodes = sorted(groups)
    np.random.default_rng(seed).shuffle(episodes)
    validation_count = max(1, min(len(episodes) - 1, round(len(episodes) * validation_ratio)))
    validation_episodes = set(episodes[:validation_count])
    train = [index for episode, values in groups.items() if episode not in validation_episodes for index in values]
    validation = [index for episode, values in groups.items() if episode in validation_episodes for index in values]
    return train, validation


def _metrics(model, loader, device) -> dict[str, object]:
    model.eval()
    totals = np.zeros(3, dtype=float)
    planner = {mode: np.zeros(3, dtype=float) for mode in PLANNER_MODES}
    with torch.inference_mode():
        for batch in loader:
            output = model(
                batch["maps"].to(device),
                batch["global_state"].to(device),
                batch["planner_id"].to(device),
            )
            error = torch.abs(output[:, :2] - batch["actions"].to(device)).cpu().numpy()
            ids = batch["planner_id"].numpy()
            totals += (error[:, 0].sum(), error[:, 1].sum(), len(error))
            for mode_id, mode in enumerate(PLANNER_MODES):
                selected = error[ids == mode_id]
                if len(selected):
                    planner[mode] += (selected[:, 0].sum(), selected[:, 1].sum(), len(selected))
    def summarize(values: np.ndarray) -> dict[str, float | int]:
        count = int(values[2])
        return {
            "rudder_mae": float(values[0] / max(count, 1)),
            "flap_mae": float(values[1] / max(count, 1)),
            "samples": count,
        }
    return {"overall": summarize(totals), "per_planner": {key: summarize(value) for key, value in planner.items()}}


def train(
    dataset_dir: Path | list[Path] | tuple[Path, ...],
    output_dir: Path,
    *,
    epochs: int = 10,
    batch_size: int = 16,
    seed: int = 42,
    validation_ratio: float = 0.2,
    device: str = "auto",
    d_model: int = 96,
    num_layers: int = 2,
) -> dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    target_device = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    )
    dataset = MarineShardDataset(dataset_dir)
    all_globals = np.stack([dataset.numpy_item(index)["global_state"] for index in range(len(dataset))])
    mean = all_globals.mean(axis=0).astype(np.float32)
    std = all_globals.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    train_indices, validation_indices = grouped_split(dataset, validation_ratio, seed)

    class NormalizedSubset(Dataset):
        def __init__(self, indices: list[int]) -> None:
            self.indices = indices
        def __len__(self) -> int:
            return len(self.indices)
        def __getitem__(self, index: int):
            value = dataset[self.indices[index]]
            value["global_state"] = (value["global_state"] - torch.from_numpy(mean)) / torch.from_numpy(std)
            return value

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(NormalizedSubset(train_indices), batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(NormalizedSubset(validation_indices), batch_size=batch_size)
    grid_size = int(dataset.numpy_item(0)["maps"].shape[-1])
    model = PlannerConditionedMarineTransformer(
        grid_size=grid_size, d_model=d_model, num_layers=num_layers
    ).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best = float("inf")
    normalization = {"global_mean": mean.tolist(), "global_std": std.tolist()}
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        count = 0
        for batch in train_loader:
            output = model(
                batch["maps"].to(target_device),
                batch["global_state"].to(target_device),
                batch["planner_id"].to(target_device),
            )
            target = batch["actions"].to(target_device)
            actuator_loss = F.mse_loss(output[:, :2] / torch.tensor((30.0, 15.0), device=target_device), target / torch.tensor((30.0, 15.0), device=target_device))
            confidence_loss = F.binary_cross_entropy(
                output[:, 2],
                batch["confidence_target"].to(target_device),
            )
            loss = actuator_loss + 0.05 * confidence_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.item()) * len(target)
            count += len(target)
        metrics = _metrics(model, validation_loader, target_device)
        validation_loss = float(metrics["overall"]["rudder_mae"]) / 30.0 + float(metrics["overall"]["flap_mae"]) / 15.0
        row = {"epoch": epoch, "train_loss": running / max(count, 1), "validation_scaled_mae": validation_loss}
        history.append(row)
        payload = checkpoint_payload(model, normalization, epoch=epoch, seed=seed, metrics=metrics)
        torch.save(payload, output_dir / "last.pt")
        if validation_loss < best:
            best = validation_loss
            torch.save(payload, output_dir / "best.pt")
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    final_metrics = _metrics(model, validation_loader, target_device)
    summary = {
        "device": str(target_device),
        "train_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "split": "episode_grouped" if len(set(int(dataset.numpy_item(i)["episode_id"]) for i in range(len(dataset)))) > 1 else "deterministic_smoke_sample",
        "best_validation_scaled_mae": best,
        "metrics": final_metrics,
        "normalization": normalization,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        nargs="+",
        required=True,
        help="One or more expert/DAgger dataset directories.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(train(args.dataset_dir, args.output_dir, epochs=args.epochs, batch_size=args.batch_size, seed=args.seed, validation_ratio=args.validation_ratio, device=args.device, d_model=args.d_model, num_layers=args.layers), indent=2))


if __name__ == "__main__":
    main()
