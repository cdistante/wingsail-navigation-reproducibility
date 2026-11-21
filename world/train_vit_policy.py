"""Train and export the environmental Vision Transformer direction policy.

This module may be launched either as ``python -m world.train_vit_policy`` or
directly with a PyTorch-capable interpreter:

    python world/train_vit_policy.py --data data/training/vit_environmental.npz ...

The direct-file form is useful when training happens in a dedicated ML Conda
environment while the simulator remains in a separate environment.
"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os

import numpy as np

try:
    from contextlib import nullcontext
except ImportError:  # Python 3.6 compatibility for the legacy local trainer.
    class nullcontext(object):
        def __enter__(self):
            return None

        def __exit__(self, *arguments):
            return False

try:
    import torch
    from torch.nn import functional as F
except ImportError as error:
    raise RuntimeError(
        "Vision Transformer training requires PyTorch. Use a compatible Python/PyTorch environment."
    ) from error

try:
    from .vit_policy import EnvironmentalVisionTransformer
except (ImportError, ValueError):
    from vit_policy import EnvironmentalVisionTransformer


def _normalisation(values, indices, axes):
    subset = values[indices]
    mean = subset.mean(axis=axes, dtype=np.float64).astype(np.float32)
    std = subset.std(axis=axes, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, 1.0e-6)


def _normalise_grids(grids, mean, std):
    return (grids - mean[None, :, None, None]) / std[None, :, None, None]


def _normalise_features(features, mean, std):
    return (features - mean[None, :]) / std[None, :]


def _angular_metrics(prediction, target):
    prediction = F.normalize(prediction, dim=1, eps=1.0e-8)
    target = F.normalize(target, dim=1, eps=1.0e-8)
    cosine = torch.clamp((prediction * target).sum(dim=1), -1.0, 1.0)
    loss = (1.0 - cosine).mean()
    angle = torch.acos(cosine).mean() * (180.0 / np.pi)
    return loss, angle


def _autocast(amp_enabled):
    """Avoid invoking CUDA AMP at all in CPU-only compatibility environments."""

    return torch.cuda.amp.autocast() if amp_enabled else nullcontext()


def _evaluate(model, grids, features, targets, indices, batch_size, device, amp_enabled=False):
    if len(indices) == 0:
        return float("nan"), float("nan")
    model.eval()
    total_loss = 0.0
    total_angle = 0.0
    total_samples = 0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            grid_batch = torch.from_numpy(grids[batch_indices]).to(device)
            feature_batch = torch.from_numpy(features[batch_indices]).to(device)
            target_batch = torch.from_numpy(targets[batch_indices]).to(device)
            with _autocast(amp_enabled):
                loss, angle = _angular_metrics(model(grid_batch, feature_batch), target_batch)
            count = len(batch_indices)
            total_loss += float(loss.cpu().item()) * count
            total_angle += float(angle.cpu().item()) * count
            total_samples += count
    return total_loss / total_samples, total_angle / total_samples


def _export_numpy_checkpoint(path, model, grid_mean, grid_std, global_mean, global_std, metadata, metrics):
    arrays = {
        "model_config_json": np.asarray(json.dumps(model.config(), sort_keys=True)),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "metrics_json": np.asarray(json.dumps(metrics, sort_keys=True)),
        "grid_mean": np.asarray(grid_mean, dtype=np.float32),
        "grid_std": np.asarray(grid_std, dtype=np.float32),
        "global_mean": np.asarray(global_mean, dtype=np.float32),
        "global_std": np.asarray(global_std, dtype=np.float32),
    }
    for name, value in model.state_dict().items():
        arrays[name] = value.detach().cpu().numpy().astype(np.float32)
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return destination


def _grouped_metrics(model, grids, features, targets, indices, groups, batch_size, device, amp_enabled):
    """Report held-out metrics by route/time partition or AIS source."""

    result = {}
    for group in sorted(set(groups[indices].tolist())):
        group_indices = indices[groups[indices] == group]
        loss, angle = _evaluate(
            model,
            grids,
            features,
            targets,
            group_indices,
            batch_size,
            device,
            amp_enabled,
        )
        result[str(group)] = {
            "samples": int(len(group_indices)),
            "loss": float(loss),
            "angular_error_degrees": float(angle),
        }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Train a patch-token Vision Transformer from environmental occupancy grids."
    )
    parser.add_argument("--data", required=True, help="Dataset archive made by world.vit_dataset.")
    parser.add_argument("--output", required=True, help="Output NumPy-inference .npz checkpoint.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument(
        "--gpus",
        default="0",
        help="Comma-separated CUDA GPU indices; use 0,1 for two Titan RTX GPUs.",
    )
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Prioritise repeatability over CUDA throughput where supported.",
    )
    arguments = parser.parse_args()
    if arguments.epochs < 1 or arguments.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive.")
    if arguments.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive.")

    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    if arguments.device == "cuda":
        if not torch.cuda.is_available():
            parser.error("Requested CUDA device is unavailable.")
        try:
            gpu_indices = tuple(int(value) for value in arguments.gpus.split(",") if value.strip())
        except ValueError:
            parser.error("--gpus must be a comma-separated list of CUDA indices.")
        if not gpu_indices or min(gpu_indices) < 0 or max(gpu_indices) >= torch.cuda.device_count():
            parser.error("--gpus contains an unavailable CUDA index.")
        torch.cuda.manual_seed_all(arguments.seed)
        device = torch.device("cuda:{0}".format(gpu_indices[0]))
    elif arguments.device != "cpu":
        backend = getattr(torch.backends, arguments.device, None)
        if backend is None or not backend.is_available():
            parser.error("Requested Torch device is unavailable.")
        gpu_indices = ()
        device = torch.device(arguments.device)
    else:
        gpu_indices = ()
        device = torch.device("cpu")
    if arguments.amp and arguments.device != "cuda":
        parser.error("--amp requires --device cuda.")
    amp_enabled = bool(arguments.amp and arguments.device == "cuda")
    if arguments.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)
    with np.load(arguments.data, allow_pickle=False) as archive:
        grids = np.asarray(archive["grids"], dtype=np.float32)
        features = np.asarray(archive["global_features"], dtype=np.float32)
        targets = np.asarray(archive["targets"], dtype=np.float32)
        split = np.asarray(archive["split"], dtype=np.int8)
        metadata = json.loads(str(archive["metadata_json"].item()))
        partitions = (
            np.asarray(archive["partitions"], dtype=str)
            if "partitions" in archive.files
            else np.asarray(["unspecified"] * len(grids), dtype=str)
        )
        traffic_sources = (
            np.asarray(archive["traffic_sources"], dtype=str)
            if "traffic_sources" in archive.files
            else np.asarray(["unspecified"] * len(grids), dtype=str)
        )
    if grids.ndim != 4 or features.ndim != 2 or targets.shape != (len(grids), 2):
        raise ValueError("Dataset arrays do not match the environmental ViT schema.")
    if (
        grids.shape[0] != features.shape[0]
        or len(split) != len(grids)
        or len(partitions) != len(grids)
        or len(traffic_sources) != len(grids)
    ):
        raise ValueError("Dataset sample dimensions are inconsistent.")
    train_indices = np.flatnonzero(split == 0)
    validation_indices = np.flatnonzero(split == 1)
    test_indices = np.flatnonzero(split == 2)
    if not len(train_indices) or not len(validation_indices) or not len(test_indices):
        raise ValueError("Dataset must contain non-empty train, validation, and test partitions.")
    print(
        "Loaded {0} samples: train={1}, validation={2}, test={3}; grid={4}, global_features={5}.".format(
            len(grids),
            len(train_indices),
            len(validation_indices),
            len(test_indices),
            tuple(grids.shape[1:]),
            features.shape[1],
        )
    )
    grid_mean, grid_std = _normalisation(grids, train_indices, axes=(0, 2, 3))
    global_mean, global_std = _normalisation(features, train_indices, axes=(0,))
    grids = _normalise_grids(grids, grid_mean, grid_std).astype(np.float32)
    features = _normalise_features(features, global_mean, global_std).astype(np.float32)
    model = EnvironmentalVisionTransformer(
        input_channels=grids.shape[1],
        grid_size=grids.shape[2],
        patch_size=arguments.patch_size,
        global_feature_count=features.shape[1],
        embed_dim=arguments.embed_dim,
        num_heads=arguments.heads,
        num_layers=arguments.layers,
    ).to(device)
    if len(gpu_indices) > 1:
        model = torch.nn.DataParallel(model, device_ids=list(gpu_indices), output_device=gpu_indices[0])
    optimiser = torch.optim.Adam(model.parameters(), lr=arguments.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_state = None
    best_validation = float("inf")
    history = []
    for epoch in range(1, arguments.epochs + 1):
        model.train()
        shuffled = train_indices.copy()
        np.random.shuffle(shuffled)
        total_training_loss = 0.0
        total_training_samples = 0
        for start in range(0, len(shuffled), arguments.batch_size):
            batch_indices = shuffled[start : start + arguments.batch_size]
            grid_batch = torch.from_numpy(grids[batch_indices]).to(device)
            feature_batch = torch.from_numpy(features[batch_indices]).to(device)
            target_batch = torch.from_numpy(targets[batch_indices]).to(device)
            optimiser.zero_grad()
            with _autocast(amp_enabled):
                loss, _ = _angular_metrics(model(grid_batch, feature_batch), target_batch)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            count = len(batch_indices)
            total_training_loss += float(loss.detach().cpu().item()) * count
            total_training_samples += count
        validation_loss, validation_angle = _evaluate(
            model,
            grids,
            features,
            targets,
            validation_indices,
            arguments.batch_size,
            device,
            amp_enabled,
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": total_training_loss / total_training_samples,
                "validation_loss": validation_loss,
                "validation_angular_error_degrees": validation_angle,
            }
        )
        print(
            "epoch {0:03d}: train_loss={1:.6f}, validation_loss={2:.6f}, "
            "validation_angle={3:.2f} deg".format(
                epoch, history[-1]["training_loss"], validation_loss, validation_angle
            )
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
    assert best_state is not None
    model.load_state_dict(best_state)
    test_loss, test_angle = _evaluate(
        model,
        grids,
        features,
        targets,
        test_indices,
        arguments.batch_size,
        device,
        amp_enabled,
    )
    test_by_partition = _grouped_metrics(
        model,
        grids,
        features,
        targets,
        test_indices,
        partitions,
        arguments.batch_size,
        device,
        amp_enabled,
    )
    test_by_traffic_source = _grouped_metrics(
        model,
        grids,
        features,
        targets,
        test_indices,
        traffic_sources,
        arguments.batch_size,
        device,
        amp_enabled,
    )
    metrics = {
        "best_validation_loss": float(best_validation),
        "test_loss": float(test_loss),
        "test_angular_error_degrees": float(test_angle),
        "history": history,
        "sample_counts": {
            "train": int(len(train_indices)),
            "validation": int(len(validation_indices)),
            "test": int(len(test_indices)),
        },
        "test_by_partition": test_by_partition,
        "test_by_traffic_source": test_by_traffic_source,
        "training_runtime": {
            "device": str(device),
            "cuda_gpus": list(gpu_indices),
            "amp": amp_enabled,
            "deterministic": bool(arguments.deterministic),
            "torch_version": torch.__version__,
        },
    }
    export_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    output = _export_numpy_checkpoint(
        arguments.output,
        export_model,
        grid_mean,
        grid_std,
        global_mean,
        global_std,
        metadata,
        metrics,
    )
    print(
        "Saved best checkpoint to {0}; test_loss={1:.6f}, test_angle={2:.2f} deg.".format(
            output, test_loss, test_angle
        )
    )


if __name__ == "__main__":
    main()
