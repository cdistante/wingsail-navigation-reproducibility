"""NumPy inference runtime for exported environmental Vision Transformer checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class EnvironmentalVisionTransformerPolicy:
    """Run a trained occupancy-grid ViT without importing PyTorch.

    The trainer exports its linear, normalisation, attention, and LayerNorm
    parameters to ``.npz``.  Keeping rollout inference in NumPy means the
    existing simulator remains fully runnable in its Python 3.13 environment,
    even when the locally available PyTorch distribution targets another
    interpreter version.
    """

    def __init__(self, checkpoint: Path | str) -> None:
        self.path = Path(checkpoint)
        if not self.path.exists():
            raise RuntimeError(f"Vision Transformer checkpoint not found: {self.path}.")
        with np.load(self.path, allow_pickle=False) as archive:
            self.arrays = {name: np.asarray(archive[name]) for name in archive.files}
        try:
            self.config = json.loads(str(self.arrays["model_config_json"].item()))
            self.metadata = json.loads(str(self.arrays["metadata_json"].item()))
        except KeyError as error:
            raise ValueError(f"{self.path} is not an exported environmental ViT checkpoint.") from error
        self.grid_mean = self.arrays["grid_mean"].astype(np.float32)
        self.grid_std = self.arrays["grid_std"].astype(np.float32)
        self.global_mean = self.arrays["global_mean"].astype(np.float32)
        self.global_std = self.arrays["global_std"].astype(np.float32)
        self._validate()

    @property
    def input_channels(self) -> int:
        return int(self.config["input_channels"])

    @property
    def grid_size(self) -> int:
        return int(self.config["grid_size"])

    @property
    def global_feature_count(self) -> int:
        return int(self.config["global_feature_count"])

    def predict(self, grid: np.ndarray, global_features: np.ndarray) -> np.ndarray:
        """Predict unit Earth-frame ``[east, north]`` vectors for one or more grids."""

        grids = np.asarray(grid, dtype=np.float32)
        features = np.asarray(global_features, dtype=np.float32)
        if grids.ndim == 3:
            grids = grids[None, ...]
        if features.ndim == 1:
            features = features[None, ...]
        if grids.shape[1:] != (self.input_channels, self.grid_size, self.grid_size):
            raise ValueError(
                "Grid shape does not match checkpoint; expected "
                f"({self.input_channels}, {self.grid_size}, {self.grid_size}), got {grids.shape[1:]}.")
        if features.shape != (grids.shape[0], self.global_feature_count):
            raise ValueError(
                "Global feature shape does not match checkpoint; expected "
                f"({grids.shape[0]}, {self.global_feature_count}), got {features.shape}.")
        grids = (grids - self.grid_mean[None, :, None, None]) / self.grid_std[
            None, :, None, None
        ]
        features = (features - self.global_mean[None, :]) / self.global_std[None, :]
        output = self._forward(grids, features)
        norm = np.linalg.norm(output, axis=1, keepdims=True)
        return output / np.maximum(norm, 1.0e-8)

    def heading(self, grid: np.ndarray, global_features: np.ndarray) -> float:
        """Return a single true-heading prediction in degrees."""

        direction = self.predict(grid, global_features)
        if len(direction) != 1:
            raise ValueError("heading expects exactly one grid and feature vector.")
        return float(np.degrees(np.arctan2(direction[0, 0], direction[0, 1])) % 360.0)

    def _forward(self, grids: np.ndarray, global_features: np.ndarray) -> np.ndarray:
        patch_size = int(self.config["patch_size"])
        embed_dim = int(self.config["embed_dim"])
        heads = int(self.config["num_heads"])
        layers = int(self.config["num_layers"])
        patches_per_side = self.grid_size // patch_size
        batch = grids.shape[0]
        patches = grids.reshape(
            batch,
            self.input_channels,
            patches_per_side,
            patch_size,
            patches_per_side,
            patch_size,
        )
        patches = patches.transpose(0, 2, 4, 1, 3, 5).reshape(batch, -1, self.input_channels * patch_size**2)
        tokens = self._linear(patches, "patch_embedding")
        class_token = np.repeat(self.arrays["class_token"], batch, axis=0)
        tokens = np.concatenate((class_token, tokens), axis=1) + self.arrays["position_embedding"]
        for index in range(layers):
            prefix = f"blocks.{index}"
            residual = tokens
            value = self._layer_norm(tokens, prefix + ".norm1")
            qkv = self._linear(value, prefix + ".qkv")
            head_dim = embed_dim // heads
            qkv = qkv.reshape(batch, qkv.shape[1], 3, heads, head_dim).transpose(2, 0, 3, 1, 4)
            query, key, value = qkv[0], qkv[1], qkv[2]
            score = np.matmul(query, key.swapaxes(-2, -1)) * (head_dim ** -0.5)
            attention = self._softmax(score, axis=-1)
            value = np.matmul(attention, value).transpose(0, 2, 1, 3).reshape(batch, -1, embed_dim)
            tokens = residual + self._linear(value, prefix + ".projection")
            residual = tokens
            value = self._layer_norm(tokens, prefix + ".norm2")
            value = self._gelu(self._linear(value, prefix + ".mlp_first"))
            tokens = residual + self._linear(value, prefix + ".mlp_second")
        grid_embedding = self._layer_norm(tokens, "final_norm")[:, 0]
        environmental_embedding = np.maximum(
            self._linear(global_features, "environment_first"), 0.0
        )
        environmental_embedding = np.maximum(
            self._linear(environmental_embedding, "environment_second"), 0.0
        )
        return self._linear(
            np.concatenate((grid_embedding, environmental_embedding), axis=1), "head"
        )

    def _linear(self, value: np.ndarray, name: str) -> np.ndarray:
        return value @ self.arrays[name + ".weight"].T + self.arrays[name + ".bias"]

    def _layer_norm(self, value: np.ndarray, name: str) -> np.ndarray:
        epsilon = float(self.config["layer_norm_epsilon"])
        mean = value.mean(axis=-1, keepdims=True)
        variance = ((value - mean) ** 2).mean(axis=-1, keepdims=True)
        normalised = (value - mean) / np.sqrt(variance + epsilon)
        return normalised * self.arrays[name + ".weight"] + self.arrays[name + ".bias"]

    @staticmethod
    def _gelu(value: np.ndarray) -> np.ndarray:
        return 0.5 * value * (
            1.0 + np.tanh(0.7978845608028654 * (value + 0.044715 * value**3))
        )

    @staticmethod
    def _softmax(value: np.ndarray, axis: int) -> np.ndarray:
        shifted = value - np.max(value, axis=axis, keepdims=True)
        exponent = np.exp(shifted)
        return exponent / exponent.sum(axis=axis, keepdims=True)

    def _validate(self) -> None:
        required = {
            "grid_mean",
            "grid_std",
            "global_mean",
            "global_std",
            "class_token",
            "position_embedding",
            "patch_embedding.weight",
            "patch_embedding.bias",
            "final_norm.weight",
            "final_norm.bias",
            "environment_first.weight",
            "environment_first.bias",
            "environment_second.weight",
            "environment_second.bias",
            "head.weight",
            "head.bias",
        }
        for index in range(int(self.config["num_layers"])):
            for suffix in (
                "norm1.weight",
                "norm1.bias",
                "qkv.weight",
                "qkv.bias",
                "projection.weight",
                "projection.bias",
                "norm2.weight",
                "norm2.bias",
                "mlp_first.weight",
                "mlp_first.bias",
                "mlp_second.weight",
                "mlp_second.bias",
            ):
                required.add(f"blocks.{index}.{suffix}")
        missing = sorted(required.difference(self.arrays))
        if missing:
            raise ValueError(f"ViT checkpoint is missing parameter(s): {', '.join(missing)}.")
        if self.grid_mean.shape != (self.input_channels,) or self.grid_std.shape != (
            self.input_channels,
        ):
            raise ValueError("ViT checkpoint grid normalisation does not match its channel count.")
        if self.global_mean.shape != (self.global_feature_count,) or self.global_std.shape != (
            self.global_feature_count,
        ):
            raise ValueError("ViT checkpoint global normalisation does not match its feature count.")
        if np.any(self.grid_std <= 0.0) or np.any(self.global_std <= 0.0):
            raise ValueError("ViT checkpoint normalisation standard deviations must be positive.")
