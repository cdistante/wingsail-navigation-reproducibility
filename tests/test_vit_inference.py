"""Checks for NumPy-only inference from exported ViT checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from world.vit_inference import EnvironmentalVisionTransformerPolicy


class ViTInferenceTests(unittest.TestCase):
    def test_numpy_runtime_predicts_a_normalised_direction(self) -> None:
        configuration = {
            "input_channels": 1,
            "grid_size": 4,
            "patch_size": 2,
            "global_feature_count": 2,
            "embed_dim": 4,
            "num_heads": 2,
            "num_layers": 1,
            "mlp_ratio": 2.0,
            "layer_norm_epsilon": 1.0e-5,
        }
        arrays: dict[str, np.ndarray] = {
            "model_config_json": np.asarray(json.dumps(configuration)),
            "metadata_json": np.asarray(json.dumps({"global_feature_names": ["a", "b"]})),
            "grid_mean": np.zeros(1, dtype=np.float32),
            "grid_std": np.ones(1, dtype=np.float32),
            "global_mean": np.zeros(2, dtype=np.float32),
            "global_std": np.ones(2, dtype=np.float32),
            "class_token": np.zeros((1, 1, 4), dtype=np.float32),
            "position_embedding": np.zeros((1, 5, 4), dtype=np.float32),
            "patch_embedding.weight": np.zeros((4, 4), dtype=np.float32),
            "patch_embedding.bias": np.zeros(4, dtype=np.float32),
            "final_norm.weight": np.ones(4, dtype=np.float32),
            "final_norm.bias": np.zeros(4, dtype=np.float32),
            "environment_first.weight": np.zeros((4, 2), dtype=np.float32),
            "environment_first.bias": np.zeros(4, dtype=np.float32),
            "environment_second.weight": np.zeros((4, 4), dtype=np.float32),
            "environment_second.bias": np.zeros(4, dtype=np.float32),
            "head.weight": np.zeros((2, 8), dtype=np.float32),
            "head.bias": np.asarray([1.0, 0.0], dtype=np.float32),
        }
        for prefix, shape in (
            ("blocks.0.norm1", (4,)),
            ("blocks.0.norm2", (4,)),
        ):
            arrays[prefix + ".weight"] = np.ones(shape, dtype=np.float32)
            arrays[prefix + ".bias"] = np.zeros(shape, dtype=np.float32)
        arrays.update(
            {
                "blocks.0.qkv.weight": np.zeros((12, 4), dtype=np.float32),
                "blocks.0.qkv.bias": np.zeros(12, dtype=np.float32),
                "blocks.0.projection.weight": np.zeros((4, 4), dtype=np.float32),
                "blocks.0.projection.bias": np.zeros(4, dtype=np.float32),
                "blocks.0.mlp_first.weight": np.zeros((8, 4), dtype=np.float32),
                "blocks.0.mlp_first.bias": np.zeros(8, dtype=np.float32),
                "blocks.0.mlp_second.weight": np.zeros((4, 8), dtype=np.float32),
                "blocks.0.mlp_second.bias": np.zeros(4, dtype=np.float32),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "vit.npz"
            np.savez_compressed(checkpoint, **arrays)
            policy = EnvironmentalVisionTransformerPolicy(checkpoint)
            prediction = policy.predict(np.zeros((1, 4, 4)), np.zeros(2))

        np.testing.assert_allclose(prediction, [[1.0, 0.0]], atol=1.0e-7)
        self.assertAlmostEqual(policy.heading(np.zeros((1, 4, 4)), np.zeros(2)), 90.0)


if __name__ == "__main__":
    unittest.main()
