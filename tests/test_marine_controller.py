from __future__ import annotations

import argparse
import shutil
import unittest
from pathlib import Path

import numpy as np
import torch

from world.boat import BoatAction, BoatState
from world.marine_controller import (
    GLOBAL_FEATURE_NAMES,
    MarineController,
    PlannerConditionedMarineTransformer,
    checkpoint_payload,
    map_tensor,
)
from world.run_legacy_fossen import (
    combined_controller_action_is_safe,
    combined_controller_cost_is_safe,
    validate_combined_controller_args,
)
from world.train_marine_controller import MarineShardDataset, grouped_split


class MarineControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.output = Path("data/results/test_marine_controller")
        cls.output.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.output, ignore_errors=True)

    def _checkpoint(self, confidence_bias: float) -> Path:
        model = PlannerConditionedMarineTransformer(
            grid_size=9, d_model=16, nhead=4, num_layers=1, dim_feedforward=32, dropout=0.0
        )
        for parameter in model.parameters():
            parameter.data.zero_()
        model.head[-1].bias.data[2] = confidence_bias
        path = self.output / f"confidence_{confidence_bias}.pt"
        torch.save(
            checkpoint_payload(
                model,
                {
                    "global_mean": [0.0] * len(GLOBAL_FEATURE_NAMES),
                    "global_std": [1.0] * len(GLOBAL_FEATURE_NAMES),
                },
            ),
            path,
        )
        return path

    def test_dataset_shape_and_grouped_split(self) -> None:
        dataset_dir = self.output / "dataset"
        dataset_dir.mkdir(exist_ok=True)
        samples = 6
        np.savez_compressed(
            dataset_dir / "marine_controller_0000.npz",
            maps=np.zeros((samples, 5, 9, 9), dtype=np.float32),
            global_state=np.zeros((samples, len(GLOBAL_FEATURE_NAMES)), dtype=np.float32),
            planner_id=np.asarray((0, 1, 2, 0, 1, 2), dtype=np.int64),
            actions=np.zeros((samples, 2), dtype=np.float32),
            episode_id=np.asarray((0, 0, 0, 1, 1, 1), dtype=np.int64),
        )
        (dataset_dir / "index.csv").write_text(
            "shard,samples\nmarine_controller_0000.npz,6\n", encoding="utf-8"
        )
        dataset = MarineShardDataset(dataset_dir)
        item = dataset[0]
        self.assertEqual(tuple(item["maps"].shape), (5, 9, 9))
        self.assertEqual(tuple(item["global_state"].shape), (len(GLOBAL_FEATURE_NAMES),))
        train, validation = grouped_split(dataset, 0.5, 7)
        train_episodes = {int(dataset.numpy_item(index)["episode_id"]) for index in train}
        validation_episodes = {int(dataset.numpy_item(index)["episode_id"]) for index in validation}
        self.assertTrue(train_episodes.isdisjoint(validation_episodes))

    def test_model_output_and_checkpoint_acceptance_fallback(self) -> None:
        maps = map_tensor(
            np.zeros((9, 9)), np.zeros((9, 9)), np.zeros((9, 9)), (4, 4), (1, 7)
        )
        features = np.zeros(len(GLOBAL_FEATURE_NAMES), dtype=np.float32)
        state = BoatState(40.0, 18.5, 0.0, 1.0, rudder=3.0, flap=-2.0)
        accepted = MarineController(self._checkpoint(10.0), device="cpu").decide(
            maps, features, "gamma", state, max_rudder_slew=5.0, max_flap_slew=3.0
        )
        self.assertTrue(accepted.accepted)
        self.assertEqual(accepted.action.rudder, 0.0)
        rejected = MarineController(self._checkpoint(-10.0), device="cpu").decide(
            maps, features, "diffusion", state, max_rudder_slew=5.0, max_flap_slew=3.0
        )
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.reason, "low_confidence")
        self.assertEqual(rejected.action.rudder, state.rudder)

    def test_cli_validation_preserves_direction_mode(self) -> None:
        base = dict(
            combined_controller_checkpoint=Path("controller.pt"),
            combined_controller_min_confidence=0.5,
            combined_controller_max_cost_increase=0.1,
            combined_controller_max_rudder_error=3.0,
            combined_controller_max_flap_error=3.0,
            dynamics="fossen",
            planner="gamma",
        )
        validate_combined_controller_args(argparse.Namespace(**base))
        with self.assertRaises(SystemExit):
            validate_combined_controller_args(
                argparse.Namespace(**{**base, "dynamics": "legacy_kinematic"})
            )
        with self.assertRaises(SystemExit):
            validate_combined_controller_args(
                argparse.Namespace(**{**base, "planner": "transformer_diffusion"})
            )

    def test_rollout_cost_gate_limits_degradation(self) -> None:
        self.assertTrue(combined_controller_cost_is_safe(108.0, 100.0, 0.1))
        self.assertFalse(combined_controller_cost_is_safe(111.0, 100.0, 0.1))
        self.assertTrue(combined_controller_cost_is_safe(-9.5, -10.0, 0.1))
        self.assertFalse(combined_controller_cost_is_safe(-8.0, -10.0, 0.1))

    def test_action_gate_requires_expert_agreement(self) -> None:
        expert = BoatAction(rudder=5.0, flap=-5.0)
        self.assertTrue(
            combined_controller_action_is_safe(
                BoatAction(rudder=7.0, flap=-3.0), expert, 3.0, 3.0
            )
        )
        self.assertFalse(
            combined_controller_action_is_safe(
                BoatAction(rudder=9.0, flap=-3.0), expert, 3.0, 3.0
            )
        )


if __name__ == "__main__":
    unittest.main()
