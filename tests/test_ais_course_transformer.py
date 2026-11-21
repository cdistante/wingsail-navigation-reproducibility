"""Tests for temporal AIS course prediction data and model utilities."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
import torch

from world.ais_course_transformer import (
    AISCourseTransformer,
    AISCoursePredictor,
    AISCourseReplayForecaster,
    AISSequenceDataset,
    angular_errors_degrees,
    build_ais_sequence_dataset,
    direction_loss,
)


class AISCourseTransformerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.csv_path = Path(self.temporary_directory.name) / "ais.csv"
        start = pd.Timestamp("2026-09-30T00:00:00Z")
        rows = []
        for vessel_index, mmsi in enumerate((111000111, 222000222, 333000333)):
            for minute in range(31):
                rows.append(
                    {
                        "receiver_timestamp": start + pd.Timedelta(minutes=minute),
                        "mmsi": mmsi,
                        "latitude": 40.0 + vessel_index * 0.1,
                        "longitude": 18.0 + vessel_index * 0.1 + minute * 0.001,
                        "speed": 3.0,
                        "course": 90.0,
                    }
                )
        pd.DataFrame(rows).to_csv(self.csv_path, index=False)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_dataset_builds_eastbound_multihorizon_targets_without_vessel_leakage(self) -> None:
        dataset = build_ais_sequence_dataset(
            self.csv_path,
            history_minutes=5,
            horizons_minutes=(2, 4),
            sample_interval_seconds=60,
            max_interpolation_gap_seconds=120,
            minimum_future_displacement_metres=10.0,
            seed=7,
        )

        self.assertEqual(dataset.features.shape[1:], (6, 5))
        self.assertEqual(dataset.targets.shape[1:], (2, 2))
        np.testing.assert_allclose(dataset.targets[..., 0], 1.0, atol=1e-5)
        np.testing.assert_allclose(dataset.targets[..., 1], 0.0, atol=1e-5)
        vessel_splits = {
            int(mmsi): set(dataset.split[dataset.mmsi == mmsi])
            for mmsi in np.unique(dataset.mmsi)
        }
        self.assertTrue(all(len(splits) == 1 for splits in vessel_splits.values()))
        self.assertEqual(set(dataset.split), {"train", "val", "test"})

    def test_dataset_archive_round_trip(self) -> None:
        dataset = build_ais_sequence_dataset(
            self.csv_path,
            history_minutes=5,
            horizons_minutes=(2, 4),
            minimum_future_displacement_metres=10.0,
        )
        archive = Path(self.temporary_directory.name) / "dataset.npz"

        dataset.save(archive)
        restored = AISSequenceDataset.load(archive)

        np.testing.assert_array_equal(restored.features, dataset.features)
        np.testing.assert_array_equal(restored.targets, dataset.targets)
        np.testing.assert_array_equal(restored.split, dataset.split)
        self.assertEqual(restored.horizons_minutes, (2, 4))

    def test_transformer_returns_one_vector_per_horizon(self) -> None:
        model = AISCourseTransformer(
            horizons=3,
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            maximum_sequence_length=8,
        )
        features = torch.zeros((4, 8, 5))

        prediction = model(features)

        self.assertEqual(tuple(prediction.shape), (4, 3, 2))

    def test_direction_loss_and_angles_are_circular(self) -> None:
        target = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        prediction = torch.tensor([[[2.0, 0.0], [0.0, 3.0]]])

        loss, _ = direction_loss(prediction, target)
        angles = angular_errors_degrees(prediction, target)

        self.assertAlmostEqual(float(loss), 0.0, places=6)
        np.testing.assert_allclose(angles.numpy(), 0.0, atol=1e-5)

    def test_checkpoint_predictor_returns_true_courses(self) -> None:
        model = AISCourseTransformer(
            horizons=2,
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            maximum_sequence_length=6,
        )
        checkpoint = Path(self.temporary_directory.name) / "model.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_config": {
                    "input_features": 5,
                    "horizons": 2,
                    "d_model": 16,
                    "nhead": 4,
                    "num_layers": 1,
                    "dim_feedforward": 32,
                    "dropout": 0.1,
                    "maximum_sequence_length": 6,
                },
                "horizons_minutes": (2, 4),
                "history_minutes": 5,
                "sample_interval_seconds": 60,
                "feature_mean": np.zeros((1, 1, 5), dtype=np.float32),
                "feature_standard_deviation": np.ones((1, 1, 5), dtype=np.float32),
            },
            checkpoint,
        )
        predictor = AISCoursePredictor(checkpoint, device="cpu")

        directions, courses = predictor.predict(np.zeros((6, 5), dtype=np.float32))

        self.assertEqual(directions.shape, (2, 2))
        self.assertEqual(courses.shape, (2,))
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-5)

    def test_replay_forecaster_batches_eligible_targets(self) -> None:
        model = AISCourseTransformer(
            horizons=2,
            d_model=16,
            nhead=4,
            num_layers=1,
            dim_feedforward=32,
            maximum_sequence_length=6,
        )
        checkpoint = Path(self.temporary_directory.name) / "batch-model.pt"
        torch.save(
            {
                "model_state": model.state_dict(),
                "model_config": {
                    "input_features": 5,
                    "horizons": 2,
                    "d_model": 16,
                    "nhead": 4,
                    "num_layers": 1,
                    "dim_feedforward": 32,
                    "dropout": 0.1,
                    "maximum_sequence_length": 6,
                },
                "horizons_minutes": (2, 4),
                "history_minutes": 5,
                "sample_interval_seconds": 60,
                "feature_mean": np.zeros((1, 1, 5), dtype=np.float32),
                "feature_standard_deviation": np.ones((1, 1, 5), dtype=np.float32),
            },
            checkpoint,
        )
        forecaster = AISCourseReplayForecaster(
            checkpoint,
            self.csv_path,
            device="cpu",
            max_interpolation_gap_seconds=120,
        )

        forecasts, stats = forecaster.predict_targets(
            [111000111, 999999999],
            "2026-09-30T00:10:00Z",
        )

        self.assertEqual(set(forecasts), {111000111})
        self.assertEqual(forecasts[111000111].directions.shape, (2, 2))
        self.assertEqual(stats["requested"], 2)
        self.assertEqual(stats["forecasted"], 1)
        self.assertEqual(stats["missing"], 1)


if __name__ == "__main__":
    unittest.main()
