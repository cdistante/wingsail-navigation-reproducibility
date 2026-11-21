"""Tests for AIS course evaluation metrics and plots."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from world.ais_course_transformer import AISSequenceDataset
from world.evaluate_ais_course_transformer import (
    angular_errors,
    horizon_metrics,
    save_evaluation_plots,
)


class AISCouseEvaluationTests(unittest.TestCase):
    def test_angular_metrics_distinguish_model_and_baseline(self) -> None:
        target = np.asarray([[[1.0, 0.0], [0.0, 1.0]]])
        prediction = target.copy()
        baseline = np.asarray([[[0.0, 1.0], [1.0, 0.0]]])

        errors = angular_errors(prediction, target)
        baseline_errors = angular_errors(baseline, target)
        rows = horizon_metrics(errors, baseline_errors, (5, 10))

        np.testing.assert_allclose(errors, 0.0)
        self.assertEqual(rows[0]["mean_error_degrees"], 0.0)
        self.assertAlmostEqual(rows[0]["mean_improvement_degrees"], 90.0)

    def test_plot_writer_creates_expected_artifacts(self) -> None:
        features = np.zeros((2, 3, 5), dtype=np.float32)
        features[:, :, 0] = np.asarray([[-0.2, -0.1, 0.0], [0.0, 0.1, 0.0]])
        targets = np.asarray(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.0, 1.0], [-1.0, 0.0]],
            ],
            dtype=np.float32,
        )
        predictions = targets.copy()
        errors = angular_errors(predictions, targets)
        baseline_errors = np.full_like(errors, 20.0)
        dataset = AISSequenceDataset(
            features=features,
            targets=targets,
            mmsi=np.asarray([1, 2]),
            timestamps_ns=np.asarray([1, 2]),
            split=np.asarray(["test", "test"]),
            horizons_minutes=(5, 10),
            history_minutes=2,
            sample_interval_seconds=60,
        )
        rows = horizon_metrics(errors, baseline_errors, (5, 10))
        with TemporaryDirectory() as directory:
            output = Path(directory)

            save_evaluation_plots(
                output,
                dataset,
                dataset.split == "test",
                features,
                targets,
                predictions,
                errors,
                baseline_errors,
                rows,
                history_path=None,
                example_count=2,
            )

            self.assertTrue((output / "horizon_error_comparison.png").exists())
            self.assertTrue((output / "error_cdf.png").exists())
            self.assertTrue((output / "prediction_examples.png").exists())
            self.assertTrue((output / "predictions.npz").exists())


if __name__ == "__main__":
    unittest.main()
