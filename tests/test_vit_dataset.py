"""Checks for environmental Vision Transformer dataset features."""

from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from world import BoatState, Current, Wave, Waypoint, Wind, WorldObservation
from world.vit_dataset import (
    AISTrafficSource,
    EnvironmentalViTDatasetBuilder,
    GLOBAL_FEATURE_NAMES,
    ViTDataset,
    direction_vector,
    global_environment_features,
    heading_from_direction,
)


class ViTDatasetTests(unittest.TestCase):
    def test_direction_round_trip_uses_east_north_navigation_axes(self) -> None:
        np.testing.assert_allclose(direction_vector(90.0), [1.0, 0.0], atol=1.0e-7)
        self.assertAlmostEqual(heading_from_direction(direction_vector(225.0)), 225.0)

    def test_global_features_include_real_environment_and_goal_geometry(self) -> None:
        state = BoatState(39.9, 18.4, heading=90.0, speed=1.2, rudder=3.0, flap=-5.0)
        observation = WorldObservation(
            time=np.datetime64("2026-07-29T07:00:00"),
            boat=state,
            wind=Wind(u=2.0, v=-3.0, speed=np.hypot(2.0, -3.0), direction=326.3),
            wave=Wave(height=1.2, direction=170.0, peak_period=5.5),
            current=Current(u=-0.1, v=0.2, speed=np.hypot(-0.1, 0.2), direction=333.4, depth=0.0),
            seabed=None,
            coast=None,
            chart=None,
            ais_targets=(),
        )
        features = global_environment_features(state, observation, Waypoint(39.9, 18.5))

        self.assertEqual(features.shape, (len(GLOBAL_FEATURE_NAMES),))
        self.assertAlmostEqual(features[0], 2.0)
        self.assertAlmostEqual(features[5], -0.1)
        self.assertAlmostEqual(features[7], 1.2)
        self.assertGreater(features[-1], 0.0)

    def test_ais_schedule_uses_exact_largest_remainder_balance(self) -> None:
        sources = (
            AISTrafficSource("none", "none", weight=3.0),
            AISTrafficSource("crossing", "synthetic", scenario="crossing", weight=2.0),
            AISTrafficSource("recorded", "recorded", path="recorded.csv", weight=6.0),
        )
        schedule = EnvironmentalViTDatasetBuilder._traffic_schedule(
            sources,
            samples=22,
            rng=np.random.default_rng(5),
        )
        self.assertEqual(
            Counter(source.name for source in schedule),
            {"none": 6, "crossing": 4, "recorded": 12},
        )

    def test_archive_persists_traffic_and_partition_provenance(self) -> None:
        dataset = ViTDataset(
            grids=np.zeros((3, 10, 2, 2), dtype=np.float32),
            global_features=np.zeros((3, len(GLOBAL_FEATURE_NAMES)), dtype=np.float32),
            targets=np.tile(direction_vector(90.0), (3, 1)),
            split=np.asarray([0, 1, 2], dtype=np.int8),
            teacher_headings=np.asarray([90.0, 90.0, 90.0], dtype=np.float32),
            traffic_sources=np.asarray(["none", "synthetic", "recorded"]),
            partitions=np.asarray(["train", "validation", "test_route_time"]),
            metadata={"schema_version": 2},
        )
        with TemporaryDirectory() as directory:
            path = dataset.save(Path(directory) / "dataset.npz")
            with np.load(path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(
                    archive["traffic_sources"], ["none", "synthetic", "recorded"]
                )
                np.testing.assert_array_equal(
                    archive["partitions"], ["train", "validation", "test_route_time"]
                )


if __name__ == "__main__":
    unittest.main()
