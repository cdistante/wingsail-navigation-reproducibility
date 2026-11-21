"""Exercise the tack feature contract and portable checkpoint inference."""

from dataclasses import replace
import json
from math import cos, radians, sin
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from world._geo import offset_latitude_longitude
from world.boat import BoatState
from world.currents import Current
from world.learned_tacks import FEATURE_NAMES, LearnedTackSelector, tack_candidate_features
from world.scenario import Waypoint
from world.waves import Wave
from world.weather import Wind
from world.world import WorldObservation


class LearnedTackTests(unittest.TestCase):
    def setUp(self):
        boat = BoatState(40.225, 18.75, heading=75.0, speed=0.7, rudder=5.0, flap=-10.0)
        self.observation = WorldObservation(
            time=np.datetime64("2026-06-10T03:00:00"),
            boat=boat,
            wind=Wind(0.0, -8.0, 8.0, 0.0),
            wave=Wave(1.25, 30.0, 6.0),
            current=Current(0.4, -0.2, 0.45, 116.6, 0.0),
            seabed=None,
            coast=None,
            chart=None,
            ais_targets=(),
        )
        self.goal = Waypoint(*offset_latitude_longitude(boat.latitude, boat.longitude, 500.0, 0.0))

    @staticmethod
    def payload():
        width = len(FEATURE_NAMES)
        means, scales = np.zeros(width), np.ones(width)
        means[0], scales[0] = 4.0, 2.0
        weights = np.zeros((2, width))
        weights[:, 0] = [2.0, -1.0]
        return {
            "version": 1,
            "feature_names": list(FEATURE_NAMES),
            "means": means.tolist(),
            "scales": scales.tolist(),
            "target_mean": -4.0,
            "target_scale": 2.0,
            "layers": [
                {"weights": weights.tolist(), "bias": [-1.0, 0.0]},
                {"weights": [[2.0, -3.0]], "bias": [0.5]},
            ],
        }

    def load(self, data):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            return LearnedTackSelector.load(path)

    def test_features_preserve_wind_current_and_candidate_geometry(self):
        features = dict(zip(FEATURE_NAMES, tack_candidate_features(self.observation, self.goal, 60.0, 35.0)))
        self.assertEqual(features["wind_speed_mps"], 8.0)
        self.assertAlmostEqual(features["wind_from_along_goal"], 0.0)
        self.assertAlmostEqual(features["wind_from_across_goal"], -1.0)
        self.assertAlmostEqual(features["current_along_goal_mps"], 0.4)
        self.assertAlmostEqual(features["current_across_goal_mps"], 0.2)
        self.assertEqual(features["wave_height_m"], 1.25)
        self.assertEqual(features["boat_speed_mps"], 0.7)
        self.assertEqual(features["boat_surge_mps"], 0.7)
        self.assertAlmostEqual(features["course_along_goal"], cos(radians(30.0)))
        self.assertAlmostEqual(features["course_across_goal"], -0.5)
        self.assertAlmostEqual(features["bow_along_wind_from"], cos(radians(35.0)))
        self.assertAlmostEqual(features["bow_across_boat"], sin(radians(-40.0)))
        self.assertAlmostEqual(features["goal_distance_m"], 500.0)

    def test_json_inference_standardizes_relu_and_restores_progress_units(self):
        selector = self.load(self.payload())
        self.assertEqual(selector.predict_progress(self.observation, self.goal, 60.0, 35.0), 9.0)

    def test_no_score_without_input_coverage(self):
        selector = self.load({**self.payload(), "domain_bounds": {"wind_speed_mps": [4.0, 10.0]}})
        self.assertIsNone(selector.predict_progress(replace(self.observation, wave=None), self.goal, 60.0, 35.0))
        outside = replace(self.observation, wind=Wind(0.0, -11.0, 11.0, 0.0))
        self.assertIsNone(selector.predict_progress(outside, self.goal, 60.0, 35.0))
        self.assertIsNone(selector.predict_progress(self.observation, self.goal, float("nan"), 35.0))

    def test_malformed_model_is_rejected_at_load(self):
        for change in (
            {"feature_names": list(reversed(FEATURE_NAMES))},
            {"target_scale": 0.0},
            {"layers": [{"weights": [[0.0]], "bias": [0.0]}]},
            {"domain_bounds": {"wind_speed_mps": [10.0, 5.0]}},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load({**self.payload(), **change})


if __name__ == "__main__":
    unittest.main()
