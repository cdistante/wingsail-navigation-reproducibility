"""Verify the scheduled-tack arrival feature and checkpoint contracts."""

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from world._geo import offset_latitude_longitude
from world.boat import BoatState
from world.currents import Current
from world.learned_tack_arrival import FEATURE_NAMES, LearnedTackArrivalSelector, tack_arrival_features
from world.learned_tacks import FEATURE_NAMES as FIRST_LEG_NAMES
from world.learned_tacks import tack_candidate_features
from world.scenario import Waypoint
from world.waves import Wave
from world.weather import Wind
from world.world import WorldObservation


class LearnedTackArrivalTests(unittest.TestCase):
    def setUp(self):
        boat = BoatState(40.225, 18.75, heading=75.0, speed=0.7)
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
        self.schedule = (55.0, 45.0, 315.0, 300.0, 240.0, 50.0, 2880.0)

    @staticmethod
    def payload():
        width = len(FEATURE_NAMES)
        means, scales = np.zeros(width), np.ones(width)
        means[0], scales[0] = 4.0, 2.0
        weights = np.zeros((2, width))
        weights[:, 0] = [2.0, -1.0]
        return {
            "version": 2,
            "feature_names": list(FEATURE_NAMES),
            "means": means.tolist(),
            "scales": scales.tolist(),
            "target_mean": 800.0,
            "target_scale": 60.0,
            "layers": [
                {"weights": weights.tolist(), "bias": [-1.0, 0.0]},
                {"weights": [[1.0 / 3.0, -1.0], [2.0, -3.0]], "bias": [0.0, 0.5]},
            ],
        }

    def load(self, data):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "arrival-model.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            return LearnedTackArrivalSelector.load(path)

    def test_features_include_distinct_continuation_and_first_leg_duration(self):
        row = tack_arrival_features(self.observation, self.goal, *self.schedule)
        first = tack_candidate_features(self.observation, self.goal, 55.0, 45.0)
        continuation = dict(zip(
            FIRST_LEG_NAMES, tack_candidate_features(self.observation, self.goal, 315.0, 300.0),
        ))
        np.testing.assert_array_equal(row[:len(first)], first)
        values = dict(zip(FEATURE_NAMES, row))
        self.assertEqual(values["has_transition"], 1.0)
        self.assertEqual(values["first_leg_seconds"], 240.0)
        self.assertEqual(values["arrival_radius_m"], 50.0)
        self.assertEqual(values["horizon_seconds"], 2880.0)
        for name, value in values.items():
            if name.startswith("next_"):
                self.assertEqual(value, continuation[name.removeprefix("next_")])

    def test_single_leg_has_no_continuation_and_invalid_switches_fail(self):
        for switch in (None, 0.0):
            row = tack_arrival_features(self.observation, self.goal, 55.0, 45.0, None, None, switch, 50.0, 2880.0)
            values = dict(zip(FEATURE_NAMES, row))
            self.assertEqual(values["has_transition"], 0.0)
            self.assertEqual(values["first_leg_seconds"], 0.0)
            self.assertTrue(all(value == 0.0 for name, value in values.items() if name.startswith("next_")))
        for next_course, next_bow, switch in ((55.0, None, 240.0), (55.0, 45.0, 0.0),
                                               (55.0, 45.0, 2880.0), (None, None, 240.0)):
            with self.subTest(next_course=next_course, next_bow=next_bow, switch=switch), self.assertRaises(ValueError):
                tack_arrival_features(
                    self.observation, self.goal, 55.0, 45.0, next_course, next_bow, switch, 50.0, 2880.0,
                )

    def test_json_inference_decodes_capture_probability_and_conditional_seconds(self):
        selector = self.load(self.payload())
        prediction = selector.predict_arrival(self.observation, self.goal, *self.schedule)
        self.assertIsNotNone(prediction)
        self.assertAlmostEqual(prediction.capture_probability, 1.0 / (1.0 + np.exp(-1.0)))
        self.assertAlmostEqual(prediction.arrival_seconds, 1190.0)

    def test_missing_inputs_and_invalid_conditional_seconds_give_no_prediction(self):
        selector = self.load({
            **self.payload(),
            "domain_bounds": {"wind_speed_mps": [4.0, 9.0], "horizon_seconds": [2880.0, 2880.0]},
        })
        self.assertIsNone(selector.predict_arrival(replace(self.observation, wave=None), self.goal, *self.schedule))
        outside = replace(self.observation, wind=Wind(0.0, -11.0, 11.0, 0.0))
        self.assertIsNone(selector.predict_arrival(outside, self.goal, *self.schedule))
        self.assertIsNone(selector.predict_arrival(self.observation, self.goal, *self.schedule[:-1], 2800.0))
        impossible = self.load({**self.payload(), "target_mean": 3000.0})
        self.assertIsNone(impossible.predict_arrival(self.observation, self.goal, *self.schedule))

    def test_rejects_progress_model_version_and_single_output(self):
        with self.assertRaises(ValueError):
            self.load({**self.payload(), "version": 1})
        payload = self.payload()
        payload["layers"][-1] = {"weights": [[1.0, 1.0]], "bias": [0.0]}
        with self.assertRaises(ValueError):
            self.load(payload)


if __name__ == "__main__":
    unittest.main()
