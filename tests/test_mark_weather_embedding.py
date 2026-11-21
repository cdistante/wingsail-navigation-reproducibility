"""Verify frozen weather features preserve the old head and causal history."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from world.ais import AISTarget
from world.boat import BoatState
from world.currents import Current
from world.mark_weather_embedding import FrozenMarkWeatherEncoder
from world.temporal_recovery import RecoveryHoldModel, TemporalRecoveryGuidance
from world.waves import Wave
from world.weather import Wind
from world.world import WorldObservation


class MarkWeatherEmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "hold.pt"
        with torch.random.fork_rng():
            torch.manual_seed(17)
            model = RecoveryHoldModel()
        torch.save({
            "weights": model.state_dict(),
            "scaling": {
                "history_mean": [.1, -.2, .3, -.4, .5],
                "history_scale": [2., 3., .4, .5, .6],
                "context_mean": [.4], "context_scale": [.7],
            },
            "temperature": .85, "confidence_threshold": .6,
        }, self.path)
        self.observation = WorldObservation(
            time=np.datetime64("2026-10-01T00:00:05", "ns"),
            boat=BoatState(40.225, 18.75, 75., .7),
            wind=Wind(1., -4., np.hypot(1., 4.), 345.96),
            current=Current(.2, -.1, np.hypot(.2, .1), 116.56, 0.),
            wave=Wave(.7, 45., 6.), seabed=None, coast=None, chart=None, ais_targets=(),
        )

    def test_export_and_original_head_match_original_classifier(self):
        encoder = FrozenMarkWeatherEncoder(self.path)
        original = TemporalRecoveryGuidance(self.path)
        for minute in (0, 5, 10, 25, 30, 35, 40, 45, 50, 55, 60, 65):
            obs = replace(self.observation, time=self.observation.time + np.timedelta64(minute, "m"),
                          wind=replace(self.observation.wind, u=1. + .1 * minute))
            for was_holding in (False, True):
                embedding = encoder.encode(obs, np.array([120., 300.]))
                self.assertEqual(embedding.values.shape, (48,))
                self.assertFalse(embedding.values.flags.writeable)
                decision = original.decide(obs, np.array([120., 300.]), was_holding=was_holding,
                                           classical_hold=False, gate_vmg=.1, hold_available=True)
                self.assertEqual(encoder.original_hold_probability(embedding, was_holding=was_holding),
                                 decision.p_hold)
        self.assertTrue(all(not parameter.requires_grad for parameter in encoder._guidance.model.parameters()))

    def test_first_sample_gap_fill_and_backwards_clock(self):
        encoder = FrozenMarkWeatherEncoder(self.path)
        first = encoder.encode(self.observation, np.array([0., 1.]))
        changed = replace(self.observation, time=self.observation.time + np.timedelta64(20, "s"),
                          wind=replace(self.observation.wind, u=20.))
        second = encoder.encode(changed, np.array([0., 1.]))
        np.testing.assert_array_equal(first.values, second.values)
        self.assertEqual(len(set(first.source_windows)), 1)
        later = replace(changed, time=self.observation.time + np.timedelta64(900, "s"))
        third = encoder.encode(later, np.array([0., 1.]))
        self.assertEqual(third.source_windows[-4:], (first.source_windows[0],) * 3 + (third.history_windows[-1],))
        self.assertLessEqual(max(third.source_windows), third.history_windows[-1])
        before = encoder.observed_weather
        with self.assertRaisesRegex(ValueError, "chronological"):
            encoder.encode(self.observation, np.array([0., 1.]))
        self.assertEqual(encoder.observed_weather, before)

    def test_weather_vector_excludes_range_vessel_and_ais(self):
        encoder = FrozenMarkWeatherEncoder(self.path)
        first = encoder.encode(self.observation, np.array([0., 100.]))
        altered = replace(self.observation, boat=replace(self.observation.boat, heading=270., speed=3.,
                                                        rudder=25., flap=-15., yaw_rate=.3),
                          ais_targets=(AISTarget(123, 40.226, 18.751, 5., 180., 180.),))
        second = encoder.encode(altered, np.array([0., 1000.]))
        np.testing.assert_array_equal(first.values, second.values)

    def test_missing_invalid_inputs_have_no_embedding(self):
        encoder = FrozenMarkWeatherEncoder(self.path)
        self.assertIsNone(encoder.encode(replace(self.observation, wind=None), np.array([0., 1.])))
        self.assertIsNone(encoder.encode(self.observation, np.array([0., 0.])))
        self.assertIsNone(encoder.encode(self.observation, np.array([np.nan, 0.])))
        self.assertIsNone(encoder.encode(self.observation, np.array([[1., 0.]])))
        with self.assertRaisesRegex(ValueError, "valid observation time"):
            encoder.encode(replace(self.observation, time=np.datetime64("NaT")), np.array([0., 1.]))
        later = replace(self.observation, time=self.observation.time + np.timedelta64(300, "s"), wind=None)
        self.assertIsNone(encoder.encode(later, np.array([0., 1.])))


if __name__ == "__main__":
    unittest.main()
