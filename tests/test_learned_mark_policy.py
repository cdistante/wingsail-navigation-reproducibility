import unittest

import numpy as np

from world.boat import BoatState
from world.currents import Current
from world.drift_guidance import DriftMode
from world.learned_mark_policy import FEATURE_NAMES, MAX_SECONDS, LearnedMarkPolicy, PolarSnapshot, candidate_batch
from world.weather import Wind
from world.waves import Wave
from world.world import WorldObservation


def mode(velocity, bow):
    return DriftMode(np.asarray(velocity, dtype=float), bow, 0.0, 0.0)


class LearnedMarkPolicyTests(unittest.TestCase):
    def test_turn_consumes_sailing_time_before_circle_capture(self):
        alternative = mode((0.0, 1.0), 0.0)
        polar = PolarSnapshot.from_modes((alternative,), (0.0, 0.0), 230.0)
        batch = candidate_batch((0.0, 90.0), 30.0, 50.0, polar, None, 5.0, .5,
                                300.0, 0.0, MAX_SECONDS)
        self.assertEqual(batch.features.shape, (1, 1, len(FEATURE_NAMES)))
        self.assertAlmostEqual(batch.captured_after_s[0, 0], 70.0)

    def test_rank_keeps_original_actions_and_abstains_for_unsupported_context(self):
        modes = (mode((1.0, 0.0), 90.0), mode((0.0, 1.0), 0.0))
        policy = LearnedMarkPolicy(np.zeros(len(FEATURE_NAMES)), np.ones(len(FEATURE_NAMES)),
                                   ((np.zeros((1, len(FEATURE_NAMES))), np.ones(1)),), {
            "range_m": (55.0, 500.0), "radius_m": (50.0, 50.0), "wind_mps": (3.0, 9.0),
            "current_mps": (0.0, .6), "wave_m": (0.0, 1.0), "decision_s": (60.0, 300.0),
            "forecast_s": (0.0, 3600.0), "ground_mps": (0.0, 2.0),
        })
        observation = WorldObservation(np.datetime64("2000-01-01"), BoatState(0.0, 0.0, 0.0, 0.0),
                                       Wind(0.0, 5.0, 5.0, 180.0), Wave(.5, 180.0, 5.0),
                                       Current(0.0, 0.0, 0.0, 0.0, 0.0), None, None, None, ())
        rows = policy.rank(observation, (0.0, 90.0), 50.0, modes, decision_seconds=300.0)
        self.assertEqual([row.candidate_index for row in rows], [1, 0])
        self.assertEqual(rows[0].cost_seconds, 40.0)
        self.assertIsNone(policy.rank(observation, (0.0, 600.0), 50.0, modes, decision_seconds=300.0))
        self.assertIsNone(policy.rank(observation, (0.0, 90.0), 50.0, modes, decision_seconds=300.0,
                                      future_modes=modes, forecast_after_seconds=120.0))


if __name__ == "__main__":
    unittest.main()
