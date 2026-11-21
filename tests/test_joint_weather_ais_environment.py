"""Causal/report-only and batch-invariant contract for the shared joint study."""

from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from data.results.learned_safe_mark_policy.reproduce.environment import load_scenarios
from data.results.learned_weather_ais_policy.reproduce.environment_joint import (
    REPORT_FEATURE_NAMES, assess_applied, evaluate, observed_weather, prepare, weather_observation,
)


class JointWeatherAISEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1]/"data/results/learned_safe_mark_policy/scenarios.json.gz"
        cls.scenarios = load_scenarios(path)
        cls.scenario = next(case for case in cls.scenarios if case.future is not None)

    def test_batched_and_single_vessels_receive_identical_masks_and_features(self):
        scenario = self.scenario
        targets = np.stack((scenario.target, scenario.target+[50., -30.], scenario.target+[-10000., 12000.]))
        bows = np.array([scenario.bow, scenario.bow+45., scenario.bow-90.])
        for mode in ("stale", "reach"):
            for profile in ("clear", "crossing", "mixed_long_dropout", "mixed_sharp_dropout"):
                context = prepare(scenario, profile, guard_mode=mode)
                for elapsed in (0., 120., 180., 900.):
                    batch = evaluate(context, elapsed, targets, bows)
                    for row in range(len(targets)):
                        single = evaluate(context, elapsed, targets[row], bows[row])
                        self.assertEqual(batch.dt, single.dt)
                        np.testing.assert_array_equal(batch.allowed[row], single.allowed[0])
                        np.testing.assert_array_equal(batch.report_features[row], single.report_features[0])
                        if batch.assessment is not None:
                            np.testing.assert_array_equal(batch.assessment.report_margin_m[row], single.assessment.report_margin_m[0])
        self.assertEqual(len(REPORT_FEATURE_NAMES), 6)

    def test_evaluation_does_not_query_analytical_truth(self):
        scenario = self.scenario
        context = prepare(scenario, "mixed_sharp_dropout")
        with patch.object(context.ais, "truth_positions", side_effect=AssertionError("Future truth queried")):
            result = evaluate(context, 120., np.stack((scenario.target, scenario.target+[10000., 10000.])), scenario.bow)
            self.assertFalse(result.audit["truth_used_for_features_or_mask"])
            self.assertFalse(hasattr(result.assessment, "truth_margin_m"))
            self.assertTrue(np.isfinite(result.report_features).all())
            with self.assertRaisesRegex(AssertionError, "Future truth"):
                assess_applied(context, 120., np.stack((scenario.target, scenario.target+[10000., 10000.])), result, 0, 0)

    def test_stale_guard_and_geographic_anchor_match_declared_protocol(self):
        scenario = self.scenario
        context = prepare(scenario, "mixed_long_dropout", origin_latitude=41.2, origin_longitude=17.4)
        result = evaluate(context, 120., scenario.target, scenario.bow)
        self.assertEqual(result.dt, 5.)
        self.assertEqual(result.audit["maximum_report_age_seconds"], 40.)
        self.assertEqual(result.audit["dropped_reports_at_observation"], 6)
        self.assertEqual(result.audit["required_additional_clearance_m"], .5*.030*45.**2)
        self.assertEqual(context.ais.origin, (41.2, 17.4))
        self.assertEqual(context.ais.reserve_metres, 3.)
        np.testing.assert_array_equal(result.allowed, result.assessment.report_margin_m > .5*.030*45.**2)

    def test_future_weather_cannot_change_observed_history_before_its_time(self):
        scenario = self.scenario
        altered = replace(scenario, future=replace(scenario.future, wind_speed=scenario.future.wind_speed+30.,
                                                   wave_height=scenario.future.wave_height+4.))
        a, b = prepare(scenario, "clear"), prepare(altered, "clear")
        for elapsed in (0., min(120., scenario.change_at/2), scenario.change_at-1.):
            np.testing.assert_array_equal(observed_weather(a, elapsed), observed_weather(b, elapsed))
            self.assertEqual(weather_observation(a, elapsed).wind, weather_observation(b, elapsed).wind)
        launch = observed_weather(a, 0.)
        self.assertEqual(launch.shape, (12, 5))
        np.testing.assert_array_equal(launch, np.repeat(launch[-1:], 12, axis=0))
        self.assertFalse(np.array_equal(observed_weather(a, scenario.change_at+300.),
                                       observed_weather(b, scenario.change_at+300.)))

    def test_selected_partial_sweep_and_safe_contingency_contract(self):
        scenario = self.scenario
        context = prepare(scenario, "head_on")
        result = evaluate(context, 0., scenario.target, scenario.bow)
        actual = assess_applied(context, 0., scenario.target, result, 0, 0, 2.)
        expected = context.ais.assess_applied(0., [0., 0.], result.batch, result.dt, 0, 0, 2.)
        np.testing.assert_array_equal(actual, expected)
        emergency = prepare(scenario, "head_on", contingency_short_step=True)
        ship = emergency.ais._snapshot(0.).prediction.positions_metres[0]
        blocked = evaluate(emergency, 0., scenario.target-ship, scenario.bow)
        self.assertEqual(blocked.dt, 5.)
        self.assertTrue(blocked.audit["contingency_short_step_used"])
        self.assertFalse(blocked.allowed.any())


if __name__ == "__main__":
    unittest.main()
