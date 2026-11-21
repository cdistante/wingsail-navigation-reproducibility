"""Independent native transcription, causality, fork, and stop checks."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter
import unittest

import numpy as np

from native_branches import NativeEngine, NativeConfig, NativeSnapshot, FEATURE_NAMES
from world._geo import normalise_time, offset_latitude_longitude, haversine_metres
from world.ais import AISTarget
from world.boat import BoatState, BoatAction
from world.currents import Current
from world.scenario import RouteWaypoint
from world.weather import Wind
from world.waves import Wave
from world.world import WorldObservation


WHEN = normalise_time("2026-01-12T06:00:00Z")


class Environment:
    def __init__(self, reports=(), truth=()):
        self.reports, self.truth = reports, truth
        self.observations, self.truth_queries = [], []

    def observe(self, state, when):
        self.observations.append(normalise_time(when))
        return WorldObservation(normalise_time(when), state, Wind(0., -8., 8., 0.),
            Wave(.8, 0., 4.), Current(.1, .05, np.hypot(.1, .05), 63.4349488229, 0.),
            None, None, None, self.reports)

    def truth_targets(self, when):
        self.truth_queries.append(normalise_time(when))
        return self.truth


def state():
    return BoatState(40., 18., 90., .8, 3., -2., 0., .8, .03, .01)


def goal(distance=300.):
    latitude, longitude = offset_latitude_longitude(40., 18., distance, 0.)
    return RouteWaypoint(latitude, longitude, "test")


RECEIPT = {}


class NativeChecks(unittest.TestCase):
    def test_transcription_matches_existing_python_plant(self):
        engine = NativeEngine(Environment())
        rng = np.random.default_rng(7712)
        maxima = {key: 0. for key in asdict(state())}
        for _ in range(24):
            before = replace(state(), heading=float(rng.uniform(0, 360)),
                surge_speed=float(rng.uniform(-.4, 1.8)), sway_speed=float(rng.uniform(-.2, .2)),
                yaw_rate=float(rng.uniform(-.04, .04)), flap=float(rng.uniform(-15, 15)))
            action = BoatAction(float(rng.uniform(-25, 25)), float(rng.uniform(-15, 15)))
            observation = engine.environment.observe(before, WHEN)
            actual, _, _ = engine.integrate(before, action, observation)
            expected = engine.dynamics.step(before, action, observation, 5.)
            for key in maxima:
                maxima[key] = max(maxima[key], abs(getattr(actual, key) - getattr(expected, key)))
            self.assertLess(haversine_metres(actual.latitude, actual.longitude, expected.latitude, expected.longitude), 1e-5)
            for key in ("heading", "speed", "surge_speed", "sway_speed", "yaw_rate", "rudder", "flap", "heel"):
                self.assertAlmostEqual(getattr(actual, key), getattr(expected, key), places=9)
        RECEIPT["python_native_parity"] = {"cases": 24, "maximum_absolute_error": maxima}

    def test_truth_cannot_change_features_or_report_admission(self):
        close = AISTarget(10, 40., 18., 0., 0., 0., length_metres=30., beam_metres=8.)
        safe_env, unsafe_truth_env = Environment(), Environment(truth=(close,))
        a, b = NativeEngine(safe_env), NativeEngine(unsafe_truth_env)
        da, db = a.decision(state(), goal(), WHEN), b.decision(state(), goal(), WHEN)
        np.testing.assert_array_equal(da.features, db.features)
        np.testing.assert_array_equal(da.allowed, db.allowed)
        self.assertEqual(da.physics_index, db.physics_index)
        self.assertFalse(unsafe_truth_env.truth_queries)
        self.assertTrue(all(instant == WHEN for instant in unsafe_truth_env.observations))
        ra, rb = a.run(state(), [goal()], WHEN, cap_seconds=10.), b.run(state(), [goal()], WHEN, cap_seconds=10.)
        self.assertEqual(ra.trace, rb.trace)
        self.assertLess(rb.truth_minimum_m, 0.)
        self.assertEqual(ra.report_minimum_m, rb.report_minimum_m)
        RECEIPT["causality"] = {"features_identical_under_different_truth": True,
            "admission_and_motion_identical": True, "negative_truth_detected_offline": True,
            "truth_queries_during_decision": 0, "feature_width": len(FEATURE_NAMES)}

    def test_unsafe_initial_overlap_stops_before_applying(self):
        close = AISTarget(10, 40., 18., 0., 0., 0., length_metres=30., beam_metres=8.)
        engine = NativeEngine(Environment(reports=(close,), truth=(close,)))
        result = engine.run(state(), [goal()], WHEN, cap_seconds=10.)
        self.assertFalse(result.success)
        self.assertEqual(result.status, "no_safe_native_action")
        self.assertEqual(result.final_state, state())
        self.assertEqual(len(result.controls), 0)
        self.assertEqual(len(result.control_attempts), 1)
        self.assertFalse(result.control_attempts[0]["applied"])
        RECEIPT["strict_stop"] = {"terminal_attempt_preserved": True, "unsafe_actions_applied": 0}

    def test_exact_matched_branch_and_causal_snapshot(self):
        engine = NativeEngine(Environment(), config=NativeConfig(forecast_horizon_s=1200))
        d = engine.decision(state(), goal(), WHEN)
        snap = NativeSnapshot(state(), (goal(),), WHEN, 0., 1200.)
        begun = perf_counter()
        baseline = engine.run(state(), [goal()], WHEN, cap_seconds=1200.)
        branches = engine.branches(snap, [d.physics_index, d.physics_index], decision=d, trace=True)
        self.assertEqual(baseline.trace, branches[0].outcome.trace)
        self.assertEqual(branches[0].outcome.trace, branches[1].outcome.trace)
        self.assertTrue(baseline.success)
        self.assertTrue(baseline.first_commit_completed)
        self.assertLessEqual(baseline.captures[0]["distance_m"], 50. + 1e-7)
        self.assertLessEqual(baseline.elapsed_s - baseline.captures[0]["elapsed_s"], .05 + 1e-8)
        RECEIPT["matched_branch"] = {"captures": baseline.captures, "identical_trace": True,
            "three_episode_wall_s": perf_counter() - begun, "summary": baseline.summary()}


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(NativeChecks)
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    RECEIPT["passed"] = outcome.wasSuccessful()
    RECEIPT["tests_run"] = outcome.testsRun
    RECEIPT["feature_names"] = list(FEATURE_NAMES)
    output = Path(__file__).resolve().parents[1] / "native_library_checks.json"
    output.write_text(json.dumps(RECEIPT, indent=2, allow_nan=True) + "\n")
    raise SystemExit(0 if outcome.wasSuccessful() else 1)
