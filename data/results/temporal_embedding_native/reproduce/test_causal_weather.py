"""Checks for weather availability, plant isolation, and frozen-runner parity."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


RESULTS = Path(__file__).resolve().parents[2]
for artifact in ("seasonal_native_training", "grpo_progress_stale_ais", "grpo_progress_efficient_ais"):
    sys.path.insert(0, str(RESULTS / artifact / "reproduce"))

from causal_weather import AsOfSeasonalArchive, CausalEfficientNativeEngine, CausalScenarioEnvironment, _AsOfGrid
from native_efficient_guard import EfficientNativeEngine
from seasonal_environment import SeasonalArchive, _Grid
from world._geo import offset_latitude_longitude
from world.boat import BoatState
from world.scenario import RouteWaypoint


HOUR = np.datetime64("2026-10-16T18:00:00", "ns")
DECISION = HOUR + np.timedelta64(30, "m")
LATITUDES = np.array([39.90, 39.98, 40.06])
LONGITUDES = np.array([17.90, 17.98, 18.06])
TIMES = (HOUR + np.arange(-1, 2) * np.timedelta64(1, "h")).astype("int64")
STATE = BoatState(40., 18., 90., .8, 3., -2., 0., .8, .03, .01)


def grid_values(initial, changed=None):
    values = np.tile(np.asarray(initial), (len(TIMES), len(LATITUDES), len(LONGITUDES), 1))
    if changed is not None:
        values[-1] = changed
    return values


def grid(initial, changed=None):
    with patch("seasonal_environment._read", return_value=(
        TIMES, LATITUDES, LONGITUDES, grid_values(initial, changed),
    )):
        return _Grid("synthetic.nc", "synthetic")


def archive(future_current=(.1, .05)):
    replay = SeasonalArchive.__new__(SeasonalArchive)
    replay._wind = grid((0., -8.))
    replay._current = grid((.1, .05), future_current)
    replay._wave = grid((.8, 4., 0., 1.))
    return replay


def waypoint(east):
    latitude, longitude = offset_latitude_longitude(STATE.latitude, STATE.longitude, east, 0.)
    return RouteWaypoint(latitude, longitude, "002")


class Scenario:
    def __init__(self, replay):
        self.archive = replay
        self.plant_observations = []

    def observe(self, state, when):
        self.plant_observations.append(when)
        return self.archive.observe(state, when, self.reports_at(when))

    def weather_history(self, latitude, longitude, when):
        return self.archive.weather_history(latitude, longitude, when)

    def reports_at(self, when):
        return ()

    def chart_path_is_clear(self, path, observation=None):
        return True


class CausalWeatherChecks(unittest.TestCase):
    def test_future_wet_mask_cannot_change_available_coastal_fallback(self):
        first = grid_values((.1, .05))
        second = first.copy()
        first[1] = second[1] = np.nan
        first[1, 0, 1] = second[1, 0, 1] = (.2, .1)
        first[-1] = np.nan
        second[-1] = (50., 50.)
        before_release = int(DECISION.astype("int64"))
        kwargs = {"times": TIMES, "latitudes": LATITUDES, "longitudes": LONGITUDES}
        a = _AsOfGrid(SimpleNamespace(fields=first, **kwargs))
        b = _AsOfGrid(SimpleNamespace(fields=second, **kwargs))
        np.testing.assert_array_equal(a.values(39.98, 17.98, before_release), (.2, .1))
        np.testing.assert_array_equal(a.values(39.98, 17.98, before_release),
                                      b.values(39.98, 17.98, before_release))
        self.assertGreater(a._fallback_count, 0)
        np.testing.assert_array_equal(b.values(39.98, 17.98, TIMES[-1]), (50., 50.))

    def test_unavailable_weather_only_changes_plant_transition(self):
        steady, changing = Scenario(archive()), Scenario(archive((1.7, -.6)))
        base = CausalEfficientNativeEngine(CausalScenarioEnvironment(steady))
        shifted = CausalEfficientNativeEngine(CausalScenarioEnvironment(changing))
        goal = waypoint(300.)
        before, after = (engine.decision(STATE, goal, DECISION) for engine in (base, shifted))
        self.assertFalse(changing.plant_observations)
        self.assertEqual(before.physics_index, after.physics_index)
        np.testing.assert_array_equal(before.features, after.features)
        np.testing.assert_array_equal(before.weather_history, after.weather_history)
        np.testing.assert_array_equal(before.allowed, after.allowed)
        self.assertEqual(before.weather_history.shape, (12, 5))
        self.assertNotEqual(
            AsOfSeasonalArchive(steady.archive).at(STATE.latitude, STATE.longitude, HOUR + np.timedelta64(1, "h")),
            AsOfSeasonalArchive(changing.archive).at(STATE.latitude, STATE.longitude, HOUR + np.timedelta64(1, "h")),
        )
        plain, perturbed = (engine.run(STATE, [goal], DECISION, cap_seconds=5.) for engine in (base, shifted))
        for field in ("choice_id", "rudder_deg", "flap_target_deg", "screened_guard_margin_m"):
            self.assertEqual(plain.controls[0][field], perturbed.controls[0][field])
        self.assertNotEqual(plain.final_state, perturbed.final_state)
        self.assertEqual(changing.plant_observations, [DECISION])

    def test_constant_weather_matches_pinned_runner_at_gate_entry(self):
        fixture = Scenario(archive())
        base = EfficientNativeEngine(fixture)
        isolated = CausalEfficientNativeEngine(CausalScenarioEnvironment(fixture))
        goal = waypoint(51.)
        original, modified = (engine.run(STATE, [goal], DECISION, cap_seconds=20.) for engine in (base, isolated))
        self.assertTrue(original.success)
        self.assertEqual(original.captures, modified.captures)
        self.assertEqual(original.trace, modified.trace)
        self.assertEqual(original.final_state, modified.final_state)
        self.assertEqual(modified.applied_chart_violations, 0)
        self.assertEqual(original.report_minimum_m, modified.report_minimum_m)
        self.assertEqual(original.truth_minimum_m, modified.truth_minimum_m)


if __name__ == "__main__":
    unittest.main()
