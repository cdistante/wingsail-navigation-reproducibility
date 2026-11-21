"""Exercise the shared ETA and first-window measures used for hold exit."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from world.boat import BoatState
from world.currents import Current
from world.drift_guidance import DriftMode
from world.forecast_eta_release import ForecastETARelease
from world.forecast_hold import _Polar
from world.fossen import Fossen3DOFSaildroneDynamics
from world.scenario import Waypoint
from world.weather import Wind


class ForecastETAReleaseTests(unittest.TestCase):
    def setUp(self):
        self.planner = ForecastETARelease(None, Fossen3DOFSaildroneDynamics(), 45., 50.)
        self.observation = SimpleNamespace(
            time=np.datetime64("2026-10-01T06:00:00"), boat=BoatState(40., 18., 60., 0.),
            wind=Wind(0., -5., 5., 0.), current=Current(0., 0., 0., 0., 0.),
        )
        self.hold = DriftMode(np.array([0., .1]), 60., 0., 0.)
        self.sail = DriftMode(np.array([0., .6]), 60., 0., 15.)
        self.goal = Waypoint(40.0025, 18.)
        self.target = np.array([0., 280.])

    def trial(self, permitted=lambda trim: True, modes=None):
        return self.planner.release_reference(
            self.observation, self.goal, self.target, self.hold,
            (self.hold, self.sail) if modes is None else modes, chart_allows=permitted,
        )

    def test_compares_two_etas_to_same_gate_and_rechecks_the_chart(self):
        with patch.object(self.planner, "_sample", return_value=_Polar.from_modes(
                self.observation, (self.sail,))) as forecast:
            self.assertIs(self.trial(), self.sail)
            details = self.planner.telemetry
            self.assertAlmostEqual(details["forecast_release_now_eta_s"], 383.3333333)
            self.assertAlmostEqual(details["forecast_release_wait_eta_s"], 633.3333333)
            self.assertAlmostEqual(details["forecast_release_eta_gain_s"], 250.)
            self.assertAlmostEqual(details["forecast_release_now_range_m"], 100.)
            self.assertAlmostEqual(details["forecast_release_wait_range_m"], 250.)
            calls = forecast.call_count
            self.assertIsNone(self.trial(permitted=lambda trim: trim is self.hold))
            self.assertEqual(forecast.call_count, calls)
            self.assertEqual(self.planner.telemetry, {})

    def test_keeps_hold_when_wait_eta_is_missing_or_gain_is_too_small(self):
        for arrivals in [(None,), ((0, 630.), (0, 626.)), ((0, 630.), (0, 640.))]:
            self.planner.invalidate_selection()
            with patch.object(self.planner, "_first_arrival", side_effect=arrivals):
                self.assertIsNone(self.trial())
                self.assertEqual(self.planner.telemetry, {})

    def test_skips_search_for_a_trim_that_projects_farther_away(self):
        away = DriftMode(np.array([0., -.6]), 60., 0., 15.)
        with patch.object(self.planner, "_first_arrival") as predict:
            self.assertIsNone(self.trial(modes=(self.hold, away)))
            predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
