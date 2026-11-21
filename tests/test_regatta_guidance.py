"""Time-to-mark regressions for the optional forecast trim selector."""
from unittest import TestCase

import numpy as np

from world.drift_guidance import DriftMode
from world.forecast_hold import _Paths, _Polar
from world.regatta_guidance import RegattaHoldPlanner, _entry_fraction


class RegattaGuidanceTests(TestCase):
    def test_mark_entry_can_occur_between_samples(self):
        positions = np.array([[0., 0.], [0., 0.], [100., 0.]])
        travel = np.array([[100., 0.], [0., 100.], [0., 0.]])
        fractions = _entry_fraction(positions, travel, np.array([100., 0.]), 25.)
        np.testing.assert_allclose(fractions, [0.75, np.inf, 0.])

    def test_earliest_arrival_accounts_for_the_turn_before_sailing(self):
        # The faster opposite tack loses time during its turn. The current
        # tack reaches the mark first even though its straight speed is lower.
        planner = object.__new__(RegattaHoldPlanner)
        planner.arrival_radius = 50.
        v = np.array([[.6, 0.], [1., 0.]])
        modes = tuple(DriftMode(velocity, bow, 0., 0.) for velocity,bow in zip(v,[90.,270.]))
        polar = _Polar(modes, v, np.array([90.,270.]), np.zeros(2), 0., v)
        paths = _Paths(np.zeros((1,2)), np.array([90.]), np.array([-1]), np.zeros(1))
        _, best = planner._expand(paths, polar, 500., np.array([250.,0.]))
        self.assertIsNotNone(best)
        self.assertEqual(best[0], 0)
        self.assertAlmostEqual(best[1], 200/.6)

    def test_snapshot_shorter_than_turn_preserves_unfinished_heading(self):
        planner = object.__new__(RegattaHoldPlanner)
        planner.arrival_radius = 50.
        v = np.array([[1., 0.]])
        polar = _Polar((DriftMode(v[0], 270., 0., 0.),),v,np.array([270.]),np.zeros(2),0.,v)
        paths = _Paths(np.zeros((1,2)), np.array([90.]), np.array([-1]), np.zeros(1))
        future, hit = planner._expand(paths, polar, 30., np.array([250.,0.]))
        self.assertIsNone(hit)
        self.assertGreater(abs(future.bow[0] - 270.), 1.)
        np.testing.assert_allclose(future.position, [[0.,0.]])
