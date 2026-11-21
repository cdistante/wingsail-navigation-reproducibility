"""Check waypoint layline selection against known ground-track geometry."""

import unittest

import numpy as np

from world.waypoint_laylines import layline_candidates


class WaypointLaylineTests(unittest.TestCase):
    def test_direct_gate_entries_precede_ranked_two_tack_laylines(self):
        candidates = layline_candidates(
            (0., 200.), [(-1., 1.), (1., 2.), (0., .5), (0., 1.)],
            [315., 45., 277., 83.], 0., 0., (0., 0.), 800., 5., 1.,
        )
        self.assertEqual([(item.first, item.second) for item in candidates[:2]], [(3, None), (2, None)])
        distances = [item.first_distance_m for item in candidates[2:]]
        self.assertEqual(distances, sorted(distances))
        self.assertIn((0, 1), [(item.first, item.second) for item in candidates[2:]])
        self.assertAlmostEqual(candidates[0].arrival_s, 83. + 195.)
        for item in candidates:
            np.testing.assert_allclose(item.vertices_en[0], (0., 0.))
            self.assertAlmostEqual(np.linalg.norm(item.vertices_en[-1] - (0., 200.)), 5.)

    def test_current_during_turn_moves_the_layline_and_keeps_true_gate_entry(self):
        first = layline_candidates(
            (0., 200.), [(-1., 1.), (1., 1.)], [315., 45.], 0., 315.,
            (0., -.2), 360., 5., 1.,
        )[0]
        self.assertEqual((first.first, first.second), (0, 1))
        north = 109. - 5. / np.sqrt(2.)
        np.testing.assert_allclose(first.vertices_en[1:4], [(0., 0.), (-north, north), (-north, north - 18.)])
        self.assertAlmostEqual(first.first_distance_m, 109. * np.sqrt(2.) - 5.)
        self.assertAlmostEqual(first.arrival_s, 2. * 109. + 90. - 5. / np.sqrt(2.))

    def test_unreachable_upwind_target_and_short_horizon_return_no_candidates(self):
        self.assertEqual(layline_candidates(
            (0., 200.), [(-1., -2.), (1., -2.), (0., np.nan), (0., 0.)],
            [315., 45., 60., 90.], 0., 315., (0., -.2), 800., 50., 1.,
        ), [])
        self.assertEqual(layline_candidates(
            (0., 200.), [(0., 1.), (-1., 1.), (1., 1.)], [60., 315., 45.],
            0., 60., (0., 0.), 60., 5., 1.,
        ), [])

    def test_minimum_tack_still_intersects_gate_when_center_line_is_earlier(self):
        candidates = layline_candidates(
            (0., 58.), [(-1., 1.), (1., 1.)], [315., 45.], 0., 315.,
            (0., 0.), 300., 5., 1., min_first_s=30.,
        )
        self.assertAlmostEqual(candidates[0].first_distance_m, 30. * np.sqrt(2.))
        self.assertAlmostEqual(np.linalg.norm(candidates[0].vertices_en[-1] - (0., 58.)), 5.)

    def test_two_tack_uses_nearest_capture_disk_tangent(self):
        candidates = layline_candidates(
            (0., 200.), [(-1., 1.), (1., 1.)], [315., 45.], 0., 315.,
            (0., 0.), 500., 50., 1.,
        )
        self.assertEqual((candidates[0].first, candidates[0].second), (0, 1))
        self.assertAlmostEqual(candidates[0].first_distance_m, 200. / np.sqrt(2.) - 50.)
        self.assertAlmostEqual(np.linalg.norm(candidates[0].vertices_en[-1] - (0., 200.)), 50.)

    def test_backward_layline_tangent_does_not_make_a_gate_feasible(self):
        args = ([(1., 0.), (-.6, .8)], [315., 45.], 0., 315.)
        self.assertEqual(layline_candidates((100., -25.), *args, (0., 0.), 500., 10., 1.), [])
        choices = layline_candidates((100., -25.), *args, (0., -20. / 90.), 500., 10., 1.)
        forward = next(item for item in choices if (item.first, item.second) == (0, 1))
        self.assertAlmostEqual(forward.first_distance_m, 100. + np.sqrt(75.))
        self.assertGreater(np.linalg.norm(forward.vertices_en[-1] - forward.vertices_en[-2]), 0.)
        self.assertAlmostEqual(np.linalg.norm(forward.vertices_en[-1] - (100., -25.)), 10.)

    def test_horizon_moves_an_overlong_tangent_to_the_first_timely_layline(self):
        args = ((0., 200.), [(-1., 1.), (1., 1.)], [315., 45.], 0., 315., (0., 0.))
        earliest_tack = layline_candidates(*args, 500., 50., 1.)[0]
        limited = layline_candidates(*args, 245., 50., 1.)[0]
        self.assertEqual((limited.first, limited.second), (0, 1))
        self.assertGreater(limited.first_distance_m, earliest_tack.first_distance_m)
        self.assertAlmostEqual(limited.arrival_s, 245.)
        self.assertAlmostEqual(np.linalg.norm(limited.vertices_en[-1] - (0., 200.)), 50.)
        self.assertEqual(layline_candidates(*args, 239., 50., 1.), [])


if __name__ == "__main__":
    unittest.main()
