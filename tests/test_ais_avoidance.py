"""Check synchronized AIS sweeps, uncertainty, and existing-overlap recovery."""

from __future__ import annotations

import math
import unittest

import numpy as np

from world import Fossen3DOFSaildroneDynamics, SaildroneMPC, Wind
from world._geo import offset_latitude_longitude
from world.ais import AISTarget
from world.ais_avoidance import AISPrediction
from world.ais_guidance import fallback_action
from world.boat import BoatState
from world.occupancy import LocalGrid
from world.world import WorldObservation


class AISPredictionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.boat = BoatState(40.225, 18.75, 90.0, 0.0)

    def test_swept_contact_is_detected_between_clear_waypoints(self) -> None:
        prediction = self._prediction(self._target(0.0, -10.0, speed=2.0))
        points = np.array([[-10.0, 0.0], [10.0, 0.0]])
        self.assertTrue(np.all(prediction.signed_margins(points, [0.0, 10.0]) > 0.0))

        assessment = prediction.assess_local_rollout(points, [0.0, 10.0])
        self.assertFalse(assessment.safe)
        self.assertFalse(assessment.recovering)
        self.assertAlmostEqual(assessment.minimum_center_distance_metres, 0.0, places=7)
        self.assertAlmostEqual(assessment.minimum_hull_clearance_metres, -3.5, places=7)
        self.assertAlmostEqual(assessment.first_contact_seconds, 5.0 - 6.0 / math.sqrt(8.0), places=7)
        self.assertAlmostEqual(assessment.first_hull_contact_seconds, 5.0 - 3.5 / math.sqrt(8.0), places=7)

    def test_spatial_crossing_at_different_times_is_clear(self) -> None:
        prediction = self._prediction(self._target(0.0, -20.0, speed=2.0))
        assessment = prediction.assess_local_rollout([[-10.0, 0.0], [10.0, 0.0]], [0.0, 10.0])
        self.assertTrue(assessment.safe)
        self.assertIsNone(assessment.first_contact_seconds)
        self.assertAlmostEqual(assessment.minimum_center_distance_metres, math.sqrt(50.0), places=7)

    def test_touch_is_contact_even_without_positive_duration_inside(self) -> None:
        prediction = self._prediction(self._target(0.0, 0.0))
        assessment = prediction.assess_local_rollout([[-10.0, 6.0], [10.0, 6.0]], [0.0, 10.0])
        self.assertFalse(assessment.safe)
        self.assertIsNone(assessment.first_hull_contact_seconds)
        self.assertAlmostEqual(assessment.first_contact_seconds, 5.0, places=6)

    def test_growing_radius_minimum_is_distinct_from_center_cpa(self) -> None:
        prediction = self._prediction(self._target(0.0, 0.0), growth=1.0)
        assessment = prediction.assess_local_rollout([[-10.0, 8.0], [10.0, 8.0]], [0.0, 10.0])
        contact, = assessment.contacts
        self.assertAlmostEqual(contact.closest_time_seconds, 5.0)
        expected_margin_time = (10.0 + 80.0 / math.sqrt(300.0)) / 2.0
        self.assertAlmostEqual(contact.minimum_clearance_time_seconds, expected_margin_time)
        self.assertAlmostEqual(assessment.first_contact_seconds, (52.0 - math.sqrt(1168.0)) / 6.0)
        self.assertGreater(assessment.minimum_hull_clearance_metres, 0.0)

    def test_buffer_recovery_stays_distinct_from_safe_and_checks_all_contacts(self) -> None:
        prediction = self._prediction(self._target(0.0, 0.0))
        escape = prediction.assess_local_rollout([[4.5, 0.0], [9.0, 0.0]], [0.0, 5.0])
        self.assertTrue(escape.initial_overlap)
        self.assertFalse(escape.safe)
        self.assertTrue(escape.recovering)
        self.assertEqual(escape.first_contact_seconds, 0.0)
        self.assertIsNone(escape.first_hull_contact_seconds)
        point = prediction.assess_local_rollout([[0.0, 0.0]], [0.0])
        self.assertFalse(point.safe)
        self.assertFalse(point.recovering)
        self.assertEqual(point.first_hull_contact_seconds, 0.0)

        worse_first = prediction.assess_local_rollout([[4.5, 0.0], [4.0, 0.0], [8.5, 0.0]], [0.0, 1.0, 5.0])
        self.assertFalse(worse_first.recovering)
        with_second = self._prediction(self._target(0.0, 0.0), self._target(14.0, 0.0, mmsi=999000102))
        incurs_second = with_second.assess_local_rollout([[4.5, 0.0], [8.5, 0.0]], [0.0, 5.0])
        self.assertFalse(incurs_second.recovering)
        self.assertGreater(incurs_second.minimum_hull_clearance_metres, 0.0)

    def test_report_age_uses_explicit_or_embedded_value_with_bounded_projection(self) -> None:
        target = self._target(0.0, 0.0, course=90.0, speed=2.0, age=200.0)
        recent = self._target(0.0, 0.0, speed=1.0, age=7.0, mmsi=999000102)
        prediction = self._prediction(target, recent, growth=0.1, report_age_seconds={recent.mmsi: 8.0})

        self.assertEqual(prediction.targets, (target, recent))
        np.testing.assert_allclose(prediction.positions_metres, [[300.0, 0.0], [0.0, 8.0]], atol=1e-8)
        np.testing.assert_allclose(prediction.centres_at([0.0, 5.0])[1], [[310.0, 0.0], [0.0, 13.0]], atol=1e-8)
        np.testing.assert_allclose(prediction.radii_at([0.0, 5.0]), [[126.0, 6.8], [126.5, 7.3]])
        first, second = prediction.stage_obstacles(5.0)
        self.assertTrue(first.extrapolation_limited)
        self.assertFalse(second.extrapolation_limited)
        self.assertEqual((first.mmsi, second.mmsi), (target.mmsi, recent.mmsi))

    def test_empty_traffic_and_invalid_rollout(self) -> None:
        prediction = self._prediction()
        self.assertTrue(prediction.assess_rollout([self.boat], [0.0]).safe)
        self.assertEqual(prediction.centres_at([0.0, 1.0]).shape, (2, 0, 2))
        with self.assertRaisesRegex(ValueError, "increasing"):
            prediction.assess_local_rollout([[0.0, 0.0], [1.0, 1.0]], [1.0, 1.0])

    def test_acceleration_allowance_covers_report_age_and_forecast_time(self) -> None:
        target = self._target(0.0, 0.0, age=30.0)
        default = self._prediction(target, growth=0.05)
        bounded = self._prediction(target, growth=0.05, acceleration_bound_mps2=0.02)
        np.testing.assert_allclose(default.radii_at([0.0, 5.0]).ravel(), [7.5, 7.75])
        np.testing.assert_allclose(bounded.radii_at([0.0, 5.0]).ravel(), [15.0, 18.25])

        boat = [[16.0, 0.0], [16.0, 0.0]]
        self.assertTrue(default.assess_local_rollout(boat, [0.0, 5.0]).safe)
        self.assertFalse(bounded.assess_local_rollout(boat, [0.0, 5.0]).safe)

    def test_report_age_budget_keeps_the_entire_applied_step(self) -> None:
        traffic = self._prediction(self._target(0.0, 0.0, age=10.0),
                                   self._target(50.0, 0.0, age=40.0, mmsi=999000102),
                                   growth=0.05, acceleration_bound_mps2=0.026)
        preview = traffic.for_preview(35.0, 5.0)
        np.testing.assert_allclose(preview.radii_at([0.0, 5.0]), traffic.radii_at([0.0, 5.0]))
        for_elapsed = traffic.radii_at([25.0, 5.0])
        np.testing.assert_allclose(preview.radii_at(90.0), [for_elapsed[0, 0], for_elapsed[1, 1]])
        self.assertTrue(np.all(traffic.radii_at(90.0) > preview.radii_at(90.0)))

    def test_clearance_reserve_increases_the_swept_buffer_by_a_fixed_distance(self) -> None:
        target = self._target(0.0, 0.0, age=30.0)
        options = dict(growth=0.05, acceleration_bound_mps2=0.02)
        base = self._prediction(target, **options)
        reserved = self._prediction(target, clearance_reserve_metres=2.0, **options)
        times = [0.0, 5.0]
        positions = [[19.0, 0.0], [19.0, 0.0]]

        self.assertEqual(reserved.clearance_reserve_metres, 2.0)
        np.testing.assert_allclose(reserved.radii_at(times), base.radii_at(times) + 2.0)
        without_reserve = base.assess_local_rollout(positions, times)
        with_reserve = reserved.assess_local_rollout(positions, times)
        self.assertTrue(without_reserve.safe)
        self.assertFalse(with_reserve.safe)
        self.assertAlmostEqual(with_reserve.minimum_clearance_metres, without_reserve.minimum_clearance_metres - 2.0)
        self.assertEqual(with_reserve.minimum_hull_clearance_metres, without_reserve.minimum_hull_clearance_metres)

        with self.assertRaisesRegex(ValueError, "non-negative"):
            self._prediction(target, clearance_reserve_metres=-1.0)

    def test_strict_fallback_rejects_hull_clear_actions_inside_uncertainty_buffer(self) -> None:
        target = self._target(5.0, 0.0)
        observation = WorldObservation(
            np.datetime64("2026-09-30T03:00:00"), self.boat, wind=Wind(0., 0., 0., 0.), wave=None, current=None,
            seabed=None, coast=None, chart=None, ais_targets=(target,),
        )
        grid = LocalGrid(np.zeros((1, 21, 21)), ("chart",), 20.0, 420.0,
                         self.boat.latitude, self.boat.longitude)
        arguments = (
            Fossen3DOFSaildroneDynamics(), SaildroneMPC(), self.boat, observation,
            grid, np.zeros((21, 21), dtype=bool), self._prediction(target), 5.0,
        )
        legacy = fallback_action(*arguments)
        self.assertIsNotNone(legacy)
        self.assertGreater(legacy.assessment.minimum_hull_clearance_metres, 0.0)
        self.assertFalse(legacy.assessment.safe)
        self.assertIsNone(fallback_action(*arguments, full_clearance_only=True))

    def _target(self, east, north, *, speed=0.0, course=0.0, age=None, mmsi=999000101) -> AISTarget:
        latitude, longitude = offset_latitude_longitude(self.boat.latitude, self.boat.longitude, east, north)
        return AISTarget(mmsi=mmsi, latitude=float(latitude), longitude=float(longitude), speed=speed,
                         course=course, heading=course, length_metres=4.0, beam_metres=3.0,
                         report_age_seconds=age)

    def _prediction(self, *targets, growth=0.0, **options) -> AISPrediction:
        observation = WorldObservation(
            np.datetime64("2026-09-30T03:00:00"), self.boat, wind=None, wave=None, current=None,
            seabed=None, coast=None, chart=None, ais_targets=targets,
        )
        return AISPrediction.from_observation(observation, own_hull_radius_metres=1.0,
                                              safety_buffer_metres=2.5, uncertainty_metres=0.0,
                                              uncertainty_growth_mps=growth, **options)
