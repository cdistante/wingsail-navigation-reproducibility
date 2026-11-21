"""Exercise forecast schedules with a known two-tack velocity polar."""

from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from world import BoatState, Current, Fossen3DOFSaildroneDynamics, MaritimeWorld, Waypoint, Wind, WorldObservation
from world._geo import offset_latitude_longitude
from world.forecast_tacks import ForecastTackPlanner
from world.learned_tack_arrival import TackArrivalPrediction


START_TIME = np.datetime64("2026-09-30T03:00:00", "s")
INITIAL_MODES = ((-1.0, 1.0), (1.0, 1.0))


class SyntheticForecastTackPlanner(ForecastTackPlanner):
    interval_seconds = 60
    horizon_intervals = 4
    tack_seconds = 30.0

    def __init__(self, future_modes=INITIAL_MODES):
        super().__init__(MaritimeWorld(), Fossen3DOFSaildroneDynamics(), 45.0, 2.0,
                         tracking_feedback=False)
        self.offsets = np.array([-45.0, 45.0])
        self.future_modes = future_modes

    def _sample(self, observation, goal, time):
        return replace(observation, time=time)

    def _polar(self, observation):
        modes = self.future_modes if observation.time >= START_TIME + np.timedelta64(60, "s") else INITIAL_MODES
        return np.array(modes), np.array([315.0, 45.0]), np.zeros(2)


class ForecastTackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.boat = BoatState(40.225, 18.75, heading=45.0, speed=1.0, surge_speed=1.0)
        self.observation = WorldObservation(
            time=START_TIME,
            boat=self.boat,
            wind=Wind(0.0, -5.0, 5.0, 0.0),
            wave=None,
            current=Current(0.0, 0.0, 0.0, 0.0, 0.0),
            seabed=None,
            coast=None,
            chart=None,
            ais_targets=(),
        )

    def waypoint(self, east: float, north: float) -> Waypoint:
        return Waypoint(*offset_latitude_longitude(self.boat.latitude, self.boat.longitude, east, north))

    def test_selects_and_executes_a_required_switch(self) -> None:
        planner = SyntheticForecastTackPlanner()
        goal = self.waypoint(0.0, 120.0)

        reference = planner.reference(self.observation, goal)

        self.assertIsNotNone(reference)
        self.assertAlmostEqual(reference.course, 45.0)
        self.assertAlmostEqual(reference.bow, 45.0)
        self.assertAlmostEqual(reference.tack_in_minutes, 1.0)
        self.assertLess(reference.closest_metres, 0.25)

        position = self.waypoint(60.0, 60.0)
        at_switch = replace(
            self.observation,
            time=START_TIME + np.timedelta64(60, "s"),
            boat=replace(self.boat, latitude=position.latitude, longitude=position.longitude),
        )
        reference = planner.reference(at_switch, goal)

        self.assertIsNotNone(reference)
        self.assertAlmostEqual(reference.course, 315.0)
        self.assertAlmostEqual(reference.bow, 315.0)
        self.assertIsNone(reference.tack_in_minutes)

    def test_future_polar_can_change_the_first_tack(self) -> None:
        goal = self.waypoint(0.0, 120.0)
        steady = SyntheticForecastTackPlanner()
        changing = SyntheticForecastTackPlanner(future_modes=((-1.0, -0.5), (1.0, 3.0)))

        steady_reference = steady.reference(self.observation, goal)
        changing_reference = changing.reference(self.observation, goal)

        self.assertIsNotNone(steady_reference)
        self.assertIsNotNone(changing_reference)
        self.assertAlmostEqual(steady_reference.course, 45.0)
        self.assertAlmostEqual(changing_reference.course, 315.0)
        self.assertAlmostEqual(changing_reference.tack_in_minutes, 1.0)
        self.assertLess(changing_reference.closest_metres, 0.25)

    def test_detects_arrival_between_interval_endpoints(self) -> None:
        planner = SyntheticForecastTackPlanner()
        goal = self.waypoint(30.0, 30.0)

        reference = planner.reference(self.observation, goal)

        self.assertIsNotNone(reference)
        self.assertAlmostEqual(reference.course, 45.0)
        self.assertIsNone(reference.tack_in_minutes)
        self.assertLess(reference.closest_metres, 0.25)

    def test_layline_eta_rescores_both_schedules_at_the_original_waypoint(self) -> None:
        planner = SyntheticForecastTackPlanner()
        planner.arrival_radius = 20.
        goal = self.waypoint(0., 100.)
        self.assertAlmostEqual(planner.reference(self.observation, goal).tack_in_minutes, 1.)
        forecast_eta, layline_eta = planner.compare_layline_arrival(self.observation, goal, 45., 315., 50.)
        self.assertAlmostEqual(forecast_eta, 130., delta=.02)
        self.assertAlmostEqual(layline_eta, 130. - 20. / np.sqrt(2.), delta=.02)

        position = self.waypoint(5., 5.)
        later = replace(self.observation, time=START_TIME + np.timedelta64(5, "s"),
                        boat=replace(self.boat, latitude=position.latitude, longitude=position.longitude))
        self.assertAlmostEqual(planner.reference(later, goal).tack_in_minutes, 55. / 60.)
        forecast_eta, layline_eta = planner.compare_layline_arrival(later, goal, 45., 315., 45.)
        self.assertAlmostEqual(forecast_eta, 125., delta=.02)
        self.assertAlmostEqual(layline_eta, 125. - 20. / np.sqrt(2.), delta=.02)
        self.assertIsNone(planner.compare_layline_arrival(later, goal, 0., None, 45.)[1])

    def test_turn_drift_continues_across_interval_boundaries(self) -> None:
        class DriftingPlanner(SyntheticForecastTackPlanner):
            interval_seconds = 120
            horizon_intervals = 2
            tack_seconds = 60.0

            def __init__(self):
                super().__init__()
                self.offsets = np.array([45.0])

            def _polar(self, observation):
                return np.array([[0.0, 1.0]]), np.array([0.0]), np.array([1.0, 0.0])

        starting_turn = replace(self.observation, boat=replace(self.boat, heading=180.0))
        planner = DriftingPlanner()
        reference = planner.reference(starting_turn, self.waypoint(30.0, 30.0))
        self.assertAlmostEqual(reference.closest_metres, 30.0, delta=0.01)

        planner = DriftingPlanner()
        near_boundary = replace(starting_turn, time=START_TIME + np.timedelta64(115, "s"))
        reference = planner.reference(near_boundary, self.waypoint(5.0, 70.0))
        self.assertAlmostEqual(reference.closest_metres, np.hypot(55.0, 5.0), delta=0.01)

    def test_intermediate_close_hauled_mode_avoids_an_unnecessary_tack(self) -> None:
        class ResolvedPlanner(SyntheticForecastTackPlanner):
            def __init__(self, refine):
                ForecastTackPlanner.__init__(
                    self, MaritimeWorld(), Fossen3DOFSaildroneDynamics(), 45.0, 2.0,
                    refine_modes=refine,
                )

            def _polar(self, observation):
                modes = np.full((len(self.offsets), 2), np.nan)
                for angle, velocity in ((45.5, (1.0, 1.0)), (-50.0, (-1.0, 0.0)),
                                        (-47.75, (-1.0, 0.5))):
                    modes[self.offsets == angle] = velocity
                return modes, self.offsets % 360.0, np.zeros(2)

        goal = self.waypoint(-60.0, 30.0)
        coarse = ResolvedPlanner(False).reference(self.observation, goal)
        refined = ResolvedPlanner(True).reference(self.observation, goal)
        self.assertIsNotNone(coarse.tack_in_minutes)
        self.assertAlmostEqual(refined.bow, 312.25)
        self.assertIsNone(refined.tack_in_minutes)
        self.assertLess(refined.closest_metres, 0.01)

    def test_tracking_feedback_waits_for_settled_motion_and_bounds_error(self) -> None:
        planner = SyntheticForecastTackPlanner()
        planner.tracking_feedback = True
        planner._tracking_reference = 45.0
        planner._tracking_stable_since = 0.0
        observation = replace(self.observation, current=Current(0.2, -0.1, 0.224, 116.6, 0.0))
        actual = planner.dynamics._body_to_earth(45.0) @ np.array([1.0, 0.0]) + np.array([0.2, -0.1])
        nominal = actual - np.array([0.3, 0.4])
        polar = np.array([nominal]), np.array([45.0]), np.zeros(2)
        with patch('world.forecast_tacks.steady_ground_polar', return_value=polar) as solve:
            planner._observe_tracking(observation, 59.0)
            self.assertFalse(planner._tracking_residuals)
            solve.assert_not_called()
            planner._observe_tracking(observation, 60.0)
            solve.assert_called_once()
        residual = planner.dynamics._body_to_earth(0.0) @ planner._tracking_residuals[1]
        np.testing.assert_allclose(residual, [0.09, 0.12], atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(residual), 0.15)
        self.assertEqual(planner.tracking_telemetry['forecast_tracking_samples'], 1)
        # A long interruption requires another settled window; the learned
        # tack estimate remains available rather than being tied to a goal.
        with patch('world.forecast_tacks.steady_ground_polar') as solve:
            planner._observe_tracking(observation, 300.0)
            solve.assert_not_called()
        self.assertIn(1, planner._tracking_residuals)

    def test_explicit_forecast_interval_preserves_the_48_minute_horizon(self) -> None:
        for interval in (60, 120):
            planner = ForecastTackPlanner(MaritimeWorld(), Fossen3DOFSaildroneDynamics(), 45.0, 50.0,
                                           interval_seconds=interval)
            self.assertEqual(planner.interval_seconds, interval)
            self.assertEqual(planner.interval_seconds * planner.horizon_intervals, 2880)
        # Test subclasses can retain their deliberately short synthetic horizon.
        synthetic = SyntheticForecastTackPlanner()
        self.assertEqual((synthetic.interval_seconds, synthetic.horizon_intervals), (60, 4))
        for interval in (30, 61, 0, True, 60.5):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                ForecastTackPlanner(MaritimeWorld(), Fossen3DOFSaildroneDynamics(), 45.0, 50.0,
                                    interval_seconds=interval)

    def test_override_suspension_discards_unapplied_reference_history_and_replans(self) -> None:
        planner = SyntheticForecastTackPlanner()
        planner.tracking_feedback = True
        goal = self.waypoint(0.0, 120.0)
        planner.reference(self.observation, goal)
        planner._tracking_residuals[1] = np.array([0.02, -0.03])
        planner._commit_until = float(START_TIME.astype('datetime64[s]').astype(int)) + 300.0
        planner._committed_mode, planner._committed_side = 1, 1
        planner._last_command_bow, planner._last_command_side = 45.0, 1

        planner.suspend("AIS steering override")

        self.assertIsNone(planner._choice)
        self.assertIsNone(planner._window)
        self.assertFalse(planner._polars)
        self.assertFalse(planner._tracking_residuals)
        self.assertIsNone(planner._tracking_reference)
        self.assertIsNone(planner._tracking_stable_since)
        self.assertIsNone(planner._tracking_last_observation)
        self.assertIsNone(planner._committed_side)
        self.assertEqual(planner.tracking_telemetry['forecast_tracking_source'], 'suspended')
        self.assertEqual(planner.tracking_telemetry['forecast_suspension_reason'], 'AIS steering override')
        resumed = replace(self.observation, time=START_TIME + np.timedelta64(60, 's'))
        with patch('world.forecast_tacks.steady_ground_polar') as observer:
            reference = planner.reference(resumed, goal)
            planner.reference(replace(resumed, time=resumed.time + np.timedelta64(5, 's')), goal)
        self.assertIsNotNone(reference)
        observer.assert_not_called()  # Override motion cannot become a settled NMPC residual.
        self.assertEqual(planner.tracking_telemetry['forecast_tracking_samples'], 0)

    def test_gate_aim_disk_is_inside_original_capture_gate(self) -> None:
        planner = SyntheticForecastTackPlanner()
        planner.arrival_radius = 50.0
        planner.gate_aim_fraction = 0.5
        goal = self.waypoint(0.0, 120.0)
        planner.continuation_goal = self.waypoint(-200.0, 120.0)
        aim, radius = planner._gate_target(self.observation, goal)
        original = np.array([0.0, 120.0])
        self.assertAlmostEqual(radius, 25.0)
        self.assertLess(aim[0], -24.9)
        for angle in np.linspace(0.0, 2.0 * np.pi, 361):
            point = aim + radius * np.array([np.cos(angle), np.sin(angle)])
            self.assertLessEqual(np.linalg.norm(point - original), 50.01)
        self.assertEqual(planner.arrival_radius, 50.0)
        self.assertEqual(planner.gate_telemetry['forecast_gate_true_radius_m'], 50.0)

    def test_gate_aim_does_not_change_legacy_or_terminal_reference(self) -> None:
        goal = self.waypoint(0.0, 120.0)
        baseline = SyntheticForecastTackPlanner().reference(self.observation, goal)
        disabled = SyntheticForecastTackPlanner()
        disabled.continuation_goal = self.waypoint(-100.0, 120.0)
        terminal = SyntheticForecastTackPlanner()
        terminal.gate_aim_fraction = 0.5
        self.assertEqual(disabled.reference(self.observation, goal), baseline)
        self.assertEqual(terminal.reference(self.observation, goal), baseline)
        self.assertEqual(disabled.gate_telemetry, {})
        self.assertEqual(terminal.gate_telemetry, {})

    def test_gate_aim_replans_changed_continuation_and_samples_original_goal(self) -> None:
        planner = SyntheticForecastTackPlanner()
        planner.gate_aim_fraction = 0.5
        goal = self.waypoint(0.0, 120.0)
        with (patch.object(planner, '_plan', wraps=planner._plan) as plan,
              patch.object(planner, '_sample', wraps=planner._sample) as sample):
            planner.continuation_goal = self.waypoint(-100.0, 120.0)
            planner.reference(self.observation, goal)
            planner.continuation_goal = self.waypoint(100.0, 120.0)
            planner.reference(self.observation, goal)
        self.assertEqual(plan.call_count, 2)
        self.assertTrue(sample.call_args_list)
        self.assertTrue(all(call.args[1] == goal for call in sample.call_args_list))
        self.assertEqual(planner._active_goal, goal)

    def test_gate_aim_uses_inner_radius_and_caps_short_continuation(self) -> None:
        planner = SyntheticForecastTackPlanner()
        planner.arrival_radius = 50.0
        planner.gate_aim_fraction = 0.5
        planner.continuation_goal = self.waypoint(0.0, 125.0)
        _, radius = planner._gate_target(self.observation, self.waypoint(0.0, 120.0))
        self.assertAlmostEqual(radius, 47.5, delta=0.01)
        travel = np.array([[20.0, 60.0]])
        target = np.array([0.0, 60.0])
        coarse, _ = planner._score_path(target, travel, np.array([60.0]), 0.0, 0)
        inner, _ = planner._score_path(target, travel, np.array([60.0]), 0.0, 0, arrival_radius=10.0)
        self.assertEqual(coarse[0], 0.0)
        self.assertNotEqual(inner[0], 0.0)
        for fraction in (-0.1, 0.51, float('nan'), float('inf')):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                ForecastTackPlanner(MaritimeWorld(), Fossen3DOFSaildroneDynamics(), 45.0, 50.0,
                                    gate_aim_fraction=fraction)

    def test_gate_aim_upwind_guard_leaves_crosswind_continuation_unchanged(self) -> None:
        planner = SyntheticForecastTackPlanner()
        planner.arrival_radius = 50.0
        planner.gate_aim_fraction = 0.5
        planner.gate_aim_upwind_angle = 60.0
        goal = self.waypoint(0.0, 120.0)
        planner.continuation_goal = self.waypoint(-200.0, 120.0)
        aim, radius = planner._gate_target(self.observation, goal)
        np.testing.assert_allclose(aim, [0.0, 120.0], atol=0.01)
        self.assertEqual(radius, 50.0)
        self.assertEqual(planner.gate_telemetry['forecast_gate_aim_guard_reason'],
                         'continuation outside upwind cone')
        west_wind = replace(self.observation, wind=Wind(5.0, 0.0, 5.0, 270.0))
        aim, radius = planner._gate_target(west_wind, goal)
        self.assertLess(aim[0], -24.9)
        self.assertEqual(radius, 25.0)
        self.assertEqual(planner.gate_telemetry['forecast_gate_aim_guard_reason'], 'applied')

    def test_maneuver_commitment_prevents_forecast_boundary_reversal(self) -> None:
        goal = self.waypoint(30.0, 30.0)
        starting_turn = replace(self.observation, time=START_TIME + np.timedelta64(45, 's'),
                                boat=replace(self.boat, heading=180.0))
        position = self.waypoint(80.0, 0.0)
        mid_turn = replace(
            self.observation, time=START_TIME + np.timedelta64(60, 's'),
            boat=replace(self.boat, latitude=position.latitude, longitude=position.longitude, heading=113.0),
        )
        free = SyntheticForecastTackPlanner()
        committed = SyntheticForecastTackPlanner()
        committed.maneuver_commit_seconds = 60.0
        self.assertAlmostEqual(free.reference(starting_turn, goal).bow, 45.0)
        self.assertAlmostEqual(committed.reference(starting_turn, goal).bow, 45.0)
        # Without commitment, the new minute's plan reverses the turn just
        # fifteen seconds after the initial bow reference was issued.
        self.assertAlmostEqual(free.reference(mid_turn, goal).bow, 315.0)
        self.assertAlmostEqual(committed.reference(mid_turn, goal).bow, 45.0)
        self.assertAlmostEqual(committed.maneuver_telemetry['forecast_maneuver_commit_remaining_s'], 45.0)
        after_deadline = replace(mid_turn, time=START_TIME + np.timedelta64(120, 's'))
        self.assertAlmostEqual(committed.reference(after_deadline, goal).bow, 315.0)
        self.assertEqual(free.maneuver_telemetry, {})

    def test_commitment_keeps_side_across_goal_handoff_and_invalid_mode(self) -> None:
        planner = SyntheticForecastTackPlanner()
        planner.maneuver_commit_seconds = 60.0
        turning = replace(self.observation, time=START_TIME + np.timedelta64(45, 's'),
                           boat=replace(self.boat, heading=180.0))
        planner.reference(turning, self.waypoint(30.0, 30.0))
        next_goal = self.waypoint(-60.0, 60.0)
        still_turning = replace(turning, time=START_TIME + np.timedelta64(60, 's'))
        self.assertAlmostEqual(planner.reference(still_turning, next_goal).bow, 45.0)

        class AlternateSameSide(SyntheticForecastTackPlanner):
            def __init__(self, has_alternative=True):
                super().__init__()
                self.offsets = np.array([-45.0, 45.0, 50.0])
                self.has_alternative = has_alternative

            def _polar(self, observation):
                other = [1.0, 1.0] if self.has_alternative else [np.nan, np.nan]
                return np.array([[-1.0, 1.0], [np.nan, np.nan], other]), np.array([315.0, 45.0, 50.0]), np.zeros(2)

        second = float(START_TIME.astype(np.int64))
        window = int(second // 60)
        alternate = AlternateSameSide()
        schedule = alternate._plan(self.observation, self.waypoint(30.0, 30.0), window, second,
                                    first_mode=1, first_side=1)
        self.assertEqual(schedule[0], 2)
        unavailable = AlternateSameSide(False)
        self.assertIsNone(unavailable._plan(self.observation, self.waypoint(30.0, 30.0), window, second,
                                           first_mode=1, first_side=1))

    def test_maneuver_and_upwind_guard_parameters_reject_invalid_values(self) -> None:
        for parameter, values in (
            ('maneuver_commit_seconds', (-1.0, float('nan'), float('inf'))),
            ('gate_aim_upwind_angle', (-1.0, 181.0, float('nan'), float('inf'))),
        ):
            for value in values:
                with self.subTest(parameter=parameter, value=value), self.assertRaises(ValueError):
                    ForecastTackPlanner(MaritimeWorld(), Fossen3DOFSaildroneDynamics(), 45.0, 50.0,
                                        **{parameter: value})

    def test_tracking_feedback_rejects_turns_reverse_motion_and_reaching(self) -> None:
        cases = (
            (45.0, replace(self.boat, yaw_rate=np.radians(1.0))),
            (45.0, replace(self.boat, surge_speed=-0.5)),
            (45.0, replace(self.boat, heading=60.0)),
            (95.0, replace(self.boat, heading=95.0)),
        )
        for bow, boat in cases:
            with self.subTest(bow=bow, boat=boat):
                planner = SyntheticForecastTackPlanner()
                planner.tracking_feedback = True
                planner._tracking_reference = bow
                planner._tracking_stable_since = 0.0
                with patch('world.forecast_tacks.steady_ground_polar') as solve:
                    planner._observe_tracking(replace(self.observation, boat=boat), 60.0)
                    solve.assert_not_called()
                self.assertFalse(planner._tracking_residuals)

    def test_tracking_correction_rotates_with_wind_and_leaves_downwind_modes(self) -> None:
        planner = SyntheticForecastTackPlanner()
        planner.offsets = np.array([-45.0, 45.0, 125.0])
        raw = np.zeros((1, 3, 2))
        # A +0.1 m/s residual along the wind-FROM axis points east when
        # future wind is FROM90°, and affects the observed tack alone.
        planner._tracking_residuals[1] = np.array([0.1, 0.0])
        planner._polars[10] = raw[0].copy(), np.array([45.0, 135.0, 215.0]), np.zeros(2)
        corrected = planner._tracking_velocities(raw, 10)
        np.testing.assert_allclose(corrected[0, 1], [0.1, 0.0], atol=1e-12)
        np.testing.assert_array_equal(corrected[0, 0], raw[0, 0])
        np.testing.assert_array_equal(corrected[0, 2], raw[0, 2])
        np.testing.assert_array_equal(raw, np.zeros((1, 3, 2)))
        np.testing.assert_array_equal(planner._polars[10][0], raw[0])

    def test_tracking_prediction_keeps_nominal_emitted_course_across_goals(self) -> None:
        planner = SyntheticForecastTackPlanner()
        planner.tracking_feedback = True
        planner._tracking_residuals[1] = np.array([0.0, 0.1])
        window = int(START_TIME.astype(np.int64)) // planner.interval_seconds
        raw_polar = np.array(INITIAL_MODES), np.array([315.0, 45.0]), np.zeros(2)

        def plan(*args, **kwargs):
            planner._polars[window] = raw_polar
            return 1, 1, window, 0.0

        with patch.object(planner, '_plan', side_effect=plan):
            for goal in (self.waypoint(0.0, 120.0), self.waypoint(0.0, 150.0)):
                reference = planner.reference(self.observation, goal)
                self.assertAlmostEqual(reference.course, 45.0)
                self.assertAlmostEqual(reference.bow, 45.0)
                self.assertIn(1, planner._tracking_residuals)
        corrected = planner._tracking_velocities(np.array([raw_polar[0]]), window)
        self.assertGreater(np.degrees(np.arctan2(*corrected[0, 1])), 45.0)

    def test_learned_progress_ranks_comparable_first_legs(self) -> None:
        planner = SyntheticForecastTackPlanner()
        planner.learned_selector = Mock()
        planner.learned_selector.predict_progress.side_effect = [20.0, 40.0]
        baseline = ((0.0, 100.0, 0.0, 0.0), 1, 1, 0, 0.0)
        alternative = ((0.0, 220.0, 60.0, 0.0), 0, 1, 3, 0.0)

        result = planner._choose_first_leg(
            self.observation, self.waypoint(0.0, 120.0), baseline, {1: baseline, 0: alternative},
            np.array(INITIAL_MODES), np.array([315.0, 45.0]), scheduled_switch=False,
        )

        self.assertEqual(result, alternative)
        self.assertEqual(planner.learned_selection['learned_predicted_progress_gain_m'], 20.0)

    def test_learned_score_preserves_a_scheduled_switch_and_arrival_class(self) -> None:
        planner = SyntheticForecastTackPlanner()
        planner.learned_selector = Mock()
        baseline = ((0.0, 100.0, 0.0, 0.0), 1, 1, 0, 0.0)
        outside_radius = ((1.0, 3.0, 0.0, 0.0), 0, 0, 0, 3.0)
        for scheduled_switch in (True, False):
            with self.subTest(scheduled_switch=scheduled_switch):
                result = planner._choose_first_leg(
                    self.observation, self.waypoint(0.0, 120.0), baseline, {1: baseline, 0: outside_radius},
                    np.array(INITIAL_MODES), np.array([315.0, 45.0]), scheduled_switch=scheduled_switch,
                )
                self.assertEqual(result, baseline)
        planner.learned_selector.predict_progress.assert_not_called()

    def test_learned_arrival_scores_first_leg_and_transition_time(self) -> None:
        planner = SyntheticForecastTackPlanner()
        window = 10
        polar = np.array(INITIAL_MODES), np.array([315.0, 45.0]), np.zeros(2)
        planner._polars = {window: polar, window + 1: polar}
        planner.arrival_selector = Mock()
        planner.arrival_selector.predict_arrival.side_effect = [
            TackArrivalPrediction(capture_probability=0.95, arrival_seconds=180.0),
            TackArrivalPrediction(capture_probability=0.94, arrival_seconds=120.0),
        ]
        baseline = ((0.0, 100.0, 0.0, 0.0), 1, 1, window, 0.0)
        alternative = ((0.0, 180.0, 30.0, 0.0), 0, 1, window + 1, 0.0)

        selected = planner._choose_arrival_schedule(
            self.observation, self.waypoint(0.0, 120.0), baseline, {1: baseline, 0: alternative},
            window, 605.0, scheduled_switch=False,
        )

        self.assertEqual(selected, alternative)
        first, second = planner.arrival_selector.predict_arrival.call_args_list
        self.assertEqual(first.args[2:], (45.0, 45.0, None, None, None, 2.0, 240.0))
        self.assertEqual(second.args[2:], (315.0, 315.0, 45.0, 45.0, 55.0, 2.0, 240.0))
        self.assertAlmostEqual(planner.learned_selection['learned_predicted_arrival_gain_s'], 55.8)

    def test_arrival_selector_preserves_capture_probability_and_committed_switch(self) -> None:
        planner = SyntheticForecastTackPlanner()
        window = 10
        planner._polars[window] = np.array(INITIAL_MODES), np.array([315.0, 45.0]), np.zeros(2)
        planner.arrival_selector = Mock()
        planner.arrival_selector.predict_arrival.side_effect = [
            TackArrivalPrediction(capture_probability=0.95, arrival_seconds=180.0),
            TackArrivalPrediction(capture_probability=0.70, arrival_seconds=60.0),
        ]
        baseline = ((0.0, 100.0, 0.0, 0.0), 1, 1, window, 0.0)
        alternative = ((0.0, 150.0, 0.0, 0.0), 0, 0, window, 0.0)
        for scheduled_switch in (True, False):
            with self.subTest(scheduled_switch=scheduled_switch):
                selected = planner._choose_arrival_schedule(
                    self.observation, self.waypoint(0.0, 120.0), baseline, {1: baseline, 0: alternative},
                    window, 605.0, scheduled_switch=scheduled_switch,
                )
                self.assertEqual(selected, baseline)
        self.assertEqual(planner.arrival_selector.predict_arrival.call_count, 2)

    def swung_away(self, heading: float = 265.0) -> WorldObservation:
        """The next window, with the plant carried onto the opposite tack.

        The recorded Otranto limit cycle behaved this way: a commanded tack
        reached its close-hauled bow, fell away through the wind and left the
        boat on the other side before the next forecast window.
        """
        return replace(
            self.observation, time=START_TIME + np.timedelta64(60, "s"),
            boat=replace(self.boat, heading=heading),
        )

    def test_zero_thresholds_keep_the_original_side_selection(self) -> None:
        planner = SyntheticForecastTackPlanner()
        goal = self.waypoint(0.0, 900.0)

        first = planner.reference(self.observation, goal)
        second = planner.reference(self.swung_away(), goal)

        self.assertAlmostEqual(first.bow, 45.0)
        self.assertAlmostEqual(second.bow, 315.0)
        self.assertEqual(planner.tack_economy_telemetry, {})

    def test_a_swung_heading_alone_does_not_buy_a_change_of_tack(self) -> None:
        planner = SyntheticForecastTackPlanner()
        planner.tack_gain_metres = 50.0
        goal = self.waypoint(0.0, 900.0)

        first = planner.reference(self.observation, goal)
        second = planner.reference(self.swung_away(), goal)

        self.assertAlmostEqual(first.bow, 45.0)
        self.assertAlmostEqual(second.bow, 45.0)
        details = planner.tack_economy_telemetry
        self.assertEqual(details['forecast_tack_side_rule'], 'side retained')
        self.assertEqual(details['forecast_tack_side_retained'], 1)
        self.assertEqual(details['forecast_tack_side_candidate'], -1)
        self.assertLess(details['forecast_tack_side_gain_m'], 50.0)
        self.assertEqual(details['forecast_tack_side_required_m'], 50.0)

    def test_a_predicted_gain_above_the_threshold_still_changes_tack(self) -> None:
        planner = SyntheticForecastTackPlanner()
        planner.tack_gain_metres = 5.0
        goal = self.waypoint(0.0, 900.0)

        planner.reference(self.observation, goal)
        second = planner.reference(self.swung_away(), goal)

        self.assertAlmostEqual(second.bow, 315.0)
        details = planner.tack_economy_telemetry
        self.assertEqual(details['forecast_tack_side_rule'], 'side change admitted')
        self.assertGreater(details['forecast_tack_side_gain_m'], 5.0)

    def test_the_threshold_never_blocks_an_already_scheduled_switch(self) -> None:
        planner = SyntheticForecastTackPlanner()
        planner.tack_gain_metres = 1e6
        goal = self.waypoint(0.0, 900.0)

        first = planner.reference(self.observation, goal)
        self.assertAlmostEqual(first.bow, 45.0)
        self.assertAlmostEqual(first.tack_in_minutes, 2.0)
        at_switch = replace(
            self.observation, time=START_TIME + np.timedelta64(120, "s"),
            boat=replace(self.boat, heading=45.0),
        )

        second = planner.reference(at_switch, goal)

        # The switch executes on its planned window even though returning to
        # the opposite side could never clear this threshold on its own.
        self.assertAlmostEqual(second.bow, 315.0)
        self.assertNotEqual(planner.tack_economy_telemetry.get('forecast_tack_side_rule'), 'side retained')

    def test_a_dropped_scheduled_mode_restores_the_threshold(self) -> None:
        # The scheduled switch only exempts the decision while that mode is
        # still a valid first leg. If its equilibrium is gone, the window
        # plans freely and must pay for crossing the wind like any other.
        planner = SyntheticForecastTackPlanner()
        planner.offsets = np.array([-60.0, -45.0, 45.0, 60.0])
        bows = np.array([295.0, 310.0, 40.0, 55.0])
        usable = np.array([(-0.9, 1.0), (-1.0, 1.0), (1.0, 1.0), (0.9, 1.0)])
        planner._polar = lambda observation: (usable, bows, np.zeros(2))
        goal = self.waypoint(0.0, 900.0)
        planner.reference(self.observation, goal)
        self.assertEqual(planner._last_command_side, 1)

        dropped = usable.copy()
        dropped[1] = np.nan
        planner.tack_gain_metres = 1e6
        planner._polars.clear()
        planner._polar = lambda observation: (dropped, bows, np.zeros(2))
        planner._choice = (2, 1, planner._window + 1, 0.0)
        swung = replace(
            self.observation, time=START_TIME + np.timedelta64(60, "s"),
            boat=replace(self.boat, heading=265.0),
        )

        reference = planner.reference(swung, goal)

        self.assertIn(reference.bow, (40.0, 55.0))
        self.assertEqual(planner.tack_economy_telemetry['forecast_tack_side_rule'], 'side retained')

    def test_the_threshold_guards_arrival_time_when_a_schedule_captures(self) -> None:
        planner = SyntheticForecastTackPlanner()
        planner._last_command_side = 1
        faster = ((0.0, 300.0, 30.0, 1.0), 0, 0, 0, 1.0)
        retained = ((0.0, 340.0, 0.0, 1.0), 1, 1, 0, 1.0)
        catalogue = {0: faster, 1: retained}

        planner.tack_gain_seconds = 60.0
        held = planner._retain_tack_side(faster, dict(catalogue), scheduled=False,
                                         committed=False, goal_changed=False)
        self.assertEqual(held, retained)
        self.assertEqual(planner.tack_economy_telemetry['forecast_tack_side_gain_s'], 40.0)
        self.assertEqual(planner.tack_economy_telemetry['forecast_tack_side_rule'], 'side retained')

        planner.tack_gain_seconds = 30.0
        taken = planner._retain_tack_side(faster, dict(catalogue), scheduled=False,
                                          committed=False, goal_changed=False)
        self.assertEqual(taken, faster)
        self.assertEqual(planner.tack_economy_telemetry['forecast_tack_side_rule'],
                         'side change admitted')

    def test_the_threshold_compares_final_distance_without_a_closing_approach(self) -> None:
        planner = SyntheticForecastTackPlanner()
        planner.tack_gain_metres = 50.0
        planner._last_command_side = 1
        closer = ((2.0, 0.0, 900.0, 30.0), 0, 0, 0, 920.0)
        retained = ((2.0, 0.0, 930.0, 0.0), 1, 1, 0, 930.0)

        held = planner._retain_tack_side(closer, {0: closer, 1: retained},
                                         scheduled=False, committed=False, goal_changed=False)

        self.assertEqual(held, retained)
        self.assertEqual(planner.tack_economy_telemetry['forecast_tack_side_gain_m'], 30.0)

        # A schedule that keeps more of the horizon is preferred outright.
        complete = ((2.0, 0.0, 950.0, 30.0), 0, 0, 0, 950.0)
        truncated = ((2.0, 3.0, 400.0, 0.0), 1, 1, 0, 400.0)
        selected = planner._retain_tack_side(complete, {0: complete, 1: truncated},
                                             scheduled=False, committed=False, goal_changed=False)
        self.assertEqual(selected, complete)
        self.assertEqual(planner.tack_economy_telemetry['forecast_tack_side_rule'],
                         'forecast availability differs')

    def test_a_retained_side_also_constrains_the_learned_reranker(self) -> None:
        # The reranker scores whatever first legs remain comparable. The
        # rejected opposite-side schedule must not be among them, or a
        # learned proposal would quietly restore the tack just refused.
        planner = SyntheticForecastTackPlanner()
        planner.tack_gain_metres = 1e6
        planner.learned_selector = Mock()
        planner.learned_selector.predict_progress.return_value = 500.0
        goal = self.waypoint(0.0, 900.0)

        planner.reference(self.observation, goal)
        reference = planner.reference(self.swung_away(), goal)

        self.assertAlmostEqual(reference.bow, 45.0)
        self.assertEqual(planner.tack_economy_telemetry['forecast_tack_side_rule'], 'side retained')
        self.assertLessEqual(planner.learned_selector.predict_progress.call_count, 1)

    def test_a_new_waypoint_chooses_its_tack_without_the_threshold(self) -> None:
        planner = SyntheticForecastTackPlanner()
        planner.tack_gain_metres = 1e6

        planner.reference(self.observation, self.waypoint(0.0, 900.0))
        second = planner.reference(self.swung_away(), self.waypoint(-200.0, 900.0))

        self.assertAlmostEqual(second.bow, 315.0)
        self.assertEqual(planner.tack_economy_telemetry, {})

    def test_a_capture_outranks_the_threshold_against_a_near_miss(self) -> None:
        planner = SyntheticForecastTackPlanner()
        planner.tack_gain_seconds = 1e6
        planner.tack_gain_metres = 1e6
        planner._last_command_side = 1
        capture = ((0.0, 400.0, 30.0, 1.0), 0, 0, 0, 1.0)
        near_miss = ((1.0, 120.0, 0.0, 240.0), 1, 1, 0, 120.0)

        selected = planner._retain_tack_side(
            capture, {0: capture, 1: near_miss},
            scheduled=False, committed=False, goal_changed=False,
        )

        self.assertEqual(selected, capture)
        self.assertEqual(planner.tack_economy_telemetry['forecast_tack_side_rule'],
                         'outcome class improves')

    def test_the_threshold_yields_when_the_retained_side_has_no_equilibrium(self) -> None:
        planner = SyntheticForecastTackPlanner()
        planner.tack_gain_metres = 1e6
        planner._last_command_side = 1
        only_option = ((1.0, 120.0, 0.0, 240.0), 0, 0, 0, 120.0)

        selected = planner._retain_tack_side(
            only_option, {0: only_option},
            scheduled=False, committed=False, goal_changed=False,
        )

        self.assertEqual(selected, only_option)
        self.assertEqual(planner.tack_economy_telemetry['forecast_tack_side_rule'],
                         'retained side unavailable')

    def test_tack_gain_thresholds_must_be_finite_and_nonnegative(self) -> None:
        for seconds, metres in ((-1.0, 0.0), (0.0, -1.0), (np.nan, 0.0), (0.0, np.inf)):
            with self.subTest(seconds=seconds, metres=metres), self.assertRaises(ValueError):
                ForecastTackPlanner(MaritimeWorld(), Fossen3DOFSaildroneDynamics(), 45.0, 50.0,
                                    tack_gain_seconds=seconds, tack_gain_metres=metres)


if __name__ == "__main__":
    unittest.main()
