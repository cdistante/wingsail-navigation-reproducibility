"""Regression for an early multi-contact AIS buffer maneuver."""

from importlib.util import find_spec
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from world import BoatAction, BoatState, Fossen3DOFSaildroneDynamics, OccupancyGrid, SaildroneMPC, Wind, WorldObservation
from world.ais_avoidance import AISPrediction
from world.ais_buffer_maneuver import AISBufferManeuver, AISBufferPlanner, _Policy, _Rollout, _local_transition
from world.ais_guidance import assess_action
from world.synthetic_ais import SyntheticAIS, SyntheticAISReporting


class AISClearanceRefinementTests(unittest.TestCase):
    @unittest.skipUnless(find_spec("casadi"), "CasADi supplies the nonlinear refinement")
    def test_time_varying_full_radii_and_deterministic_iteration_limit(self):
        import casadi as ca
        from world.ais_clearance_refinement import refine_clearance

        state, command, forcing = ca.SX.sym("x", 9), ca.SX.sym("u", 2), ca.SX.sym("f", 5)
        step = ca.Function("clearance_test_step", [state, command, forcing], [
            ca.vertcat(state[:2] + command, state[2:7], command),
        ])
        # A stationary seed starts clear, but an expanding reported-contact
        # envelope approaches it. Moving away can retain the initial margin.
        # Collapsing the radii to a single worst radius would instead make the
        # seed already optimal at time zero and miss this improvement.
        radii = np.linspace(0.5, 1.5, 17)[:, None]
        arguments = (step, np.zeros(5), np.zeros((5, 9)), np.zeros((4, 2)),
                     np.array([[2., 0.]]), np.zeros((1, 2)), radii,
                     -np.ones(2), np.ones(2), np.ones(2), 1.)
        first = refine_clearance(*arguments, max_iterations=60, cpu_seconds=None)
        second = refine_clearance(*arguments, max_iterations=60, cpu_seconds=None)
        self.assertIsNotNone(first)
        np.testing.assert_allclose(first, second, rtol=0, atol=1e-9)
        self.assertLess(first[0, 0], -0.05)
        self.assertEqual(first.shape, (4, 2))
        retained = refine_clearance(*arguments, max_iterations=60, cpu_seconds=None, keep_best_iterate=True)
        self.assertIsNotNone(retained)
        self.assertLess(retained[0, 0], -.05)


class FastRecoveryPhysicsTests(unittest.TestCase):
    @unittest.skipUnless(find_spec("casadi"), "CasADi supplies the native batch kernel")
    def test_batched_heading_catalog_matches_scalar_controls_and_positions(self):
        start = BoatState(40.225, 18.746666667, heading=45, speed=0, flap=-10)
        observation = WorldObservation(
            time=np.datetime64("2026-09-30T03:00:00"), boat=start,
            wind=Wind(-.3, -3.46, 3.47, 5), wave=None, current=None,
            seabed=None, coast=None, chart=None, ais_targets=(),
        )
        planner = AISBufferPlanner(Fossen3DOFSaildroneDynamics(), SaildroneMPC(), 5, 45)
        policies = [_Policy(45, -15), _Policy(80, 15, .45, -15),
                    _Policy(160, 15, .65, next_heading=210, next_trim=-15, switch_after_seconds=20),
                    _Policy(50, 0, steering_gain=0)]
        native = _local_transition(planner.dynamics.parameters, 5)
        batch_headings, scalar_headings = [], []

        class ObservedTransition:
            def map(self, number):
                mapped = native.map(number)

                def evaluate(*args):
                    result = mapped(*args)
                    batch_headings.append(np.asarray(result)[2].copy())
                    return result
                return evaluate

            def __call__(self, *args):
                result = native(*args)
                scalar_headings.append(float(result[2]))
                return result

        with patch("world.ais_buffer_maneuver._local_transition", return_value=ObservedTransition()):
            batch = planner._coarse_rollouts(observation, policies, 60)
            for index, (policy, batched) in enumerate(zip(policies, batch)):
                scalar_headings.clear()
                scalar = planner._rollout(observation, policy, 60, 0, dense=False)
                np.testing.assert_allclose(batched.positions, scalar.positions, rtol=0, atol=1e-10)
                np.testing.assert_array_equal(batched.times, scalar.times)
                np.testing.assert_allclose([(a.rudder, a.flap) for a in batched.controls],
                                           [(a.rudder, a.flap) for a in scalar.controls], rtol=0, atol=1e-10)
                np.testing.assert_allclose(np.asarray(batch_headings)[:, index], scalar_headings,
                                           rtol=0, atol=1e-10)
        # Preset commands retain the scalar continuation, including the switch
        # back to feedback after the supplied sequence has ended.
        preset = replace(policies[0], controls=(BoatAction(5, -5), BoatAction(10, 0), BoatAction(15, 5)))
        batched, = planner._coarse_rollouts(observation, [preset], 60)
        scalar = planner._rollout(observation, preset, 60, 0, dense=False)
        np.testing.assert_allclose(batched.positions, scalar.positions, rtol=0, atol=1e-10)
        np.testing.assert_allclose([(a.rudder, a.flap) for a in batched.controls],
                                   [(a.rudder, a.flap) for a in scalar.controls], rtol=0, atol=1e-10)

    @unittest.skipUnless(find_spec("casadi"), "CasADi supplies the native batch kernel")
    def test_batched_feedback_matches_native_policy_rollouts(self):
        from world.ais_fast_recovery import FeedbackFamily, _rollouts

        start = BoatState(40.225, 18.746666667, heading=45, speed=0, flap=-10)
        observation = WorldObservation(
            time=np.datetime64("2026-09-30T03:00:00"), boat=start,
            wind=Wind(-.3, -3.46, 3.47, 5), wave=None, current=None,
            seabed=None, coast=None, chart=None, ais_targets=(),
        )
        planner = AISBufferPlanner(Fossen3DOFSaildroneDynamics(), SaildroneMPC(), 5, 45)
        state, forcing = planner._motion_inputs(observation)
        families = [FeedbackFamily(135, 40, -15, .8, 2.5, 20),
                    FeedbackFamily(315, 55, 15, .5, 2.5, 10)]
        states, commands = _rollouts(_local_transition(planner.dynamics.parameters, 5), state, forcing,
                                     families, planner.lower, planner.upper, planner.slew, 5, 12)
        for index, family in enumerate(families):
            policy = _Policy(family.pulse_heading, family.flap, family.gain,
                             next_heading=family.heading, next_trim=family.flap,
                             switch_after_seconds=family.pulse_seconds)
            native = planner._rollout(observation, policy, 60, 0, dense=False)
            np.testing.assert_allclose(states[:, index, :2], native.positions, rtol=0, atol=1e-10)
            np.testing.assert_allclose(np.degrees(commands[:, index]),
                                       [(action.rudder, action.flap) for action in native.controls], rtol=0, atol=1e-10)

    @unittest.skipUnless(find_spec("casadi"), "CasADi supplies the native batch kernel")
    def test_slew_uses_previous_command_when_actual_flap_lags(self):
        import casadi as ca
        from world.ais_fast_recovery import FeedbackFamily, _rollouts

        state, control, forcing = ca.SX.sym("x", 9), ca.SX.sym("u", 2), ca.SX.sym("p", 5)
        step = ca.Function("lagged_flap_test", [state, control, forcing], [ca.vertcat(state[:7], control)])
        states, commands = _rollouts(step, np.zeros(9), np.zeros(5),
                                     [FeedbackFamily(0, 0, 15, 1, 2.5, 0)],
                                     np.array([-30, -15]), np.array([30, 15]), np.array([7.5, 5]), 5, 3)
        np.testing.assert_allclose(np.degrees(commands[:, 0, 1]), [5, 10, 15], rtol=0, atol=1e-10)
        np.testing.assert_array_equal(states[:, 0, 6], 0)


class AISBufferManeuverTests(unittest.TestCase):
    def _recovery_context(self):
        start = BoatState(40.225, 18.746666667, heading=90, speed=0, flap=-10)
        target, = SyntheticAIS("crossing", start.latitude, start.longitude).load().sample(0)
        observation = WorldObservation(
            time=np.datetime64("2026-09-30T03:00:00"), boat=start,
            wind=Wind(0, 8, 8, 180), wave=None, current=None, seabed=None, coast=None, chart=None,
            ais_targets=(target,),
        )
        grid = OccupancyGrid(span=820, resolution=20).build(start, observation)
        planner = AISBufferPlanner(Fossen3DOFSaildroneDynamics(), SaildroneMPC(), 5.0, 45.0)
        traffic = replace(AISPrediction.from_observation(observation),
                          positions_metres=np.array([[100., 0.]]), velocities_mps=np.zeros((1, 2)),
                          clearance_radii_metres=np.array([150.])).for_preview(35, 5)
        action = BoatAction(0, -10)
        rollout = _Rollout(action, np.zeros((2, 2)), np.array([0., 5.]), (action,))
        return planner, observation, grid, np.zeros((41, 41), dtype=bool), traffic, rollout

    def test_recovery_requires_full_uncertainty_and_chart_clearance(self):
        planner, observation, grid, blocked, traffic, rollout = self._recovery_context()
        nominal = planner.nominal_traffic(traffic)
        legacy = _Policy(90, -15, controls=rollout.controls)
        recovery = replace(legacy, recovery_controls=True)
        # Historical uncapped previews retain their nominal-only semantics.
        uncapped = replace(traffic, preview_growth_seconds=None)
        self.assertIsNotNone(planner._screen(rollout, legacy, 0, uncapped, nominal, grid, blocked))
        # A bounded full-uncertainty preview must reject the same trajectory
        # for ordinary headings as well as explicit recovery controls.
        self.assertIsNone(planner._screen(rollout, legacy, 0, traffic, nominal, grid, blocked))
        self.assertIsNone(planner._screen(rollout, recovery, 0, traffic, nominal, grid, blocked))
        safe = replace(traffic, clearance_radii_metres=np.array([80.]))
        accepted = planner._screen(rollout, recovery, 0, safe, nominal, grid, blocked)
        self.assertTrue(accepted.recovery_controls)
        self.assertGreater(accepted.predicted_margin_metres, 0)
        self.assertEqual(accepted.recovery_remaining_seconds, 5)
        blocked[20, 20] = True
        self.assertIsNone(planner._screen(rollout, recovery, 0, safe, nominal, grid, blocked))

    def test_recovery_is_attempted_before_fallback_with_no_nominal_heading(self):
        planner, observation, grid, blocked, traffic, rollout = self._recovery_context()
        policy = _Policy(90, -15, controls=rollout.controls, recovery_controls=True)
        accepted = AISBufferManeuver(rollout.first, 90, 30, 8, refined_controls=True,
                                     recovery_controls=True, recovery_remaining_seconds=300)
        with patch.object(planner, "_policies", return_value=[_Policy(90, -15)]), \
                patch.object(planner, "_refine_downwind_turns", return_value=[]), \
                patch.object(planner, "_rollout", return_value=rollout), \
                patch.object(planner, "_screen", return_value=None), \
                patch.object(planner, "_recover_policy", return_value=(policy, accepted)) as recover:
            result = planner.choose(observation, grid, blocked, traffic, None, True)
        self.assertIs(result, accepted)
        recover.assert_called_once()
        self.assertIs(planner.policy, policy)

    def test_recovery_retention_rechecks_revised_reports(self):
        planner, observation, grid, blocked, traffic, rollout = self._recovery_context()
        planner.policy = _Policy(90, -15, controls=rollout.controls, recovery_controls=True)
        planner.duration, planner.elapsed = 300, 5
        accepted = AISBufferManeuver(rollout.first, 90, 30, 8, recovery_controls=True)
        with patch.object(planner, "_report_course_revised", return_value=True), \
                patch.object(planner, "_rollout", return_value=rollout), \
                patch.object(planner, "_screen", return_value=accepted) as screen, \
                patch.object(planner, "_recover_policy") as recover:
            self.assertIs(planner.choose(observation, grid, blocked, traffic, 90, True), accepted)
        self.assertIs(screen.call_args.args[3], traffic)
        self.assertEqual(planner.elapsed, 10)
        recover.assert_not_called()

    def test_recovery_is_attempted_after_all_dense_candidates_fail(self):
        planner, observation, grid, blocked, traffic, rollout = self._recovery_context()
        seed = _Policy(90, -15)
        policy = replace(seed, controls=rollout.controls, recovery_controls=True)
        coarse_pass = AISBufferManeuver(rollout.first, 90, 30, 8)
        recovery_pass = replace(coarse_pass, recovery_controls=True)
        with patch.object(planner, "_policies", return_value=[seed]), \
                patch.object(planner, "_refine_downwind_turns", return_value=[]), \
                patch.object(planner, "_rollout", return_value=rollout), \
                patch.object(planner, "_screen", side_effect=[coarse_pass, None]), \
                patch.object(planner, "_recover_policy", return_value=(policy, recovery_pass)) as recover:
            self.assertIs(planner.choose(observation, grid, blocked, traffic, 90, True), recovery_pass)
        recover.assert_called_once()

    def test_recovery_prefers_starboard_only_after_full_margin_priority(self):
        action = BoatAction(0, 0)
        right = _Rollout(action, np.array([[0., 0.], [10., 10.], [10., 50.]]),
                         np.array([0., 30., 300.]), (action,))
        left = replace(right, positions=right.positions * np.array([-1., 1.]))
        starboard, turn, override = AISBufferPlanner._recovery_rank(right, 8, 0, True)
        port, port_turn, port_override = AISBufferPlanner._recovery_rank(left, 8, 0, True)
        self.assertGreater(starboard, port)
        self.assertEqual(turn, 45)
        self.assertEqual(port_turn, -45)
        self.assertFalse(override)
        self.assertTrue(port_override)
        barely_clear, _, _ = AISBufferPlanner._recovery_rank(right, .05, 0, True)
        self.assertGreater(port, barely_clear)
        # Seed bow headings are absent from the score: the actual ground
        # departure determines the bias, including during astern motion.

    def test_unsafe_remaining_recovery_is_not_reused(self):
        planner, observation, grid, blocked, traffic, rollout = self._recovery_context()
        planner.policy = _Policy(90, -15, controls=rollout.controls, recovery_controls=True)
        planner.duration, planner.elapsed = 300, 5
        with patch.object(planner, "_policies", return_value=[]), \
                patch.object(planner, "_refine_downwind_turns", return_value=[]), \
                patch.object(planner, "_rollout", return_value=rollout), \
                patch.object(planner, "_screen", return_value=None), \
                patch.object(planner, "_recover_policy", return_value=None) as recover:
            self.assertIsNone(planner.choose(observation, grid, blocked, traffic, 90, True))
        recover.assert_called_once()
        self.assertIsNone(planner.policy)

    def test_bounded_ordinary_incumbent_is_not_retained_when_full_preview_is_unsafe(self):
        planner, observation, grid, blocked, traffic, rollout = self._recovery_context()
        planner.policy = _Policy(90, -15, controls=rollout.controls)
        planner.duration, planner.elapsed = 300, 5
        with patch.object(planner, "_report_course_revised", return_value=False), \
                patch.object(planner, "_policies", return_value=[]), \
                patch.object(planner, "_refine_downwind_turns", return_value=[]), \
                patch.object(planner, "_rollout", return_value=rollout), \
                patch.object(planner, "_recover_policy", return_value=None) as recover:
            self.assertIsNone(planner.choose(observation, grid, blocked, traffic, 90, True))
        recover.assert_called_once()
        self.assertIsNone(planner.policy)

    def test_unsuccessful_fast_recovery_does_not_start_a_long_search(self):
        from world.ais_fast_recovery import FeedbackFamily, FastRecoverySeed

        planner, observation, grid, blocked, traffic, rollout = self._recovery_context()
        fast = FastRecoverySeed(FeedbackFamily(90, 90, -15, .5, 2.5, 0),
                                np.zeros((1, 2)), np.zeros((2, 9)), -50.)
        with patch("world.ais_buffer_maneuver._local_transition", return_value=object()), \
                patch("world.ais_fast_recovery.search_feedback", return_value=(fast, {})), \
                patch("world.ais_clearance_refinement.refine_clearance", return_value=None) as polish, \
                patch.object(planner, "_rollout", return_value=rollout), \
                patch.object(planner, "_recovery_policies") as long_seeds:
            result = planner._recover_policy(observation, traffic, planner.nominal_traffic(traffic), grid, blocked)
        self.assertIsNone(result)
        long_seeds.assert_not_called()
        self.assertEqual(polish.call_args.kwargs["max_iterations"], 15)
        self.assertTrue(polish.call_args.kwargs["keep_best_iterate"])
        self.assertIn("no full-clearance sequence", planner.recovery_search_profile["recovery_search_outcome"])

    def test_positive_full_reserve_sequence_is_not_rejected_by_ranking_headroom(self):
        from world.ais_fast_recovery import FeedbackFamily, FastRecoverySeed

        planner, observation, grid, blocked, traffic, rollout = self._recovery_context()
        traffic = replace(traffic, clearance_radii_metres=np.array([99.]))
        fast = FastRecoverySeed(FeedbackFamily(90, 90, -15, .5, 2.5, 0),
                                np.zeros((1, 2)), np.zeros((2, 9)), .75)
        with patch("world.ais_buffer_maneuver._local_transition", return_value=object()), \
                patch("world.ais_fast_recovery.search_feedback", return_value=(fast, {})), \
                patch("world.ais_clearance_refinement.refine_clearance", return_value=None), \
                patch.object(planner, "_rollout", return_value=rollout):
            result = planner._recover_policy(observation, traffic, planner.nominal_traffic(traffic), grid, blocked)
        self.assertIsNotNone(result)
        self.assertGreater(result[1].predicted_margin_metres, 0)
        self.assertLess(result[1].predicted_margin_metres, 6)

    def test_recovery_suspends_for_swept_encounter_and_active_colreg(self) -> None:
        start = BoatState(40.225, 18.746666667, heading=90, speed=0)
        target, = SyntheticAIS("crossing", start.latitude, start.longitude).load().sample(0)
        observation = WorldObservation(
            time=np.datetime64("2026-09-30T03:00:00"), boat=start,
            wind=None, wave=None, current=None, seabed=None, coast=None, chart=None,
            ais_targets=(target,),
        )
        grid = OccupancyGrid(span=820, resolution=20).build(start, observation)
        planner = AISBufferPlanner(Fossen3DOFSaildroneDynamics(), SaildroneMPC(), 5.0, 45.0)
        prediction = replace(
            AISPrediction.from_observation(observation), positions_metres=np.array([[900.0, 0.0]]),
        )
        outbound = replace(prediction, velocities_mps=np.array([[5.0, 0.0]]))
        crossing = replace(prediction, velocities_mps=np.array([[-5.0, 0.0]]))
        self.assertFalse(planner.blocks_open_water_recovery(grid, outbound, False))
        self.assertTrue(planner.blocks_open_water_recovery(grid, outbound, True))
        self.assertTrue(planner.blocks_open_water_recovery(grid, crossing, False))
        self.assertFalse(planner.blocks_open_water_recovery(grid, crossing, False, candidate_handoff=True))
        self.assertTrue(planner.blocks_open_water_recovery(grid, crossing, True, candidate_handoff=True))
        planner.escape_seed = np.zeros((1, 2))
        self.assertTrue(planner.blocks_open_water_recovery(grid, outbound, False, candidate_handoff=True))
        planner.escape_seed = None
        no_contacts = AISPrediction.from_observation(replace(observation, ais_targets=()))
        self.assertFalse(planner.blocks_open_water_recovery(grid, no_contacts, False))
        stagnant = replace(prediction, velocities_mps=np.zeros((1, 2)), acceleration_bound_mps2=0.026)
        self.assertTrue(planner.blocks_open_water_recovery(grid, stagnant.for_preview(35, 5), False))

    def test_active_plan_reconsideration_requires_fresh_revised_course(self) -> None:
        start = BoatState(40.225, 18.746666667, heading=90, speed=0)
        target, = SyntheticAIS("crossing", start.latitude, start.longitude).load().sample(0)
        plant = Fossen3DOFSaildroneDynamics()
        planner = AISBufferPlanner(plant, SaildroneMPC(), 5.0, 45.0)

        def prediction(age, course, seconds=0, speed=target.speed):
            observation = WorldObservation(
                time=np.datetime64("2026-09-30T03:00:00") + np.timedelta64(seconds, "s"),
                boat=start, wind=None, wave=None, current=None,
                seabed=None, coast=None, chart=None,
                ais_targets=(replace(target, report_age_seconds=age, course=course, speed=speed),),
            )
            return AISPrediction.from_observation(observation)

        self.assertFalse(planner._report_course_revised(prediction(40, 359)))
        self.assertFalse(planner._report_course_revised(prediction(45, 359)))
        self.assertTrue(planner._report_course_revised(prediction(1, 2)))
        self.assertFalse(planner._report_course_revised(prediction(6, 2)))
        self.assertFalse(planner._report_course_revised(prediction(1, 2.5)))
        self.assertTrue(planner._report_course_revised(prediction(1, 2.5, 5, target.speed + 0.1)))

    @unittest.skipUnless(find_spec("casadi"), "CasADi accelerates the full maneuver search")
    def test_mixed_traffic_plans_from_launch_and_keeps_the_passing_maneuver(self) -> None:
        self._check_mixed_maneuver(None, 105)

    @unittest.skipUnless(find_spec("casadi"), "CasADi accelerates the full maneuver search")
    def test_mixed_traffic_clears_both_passes_during_delayed_and_dropped_reports(self) -> None:
        self._check_mixed_maneuver(SyntheticAISReporting(5, 10, ((85, 105), (115, 135))), 145)

    def _check_mixed_maneuver(self, reporting: SyntheticAISReporting | None, end: int) -> None:
        start = BoatState(40.225, 18.746666667, heading=90, speed=0, flap=-10)
        traffic = SyntheticAIS("mixed", start.latitude, start.longitude, reporting=reporting).load()
        plant = Fossen3DOFSaildroneDynamics()
        planner = AISBufferPlanner(
            plant, SaildroneMPC(), 5.0, 45.0,
            reports_refreshed_each_step=traffic.reports_refreshed_each_step,
        )
        grid_builder = OccupancyGrid(span=820, resolution=20)
        chart = np.zeros((41, 41), dtype=bool)

        def observe(boat, seconds):
            return WorldObservation(
                time=np.datetime64("2026-09-30T03:00:00") + np.timedelta64(seconds, "s"),
                boat=boat, wind=Wind(0, 8, 8, 180), wave=None, current=None,
                seabed=None, coast=None, chart=None, ais_targets=tuple(traffic.sample(seconds)),
            )

        state = start
        first_policy = None
        margins = []
        for seconds in range(0, end, 5):
            observation = observe(state, seconds)
            grid = grid_builder.build(state, observation)
            prediction = AISPrediction.from_observation(observation)
            maneuver = planner.choose(observation, grid, chart, prediction, 90, seconds == 0)
            self.assertIsNotNone(maneuver)
            if seconds == 0:
                first_policy = planner.policy
                remote = replace(prediction, positions_metres=prediction.positions_metres + 20_000.0)
                self.assertTrue(planner.blocks_open_water_recovery(grid, remote, False))
                self.assertTrue(planner.blocks_open_water_recovery(grid, remote, False, candidate_handoff=True))
                if reporting is None:
                    self.assertGreater(maneuver.refreshed_margin_metres, 0.1)
                else:
                    self.assertIsNone(maneuver.refreshed_margin_metres)
                self.assertLess(maneuver.predicted_margin_metres, 0.0)
            self.assertIs(planner.policy, first_policy)
            self.assertLessEqual(abs(maneuver.action.rudder - state.rudder), 7.5 + 1e-10)
            self.assertLessEqual(abs(maneuver.action.flap - state.flap), 5.0 + 1e-10)
            chart_clear, swept = assess_action(plant, state, observation, maneuver.action, 5, grid, chart, prediction)
            self.assertTrue(chart_clear and swept.safe)
            margins.append(swept.minimum_clearance_metres)
            state = plant.step(state, maneuver.action, observation, 5.0)
        self.assertGreater(min(margins), 0.15)
