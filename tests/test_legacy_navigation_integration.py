"""Smoke tests for imported legacy planners with Maritime World interfaces."""

from __future__ import annotations

import json
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
import torch

from world import AISTarget, BoatState, HarmonicPlan, LegacyPlannerAdapter, LocalGrid, MaritimeWorld, Route, Waypoint, WorldObservation
from world._geo import offset_latitude_longitude
from world import run_legacy_fossen as runner
from world.currents import Current
from world.drift_guidance import DriftMode, gate_velocity
from world.fossen import Fossen3DOFSaildroneDynamics
from world.forecast_hold import ForecastHoldPlanner
from world.forecast_tacks import ForecastTackPlanner, ForecastTackReference
from world.local_map import LegacyLocalMap
from world.run_legacy_fossen import mission_goal_reached, parser
from world.temporal_recovery import RecoveryHoldModel, TemporalRecoveryGuidance
from world.waves import Wave
from world.weather import Wind


class LegacyNavigationIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.latitude = 39.9
        self.longitude = 18.4
        channels = (
            "ais_occupied",
            "depth",
            "coast_distance",
            "wind_u",
            "wind_v",
            "wave_height",
            "current_u",
            "current_v",
            "goal",
            "restricted",
        )
        self.grid = LocalGrid(
            data=np.zeros((10, 41, 41), dtype=np.float32),
            channels=channels,
            resolution=115.0,
            span=4715.0,
            origin_latitude=self.latitude,
            origin_longitude=self.longitude,
        )
        state = BoatState(self.latitude, self.longitude, heading=0.0, speed=1.0)
        self.observation = WorldObservation(
            time=np.datetime64("2026-07-29T06:00:00"),
            boat=state,
            wind=Wind(0.0, 8.0, 8.0, 180.0),
            wave=None,
            current=None,
            seabed=None,
            coast=None,
            chart=None,
            ais_targets=(),
        )
        goal_lat, goal_lon = offset_latitude_longitude(
            self.latitude, self.longitude, 1500.0, 0.0
        )
        self.goal = Waypoint(float(goal_lat), float(goal_lon))
        self.local_map = LegacyLocalMap(
            tensor=self.grid,
            static_blocked=np.zeros((41, 41), dtype=bool),
        )

    def test_all_three_legacy_planners_return_current_harmonic_plan_contract(self) -> None:
        for mode in ("standard", "gamma", "diffusion"):
            with self.subTest(mode=mode):
                plan = LegacyPlannerAdapter(mode=mode).plan(
                    self.local_map, self.observation, self.goal
                )
                self.assertEqual(plan.potential.shape, (41, 41))
                self.assertEqual(plan.blocked.shape, (41, 41))
                self.assertTrue(plan.has_path)
                self.assertIsNotNone(plan.desired_heading)

    def test_static_global_obstacle_plane_is_respected(self) -> None:
        blocked = np.zeros((41, 41), dtype=bool)
        blocked[20, 23] = True
        local_map = LegacyLocalMap(self.grid, blocked)
        plan = LegacyPlannerAdapter(mode="gamma").plan(
            local_map, self.observation, self.goal
        )
        self.assertTrue(plan.blocked[20, 23])

    def test_open_water_reference_separates_ground_track_and_bow_under_drift(self) -> None:
        north = Waypoint(*offset_latitude_longitude(self.latitude, self.longitude, 0.0, 460.0))
        observation = replace(
            self.observation, boat=replace(self.observation.boat, heading=85.0),
            current=Current(0.25, 0.0, 0.25, 90.0, 0.0),
        )
        planner = LegacyPlannerAdapter(fossen_dynamics=Fossen3DOFSaildroneDynamics())

        def polar(sample, dynamics, offsets):
            velocities = np.full((len(offsets), 2), np.nan)
            velocities[0] = (0.20, 0.25)
            velocities[2] = (0.0, 0.55)
            return velocities, (sample.wind.direction + offsets) % 360.0, np.array([0.25, 0.0])

        with patch("world.legacy_planner_adapter.steady_ground_polar", side_effect=polar) as solve:
            reference = planner.plan(self.local_map, observation, north)
            self.assertAlmostEqual(reference.desired_heading, 0.0)
            self.assertAlmostEqual(reference.desired_bow_heading, 85.0)
            self.assertIsNone(reference.forecast_closest_metres)
            self.assertGreater(reference.navigation_target.latitude, self.latitude)
            self.assertAlmostEqual(reference.navigation_target.longitude, self.longitude)
            weak = replace(observation, current=Current(0.02, 0.0, 0.02, 90.0, 0.0))
            reference = planner.plan(self.local_map, weak, north)
            self.assertIsNone(reference.desired_bow_heading)
            self.assertAlmostEqual(reference.desired_heading, 0.0)
        solve.assert_called_once()

    def test_open_water_fallback_avoids_fastest_departure_from_unreachable_waypoint(self) -> None:
        observation = replace(
            self.observation, boat=replace(self.observation.boat, heading=20.0),
            current=Current(-0.25, 0.0, 0.25, 270.0, 0.0),
        )
        planner = LegacyPlannerAdapter(fossen_dynamics=Fossen3DOFSaildroneDynamics())

        def polar(sample, dynamics, offsets):
            velocities = np.full((len(offsets), 2), np.nan)
            velocities[0] = (-0.40, 0.0)
            velocities[2] = (-0.04, 0.0)
            return velocities, (sample.wind.direction + offsets) % 360.0, np.array([-0.25, 0.0])

        with patch("world.legacy_planner_adapter.steady_ground_polar", side_effect=polar):
            reference = planner.plan(self.local_map, observation, self.goal)
        self.assertAlmostEqual(reference.desired_heading, 270.0)
        self.assertAlmostEqual(reference.desired_bow_heading, 85.0)

    def test_closest_waypoint_layline_respects_chart_and_ais_handoff(self) -> None:
        goal = Waypoint(*offset_latitude_longitude(self.latitude, self.longitude, 140., 400.))
        observation = replace(
            self.observation, wind=Wind(0., -8., 8., 0.),
            boat=replace(self.observation.boat, heading=314.5),
            current=Current(0., 0., 0., 0., 0.),
        )
        planner = LegacyPlannerAdapter(fossen_dynamics=Fossen3DOFSaildroneDynamics(),
                                       closest_waypoint_layline=True)

        def polar(sample, dynamics, offsets):
            velocities = np.full((len(offsets), 2), np.nan)
            velocities[len(offsets) // 2 - 1] = (-.4, .4)
            velocities[len(offsets) // 2] = (.6, .4)
            return velocities, offsets % 360., np.zeros(2)

        with patch("world.legacy_planner_adapter.steady_ground_polar", side_effect=polar):
            plan = planner.plan(self.local_map, observation, goal, time_aware_ais=True,
                                suspend_recovery=False)
            self.assertAlmostEqual(plan.desired_heading, 315.)
            self.assertAlmostEqual(plan.desired_bow_heading, 314.5)
            self.assertEqual(planner.waypoint_layline_telemetry["waypoint_layline_source"], "closest reachable tack")
            self.assertAlmostEqual(planner.waypoint_layline_telemetry["waypoint_layline_second_bow_deg"], 45.5)

            planner.plan(self.local_map, observation, goal, time_aware_ais=True, suspend_recovery=True)
            self.assertEqual(planner.waypoint_layline_telemetry, {})

            blocked = self.local_map.static_blocked.copy()
            blocked[19, 19] = True
            plan = planner.plan(LegacyLocalMap(self.grid, blocked), observation, goal,
                                time_aware_ais=True, suspend_recovery=False)
            self.assertAlmostEqual(plan.desired_bow_heading, 45.5)

            forecast = ForecastTackPlanner(None, planner.fossen_dynamics, 45., 50.)
            planner.forecast_tacks = forecast
            with patch.object(forecast, "reference", return_value=ForecastTackReference(55., 40., 42., None)):
                plan = planner.plan(self.local_map, observation, goal, time_aware_ais=True,
                                    suspend_recovery=False)
                self.assertAlmostEqual(plan.desired_bow_heading, 40.)
                self.assertEqual(planner.waypoint_layline_telemetry, {})

    def test_layline_priority_can_replace_hold_only_when_admitted_and_feasible(self) -> None:
        goal = Waypoint(*offset_latitude_longitude(self.latitude, self.longitude, 140., 400.))
        observation = replace(
            self.observation, wind=Wind(0., -8., 8., 0.),
            boat=replace(self.observation.boat, heading=314.5),
            current=Current(0., 0., 0., 0., 0.),
        )
        dynamics = Fossen3DOFSaildroneDynamics()
        forecast = ForecastTackPlanner(None, dynamics, 45., 50.)
        planner = LegacyPlannerAdapter(fossen_dynamics=dynamics, limit_drift=True,
                                       forecast_tacks=forecast, prioritize_waypoint_layline=True)
        planner._drift_guidance.mode = "holding"
        velocities = np.full((14, 2), np.nan)
        velocities[6:8] = [(-.4, .4), (.6, .4)]
        bows = (np.array((-160., -125., -95., -75., -60., -50., -45.5,
                          45.5, 50., 60., 75., 95., 125., 160.)) % 360.)
        weak_trim = DriftMode(np.array([0., -.1]), 314.5, 0., -15.)
        with (
            patch("world.legacy_planner_adapter.steady_ground_polar",
                  return_value=(velocities, bows, np.zeros(2))) as polar,
            patch.object(forecast, "reference", return_value=ForecastTackReference(55., 40., 42., None)) as reference,
            patch.object(planner._drift_guidance, "_steady_modes", return_value=[weak_trim]),
        ):
            plan = planner.plan(self.local_map, observation, goal, time_aware_ais=True, suspend_recovery=True)
            self.assertEqual(planner.waypoint_layline_telemetry, {})
            polar.assert_not_called()

            planner._drift_guidance.mode = "holding"
            plan = planner.plan(self.local_map, observation, goal, time_aware_ais=True, suspend_recovery=False)
            self.assertAlmostEqual(plan.desired_heading, 315.)
            self.assertIsNone(plan.drift_mode)
            self.assertIsNone(plan.forecast_closest_metres)
            self.assertEqual(planner.waypoint_layline_telemetry["waypoint_layline_policy"], "priority")
            reference.assert_not_called()

            weak = velocities.copy()
            weak[6:8, 1] = -.1
            polar.return_value = weak, bows, np.zeros(2)
            later = replace(observation, time=observation.time + np.timedelta64(120, "s"))
            plan = planner.plan(self.local_map, later, goal, time_aware_ais=True, suspend_recovery=False)
            self.assertEqual(plan.drift_mode, "holding")
            self.assertEqual(planner.waypoint_layline_telemetry, {})

    def test_drift_gate_distinguishes_radial_motion_from_sustainable_progress(self) -> None:
        velocity = np.array([[0.45, -0.10], [-0.15, -0.20]])
        target = np.array([300.0, 400.0])
        self.assertGreater(float(velocity[0] @ target), 0.0)
        self.assertLess(gate_velocity(velocity, target), 0.0)

    def test_closest_layline_eta_needs_two_comparable_etas_and_a_strict_win(self) -> None:
        goal = Waypoint(*offset_latitude_longitude(self.latitude, self.longitude, 140., 400.))
        observation = replace(
            self.observation, wind=Wind(0., -8., 8., 0.),
            boat=replace(self.observation.boat, heading=314.5), current=Current(0., 0., 0., 0., 0.),
        )
        dynamics = Fossen3DOFSaildroneDynamics()
        forecast = ForecastTackPlanner(None, dynamics, 45., 50.)
        planner = LegacyPlannerAdapter(fossen_dynamics=dynamics, forecast_tacks=forecast,
                                       compare_waypoint_layline_eta=True)
        velocities = np.full((14, 2), np.nan)
        velocities[6:8] = [(-.4, .4), (.6, .4)]
        with (
            patch("world.legacy_planner_adapter.steady_ground_polar",
                  side_effect=lambda sample, dynamics, offsets: (velocities, offsets % 360., np.zeros(2))),
            patch.object(forecast, "reference", return_value=ForecastTackReference(55., 40., 42., None)),
            patch.object(forecast, "compare_layline_arrival") as compare,
            patch.object(forecast, "turn_in_progress", return_value=False) as in_turn,
        ):
            for etas in [(400., 400.), (400., 600.), (400., None), (None, 300.)]:
                compare.return_value = etas
                plan = planner.plan(self.local_map, observation, goal, time_aware_ais=True, suspend_recovery=False)
                self.assertEqual(plan.desired_bow_heading, 40.)
                self.assertEqual(planner.waypoint_layline_telemetry, {})
            compare.return_value = (400., 380.)
            plan = planner.plan(self.local_map, observation, goal, time_aware_ais=True, suspend_recovery=False)
            self.assertEqual(plan.desired_bow_heading, 314.5)
            self.assertEqual(planner.waypoint_layline_telemetry["waypoint_layline_eta_advantage_s"], 20.)
            self.assertEqual(planner.waypoint_layline_telemetry["waypoint_layline_comparison_radius_m"], 50.)

            compare.reset_mock()
            in_turn.return_value = True
            plan = planner.plan(self.local_map, observation, goal, time_aware_ais=True, suspend_recovery=False)
            self.assertEqual(plan.desired_bow_heading, 40.)
            compare.assert_not_called()

    def test_drift_limiter_checks_chart_and_releases_to_an_off_grid_waypoint(self) -> None:
        goal = Waypoint(*offset_latitude_longitude(self.latitude, self.longitude, 0.0, 2600.0))
        current = Current(0.20, -0.25, 0.32, 141.0, 0.0)
        observation = replace(self.observation, current=current)
        planner = LegacyPlannerAdapter(fossen_dynamics=Fossen3DOFSaildroneDynamics(), limit_drift=True)
        weak = [
            DriftMode(np.array([0.20, -0.10]), 55.0, 10.0, 15.0),
            DriftMode(np.array([-0.04, -0.09]), 305.0, -9.0, -15.0),
        ]
        strong = [DriftMode(np.array([0.0, 0.30]), 305.0, -10.0, -15.0)]
        with patch("world.drift_guidance.DriftGuidance._steady_modes", side_effect=[weak, strong]):
            first = planner.plan(self.local_map, observation, goal)
            self.assertEqual(first.drift_mode, "holding")
            self.assertEqual(first.desired_bow_heading, 305.0)
            self.assertEqual(first.steady_rudder_degrees, -9.0)
            self.assertAlmostEqual(first.drift_range_m, 0.0)
            self.assertTrue(first.target_on_boundary)
            updated = replace(observation, time=observation.time + np.timedelta64(300, "s"))
            with patch.object(planner, "_steady_open_water_reference", return_value=(0.0, 305.0, goal)) as solve:
                second = planner.plan(self.local_map, updated, goal)
            self.assertEqual(second.drift_mode, "recovering")
            self.assertGreater(second.gate_vmg_mps, 0.04)
            self.assertIsNone(second.steady_rudder_degrees)
            self.assertEqual(second.desired_bow_heading, 305.0)
            self.assertEqual(solve.call_args.kwargs["recovery_vmg"], second.gate_vmg_mps)
            self.assertIsNone(solve.call_args.kwargs["recovery_goal"])

        blocked = np.zeros((41, 41), dtype=bool)
        blocked[18, 20] = True
        local_map = LegacyLocalMap(self.grid, blocked)
        with patch("world.drift_guidance.DriftGuidance._steady_modes") as forbidden:
            reference = planner.plan(local_map, observation, goal)
            self.assertIsNone(reference.drift_mode)
        forbidden.assert_not_called()

    def test_recovery_keeps_tack_until_the_opposite_ray_reaches_layline(self) -> None:
        planner = LegacyPlannerAdapter(
            fossen_dynamics=Fossen3DOFSaildroneDynamics(), limit_drift=True, commit_recovery_tacks=True,
        )
        observation = replace(
            self.observation, boat=replace(self.observation.boat, heading=135.0),
            current=Current(0.0, 0.0, 0.0, 0.0, 0.0),
        )
        window = int(observation.time.astype("datetime64[s]").astype(np.int64)) // 120
        planner._steady_polar = (
            window, np.array([[0.40, 0.15], [-0.40, 0.15]]),
            np.array([135.0, 225.0]), np.zeros(2),
        )
        planner._recovery_tack = (self.goal, -1, 0.0)
        start, blocked = (20, 20), self.local_map.static_blocked
        reference = planner._steady_open_water_reference(
            observation, np.array([0.0, 400.0]), self.grid, blocked, start,
            recovery_vmg=0.15, recovery_goal=self.goal,
        )
        self.assertEqual(reference[1], 135.0)

        reference = planner._steady_open_water_reference(
            observation, np.array([-200.0, 75.0]), self.grid, blocked, start,
            recovery_vmg=0.15, recovery_goal=self.goal,
        )
        self.assertEqual(reference[1], 225.0)
        self.assertEqual(planner._recovery_tack[1], 1)

        settling = replace(observation, time=observation.time + np.timedelta64(5, "s"))
        reference = planner._steady_open_water_reference(
            settling, np.array([200.0, 75.0]), self.grid, blocked, start,
            recovery_vmg=0.15, recovery_goal=self.goal,
        )
        self.assertEqual(reference[1], 225.0)

    def test_recovery_tack_commitment_extends_to_the_next_waypoint_only_when_enabled(self) -> None:
        current = Current(0.20, -0.25, 0.32, 141.0, 0.0)
        observation = replace(self.observation, current=current)
        strong = [DriftMode(np.array([0.0, 0.30]), 305.0, -10.0, -15.0)]
        for enabled in (False, True):
            with self.subTest(commit_recovery_tacks=enabled):
                planner = LegacyPlannerAdapter(
                    fossen_dynamics=Fossen3DOFSaildroneDynamics(),
                    limit_drift=True, commit_recovery_tacks=enabled,
                )
                planner._drift_guidance.released = True
                goal = Waypoint(*offset_latitude_longitude(self.latitude, self.longitude, 0.0, 260.0))
                with (
                    patch("world.drift_guidance.DriftGuidance._steady_modes", return_value=strong),
                    patch.object(planner, "_steady_open_water_reference", return_value=(0.0, 305.0, goal)) as solve,
                ):
                    plan = planner.plan(self.local_map, observation, goal)
                if enabled:
                    self.assertEqual(plan.drift_mode, "recovering")
                    self.assertEqual(solve.call_args.kwargs["recovery_goal"], goal)
                else:
                    self.assertIsNone(plan.drift_mode)
                    self.assertNotIn("recovery_goal", solve.call_args.kwargs)

    def _temporal_navigation(self, *, confirm_release: bool = False):
        model = RecoveryHoldModel()
        with torch.no_grad():
            model.head[-1].weight.zero_()
            model.head[-1].bias.copy_(torch.tensor([2.0, -2.0]))
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "temporal_hold.pt"
            torch.save({
                "weights": model.state_dict(), "temperature": 1.0, "confidence_threshold": 0.8,
                "scaling": {
                    "history_mean": torch.zeros(5), "history_scale": torch.ones(5),
                    "context_mean": torch.zeros(1), "context_scale": torch.ones(1),
                },
            }, checkpoint)
            policy = TemporalRecoveryGuidance(checkpoint, confirm_release=confirm_release)
        planner = LegacyPlannerAdapter(
            fossen_dynamics=Fossen3DOFSaildroneDynamics(), limit_drift=True,
            commit_recovery_tacks=True, temporal_recovery=policy,
        )
        observation = replace(self.observation, current=Current(0.20, -0.25, 0.32, 141.0, 0.0),
                              wave=Wave(0.5, 160.0, 5.0))
        goal = Waypoint(*offset_latitude_longitude(self.latitude, self.longitude, 0.0, 2600.0))
        return planner, policy, observation, goal

    def test_temporal_policy_can_hold_then_release_before_the_classical_threshold(self) -> None:
        planner, policy, observation, goal = self._temporal_navigation()
        first_modes = [DriftMode(np.array([0.0, 0.06]), 305.0, -9.0, -15.0)]
        next_modes = [DriftMode(np.array([0.0, 0.03]), 305.0, -9.0, -15.0)]
        with (
            patch("world.drift_guidance.DriftGuidance._steady_modes", side_effect=[first_modes, next_modes]),
            patch.object(policy, "_hold_probability", wraps=policy._hold_probability) as predict,
            patch.object(planner, "_steady_open_water_reference", return_value=(0.0, 305.0, goal)) as layline,
        ):
            initial = planner.plan(self.local_map, observation, goal)
            hold = planner.temporal_recovery_telemetry
            self.assertEqual(initial.drift_mode, "holding")
            self.assertEqual(initial.steady_flap_degrees, -15.0)
            self.assertEqual(hold["temporal_recovery_raw_action"], "hold")
            self.assertEqual(hold["temporal_recovery_applied_action"], "hold")
            self.assertEqual(hold["temporal_recovery_classical_action"], "sail")
            self.assertEqual(hold["temporal_recovery_source"], "learned")
            self.assertEqual(hold["temporal_recovery_window_utc"], "2026-07-29T06:00:00Z")
            with torch.no_grad():
                policy.model.head[-1].bias.mul_(-1)
            middle = replace(observation, time=observation.time + np.timedelta64(5, "s"))
            self.assertEqual(planner.plan(self.local_map, middle, goal).drift_mode, "holding")
            predict.assert_called_once()
            next_window = replace(observation, time=observation.time + np.timedelta64(300, "s"))
            released = planner.plan(self.local_map, next_window, goal)
            sail = planner.temporal_recovery_telemetry
            self.assertEqual(released.drift_mode, "recovering")
            self.assertIsNone(released.steady_flap_degrees)
            self.assertEqual(sail["temporal_recovery_raw_action"], "sail")
            self.assertEqual(sail["temporal_recovery_applied_action"], "sail")
            self.assertEqual(sail["temporal_recovery_classical_action"], "hold")
            self.assertEqual(sail["temporal_recovery_source"], "learned")
            self.assertEqual(predict.call_count, 2)
            layline.assert_called_once()

    def test_temporal_policy_vetoes_release_with_negative_gate_progress(self) -> None:
        planner, policy, observation, goal = self._temporal_navigation()
        modes = [DriftMode(np.array([0.0, -0.05]), 305.0, -9.0, -15.0)]
        with patch("world.drift_guidance.DriftGuidance._steady_modes", return_value=modes):
            initial = planner.plan(self.local_map, observation, goal)
            self.assertEqual(initial.drift_mode, "holding")
            with torch.no_grad():
                policy.model.head[-1].bias.mul_(-1)
            next_window = replace(observation, time=observation.time + np.timedelta64(300, "s"))
            vetoed = planner.plan(self.local_map, next_window, goal)
            self.assertEqual(vetoed.drift_mode, "holding")
        result = planner.temporal_recovery_telemetry
        self.assertEqual(result["temporal_recovery_raw_action"], "sail")
        self.assertEqual(result["temporal_recovery_applied_action"], "hold")
        self.assertEqual(result["temporal_recovery_source"], "classical fallback")
        self.assertEqual(result["temporal_recovery_fallback_reason"], "no physically viable gate approach")

    def test_temporal_policy_vetoes_a_hold_trim_whose_local_track_is_blocked(self) -> None:
        planner, _, observation, goal = self._temporal_navigation()
        modes = [DriftMode(np.array([0.06, 0.0]), 305.0, -9.0, -15.0)]
        blocked = np.zeros((41, 41), dtype=bool)
        blocked[20, 21] = True
        local_map = LegacyLocalMap(self.grid, blocked)
        with patch("world.drift_guidance.DriftGuidance._steady_modes", return_value=modes):
            vetoed = planner.plan(local_map, observation, goal)
        self.assertIsNone(vetoed.drift_mode)
        self.assertIsNone(vetoed.steady_flap_degrees)
        self.assertNotIn((20, 21), vetoed.path_cells)
        result = planner.temporal_recovery_telemetry
        self.assertEqual(result["temporal_recovery_raw_action"], "hold")
        self.assertEqual(result["temporal_recovery_applied_action"], "planner")
        self.assertEqual(result["temporal_recovery_source"], "classical fallback")
        self.assertEqual(result["temporal_recovery_fallback_reason"], "no chart-clear hold trim")

    def test_temporal_release_confirmation_requires_consecutive_windows_and_prevents_chatter(self) -> None:
        planner, _, observation, goal = self._temporal_navigation(confirm_release=True)
        modes = [[DriftMode(np.array([0.0, vmg]), 305.0, -9.0, -15.0)]
                 for vmg in (-0.02, 0.05, 0.05, 0.055, 0.03)]
        with (
            patch("world.drift_guidance.DriftGuidance._steady_modes", side_effect=modes),
            patch.object(planner, "_steady_open_water_reference", return_value=(0.0, 305.0, goal)),
        ):
            for elapsed, expected_mode in ((0, "holding"), (300, "holding"), (900, "holding"),
                                           (1200, "recovering"), (1500, "recovering")):
                with self.subTest(elapsed_seconds=elapsed):
                    updated = replace(observation, time=observation.time + np.timedelta64(elapsed, "s"))
                    plan = planner.plan(self.local_map, updated, goal)
                    self.assertEqual(plan.drift_mode, expected_mode)
                    result = planner.temporal_recovery_telemetry
                    self.assertEqual(result["temporal_recovery_raw_action"], "hold")
                    if expected_mode == "recovering":
                        self.assertEqual(result["temporal_recovery_applied_action"], "sail")
                        self.assertEqual(result["temporal_recovery_source"], "physical release")
                        guard = "sustained feasible release" if elapsed == 1200 else "confirmed release remains feasible"
                        self.assertEqual(result["temporal_recovery_release_guard"], guard)
                    else:
                        self.assertEqual(result["temporal_recovery_applied_action"], "hold")
                        self.assertEqual(result["temporal_recovery_source"], "learned")

    def test_temporal_release_confirmation_resets_on_goal_change_and_lost_progress(self) -> None:
        _, policy, observation, goal = self._temporal_navigation(confirm_release=True)
        other = Waypoint(goal.latitude + 0.001, goal.longitude)
        target = np.array([0.0, 2600.0])
        for elapsed, active, gate, was_holding, expected_source in (
            (0, goal, 0.05, True, "learned"),
            (300, other, 0.05, True, "learned"),
            (600, other, 0.05, True, "physical release"),
            (605, goal, 0.05, True, "learned"),
            (900, goal, -0.02, False, "learned"),
            (1200, goal, 0.05, True, "learned"),
            (1500, goal, 0.05, True, "physical release"),
        ):
            with self.subTest(elapsed_seconds=elapsed):
                updated = replace(observation, time=observation.time + np.timedelta64(elapsed, "s"))
                choice = policy.decide(updated, target, goal=active, was_holding=was_holding,
                                       classical_hold=gate < (0.04 if was_holding else 0.01),
                                       gate_vmg=gate, hold_available=True)
                self.assertEqual(choice.telemetry()["temporal_recovery_source"], expected_source)
                self.assertEqual(choice.hold, expected_source == "learned")

    def test_temporal_release_confirmation_defaults_off(self) -> None:
        planner, _, observation, goal = self._temporal_navigation()
        modes = [DriftMode(np.array([0.0, 0.05]), 305.0, -9.0, -15.0)]
        with patch("world.drift_guidance.DriftGuidance._steady_modes", return_value=modes):
            for elapsed in (0, 300, 600):
                updated = replace(observation, time=observation.time + np.timedelta64(elapsed, "s"))
                self.assertEqual(planner.plan(self.local_map, updated, goal).drift_mode, "holding")
                result = planner.temporal_recovery_telemetry
                self.assertEqual(result["temporal_recovery_classical_action"], "sail")
                self.assertEqual(result["temporal_recovery_applied_action"], "hold")
                self.assertEqual(result["temporal_recovery_source"], "learned")

    def _temporal_forecast_navigation(self, *, sail: bool = False):
        _, policy, observation, _ = self._temporal_navigation()
        if sail:
            with torch.no_grad():
                policy.model.head[-1].bias.mul_(-1)
        dynamics = Fossen3DOFSaildroneDynamics()
        forecast = ForecastTackPlanner(MaritimeWorld(), dynamics, 45.0, 50.0,
                                      tracking_feedback=True, maneuver_commit_seconds=60.0)
        planner = LegacyPlannerAdapter(
            fossen_dynamics=dynamics, limit_drift=True,
            temporal_recovery=policy, forecast_tacks=forecast,
        )
        return planner, policy, forecast, observation

    def test_temporal_hold_takes_priority_then_release_uses_forecast_sailing(self) -> None:
        planner, policy, forecast, observation = self._temporal_forecast_navigation()
        modes = [DriftMode(np.array([0.10, 0.0]), 85.0, 5.0, 15.0)]
        forecast._choice = (1, 2, 10, 0.0)
        forecast._tracking_residuals[1] = np.array([0.02, 0.01])
        with (
            patch("world.drift_guidance.DriftGuidance._steady_modes", return_value=modes),
            patch.object(forecast, "reference", return_value=ForecastTackReference(90.0, 85.0, 0.0, None)) as sailing,
        ):
            held = planner.plan(self.local_map, observation, self.goal)
            self.assertEqual(held.drift_mode, "holding")
            self.assertEqual(held.steady_flap_degrees, 15.0)
            self.assertIsNone(held.forecast_closest_metres)
            sailing.assert_not_called()
            self.assertIsNone(forecast._choice)
            self.assertFalse(forecast._tracking_residuals)
            self.assertEqual(forecast.tracking_telemetry["forecast_suspension_reason"], "holding trim selected")
            self.assertEqual(planner.temporal_recovery_telemetry["temporal_recovery_applied_action"], "hold")

            with torch.no_grad():
                policy.model.head[-1].bias.mul_(-1)
            resumed = replace(observation, time=observation.time + np.timedelta64(300, "s"))
            released = planner.plan(self.local_map, resumed, self.goal)
            self.assertEqual(released.drift_mode, "recovering")
            self.assertIsNone(released.steady_flap_degrees)
            self.assertEqual(released.forecast_closest_metres, 0.0)
            self.assertEqual(released.desired_bow_heading, 85.0)
            self.assertEqual(planner.temporal_recovery_telemetry["temporal_recovery_applied_action"], "sail")
            sailing.assert_called_once_with(resumed, self.goal)

    def test_ais_defers_both_temporal_and_forecast_and_resumes_from_current_boat(self) -> None:
        planner, policy, forecast, observation = self._temporal_forecast_navigation(sail=True)
        target = AISTarget(100000001, self.latitude, self.longitude + 0.1, 5.0, 180.0, 180.0)
        observed = replace(observation, ais_targets=(target,))
        modes = [DriftMode(np.array([0.10, 0.0]), 85.0, 5.0, 15.0)]
        forecast._tracking_reference = 85.0
        forecast._tracking_residuals[1] = np.array([0.02, -0.03])
        with (
            patch("world.drift_guidance.DriftGuidance._steady_modes", return_value=modes),
            patch.object(policy, "decide", wraps=policy.decide) as learned,
            patch.object(policy, "reset_release_confirmation", wraps=policy.reset_release_confirmation) as confirmation,
            patch.object(forecast, "reference", return_value=ForecastTackReference(90.0, 85.0, 0.0, None)) as sailing,
        ):
            blocked = planner.plan(self.local_map, observed, self.goal,
                                   time_aware_ais=True, suspend_recovery=True)
            self.assertIsNone(blocked.forecast_closest_metres)
            learned.assert_not_called()
            sailing.assert_not_called()
            confirmation.assert_called()
            self.assertFalse(forecast._tracking_residuals)
            self.assertIsNone(forecast._tracking_reference)
            self.assertEqual(planner.temporal_recovery_telemetry["temporal_recovery_fallback_reason"], "AIS encounter")
            clear = replace(observed, time=observed.time + np.timedelta64(5, "s"))
            resumed = planner.plan(self.local_map, clear, self.goal,
                                   time_aware_ais=True, suspend_recovery=False)
            self.assertEqual(resumed.forecast_closest_metres, 0.0)
            learned.assert_called_once()
            sailing.assert_called_once_with(clear, self.goal)

    def test_chart_rejection_discards_unapplied_forecast_history(self) -> None:
        _, _, forecast, observation = self._temporal_forecast_navigation(sail=True)
        planner = LegacyPlannerAdapter(fossen_dynamics=forecast.dynamics, forecast_tacks=forecast)
        blocked = self.local_map.static_blocked.copy()
        blocked[19, 20] = True  # Block the proposed north ray, leaving the eastward GPX approach clear.
        forecast._tracking_reference = 0.0
        forecast._tracking_residuals[1] = np.array([0.01, 0.01])
        with patch.object(forecast, "reference", return_value=ForecastTackReference(0.0, 355.0, 0.0, None)):
            plan = planner.plan(LegacyLocalMap(self.grid, blocked), observation, self.goal)
        self.assertIsNone(plan.forecast_closest_metres)
        self.assertFalse(forecast._tracking_residuals)
        self.assertIsNone(forecast._tracking_reference)
        self.assertEqual(forecast.tracking_telemetry["forecast_suspension_reason"],
                         "forecast reference unavailable or chart rejected")

    def test_late_actuator_override_invalidates_forecast_and_learned_applied_claim(self) -> None:
        planner, _, forecast, _ = self._temporal_forecast_navigation(sail=True)
        planner.temporal_recovery_telemetry = {
            "temporal_recovery_raw_action": "sail",
            "temporal_recovery_p_hold": 0.02,
            "temporal_recovery_applied_action": "sail",
            "temporal_recovery_source": "learned",
        }
        forecast._tracking_reference = 85.0
        forecast._tracking_residuals[1] = np.array([0.02, 0.01])
        forecast._choice = (1, 2, 10, 0.0)
        proposed = {
            **planner.temporal_recovery_telemetry,
            "bow_reference_deg": 85.0,
            "forecast_closest_m": 0.0,
            "forecast_tracking_residual_east_mps": 0.02,
        }

        runner.suspend_nominal_guidance(planner, proposed, "AIS action fallback")

        self.assertIsNone(forecast._choice)
        self.assertIsNone(forecast._tracking_reference)
        self.assertFalse(forecast._tracking_residuals)
        self.assertNotIn("bow_reference_deg", proposed)
        self.assertNotIn("forecast_closest_m", proposed)
        self.assertNotIn("forecast_tracking_residual_east_mps", proposed)
        self.assertEqual(proposed["temporal_recovery_applied_action"], "planner")
        self.assertEqual(proposed["temporal_recovery_source"], "classical fallback")
        self.assertEqual(proposed["temporal_recovery_raw_action"], "sail")
        self.assertEqual(proposed["temporal_recovery_p_hold"], 0.02)
        self.assertEqual(proposed["nominal_guidance_suspended"], "AIS action fallback")

    def _forecast_holding(self):
        observation = replace(self.observation, boat=replace(self.observation.boat, heading=300.0),
                              current=Current(0.20, -0.25, 0.32, 141.0, 0.0), wave=Wave(0.5, 160.0, 5.0))
        goal = Waypoint(*offset_latitude_longitude(self.latitude, self.longitude, 0.0, 2600.0))
        world = MaritimeWorld(weather=Mock(), currents=Mock(), waves=Mock())
        world.weather.sample.return_value = observation.wind
        world.currents.sample.return_value = observation.current
        world.waves.sample.return_value = observation.wave
        dynamics = Fossen3DOFSaildroneDynamics()
        forecast = ForecastHoldPlanner(world, dynamics, 45.0)
        planner = LegacyPlannerAdapter(fossen_dynamics=dynamics, limit_drift=True, forecast_hold=forecast)
        hold = [DriftMode(np.array([0.12, 0.0]), 60.0, 10.0, 15.0),
                DriftMode(np.array([0.0, -0.02]), 300.0, -9.0, -15.0)]
        strong = [DriftMode(np.array([0.0, 0.35]), 315.0, -5.0, -15.0),
                  DriftMode(np.array([0.2, 0.35]), 45.0, 5.0, 15.0),
                  DriftMode(np.array([-0.2, 0.35]), 320.0, -5.0, -15.0)]
        return planner, forecast, observation, goal, hold, strong

    def test_forecast_hold_uses_archive_weather_only_for_holding_trims(self) -> None:
        planner, forecast, observation, goal, hold, strong = self._forecast_holding()
        with (
            patch.object(forecast._sampler, "_steady_modes", return_value=strong),
            patch("world.drift_guidance.DriftGuidance._steady_modes", side_effect=[hold, strong]),
            patch.object(planner, "_steady_open_water_reference", return_value=(0.0, 315.0, goal)) as layline,
        ):
            initial = planner.plan(self.local_map, observation, goal)
            self.assertEqual(initial.drift_mode, "holding")
            self.assertEqual(initial.desired_bow_heading, hold[1].bow)
            applied = planner.forecast_hold_telemetry
            self.assertEqual(applied["forecast_hold_source"], "archive forecast")
            self.assertTrue(applied["forecast_hold_predicted_release"])
            self.assertAlmostEqual(applied["forecast_hold_gate_vmg_mps"], 0.35)
            self.assertEqual(applied["forecast_hold_end_utc"], "2026-07-29T06:05:00Z")
            forecast.world.weather.sample.assert_called_once_with(
                goal.latitude, goal.longitude, np.datetime64("2026-07-29T06:07:30"),
            )
            following = replace(observation, time=observation.time + np.timedelta64(300, "s"))
            release = planner.plan(self.local_map, following, goal)
            self.assertEqual(release.drift_mode, "recovering")
            self.assertEqual(release.desired_bow_heading, 315.0)
            self.assertIsNone(release.steady_flap_degrees)
            self.assertEqual(planner.forecast_hold_telemetry, {})
            forecast.world.weather.sample.assert_called_once()
            layline.assert_called_once()

    def test_forecast_hold_falls_back_when_a_cached_trim_becomes_blocked(self) -> None:
        planner, forecast, observation, goal, hold, strong = self._forecast_holding()
        with (
            patch.object(forecast._sampler, "_steady_modes", return_value=strong),
            patch("world.drift_guidance.DriftGuidance._steady_modes", return_value=hold),
        ):
            self.assertEqual(planner.plan(self.local_map, observation, goal).desired_bow_heading, hold[1].bow)
            blocked = np.zeros((41, 41), dtype=bool)
            blocked[21, 20] = True
            updated = replace(observation, time=observation.time + np.timedelta64(5, "s"))
            fallback = planner.plan(LegacyLocalMap(self.grid, blocked), updated, goal)
        self.assertEqual(fallback.drift_mode, "holding")
        self.assertEqual(fallback.desired_bow_heading, hold[0].bow)
        self.assertEqual(planner.forecast_hold_telemetry["forecast_hold_source"], "classical fallback")
        self.assertEqual(planner.forecast_hold_telemetry["forecast_hold_fallback_reason"], "planned trim blocked")
        forecast.world.weather.sample.assert_called_once()

    def test_unavailable_hold_forecast_uses_the_default_drift_trim(self) -> None:
        planner, forecast, observation, goal, hold, _ = self._forecast_holding()
        classical = LegacyPlannerAdapter(fossen_dynamics=planner.fossen_dynamics, limit_drift=True)
        forecast.world.weather.sample.side_effect = ValueError("no archive at next timestamp")
        with patch("world.drift_guidance.DriftGuidance._steady_modes", return_value=hold):
            fallback = planner.plan(self.local_map, observation, goal)
            reference = classical.plan(self.local_map, observation, goal)
        self.assertEqual(fallback.desired_heading, reference.desired_heading)
        self.assertEqual(fallback.desired_bow_heading, reference.desired_bow_heading)
        self.assertEqual(fallback.steady_flap_degrees, reference.steady_flap_degrees)
        self.assertEqual(classical.forecast_hold_telemetry, {})
        self.assertEqual(planner.forecast_hold_telemetry["forecast_hold_fallback_reason"], "archive forecast unavailable")

    def test_temporal_recovery_suspends_for_an_ais_encounter_and_renews_its_hold(self) -> None:
        planner, policy, observation, goal = self._temporal_navigation(confirm_release=True)
        target = AISTarget(100000001, self.latitude, self.longitude + 0.1, 5.0, 180.0, 180.0)
        observed = replace(observation, ais_targets=(target,))
        stable = [DriftMode(np.array([0.0, 0.055]), 305.0, -9.0, -15.0)]
        with (
            patch("world.drift_guidance.DriftGuidance._steady_modes", return_value=stable) as polar,
            patch.object(planner, "_steady_open_water_reference", return_value=(0.0, 305.0, goal)) as layline,
            patch.object(policy, "_hold_probability", wraps=policy._hold_probability) as predict,
        ):
            for elapsed, expected_mode in ((0, "holding"), (300, "holding"), (600, "recovering")):
                updated = replace(observed, time=observed.time + np.timedelta64(elapsed, "s"))
                plan = planner.plan(self.local_map, updated, goal, time_aware_ais=True, suspend_recovery=False)
                self.assertEqual(plan.drift_mode, expected_mode)
            self.assertTrue(layline.call_args.kwargs["allow_screened_contacts"])
            self.assertEqual(layline.call_args.args[0].ais_targets, (target,))
            self.assertEqual(planner.temporal_recovery_telemetry["temporal_recovery_source"], "physical release")

            encounter = replace(observed, time=observed.time + np.timedelta64(605, "s"))
            bypassed = planner.plan(self.local_map, encounter, goal, time_aware_ais=True, suspend_recovery=True)
            self.assertIsNone(bypassed.drift_mode)
            self.assertIsNone(bypassed.steady_flap_degrees)
            self.assertEqual(planner.temporal_recovery_telemetry["temporal_recovery_fallback_reason"], "AIS encounter")
            self.assertEqual(predict.call_count, 3)

            moved = replace(observed.boat, latitude=self.latitude + 0.0002)
            moved_grid = replace(self.grid, origin_latitude=moved.latitude)
            moved_map = LegacyLocalMap(moved_grid, self.local_map.static_blocked)
            clear = replace(observed, boat=moved, time=observed.time + np.timedelta64(610, "s"))
            recovered = planner.plan(moved_map, clear, goal, time_aware_ais=True, suspend_recovery=False)
            self.assertEqual(recovered.drift_mode, "holding")
            self.assertEqual(planner.temporal_recovery_telemetry["temporal_recovery_source"], "learned")
            self.assertEqual(planner._drift_guidance.anchor, Waypoint(moved.latitude, moved.longitude))
            self.assertEqual(predict.call_count, 4)
            self.assertEqual(polar.call_count, 4)
            self.assertEqual(len(policy.observed_weather), 3)
            proposal = planner.temporal_recovery_telemetry.copy()
            planner.suspend_open_water_recovery("AIS steering override")
            suspension = planner.temporal_recovery_telemetry
            self.assertEqual(suspension["temporal_recovery_raw_action"], proposal["temporal_recovery_raw_action"])
            self.assertEqual(suspension["temporal_recovery_p_hold"], proposal["temporal_recovery_p_hold"])
            self.assertEqual(suspension["temporal_recovery_applied_action"], "planner")
            self.assertEqual(suspension["temporal_recovery_fallback_reason"], "AIS steering override")

    def test_ais_pause_discards_forecast_trim_but_keeps_archive_samples(self) -> None:
        planner, forecast, observation, goal, hold, strong = self._forecast_holding()
        new_trim = DriftMode(np.array([0.0, -0.04]), 315.0, -8.0, -15.0)
        with (
            patch.object(forecast._sampler, "_steady_modes", return_value=strong),
            patch("world.drift_guidance.DriftGuidance._steady_modes", side_effect=[hold, [new_trim]]),
        ):
            initial = planner.plan(self.local_map, observation, goal, time_aware_ais=True, suspend_recovery=False)
            self.assertEqual(initial.desired_bow_heading, hold[1].bow)
            encounter = replace(observation, time=observation.time + np.timedelta64(5, "s"))
            safety_plan = planner.plan(self.local_map, encounter, goal, time_aware_ais=True)
            self.assertIsNone(safety_plan.drift_mode)
            self.assertEqual(planner.forecast_hold_telemetry["forecast_hold_fallback_reason"], "AIS encounter")

            clear = replace(observation, time=observation.time + np.timedelta64(10, "s"))
            resumed = planner.plan(self.local_map, clear, goal, time_aware_ais=True, suspend_recovery=False)
            self.assertEqual(resumed.drift_mode, "holding")
            self.assertEqual(resumed.desired_bow_heading, new_trim.bow)
            self.assertEqual(planner.forecast_hold_telemetry["forecast_hold_source"], "archive forecast")
            forecast.world.weather.sample.assert_called_once()
            planner.suspend_open_water_recovery("AIS action fallback")
            suspension = planner.forecast_hold_telemetry
            self.assertEqual(suspension["forecast_hold_fallback_reason"], "AIS action fallback")
            self.assertNotIn("forecast_hold_range_at_end_m", suspension)
            self.assertNotIn("forecast_hold_trim_bow_deg", suspension)
            self.assertIn("forecast_hold_window_utc", suspension)

    def test_round_trip_cli_uses_goal_as_turnaround(self) -> None:
        arguments = parser().parse_args(["--return-to-start"])

        self.assertTrue(arguments.return_to_start)

    def test_route_gpx_cli_accepts_named_round_trip(self) -> None:
        arguments = parser().parse_args(
            [
                "--route-gpx",
                "current_route/naples_8_8nm_round_trip.gpx",
                "--route-name",
                "naples_8_8nm",
                "--return-to-start",
            ]
        )

        self.assertEqual(arguments.route_gpx.name, "naples_8_8nm_round_trip.gpx")
        self.assertEqual(arguments.route_name, "naples_8_8nm")
        self.assertTrue(arguments.return_to_start)

    def test_route_endpoint_requires_final_waypoint_to_be_active(self) -> None:
        state = BoatState(
            self.goal.latitude,
            self.goal.longitude,
            heading=0.0,
            speed=0.0,
        )

        self.assertFalse(
            mission_goal_reached(
                state,
                self.goal,
                50.0,
                waypoint_index=1,
                waypoint_count=4,
            )
        )
        self.assertTrue(
            mission_goal_reached(
                state,
                self.goal,
                50.0,
                waypoint_index=3,
                waypoint_count=4,
            )
        )

    def test_round_trip_outcome_records_arrivals_at_the_step_limit(self) -> None:
        route_path = Path(__file__).resolve().parents[1] / "current_route" / "otranto_short_test_loop.gpx"
        route = Route.load_gpx(route_path)
        arrival_points = route.waypoints[1:] + tuple(reversed(route.waypoints[:-1]))
        expected_waypoints = ("002", "003", "004", "005", "004", "003", "002", "001")
        transitions = [
            (BoatState(point.latitude, point.longitude, heading=0.0, speed=1.0), np.zeros(2))
            for point in arrival_points
        ]

        for limit, status, next_waypoint in ((4, "step_limit", "004"), (8, "completed", None)):
            with self.subTest(limit=limit):
                arguments = parser().parse_args(
                    [
                        "--route-gpx", str(route_path),
                        "--return-to-start",
                        "--steps", str(limit),
                        "--plot-every", "0",
                        "--dynamics", "legacy_kinematic",
                    ]
                )
                output = StringIO()
                with (
                    patch.object(runner, "GlobalOccupancyMap"),
                    patch.object(runner, "LegacyLocalMapBuilder") as builder,
                    patch.object(runner, "LegacyPlannerAdapter") as planner,
                    patch.object(runner, "legacy_kinematic_step", side_effect=transitions) as step,
                    redirect_stdout(output),
                ):
                    builder.return_value.build.return_value = self.local_map
                    planner.return_value.plan.return_value.target_cell = (19, 20)
                    planner.return_value.plan.return_value.desired_heading = 0.0
                    trajectory, ais_tracks = runner.run(arguments)

                self.assertEqual(step.call_count, limit)
                self.assertEqual(len(trajectory), limit + 1)
                self.assertEqual(ais_tracks, {})
                lines = output.getvalue().splitlines()
                summary, = [line for line in lines if line.startswith("Mission outcome: ")]
                outcome = json.loads(summary.removeprefix("Mission outcome: "))
                self.assertEqual(outcome["status"], status)
                self.assertEqual(outcome["steps"], limit)
                self.assertEqual(outcome["active_waypoint"], next_waypoint)
                self.assertEqual(outcome["phase"], "return")
                self.assertEqual(outcome["departure_waypoint"], "001")
                self.assertEqual(
                    [
                        (arrival["waypoint"], arrival["phase"], arrival["step"])
                        for arrival in outcome["waypoint_arrivals"]
                    ],
                    [
                        (name, "outbound" if step <= 4 else "return", step)
                        for step, name in enumerate(expected_waypoints[:limit], start=1)
                    ],
                )
                self.assertTrue(all(
                    arrival["distance_metres"] <= outcome["arrival_radius_metres"]
                    for arrival in outcome["waypoint_arrivals"]
                ))
                self.assertIn("Route waypoint 005 reached at step 4.", lines)
                if status == "completed":
                    self.assertIn("Route waypoint 001 reached at step 8.", lines)

    def test_marine_environment_selects_real_ocean_layers(self) -> None:
        arguments = parser().parse_args(["--environment", "marine"])

        self.assertEqual(arguments.environment, "marine")

    def test_synthetic_ais_uses_the_selected_gpx_departure(self) -> None:
        route_path = Path(__file__).resolve().parents[1] / "current_route" / "otranto_short_test_loop.gpx"
        departure = Route.load_gpx(route_path).departure
        arguments = parser().parse_args([
            "--route-gpx", str(route_path), "--ais-scenario", "mixed",
            "--start-latitude", "40.5", "--start-longitude", "18.5",
            "--plot-every", "0", "--steps", "0",
        ])
        with (
            patch.object(runner, "GlobalOccupancyMap"),
            patch.object(runner.SyntheticAIS, "load", autospec=True, side_effect=runner.SyntheticAIS.load) as load,
            redirect_stdout(StringIO()),
        ):
            trajectory, _ = runner.run(arguments)

        ais, = load.call_args.args
        self.assertEqual((ais.origin_latitude, ais.origin_longitude), (departure.latitude, departure.longitude))
        self.assertEqual((trajectory[0].latitude, trajectory[0].longitude), (departure.latitude, departure.longitude))

    def test_time_aware_fallback_advances_physics_or_reports_chart_stop(self) -> None:
        arguments = parser().parse_args([
            "--time-aware-ais", "--steps", "1", "--plot-every", "0", "--dt", "5",
            "--start-latitude", str(self.latitude), "--start-longitude", str(self.longitude),
            "--goal-latitude", str(self.goal.latitude), "--goal-longitude", str(self.goal.longitude),
            "--heading", "90", "--flap=-10",
        ])
        for unblocked, status, steps in ((True, "step_limit", 1), (False, "no_chart_clear_action", 0)):
            with self.subTest(unblocked=unblocked):
                shape = (self.grid.cells, self.grid.cells)
                plan = HarmonicPlan(
                    potential=np.zeros(shape), blocked=np.full(shape, not unblocked),
                    path_cells=((20, 20),), desired_heading=None, target_cell=(20, 23),
                    target_on_boundary=False, converged=False, iterations=0,
                )
                output = StringIO()
                with (
                    patch.object(runner, "GlobalOccupancyMap"),
                    patch.object(runner, "LegacyLocalMapBuilder") as builder,
                    patch.object(runner, "LegacyPlannerAdapter") as planner,
                    redirect_stdout(output),
                ):
                    builder.return_value.build.return_value = self.local_map
                    planner.return_value.plan.return_value = plan
                    trajectory, _ = runner.run(arguments)

                self.assertEqual(len(trajectory), steps + 1)
                line = next(line for line in output.getvalue().splitlines()
                            if line.startswith("Mission outcome: "))
                self.assertEqual(json.loads(line.removeprefix("Mission outcome: "))["status"], status)
                if unblocked:
                    self.assertEqual(trajectory[-1].flap, -5.0)
                    self.assertGreater(trajectory[-1].speed, 0)
                    self.assertIn('"fallback depower": 1', output.getvalue())

    def test_transformer_diffusion_cli_accepts_both_transformer_checkpoints(self) -> None:
        arguments = parser().parse_args(
            [
                "--planner",
                "transformer_diffusion",
                "--transformer-diffusion-checkpoint",
                "own_ship.pt",
                "--ais-course-checkpoint",
                "ais_course.pt",
                "--ais-replay",
                "replay.csv",
                "--live-plot",
            ]
        )

        self.assertEqual(arguments.planner, "transformer_diffusion")
        self.assertEqual(arguments.transformer_diffusion_checkpoint.name, "own_ship.pt")
        self.assertEqual(arguments.ais_course_checkpoint.name, "ais_course.pt")


if __name__ == "__main__":
    unittest.main()
