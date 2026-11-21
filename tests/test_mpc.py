"""Tests for the discrete nonlinear saildrone MPC."""

from __future__ import annotations

from dataclasses import replace
from math import cos, radians, sin
import unittest

import numpy as np

from world import (
    BoatAction, BoatState, Current, Fossen3DOFSaildroneDynamics, HarmonicPlan,
    SaildroneMPC, Wave, Waypoint, Wind, WorldObservation,
)
from world._geo import offset_latitude_longitude
from world.boat import BoatDynamics


class TurningDynamics(BoatDynamics):
    """Small deterministic model used to isolate the MPC optimisation logic."""

    def step(self, state, action, environment, dt):
        heading = (state.heading - action.rudder * dt) % 360.0
        speed = 1.0 + abs(action.flap) / 15.0
        return BoatState(
            state.latitude,
            state.longitude,
            heading=heading,
            speed=speed,
            rudder=action.rudder,
            flap=action.flap,
        )


class CurrentAffectedDynamics(BoatDynamics):
    """Model with a fixed 20° COG offset from the hull heading."""

    def step(self, state, action, environment, dt):
        heading = (state.heading - action.rudder * dt) % 360.0
        course = (heading - 20.0) % 360.0
        latitude, longitude = offset_latitude_longitude(
            state.latitude,
            state.longitude,
            dt * sin(radians(course)),
            dt * cos(radians(course)),
        )
        return BoatState(
            latitude,
            longitude,
            heading=heading,
            speed=1.0,
            rudder=action.rudder,
            flap=action.flap,
        )


class SaildroneMPCTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = BoatState(39.9, 18.4, heading=45.0, speed=0.0, flap=0.0)
        self.observation = WorldObservation(
            time="2026-07-26T21:00:00Z",
            boat=self.state,
            wind=None,
            wave=None,
            current=None,
            seabed=None,
            coast=None,
            chart=None,
            ais_targets=(),
        )

    def test_mpc_selects_a_starboard_command_for_an_eastward_reference(self) -> None:
        controller = SaildroneMPC(prediction_horizon_steps=4)
        result = controller.choose_action(
            TurningDynamics(),
            self.state,
            self.observation,
            desired_heading=90.0,
        )
        self.assertLess(result.action.rudder, 0.0)
        self.assertEqual(len(result.predicted_states), 4)
        self.assertEqual(
            result.candidates_evaluated,
            len(controller.rudder_candidates_degrees) * len(controller.flap_candidates_degrees),
        )

    def test_mpc_action_respects_the_configured_candidate_bounds(self) -> None:
        controller = SaildroneMPC(prediction_horizon_steps=2)
        result = controller.choose_action(
            TurningDynamics(),
            self.state,
            self.observation,
            desired_heading=0.0,
        )
        self.assertIn(result.action.rudder, controller.rudder_candidates_degrees)
        self.assertIn(result.action.flap, controller.flap_candidates_degrees)

    def test_mpc_respects_command_slew_bounds(self) -> None:
        controller = SaildroneMPC(prediction_horizon_steps=2)
        result = controller.choose_action(
            TurningDynamics(),
            self.state,
            self.observation,
            desired_heading=180.0,
        )
        self.assertLessEqual(
            abs(result.action.rudder - self.state.rudder),
            controller.max_rudder_change_per_update_degrees,
        )
        self.assertLessEqual(
            abs(result.action.flap - self.state.flap),
            controller.max_flap_change_per_update_degrees,
        )

    def test_mpc_tracks_ground_course_when_current_offsets_heading(self) -> None:
        state = BoatState(39.9, 18.4, heading=90.0, speed=1.0, flap=0.0)
        observation = WorldObservation(
            time="2026-07-26T21:00:00Z",
            boat=state,
            wind=None,
            wave=None,
            current=None,
            seabed=None,
            coast=None,
            chart=None,
            ais_targets=(),
        )
        result = SaildroneMPC(prediction_horizon_steps=1).choose_action(
            CurrentAffectedDynamics(),
            state,
            observation,
            desired_heading=90.0,
        )
        # Although the hull already points east, its COG is initially 70°.
        # A starboard command increases heading and restores the eastbound COG.
        self.assertLess(result.action.rudder, 0.0)

    def test_normal_sailing_heel_is_inside_the_mpc_soft_limit(self) -> None:
        controller = SaildroneMPC()
        level = BoatState(39.9, 18.4, heading=90.0, speed=1.0, heel=0.0)
        heeled = BoatState(39.9, 18.4, heading=90.0, speed=1.0, heel=5.0)
        level_cost = controller._state_cost(
            level,
            level,
            wind=None,
            desired_heading=90.0,
            local_grid=None,
            harmonic_plan=None,
        )
        heeled_cost = controller._state_cost(
            level,
            heeled,
            wind=None,
            desired_heading=90.0,
            local_grid=None,
            harmonic_plan=None,
        )
        self.assertAlmostEqual(heeled_cost, level_cost)

    def test_mpc_crosses_the_wind_to_reach_the_opposite_tack(self) -> None:
        # From the Brindisi replay, the old cost kept the flap on the original
        # tack even when the reference moved to the other side of the wind.
        state = BoatState(
            40.597755, 18.196700, heading=12.49, speed=0.159, rudder=7.5,
            flap=15.0, surge_speed=0.251, sway_speed=0.0227, yaw_rate=0.0076,
        )
        observation = replace(
            self.observation,
            boat=state,
            wind=Wind(2.023, -2.965, 3.59, 325.69),
            wave=Wave(0.554, 25.84, 4.73),
            current=Current(0.0083, -0.1053, 0.1056, 175.50, 0.0),
        )
        reference = 274.0
        latitude, longitude = offset_latitude_longitude(
            state.latitude, state.longitude,
            150.0 * sin(radians(reference)), 150.0 * cos(radians(reference)),
        )
        plan = HarmonicPlan(
            potential=np.zeros((3, 3)), blocked=np.zeros((3, 3), dtype=bool),
            path_cells=((1, 1), (1, 0)), desired_heading=reference,
            target_cell=(1, 0), target_on_boundary=False, converged=True, iterations=0,
            navigation_target=Waypoint(latitude, longitude),
        )
        dynamics = Fossen3DOFSaildroneDynamics()
        controller = SaildroneMPC()
        for _ in range(18):
            result = controller.choose_action(
                dynamics, state, observation, reference, harmonic_plan=plan,
            )
            self.assertTrue(controller._within_actuator_slew(state, result.action))
            state = dynamics.step(state, result.action, observation, 5.0)
            if self._on_requested_tack(state, observation.wind.direction, reference):
                break
        self.assertTrue(self._on_requested_tack(state, observation.wind.direction, reference))
        self.assertLess(state.flap, 0.0)

    @staticmethod
    def _on_requested_tack(state: BoatState, wind_from: float, reference: float) -> bool:
        bow = (state.heading - wind_from + 180.0) % 360.0 - 180.0
        return bow < -45.0 and abs(state.heading - reference) < 15.0


if __name__ == "__main__":
    unittest.main()
