"""Regression checks for the physical and legacy saildrone models."""

from __future__ import annotations

import math
import unittest
from dataclasses import fields

import numpy as np

from config import SAILDRONE, SAILDRONE_SURGE_RESISTANCE_TABLE_N
from world import (
    BoatAction,
    BoatState,
    Fossen3DOFParameters,
    Fossen3DOFSaildroneDynamics,
    SaildroneDynamics,
    SaildroneParameters,
    Wind,
    WorldObservation,
)


def observation(*, wind: Wind) -> WorldObservation:
    """Build the smallest environmental input needed by the dynamics."""

    boat = BoatState(latitude=40.0, longitude=18.0, heading=90.0, speed=0.0)
    return WorldObservation(
        time=np.datetime64("2026-07-26T21:00:00", "ns"),
        boat=boat,
        wind=wind,
        wave=None,
        current=None,
        seabed=None,
        coast=None,
        chart=None,
        ais_targets=(),
    )


class SaildroneDynamicsTests(unittest.TestCase):
    """Confirm that propulsion follows forces rather than the old sine curve."""

    def setUp(self) -> None:
        # Air flows north; therefore the meteorological FROM direction is south.
        self.wind = Wind(u=0.0, v=10.0, speed=10.0, direction=180.0)

    def test_neutral_wing_does_not_drive_a_crosswind_course(self) -> None:
        dynamics = SaildroneDynamics()
        state = BoatState(40.0, 18.0, heading=90.0, speed=0.0, flap=0.0)

        next_state = dynamics.step(
            state,
            BoatAction(rudder=0.0, flap=0.0),
            observation(wind=self.wind),
            dt=1.0,
        )

        self.assertAlmostEqual(next_state.speed, 0.0, places=10)

    def test_flap_deflection_creates_crosswind_propulsion(self) -> None:
        dynamics = SaildroneDynamics()
        state = BoatState(40.0, 18.0, heading=90.0, speed=0.0, flap=-10.0)

        next_state = dynamics.step(
            state,
            BoatAction(rudder=0.0, flap=-10.0),
            observation(wind=self.wind),
            dt=1.0,
        )

        self.assertGreater(next_state.speed, 0.0)

    def test_quadratic_hull_resistance_slows_an_overfast_boat(self) -> None:
        dynamics = SaildroneDynamics()
        state = BoatState(40.0, 18.0, heading=90.0, speed=4.0, flap=-10.0)

        next_state = dynamics.step(
            state,
            BoatAction(rudder=0.0, flap=-10.0),
            observation(wind=self.wind),
            dt=1.0,
        )

        self.assertLess(next_state.speed, state.speed)

    def test_legacy_visualisation_still_uses_original_speed_curve(self) -> None:
        dynamics = SaildroneDynamics(SaildroneParameters.legacy_visualisation())
        state = BoatState(40.0, 18.0, heading=90.0, speed=0.0, flap=-10.0)

        next_state = dynamics.step(
            state,
            BoatAction(rudder=0.0, flap=-10.0),
            observation(wind=self.wind),
            dt=0.001,
        )

        expected_speed = 2_000_000.0 * math.sin(math.radians(100.0))
        self.assertAlmostEqual(next_state.speed, expected_speed, places=6)

    def test_fossen_mass_matrix_is_symmetric_positive_definite(self) -> None:
        matrix = Fossen3DOFParameters().mass_matrix
        np.testing.assert_allclose(matrix, matrix.T)
        self.assertTrue(np.all(np.linalg.eigvalsh(matrix) > 0.0))

    def test_supplied_cfd_values_are_used_by_the_default_model(self) -> None:
        parameters = Fossen3DOFParameters()

        self.assertEqual(parameters.mass, 52.2)
        self.assertAlmostEqual(parameters.wing_sail_area, 1.203)
        self.assertAlmostEqual(parameters.rudder_area, 0.08442925917762516)
        self.assertAlmostEqual(parameters.rudder_lift_slope, 2.84505091180687)
        self.assertEqual(parameters.surge_resistance_table, SAILDRONE_SURGE_RESISTANCE_TABLE_N)
        self.assertEqual(SAILDRONE["righting_moment_per_degree_nm"], 2.30)

    def test_fossen_surge_damping_matches_supplied_resistance_curve(self) -> None:
        dynamics = Fossen3DOFSaildroneDynamics()
        speed, resistance = SAILDRONE_SURGE_RESISTANCE_TABLE_N[3]

        damping = dynamics._surge_damping_coefficient(speed)

        self.assertAlmostEqual(damping * speed, resistance)

    def test_fossen_model_tracks_surge_sway_and_yaw(self) -> None:
        dynamics = Fossen3DOFSaildroneDynamics()
        state = BoatState(40.0, 18.0, heading=90.0, speed=0.0, flap=-10.0)

        next_state = dynamics.step(
            state,
            BoatAction(rudder=3.0, flap=-10.0),
            observation(wind=self.wind),
            dt=1.0,
        )

        self.assertIsNotNone(next_state.surge_speed)
        self.assertGreater(next_state.surge_speed, 0.0)
        self.assertTrue(np.isfinite(next_state.sway_speed))
        self.assertLess(next_state.yaw_rate, 0.0)

    def test_fossen_flap_motion_is_consistent_across_step_lengths(self) -> None:
        dynamics = Fossen3DOFSaildroneDynamics(Fossen3DOFParameters(flap_rate=4.0))
        initial = BoatState(40.0, 18.0, heading=90.0, speed=0.0, flap=-15.0)
        action = BoatAction(rudder=3.0, flap=15.0)
        forcing = observation(wind=self.wind)

        whole_step = dynamics.step(initial, action, forcing, dt=5.0)
        split_steps = initial
        for _ in range(5):
            split_steps = dynamics.step(split_steps, action, forcing, dt=1.0)

        self.assertAlmostEqual(whole_step.flap, 5.0)
        for field in fields(BoatState):
            with self.subTest(field=field.name):
                self.assertAlmostEqual(
                    getattr(whole_step, field.name), getattr(split_steps, field.name), places=10
                )


if __name__ == "__main__":
    unittest.main()
