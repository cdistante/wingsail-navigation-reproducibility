"""Check the direct controller's same-side initialization without code generation."""

import unittest
from unittest.mock import Mock

import numpy as np

from world.acados_nmpc import AcadosNMPC
from world.boat import BoatState
from world.harmonic_planner import HarmonicPlan
from world.mpc import SaildroneMPC
from world.weather import Wind
from world.world import WorldObservation


class AcadosNMPCSeedTests(unittest.TestCase):
    def test_cold_close_hauled_seed_moves_toward_the_requested_trim(self) -> None:
        for side in (1, -1):
            with self.subTest(side=side):
                controller = AcadosNMPC.__new__(AcadosNMPC)
                controller.mode = "direct"
                controller.stages = 5
                controller.lower = np.radians([-30.0, -15.0])
                controller.upper = np.radians([30.0, 15.0])
                controller.slew = np.radians([7.5, 5.0])
                controller._warm_bow = None
                controller._warm_commands = None
                controller._solver = Mock()
                controller._discrete_step = lambda state, command, _: np.r_[state[:7], command]

                wind_from = np.radians(320.0)
                parameters = np.zeros(14)
                parameters[:2] = -5.0 * np.array([np.sin(wind_from), np.cos(wind_from)])
                parameters[9] = wind_from + side * np.radians(45.5)
                parameters[11:14] = 1.0  # Close-hauled with a supplied bow reference.
                state = BoatState(40.225, 18.75, (320.0 + side * 85.0) % 360, 0.0, flap=-side * 10.0)

                controller._seed(state, parameters)

                commands = [call.args[2] for call in controller._solver.set.call_args_list if call.args[1] == "u"]
                np.testing.assert_allclose(np.degrees(commands[0]), side * np.array([7.5, -5.0]), atol=1e-10)
                self.assertAlmostEqual(np.degrees(commands[-1][1]), side * 15.0)

    def test_planned_reach_reseeds_when_warm_solver_is_backing(self) -> None:
        controller = AcadosNMPC.__new__(AcadosNMPC)
        controller.mode = "direct"
        controller.stages = 5
        controller.no_go_angle_degrees = 45.0
        controller.fallback = SaildroneMPC()
        controller.lower = np.radians([-30.0, -15.0])
        controller.upper = np.radians([30.0, 15.0])
        controller.slew = np.radians([7.5, 5.0])
        controller._warm_bow = np.radians(350.0)
        controller._warm_commands = np.tile(np.radians([3.0, -4.9]), (controller.stages, 1))
        controller._solver = Mock()
        controller._discrete_step = lambda state, command, _: np.r_[state[:7], command]
        state = BoatState(40.225, 18.75, 180.0, 0.16, rudder=3.0, flap=-4.9, surge_speed=-0.3)
        wind_from = np.radians(85.0)
        wind = Wind(-3.44 * np.sin(wind_from), -3.44 * np.cos(wind_from), 3.44, 85.0)
        observation = WorldObservation(np.datetime64("2026-10-01"), state, wind, None, None, None, None, None, ())
        plan = HarmonicPlan(
            np.zeros((3, 3)), np.zeros((3, 3), dtype=bool), ((1, 1), (0, 1)), 31.3,
            (0, 1), False, True, 1, desired_bow_heading=350.0,
        )

        parameters = controller._stage_parameters(state, observation, 31.3, plan, None, None)
        self.assertEqual(parameters[12], 0.0)  # The desired bow is a reach, not close-hauled.
        self.assertEqual(parameters[13], 1.0)
        self.assertLess(parameters[11], 0.5)
        controller._seed(state, parameters)
        first = next(call.args[2] for call in controller._solver.set.call_args_list if call.args[1] == "u")
        np.testing.assert_allclose(np.degrees(first), [-4.5, -9.9], atol=1e-10)


if __name__ == "__main__":
    unittest.main()
