"""Compare the generated-model transition with the simulator's Fossen step."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from world._geo import local_east_north_metres
from world.acados_fossen import discrete_fossen
from world.boat import BoatAction, BoatState
from world.currents import Current
from world.fossen import Fossen3DOFSaildroneDynamics
from world.waves import Wave
from world.weather import Wind
from world.world import WorldObservation

try:
    import casadi as ca
except ImportError:
    ca = None


@unittest.skipUnless(ca is not None, "CasADi is an optional NMPC dependency")
class AcadosFossenTests(unittest.TestCase):
    def test_five_second_transition_matches_simulator(self) -> None:
        dynamics = Fossen3DOFSaildroneDynamics()
        dt_seconds = 5.0
        transition = self._transition(dynamics, dt_seconds)
        for name, surge, sway, yaw_rate, command, forcing in (
            ("wind, waves and current", 1.35, -0.32, 0.08, (16.0, -13.0), (5.0, -7.0, -0.28, 0.41, 0.65)),
            ("reverse above resistance table", -2.45, 0.16, -0.12, (-35.0, 25.0), (-2.5, 4.3, 0.0, -0.12, 1.2)),
            ("zero wind and water velocities", 0.0, 0.0, 0.0, (22.0, -10.0), (0.0, 0.0, 0.0, 0.0, 0.0)),
        ):
            with self.subTest(name=name):
                self._check_step(dynamics, transition, dt_seconds, surge, sway, yaw_rate, command, forcing)

    def test_custom_damping_and_flap_slew_match_simulator(self) -> None:
        defaults = Fossen3DOFSaildroneDynamics().parameters
        parameters = replace(defaults, surge_resistance_table=(), flap_rate=2.0)
        dynamics = Fossen3DOFSaildroneDynamics(parameters)
        dt_seconds = 0.17
        transition = self._transition(dynamics, dt_seconds)
        result = self._check_step(dynamics, transition, dt_seconds, 0.8, -0.15, 0.06,
                                  (45.0, -40.0), (7.0, 2.0, 0.32, -0.44, 0.0))
        self.assertAlmostEqual(np.degrees(result[6]), 8.0 - parameters.flap_rate * dt_seconds)
        np.testing.assert_allclose(np.degrees(result[7:]), [parameters.rudder_max, parameters.flap_min])

    def test_compact_transition_jacobian_supports_c_generation(self) -> None:
        dynamics = Fossen3DOFSaildroneDynamics()
        for symbol_type in (ca.MX, ca.SX):
            with self.subTest(symbol_type=symbol_type):
                state = symbol_type.sym("state", 9)
                action = symbol_type.sym("action", 2)
                parameters = symbol_type.sym("parameters", 5)
                transition = discrete_fossen(dynamics, state, action, parameters, 5.0)
                jacobian = ca.jacobian(transition, ca.vertcat(state, action))
                function = ca.Function("fossen_jacobian", [state, action, parameters], [jacobian])
                rest_jacobian = function(np.zeros(9), np.zeros(2), np.zeros(5))
                self.assertTrue(np.isfinite(np.asarray(rest_jacobian)).all())
                with TemporaryDirectory() as directory:
                    generator = ca.CodeGenerator("fossen_jacobian.c")
                    generator.add(function)
                    generated = generator.generate(f"{directory}/")
                    self.assertTrue(Path(generated).is_file())

    @staticmethod
    def _transition(dynamics, dt_seconds):
        state = ca.MX.sym("state", 9)
        action = ca.MX.sym("action", 2)
        parameters = ca.MX.sym("parameters", 7)
        next_state = discrete_fossen(dynamics, state, action, parameters, dt_seconds)
        return ca.Function("transition", [state, action, parameters], [next_state])

    def _check_step(self, dynamics, transition, dt_seconds, surge, sway, yaw_rate, command, forcing):
        east, north = 17.0, -29.0
        state = BoatState(40.1, 18.5, heading=359.3, speed=0.0, flap=8.0,
                          surge_speed=surge, sway_speed=sway, yaw_rate=yaw_rate)
        wind_east, wind_north, current_east, current_north, wave_height = forcing
        observation = WorldObservation(
            time=np.datetime64("2026-09-30T03:00:00"),
            boat=state,
            wind=Wind(wind_east, wind_north, np.hypot(wind_east, wind_north), 0.0),
            wave=Wave(wave_height, 0.0, 5.0) if wave_height > 0.0 else None,
            current=Current(current_east, current_north, np.hypot(current_east, current_north), 0.0, 0.0),
            seabed=None,
            coast=None,
            chart=None,
            ais_targets=(),
        )
        expected = dynamics.step(state, BoatAction(*command), observation, dt_seconds)
        expected_east, expected_north = local_east_north_metres(
            expected.latitude, expected.longitude, state.latitude, state.longitude,
        )
        state_vector = [east, north, np.radians(state.heading), surge, sway, yaw_rate,
                        np.radians(state.flap), 0.12, -0.11]
        result = np.asarray(transition(state_vector, np.radians(command), [*forcing, 11.0, -13.0])).ravel()
        # The geographic simulator recomputes longitude scale at each latitude.
        np.testing.assert_allclose(result[:2], [east + expected_east, north + expected_north],
                                   atol=2e-5, rtol=0.0)
        heading_error = (result[2] - np.radians(expected.heading) + np.pi) % (2 * np.pi) - np.pi
        self.assertAlmostEqual(heading_error, 0.0, places=12)
        np.testing.assert_allclose(
            result[3:8],
            [expected.surge_speed, expected.sway_speed, expected.yaw_rate,
             np.radians(expected.flap), np.radians(expected.rudder)],
            atol=1e-12,
            rtol=0.0,
        )
        self.assertAlmostEqual(
            result[8],
            np.radians(np.clip(command[1], dynamics.parameters.flap_min, dynamics.parameters.flap_max)),
        )
        return result
