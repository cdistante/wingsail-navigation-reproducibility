"""Exercise the moving AIS parameter and safety contracts without native codegen."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from world._geo import offset_latitude_longitude
from world.acados_nmpc import AISControlInfeasible, AcadosNMPC
from world.ais import AISTarget
from world.ais_avoidance import AISPrediction
from world.boat import BoatAction, BoatState
from world.fossen import Fossen3DOFSaildroneDynamics
from world.mpc import MPCResult
from world.weather import Wind
from world.world import WorldObservation


class AcadosAISTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dynamics = Fossen3DOFSaildroneDynamics()
        self.state = BoatState(40.225, 18.75, 90.0, 0.0, surge_speed=0.0)
        self.observation = WorldObservation(
            time=np.datetime64("2026-10-01T13:00:00"), boat=self.state,
            wind=Wind(0.0, -6.0, 6.0, 0.0), wave=None, current=None,
            seabed=None, coast=None, chart=None,
            ais_targets=(self.target(1, 25.0, 0.0), self.target(2, 70.0, 12.0)),
        )
        self.traffic = AISPrediction.from_observation(self.observation, uncertainty_growth_mps=0.2)
        self.solver = Mock()
        self.solver.solve.return_value = 0
        self.solver.get_cost.return_value = 1.0
        self.solver.get.side_effect = lambda _index, variable: np.zeros(2 if variable == "u" else 9)
        scratch = TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        with patch.object(AcadosNMPC, "_build_solver", return_value=(self.solver, lambda x, u, p: x)):
            self.controller = AcadosNMPC(
                self.dynamics, ais_capacity=1, dt_seconds=5.0, horizon_seconds=15.0,
                build_directory=Path(scratch.name),
            )
        self.fallback = BoatAction(rudder=1.0, flap=-1.0)
        self.controller.fallback = Mock(wraps=self.controller.fallback)
        self.controller.fallback.choose_action = Mock(return_value=MPCResult(self.fallback, 2.0, (), 1))

    def target(self, mmsi: int, east: float, speed: float) -> AISTarget:
        latitude, longitude = offset_latitude_longitude(self.state.latitude, self.state.longitude, east, 0.0)
        return AISTarget(mmsi, float(latitude), float(longitude), speed=speed, course=270.0, heading=None,
                         length_metres=5.0, beam_metres=2.0)

    def choose(self):
        return self.controller.choose_action(self.dynamics, self.state, self.observation, 90.0,
                                             ais_context=self.traffic)

    def test_approaching_contact_keeps_one_ranked_slot_through_terminal_knot(self) -> None:
        safe = SimpleNamespace(safe=True, recovering=False)
        with patch("world.ais_guidance.assess_action", return_value=(True, safe)):
            self.choose()
        calls = [call.args for call in self.solver.set.call_args_list if call.args[1] == "p"]
        self.assertEqual([stage for stage, _, _ in calls], list(range(4)))
        stages = np.stack([value for _, _, value in calls])
        times = np.arange(4) * 5.0
        np.testing.assert_allclose(stages[:, 14:16], self.traffic.centres_at(times)[:, 1, :])
        np.testing.assert_allclose(stages[:, 16], self.traffic.hull_radii_metres[1])
        np.testing.assert_allclose(stages[:, 17], self.traffic.radii_at(times)[:, 1])
        np.testing.assert_array_equal(stages[:, 18], np.ones(4))

    def test_ais_veto_is_distinct_from_solver_failure_and_checks_all_targets(self) -> None:
        unsafe = SimpleNamespace(safe=False, recovering=False)
        safe = SimpleNamespace(safe=True, recovering=False)
        with patch("world.ais_guidance.assess_action", side_effect=((True, unsafe), (True, safe))) as shield:
            result = self.choose()

        self.assertEqual(result.action, self.fallback)
        self.assertEqual(result.control_source, "grid-mpc")
        self.assertEqual(result.solver_status, 0)
        self.assertIsNone(result.fallback_reason)
        self.assertEqual(result.safety_intervention, "acados first step violates moving AIS clearance")
        self.assertEqual(shield.call_count, 2)
        for call in shield.call_args_list:
            self.assertIs(call.args[-1], self.traffic)
            self.assertEqual(len(call.args[-1].targets), 2)

    def test_both_actions_vetoed_reports_infeasible_without_returning_a_command(self) -> None:
        unsafe = SimpleNamespace(safe=False, recovering=False)
        with patch("world.ais_guidance.assess_action", return_value=(True, unsafe)):
            with self.assertRaises(AISControlInfeasible) as error:
                self.choose()

        self.assertEqual(error.exception.solver_status, 0)
        self.assertIsNone(error.exception.fallback_reason)
        self.assertIn("grid MPC first step", error.exception.safety_intervention)

    def test_solver_failure_stays_separate_from_a_safe_grid_action(self) -> None:
        self.solver.solve.return_value = 4
        safe = SimpleNamespace(safe=True, recovering=False)
        with patch("world.ais_guidance.assess_action", return_value=(True, safe)) as shield:
            result = self.choose()

        self.assertEqual(result.control_source, "grid-mpc")
        self.assertEqual(result.fallback_reason, "acados status 4")
        self.assertIsNone(result.safety_intervention)
        shield.assert_called_once()

    def test_explicit_buffer_recovery_accepts_native_action(self) -> None:
        recovery = SimpleNamespace(safe=False, recovering=True)
        with patch("world.ais_guidance.assess_action", return_value=(True, recovery)):
            result = self.choose()

        self.assertEqual(result.control_source, "acados")
        self.assertEqual(result.safety_intervention, "moving AIS clearance recovery")
        self.controller.fallback.choose_action.assert_not_called()


if __name__ == "__main__":
    unittest.main()
