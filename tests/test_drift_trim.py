from types import SimpleNamespace
import unittest

from world import BoatState, Fossen3DOFSaildroneDynamics, SaildroneMPC
from world.run_legacy_fossen import drift_trim_action


class DriftTrimTests(unittest.TestCase):
    def test_holding_respects_rudder_and_flap_slew(self):
        dynamics = Fossen3DOFSaildroneDynamics()
        mpc = SaildroneMPC()
        state = BoatState(40.225, 18.75, 180.0, 0.0, rudder=10.0, flap=5.0)
        plan = SimpleNamespace(
            drift_mode="holding", desired_bow_heading=90.0,
            steady_rudder_degrees=-8.0, steady_flap_degrees=-15.0,
        )
        action = drift_trim_action(dynamics, mpc, state, plan)
        self.assertIsNotNone(action)
        self.assertAlmostEqual(action.rudder, 17.5)
        self.assertAlmostEqual(action.flap, 0.0)
        self.assertTrue(mpc._within_actuator_slew(state, action))

        plan.drift_mode = "recovering"
        self.assertIsNone(drift_trim_action(dynamics, mpc, state, plan))


if __name__ == "__main__":
    unittest.main()
