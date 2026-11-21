"""Tests for preliminary GPX route steering guidance."""

from __future__ import annotations

import unittest

from world import BoatState, RouteWaypoint, WaypointGuidance, initial_bearing_degrees


class WaypointGuidanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.departure = (39.902416747, 18.414363207)
        self.goal = RouteWaypoint(39.884456929, 19.576520777, name="002")

    def test_first_route_leg_has_an_eastward_bearing(self) -> None:
        bearing = initial_bearing_degrees(*self.departure, self.goal.latitude, self.goal.longitude)
        self.assertGreater(bearing, 85.0)
        self.assertLess(bearing, 95.0)

    def test_eastward_goal_commands_starboard_turn_from_45_degrees(self) -> None:
        state = BoatState(*self.departure, heading=45.0, speed=1.0)
        action = WaypointGuidance(flap_degrees=-10.0).action(state, self.goal)
        # Positive rudder is port in the Fossen model; starboard is negative.
        self.assertLess(action.rudder, 0.0)
        self.assertEqual(action.flap, -10.0)

    def test_completed_mission_neutralises_controls(self) -> None:
        state = BoatState(*self.departure, heading=45.0, speed=1.0)
        self.assertEqual(WaypointGuidance().action(state, None).rudder, 0.0)
        self.assertEqual(WaypointGuidance().action(state, None).flap, 0.0)

    def test_heading_reference_uses_the_same_starboard_rudder_convention(self) -> None:
        state = BoatState(*self.departure, heading=45.0, speed=1.0)
        action = WaypointGuidance(flap_degrees=-10.0).action_to_heading(state, 90.0)
        self.assertLess(action.rudder, 0.0)
        self.assertEqual(action.flap, -10.0)


if __name__ == "__main__":
    unittest.main()
