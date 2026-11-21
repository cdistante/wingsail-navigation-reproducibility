"""Tests for the conservative AIS-informed COLREG simulation policy."""

from __future__ import annotations

import unittest

import numpy as np

from world import (
    AISTarget,
    BoatState,
    ColregEncounter,
    ColregPolicy,
    LocalGrid,
    WorldObservation,
)
from world._geo import offset_latitude_longitude


class ColregPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.latitude = 39.9
        self.longitude = 18.4
        self.state = BoatState(self.latitude, self.longitude, heading=0.0, speed=3.0)
        self.policy = ColregPolicy(
            assessment_horizon_seconds=300.0,
            risk_dcpa_metres=25.0,
            emergency_dcpa_metres=10.0,
            starboard_turn_clearance_metres=100.0,
        )
        self.grid = LocalGrid(
            data=np.zeros((10, 41, 41), dtype=np.float32),
            channels=(
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
            ),
            resolution=5.0,
            span=205.0,
            origin_latitude=self.latitude,
            origin_longitude=self.longitude,
        )

    def decision(self, east: float, north: float, course: float, speed: float) -> object:
        latitude, longitude = offset_latitude_longitude(
            self.latitude, self.longitude, east, north
        )
        target = AISTarget(
            mmsi=999000001,
            latitude=float(latitude),
            longitude=float(longitude),
            speed=speed,
            course=course,
            heading=course,
        )
        observation = WorldObservation(
            time=np.datetime64("2026-07-28T10:00:00"),
            boat=self.state,
            wind=None,
            wave=None,
            current=None,
            seabed=None,
            coast=None,
            chart=None,
            ais_targets=(target,),
        )
        return self.policy.assess(self.state, observation)[0]

    def test_head_on_encounter_is_a_starboard_give_way_case(self) -> None:
        decision = self.decision(0.0, 300.0, 180.0, 3.0)
        self.assertEqual(decision.encounter, ColregEncounter.HEAD_ON_GIVE_WAY)
        self.assertTrue(decision.risk_of_collision)
        self.assertTrue(decision.give_way)
        self.assertAlmostEqual(decision.time_to_cpa_seconds, 50.0, delta=0.1)

    def test_starboard_crossing_is_a_give_way_case(self) -> None:
        decision = self.decision(300.0, 300.0, 270.0, 3.0)
        self.assertEqual(decision.encounter, ColregEncounter.CROSSING_GIVE_WAY)
        self.assertTrue(decision.give_way)
        self.assertGreater(decision.relative_bearing_degrees, 0.0)

    def test_port_crossing_is_a_stand_on_case(self) -> None:
        decision = self.decision(-300.0, 300.0, 90.0, 3.0)
        self.assertEqual(decision.encounter, ColregEncounter.CROSSING_STAND_ON)
        self.assertFalse(decision.give_way)
        self.assertLess(decision.relative_bearing_degrees, 0.0)

    def test_overtaking_remains_a_give_way_case(self) -> None:
        self.state = BoatState(self.latitude, self.longitude, heading=0.0, speed=4.0)
        decision = self.decision(0.0, 300.0, 0.0, 2.0)
        self.assertEqual(decision.encounter, ColregEncounter.OVERTAKING_GIVE_WAY)
        self.assertTrue(decision.give_way)

    def test_give_way_reserves_the_port_forward_turn_sector(self) -> None:
        decision = self.decision(300.0, 300.0, 270.0, 3.0)
        mask = self.policy.extra_blocked_mask(self.grid, self.state, (decision,))
        # Northwest is a port-forward cell for a northbound boat.
        self.assertTrue(mask[10, 10])
        # Northeast remains available for a starboard alteration.
        self.assertFalse(mask[10, 30])


if __name__ == "__main__":
    unittest.main()
