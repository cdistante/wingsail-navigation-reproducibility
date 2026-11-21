"""Departure admission tests for a staged vessel and received AIS."""
from dataclasses import replace
import unittest

import numpy as np

from world._geo import offset_latitude_longitude
from world.ais import AISTarget
from world.ais_departure_gate import DepartureGateSettings, assess_departure
from world.boat import BoatState
from world.world import WorldObservation


class AISDepartureGateTests(unittest.TestCase):
    def setUp(self):
        boat = BoatState(40.225, 18.746666667, heading=45.0, speed=0.0)
        lat, lon = offset_latitude_longitude(boat.latitude, boat.longitude, 900.0, 0.0)
        self.report = AISTarget(
            mmsi=999000201, latitude=float(lat), longitude=float(lon), course=90.0, heading=90.0, speed=3.0,
            report_age_seconds=20.0, length_metres=60.0, beam_metres=12.0,
        )
        self.observation = WorldObservation(
            np.datetime64("2026-09-30T23:50:00"), boat, None, None, None, None, None, None, (self.report,),
        )

    def assess(self, *targets):
        return assess_departure(
            replace(self.observation, ais_targets=targets), DepartureGateSettings(), 5.0,
            acceleration_bound_mps2=.026, clearance_reserve_metres=3.0,
        )

    def test_clear_receding_fresh_report_can_release(self):
        result = self.assess(self.report)
        self.assertTrue(result.eligible)
        self.assertGreater(result.min_uncapped_startup_margin_m, 0.0)
        self.assertGreater(result.min_preview_margin_m, 0.0)

    def test_silence_and_held_report_do_not_authorize_launch(self):
        self.assertFalse(self.assess().eligible)
        stale = self.assess(replace(self.report, report_age_seconds=25.0))
        self.assertFalse(stale.fresh_reports)
        self.assertFalse(stale.eligible)

    def test_all_reported_contacts_must_be_receding(self):
        approaching = replace(self.report, mmsi=999000202, course=270.0)
        result = self.assess(self.report, approaching)
        self.assertTrue(result.fresh_reports)
        self.assertLess(result.min_robust_receding_mps, 0.0)
        self.assertFalse(result.eligible)

    def test_receding_contact_can_still_conflict_with_uncapped_startup_tube(self):
        boat = self.observation.boat
        lat, lon = offset_latitude_longitude(boat.latitude, boat.longitude, 145.0, 0.0)
        result = self.assess(replace(self.report, latitude=float(lat), longitude=float(lon), speed=2.0))
        self.assertGreater(result.min_robust_receding_mps, 0.0)
        self.assertGreater(result.min_preview_margin_m, 0.0)
        self.assertLess(result.min_uncapped_startup_margin_m, 0.0)
        self.assertFalse(result.eligible)


if __name__ == "__main__":
    unittest.main()
