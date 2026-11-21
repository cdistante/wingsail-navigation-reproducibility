"""Regression tests for live OpenCPN AIS decoding."""

from __future__ import annotations

import unittest

from world.ais import AISTarget
from world.live_ais import LiveAIS
from world.nmea import aivdm_position_report


class LiveAISTests(unittest.TestCase):
    def test_position_report_decodes_to_target_with_metres_per_second(self) -> None:
        sentence = aivdm_position_report(
            AISTarget(
                mmsi=123456789,
                latitude=39.7895,
                longitude=18.3437,
                speed=4.2,
                course=91.0,
                heading=90.0,
            )
        )
        target = LiveAIS._decode(sentence)
        self.assertEqual(target.mmsi, 123456789)
        self.assertAlmostEqual(target.latitude, 39.7895, places=4)
        self.assertAlmostEqual(target.longitude, 18.3437, places=4)
        self.assertAlmostEqual(target.speed, 4.2, places=1)
        self.assertAlmostEqual(target.course, 91.0, places=1)


if __name__ == "__main__":
    unittest.main()
