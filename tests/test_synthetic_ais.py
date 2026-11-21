"""Tests for repeatable AIS encounters and OpenCPN AIVDM encoding."""

from __future__ import annotations

import unittest

from world import SyntheticAIS, aivdm_position_report, aivdm_static_data_sentences
from world._geo import local_east_north_metres
from world.nmea import nmea_checksum
from world.synthetic_ais import SyntheticAISReporting


def _decode_six_bit(payload: str) -> str:
    values = []
    for character in payload:
        code = ord(character)
        values.append(code - 48 if code <= 87 else code - 56)
    return "".join(f"{value:06b}" for value in values)


class SyntheticAISTests(unittest.TestCase):
    def setUp(self) -> None:
        self.origin = (39.902416747, 18.414363207)

    def test_crossing_target_moves_on_declared_course(self) -> None:
        source = SyntheticAIS("crossing", *self.origin).load()
        start, later = source.targets(0.0)[0], source.targets(10.0)[0]
        east, north = local_east_north_metres(
            start.latitude, start.longitude, *self.origin
        )
        moved_east, moved_north = local_east_north_metres(
            later.latitude, later.longitude, start.latitude, start.longitude
        )
        self.assertAlmostEqual(float(east), 90.0, places=1)
        self.assertAlmostEqual(float(north), -400.0, places=1)
        self.assertAlmostEqual(float(moved_east), 0.0, places=3)
        self.assertAlmostEqual(float(moved_north), 35.0, places=2)

    def test_type_one_report_contains_the_target_mmsi_and_checksum(self) -> None:
        target = SyntheticAIS("crossing", *self.origin).load().targets(0.0)[0]
        sentence = aivdm_position_report(target)
        self.assertTrue(sentence.startswith("!AIVDM,1,1,,A,"))
        body, checksum = sentence[1:].strip().split("*")
        self.assertEqual(int(checksum, 16), nmea_checksum(body))
        payload = body.split(",")[5]
        bits = _decode_six_bit(payload)
        self.assertEqual(int(bits[:6], 2), 1)
        self.assertEqual(int(bits[8:38], 2), target.mmsi)

    def test_delayed_packets_hold_the_last_delivered_source_position(self) -> None:
        reporting = SyntheticAISReporting(5, 10, ((85, 105), (115, 135)))
        receiver = SyntheticAIS("mixed", *self.origin, reporting=reporting).load()
        truth = SyntheticAIS("mixed", *self.origin).load()
        self.assertFalse(receiver.reports_refreshed_each_step)
        self.assertTrue(truth.reports_refreshed_each_step)

        for when, emitted in ((0, -10), (95, 80), (110, 80), (115, 105),
                              (120, 110), (140, 110), (145, 135)):
            with self.subTest(when=when):
                reports = receiver.targets(when)
                self.assertEqual(len(reports), 3)
                self.assertTrue(all(target.report_age_seconds == when - emitted for target in reports))
                if emitted >= 0:
                    self.assertEqual(
                        [(target.latitude, target.longitude) for target in reports],
                        [(target.latitude, target.longitude) for target in truth.targets(emitted)],
                    )
                else:
                    original_tug = truth.targets(0)[-1]
                    east, north = local_east_north_metres(
                        reports[-1].latitude, reports[-1].longitude, original_tug.latitude, original_tug.longitude,
                    )
                    self.assertAlmostEqual(east, -30.0, places=3)
                    self.assertAlmostEqual(north, 0.0, places=3)

    def test_type_five_static_report_is_a_valid_two_fragment_message(self) -> None:
        target = SyntheticAIS("head_on", *self.origin).load().targets(0.0)[0]
        first, second = aivdm_static_data_sentences(target, sequence_id="7")
        self.assertTrue(first.startswith("!AIVDM,2,1,7,A,"))
        self.assertTrue(second.startswith("!AIVDM,2,2,7,A,"))
        for sentence in (first, second):
            body, checksum = sentence[1:].strip().split("*")
            self.assertEqual(int(checksum, 16), nmea_checksum(body))


    def test_turning_target_changes_only_after_a_new_report_is_delivered(self) -> None:
        reporting = SyntheticAISReporting(5, 10, ((85, 105), (115, 135)))
        receiver = SyntheticAIS("mixed", *self.origin, reporting=reporting, manoeuvring=True).load()
        truth = SyntheticAIS("mixed", *self.origin, manoeuvring=True).load()
        self.assertFalse(truth.reports_refreshed_each_step)

        stale = receiver.targets(110)[-1]
        actual = truth.targets(110)[-1]
        self.assertEqual(stale.report_age_seconds, 30.0)
        self.assertAlmostEqual(stale.course, 90.0)
        self.assertAlmostEqual(stale.speed, 3.0)
        self.assertAlmostEqual(actual.course, 88.5)
        self.assertAlmostEqual(actual.speed, 3.08)

        updated = receiver.targets(115)[-1]
        at_source = truth.targets(105)[-1]
        self.assertEqual(updated.report_age_seconds, 10.0)
        self.assertEqual((updated.latitude, updated.longitude), (at_source.latitude, at_source.longitude))
        self.assertAlmostEqual(updated.course, actual.course)
        self.assertAlmostEqual(updated.speed, actual.speed)

    def test_sharp_mixed_profile_changes_course_and_speed_in_the_same_windows(self) -> None:
        source = SyntheticAIS("mixed", *self.origin, sharp_manoeuvres=True).load()
        initial = source.targets(85.0)
        after = source.targets(135.0)
        self.assertFalse(source.reports_refreshed_each_step)
        for start, finish, turn, speed_change in zip(initial, after, (-6, 6, -6), (-0.35, -0.35, 0.35)):
            with self.subTest(mmsi=start.mmsi):
                self.assertAlmostEqual((finish.course - start.course + 180) % 360 - 180, turn)
                self.assertAlmostEqual(finish.speed - start.speed, speed_change)

    def test_launch_lead_preserves_the_encounter_clock_and_report_ages(self) -> None:
        reporting = SyntheticAISReporting(5, 10, ((85, 105), (115, 135)))
        original = SyntheticAIS("mixed", *self.origin, reporting=reporting, sharp_manoeuvres=True).load()
        early = SyntheticAIS("mixed", *self.origin, reporting=reporting, sharp_manoeuvres=True,
                             launch_lead_seconds=45.0).load()
        self.assertEqual(early.targets(0)[0].report_age_seconds, 10.0)
        for epoch in (0.0, 95.0, 110.0, 115.0, 145.0):
            self.assertEqual(early.targets(epoch + 45.0), original.targets(epoch))


if __name__ == "__main__":
    unittest.main()
