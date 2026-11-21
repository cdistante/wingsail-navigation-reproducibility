"""Regression coverage for replaying recorded tack references by mission leg."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from world._geo import local_east_north_metres, offset_latitude_longitude
from world.animate_episode import EpisodeFrame
from world.boat import BoatState
from world.navigation_replay import NavigationRecorder
from world.scenario import Route, RouteWaypoint, Waypoint
from world.tack_plan_replay import TackPlanReplay
from world.world import WorldObservation


class TackPlanReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "navigation.jsonl"
        self.departure = np.datetime64("2026-09-30T03:00:00", "ns")
        self.route = Route(
            "Tack comparison",
            (
                RouteWaypoint(40.25, 18.5, "001"),
                RouteWaypoint(40.26, 18.5, "002"),
                RouteWaypoint(40.27, 18.52, "003"),
            ),
            self.path.with_suffix(".gpx"),
        )

    def test_each_phase_and_waypoint_has_its_own_update_index(self) -> None:
        self._write_replay(
            ("outbound", "002", 55.0, 60.0),
            ("outbound", "002", 70.0, 77.0),
            ("outbound", "003", 81.0, 83.0),
            ("return", "002", 236.0, 231.0),
            ("return", "001", 214.0, 211.0),
        )
        replay = TackPlanReplay(self.path, self.route, self.departure, 5.0)

        for phase, name, leg_step, course, bow, recorded_start in (
            ("outbound", "002", 0, 55.0, 60.0, 0),
            ("outbound", "002", 1, 70.0, 77.0, 1),
            ("outbound", "003", 0, 81.0, 83.0, 2),
            ("return", "002", 0, 236.0, 231.0, 3),
            ("return", "001", 0, 214.0, 211.0, 4),
        ):
            with self.subTest(phase=phase, name=name, leg_step=leg_step):
                reference = replay.reference(phase, name, leg_step)
                self.assertEqual((reference.course, reference.bow), (course, bow))
                self.assertEqual(reference.start, self._position(recorded_start))

        self.assertEqual(replay.reference("outbound", "002", 500), replay.reference("outbound", "002", 1))
        self.assertEqual(replay.reference("return", "002", 500), replay.reference("return", "002", 0))
        self.assertIsNone(replay.reference("return", "003", 0))

    def test_direct_reference_targets_the_waypoint_and_pins_the_segment_start(self) -> None:
        self._write_replay(
            ("outbound", "002", 359.98, None),
            ("outbound", "002", 0.0, None),
            ("outbound", "002", 0.0, 24.0),
            ("outbound", "002", 90.0, None),
            ("outbound", "002", 0.0, None),
        )
        replay = TackPlanReplay(self.path, self.route, self.departure, 5.0)
        target = self.route.waypoints[1]

        for leg_step, recorded_start in ((0, 0), (1, 0), (4, 4)):
            with self.subTest(direct_step=leg_step):
                reference = replay.reference("outbound", "002", leg_step)
                self.assertEqual(reference.target, target)
                self.assertEqual(reference.start, self._position(recorded_start))

        for leg_step, expected_offset in ((2, (0.0, 150.0)), (3, (150.0, 0.0))):
            with self.subTest(tack_step=leg_step):
                reference = replay.reference("outbound", "002", leg_step)
                self.assertEqual(reference.start, self._position(leg_step))
                offset = local_east_north_metres(
                    reference.target.latitude, reference.target.longitude,
                    reference.start.latitude, reference.start.longitude,
                )
                np.testing.assert_allclose(offset, expected_offset, atol=1e-7, rtol=0.0)

    def test_rejects_missing_and_nonfinite_angles(self) -> None:
        self._write_replay(("outbound", "002", 55.0, 60.0))
        lines = self.path.read_text().splitlines()
        for field, value in (
            ("heading_reference_deg", None),
            ("heading_reference_deg", float("nan")),
            ("bow_reference_deg", float("inf")),
        ):
            with self.subTest(field=field, value=value):
                update = json.loads(lines[-1])
                update["navigation"][field] = value
                self.path.write_text("\n".join([*lines[:-1], json.dumps(update)]) + "\n")
                with self.assertRaisesRegex(ValueError, "Invalid tack reference at replay step 1"):
                    TackPlanReplay(self.path, self.route, self.departure, 5.0)

    def test_tack_ray_ends_within_capture_circle_when_it_crosses(self) -> None:
        start = self.route.departure
        target = RouteWaypoint(*offset_latitude_longitude(start.latitude, start.longitude, 18.0, 50.0), "002")
        self.route = replace(self.route, waypoints=(start, target, self.route.destination))
        self._write_replay(("outbound", "002", 0.0, 45.0))
        replay = TackPlanReplay(self.path, self.route, self.departure, 5.0)
        reference = replay.reference("outbound", "002", 0)
        offset = local_east_north_metres(reference.target.latitude, reference.target.longitude,
                                         start.latitude, start.longitude)
        np.testing.assert_allclose(offset, (0.0, 50.0), atol=1e-7, rtol=0.0)

    def test_requires_the_same_ordered_route_and_control_timestamps(self) -> None:
        self._write_replay(("outbound", "002", 55.0, None), ("outbound", "002", 60.0, None))
        reordered = replace(self.route, waypoints=tuple(reversed(self.route.waypoints)))
        with self.assertRaisesRegex(ValueError, "same ordered route"):
            TackPlanReplay(self.path, reordered, self.departure, 5.0)
        with self.assertRaisesRegex(ValueError, "same departure time"):
            TackPlanReplay(self.path, self.route, self.departure + np.timedelta64(1, "s"), 5.0)
        with self.assertRaisesRegex(ValueError, "same controller update interval"):
            TackPlanReplay(self.path, self.route, self.departure, 4.0)
        lines = self.path.read_text().splitlines()
        update = json.loads(lines[-1])
        update["observation"]["time"] = str(self.departure + np.timedelta64(11, "s"))
        self.path.write_text("\n".join([*lines[:-1], json.dumps(update)]) + "\n")
        with self.assertRaisesRegex(ValueError, "same controller update interval"):
            TackPlanReplay(self.path, self.route, self.departure, 5.0)

    def _position(self, step: int) -> Waypoint:
        return Waypoint(self.route.departure.latitude + step * 0.0001, self.route.departure.longitude)

    def _write_replay(self, *updates: tuple[str, str, float, float | None]) -> None:
        with NavigationRecorder(self.path, route=self.route.waypoints) as recorder:
            recorder.record(self._frame(0, {}))
            for step, (phase, name, course, bow) in enumerate(updates, 1):
                recorder.record(self._frame(step, {
                    "phase": phase,
                    "waypoint": name,
                    "heading_reference_deg": course,
                    "bow_reference_deg": bow,
                }))

    def _frame(self, step: int, navigation: dict) -> EpisodeFrame:
        position = self._position(step)
        state = BoatState(position.latitude, position.longitude, heading=0.0, speed=1.0)
        observation = WorldObservation(
            self.departure + np.timedelta64(step * 5, "s"), state,
            wind=None, wave=None, current=None, seabed=None, coast=None, chart=None, ais_targets=(),
        )
        return EpisodeFrame(state, observation, navigation=navigation)
