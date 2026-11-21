"""Tests for the local live navigation display."""

from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from world import AISTarget, BoatState, Waypoint, Wind, WorldObservation
from world.ais_course_transformer import AISCourseForecast
from world._geo import offset_latitude_longitude
from world.animate_episode import EpisodeFrame, SaildroneAnimation


def observation(
    state: BoatState,
    target: AISTarget,
    *additional_targets: AISTarget,
) -> WorldObservation:
    return WorldObservation(
        time=np.datetime64("2026-09-29T23:23:00"),
        boat=state,
        wind=None,
        wave=None,
        current=None,
        seabed=None,
        coast=None,
        chart=None,
        ais_targets=(target, *additional_targets),
    )


class SaildroneAnimationTests(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_recorded_wind_grid_draws_eight_barbs_clear_of_the_boat(self) -> None:
        state = BoatState(40.25, 18.55, heading=0.0, speed=1.0)
        target = AISTarget(1, 40.251, 18.551, 1.0, 90.0, 90.0)
        forecast = replace(observation(state, target), wind=Wind(3.0, 4.0, 5.0, 216.87))
        wind_grid = (np.full(25, 6.0), np.full(25, 8.0))
        animation = SaildroneAnimation(
            [EpisodeFrame(state, forecast, wind_grid=wind_grid)],
            interval_ms=1, local_frame=True, view_radius_metres=800.0, boat_scale=2.5,
        )

        animation.update(0)

        distances = np.hypot(*animation.wind_dots.get_data())
        self.assertEqual(len(distances), 8)
        self.assertGreater(distances.min(), animation.sail_length + animation.flap_length)
        self.assertEqual(animation.frames[0].wind_grid[0].shape, (25,))

    def test_local_ais_arrow_uses_relative_position_and_course(self) -> None:
        latitude, longitude = 40.25, 18.55
        target_latitude, target_longitude = offset_latitude_longitude(
            latitude,
            longitude,
            100.0,
            200.0,
        )
        state = BoatState(latitude, longitude, heading=0.0, speed=1.0)
        target = AISTarget(
            mmsi=374552000,
            latitude=target_latitude,
            longitude=target_longitude,
            speed=5.0,
            course=90.0,
            heading=90.0,
        )
        frame = EpisodeFrame(state, observation(state, target))
        animation = SaildroneAnimation(
            [frame],
            interval_ms=1,
            local_frame=True,
            view_radius_metres=1_000.0,
        )

        animation.update(0)

        east, north = animation.ais_targets.get_offsets()[0]
        self.assertAlmostEqual(float(east), 100.0, places=3)
        self.assertAlmostEqual(float(north), 200.0, places=3)
        self.assertGreater(float(animation.ais_targets.U[0]), 0.0)
        self.assertAlmostEqual(float(animation.ais_targets.V[0]), 0.0, places=6)

    def test_local_view_hides_start_dot_and_keeps_hull_visible(self) -> None:
        latitude, longitude = 40.25, 18.55
        state = BoatState(latitude, longitude, heading=0.0, speed=1.0)
        target = AISTarget(1, latitude, longitude, 1.0, 0.0, 0.0)
        animation = SaildroneAnimation(
            [EpisodeFrame(state, observation(state, target))],
            interval_ms=1,
            local_frame=True,
            view_radius_metres=2_500.0,
        )

        animation.update(0)

        self.assertFalse(animation.start_marker.get_visible())
        self.assertGreaterEqual(animation.boat_length, 50.0)
        self.assertLess(animation.boat_length, 150.0)
        self.assertEqual(animation.axis.get_title(), "Boat-centred local navigation (north up)")

    def test_scaled_bow_stays_at_heading_when_ground_motion_is_astern(self) -> None:
        state = BoatState(40.25, 18.55, heading=90.0, speed=0.15)
        target = AISTarget(1, state.latitude, state.longitude, 1.0, 0.0, 0.0)
        frame = EpisodeFrame(state, observation(state, target), navigation={"course_over_ground_deg": 270.0})
        animation = SaildroneAnimation([frame], interval_ms=1, boat_scale=7.0)

        with patch.object(animation.ground_course_arrow, "set_positions") as set_course:
            animation.update(0)

        length = animation.boat_length
        hull = animation.hull.get_xy()
        np.testing.assert_allclose(hull[2], [length / 2.0, 0.0], atol=1e-8)
        self.assertAlmostEqual(hull[2, 0] - hull[1, 0], 7.0 * 4.0)
        np.testing.assert_allclose(animation.bow_marker.get_data(), [[length / 2.0], [0.0]], atol=1e-8)
        self.assertAlmostEqual(animation.rudder.get_xdata()[0], -length / 2.0)
        start, tip = set_course.call_args.args
        self.assertLess(tip[0], start[0])
        self.assertLess(start[0], -length / 2.0)
        self.assertTrue(animation.ground_course_arrow.get_visible())
        self.assertIn("Bow heading", animation.readout.get_text())
        self.assertIn("Actual COG deg 270.00", animation.readout.get_text())

        animation.frames[0] = replace(frame, navigation={})
        animation.update(0)

        self.assertFalse(animation.ground_course_arrow.get_visible())

    def test_timed_ais_draws_age_adjusted_position_and_keeps_source_report(self) -> None:
        state = BoatState(40.25, 18.55, heading=0.0, speed=1.0)
        target = AISTarget(1, state.latitude, state.longitude, 2.0, 90.0, 90.0, report_age_seconds=20.0)
        frame = EpisodeFrame(state, observation(state, target), navigation={"ais_mode": "route"})
        animation = SaildroneAnimation([frame], interval_ms=1, local_frame=True)
        animation.update(0)

        np.testing.assert_allclose(animation.ais_targets.get_offsets(), [(40.0, 0.0)], atol=1e-6)
        self.assertIs(animation.frames[0].observation.ais_targets[0], target)
        self.assertIn("AIS arrows: age-adjusted COG/SOG estimates", animation.readout.get_text())

    def test_local_view_rebuilds_arrows_and_filters_distant_targets(self) -> None:
        latitude, longitude = 40.25, 18.55
        near_latitude, near_longitude = offset_latitude_longitude(
            latitude, longitude, 100.0, 0.0
        )
        far_latitude, far_longitude = offset_latitude_longitude(
            latitude, longitude, 2_000.0, 0.0
        )
        state = BoatState(latitude, longitude, heading=0.0, speed=1.0)
        near = AISTarget(1, near_latitude, near_longitude, 2.0, 90.0, 90.0)
        far = AISTarget(2, far_latitude, far_longitude, 2.0, 90.0, 90.0)
        frame = EpisodeFrame(state, observation(state, near, far))
        animation = SaildroneAnimation(
            [frame],
            interval_ms=1,
            local_frame=True,
            view_radius_metres=1_000.0,
        )

        animation.update(0)

        self.assertEqual(animation.ais_targets.N, 1)
        self.assertEqual(len(animation.ais_targets.get_offsets()), 1)

    def test_local_trajectory_moves_behind_centred_boat(self) -> None:
        latitude, longitude = 40.25, 18.55
        next_latitude, next_longitude = offset_latitude_longitude(
            latitude,
            longitude,
            20.0,
            0.0,
        )
        target = AISTarget(1, latitude, longitude, 1.0, 0.0, 0.0)
        start = BoatState(latitude, longitude, heading=90.0, speed=1.0)
        moved = BoatState(next_latitude, next_longitude, heading=90.0, speed=1.0)
        first = EpisodeFrame(start, observation(start, target))
        second = EpisodeFrame(moved, observation(moved, target))
        animation = SaildroneAnimation(
            [first],
            interval_ms=1,
            local_frame=True,
            view_radius_metres=1_000.0,
        )

        animation.add_frame(second)
        animation.update(1)

        path_east = np.asarray(animation.path.get_xdata(), dtype=float)
        self.assertAlmostEqual(float(path_east[-1]), 0.0, places=6)
        self.assertAlmostEqual(float(path_east[0]), -20.0, places=3)

    def test_fixed_local_view_moves_boat_across_map(self) -> None:
        latitude, longitude = 40.25, 18.55
        next_latitude, next_longitude = offset_latitude_longitude(
            latitude,
            longitude,
            20.0,
            0.0,
        )
        target = AISTarget(1, latitude, longitude, 1.0, 0.0, 0.0)
        start = BoatState(latitude, longitude, heading=90.0, speed=1.0)
        moved = BoatState(next_latitude, next_longitude, heading=90.0, speed=1.0)
        animation = SaildroneAnimation(
            [EpisodeFrame(start, observation(start, target))],
            interval_ms=1,
            fixed_local_view=True,
            view_radius_metres=1_000.0,
        )

        animation.add_frame(EpisodeFrame(moved, observation(moved, target)))
        animation.update(1)

        rendered_boat_east = float(animation.sail.get_xdata()[0])
        path_east = np.asarray(animation.path.get_xdata(), dtype=float)
        self.assertAlmostEqual(rendered_boat_east, 20.0, places=3)
        self.assertAlmostEqual(float(path_east[0]), 0.0, places=6)
        self.assertAlmostEqual(float(path_east[-1]), 20.0, places=3)
        self.assertEqual(
            animation.axis.get_title(),
            "Fixed local navigation from start (north up)",
        )
        self.assertIn("E   +20.0 m", animation.readout.get_text())

    def test_fixed_local_view_draws_mission_route(self) -> None:
        latitude, longitude = 40.25, 18.55
        end_latitude, end_longitude = offset_latitude_longitude(
            latitude,
            longitude,
            200.0,
            300.0,
        )
        state = BoatState(latitude, longitude, heading=0.0, speed=1.0)
        target = AISTarget(1, latitude, longitude, 1.0, 0.0, 0.0)
        animation = SaildroneAnimation(
            [EpisodeFrame(state, observation(state, target))],
            interval_ms=1,
            fixed_local_view=True,
            view_radius_metres=1_000.0,
        )

        animation.set_route(
            [Waypoint(latitude, longitude), Waypoint(end_latitude, end_longitude)]
        )

        route_east = np.asarray(animation.route_line.get_xdata(), dtype=float)
        route_north = np.asarray(animation.route_line.get_ydata(), dtype=float)
        self.assertAlmostEqual(float(route_east[0]), 0.0, places=6)
        self.assertAlmostEqual(float(route_north[0]), 0.0, places=6)
        self.assertAlmostEqual(float(route_east[1]), 200.0, places=3)
        self.assertAlmostEqual(float(route_north[1]), 300.0, places=3)

    def test_ais_forecast_line_uses_speed_horizon_and_predicted_direction(self) -> None:
        latitude, longitude = 40.25, 18.55
        target_latitude, target_longitude = offset_latitude_longitude(
            latitude, longitude, 100.0, 200.0
        )
        state = BoatState(latitude, longitude, heading=0.0, speed=1.0)
        target = AISTarget(
            374552000,
            target_latitude,
            target_longitude,
            2.0,
            90.0,
            90.0,
        )
        forecast = AISCourseForecast(
            mmsi=374552000,
            history_end=np.datetime64("2026-09-29T23:23:00").astype(object),
            horizons_minutes=(5, 10, 15),
            directions=np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]),
            courses_degrees=np.asarray([90.0, 0.0, 270.0]),
        )
        frame = EpisodeFrame(
            state,
            observation(state, target),
            {374552000: forecast},
        )
        animation = SaildroneAnimation(
            [frame],
            interval_ms=1,
            fixed_local_view=True,
            view_radius_metres=5_000.0,
        )

        animation.update(0)

        segment = np.asarray(animation.ais_forecast_lines[5].get_segments()[0])
        np.testing.assert_allclose(segment[0], [100.0, 200.0], atol=1e-3)
        np.testing.assert_allclose(segment[1], [700.0, 200.0], atol=1e-3)


if __name__ == "__main__":
    unittest.main()
