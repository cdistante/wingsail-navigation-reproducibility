"""Regression coverage for persisted navigation frames and playback."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib
import numpy as np
from matplotlib.text import Text
from pandas import Timestamp

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from world import (
    AISTarget, BoatState, ChartSample, Coast, Current, LocalGrid, RouteWaypoint,
    Seabed, Wave, Wind, WorldObservation,
)
from world._geo import offset_latitude_longitude
from world.ais_course_transformer import AISCourseForecast
from world.animate_episode import EpisodeFrame
from world import navigation_replay
from world.navigation_replay import (
    NavigationRecorder, NavigationReplayPlayer, export_navigation_video, load_navigation_replay,
)


class NavigationReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.addCleanup(plt.close, "all")
        self.file = Path(directory.name) / "run.jsonl.gz"
        self.state = BoatState(40.25, 18.55, 90.0, 1.2, rudder=5.0, flap=-8.0, heel=3.0)
        self.target = AISTarget(999000101, 40.251, 18.551, 4.1, 220.0, 219.0, name="AIS ferry")
        self.observation = WorldObservation(
            time=np.datetime64("2026-10-02T03:00:00.123456789", "ns"),
            boat=self.state,
            wind=Wind(3.0, 4.0, 5.0, 216.87),
            wave=Wave(1.2, 45.0, 4.0),
            current=Current(0.2, 0.0, 0.2, 90.0, 0.0),
            seabed=Seabed(-135.0, 135.0, False),
            coast=Coast(600.0, False),
            chart=ChartSample(False, False),
            ais_targets=(self.target,),
        )
        self.forecast = AISCourseForecast(
            self.target.mmsi, Timestamp("2026-10-02T03:00:00.123456789Z"), (5, 10, 15),
            np.asarray([[1., 0.], [0., 1.], [-1., 0.]]), np.asarray([90., 0., 270.]),
        )
        self.frame = EpisodeFrame(
            self.state, self.observation, {self.target.mmsi: self.forecast},
            navigation={"step": 3, "waypoint": "002", "heading_reference_deg": 92.3},
        )
        self.grid = LocalGrid(np.zeros((1, 3, 3)), ("depth",), 20.0, 60.0, 40.25, 18.55)
        self.mask = np.zeros((3, 3), dtype=bool)
        self.mask[0, 1] = True
        self.route = (RouteWaypoint(40.25, 18.55, "001"), RouteWaypoint(40.251, 18.552, "002"))

    def test_round_trip_preserves_the_display_inputs(self) -> None:
        with NavigationRecorder(self.file, route=self.route, route_name="Otranto") as recorder:
            recorded = recorder.record(self.frame, local_grid=self.grid, blocked=self.mask)

        replay = load_navigation_replay(self.file)
        restored = replay.frames[0]
        self.assertEqual(replay.route, self.route)
        self.assertEqual(replay.route_name, "Otranto")
        self.assertEqual(restored.state, self.state)
        self.assertEqual(restored.observation, self.observation)
        self.assertEqual(restored.navigation, self.frame.navigation)
        forecast = restored.ais_course_forecasts[self.target.mmsi]
        self.assertEqual(forecast.history_end, self.forecast.history_end)
        self.assertEqual(forecast.horizons_minutes, self.forecast.horizons_minutes)
        np.testing.assert_array_equal(forecast.directions, self.forecast.directions)
        np.testing.assert_array_equal(forecast.courses_degrees, self.forecast.courses_degrees)
        np.testing.assert_array_equal(restored.wind_grid, recorded.wind_grid)
        np.testing.assert_allclose(restored.wind_grid[0], self.observation.wind.u * 1.94384449)
        np.testing.assert_allclose(restored.wind_grid[1], self.observation.wind.v * 1.94384449)
        self.assertEqual(replay.grids[0].resolution, self.grid.resolution)
        np.testing.assert_array_equal(replay.grids[0].blocked, self.mask)

    def test_scrubbing_restores_the_occupancy_overlay_and_route_at_that_frame(self) -> None:
        latitude, longitude = offset_latitude_longitude(40.25, 18.55, 20.0, 0.0)
        moved = replace(self.state, latitude=float(latitude), longitude=float(longitude))
        still_water = Current(0.0, 0.0, 0.0, 0.0, 0.0)
        second_frame = EpisodeFrame(
            moved, replace(self.observation, boat=moved, current=still_water),
            navigation={"step": 4},
        )
        with NavigationRecorder(self.file, route=self.route, local_frame=True) as recorder:
            recorder.record(self.frame, local_grid=self.grid, blocked=self.mask)
            recorder.record(second_frame, local_grid=self.grid, blocked=~self.mask)

        player = NavigationReplayPlayer(load_navigation_replay(self.file))
        view = player.animation
        self.assertIn("Step: 0003", view.readout.get_text())
        self.assertIn("Waypoint 002", view.readout.get_text())
        self.assertEqual(len(view.ais_forecast_lines[5].get_segments()), 1)
        self.assertTrue(view.local_grid_image.get_visible())
        self.assertTrue(view.current_arrow.get_visible())
        np.testing.assert_allclose(view.local_grid_image.get_extent(), [-30, 30, -30, 30])
        self.assertAlmostEqual(view.route_line.get_xdata()[0], 0.0)
        self.assertEqual(view.route_labels[0].get_text(), "001")

        player._slider.set_val(1)
        self.assertIn("Step: 0004", view.readout.get_text())
        self.assertEqual(len(view.ais_forecast_lines[5].get_segments()), 0)
        self.assertFalse(view.current_arrow.get_visible())
        np.testing.assert_array_equal(view.local_grid_image.get_array(), ~self.mask)
        np.testing.assert_allclose(view.local_grid_image.get_extent(), [-50, 10, -30, 30], atol=1e-6)
        self.assertAlmostEqual(view.route_line.get_xdata()[0], -20.0, places=6)
        self.assertAlmostEqual(view.route_labels[0].xy[0], -20.0, places=6)
        player._slider.set_val(0)
        np.testing.assert_array_equal(view.local_grid_image.get_array(), self.mask)
        self.assertEqual(len(view.ais_forecast_lines[5].get_segments()), 1)
        self.assertTrue(view.current_arrow.get_visible())

        player.toggle()
        self.assertEqual(player._play_button.label.get_text(), "Pause")
        player._advance()
        self.assertEqual(player._slider.val, 1)
        player._advance()
        self.assertEqual(player._play_button.label.get_text(), "Play")

    def test_video_renders_every_frame_without_player_controls(self) -> None:
        methods = ("acados NMPC · Direct path following", "Fossen 3-DOF · Gamma harmonic planner")
        with NavigationRecorder(self.file, route=self.route, interval_ms=125, method_lines=methods) as recorder:
            recorder.record(self.frame, local_grid=self.grid, blocked=self.mask)
            recorder.record(replace(self.frame, navigation={"step": 4}))
        replay = load_navigation_replay(self.file)
        self.assertEqual(replay.method_lines, methods)
        destination = self.file.with_name("video.mp4")

        @contextmanager
        def save_video(figure, output_path, *, dpi):
            labels = {text.get_text() for text in figure.findobj(Text)}
            self.assertTrue(labels.isdisjoint({"Play", "Pause", "Frame"}))
            self.assertEqual(dpi, figure.dpi)
            yield
            self.assertTrue(any(
                text.get_text().startswith("\n".join(methods) + "\n\nStep: 0004")
                for text in figure.findobj(Text)
            ))
            output_path.write_bytes(b"encoded video")

        with (
            patch.object(navigation_replay, "_ffmpeg_path", return_value="ffmpeg"),
            patch.object(navigation_replay, "FFMpegWriter") as writer,
            patch.object(
                navigation_replay, "_render_replay_frame", wraps=navigation_replay._render_replay_frame,
            ) as render,
        ):
            writer.return_value.saving.side_effect = save_video
            self.assertEqual(export_navigation_video(replay, destination), destination)

        self.assertEqual(writer.call_args.kwargs["fps"], 8.0)
        self.assertEqual(writer.return_value.grab_frame.call_count, 2)
        self.assertEqual([call.args[2] for call in render.call_args_list], [0, 1])
        self.assertEqual(destination.read_bytes(), b"encoded video")
        with self.assertRaises(FileExistsError):
            export_navigation_video(replay, destination)

    def test_video_rejects_invalid_rate_and_extension(self) -> None:
        with NavigationRecorder(self.file) as recorder:
            recorder.record(self.frame)
        replay = load_navigation_replay(self.file)
        with self.assertRaisesRegex(ValueError, "must end in .mp4"):
            export_navigation_video(replay, self.file.with_name("video.gif"))
        for fps in (0, float("nan"), float("inf")):
            with self.subTest(fps=fps), self.assertRaisesRegex(ValueError, "finite and positive"):
                export_navigation_video(replay, self.file.with_name("video.mp4"), fps=fps)


if __name__ == "__main__":
    unittest.main()
