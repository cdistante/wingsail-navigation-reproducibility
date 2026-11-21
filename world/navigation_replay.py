"""Record live navigation frames and play them without running the simulator."""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from subprocess import CalledProcessError
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.widgets import Button, Slider
from pandas import Timestamp

from world import (
    AISTarget,
    BoatState,
    ChartSample,
    Coast,
    Current,
    LocalGrid,
    RouteWaypoint,
    Seabed,
    Wave,
    Waypoint,
    Weather,
    Wind,
    WorldObservation,
)
from world._geo import local_east_north_metres
from world.ais_course_transformer import AISCourseForecast
from world.animate_episode import EpisodeFrame, SaildroneAnimation


FORMAT = "maritime_world.navigation_replay"
VERSION = 1


@dataclass(frozen=True)
class ReplayView:
    interval_ms: int
    show_environment: bool
    view_radius_metres: float
    boat_scale: float
    local_frame: bool
    fixed_local_view: bool


@dataclass(frozen=True)
class ReplayGrid:
    blocked: np.ndarray
    resolution: float
    origin_latitude: float
    origin_longitude: float


@dataclass(frozen=True)
class NavigationReplay:
    route: tuple[RouteWaypoint, ...]
    route_name: str | None
    view: ReplayView
    frames: tuple[EpisodeFrame, ...]
    grids: tuple[ReplayGrid | None, ...]
    method_lines: tuple[str, ...] = ()


def _open_text(path: Path, mode: str):
    opener = gzip.open if path.suffix == ".gz" else open
    return opener(path, mode, encoding="utf-8")


def _encode_json(value):
    if isinstance(value, np.datetime64):
        return np.datetime_as_string(value, unit="ns")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Cannot record a {type(value).__name__} value in navigation replay.")


class NavigationRecorder:
    """Stream data for every displayed frame as optionally compressed JSON lines."""

    def __init__(
        self,
        path: Path | str,
        *,
        route: Iterable[Waypoint] = (),
        route_name: str | None = None,
        interval_ms: int = 100,
        show_environment: bool = True,
        view_radius_metres: float = 250.0,
        boat_scale: float = 1.0,
        local_frame: bool = False,
        fixed_local_view: bool = False,
        method_lines: tuple[str, ...] = (),
    ) -> None:
        self.view = ReplayView(
            interval_ms, show_environment, view_radius_metres, boat_scale,
            local_frame, fixed_local_view,
        )
        header = {
            "format": FORMAT,
            "version": VERSION,
            "route": [asdict(waypoint) for waypoint in route],
            "route_name": route_name,
            "view": asdict(self.view),
        }
        if method_lines:
            header["method_lines"] = list(method_lines)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = _open_text(path, "xt")
        try:
            self._write(header)
        except Exception:
            self.close()
            raise

    def __enter__(self) -> NavigationRecorder:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        self._file.close()

    def _write(self, row: dict) -> None:
        line = json.dumps(row, default=_encode_json, allow_nan=False, separators=(",", ":"))
        self._file.write(line + "\n")

    def record(
        self,
        frame: EpisodeFrame,
        *,
        local_grid: LocalGrid | None = None,
        blocked: np.ndarray | None = None,
        weather: Weather | None = None,
    ) -> EpisodeFrame:
        """Append a frame and return it with sampled wind data for the live view."""

        grid = None
        if local_grid is not None and blocked is not None:
            if blocked.shape != (local_grid.cells, local_grid.cells):
                raise ValueError("Replay obstacle mask must match the local grid.")
            grid = {
                "cells": local_grid.cells,
                "resolution": local_grid.resolution,
                "origin_latitude": local_grid.origin_latitude,
                "origin_longitude": local_grid.origin_longitude,
                "blocked": base64.b64encode(np.packbits(blocked).tobytes()).decode("ascii"),
            }
        elif local_grid is not None or blocked is not None:
            raise ValueError("A replay occupancy overlay requires both grid and mask.")

        if self.view.show_environment and frame.observation.wind is not None and frame.wind_grid is None:
            frame = replace(frame, wind_grid=SaildroneAnimation.sample_wind_grid(
                frame.state, frame.observation, weather,
            ))
        self._write({
            "state": asdict(frame.state),
            "observation": asdict(frame.observation),
            "ais_course_forecasts": [asdict(forecast) for forecast in frame.ais_course_forecasts.values()],
            "navigation": frame.navigation,
            "wind_grid": frame.wind_grid,
            "grid": grid,
        })
        return frame


def _decode_grid(row: dict | None) -> ReplayGrid | None:
    if row is None:
        return None
    cells = row["cells"]
    raw = base64.b64decode(row["blocked"], validate=True)
    if not isinstance(cells, int) or cells <= 0 or len(raw) != (cells * cells + 7) // 8:
        raise ValueError("Invalid replay obstacle mask dimensions.")
    values = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), count=cells * cells)
    return ReplayGrid(
        values.reshape(cells, cells).astype(bool), row["resolution"],
        row["origin_latitude"], row["origin_longitude"],
    )


def _decode_episode_frame(row: dict) -> EpisodeFrame:
    observation = row["observation"]
    observation["boat"] = BoatState(**observation["boat"])
    observation["time"] = np.datetime64(observation["time"], "ns")
    observation["ais_targets"] = tuple(AISTarget(**target) for target in observation["ais_targets"])
    for name, value_type in (
        ("wind", Wind), ("wave", Wave), ("current", Current), ("seabed", Seabed),
        ("coast", Coast), ("chart", ChartSample),
    ):
        if observation[name] is not None:
            observation[name] = value_type(**observation[name])

    forecasts = {}
    for data in row["ais_course_forecasts"]:
        forecast = AISCourseForecast(
            mmsi=data["mmsi"],
            history_end=Timestamp(data["history_end"]),
            horizons_minutes=tuple(data["horizons_minutes"]),
            directions=np.asarray(data["directions"], dtype=float),
            courses_degrees=np.asarray(data["courses_degrees"], dtype=float),
        )
        forecasts[forecast.mmsi] = forecast

    wind_grid = None
    if row["wind_grid"] is not None:
        east, north = row["wind_grid"]
        wind_grid = (np.asarray(east, dtype=float), np.asarray(north, dtype=float))
    return EpisodeFrame(
        BoatState(**row["state"]), WorldObservation(**observation), forecasts,
        navigation=row["navigation"], wind_grid=wind_grid,
    )


def load_navigation_replay(path: Path | str) -> NavigationReplay:
    """Read a recording for inspection or playback without opening its data sources."""

    with _open_text(Path(path), "rt") as recording:
        header = json.loads(recording.readline())
        if not isinstance(header, dict) or header.get("format") != FORMAT or header.get("version") != VERSION:
            raise ValueError("Unsupported navigation replay format or version.")
        view = ReplayView(**header["view"])
        method_lines = header.get("method_lines", [])
        if not isinstance(method_lines, list) or any(not isinstance(line, str) for line in method_lines):
            raise ValueError("Invalid navigation replay method labels.")
        route = tuple(RouteWaypoint(**point) for point in header["route"])
        frames, grids = [], []
        for line_number, line in enumerate(recording, start=2):
            try:
                row = json.loads(line)
                grids.append(_decode_grid(row["grid"]))
                frames.append(_decode_episode_frame(row))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid navigation replay frame on line {line_number}: {error}") from error
    if not frames:
        raise ValueError("The navigation recording contains no frames.")
    return NavigationReplay(route, header["route_name"], view, tuple(frames), tuple(grids), tuple(method_lines))


def _render_replay_frame(animation: SaildroneAnimation, grid: ReplayGrid | None, index: int) -> None:
    if grid is None:
        animation.local_grid_image.set_visible(False)
    else:
        origin = animation.origin
        east, north = local_east_north_metres(
            grid.origin_latitude, grid.origin_longitude, origin.latitude, origin.longitude,
        )
        animation.set_local_grid(grid.blocked, grid.resolution, east, north, frame_index=index)
    animation.update(index)


def _ffmpeg_path() -> str:
    if FFMpegWriter.isAvailable():
        return FFMpegWriter.bin_path()
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        return get_ffmpeg_exe()
    except (ImportError, RuntimeError) as error:
        raise RuntimeError(
            "MP4 export requires FFmpeg on PATH or the optional imageio-ffmpeg package."
        ) from error


def export_navigation_video(
    replay: NavigationReplay,
    destination: Path | str,
    *,
    fps: float | None = None,
) -> Path:
    """Render each recorded frame to H.264 MP4 using the replay's navigation view."""

    path = Path(destination)
    if path.suffix.lower() != ".mp4":
        raise ValueError("The video output path must end in .mp4.")
    if fps is None:
        if replay.view.interval_ms <= 0:
            raise ValueError("The recorded interval must be positive when deriving the video frame rate.")
        fps = 1000.0 / replay.view.interval_ms
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("The video frame rate must be finite and positive.")
    if path.exists():
        raise FileExistsError(f"The video output already exists: {path}")
    ffmpeg = _ffmpeg_path()

    animation = SaildroneAnimation(list(replay.frames), method_lines=replay.method_lines, **asdict(replay.view))
    animation.set_route(replay.route)
    figure = animation.figure
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with plt.rc_context({"animation.ffmpeg_path": ffmpeg}):
            writer = FFMpegWriter(
                fps=fps, codec="h264", extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            )
            with TemporaryDirectory(prefix=".navigation_video_", dir=path.parent) as directory:
                temporary_path = Path(directory) / "replay.mp4"
                with writer.saving(figure, temporary_path, dpi=figure.dpi):
                    for index, grid in enumerate(replay.grids):
                        _render_replay_frame(animation, grid, index)
                        if index == 0:
                            figure.canvas.draw()
                        writer.grab_frame()
                if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
                    raise RuntimeError("FFmpeg did not produce a video.")
                if path.exists():
                    raise FileExistsError(f"The video output already exists: {path}")
                temporary_path.replace(path)
    finally:
        plt.close(figure)
    return path


class NavigationReplayPlayer:
    """Render a recording with a pause button and a seekable frame slider."""

    def __init__(self, replay: NavigationReplay, *, interval_ms: int | None = None) -> None:
        self.replay = replay
        view_options = asdict(replay.view)
        if interval_ms is not None:
            view_options["interval_ms"] = interval_ms
        self.animation = SaildroneAnimation(
            list(replay.frames), constrained_layout=False, method_lines=replay.method_lines, **view_options,
        )
        self.animation.set_route(replay.route)
        self.figure = self.animation.figure
        self.figure.subplots_adjust(bottom=0.17)
        if self.figure.canvas.manager is not None:
            route_label = f" — {replay.route_name}" if replay.route_name else ""
            self.figure.canvas.manager.set_window_title(f"Saildrone replay{route_label}")

        self._last_index = len(replay.frames) - 1
        self._timer = self.figure.canvas.new_timer(interval=self.animation.interval_ms)
        self._timer.add_callback(self._advance)
        self._playing = False
        button_axis = self.figure.add_axes((0.13, 0.05, 0.10, 0.04))
        self._play_button = Button(button_axis, "Play")
        self._play_button.on_clicked(self.toggle)
        slider_axis = self.figure.add_axes((0.32, 0.05, 0.53, 0.04))
        self._slider = Slider(slider_axis, "Frame", 0, max(1, self._last_index), valinit=0, valstep=1, valfmt="%d")
        self._slider.drawon = False  # The seek callback draws after the artists have updated.
        self._slider.set_active(self._last_index > 0)
        self._slider.on_changed(self._seek)
        self.figure.canvas.mpl_connect("button_press_event", self._on_click)
        self.figure.canvas.mpl_connect("key_press_event", self._on_key)
        self.figure.canvas.mpl_connect("close_event", self._pause)
        self._seek(0)

    def _seek(self, value: float) -> None:
        index = min(int(value), self._last_index)
        _render_replay_frame(self.animation, self.replay.grids[index], index)
        self.figure.canvas.draw_idle()

    def _advance(self) -> None:
        if self._slider.val >= self._last_index:
            self._pause()
        else:
            self._slider.set_val(self._slider.val + 1)

    def _pause(self, _event=None) -> None:
        self._timer.stop()
        self._playing = False
        self._play_button.label.set_text("Play")
        self.figure.canvas.draw_idle()

    def toggle(self, _event=None) -> None:
        if self._playing:
            self._pause()
        elif self._last_index:
            if self._slider.val >= self._last_index:
                self._slider.set_val(0)
            self._play_button.label.set_text("Pause")
            self._playing = True
            self._timer.start()
            self.figure.canvas.draw_idle()

    def _on_key(self, event) -> None:
        if event.key == " ":
            self.toggle()
        elif event.key in {"left", "right"}:
            self._pause()
            direction = -1 if event.key == "left" else 1
            self._slider.set_val(np.clip(self._slider.val + direction, 0, self._last_index))

    def _on_click(self, event) -> None:
        if self._playing and event.inaxes == self._slider.ax:
            self._pause()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Space toggles playback. Left/right arrows step through frames while paused.",
    )
    parser.add_argument("recording", type=Path, help="Navigation recording (.jsonl or .jsonl.gz).")
    parser.add_argument("--interval-ms", type=int, help="Override the recorded playback interval.")
    parser.add_argument("--export-video", type=Path, metavar="PATH", help="Save the replay as an MP4 file.")
    parser.add_argument("--fps", type=float, help="Video frame rate. Defaults to the playback rate.")
    arguments = parser.parse_args()
    if arguments.interval_ms is not None and arguments.interval_ms < 1:
        parser.error("--interval-ms must be positive.")
    if arguments.fps is not None and arguments.export_video is None:
        parser.error("--fps requires --export-video.")
    if arguments.fps is not None and (not math.isfinite(arguments.fps) or arguments.fps <= 0):
        parser.error("--fps must be finite and positive.")
    try:
        replay = load_navigation_replay(arguments.recording)
    except (EOFError, KeyError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    if arguments.export_video is not None:
        plt.switch_backend("Agg")
        fps = arguments.fps
        if fps is None and arguments.interval_ms is not None:
            fps = 1000.0 / arguments.interval_ms
        try:
            path = export_navigation_video(replay, arguments.export_video, fps=fps)
        except CalledProcessError as error:
            parser.error(f"FFmpeg could not encode the MP4 file (exit status {error.returncode}).")
        except (OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        print(f"Saved replay video to {path}")
        return

    player = NavigationReplayPlayer(replay, interval_ms=arguments.interval_ms)
    player.toggle()
    plt.show()


if __name__ == "__main__":
    main()
