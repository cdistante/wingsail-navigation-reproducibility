"""Render and verify representative labeled MP4s from saved raw replays."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import gzip
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from unittest.mock import patch
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
from matplotlib.patches import Circle
import numpy as np

from run_study import METHODS, ROUTES, STUDY, sha256_stream, write_json
from world import navigation_replay as replay_io
from world._geo import local_east_north_metres


MATCHED = tuple(f"brindisi_short__090-05__{method}" for method in METHODS)
ROUTE_VIEWS = (
    "brindisi_003__000-05__forecast",
    "brindisi_004__090+05__forecast",
    "costiera__180-05__forecast",
    "otranto_gate__270+05__forecast",
    "otranto_003__000+05__forecast",
    "otranto_005__090-05__forecast",
    "naples__180+05__forecast",
    "corfu__270-05__forecast",
)
REPRESENTATIVE = MATCHED + ROUTE_VIEWS
MISSION_PREFIX = "Mission outcome: "
DISPLAY_KEYS = {
    "step", "phase", "waypoint", "waypoint_distance_m", "heading_reference_deg",
    "course_over_ground_deg", "bow_reference_deg", "forecast_tack_in_min",
    "forecast_closest_m", "rudder_command_deg", "flap_command_deg", "solver_fallback",
}
OUTCOMES = {"completed": "Completed", "step_limit": "Step limit",
            "no_chart_clear_action": "No chart-clear action"}


def mission_label(status: str) -> str:
    return OUTCOMES.get(status, status.replace("_", " ").capitalize())


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return sha256_stream(stream)


def run_folder(study: Path, run_id: str) -> Path:
    if re.fullmatch(r"[a-z0-9_]+__\d{3}[+-]\d{2}__[a-z0-9_]+", run_id) is None:
        raise ValueError(f"Invalid navigation run ID {run_id!r}")
    route, _, method = run_id.split("__")
    if route not in ROUTES or method not in METHODS:
        raise ValueError(f"Unknown navigation route or method in {run_id!r}")
    return study / "runs" / run_id


def ready(folder: Path) -> bool:
    receipt = folder / "execution.json"
    if not receipt.is_file() or not all(
        (folder / name).is_file() for name in ("navigation.jsonl.gz", "track.gpx", "run.log")
    ):
        return False
    result = json.loads(receipt.read_text())
    if result["return_code"] == 0:
        return True
    try:
        return recorded_mission(folder, result["return_code"])["status"] == "no_feasible_heading"
    except (OSError, ValueError, KeyError):
        return False


def recorded_mission(folder: Path, return_code: int) -> dict:
    if return_code != 0:
        from summarize import load_mission

        return load_mission(folder)
    [mission] = [json.loads(line.removeprefix(MISSION_PREFIX))
                 for line in (folder / "run.log").read_text().splitlines()
                 if line.startswith(MISSION_PREFIX)]
    return mission


def camera_for(study: Path, group: tuple[str, ...]) -> list[float]:
    folder = run_folder(study, group[0])
    with gzip.open(folder / "navigation.jsonl.gz", "rt", encoding="utf-8") as stream:
        route = json.loads(next(stream))["route"]
        origin = json.loads(next(stream))["state"]
    points = [(point["latitude"], point["longitude"]) for point in route]
    for run_id in group:
        track = ET.parse(run_folder(study, run_id) / "track.gpx")
        points.extend(
            (float(point.attrib["lat"]), float(point.attrib["lon"]))
            for point in track.iter() if point.tag.rsplit("}", 1)[-1] == "trkpt"
        )
    coordinates = np.asarray(points)
    east, north = local_east_north_metres(
        coordinates[:, 0], coordinates[:, 1], origin["latitude"], origin["longitude"],
    )
    low, high = np.array([east.min(), north.min()]), np.array([east.max(), north.max()])
    margin = max(90.0, 0.11 * float(max(high - low)))
    half_span = np.maximum((high - low) / 2 + margin, [330.0, 260.0])
    middle = (high + low) / 2
    return [float(middle[0] - half_span[0]), float(middle[0] + half_span[0]),
            float(middle[1] - half_span[1]), float(middle[1] + half_span[1])]


def choose_frames(count: int, capture_steps: list[int], limit: int) -> list[int]:
    retain = {0, count - 1}
    for step in capture_steps:
        retain.update(range(max(step - 1, 0), min(step + 2, count)))
    if len(retain) > limit:
        raise ValueError(f"The requested frame limit {limit} cannot retain the capture events.")
    if count <= limit:
        return list(range(count))
    regular = [index for index in range(count) if index not in retain]
    sampling = np.linspace(0, len(regular) - 1, limit - len(retain), dtype=int)
    return sorted(retain | {regular[index] for index in sampling})


def displayed_frame(frame, step: int, capture: str | None):
    navigation = {key: value for key, value in frame.navigation.items() if key in DISPLAY_KEYS}
    navigation["step"] = step
    extras = {
        "learned_predicted_progress_gain_m": "predicted_progress_gain_m",
        "learned_predicted_arrival_gain_s": "predicted_eta_gain_s",
    }
    decision = frame.navigation.get("learned_tack_selection")
    if decision is not None:
        navigation["tack_selection"] = {
            "classical input outside training support": "classical · outside model support",
            "classical no-progress forecast": "classical · forecast no progress",
        }.get(decision, decision)
    for key, label in extras.items():
        value = frame.navigation.get(key)
        if value is not None:
            navigation[label] = value
    if capture is not None:
        navigation["waypoint_capture"] = capture
    return replace(frame, navigation=navigation)


def chart_animation(camera: list[float], radius_m: float):
    span = max(camera[1] - camera[0], camera[3] - camera[2])
    scale = max(1.0, span / 1800.0)

    class StudyAnimation(replay_io.SaildroneAnimation):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.axis.set_xlim(camera[:2])
            self.axis.set_ylim(camera[2:])
            self.axis.set_title("Saved route and vessel track · north up")
            self.path.set_color("#144364")
            self.path.set_linewidth(2.0)
            side = self.WIND_GRID_SIDE
            self.drawn_wind_cells[:] = False
            self.drawn_wind_cells[[0, side - 1, side * (side - 1), side * side - 1]] = True

        @classmethod
        def _wind_grid_offsets(cls):
            return super()._wind_grid_offsets() * scale

        def _set_environment_indicators(self, mast, frame):
            super()._set_environment_indicators(mast, frame)
            if self.current_arrow.get_visible() and scale > 1.0:
                current = frame.observation.current
                length = scale * min(40.0, 15.0 + 100.0 * current.speed)
                self.current_arrow.set_positions(mast, mast + length * self._bearing_vector(current.direction))

        def set_route(self, route):
            super().set_route(route)
            for index, point in enumerate(self.route_positions[1:]):
                disk = Circle(point, radius_m, alpha=0.16, facecolor="#2477b8", edgecolor="none",
                              zorder=1.0, label=f"{radius_m:g} m capture radius" if index == 0 else None)
                self.axis.add_patch(disk)
                if np.any(np.linalg.norm(self.route_positions[:index + 1] - point, axis=1) < 0.015 * span):
                    label = self.route_labels[index + 1]
                    label.set_position((-6, -12))
                    label.set_horizontalalignment("right")
            handles, labels = self.axis.get_legend_handles_labels()
            visible = [(handle, label) for handle, label in zip(handles, labels) if not label.startswith("AIS")]
            self.readout.axes.legend(*zip(*visible), loc="lower left", ncol=2, fontsize=8.5)

    return StudyAnimation


def verify_decode(path: Path, expected_frames: int) -> None:
    check = subprocess.run(
        [replay_io._ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-xerror", "-threads", "1",
         "-i", str(path), "-map", "0:v:0", "-progress", "pipe:1", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    ).stdout
    counts = re.findall(r"^frame=(\d+)$", check, re.MULTILINE)
    if not counts or int(counts[-1]) != expected_frames or "progress=end" not in check:
        raise ValueError(f"{path.name}: decoded {counts[-1:]}, expected {expected_frames} frames")


def render(study: Path, run_id: str, camera: list[float], fps: float, frame_limit: int) -> dict:
    folder = run_folder(study, run_id)
    replay_path, track = folder / "navigation.jsonl.gz", folder / "track.gpx"
    meta = json.loads((folder / "execution.json").read_text())
    mission = recorded_mission(folder, meta["return_code"])
    replay = replay_io.load_navigation_replay(replay_path)
    count = len(replay.frames)
    if count != mission["steps"] + 1:
        raise ValueError(f"{run_id}: replay and run log record different step counts.")
    arrivals = mission["waypoint_arrivals"]
    captures = defaultdict(list)
    for point in arrivals:
        captures[point["step"]].append(point["waypoint"])
    wind_label = f"Wind FROM {meta['wind_from']:g}°"
    if not any(label.startswith(wind_label) for label in replay.method_lines):
        raise ValueError(f"{run_id}: the recording is missing the experimental wind label.")
    chosen = choose_frames(count, list(captures), frame_limit)
    elapsed = mission["steps"] * meta["dt_s"]
    outcome = mission_label(mission["status"])
    text = (f"{replay.route_name.replace('_', ' ')} · {meta['departure'].replace('T', ' ').replace('Z', ' UTC')}",
            *replay.method_lines,
            f"{outcome} · {len(arrivals)}/{len(replay.route) - 1} waypoints",
            f"{elapsed:g} s elapsed · {meta['cap_s']:g} s limit")
    span = max(camera[1] - camera[0], camera[3] - camera[2])
    view = replace(replay.view, fixed_local_view=True, local_frame=False,
                   boat_scale=max(replay.view.boat_scale, 0.024 * span / 18.0))
    visual = replace(
        replay, view=view, method_lines=text,
        frames=tuple(displayed_frame(replay.frames[index], index, " → ".join(captures[index]) or None)
                     for index in chosen),
        grids=tuple(replay.grids[index] for index in chosen),
    )
    video = study / "videos" / f"{run_id}.mp4"
    receipt = video.with_suffix(".json")
    config = {
        "run_id": run_id, "route": meta["route"], "method": meta["method"], "scope": meta["scope"],
        "method_label": meta["method_label"], "wind_from": meta["wind_from"], "departure": meta["departure"],
        "labels": text, "mission_status": mission["status"], "waypoint_count": len(replay.route) - 1,
        "arrivals": arrivals, "simulated_elapsed_s": elapsed, "cap_s": meta["cap_s"],
        "radius_m": meta["radius_m"], "camera_en_m": camera, "fps": fps, "source_frame_indices": chosen,
        "replay_frames": count, "replay_sha256": digest(replay_path), "gpx_sha256": digest(track),
        "render_script_sha256": digest(Path(__file__)),
        "video": str(video.relative_to(study)), "replay": str(replay_path.relative_to(study)),
        "gpx": str(track.relative_to(study)),
    }
    if video.is_file() and receipt.is_file():
        stored = json.loads(receipt.read_text())
        expected = {**config, "labels": list(text)}
        # Keep the generation hash in the receipt when re-verifying an existing clip.
        expected.pop("render_script_sha256")
        if all(stored.get(key) == val for key, val in expected.items()) and stored["video_sha256"] == digest(video):
            verify_decode(video, len(chosen))
            return stored
        raise ValueError(f"{run_id}: existing video uses different inputs. Choose a different destination.")
    if video.is_file():
        raise ValueError(f"{run_id}: existing video lacks a verification receipt.")

    class SingleThreadWriter(replay_io.FFMpegWriter):
        def __init__(self, *args, extra_args=None, **kwargs):
            super().__init__(*args, extra_args=[*(extra_args or ()), "-threads", "1"], **kwargs)

    with (patch.object(replay_io, "SaildroneAnimation", chart_animation(camera, meta["radius_m"])),
          patch.object(replay_io, "FFMpegWriter", SingleThreadWriter)):
        replay_io.export_navigation_video(visual, video, fps=fps)
    verify_decode(video, len(chosen))
    record = {**config, "decoded_frames": len(chosen), "video_sha256": digest(video)}
    write_json(receipt, record)
    print(f"{run_id}: verified {len(chosen)}/{count} frames", flush=True)
    return record


def write_index(study: Path) -> None:
    records = [json.loads(path.read_text()) for path in (study / "videos").glob("*.json")]
    by_id = {row["run_id"]: row for row in records}
    ordered = [by_id[run_id] for run_id in REPRESENTATIVE if run_id in by_id]
    extras = [by_id[run_id] for run_id in sorted(by_id.keys() - set(REPRESENTATIVE))]
    write_json(study / "video_verification.json", [*ordered, *extras])
    lines = [
        "# Wind direction study playback", "",
        "The default Brindisi selection uses 85° wind FROM and identical north-up cameras across all ten methods.",
        "The default selection also includes classical forecast clips on the other eight GPXs, each with a preset wind.",
        "Each MP4 retains the first and last frame and the frames immediately around each capture. Raw replays "
        "preserve every recorded five-second update. Shading shows local blocked grid cells at the displayed frame. "
        "Chart symbols are enlarged for visibility.", "",
        "| Route | Wind FROM | Method | Waypoints | Simulated time | Video and files |",
        "|---|---:|---|---|---:|---|",
    ]
    for row in [*ordered, *extras]:
        path = str(Path(ROUTES[row["route"]].file).stem)
        result = f"{len(row['arrivals'])}/{row['waypoint_count']} · {mission_label(row['mission_status'])}"
        assets = " · ".join(f"[{label}]({row[key]})" for label, key in
                            (("MP4", "video"), ("Replay", "replay"), ("GPX", "gpx")))
        lines.append(f"| {path} | {row['wind_from']:g}° | {row['method_label']} | {result} | "
                     f"{row['simulated_elapsed_s']:g} s | {assets} |")
    lines.extend(("", "[Frame indices, method labels, and SHA-256 digests](video_verification.json)", ""))
    (study / "playback.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--run-id", action="append", help="Export a specific saved run ID. Repeat for more runs.")
    parser.add_argument("--ready", action="store_true", help="Skip runs whose execution receipts are pending.")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--frame-limit", type=int, default=300)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--camera", type=float, nargs=4, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.jobs < 1 or args.frame_limit < 2 or not math.isfinite(args.fps) or args.fps <= 0.0:
        parser.error("Use at least one worker, two video frames and a positive finite frame rate.")
    study = args.study.resolve()
    selected = tuple(dict.fromkeys(args.run_id or REPRESENTATIVE))
    if args.camera is not None:
        if len(selected) != 1:
            parser.error("Render one run per worker.")
        render(study, selected[0], args.camera, args.fps, args.frame_limit)
        return

    camera_by_group = {}
    commands = []
    pending = []
    for run_id in selected:
        group = MATCHED if run_id in MATCHED else (run_id,)
        if not all(ready(run_folder(study, member)) for member in group):
            pending.append(run_id)
            continue
        if group not in camera_by_group:
            camera_by_group[group] = camera_for(study, group)
        commands.append([sys.executable, "-B", str(Path(__file__).resolve()), "--study", str(study),
                         "--run-id", run_id, "--fps", str(args.fps), "--frame-limit", str(args.frame_limit),
                         "--camera", *(str(value) for value in camera_by_group[group])])
    if pending and not args.ready:
        raise SystemExit("Runs still pending: " + ", ".join(pending))
    with ThreadPoolExecutor(max_workers=args.jobs) as workers:
        list(workers.map(lambda command: subprocess.run(command, check=True), commands))
    write_index(study)
    print(f"{len(commands)} clips verified, {len(pending)} pending from this selection.", flush=True)


if __name__ == "__main__":
    main()
