"""Export the matched Brindisi 95-degree full and masked arrival replays."""

from __future__ import annotations

import argparse
from dataclasses import replace
from os.path import relpath
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory
from unittest.mock import patch

import render
from run_study import STUDY, write_json
from run_transition_ablation import METHOD


CONTROL = "brindisi_short__090+05__arrival"
MASKED = "brindisi_short__090+05__" + METHOD
FEATURE_LABELS = {
    CONTROL: "Full arrival selector · continuation inputs enabled",
    MASKED: "Ablated arrival selector · continuation inputs masked",
}


def render_pair(main: Path, output: Path):
    folders = {
        CONTROL: main / "runs" / CONTROL,
        MASKED: output / "runs" / MASKED,
    }
    if not all(render.ready(folder) for folder in folders.values()):
        raise ValueError("Both Brindisi 95° replays must finish before export.")
    load = render.replay_io.load_navigation_replay

    def add_feature_label(path):
        replay = load(path)
        label = FEATURE_LABELS[path.parent.name]
        return replace(replay, method_lines=(*replay.method_lines, label))

    render.METHODS[METHOD] = render.METHODS["arrival"]
    videos = output / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    records = []
    with TemporaryDirectory(prefix="transition_ablation_video_") as directory:
        stage = Path(directory)
        (stage / "runs").mkdir()
        for run_id, folder in folders.items():
            (stage / "runs" / run_id).symlink_to(folder, target_is_directory=True)
        camera = render.camera_for(stage, (CONTROL, MASKED))
        with patch.object(render.replay_io, "load_navigation_replay", add_feature_label):
            for run_id, folder in folders.items():
                record = render.render(stage, run_id, camera, fps=20.0, frame_limit=300)
                video = videos / Path(record["video"]).name
                copy2(stage / record["video"], video)
                if render.digest(video) != record["video_sha256"]:
                    raise ValueError(f"{video.name} changed after its complete decode.")
                record.update(video=relpath(video, output),
                              replay=relpath(folder / "navigation.jsonl.gz", output),
                              gpx=relpath(folder / "track.gpx", output))
                records.append(record)
    write_json(output / "video_verification.json", records)
    lines = [
        "# Brindisi schedule-feature ablation playback", "",
        "Both clips use the Brindisi short route under 95° wind FROM. The camera, archive start and 50 m waypoint radius match.", "",
        "| Arrival model | Waypoint order | Elapsed | Video and data |", "|---|---|---:|---|",
    ]
    for record in records:
        arrivals = " → ".join(event["waypoint"] for event in record["arrivals"])
        links = " · ".join(f"[{label}]({record[key]})" for label, key in
                           (("MP4", "video"), ("Replay", "replay"), ("GPX", "gpx")))
        lines.append(f'| {FEATURE_LABELS[record["run_id"]]} | {arrivals} | '
                     f'{record["simulated_elapsed_s"]:g} s | {links} |')
    lines.extend(("", "[Video checks, exact frame indices, and SHA-256 digests](video_verification.json)", ""))
    (output / "playback.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-study", type=Path, default=STUDY)
    parser.add_argument("--output", type=Path, default=STUDY / "transition_ablation")
    args = parser.parse_args()
    render_pair(args.main_study.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
