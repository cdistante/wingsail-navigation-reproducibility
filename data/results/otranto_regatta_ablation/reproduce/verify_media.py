"""Verify the full media matrix and decode representative evidence frames."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import re
import subprocess

from render_round_trip import (
    BASE, BASE_RENDERER, BASE_SOURCES, EXPECTED_ARMS, PREDECESSOR, STUDY, digest,
    first_002_elapsed, metadata_for, read_sweeps,
)


def require(condition, arm, reason):
    if not condition:
        raise ValueError(f"{arm}: {reason}")


def inspect_previews(study, arm, receipt, renderer_hash):
    path = study / "media_preview" / f"{arm}_layout.json"
    rows = json.loads(path.read_text())
    seen = set()
    for row in rows:
        x, y, width, height = row["text_bbox"]
        _, legend_y, _, legend_height = row["legend_bbox"]
        cx, cy, canvas_width, canvas_height = row["canvas_bbox"]
        require(x >= cx and x + width <= cx + canvas_width and y + height <= cy + canvas_height
                and y >= legend_y + legend_height + 6, arm, "clipped text or legend overlap")
        require(row["render_script_sha256"] == renderer_hash
                and row["preview_replay_sha256"] == receipt["replay_sha256"]
                and row["preview_scope"] == "closed native full mission",
                arm, "preview provenance differs from encoded input")
        require(receipt["source_frame_indices"][row["video_frame_index"]] == row["source_step"],
                arm, "preview frame mapping differs from encoded frame mapping")
        seen.add(row["source_step"])
    require(len(rows) > 1 and any(row["launch_summary_visible"] for row in rows), arm, "incomplete launch preview")
    if receipt["underway_retained_source_steps"]:
        require(any(row["tactical_inset_bbox"] for row in rows), arm, "missing tactical inset preview")
    require(seen.issuperset(capture["step"] for capture in receipt["arrivals"]), arm, "missing capture preview")
    return {
        "layout_sha256": digest(path), "checked_frames": len(rows),
        "checked_tactical_insets": sum(row["tactical_inset_bbox"] is not None for row in rows),
        "capture_fixes_checked": len(receipt["arrivals"]),
        "preview_png_sha256": {Path(row["png"]).name: digest(Path(row["png"])) for row in rows},
    }


def select_frames(receipt, frames, tactical, critical, first_capture, worst_steps):
    indices = receipt["source_frame_indices"]
    picks = {0, receipt["launch_summary_frames"], len(indices) - 1}
    picks.update(indices.index(step) for step in critical if step in tactical)
    picks.update(indices.index(step) for step in worst_steps)
    if first_capture is not None:
        picks.add(indices.index(first_capture))
    for recovering in (False, True):
        astern = [index for index, step in enumerate(indices)
                  if frames[step]["state"].get("surge_speed") is not None
                  and frames[step]["state"]["surge_speed"] < -0.05
                  and bool(frames[step]["navigation"].get("ais_recovery_active")) == recovering]
        if astern:
            picks.add(min(astern, key=lambda index: frames[indices[index]]["state"]["surge_speed"]))
    return sorted(picks)


def decode_stills(study, arm, video, indices, chosen):
    output = study / "media_verification" / "decoded"
    output.mkdir(parents=True, exist_ok=True)
    pattern = output / f"{arm}_decoded_%03d.png"
    selection = "select=" + "+".join(f"not(n-{index})" for index in chosen)
    command = [str(BASE.replay_io._ffmpeg_path()), "-v", "error", "-xerror", "-threads", "1", "-i", str(video),
               "-vf", selection, "-fps_mode", "passthrough", "-y", str(pattern)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    stills = []
    for counter, index in enumerate(chosen, 1):
        png = output / f"{arm}_decoded_{counter:03d}.png"
        require(png.is_file() and png.stat().st_size > 10_000, arm, "decoded evidence frame missing")
        stills.append({"source_step": indices[index], "encoded_frame": index,
                       "path": str(png.relative_to(study)), "sha256": digest(png)})
    return stills


def case(study, arm, audit, config, renderer_hash):
    directory = study / "runs" / arm
    metadata = metadata_for(study, arm)
    video = directory / "navigation.mp4"
    replay = directory / "navigation.jsonl.gz"
    playback = directory / "navigation_playback.jsonl.gz"
    receipt_path = directory / "video_verification.json"
    receipt = json.loads(receipt_path.read_text())
    require(receipt["run_id"] == arm and receipt["render_script_sha256"] == renderer_hash,
            arm, "render receipt identifies another arm or artifact")
    require(digest(video) == receipt["video_sha256"] and digest(replay) == receipt["replay_sha256"] == audit["replay_sha256"],
            arm, "frozen replay or encoded output checksum changed")
    require(digest(directory / "track.gpx") == receipt["gpx_sha256"], arm, "frozen GPX checksum changed")
    require(digest(directory / "departure_gate.json") == receipt["departure_gate_summary_sha256"]
            and digest(directory / "departure_gate.jsonl") == receipt["departure_gate_log_sha256"],
            arm, "frozen gate telemetry checksum changed")
    require(any(label.startswith(f"Regatta {metadata['regatta_mode'].upper()} · ") for label in receipt["labels"]),
            arm, "video treatment label is missing")
    arrivals = audit["goal_captures"]
    require(receipt["arrivals"] == arrivals, arm, "video capture sequence differs from independent audit")
    first = first_002_elapsed(arrivals)
    first_label = "First 002: unreached" if first is None else f"First 002: {BASE.clock(first)} after launch"
    require(first_label in receipt["labels"], arm, "video first-mark timing label differs")

    with gzip.open(replay, "rt", encoding="utf-8") as stream:
        next(stream)
        frames = [json.loads(line) for line in stream]
    indices = receipt["source_frame_indices"]
    launch_frames = receipt["launch_summary_frames"]
    present = set(indices)
    require(len(frames) == receipt["source_frame_count"] == audit["frames"], arm, "source frame count differs")
    require(launch_frames > 0 and indices[:launch_frames] == [0] * launch_frames and indices[-1] == len(frames) - 1
            and indices == sorted(indices) and receipt["decoded_frames"] == len(indices), arm, "encoded source index invalid")
    require(set(range(min(181, len(frames)))).issubset(present), arm, "opening five-second sources missing")
    colreg_steps = {row["navigation"]["step"] for row in frames if row["navigation"].get("ais_colreg_active_contacts", 0) > 0}
    require(colreg_steps.issubset(present), arm, "COLREG source fixes missing")
    for capture in arrivals:
        step = capture["step"]
        require(set(range(max(0, step - 1), min(len(frames), step + 2))).issubset(present), arm, "capture neighborhood missing")
    lower, upper = config["underway_frame_retention_original_epoch_s"]
    epoch_at_launch = receipt["gate_wait_seconds"] - 40
    tactical = {step for step in range(len(frames)) if lower <= epoch_at_launch + 5 * step <= upper}
    require(tactical == set(receipt["underway_retained_source_steps"]) and tactical.issubset(present),
            arm, "underway encounter source updates missing")
    delay_seconds = metadata["reporting"]["delay_s"]
    critical_epochs = [config["underway_first_delivery_s"], config["underway_anchor_elapsed_s"],
                       *(end + delay_seconds for _, end in config["underway_drop_emission_windows_s"])]
    critical_steps = {int((epoch - epoch_at_launch) // 5) + offset for epoch in critical_epochs for offset in (-1, 0)}
    worst = [detail["action_step"] for label in ("truth_with_reserve", "projected_with_reserve")
             if (detail := audit["minimum_details"].get(label)) is not None]
    require(set(worst).issubset(present), arm, "minimum clearance source updates missing")
    first_capture = None if first is None else first // BASE.UPDATE_SECONDS
    chosen = select_frames(receipt, frames, tactical, critical_steps, first_capture, worst)
    BASE.reference.verify_decode(video, len(indices))
    stills = decode_stills(study, arm, video, indices, chosen)
    checks = inspect_previews(study, arm, receipt, renderer_hash)
    print(f"{arm}: {len(indices)} decoded frames, {len(stills)} evidence stills", flush=True)
    return ({
        "base_arm": metadata["base_arm"], "regatta_mode": metadata["regatta_mode"],
        "source_frames": len(frames), "decoded_frames": len(indices), "launch_summary_frames": launch_frames,
        "unique_physical_steps": len(present), "captures_and_neighbors_retained": len(arrivals),
        "all_COLREG_updates_retained": len(colreg_steps), "first900s_all_updates_retained": True,
        "tactical_updates_retained": len(tactical), "video_receipt_sha256": digest(receipt_path),
        "video_sha256": digest(video), "replay_sha256": digest(replay),
        "playback_replay_sha256": digest(playback), "still_frames": stills,
    }, checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    args = parser.parse_args()
    study = args.study.resolve()
    index = study / "playback.md"
    paths = re.findall(r"\]\(([^)]+)\)", index.read_text())
    require(len(paths) == 7 * len(EXPECTED_ARMS) and len(set(paths)) == len(paths)
            and all((study / path).is_file() for path in paths), "Playback index", "broken or missing indexed artifacts")
    config = json.loads((study / "underway_config.json").read_text())
    sweeps = read_sweeps(study)
    require(set(sweeps) == set(EXPECTED_ARMS), "Matrix", "independent audit does not cover all sixteen runs")
    media, previews = {}, {}
    renderer_hash = digest(BASE_RENDERER)
    for arm in EXPECTED_ARMS:
        media[arm], previews[arm] = case(study, arm, sweeps[arm], config, renderer_hash)
    provenance = {
        "renderer_sha256": renderer_hash,
        "render_adapter_sha256": digest(Path(__file__).with_name("render_round_trip.py")),
        "renderer_dependencies_sha256": {str(path.relative_to(PREDECESSOR.parent)): checksum
                                           for path, checksum in BASE_SOURCES.items()},
        "verification_script_sha256": digest(Path(__file__)),
        "playback_index_sha256": digest(index),
    }
    BASE.reference.write_json(study / "media_verification.json", {
        **provenance, "runs": media, "all_sixteen_mp4s_fully_decoded": True,
        "independent_sweep_sha256": digest(study / "independent_traffic_sweep.json"),
        "selection_scope": "Opening and final, both underway dropout renewals, vessel acquisition, minimum clearance, first 002 and signed sternway when available.",
    })
    BASE.reference.write_json(study / "media_layout_verification.json", {
        **provenance, "runs": previews, "all_sixteen_renderer_receipts_present": True,
        "all_final_previews_passed": True, "checked_preview_frames": sum(row["checked_frames"] for row in previews.values()),
    })


if __name__ == "__main__":
    main()
