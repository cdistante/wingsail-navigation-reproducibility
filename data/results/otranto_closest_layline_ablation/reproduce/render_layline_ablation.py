"""Export and verify labeled MP4s for the matched Otranto waypoint layline runs."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import lru_cache
import gzip
from hashlib import file_digest
import json
from pathlib import Path
import runpy
from unittest.mock import patch


STUDY = Path(__file__).resolve().parents[1]
PINNED_RENDERER = STUDY.with_name("otranto_regatta_ablation") / "reproduce/render_round_trip.py"
PINNED_SHA256 = "4f02b9aa1b5657740ee15e7871edc219e1ae3c2efcd71634b95176ea7dc2dbaf"
WINDS = ("005", "355")
MODES = ("off", "on")
FLAG = "--closest-waypoint-layline"


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


if digest(PINNED_RENDERER) != PINNED_SHA256:
    raise ValueError("The pinned Otranto regatta renderer changed.")
BASE = runpy.run_path(str(PINNED_RENDERER))["BASE"]
DISPLAYED_FRAME = BASE.displayed_frame
ROUND_TRIP_ANIMATION = BASE.round_trip_animation


def run_id(wind, mode):
    return f"crossing_fan__otranto_{wind}__layline_{mode}"


def read_json(path):
    return json.loads(path.read_text())


def execution(study, arm):
    metadata = read_json(study / "runs" / arm / "execution.json")
    fleet, wind, mode = arm.split("__")
    on = mode == "layline_on"
    command = metadata["command"]
    if (
        fleet != "crossing_fan"
        or mode not in {"layline_off", "layline_on"}
        or command.count(FLAG) != int(on)
        or metadata["wind_from_degrees"] != int(wind.removeprefix("otranto_"))
        or metadata["return_code"] != 0
    ):
        raise ValueError(f"{arm}: treatment, wind or closing receipt mismatch.")
    required = {"--time-aware-ais", "--colreg", "--hybrid-forecast-tacks", "--native-control-timing"}
    if not required.issubset(command) or "--regatta-time-to-mark" in command:
        raise ValueError(f"{arm}: inconsistent controller, guidance or safety flags.")
    return metadata


@lru_cache(maxsize=None)
def layline_events(replay):
    """Track applied references and preserve the first fix of each type per leg."""
    counts, anchors, seen = Counter(), set(), set()
    with gzip.open(replay, "rt") as stream:
        next(stream)
        for step, line in enumerate(stream):
            nav = json.loads(line)["navigation"]
            if nav.get("step") != step:
                raise ValueError(f"{replay}: replay step and frame index differ.")
            source = nav.get("waypoint_layline_source")
            if source is None:
                continue
            counts[source] += 1
            key = (nav.get("phase"), nav.get("waypoint"), source)
            if key not in seen:
                anchors.update((max(0, step - 1), step, step + 1))
                seen.add(key)
    return dict(counts), anchors


def timing_for(study, arm):
    return read_json(study / "independent_native_control_timing.json")["cases"][arm]


def method_lines(study, arm):
    receipt = execution(study, arm)
    folder = study / "runs" / arm
    mission = BASE.reference.recorded_mission(folder, receipt["return_code"])
    timing = timing_for(study, arm)
    if timing["applied_attempts"] != mission["steps"]:
        raise ValueError(f"{arm}: independent control timing and trajectory differ.")
    on = arm.endswith("__layline_on")
    counts, _ = layline_events(folder / "navigation.jsonl.gz")
    if counts and not on:
        raise ValueError(f"{arm}: active waypoint layline telemetry in the off arm.")
    point = next((event for event in mission["waypoint_arrivals"]
                  if event["phase"] == "outbound" and event["waypoint"] == "002"), None)
    first_002 = BASE.clock(BASE.UPDATE_SECONDS * point["step"]) if point else "unreached"
    maximum = timing["maximum_complete_native_control_decision_s"]
    budget = "PASS" if timing["all_complete_native_control_attempts_below_5s"] else "FAIL"
    return (
        f"Closest waypoint laylines {'ON' if on else 'OFF'}",
        "Crossing fan · Gamma + Direct acados NMPC",
        "Weather Transformer + AIS · committed-schedule veto",
        "Classical hold/release · AIS candidate handoff",
        f"Closest-layline guidance at {sum(counts.values()):,} recorded fixes",
        f"Outbound 002 {first_002} after launch",
        f"Complete native control max {maximum:.3f} s · 5 s budget {budget}",
        "AIS/COLREG · 20 s delay · staged and underway gaps",
        "25 m nominal AIS gap + full uncertainty + 3 m reserve",
    )


def displayed_frame(frame, step, capture, truth_margin=None, method=None):
    visual = DISPLAYED_FRAME(frame, step, capture, truth_margin, method)
    nav = visual.navigation.copy()
    raw = frame.navigation
    if (source := raw.get("waypoint_layline_source")) is not None:
        nav["Weather action"] = "waypoint layline"
        nav["Waypoint layline"] = source
        nav["First-leg distance m"] = raw["waypoint_layline_distance_m"]
        nav["Estimated gate entry s"] = raw["waypoint_layline_eta_s"]
        if (second := raw.get("waypoint_layline_second_bow_deg")) is not None:
            nav["Finishing tack bow deg"] = second
    if (seconds := raw.get("native_control_decision_seconds")) is not None:
        nav["Complete native control wall s"] = seconds
    return replace(visual, navigation=nav)


def render_one(task, preview):
    _, arm, *_ = task
    _, wind, mode = arm.split("__")
    bearing = int(wind.removeprefix("otranto_"))
    treatment = mode.removeprefix("layline_").upper()

    def animation(*args, **kwargs):
        base = ROUND_TRIP_ANIMATION(*args, **kwargs)

        class LaylineAnimation(base):
            def __init__(self, *animation_args, **animation_kwargs):
                super().__init__(*animation_args, **animation_kwargs)
                self.axis.set_position((.072, .13, .523, .77))
                self.axis.set_title(
                    f"Otranto · wind FROM {bearing}° · closest laylines {treatment} · north up"
                )

        return LaylineAnimation

    with (patch.object(BASE, "method_lines", method_lines),
          patch.object(BASE, "displayed_frame", displayed_frame),
          patch.object(BASE, "round_trip_animation", animation)):
        return BASE.render(task, preview)


def anchors_for(study, arm, audit):
    folder = study / "runs" / arm
    mission = BASE.reference.recorded_mission(folder, 0)
    last = mission["steps"]
    _, layline = layline_events(folder / "navigation.jsonl.gz")
    anchors = {0, last, *layline}
    at_steps = [entry["step"] for entry in mission["waypoint_arrivals"]]
    for key in ("truth_with_reserve", "projected_with_reserve"):
        if (detail := audit["minimum_details"].get(key)) is not None:
            at_steps.append(detail["action_step"])
    for step in at_steps:
        anchors.update((step - 1, step, step + 1))
    return sorted(step for step in anchors if 0 <= step <= last)


def media_proof(study, output, audit, anchors):
    arm = output["run_id"]
    folder = study / "runs" / arm
    video = folder / "navigation.mp4"
    replay = folder / "navigation.jsonl.gz"
    replay_sha256 = digest(replay)
    video_sha256 = digest(video)
    indices = output["source_frame_indices"]
    if (
        replay_sha256 != audit["replay_sha256"]
        or replay_sha256 != output["replay_sha256"]
        or video_sha256 != output["video_sha256"]
        or len(indices) != output["decoded_frames"]
        or indices != sorted(indices)
        or not set(anchors).issubset(indices)
        or not set(output["underway_retained_source_steps"]).issubset(indices)
    ):
        raise ValueError(f"{arm}: failed encoded frame/source binding.")
    counts, _ = layline_events(replay)
    return {
        "run_id": arm,
        "closest_waypoint_layline_enabled": arm.endswith("__layline_on"),
        "method_labels": output["labels"],
        "mission_status": output["mission_status"],
        "sailing_elapsed_s": output["simulated_elapsed_s"],
        "decoded_frames": output["decoded_frames"],
        "fps": output["fps"],
        "source_replay_sha256": replay_sha256,
        "video_sha256": video_sha256,
        "video_receipt_sha256": digest(folder / "video_verification.json"),
        "execution_receipt_sha256": digest(folder / "execution.json"),
        "complete_native_control_timing": timing_for(study, arm),
        "required_source_steps": anchors,
        "retained_underway_frames": len(output["underway_retained_source_steps"]),
        "layline_guidance_fixes": counts,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--winds", nargs="+", choices=WINDS, default=WINDS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    study = args.study.resolve()
    winds = tuple(dict.fromkeys(args.winds))
    sweep = study / "independent_traffic_sweep.json"
    audits = {case["case"]: case for case in read_json(sweep)}
    cameras, required, tasks = {}, {}, []
    for wind in winds:
        arms = tuple(run_id(wind, mode) for mode in MODES)
        for arm in arms:
            execution(study, arm)
            required[arm] = anchors_for(study, arm, audits[arm])
        with patch.object(BASE.reference, "run_folder", BASE.source_directory):
            cameras[wind] = BASE.reference.camera_for(study, arms)
        common = set(BASE.shared_anchor_steps(study, arms))
        common.update(step for arm in arms for step in required[arm])
        tasks.extend((study, arm, cameras[wind], 12, 420, sorted(common)) for arm in arms)
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        outputs = list(pool.map(render_one, tasks, [args.preview] * len(tasks)))
    if args.preview is not None:
        return
    proofs = [media_proof(study, output, audits[output["run_id"]], required[output["run_id"]])
              for output in outputs]
    suffix = "" if set(winds) == set(WINDS) else "_" + "_".join(winds)
    report = {
        "renderer_sha256": digest(Path(__file__)),
        "pinned_adapter_sha256": PINNED_SHA256,
        "native_clearance_audit_sha256": digest(sweep),
        "native_control_timing_audit_sha256": digest(study / "independent_native_control_timing.json"),
        "shared_cameras_by_wind": cameras,
        "runs": proofs,
    }
    BASE.reference.write_json(study / f"native_video_verification{suffix}.json", report)
    for proof in proofs:
        print(f"{proof['run_id']}: fully decoded {proof['decoded_frames']} frames", flush=True)


if __name__ == "__main__":
    main()
