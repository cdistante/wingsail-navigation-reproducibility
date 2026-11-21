"""Export and verify priority-mode Otranto MP4s from closed native replays."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from hashlib import file_digest
import json
from pathlib import Path
import runpy
from unittest.mock import patch


STUDY = Path(__file__).resolve().parents[1]
BASELINE = STUDY.with_name("otranto_closest_layline_ablation")
PINNED_RENDERER = BASELINE / "reproduce/render_layline_ablation.py"
PINNED_SHA256 = "2b118b5ce92315724928af7939f2f726971d236d1ec332475c92fc16b8158a60"
WINDS = ("005", "355")
FLAG = "--prioritize-waypoint-layline"


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


if digest(PINNED_RENDERER) != PINNED_SHA256:
    raise ValueError("The pinned closest-layline media adapter changed.")
PRIOR = runpy.run_path(str(PINNED_RENDERER))
BASE = PRIOR["BASE"]


def run_id(wind):
    return f"crossing_fan__otranto_{wind}__layline_priority"


def baseline_id(wind):
    return f"crossing_fan__otranto_{wind}__layline_off"


def read_json(path):
    return json.loads(path.read_text())


def execution(study, arm):
    receipt = read_json(study / "runs" / arm / "execution.json")
    command = receipt["command"]
    required = {FLAG, "--time-aware-ais", "--colreg", "--hybrid-forecast-tacks", "--native-control-timing"}
    if (
        arm != run_id(f"{int(receipt['wind_from_degrees']):03d}")
        or receipt["layline_mode"] != "priority"
        or receipt["prioritize_waypoint_layline"] is not True
        or receipt["closest_waypoint_layline"] is not True
        or command.count(FLAG) != 1
        or not required.issubset(command)
        or "--regatta-time-to-mark" in command
        or receipt["return_code"] != 0
    ):
        raise ValueError(f"{arm}: inconsistent treatment, safety flags or closing receipt.")
    return receipt


def paired_camera(study, wind):
    """Fit both the new track and the already verified baseline track."""
    arm, baseline = run_id(wind), baseline_id(wind)

    def folder(priority_study, name):
        source = BASELINE if name == baseline else priority_study
        return source / "runs" / name

    with patch.object(BASE.reference, "run_folder", folder):
        return BASE.reference.camera_for(study, (arm, baseline))


def method_lines(study, arm):
    receipt = execution(study, arm)
    folder = study / "runs" / arm
    mission = BASE.reference.recorded_mission(folder, receipt["return_code"])
    timing = PRIOR["timing_for"](study, arm)
    if timing["applied_attempts"] != mission["steps"]:
        raise ValueError(f"{arm}: independently timed controls and trajectory differ.")
    counts, _ = PRIOR["layline_events"](folder / "navigation.jsonl.gz")
    mark = next((item for item in mission["waypoint_arrivals"]
                 if item["phase"] == "outbound" and item["waypoint"] == "002"), None)
    elapsed = BASE.clock(BASE.UPDATE_SECONDS * mark["step"]) if mark else "unreached"
    maximum = timing["maximum_complete_native_control_decision_s"]
    budget = "PASS" if timing["all_complete_native_control_attempts_below_5s"] else "FAIL"
    return (
        "Closest feasible layline PRIORITY after AIS screen",
        "Crossing fan · Gamma + Direct acados NMPC",
        "Weather Transformer + AIS · committed-schedule veto",
        "Classical hold/forecast when no feasible layline",
        f"Priority layline applied at {sum(counts.values()):,} fixes",
        f"Outbound 002 {elapsed} after launch",
        f"Complete native control max {maximum:.3f} s · 5 s budget {budget}",
        "AIS/COLREG · 20 s delay · staged and underway gaps",
        "25 m nominal AIS gap + full uncertainty + 3 m reserve",
    )


def displayed_frame(frame, step, capture, truth_margin=None, method=None):
    visual = PRIOR["displayed_frame"](frame, step, capture, truth_margin, method)
    if frame.navigation.get("waypoint_layline_source") is None:
        return visual
    navigation = visual.navigation.copy()
    navigation["Weather action"] = "priority waypoint layline"
    return replace(visual, navigation=navigation)


def render_task(task, preview):
    _, arm, *_ = task
    _, wind, _ = arm.split("__")
    bearing = int(wind.removeprefix("otranto_"))

    def animation(*args, **kwargs):
        base = PRIOR["ROUND_TRIP_ANIMATION"](*args, **kwargs)

        class PriorityAnimation(base):
            def __init__(self, *animation_args, **animation_kwargs):
                super().__init__(*animation_args, **animation_kwargs)
                self.axis.set_position((.072, .13, .523, .77))
                self.axis.set_title(f"Otranto · wind FROM {bearing}° · layline priority · north up")

        return PriorityAnimation

    with (patch.object(BASE, "method_lines", method_lines),
          patch.object(BASE, "displayed_frame", displayed_frame),
          patch.object(BASE, "round_trip_animation", animation)):
        return BASE.render(task, preview)


def media_proof(study, result, audit, required, wind):
    proof = PRIOR["media_proof"](study, result, audit, required)
    baseline = BASELINE / "runs" / baseline_id(wind)
    baseline_receipt = read_json(baseline / "video_verification.json")
    replay_sha256 = digest(baseline / "navigation.jsonl.gz")
    video_sha256 = digest(baseline / "navigation.mp4")
    if (replay_sha256 != baseline_receipt["replay_sha256"]
            or video_sha256 != baseline_receipt["video_sha256"]
            or "Closest waypoint laylines OFF" not in baseline_receipt["labels"]):
        raise ValueError(f"{wind}: the matched baseline replay, video or mode changed.")
    return {
        **proof,
        "closest_waypoint_layline_enabled": True,
        "prioritize_waypoint_layline": True,
        "matched_baseline": {
            "run_id": baseline_id(wind),
            "replay": str(baseline / "navigation.jsonl.gz"),
            "video": str(baseline / "navigation.mp4"),
            "replay_sha256": replay_sha256,
            "video_sha256": video_sha256,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--winds", nargs="+", choices=WINDS, default=WINDS)
    parser.add_argument("--workers", type=int, default=2)
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
        arm = run_id(wind)
        execution(study, arm)
        required[arm] = PRIOR["anchors_for"](study, arm, audits[arm])
        cameras[wind] = paired_camera(study, wind)
        anchors = set(required[arm]) | set(BASE.shared_anchor_steps(study, (arm,)))
        tasks.append((study, arm, cameras[wind], 12, 420, sorted(anchors)))
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        results = list(pool.map(render_task, tasks, [args.preview] * len(tasks)))
    if args.preview is not None:
        return
    proofs = [media_proof(study, result, audits[arm], required[arm], wind)
              for wind, arm, result in zip(winds, required, results, strict=True)]
    suffix = "" if set(winds) == set(WINDS) else "_" + "_".join(winds)
    report = {
        "renderer_sha256": digest(Path(__file__)),
        "pinned_layline_adapter_sha256": PINNED_SHA256,
        "native_clearance_audit_sha256": digest(sweep),
        "native_control_timing_audit_sha256": digest(study / "independent_native_control_timing.json"),
        "cameras_fit_priority_and_baseline": cameras,
        "runs": proofs,
    }
    BASE.reference.write_json(study / f"native_video_verification{suffix}.json", report)
    for proof in proofs:
        print(f"{proof['run_id']}: fully decoded {proof['decoded_frames']} frames", flush=True)


if __name__ == "__main__":
    main()
