"""Export and verify ETA-gated Otranto MP4s from closed native replays."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import lru_cache
import gzip
from hashlib import file_digest
import json
from math import isclose, isfinite
from pathlib import Path
import runpy
from time import sleep
from unittest.mock import patch


STUDY = Path(__file__).resolve().parents[1]
BASELINE = STUDY.with_name("otranto_closest_layline_ablation")
PINNED_RENDERER = STUDY.with_name("otranto_layline_priority") / "reproduce/render_layline_priority.py"
PINNED_SHA256 = "fc09832b4e595945faf75d1bfc6309fc399cfe11f1a5bfb8d22dff0003d3fd60"
WINDS = ("005", "355")
FLAG = "--compare-waypoint-layline-eta"


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


if digest(PINNED_RENDERER) != PINNED_SHA256:
    raise ValueError("The pinned priority-media adapter changed.")
PRIORITY = runpy.run_path(str(PINNED_RENDERER))
LAYLINE = PRIORITY["PRIOR"]
BASE = PRIORITY["BASE"]


def run_id(wind):
    return f"crossing_fan__otranto_{wind}__layline_eta"


def baseline_id(wind):
    return f"crossing_fan__otranto_{wind}__layline_off"


def read_json(path):
    return json.loads(path.read_text())


def execution(study, arm):
    receipt = read_json(study / "runs" / arm / "execution.json")
    command = receipt["command"]
    required = {FLAG, "--time-aware-ais", "--colreg", "--hybrid-forecast-tacks", "--native-control-timing"}
    single_policy = {"--regatta-time-to-mark", "--prioritize-waypoint-layline", "--closest-waypoint-layline"}
    if (
        arm != run_id(f"{int(receipt['wind_from_degrees']):03d}")
        or receipt["layline_mode"] != "eta"
        or receipt["compare_waypoint_layline_eta"] is not True
        or receipt["closest_waypoint_layline"] is not True
        or command.count(FLAG) != 1
        or not required.issubset(command)
        or single_policy.intersection(command)
        or receipt["return_code"] != 0
    ):
        raise ValueError(f"{arm}: inconsistent ETA treatment, safety flags or closing receipt.")
    return receipt


def eta_values(nav):
    """Return the two recorded gate estimates and their advantage in seconds."""
    layline = float(nav["waypoint_layline_eta_s"])
    forecast = float(nav["waypoint_layline_forecast_eta_s"])
    advantage = float(nav["waypoint_layline_eta_advantage_s"])
    if (nav.get("waypoint_layline_policy") != "eta"
            or float(nav["waypoint_layline_comparison_radius_m"]) != 50.0
            or not all(isfinite(value) for value in (layline, forecast, advantage))
            or not 0.0 <= layline < forecast
            or not isclose(forecast - layline, advantage, abs_tol=1e-6)):
        raise ValueError("An applied ETA-gated frame lacks two comparable ordered 50 m arrival estimates.")
    return layline, forecast, advantage


@lru_cache(maxsize=None)
def eta_events(replay):
    """Find the first applied comparison per leg and the largest predicted gain."""
    counts, first_by_leg = Counter(), {}
    first, largest = None, None
    with gzip.open(replay, "rt") as stream:
        next(stream)
        for step, line in enumerate(stream):
            nav = json.loads(line)["navigation"]
            if nav.get("step") != step:
                raise ValueError(f"{replay}: replay step and frame index differ.")
            if (source := nav.get("waypoint_layline_source")) is None:
                continue
            layline, forecast, advantage = eta_values(nav)
            event = {
                "source_step": step, "phase": nav["phase"], "waypoint": nav["waypoint"],
                "layline_eta_s": layline, "forecast_eta_s": forecast, "advantage_s": advantage,
            }
            counts[source] += 1
            first_by_leg.setdefault((nav["phase"], nav["waypoint"]), step)
            if first is None:
                first = event
            if largest is None or advantage > largest["advantage_s"]:
                largest = event
    return dict(counts), list(first_by_leg.values()), first, largest


def paired_camera(study, wind):
    arm, baseline = run_id(wind), baseline_id(wind)

    def folder(eta_study, name):
        source = BASELINE if name == baseline else eta_study
        return source / "runs" / name

    with patch.object(BASE.reference, "run_folder", folder):
        return BASE.reference.camera_for(study, (arm, baseline))


def method_lines(study, arm):
    receipt = execution(study, arm)
    folder = study / "runs" / arm
    mission = BASE.reference.recorded_mission(folder, receipt["return_code"])
    timing = LAYLINE["timing_for"](study, arm)
    if timing["applied_attempts"] != mission["steps"]:
        raise ValueError(f"{arm}: independently timed controls and trajectory differ.")
    counts, _, _, _ = eta_events(folder / "navigation.jsonl.gz")
    mark = next((item for item in mission["waypoint_arrivals"]
                 if item["phase"] == "outbound" and item["waypoint"] == "002"), None)
    elapsed = BASE.clock(BASE.UPDATE_SECONDS * mark["step"]) if mark else "unreached"
    maximum = timing["maximum_complete_native_control_decision_s"]
    budget = "PASS" if timing["all_complete_native_control_attempts_below_5s"] else "FAIL"
    return (
        "Closest layline gated by earlier predicted arrival",
        "Shared archived forecast · same 50 m waypoint gate",
        "Crossing fan · Gamma + Direct acados NMPC",
        "Weather Transformer + AIS · committed-schedule veto",
        f"ETA-gated layline applied at {sum(counts.values()):,} fixes",
        f"Outbound 002 {elapsed} after launch",
        f"Complete native control max {maximum:.3f} s · 5 s budget {budget}",
        "AIS/COLREG · 20 s delay · staged and underway gaps",
        "25 m nominal AIS gap + full uncertainty + 3 m reserve",
    )


def displayed_frame(frame, step, capture, truth_margin=None, method=None):
    visual = LAYLINE["displayed_frame"](frame, step, capture, truth_margin, method)
    raw = frame.navigation
    if raw.get("waypoint_layline_source") is None:
        return visual
    layline, forecast, advantage = eta_values(raw)
    navigation = visual.navigation.copy()
    navigation.pop("Estimated gate entry s", None)
    navigation["Weather action"] = "ETA-gated waypoint layline"
    gain = f"{advantage:.2f}" if advantage >= .01 else "<0.01"
    navigation["50 m ETA"] = f"{layline:.2f}s layline · {forecast:.2f}s forecast · {gain}s gain"
    return replace(visual, navigation=navigation)


def anchors_for(study, arm, audit):
    folder = study / "runs" / arm
    mission = BASE.reference.recorded_mission(folder, 0)
    _, first_by_leg, _, largest = eta_events(folder / "navigation.jsonl.gz")
    anchors = set(LAYLINE["anchors_for"](study, arm, audit))
    if largest is not None:
        first_by_leg = [*first_by_leg, largest["source_step"]]
    for step in first_by_leg:
        anchors.update((step - 1, step, step + 1))
    return sorted(step for step in anchors if 0 <= step <= mission["steps"])


def render_task(task, preview):
    _, arm, *_ = task
    _, wind, _ = arm.split("__")
    bearing = int(wind.removeprefix("otranto_"))

    def animation(*args, **kwargs):
        base = LAYLINE["ROUND_TRIP_ANIMATION"](*args, **kwargs)

        class ETAAnimation(base):
            def __init__(self, *animation_args, **animation_kwargs):
                super().__init__(*animation_args, **animation_kwargs)
                self.axis.set_position((.072, .13, .523, .77))
                self.axis.set_title(f"Otranto · wind FROM {bearing}° · ETA-gated laylines · north up")

        return ETAAnimation

    with (patch.object(BASE, "method_lines", method_lines),
          patch.object(BASE, "displayed_frame", displayed_frame),
          patch.object(BASE, "round_trip_animation", animation)):
        return BASE.render(task, preview)


def media_proof(study, result, audit, required, wind):
    proof = PRIORITY["media_proof"](study, result, audit, required, wind)
    proof.pop("prioritize_waypoint_layline")
    _, _, first, largest = eta_events(study / "runs" / run_id(wind) / "navigation.jsonl.gz")
    return {
        **proof,
        "compare_waypoint_layline_eta": True,
        "first_applied_eta_comparison": first,
        "greatest_predicted_eta_advantage": largest,
    }


def ready_winds(study, pending):
    sweep = study / "independent_traffic_sweep.json"
    complete_timing = study / "independent_native_control_timing.json"
    if not (sweep.is_file() and complete_timing.is_file()):
        return []
    audited = {case["case"] for case in read_json(sweep)}
    timed = read_json(complete_timing)["cases"]
    return [
        wind for wind in pending
        if (arm := run_id(wind)) in audited and arm in timed
        and (study / "runs" / arm / "execution.json").is_file()
        and (study / "independent_traffic_history" / f"{arm}.csv").is_file()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--winds", nargs="+", choices=WINDS, default=WINDS)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--watch", action="store_true", help="Export each wind when its independent audit closes")
    args = parser.parse_args()
    if args.workers < 1 or (args.watch and args.preview is not None):
        parser.error("Use a positive worker count and choose either previews or watch mode.")
    study = args.study.resolve()
    winds = tuple(dict.fromkeys(args.winds))
    pending = list(winds)
    sweep = study / "independent_traffic_sweep.json"
    cameras, proofs = {}, {}
    with ProcessPoolExecutor(max_workers=min(args.workers, len(pending))) as pool:
        while pending:
            ready = ready_winds(study, pending)
            if not ready:
                if not args.watch:
                    raise ValueError("The requested ETA runs lack complete independent clearance or control receipts.")
                sleep(5)
                continue
            audits = {case["case"]: case for case in read_json(sweep)}
            required, tasks = {}, []
            for wind in ready:
                arm = run_id(wind)
                execution(study, arm)
                required[arm] = anchors_for(study, arm, audits[arm])
                cameras[wind] = paired_camera(study, wind)
                anchors = set(required[arm]) | set(BASE.shared_anchor_steps(study, (arm,)))
                tasks.append((study, arm, cameras[wind], 12, 420, sorted(anchors)))
            results = list(pool.map(render_task, tasks, [args.preview] * len(tasks)))
            for wind, result in zip(ready, results, strict=True):
                pending.remove(wind)
                if args.preview is not None:
                    continue
                arm = run_id(wind)
                proof = media_proof(study, result, audits[arm], required[arm], wind)
                proofs[wind] = proof
                print(f"{arm}: fully decoded {proof['decoded_frames']} frames", flush=True)
    if args.preview is not None:
        return
    suffix = "" if set(winds) == set(WINDS) else "_" + "_".join(winds)
    report = {
        "renderer_sha256": digest(Path(__file__)),
        "pinned_priority_adapter_sha256": PINNED_SHA256,
        "native_clearance_audit_sha256": digest(sweep),
        "native_control_timing_audit_sha256": digest(study / "independent_native_control_timing.json"),
        "cameras_fit_eta_and_baseline": cameras,
        "runs": [proofs[wind] for wind in winds],
    }
    BASE.reference.write_json(study / f"native_video_verification{suffix}.json", report)


if __name__ == "__main__":
    main()
