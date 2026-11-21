"""Render the controlled regatta ablation with the pinned underway renderer."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch


STUDY = Path(__file__).resolve().parents[1]
PREDECESSOR = STUDY.with_name("otranto_underway_ais_retest")
BASE_SOURCES = {
    PREDECESSOR / "reproduce/render_round_trip.py": "bef7f20f5b75dd0c3f243c098fd1a1fe49bb49a614fb583e41be253e7f83e4e6",
    PREDECESSOR / "reproduce/audit_independent.py": "2b4c281813ac4ed3f0852a92844cbf8b9cef4b10b26397d497f827775ee07d77",
    STUDY.with_name("wind_direction_ablation_20261118") / "reproduce/render.py": "83e57040ec37c98d4ca542b30bd2f8c85f985397228b19470db22900f6c041fa",
}
BASE_RENDERER = next(iter(BASE_SOURCES))
FLAG = "--regatta-time-to-mark"
EXPECTED_ARMS = tuple(
    f"{fleet}__otranto_{wind}__{policy}__regatta_{mode}"
    for fleet in ("crossing_fan", "opposed_crossing")
    for wind in ("005", "355")
    for policy in ("classical", "transformer")
    for mode in ("off", "on")
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_base():
    for path, checksum in BASE_SOURCES.items():
        if digest(path) != checksum:
            raise ValueError(f"Pinned presentation dependency changed: {path}")
    spec = importlib.util.spec_from_file_location("_pinned_underway_renderer", BASE_RENDERER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {BASE_RENDERER}")
    sys.path.insert(0, str(BASE_RENDERER.parent))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base()


def metadata_for(study, arm):
    path = study / "runs" / arm / "execution.json"
    metadata = json.loads(path.read_text())
    base, mode = arm.rsplit("__regatta_", 1)
    enabled = FLAG in metadata["command"]
    if (metadata["base_arm"] != base or metadata["regatta_mode"] != mode
            or metadata["regatta_enabled"] != enabled or enabled != (mode == "on")):
        raise ValueError(f"{arm} has an inconsistent regatta flag or treatment label")
    return metadata


def first_002_elapsed(arrivals):
    return next((point["step"] * BASE.UPDATE_SECONDS for point in arrivals
                 if point["phase"] == "outbound" and point["waypoint"] == "002"), None)


def method_lines(study, arm):
    metadata = metadata_for(study, arm)
    enabled = metadata["regatta_enabled"]
    mission = BASE.reference.recorded_mission(study / "runs" / arm, 0)
    first_002 = first_002_elapsed(mission["waypoint_arrivals"])
    return (
        f"Regatta {'ON' if enabled else 'OFF'} · Gamma + Direct acados NMPC",
        f"Encounter: {metadata['scenario_id'].replace('_', ' ')}",
        "Frozen Transformer hold/release + physical confirmation" if metadata["method"] == "transformer"
        else "Classical hold/release guidance",
        "10 h race time-to-mark hold · refined tacks on both legs" if enabled
        else "Original forecast hold · coarse outbound · refined return",
        f"First 002: {BASE.clock(first_002)} after launch" if first_002 is not None else "First 002: unreached",
        "Protected-staging AIS gate · strict receding-horizon fallback",
        "AIS/COLREG: 20 s delay · staged and underway dropouts",
        "25 m AIS gap + full uncertainty + 3 m reserve",
    )


BASE.method_lines = method_lines


def render_task(task, preview_directory):
    return BASE.render(task, preview_directory)


def write_index(study, results):
    lines = [
        "# Otranto regatta-mode ablation playback", "",
        "The sixteen fresh recordings pair regatta OFF and ON for each fixed traffic, wind and weather-policy case. "
        "Both treatments use Gamma guidance, Direct acados NMPC, the 50 m waypoint gates and the same chart and AIS/COLREG protections. "
        "Regatta ON uses the 10 h time-to-mark hold reference and refined forecast tacks on both legs. "
        "OFF uses the original forecast hold and return-only refinements.", "",
        "The opening four seconds show the original gate telemetry over the recorded launch state. "
        "Each clip then shows the sailed route, with a close view of the second encounter from original traffic time 420–900 s. "
        "Reports have a 20 s delay, two 40 s staged emission gaps and two 40 s underway emission gaps. "
        "An offline gray diamond shows the new vessel before its first received report. "
        "Blue circles are the 50 m gates and orange dots are the actual capture fixes. "
        "The yellow bow and purple course-over-ground arrow use separate recorded directions. "
        "Wind barbs show FROM, the current arrow shows TO, and signed water surge records sternway.", "",
        "Race times run from actual launch. Sailed time ends at completion or the recorded native stop. "
        "The clearance column gives the smaller audited truth- and report-centered margin after full AIS uncertainty and the additional 3 m reserve. "
        "The timing mark uses the strict 5 s complete AIS acceptance/stop budget and includes a final unapplied attempt when recorded. "
        "The departure gate has a separate measured timer.", "",
        "Replay links use the corrected presentation header with the effective ON/OFF settings. "
        "The recorded navigation data lines are identical to the raw replay linked beside it.", "",
        "| Traffic | Wind FROM | Weather policy | Regatta | First 002 | Sailed | Captures | Outcome | AIS margin (m) | Complete AIS (s) | Files |",
        "|---|---:|---|---|---:|---:|---:|---|---:|---:|---|",
    ]
    by_arm = {result["run_id"]: result for result in results}
    independent = read_sweeps(study)
    for arm in EXPECTED_ARMS:
        result = by_arm[arm]
        metadata = metadata_for(study, arm)
        capture = first_002_elapsed(result["arrivals"])
        first_mark = "—" if capture is None else BASE.clock(capture)
        weather = "Transformer + confirmation" if metadata["method"] == "transformer" else "Classical"
        timing = result["AIS_applied_decision_latency"]
        timing_text = f"{timing['maximum_seconds']:.3f} ({'pass' if timing['all_calls_below_5s'] else 'fail'})"
        audit = independent[arm]
        margins = (audit["configured_truth_centered_minimum_m"], audit["configured_model_minimum_m"])
        margin_text = f"{min(margins):+.2f}" if all(m is not None for m in margins) else "—"
        prefix = f"runs/{arm}"
        files = (f"[MP4]({prefix}/navigation.mp4) · [Replay]({prefix}/navigation_playback.jsonl.gz) · "
                 f"[Raw]({prefix}/navigation.jsonl.gz) · "
                 f"[GPX]({prefix}/track.gpx) · [Video audit]({prefix}/video_verification.json) · "
                 f"[Gate]({prefix}/departure_gate.json) · [Gate log]({prefix}/departure_gate.jsonl)")
        row = (f"{metadata['scenario_id'].replace('_', ' ')}", f"{metadata['wind_from_degrees']:g}°", weather,
               metadata["regatta_mode"].upper(), first_mark, BASE.clock(result["simulated_elapsed_s"]),
               f"{len(result['arrivals'])}/8", BASE.mission_label(result["mission_status"]), margin_text,
               timing_text, files)
        lines.append("| " + " | ".join(row) + " |")
    index = study / "playback.md"
    working = index.with_suffix(".md.tmp")
    working.write_text("\n".join((*lines, "")), encoding="utf-8")
    working.replace(index)


def read_sweeps(study):
    return {row["case"]: row for row in json.loads((study / "independent_traffic_sweep.json").read_text())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--preview", type=Path, help="Render evidence PNGs without encoding")
    parser.add_argument("--arms", nargs="+", choices=EXPECTED_ARMS, help="Select arms for previews")
    args = parser.parse_args()
    if args.jobs < 1 or (args.arms and args.preview is None):
        parser.error("Use at least one worker and select arms only for previews")
    study = args.study.resolve()
    arms = tuple(args.arms or EXPECTED_ARMS)
    for arm in arms:
        metadata_for(study, arm)
    missions = {arm: BASE.reference.recorded_mission(study / "runs" / arm, 0) for arm in arms}
    short = [arm for arm in arms if missions[arm]["status"] != "completed" and missions[arm]["steps"] <= 120]
    long = [arm for arm in arms if arm not in short]
    cameras = {}
    with patch.object(BASE.reference, "run_folder", BASE.source_directory):
        if short:
            camera = BASE.short_event_camera(study, short)
            cameras.update({arm: camera for arm in short})
        if long:
            camera = BASE.reference.camera_for(study, tuple(long))
            cameras.update({arm: camera for arm in long})
    # Include the independently measured tightest intervals even when they
    # happen away from the opening encounters or ordinary renderer anchors.
    anchors = set(BASE.shared_anchor_steps(study, arms))
    independent = read_sweeps(study)
    for arm in arms:
        for measurement in ("truth_with_reserve", "projected_with_reserve"):
            detail = independent[arm]["minimum_details"].get(measurement)
            if detail is not None:
                step = detail["action_step"]
                anchors.update((max(0, step - 1), step, step + 1))
    anchors = sorted(anchors)
    tasks = [(study, arm, cameras[arm], 12, 420, anchors) for arm in arms]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(render_task, tasks, [args.preview] * len(arms)))
    if args.preview is None:
        write_index(study, results)


if __name__ == "__main__":
    main()
