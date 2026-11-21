"""Render each closed native wind pair with the pinned traffic replay layout."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import re
import runpy
import subprocess
from tempfile import NamedTemporaryFile
from unittest.mock import patch


STUDY = Path(__file__).resolve().parents[1]
PINNED = STUDY.with_name("otranto_regatta_ablation") / "reproduce/render_round_trip.py"
PINNED_SHA256 = "4f02b9aa1b5657740ee15e7871edc219e1ae3c2efcd71634b95176ea7dc2dbaf"
WINDS = ("005", "355")
MODES = ("physics", "hybrid")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


if digest(PINNED) != PINNED_SHA256:
    raise ValueError("The pinned regatta renderer changed.")
BASE = runpy.run_path(str(PINNED))["BASE"]


def arm_name(wind, mode):
    return f"crossing_fan__otranto_{wind}__{mode}"


def execution(study, arm):
    path = study / "runs" / arm / "execution.json"
    receipt = json.loads(path.read_text())
    enabled = arm.endswith("__hybrid")
    if receipt.get("native_hybrid_enabled") is not enabled:
        raise ValueError(f"{arm}: method does not match the execution receipt")
    if receipt.get("regatta_enabled") is not False or receipt["method"] != "classical":
        raise ValueError(f"{arm}: expected shared classical hold and forecast guidance")
    if "--ais-candidate-handoff" not in receipt["command"]:
        raise ValueError(f"{arm}: missing shared native AIS admission")
    wind = arm.split("__")[1].removeprefix("otranto_")
    if receipt["wind_from_degrees"] != int(wind):
        raise ValueError(f"{arm}: wind does not match the execution receipt")
    return receipt


def full_control_stats(study, arm):
    path = study / "independent_native_control_timing.json"
    return json.loads(path.read_text())["cases"][arm]


def hybrid_check_label(study, arm):
    report = json.loads((study / "independent_hybrid_policy_telemetry.json").read_text())["cases"][arm]
    counts = report["unique_decision_outcomes"]
    accepted = counts.get("accepted first trim", 0)
    return f"Tack checks {report['unique_timestamped_decisions']} · proposals accepted {accepted}"


def method_lines(study, arm):
    metadata = execution(study, arm)
    folder = study / "runs" / arm
    mission = BASE.reference.recorded_mission(folder, metadata["return_code"])
    timing = full_control_stats(study, arm)
    if not mission["return_to_start"] or mission["arrival_radius_metres"] != 50.:
        raise ValueError(f"{arm}: expected native 50 m round trip")
    if timing["applied_attempts"] != mission["steps"]:
        raise ValueError(f"{arm}: complete control timing and replay steps differ")
    def capture(waypoint):
        event = next((p for p in mission["waypoint_arrivals"] if
                      p["phase"] == "outbound" and p["waypoint"] == waypoint), None)
        return BASE.clock(5 * event["step"]) if event else "unreached"
    selected = metadata["native_hybrid_enabled"]
    maximum = timing["maximum_complete_native_control_decision_s"]
    budget = "PASS" if timing["all_complete_native_control_attempts_below_5s"] else "FAIL"
    return (
        "Hybrid tack proposals: weather Transformer + AIS" if selected
        else "Physics forecast tacks",
        "48 min physics forecast veto · 180 s gain rule" if selected
        else "Classical tack choices · native AIS and chart screen",
        *((hybrid_check_label(study, arm),) if selected else ()),
        "Crossing fan · Gamma + Direct acados NMPC",
        "Classical hold/release · candidate handoff · 50 m gates",
        f"Outbound 002 {capture('002')} · 003 {capture('003')} after launch",
        f"Complete control max {maximum:.3f} s · 5 s budget {budget}",
        "AIS/COLREG · 20 s delay · staged and underway gaps",
        "25 m nominal AIS gap + full uncertainty + 3 m reserve",
    )


def displayed_frame(original, frame, step, capture, truth_margin, method):
    visual = original(frame, step, capture, truth_margin, method)
    navigation = visual.navigation.copy()
    own = frame.navigation
    # This block describes the optional upstream tack proposer. The base
    # renderer independently shows AIS authority and the actual applied margin.
    decision = own.get("hybrid_tack_decision")
    if decision is not None:
        navigation["Tack proposal check"] = decision
    proposed = own.get("hybrid_tack_proposed_bow_deg")
    physics = own.get("hybrid_tack_physics_bow_deg")
    if proposed is not None and physics is not None and proposed != physics:
        navigation["Proposed and physics bow"] = f"{proposed:.1f}° / {physics:.1f}°"
    learned_eta = own.get("hybrid_tack_learned_eta_s")
    physics_eta = own.get("hybrid_tack_physics_eta_s")
    if learned_eta is not None or physics_eta is not None:
        learned = f"{learned_eta:.0f}s" if learned_eta is not None else "miss"
        classical = f"{physics_eta:.0f}s" if physics_eta is not None else "miss"
        navigation["Forecast ETA learned / physics"] = f"{learned} / {classical}"
    wall = own.get("native_control_decision_seconds")
    if isinstance(wall, (int, float)):
        navigation["Complete control wall s"] = wall
    return replace(visual, navigation=navigation)


def render_one(task, preview):
    original = BASE.displayed_frame

    def visual(frame, step, capture, truth_margin=None, method=None):
        return displayed_frame(original, frame, step, capture, truth_margin, method)

    with patch.object(BASE, "method_lines", method_lines), patch.object(BASE, "displayed_frame", visual):
        return BASE.render(task, preview)


def mandatory_steps(arm, audit):
    folder = STUDY / "runs" / arm
    metadata = execution(STUDY, arm)
    mission = BASE.reference.recorded_mission(folder, metadata["return_code"])
    last = mission["steps"]
    anchors = {0, last}
    for arrival in mission["waypoint_arrivals"]:
        anchors.update(i for i in range(arrival["step"] - 1, arrival["step"] + 2) if 0 <= i <= last)
    for name in ("truth_with_reserve", "projected_with_reserve"):
        detail = audit["minimum_details"].get(name)
        if detail is not None:
            step = detail["action_step"]
            anchors.update(i for i in range(step - 1, step + 2) if 0 <= i <= last)
    return sorted(anchors)


def hybrid_steps(arm):
    """Keep leg handoffs, accepted replans and the first fix of other verdicts."""
    replay = STUDY / "runs" / arm / "navigation.jsonl.gz"
    first, waypoint_first, accepted = {}, {}, {}
    last = 0
    with gzip.open(replay, "rt") as stream:
        next(stream)
        for index, line in enumerate(stream):
            last = index
            raw = json.loads(line)["navigation"]
            phase, waypoint = raw.get("phase"), raw.get("waypoint")
            decision = raw.get("hybrid_tack_decision")
            waypoint_first.setdefault((phase, waypoint, None), index)
            if decision is None:
                continue
            first.setdefault(decision, index)
            waypoint_first.setdefault((phase, waypoint, decision), index)
            if decision == "accepted first trim":
                accepted.setdefault(raw["hybrid_tack_decision_utc"], index)
    selected = {*first.values(), *waypoint_first.values(), *accepted.values()}
    return {i for step in selected for i in (step - 1, step, step + 1) if 0 <= i <= last}


def decode(video_record, anchors, audit):
    arm = video_record["run_id"]
    folder = STUDY / "runs" / arm
    video = folder / "navigation.mp4"
    replay = folder / "navigation.jsonl.gz"
    if digest(video) != video_record["video_sha256"]:
        raise ValueError(f"{arm}: video differs from render receipt")
    if digest(replay) != video_record["replay_sha256"] or digest(replay) != audit["replay_sha256"]:
        raise ValueError(f"{arm}: video and audit do not reference the same replay")
    source_indices = video_record["source_frame_indices"]
    if not set(anchors).issubset(source_indices):
        raise ValueError(f"{arm}: a capture, clearance or proposal audit fix was omitted")
    if not set(video_record["underway_retained_source_steps"]).issubset(source_indices):
        raise ValueError(f"{arm}: underway encounter was undersampled")
    if len(source_indices) != video_record["decoded_frames"] or source_indices != sorted(source_indices):
        raise ValueError(f"{arm}: incorrect or nonchronological replay mapping")
    ffmpeg = str(BASE.replay_io._ffmpeg_path())
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-xerror", "-threads", "1",
               "-i", str(video), "-map", "0:v:0", "-progress", "pipe:1", "-f", "null", "-"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    counts = re.findall(r"^frame=(\d+)$", result.stdout, re.MULTILINE)
    if not counts or int(counts[-1]) != video_record["decoded_frames"] or "progress=end" not in result.stdout:
        raise ValueError(f"{arm}: decoder did not reach the expected final video frame")
    progress = folder / "ffmpeg_decode.log"
    progress.write_text(result.stdout)
    return {
        "run_id": arm, "method_labels": method_lines(STUDY, arm), "decoded_frames": int(counts[-1]),
        "ffmpeg_command": command, "ffmpeg_progress_sha256": digest(progress),
        "video_sha256": digest(video), "source_replay_sha256": digest(replay),
        "video_receipt_sha256": digest(folder / "video_verification.json"),
        "execution_receipt_sha256": digest(folder / "execution.json"),
        "full_control_timing": full_control_stats(STUDY, arm),
        "anchors_verified": anchors, "retained_underway_frames": len(video_record["underway_retained_source_steps"]),
    }


def write_json(path, body):
    with NamedTemporaryFile(mode="w", prefix=path.stem, suffix=".tmp", dir=path.parent,
                            encoding="utf-8", delete=False) as stream:
        json.dump(body, stream, indent=2, allow_nan=False)
        stream.write("\n")
        name = Path(stream.name)
    name.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winds", nargs="+", choices=WINDS, default=WINDS)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Use at least one rendering worker.")
    sweep = STUDY / "independent_traffic_sweep.json"
    audits = {row["case"]: row for row in json.loads(sweep.read_text())}
    cameras, mandatory, tasks = {}, {}, []
    for wind in dict.fromkeys(args.winds):
        arms = tuple(arm_name(wind, mode) for mode in MODES)
        if any(arm not in audits for arm in arms):
            raise ValueError(f"{wind}°: both closed independent audits are required for the paired camera")
        execution(STUDY, arms[0])
        execution(STUDY, arms[1])
        with patch.object(BASE.reference, "run_folder", BASE.source_directory):
            cameras[wind] = BASE.reference.camera_for(STUDY, arms)
        anchors = set(BASE.shared_anchor_steps(STUDY, arms))
        for arm in arms:
            wanted = mandatory_steps(arm, audits[arm])
            mandatory[arm] = sorted(set(wanted) | hybrid_steps(arm))
            anchors.update(mandatory[arm])
        for arm in arms:
            tasks.append((STUDY, arm, cameras[wind], 12, 420, sorted(anchors)))
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        output = list(pool.map(render_one, tasks, [args.preview] * len(tasks)))
    if args.preview is not None:
        return
    proofs = [decode(record, mandatory[record["run_id"]], audits[record["run_id"]]) for record in output]
    receipt = {"pinned_adapter_sha256": PINNED_SHA256, "renderer_sha256": digest(Path(__file__).resolve()),
               "native_clearance_audit_sha256": digest(sweep), "shared_cameras_by_wind": cameras,
               "native_control_timing_audit_sha256": digest(STUDY / "independent_native_control_timing.json"),
               "native_hybrid_policy_audit_sha256": digest(STUDY / "independent_hybrid_policy_telemetry.json"),
               "runs": proofs}
    suffix = "" if set(args.winds) == set(WINDS) else "_" + "_".join(dict.fromkeys(args.winds))
    write_json(STUDY / f"native_video_verification{suffix}.json", receipt)
    for row in proofs:
        print(row["run_id"], row["decoded_frames"], "decoded frames, all required fixes retained.", flush=True)


if __name__ == "__main__":
    main()
