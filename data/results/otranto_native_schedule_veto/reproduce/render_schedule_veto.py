"""Label and verify native committed-schedule-veto videos from closed replays."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


STUDY = Path(__file__).resolve().parents[1]
PINNED_DIR = STUDY.with_name("otranto_native_hybrid") / "reproduce"
PINNED_RENDERER = PINNED_DIR / "render_native_hybrid.py"
PINNED_SHA256 = "ec7434f87f949932e0339ca1c71f0b10e415674085d554031098dd6cffc79da7"
DIAGNOSTIC_STEP = 10959


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


if digest(PINNED_RENDERER) != PINNED_SHA256:
    raise ValueError("The proven native media adapter changed.")
sys.path.insert(0, str(PINNED_DIR))
import render_native_hybrid as media


_previous_method_lines = media.method_lines
_previous_displayed_frame = media.displayed_frame
_previous_mandatory_steps = media.mandatory_steps


def method_lines(study, arm):
    receipt = media.execution(study, arm)
    if receipt.get("native_schedule_veto") is not receipt["native_hybrid_enabled"]:
        raise ValueError(f"{arm}: corrected-schedule flag differs from the recorded method")
    lines = list(_previous_method_lines(study, arm))
    if receipt["native_schedule_veto"]:
        lines[0:2] = (
            "Hybrid tacks · corrected committed-schedule veto",
            "Weather Transformer + AIS propose the first trim",
            "50 m capture precedence · >180 s gain if both reach",
        )
    return tuple(lines)


def displayed_frame(original, frame, step, capture, truth_margin, method):
    visual = _previous_displayed_frame(original, frame, step, capture, truth_margin, method)
    nav = dict(visual.navigation)
    data = frame.navigation
    nav.pop("Forecast ETA learned / physics", None)
    if data.get("hybrid_tack_decision") is not None:
        nav["Schedule veto"] = nav.pop("Tack proposal check")
    if data.get("hybrid_tack_eta_source") == "native committed schedule":
        def describe(label):
            mode = data.get(f"hybrid_tack_{label}_schedule_class")
            if mode == 0:
                return f"capture {data[f'hybrid_tack_{label}_eta_s']:.0f}s"
            if mode == 1:
                return f"approach {data[f'hybrid_tack_{label}_closest_m']:.0f}m"
            if mode == 2:
                return f"miss at {data[f'hybrid_tack_{label}_closest_m']:.0f}m"
            return "none"
        nav["Schedule learned / physics"] = f"{describe('learned')} / {describe('physics')}"
    if (step == DIAGNOSTIC_STEP and frame.observation.wind is not None and
            abs(frame.observation.wind.direction - 355.) < 1e-7):
        nav["Comparison fix"] = "355° tack decision at step 10959"
    return media.replace(visual, navigation=nav)


def mandatory_steps(arm, audit):
    required = set(_previous_mandatory_steps(arm, audit))
    if "__otranto_355__" in arm:
        frames = audit["frames"]
        required.update(i for i in range(DIAGNOSTIC_STEP - 1, DIAGNOSTIC_STEP + 2) if i < frames)
    return sorted(required)


# Patch the detached rendering module in both parent and spawned workers.
# This leaves the pinned renderer on disk intact.
media.STUDY = STUDY
media.method_lines = method_lines
media.displayed_frame = displayed_frame
media.mandatory_steps = mandatory_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winds", nargs="+", choices=media.WINDS, default=media.WINDS)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Use at least one render worker.")
    sweep = STUDY / "independent_traffic_sweep.json"
    audits = {row["case"]: row for row in json.loads(sweep.read_text())}
    cameras, mandatory, tasks = {}, {}, []
    for wind in dict.fromkeys(args.winds):
        arms = tuple(media.arm_name(wind, mode) for mode in media.MODES)
        if any(arm not in audits for arm in arms):
            raise ValueError(f"{wind}°: missing independent closed-run audits")
        for arm in arms:
            media.execution(STUDY, arm)
            mandatory[arm] = sorted(set(mandatory_steps(arm, audits[arm])) | media.hybrid_steps(arm))
        with media.patch.object(media.BASE.reference, "run_folder", media.BASE.source_directory):
            cameras[wind] = media.BASE.reference.camera_for(STUDY, arms)
        shared = set(media.BASE.shared_anchor_steps(STUDY, arms))
        shared.update(i for arm in arms for i in mandatory[arm])
        tasks.extend((STUDY, arm, cameras[wind], 12, 420, sorted(shared)) for arm in arms)
    with media.ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as pool:
        output = list(pool.map(media.render_one, tasks, [args.preview] * len(tasks)))
    if args.preview is not None:
        return
    proofs = [media.decode(record, mandatory[record["run_id"]], audits[record["run_id"]]) for record in output]
    for proof in proofs:
        arm = proof["run_id"]
        if "__otranto_355__" in arm:
            proof["diagnostic_fix_retained"] = DIAGNOSTIC_STEP in proof["anchors_verified"]
    receipt = {
        "pinned_adapter_sha256": media.PINNED_SHA256,
        "pinned_native_media_sha256": PINNED_SHA256,
        "renderer_sha256": digest(Path(__file__).resolve()),
        "native_clearance_audit_sha256": digest(sweep),
        "native_control_timing_audit_sha256": digest(STUDY / "independent_native_control_timing.json"),
        "native_hybrid_policy_audit_sha256": digest(STUDY / "independent_hybrid_policy_telemetry.json"),
        "diagnostic_probe_sha256": digest(STUDY / "veto_regression_probe.json"),
        "shared_cameras_by_wind": cameras,
        "native_schedule_veto_scope": (
            "The recorded committed native schedules are compared by first 50 m capture class, then arrival gain."
        ),
        "runs": proofs,
    }
    suffix = "" if set(args.winds) == set(media.WINDS) else "_" + "_".join(dict.fromkeys(args.winds))
    destination = STUDY / f"native_video_verification{suffix}.json"
    media.write_json(destination, receipt)
    for proof in proofs:
        print(proof["run_id"], proof["decoded_frames"], "decoded frames, all required replay fixes retained", flush=True)


if __name__ == "__main__":
    main()
