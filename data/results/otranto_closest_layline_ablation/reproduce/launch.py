"""Rerun the matched Otranto loops with the closest-waypoint-layline flag off/on."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import runpy
import shlex
import sys
import tempfile
from tempfile import mkdtemp
from threading import Barrier


STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[2]
REFERENCE = STUDY.with_name("otranto_native_schedule_veto")
FLAG = "--closest-waypoint-layline"
WINDS = (5, 355)
MODES = ("off", "on")
LOCAL_OPTIONS = ("--acados-build-dir", "--output-dir", "--output-gpx", "--record-replay")
CHANGED_SOURCE = {"world/legacy_planner_adapter.py", "world/run_legacy_fossen.py"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def paired_command(command):
    normalized = [value for value in command if value != FLAG]
    for flag in LOCAL_OPTIONS:
        normalized[normalized.index(flag) + 1] = flag
    return normalized


def plan_pair(wind, helpers, dry_run):
    load, digest = helpers["load"], helpers["digest"]
    prior_folder = REFERENCE / "runs" / f"crossing_fan__otranto_{wind:03d}__hybrid"
    prior_path = prior_folder / "execution.json"
    prior = load(prior_path)
    require(prior["return_code"] == 0 and prior["source_unchanged"]
            and prior["native_hybrid_enabled"] and prior["native_schedule_veto"]
            and prior["radius_m"] == 50 and prior["dt_s"] == 5 and prior["maximum_steps"] == 17280
            and prior["return_to_start"] and prior["predictive_fallback"],
            f"Historical reference for {wind}° does not match this experiment.")
    require(prior["command"][:2] == [sys.executable, "-B"] and FLAG not in prior["command"],
            f"Reference runtime or layline option differs for {wind}°.")
    require(all(option in prior["command"] for option in
                ("--colreg", "--time-aware-ais", "--ais-candidate-handoff", "--native-control-timing")),
            f"Reference omitted an AIS or control audit option for {wind}°.")

    archive = STUDY / "archive_field_fingerprints.json"
    require(digest(archive) == prior["archive_field_fingerprint_sha256"],
            "The refreshed regional forcing fingerprint differs from the reference.")
    source = {name: digest(ROOT / name) for name in prior["source_sha256"]}
    changes = {name: {"prior": old, "current": source[name]} for name, old in prior["source_sha256"].items()
               if source[name] != old}
    require(set(changes) == CHANGED_SOURCE, f"Unexpected source changes from the reference: {list(changes)}")
    for path in (archive, Path(__file__), ROOT / "world/waypoint_laylines.py",
                 STUDY / "reproduce/fingerprint_archive.py", STUDY / "reproduce/run_archive.py"):
        source[path.relative_to(ROOT).as_posix()] = digest(path)
    stale = {
        "return_code", "wall_seconds", "started_utc", "finished_utc", "changed_inputs", "source_unchanged",
        "actual_departure_utc", "gate_wait_seconds", "departure_gate_log_sha256",
        "departure_gate_summary_sha256", "maximum_gate_decision_s",
    }
    plans = []
    for mode in MODES:
        arm = f"crossing_fan__otranto_{wind:03d}__layline_{mode}"
        output = STUDY / "runs" / arm
        require(dry_run or not output.exists(), f"Run already exists: {output}.")
        build = (str(Path(tempfile.gettempdir()) / f"maritime_{arm}_BUILD") if dry_run
                 else mkdtemp(prefix=f"maritime_{arm}_"))
        command = prior["command"].copy()
        paths = (build, output, output / "track.gpx", output / "navigation.jsonl.gz")
        for option, path in zip(LOCAL_OPTIONS, paths, strict=True):
            require(command.count(option) == 1, f"Expected exactly one {option} argument.")
            command[command.index(option) + 1] = str(path)
        enabled = mode == "on"
        if enabled:
            command.append(FLAG)
        require(paired_command(command) == paired_command(prior["command"]),
                f"{arm} command differs beyond output/build locations and the registered layline option.")
        metadata = {name: value for name, value in prior.items() if name not in stale}
        metadata.update(
            run_id=arm, study_phase="closest waypoint layline matched ablation",
            retest_study_id=STUDY.name, command=command, acados_build_dir=build,
            concurrent_workers_requested=len(WINDS) * len(MODES),
            layline_mode=mode, closest_waypoint_layline=enabled,
            guidance_label=f"Native schedule-veto hybrid · closest waypoint layline {mode}",
            source_sha256=source, native_common_source_sha256=source,
            changes_from_reference_sources=changes, native_reference_execution=str(prior_path),
            native_reference_execution_sha256=digest(prior_path),
            native_reference_replay_sha256=digest(prior_folder / "navigation.jsonl.gz"),
        )
        plans.append((output, metadata))
    return plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    helpers = runpy.run_path(str(STUDY.with_name("otranto_regatta_ablation") / "reproduce/launch.py"))
    plans = [arm for wind in WINDS for arm in plan_pair(wind, helpers, args.dry_run)]
    for _, metadata in plans:
        print(shlex.join(metadata["command"]), flush=True)
    if args.dry_run:
        return
    save = helpers["save"]
    save(STUDY / "launch_preflight.json", {
        "start_wave_utc": helpers["utc_now"](),
        "reference_study": str(REFERENCE),
        "archive_field_fingerprint_sha256": plans[0][1]["archive_field_fingerprint_sha256"],
        "same_inputs": "Closest-waypoint-layline flag alone varies within each pair.",
        "normalized_command_sha256": {
            str(wind): hashlib.sha256(json.dumps(paired_command(plans[i * len(MODES)][1]["command"])).encode()).hexdigest()
            for i, wind in enumerate(WINDS)
        },
        "runs": [metadata["run_id"] for _, metadata in plans],
    })
    for output, metadata in plans:
        helpers["stage"](output, metadata)
    barrier = Barrier(len(plans))
    records = {}
    with ThreadPoolExecutor(max_workers=len(plans)) as pool:
        tasks = [pool.submit(helpers["run_case"], output, metadata, barrier) for output, metadata in plans]
        for task in as_completed(tasks):
            row = task.result()
            records[row["run_id"]] = row
            save(STUDY / "launch_summary.json", [records[name] for name in sorted(records)])
    if any(row["return_code"] != 0 or not row["source_unchanged"] for row in records.values()):
        raise SystemExit("An attempt failed or a pinned input changed. All attempts were retained.")


if __name__ == "__main__":
    main()
