"""Rerun the matched Otranto loops with layline priority after AIS screening."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
BASELINE = STUDY.with_name("otranto_closest_layline_ablation")
FLAG = "--prioritize-waypoint-layline"
WINDS = (5, 355)
LOCAL_OPTIONS = ("--acados-build-dir", "--output-dir", "--output-gpx", "--record-replay")
CHANGED_SOURCE = {"world/legacy_planner_adapter.py", "world/run_legacy_fossen.py"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalized(command):
    normalized = [item for item in command if item != FLAG]
    for flag in LOCAL_OPTIONS:
        require(normalized.count(flag) == 1, f"Expected one {flag}.")
        normalized[normalized.index(flag) + 1] = flag
    return normalized


def plan(wind, helpers, dry_run):
    load, digest = helpers["load"], helpers["digest"]
    off = BASELINE / "runs" / f"crossing_fan__otranto_{wind:03d}__layline_off"
    prior_path = off / "execution.json"
    prior = load(prior_path)
    audit = load(BASELINE / "independent_final_summary.json")
    require(audit["all_acceptance_checks_passed"] and prior["return_code"] == 0 and prior["source_unchanged"]
            and prior["layline_mode"] == "off" and prior["native_hybrid_enabled"]
            and prior["radius_m"] == 50 and prior["dt_s"] == 5 and prior["maximum_steps"] == 17280,
            f"The {wind}° baseline does not match the accepted protocol.")
    command = prior["command"].copy()
    require(command[:2] == [sys.executable, "-B"]
            and FLAG not in command and "--closest-waypoint-layline" not in command,
            f"The {wind}° baseline command has a different runtime or guidance option.")

    archive = STUDY / "archive_field_fingerprints.json"
    require(digest(archive) == digest(BASELINE / archive.name) == prior["archive_field_fingerprint_sha256"],
            "The refreshed archive differs from the baseline.")
    sources = {name: digest(ROOT / name) for name in prior["source_sha256"]}
    changes = {name: {"baseline": old, "priority": sources[name]} for name, old in prior["source_sha256"].items()
               if sources[name] != old}
    require(set(changes) == CHANGED_SOURCE, f"Unexpected changes since baseline: {list(changes)}")
    for path in (archive, Path(__file__), STUDY / "reproduce/fingerprint_archive.py",
                 STUDY / "reproduce/run_archive.py"):
        sources[path.relative_to(ROOT).as_posix()] = digest(path)

    run_id = f"crossing_fan__otranto_{wind:03d}__layline_priority"
    output = STUDY / "runs" / run_id
    require(dry_run or not output.exists(), f"Run already exists: {output}.")
    build = (str(Path(tempfile.gettempdir()) / f"maritime_{run_id}_BUILD") if dry_run
             else mkdtemp(prefix=f"maritime_{run_id}_"))
    for flag, path in zip(LOCAL_OPTIONS, (build, output, output / "track.gpx", output / "navigation.jsonl.gz"), strict=True):
        command[command.index(flag) + 1] = str(path)
    command.append(FLAG)
    require(normalized(command) == normalized(prior["command"]), f"Non-treatment invocation changed at {wind}°.")

    stale = {
        "return_code", "wall_seconds", "started_utc", "finished_utc", "changed_inputs", "source_unchanged",
        "actual_departure_utc", "gate_wait_seconds", "departure_gate_log_sha256",
        "departure_gate_summary_sha256", "maximum_gate_decision_s",
    }
    metadata = {key: value for key, value in prior.items() if key not in stale}
    metadata.update(
        run_id=run_id, study_phase="closest waypoint layline priority after AIS screening", retest_study_id=STUDY.name,
        layline_mode="priority", prioritize_waypoint_layline=True, closest_waypoint_layline=True,
        command=command, acados_build_dir=build, concurrent_workers_requested=len(WINDS),
        guidance_label="Native schedule-veto hybrid · closest feasible waypoints laylines take priority after AIS screen",
        source_sha256=sources, native_common_source_sha256=sources,
        changes_from_layline_off_sources=changes, layline_off_execution=str(prior_path),
        layline_off_execution_sha256=digest(prior_path),
        layline_off_replay_sha256=digest(off / "navigation.jsonl.gz"),
    )
    return output, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    helpers = runpy.run_path(str(STUDY.with_name("otranto_regatta_ablation") / "reproduce/launch.py"))
    plans = [plan(wind, helpers, args.dry_run) for wind in WINDS]
    for _, metadata in plans:
        print(shlex.join(metadata["command"]), flush=True)
    if args.dry_run:
        return
    save = helpers["save"]
    save(STUDY / "launch_preflight.json", {
        "planned_wave_utc": helpers["utc_now"](),
        "reference_study": str(BASELINE),
        "archive_field_fingerprint_sha256": plans[0][1]["archive_field_fingerprint_sha256"],
        "treatment": FLAG,
        "same_inputs": "The command adds priority guidance and changes only build and output paths.",
        "runs": {meta["run_id"]: {name: meta[name] for name in ("layline_off_execution_sha256", "layline_off_replay_sha256")}
                 for _, meta in plans},
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
    if any(result["return_code"] != 0 or not result["source_unchanged"] for result in records.values()):
        raise SystemExit("An attempt failed or a bound input changed. All traces were retained.")


if __name__ == "__main__":
    main()
