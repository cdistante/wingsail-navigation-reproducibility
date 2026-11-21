"""Run crossing-fan Otranto physics/hybrid pairs with identical AIS and marine inputs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys
from tempfile import mkdtemp
from threading import Barrier


STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[2]
BASE = STUDY.with_name("otranto_regatta_ablation")
MODEL = STUDY.with_name("learned_weather_ais_policy")
ACTOR = STUDY / "reproduce/run_archive.py"
HYBRID_ACTOR = STUDY / "reproduce/run_native_hybrid.py"
COMMON_FLAGS = ("--native-control-timing", "--ais-candidate-handoff")
TREATMENT_FLAG = "--hybrid-forecast-tacks"
TREATMENT_VALUE = "--hybrid-model-dir"
WINDS = (5, 355)
PINNED_RUNNER = "a3b1f9bde1dea42ad83e9b98c88ee3836748b74212cbe9a1d8b75b2eebccfe97"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalized_pair(command):
    normalized = command.copy()
    normalized[2] = ACTOR.name
    for option in ("--acados-build-dir", "--output-dir", "--output-gpx", "--record-replay"):
        normalized[normalized.index(option)+1] = option
    if TREATMENT_FLAG in normalized:
        normalized.remove(TREATMENT_FLAG)
        index = normalized.index(TREATMENT_VALUE)
        del normalized[index:index+2]
    return normalized


def verify_archive(digest):
    receipt = STUDY / "archive_field_fingerprints.json"
    reference = BASE / receipt.name
    if not receipt.exists():
        subprocess.run([sys.executable, "-B", str(STUDY / "reproduce/fingerprint_archive.py")],
                       cwd=ROOT, check=True)
    require(digest(receipt) == digest(reference), "Regional archived marine fields differ.")
    return receipt


def original_receipt(wind):
    folder = BASE / "runs" / f"crossing_fan__otranto_{wind:03d}__classical__regatta_off"
    path = folder / "execution.json"
    return folder, path, json.loads(path.read_text())


def model_sources(model, digest):
    # Bind deployable checkpoints, recipe metadata, and all runtime-side experiment helpers.
    files = [model / name for name in ("joint_critic.npz", "fresh_weather_encoder.pt", "training_receipt.json")]
    files.extend(sorted((model / "reproduce").glob("*.py")))
    return {str(path.relative_to(ROOT)): digest(path) for path in files}


def plan(wind, mode, model, archive, helpers):
    old_folder, old_path, previous = original_receipt(wind)
    digest = helpers["digest"]
    require(previous["return_code"] == 0 and previous["source_unchanged"]
            and previous["regatta_mode"] == "off" and previous["radius_m"] == 50
            and previous["dt_s"] == 5 and previous["maximum_steps"] == 17280,
            f"Unexpected reference parameters for wind {wind}.")
    common, changes = {}, {}
    for name, old_hash in previous["source_sha256"].items():
        source = ROOT / name
        current = digest(source)
        if current != old_hash:
            changes[name] = {"reference": old_hash, "current": current}
        if source.is_relative_to(BASE):
            relative = source.relative_to(BASE)
            candidate = STUDY / relative
            if relative == Path("reproduce/launch.py"):
                continue
            require(digest(candidate) == old_hash, f"Cloned source or fixture differs: {candidate}")
            source = candidate
        common[str(source.relative_to(ROOT))] = digest(source)
    require(set(changes) == {"world/ais_buffer_maneuver.py", "world/run_legacy_fossen.py"},
            f"Unexpected changes to original native sources: {list(changes)}")
    require(common["world/run_legacy_fossen.py"] == PINNED_RUNNER, "Pinned common native runner changed.")
    for path in (Path(__file__), archive):
        common[str(path.relative_to(ROOT))] = digest(path)
    sources = common.copy()
    model_hashes = {}
    if mode == "hybrid":
        model_hashes = model_sources(model, digest)
        sources.update(model_hashes)
        for path in (HYBRID_ACTOR, ROOT / "world/hybrid_tack_guidance.py"):
            sources[str(path.relative_to(ROOT))] = digest(path)

    name = f"crossing_fan__otranto_{wind:03d}__{mode}"
    out = STUDY / "runs" / name
    require(not out.exists(), f"{out} exists; native results are preserved.")
    build = mkdtemp(prefix=f"maritime_native_{wind:03d}_{mode}_")
    command = previous["command"].copy()
    require(command[:2] == [sys.executable, "-B"], "Native actor interpreter changed.")
    command[2] = str(HYBRID_ACTOR if mode == "hybrid" else ACTOR)
    metadata = {key: value for key, value in previous.items() if key not in {
        "return_code", "wall_seconds", "started_utc", "finished_utc", "source_unchanged", "changed_inputs",
        "actual_departure_utc", "gate_wait_seconds", "departure_gate_summary_sha256",
        "departure_gate_log_sha256", "maximum_gate_decision_s",
    }}
    for key in ("traffic_fixture", "traffic_fixture_json", "traffic_truth_trace_csv", "truth_trace",
                "underway_traffic_fixture", "underway_truth_trace_csv", "retest_config"):
        metadata[key] = str(STUDY / Path(metadata[key]).relative_to(BASE))
    for option, value in (
        ("--acados-build-dir", build), ("--output-dir", out), ("--output-gpx", out/"track.gpx"),
        ("--record-replay", out/"navigation.jsonl.gz"), ("--traffic-fixture", metadata["traffic_fixture"]),
        ("--underway-traffic-fixture", metadata["underway_traffic_fixture"]),
    ):
        command[command.index(option)+1] = str(value)
    command.extend(COMMON_FLAGS)
    if mode == "hybrid":
        command.extend((TREATMENT_FLAG, TREATMENT_VALUE, str(model)))
        physics = STUDY / "runs" / name.replace("__hybrid", "__physics") / "experiment.json"
        reference = json.loads(physics.read_text())
        require(normalized_pair(reference["command"]) == normalized_pair(command),
                f"Physics and hybrid differ outside the registered treatment for wind {wind}.")
        require(reference["native_common_source_sha256"] == common,
                f"Common sources changed after physics launched for wind {wind}.")

    model_checksum = hashlib.sha256(json.dumps(model_hashes, sort_keys=True).encode()).hexdigest() if model_hashes else None
    metadata.update(
        run_id=name, study_phase="native physics/hybrid matched traffic", retest_study_id=STUDY.name,
        native_hybrid_enabled=mode == "hybrid", native_hybrid_mode=mode,
        native_hybrid_checkpoint=str(model) if model_hashes else None,
        native_hybrid_checkpoint_sha256=model_checksum, native_hybrid_model_files_sha256=model_hashes,
        native_common_source_sha256=common, native_control_timing=True, native_AIS_candidate_handoff=True,
        command=command, acados_build_dir=build, concurrent_workers_requested=len(WINDS),
        baseline_directory=str(old_folder), baseline_execution_sha256=digest(old_path),
        baseline_replay_sha256=digest(old_folder / "navigation.jsonl.gz"), source_sha256=sources,
        archive_field_fingerprint_sha256=digest(archive), changes_from_original_sources=changes,
        guidance_label="Physics forecast tacks and holding trims" if mode == "physics"
        else "Guarded learned ranking of native forecast tack schedules",
        guidance_objective="Shared physics and AIS safety; same candidate admission both arms",
    )
    return out, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("physics", "hybrid"), required=True)
    parser.add_argument("--model-dir", type=Path, default=MODEL)
    args = parser.parse_args()
    helpers = runpy.run_path(str(BASE / "reproduce/launch.py"))
    archive = verify_archive(helpers["digest"])
    plans = [plan(wind, args.mode, args.model_dir.resolve(), archive, helpers) for wind in WINDS]
    preflight = {"mode": args.mode, "UTC": datetime.now(timezone.utc).isoformat(),
                 "host_load": list(os.getloadavg()), "logical_cpus": os.cpu_count(),
                 "pair_control": "Common inputs and normalized command checked against live physics receipts.",
                 "launch_scope": "Two winds start together in each mode. Physics and hybrid modes have staggered starts.",
                 "archive_fingerprint_sha256": helpers["digest"](archive),
                 "commands": {metadata["run_id"]: metadata["command"] for _, metadata in plans}}
    helpers["save"](STUDY / f"launch_preflight_{args.mode}.json", preflight)
    for out, metadata in plans:
        print(shlex.join(metadata["command"]), flush=True)
        helpers["stage"](out, metadata)
    barrier = Barrier(len(plans))
    records = {}
    with ThreadPoolExecutor(max_workers=len(plans)) as pool:
        tasks = [pool.submit(helpers["run_case"], out, metadata, barrier) for out, metadata in plans]
        for task in as_completed(tasks):
            row = task.result()
            records[row["run_id"]] = row
            helpers["save"](STUDY / f"launch_summary_{args.mode}.json", [records[name] for name in sorted(records)])
    if any(row["return_code"] != 0 or not row["source_unchanged"] for row in records.values()):
        raise SystemExit("A native attempt failed or an input changed. Raw results retained.")


if __name__ == "__main__":
    main()
