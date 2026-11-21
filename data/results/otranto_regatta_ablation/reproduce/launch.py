"""Launch paired fresh Otranto loops with only the regatta flag toggled."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[2]
REFERENCE = STUDY.with_name("otranto_underway_ais_retest")
FLAG = "--regatta-time-to-mark"
MODES = ("on", "off")
ARMS = tuple(f"{encounter}__otranto_{wind:03d}__{method}"
             for encounter in ("crossing_fan", "opposed_crossing")
             for wind in (355, 5) for method in ("classical", "transformer"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_hash(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"Reference input changed at {path}: expected {expected}, got {actual}")


def runtime_receipt(reference_version):
    source = os.environ["ACADOS_SOURCE_DIR"]
    commit = subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip()
    packages = {}
    for name in reference_version["packages"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "source checkout / not registered as a distribution"
    if (sys.version, packages, commit) != (
        reference_version["python"], reference_version["packages"], reference_version["acados"]["commit"]
    ):
        raise ValueError("The Python packages or acados commit differ from the frozen reference.")
    return {
        "recorded_utc": utc_now(), "python": sys.version, "python_executable": sys.executable,
        "platform": platform.platform(), "logical_cpus": os.cpu_count(),
        "available_logical_cpus_at_least_worker_count": (os.cpu_count() or 0) >= 16,
        "host_load_averages_before_run": list(os.getloadavg()), "packages": packages,
        "acados": {"source": source, "commit": commit},
        "wave": "All 16 fresh runs start in one wave with adjacent on/off workers and one numerical thread each.",
        "latency_scope": "Each actual AIS selection through applied command or strict stop; virtual time pauses during calculation.",
    }


def prepare_reference():
    """Bind source, model, fixture and archive bytes to the prior completed study."""
    delivery = load(REFERENCE / "deliverable_verification.json")
    expected = delivery["receipts"]["audit_outputs_sha256.json"]
    manifest = REFERENCE / "audit_outputs_sha256.json"
    verify_hash(manifest, expected)
    reference_hashes = load(manifest)
    for rel in ("protocol.json", "underway_config.json", "baseline_provenance.json",
                "archive_field_fingerprints.json", "software_versions.json", "reproduce/fingerprint_archive.py"):
        verify_hash(REFERENCE / rel, reference_hashes[rel])
    verify_hash(STUDY / "reproduce/fingerprint_archive.py", reference_hashes["reproduce/fingerprint_archive.py"])
    runtime = runtime_receipt(load(REFERENCE / "software_versions.json"))
    for arm in ARMS:
        for name in ("execution.json", "experiment.json", "navigation.jsonl.gz", "departure_gate.json"):
            rel = Path("runs") / arm / name
            verify_hash(REFERENCE / rel, reference_hashes[rel.as_posix()])
    reference_execution = {arm: load(REFERENCE / "runs" / arm / "execution.json") for arm in ARMS}
    for arm, execution in reference_execution.items():
        if execution["return_code"] != 0 or not execution["source_unchanged"] or execution["radius_m"] != 50:
            raise ValueError(f"Reference arm {arm} is incomplete or differs from 50 m.")
        for name, checksum in execution["source_sha256"].items():
            verify_hash(ROOT / name, checksum)
            original = ROOT / name
            if original.is_relative_to(REFERENCE):
                rel = original.relative_to(REFERENCE)
                if rel != Path("reproduce/launch.py"):
                    verify_hash(STUDY / rel, checksum)
    subprocess.run([sys.executable, "-B", str(STUDY / "reproduce/fingerprint_archive.py")],
                   cwd=ROOT, check=True)
    verify_hash(STUDY / "archive_field_fingerprints.json", reference_hashes["archive_field_fingerprints.json"])
    return runtime, reference_execution, expected, reference_hashes


def command_for(prior, output, build, mode):
    command = prior.copy()
    if command[:2] != [sys.executable, "-B"] or command.count(FLAG) != 1:
        raise ValueError("Historical invocation or regatta flag differs from the runtime.")
    command[2] = str(STUDY / Path(command[2]).relative_to(REFERENCE))
    replacements = {
        "--acados-build-dir": str(build), "--output-dir": str(output),
        "--output-gpx": str(output / "track.gpx"), "--record-replay": str(output / "navigation.jsonl.gz"),
    }
    for option in ("--traffic-fixture", "--underway-traffic-fixture"):
        old_fixture = Path(command[command.index(option) + 1])
        replacements[option] = str(STUDY / old_fixture.relative_to(REFERENCE))
    for option, value in replacements.items():
        if command.count(option) != 1:
            raise ValueError(f"Reference option appears unexpectedly: {option}")
        command[command.index(option) + 1] = value
    if mode == "off":
        command.remove(FLAG)
    return command


def normalized(command, *, remove_regatta=False):
    result = command.copy()
    result[2] = Path(result[2]).name
    for option in ("--acados-build-dir", "--output-dir", "--output-gpx", "--record-replay"):
        result[result.index(option) + 1] = option
    for option in ("--traffic-fixture", "--underway-traffic-fixture"):
        filename = Path(result[result.index(option) + 1])
        result[result.index(option) + 1] = str(Path(*filename.parts[-3:]))
    if remove_regatta and FLAG in result:
        result.remove(FLAG)
    return result


def planned_run(arm, mode, reference, dry_run):
    name = f"{arm}__regatta_{mode}"
    output = STUDY / "runs" / name
    primary = REFERENCE / "runs" / arm
    build = (Path(tempfile.gettempdir()) / f"maritime_ablation_{name}_BUILD" if dry_run else
             Path(tempfile.mkdtemp(prefix=f"maritime_ablation_{name}_")))
    metadata = load(primary / "experiment.json")
    for key in ("traffic_fixture", "traffic_fixture_json", "traffic_truth_trace_csv", "truth_trace",
                "underway_traffic_fixture", "underway_truth_trace_csv", "retest_config"):
        metadata[key] = str(STUDY / Path(metadata[key]).relative_to(REFERENCE))
    source_paths = []
    for name_key in reference["source_sha256"]:
        path = ROOT / name_key
        if path.is_relative_to(REFERENCE):
            path = STUDY / path.relative_to(REFERENCE)
        source_paths.append(path)
    sources = {str(path.relative_to(ROOT)): digest(path) for path in source_paths}
    command = command_for(reference["command"], output, build, mode)
    if normalized(command, remove_regatta=True) != normalized(reference["command"], remove_regatta=True):
        raise ValueError(f"{name} changes a scenario input in addition to the regatta flag.")
    if mode == "on" and normalized(command) != normalized(reference["command"]):
        raise ValueError(f"{name} differs from the prior regatta-on invocation.")
    enabled = mode == "on"
    metadata.update(
        base_arm=arm, regatta_mode=mode, regatta_enabled=enabled, run_id=name, command=command,
        treatment="regatta-time-to-mark flag, including its outbound refinement activation",
        retest_study_id=STUDY.name, study_phase="full matched regatta on/off comparison",
        concurrent_workers_requested=16, acados_build_dir=str(build), source_sha256=sources,
        baseline_directory=str(primary), baseline_execution_sha256=digest(primary / "execution.json"),
        baseline_replay_sha256=digest(primary / "navigation.jsonl.gz"),
    )
    if not enabled:
        metadata.update(
            extra_guidance_cli=[], guidance_objective="Original forecast hold and tacks, with return-phase refinement",
            guidance_label="Gamma + Direct acados NMPC; regatta off with original forecast hold/tacks and return refinement",
        )
    return output, metadata


def stage(output, metadata):
    output.mkdir(parents=True, exist_ok=False)
    (output / "command.txt").write_text(shlex.join(metadata["command"]) + "\n")
    for rel, expected in metadata["source_sha256"].items():
        destination = output / "source_snapshot" / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, destination)
        verify_hash(destination, expected)
    save(output / "experiment.json", metadata)


def run_case(output, metadata, start_wave):
    env = os.environ.copy()
    env.update(metadata["thread_environment"], MPLBACKEND="Agg")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    start_wave.wait()
    start_utc, start = utc_now(), time.monotonic()
    with (output / "run.log").open("x", buffering=1) as stream:
        try:
            result = subprocess.run(metadata["command"], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            return_code = result.returncode
        except OSError as error:
            print(f"Invocation failed: {error}", file=stream, flush=True)
            return_code = 127
    wall = time.monotonic() - start
    changed = [rel for rel, expected in metadata["source_sha256"].items() if digest(ROOT / rel) != expected]
    record = {**metadata, "return_code": return_code, "wall_seconds": wall,
              "started_utc": start_utc, "finished_utc": utc_now(),
              "changed_inputs": changed, "source_unchanged": not changed}
    gate_path = output / "departure_gate.json"
    if gate_path.is_file():
        gate = load(gate_path)
        record.update(actual_departure_utc=gate.get("actual_departure_utc"),
                      gate_wait_seconds=gate.get("gate_wait_seconds"),
                      departure_gate_summary_sha256=digest(gate_path),
                      departure_gate_log_sha256=gate.get("log_sha256"),
                      maximum_gate_decision_s=gate.get("maximum_gate_decision_s"))
    save(output / "execution.json", record)
    print(f"{output.name}: exit {return_code}, wall {wall:.1f} s, input changes {changed}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    reserved = [STUDY / "runs" / f"{arm}__regatta_{mode}" for arm in ARMS for mode in MODES]
    if not args.dry_run and any(path.exists() for path in reserved):
        raise SystemExit("A paired-run directory already exists. The launcher preserves all earlier attempts.")
    runtime, reference, manifest_sha, hashes = prepare_reference()
    receipt = {
        "reference_study": str(REFERENCE), "reference_manifest_sha256": manifest_sha,
        "checked_inputs": ["GPX", "bathymetry", "original and underway AIS fixture/trace files",
                           "all prior source hashes", "regional archive arrays and coordinates", "frozen Transformer/model code"],
        "reference_archive_fingerprint_sha256": hashes["archive_field_fingerprints.json"],
        "runtime": runtime, "pairs": {},
    }
    plans = []
    for arm in ARMS:
        pair = [planned_run(arm, mode, reference[arm], args.dry_run) for mode in MODES]
        on, off = [metadata for _, metadata in pair]
        assert normalized(on["command"], remove_regatta=True) == normalized(off["command"])
        assert on["source_sha256"] == off["source_sha256"]
        assert on["thread_environment"] == off["thread_environment"]
        pair_digest = hashlib.sha256(json.dumps(normalized(off["command"])).encode()).hexdigest()
        receipt["pairs"][arm] = {
            "regatta_on": on["run_id"], "regatta_off": off["run_id"],
            "reference_execution_sha256": on["baseline_execution_sha256"],
            "normalized_shared_command_sha256": pair_digest, "only_treatment_flag_differs": True,
        }
        for output, metadata in pair:
            metadata["started_utc"] = utc_now()
            print(shlex.join(metadata["command"]), flush=True)
            plans.append((output, metadata))
    if args.dry_run:
        return
    save(STUDY / "launch_preflight.json", receipt)
    for output, metadata in plans:
        stage(output, metadata)
    print("Starting one wave of 16 paired fresh runs.", flush=True)
    records = {}
    start_wave = threading.Barrier(len(plans))
    with ThreadPoolExecutor(max_workers=len(plans)) as pool:
        tasks = [pool.submit(run_case, output, metadata, start_wave) for output, metadata in plans]
        for task in as_completed(tasks):
            result = task.result()
            records[result["run_id"]] = result
            save(STUDY / "launch_summary.json", [records[name] for name in sorted(records)])
    if any(record["return_code"] != 0 or not record["source_unchanged"] for record in records.values()):
        raise SystemExit("At least one process failed or inputs changed. All available outcomes are retained.")


if __name__ == "__main__":
    main()
