"""Run predictive-fallback Otranto retest arms without overwriting any outcome."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time


STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[2]
CHECKPOINT = ROOT / "data/results/otranto_temporal_hold_20261118/models/temporal_hold.pt"
PROTOCOL = STUDY / "protocol.json"
BASELINE = STUDY.with_name("otranto_fast_recovery_new_traffic")
RETEST_CONFIG = STUDY / "retest_config.json"
DEPARTURE = "2026-09-30T23:50:00Z"
ENCOUNTER = "2026-09-30T23:50:40Z"
SCENARIOS = ("crossing_fan", "opposed_crossing")
ARMS = [f"{scenario}__otranto_{wind:03d}__{method}" for scenario in SCENARIOS
        for wind in (355,5) for method in ("classical","transformer")]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_case(args, method, wind, encounter):
    protocol=json.loads(PROTOCOL.read_text())
    fixture=STUDY/protocol['encounters'][encounter]['definition']
    truth=STUDY/protocol['encounters'][encounter]['truth_trace']
    report=protocol['reporting']
    name = args.name or f"{encounter}__otranto_{wind:03d}__{method}"
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("Use a plain isolated run name.")
    study = args.study.resolve()
    output = study / "runs" / name
    if output.exists():
        raise ValueError(f"{output} already exists; recordings are never overwritten.")
    checkpoint = args.checkpoint.resolve()
    build = (str(Path(tempfile.gettempdir()) / f"maritime_predictive_retest_{name}_BUILD") if args.dry_run else
             tempfile.mkdtemp(prefix=f"maritime_predictive_retest_{name}_"))
    command = [
        args.python, "-B", str(STUDY / "reproduce/run_archive.py"),
        "--regional-otranto", "--batched-wind-grid", "--wind-from", str(wind),
        "--traffic-fixture", str(fixture),
        "--planner", "gamma", "--controller", "acados-direct", "--acados-build-dir", build,
        "--dynamics", "fossen", "--environment", "marine",
        "--route-gpx", "current_route/otranto_short_test_loop.gpx", "--return-to-start",
        "--arrival-radius", "50", "--dt", "5", "--heading", "45", "--flap=-10",
        "--time", DEPARTURE, "--steps", str(args.steps),
        "--local-span", "4715", "--local-resolution", "115", "--obstacle-buffer", "115",
        "--plot-every", "0", "--live-plot-frame", "fixed",
        "--live-plot-radius", "2000", "--live-plot-boat-scale", "4",
        "--gpx-checkpoint-every", "100", "--output-dir", str(output),
        "--output-gpx", str(output / "track.gpx"),
        "--record-replay", str(output / "navigation.jsonl.gz"),
        "--forecast-tacks", "--forecast-tack-refinement", "--forecast-tracking-feedback",
        "--forecast-tack-step", "60", "--forecast-refinement-phase", "return",
        "--forecast-gate-aim-fraction", "0.5", "--forecast-gate-aim-upwind-angle", "60",
        "--forecast-maneuver-commit-seconds", "60", "--limit-upwind-drift", "--forecast-hold-tacks",
        "--ais-scenario", "mixed", "--ais-maneuvering-sharp", "--time-aware-ais", "--colreg",
        "--ais-acceleration-bound", "0.026", "--ais-clearance-reserve", "3",
        "--ais-preview-age-budget", str(report['preview_age_budget_s']),
        "--ais-launch-lead", str(protocol['launch_lead_s']),
        "--ais-report-delay", str(report['delay_s']),
        "--ais-predictive-fallback",
    ]
    for start,stop in report['drop_emission_windows_s']:
        command += ['--ais-drop-reports',str(start),str(stop)]
    if method == "transformer":
        command += ["--temporal-recovery-model", str(checkpoint), "--confirm-temporal-release"]
    paths = sorted((ROOT / "world").rglob("*.py"))
    paths += [ROOT / "config.py", Path(__file__), STUDY / "reproduce/run_archive.py",
              STUDY / "reproduce/traffic_fixture.py", PROTOCOL, RETEST_CONFIG, STUDY / "baseline_provenance.json", checkpoint,
              ROOT / "current_route/otranto_short_test_loop.gpx",
              ROOT / "data/bathymetry/gebco_2026_n44.0_s37.0_w14.0_e22.0.nc",
              fixture, truth]
    sources = {str(path.relative_to(ROOT)): digest(path) for path in paths}
    baseline = BASELINE / "runs" / f"{encounter}__otranto_{wind:03d}__{method}"
    metadata = {
        "method": method, "run_id": name, "encounter_id":encounter, "scenario_id":encounter,
        "traffic_fixture":str(fixture),
        "traffic_fixture_json":str(fixture),"traffic_fixture_sha256":digest(fixture),
        "traffic_truth_trace_csv":str(truth),"truth_trace":str(truth),"traffic_truth_trace_sha256":digest(truth),
        "reporting":report,"protocol_sha256":digest(PROTOCOL),
        "retest_study_id": STUDY.name, "retest_config": str(RETEST_CONFIG),
        "retest_config_sha256": digest(RETEST_CONFIG),
        "baseline_provenance_sha256": digest(STUDY / "baseline_provenance.json"),
        "controller_mode": "receding-horizon fallback with strict applied clearance",
        "predictive_fallback": True,
        "traffic_runtime_input":"Analytical parameter JSON only; truth trace copied for independent audit and not read by controller.",
        "wind_from_degrees": wind, "wind_from": wind,
        "study_phase": "predictive-fallback full retest" if args.steps == 17280 else "engineering pilot",
        "fresh_subprocess": True,
        "AIS_latency_requirement_s": 5.0,
        "launcher_solver_warmup": False,
        "baseline_directory": str(baseline),
        "baseline_execution_sha256": digest(baseline / "execution.json"),
        "baseline_replay_sha256": digest(baseline / "navigation.jsonl.gz"),
        "departure_utc": DEPARTURE, "encounter_utc": ENCOUNTER,
        "dt_s": 5, "radius_m": 50, "maximum_steps": args.steps, "cap_s": 5 * args.steps,
        "return_to_start": True, "forecast_hold_tacks": True,
        "forecast_override_scope": "plant, future tack/hold samples, temporal observations and replay barbs",
        "command": command, "cwd": str(ROOT), "acados_build_dir": build,
        "checkpoint": str(checkpoint) if method == "transformer" else None,
        "frozen_checkpoint_sha256": digest(checkpoint),
        "started_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": sources,
        "thread_environment": {key: "1" for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                                    "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
    }
    print(shlex.join(command), flush=True)
    if args.dry_run:
        print(json.dumps(metadata, indent=2), flush=True)
        return metadata
    output.mkdir(parents=True, exist_ok=False)
    (output / "command.txt").write_text(shlex.join(command) + "\n")
    snapshot = output / "source_snapshot"
    for path in paths:
        target = snapshot / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    save(output / "experiment.json", metadata)
    env = os.environ.copy()
    env.update(metadata["thread_environment"], MPLBACKEND="Agg")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    start = time.monotonic()
    with (output / "run.log").open("w", buffering=1) as stream:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    changed = [str(path.relative_to(ROOT)) for path in paths if digest(path) != sources[str(path.relative_to(ROOT))]]
    record = {**metadata, "return_code": result.returncode, "wall_seconds": time.monotonic() - start,
              "finished_utc": datetime.now(timezone.utc).isoformat(), "changed_inputs": changed,
              "source_unchanged": not changed}
    save(output / "execution.json", record)
    print(f"{name}: exit {result.returncode}; wall {record['wall_seconds']:.1f}s; input changes {changed}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    protocol=json.loads(PROTOCOL.read_text())
    parser.add_argument('--encounter','--scenario',dest='encounter',choices=tuple(protocol['encounters']),action='append')
    parser.add_argument("--method", choices=("classical", "transformer"), action="append")
    parser.add_argument("--wind-from", choices=(355, 5), type=int, action="append")
    parser.add_argument("--name", help="Isolated folder name, for a single arm only.")
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--python", default=os.environ.get("MARITIME_ACADOS_PYTHON", sys.executable))
    parser.add_argument("--steps", type=int, default=17280)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cases = [(method, wind, encounter)
             for encounter in dict.fromkeys(args.encounter or protocol['encounters'])
             for wind in dict.fromkeys(args.wind_from or (355, 5))
             for method in dict.fromkeys(args.method or ("classical", "transformer"))]
    if args.steps <= 0 or not 1 <= args.jobs <= 4 or (args.name and len(cases) != 1):
        parser.error("Use positive steps,1--4 jobs and --name only with a single experimental arm.")
    if args.dry_run:
        for method, wind, encounter in cases:
            run_case(args, method, wind, encounter)
        return
    # A complete8-arm matrix uses two explicit4-worker waves; no later arm
    # starts early merely because a preceding case stopped or finished sooner.
    records=[]
    destination=args.study.resolve()/'launch_summary.json'
    historical=json.loads(destination.read_text()) if destination.exists() else []
    for start in range(0,len(cases),args.jobs):
        wave=cases[start:start+args.jobs]
        print(f"Starting wave {start//args.jobs+1}: {len(wave)} isolated processes",flush=True)
        with ThreadPoolExecutor(max_workers=len(wave)) as pool:
            tasks=[pool.submit(run_case,args,method,wind,encounter) for method,wind,encounter in wave]
            records.extend(task.result() for task in as_completed(tasks))
        accumulated={row['run_id']:row for row in [*historical,*records]}
        save(destination,[accumulated[name] for name in sorted(accumulated)])
    if any(record["return_code"] != 0 or not record["source_unchanged"] for record in records):
        raise SystemExit("At least one run failed or an input changed; every outcome is retained in its execution receipt.")


if __name__ == "__main__":
    main()
