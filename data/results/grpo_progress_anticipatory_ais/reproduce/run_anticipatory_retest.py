"""Replay the frozen held-out policies with causal anticipatory AIS screening."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
from importlib.metadata import version
import json
import multiprocessing
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

STUDY = Path(__file__).resolve().parents[1]
V2 = STUDY.parents[2]
BASELINE = STUDY.with_name("grpo_progress_stale_ais")
sys.path.insert(0, str(BASELINE / "reproduce"))

import run_stale_retest as stale

MANIFEST = stale.MANIFEST
METHODS = stale.METHODS
CHECKPOINT = stale.CHECKPOINT
STRENGTH = stale.STRENGTH
seasonal = stale.seasonal
GUARD_SOURCE = Path(__file__).with_name("native_anticipatory_guard.py")
GUARD = {
    "implementation": "AnticipatoryNativeEngine",
    "baseline": BASELINE.name,
    "report_guard_uncertainty_multiplier": 2,
    "reserve_m": 3.0,
    "evaluation_envelope": "original full report and truth swept radii",
}
ACTIVE_ENGINE = None


def scenario_parts(spec):
    global ACTIVE_ENGINE
    from native_anticipatory_guard import AnticipatoryNativeEngine

    engine, state, goals, start, cap = seasonal.scenario_parts(spec)
    anticipation = AnticipatoryNativeEngine(engine.environment, config=engine.config, dynamics=engine.dynamics)
    if anticipation.config.reserve_m != GUARD["reserve_m"]:
        raise ValueError("The reserve differs from the locked stale-report experiment.")
    ACTIVE_ENGINE = anticipation
    return anticipation, state, goals, start, cap


def init_worker(output, method, checkpoint):
    stale.reference_runner.init_worker(output, method, checkpoint)
    # Preserve the native executor, GRPO selection, and telemetry schema.
    stale.reference_runner.seasonal = SimpleNamespace(
        scenario_parts=scenario_parts,
        finite_records=seasonal.finite_records,
        measured_margin=seasonal.measured_margin,
        json_value=seasonal.json_value,
    )


def execute(spec):
    global ACTIVE_ENGINE
    try:
        row = stale.reference_runner.execute(spec)
        events = list(getattr(ACTIVE_ENGINE, "anticipation_events", ()))
        trace = stale.reference_runner.OUTPUT / row["trajectory"]
        with gzip.open(trace, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        payload["anticipation_events"] = events
        staged = trace.with_suffix(trace.suffix + ".part")
        with gzip.open(staged, "wt", encoding="utf-8", compresslevel=4) as stream:
            json.dump(payload, stream, default=seasonal.json_value, separators=(",", ":"), allow_nan=False)
        staged.replace(trace)
        row["anticipation_events"] = len(events)
        return row
    finally:
        ACTIVE_ENGINE = None


def prior_dir(method):
    return BASELINE / "native" / method


def baseline_inputs(method):
    directory = prior_dir(method)
    inputs_path = directory / "inputs.json"
    run_path = directory / "run.json"
    receipt = json.loads(inputs_path.read_text())
    run = json.loads(run_path.read_text())
    if (receipt["method"] != method or run["method"] != method
            or run["inputs_sha256"] != seasonal.digest(inputs_path)
            or run["episodes_sha256"] != seasonal.digest(directory / "episodes.csv")):
        raise ValueError(f"The stale-guard baseline receipt changed for {method}.")
    return receipt


def locked_protocol():
    scenarios, previous = stale.locked_protocol()
    scenario_ids = [spec["id"] for spec in scenarios]
    baseline_receipts = {}
    for method in METHODS:
        recorded = baseline_inputs(method)
        if (recorded["scenarios"] != scenario_ids
                or any(recorded[key] != value for key, value in previous.items())):
            raise ValueError(f"The prior stale-guard comparison differs from the locked inputs for {method}.")
        baseline_receipts[method] = {
            name: seasonal.digest(prior_dir(method) / name)
            for name in ("inputs.json", "run.json", "episodes.csv")
        }

    provenance = {
        **previous,
        "source_sha256": {**previous["source_sha256"], **{
            str(path.relative_to(V2)): seasonal.digest(path)
            for path in (Path(__file__), GUARD_SOURCE)
        }},
        "guard": GUARD,
        "prior_stale_guard": previous["guard"],
        "prior_stale_receipts_sha256": baseline_receipts,
    }
    return scenarios, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scenario", help="Run one locked test case in an isolated probe directory.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.scenario and (args.output is not None
                          and args.output.resolve().is_relative_to(STUDY / "native")):
        parser.error("Single-case probes must be outside the main native output directory.")
    full_roster, provenance = locked_protocol()
    scenarios = [spec for spec in full_roster if args.scenario in (None, spec["id"])]
    if not scenarios:
        parser.error(f"Unknown locked case {args.scenario}")

    default_output = STUDY / "native" / args.method
    if args.scenario:
        default_output = STUDY / "probe" / args.scenario / args.method
    output = (args.output or default_output).resolve()
    checkpoint = CHECKPOINT if args.method == "grpo_progress" else None
    packages = {package: version(package) for package in ("numpy", "scipy", "xarray", "casadi", "netCDF4", "pyyaml")}
    if packages != baseline_inputs(args.method)["package_versions"]:
        raise ValueError(f"Simulator package versions differ from the previous {args.method} run.")

    output.mkdir(parents=True, exist_ok=False)
    (output / "telemetry").mkdir()
    receipt = {
        **provenance, "method": args.method, "scenarios": [spec["id"] for spec in scenarios],
        "probe": args.scenario is not None, "workers": min(args.workers, len(scenarios)),
        "platform": platform.platform(), "python": platform.python_version(), "package_versions": packages,
        "checkpoint": str(checkpoint.relative_to(V2)) if checkpoint else None,
        "checkpoint_sha256": seasonal.digest(checkpoint) if checkpoint else None,
        "adapter_sha256": seasonal.digest(stale.ADAPTER) if checkpoint else None,
        "legacy_support_sha256": seasonal.digest(stale.SUPPORT) if checkpoint else None,
        "residual_strength": STRENGTH if checkpoint else 0.,
    }
    seasonal.save_json(output / "inputs.json", receipt)
    init_args = (str(output), args.method, str(checkpoint) if checkpoint else None)
    rows = {}
    if len(scenarios) == 1:
        init_worker(*init_args)
        row = execute(scenarios[0])
        rows[row["scenario"]] = row
    else:
        with ProcessPoolExecutor(max_workers=receipt["workers"], mp_context=multiprocessing.get_context("spawn"),
                                 initializer=init_worker, initargs=init_args) as pool:
            jobs = {pool.submit(execute, spec): spec["id"] for spec in scenarios}
            for job in as_completed(jobs):
                case = jobs[job]
                try:
                    row = job.result()
                except Exception as error:
                    raise RuntimeError(f"Anticipatory native run failed in {args.method}/{case}.") from error
                rows[case] = row
                print(f"{args.method:<13} {case} {row['capture_count']}/{row['expected_captures']} "
                      f"{row['status']} {row['duration_s']:.2f}s changes={row['learned_changes']} "
                      f"anticipation={row['anticipation_events']}", flush=True)
    ordered_rows = [rows[case] for case in receipt["scenarios"]]
    if len(ordered_rows) == 1:
        row = ordered_rows[0]
        print(f"{args.method} {row['scenario']} {row['capture_count']}/{row['expected_captures']} "
              f"{row['status']} {row['duration_s']:.2f}s truth={row['min_truth_margin_m']} "
              f"anticipation={row['anticipation_events']}", flush=True)

    episodes = output / "episodes.csv"
    seasonal.save_csv(episodes, ordered_rows)
    seasonal.save_json(output / "run.json", {
        "method": args.method, "inputs_sha256": seasonal.digest(output / "inputs.json"),
        "episodes_sha256": seasonal.digest(episodes), "summary": seasonal.summary(ordered_rows),
        "trace_sha256es": {row["trajectory"]: seasonal.digest(output / row["trajectory"]) for row in ordered_rows},
    })


if __name__ == "__main__":
    main()
