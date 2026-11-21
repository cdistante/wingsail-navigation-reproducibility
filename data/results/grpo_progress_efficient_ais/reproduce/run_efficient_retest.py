"""Rerun the fixed native policies with the shorter-detour AIS experiment."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
import json
import multiprocessing
from pathlib import Path
import platform
import sys

STUDY = Path(__file__).resolve().parents[1]
V2 = STUDY.parents[2]
EARLY = STUDY.with_name("grpo_progress_anticipatory_ais")
sys.path.insert(0, str(EARLY / "reproduce"))

import run_anticipatory_retest as earlier

MANIFEST = earlier.MANIFEST
METHODS = earlier.METHODS
CHECKPOINT = earlier.CHECKPOINT
STRENGTH = earlier.STRENGTH
seasonal = earlier.seasonal
stale = earlier.stale
GUARD_SOURCE = Path(__file__).with_name("native_efficient_guard.py")
GUARD = {
    **earlier.GUARD,
    "implementation": "EfficientNativeEngine",
    "baseline": EARLY.name,
}


def scenario_parts(spec):
    from native_efficient_guard import EfficientNativeEngine

    engine, state, goals, start, cap = seasonal.scenario_parts(spec)
    candidate = EfficientNativeEngine(engine.environment, config=engine.config, dynamics=engine.dynamics)
    if candidate.config.reserve_m != GUARD["reserve_m"]:
        raise ValueError("The reserve differs from the locked earlier-avoidance experiment.")
    earlier.ACTIVE_ENGINE = candidate
    return candidate, state, goals, start, cap


def init_worker(output, method, checkpoint):
    earlier.scenario_parts = scenario_parts
    earlier.init_worker(output, method, checkpoint)


def prior_dir(method, comparison="earlier"):
    root = {"earlier": EARLY, "robust_only": earlier.BASELINE}[comparison]
    return root / "native" / method


def baseline_inputs(method):
    directory = prior_dir(method)
    inputs_path = directory / "inputs.json"
    receipt = json.loads(inputs_path.read_text())
    run = json.loads((directory / "run.json").read_text())
    if (receipt["method"] != method or run["method"] != method or receipt["probe"]
            or run["inputs_sha256"] != seasonal.digest(inputs_path)
            or run["episodes_sha256"] != seasonal.digest(directory / "episodes.csv")):
        raise ValueError(f"The earlier-avoidance baseline receipt changed for {method}.")
    return receipt


def locked_protocol():
    scenarios, previous = earlier.locked_protocol()
    scenario_ids = [spec["id"] for spec in scenarios]
    baseline_receipts = {}
    for method in METHODS:
        recorded = baseline_inputs(method)
        if (recorded["scenarios"] != scenario_ids
                or any(recorded[key] != value for key, value in previous.items())):
            raise ValueError(f"The earlier-avoidance baseline differs from the original inputs for {method}.")
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
        "prior_early_guard": previous["guard"],
        "prior_early_receipts_sha256": baseline_receipts,
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
    packages = {name: version(name) for name in ("numpy", "scipy", "xarray", "casadi", "netCDF4", "pyyaml")}
    if packages != baseline_inputs(args.method)["package_versions"]:
        raise ValueError(f"Simulator package versions differ from the earlier {args.method} run.")

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
    if len(scenarios) == 1:
        init_worker(*init_args)
        row = earlier.execute(scenarios[0])
        ordered_rows = [row]
        print(f"{args.method} {row['scenario']} {row['capture_count']}/{row['expected_captures']} "
              f"{row['status']} {row['duration_s']:.2f}s truth={row['min_truth_margin_m']} "
              f"anticipation={row['anticipation_events']}", flush=True)
    else:
        rows = {}
        with ProcessPoolExecutor(max_workers=receipt["workers"], mp_context=multiprocessing.get_context("spawn"),
                                 initializer=init_worker, initargs=init_args) as pool:
            jobs = {pool.submit(earlier.execute, spec): spec["id"] for spec in scenarios}
            for job in as_completed(jobs):
                case = jobs[job]
                try:
                    row = job.result()
                except Exception as error:
                    raise RuntimeError(f"Efficient native run failed in {args.method}/{case}.") from error
                rows[case] = row
                print(f"{args.method:<13} {case} {row['capture_count']}/{row['expected_captures']} "
                      f"{row['status']} {row['duration_s']:.2f}s changes={row['learned_changes']} "
                      f"anticipation={row['anticipation_events']}", flush=True)
        ordered_rows = [rows[case] for case in receipt["scenarios"]]

    episodes = output / "episodes.csv"
    seasonal.save_csv(episodes, ordered_rows)
    seasonal.save_json(output / "run.json", {
        "method": args.method, "inputs_sha256": seasonal.digest(output / "inputs.json"),
        "episodes_sha256": seasonal.digest(episodes), "summary": seasonal.summary(ordered_rows),
        "trace_sha256es": {row["trajectory"]: seasonal.digest(output / row["trajectory"]) for row in ordered_rows},
    })


if __name__ == "__main__":
    main()
