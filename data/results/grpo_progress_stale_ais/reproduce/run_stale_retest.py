"""Retest the locked physics and GRPO policies with the robust stale-AIS guard."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
import json
import multiprocessing
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

STUDY = Path(__file__).resolve().parents[1]
V2 = STUDY.parents[2]
REFERENCE = STUDY.with_name("grpo_progress_native")
sys.path.insert(0, str(REFERENCE / "reproduce"))

import run_native_comparison as reference_runner
from native_stale_guard import RobustNativeEngine


MANIFEST = reference_runner.MANIFEST
METHODS = reference_runner.METHODS
STRENGTH = reference_runner.STRENGTH
seasonal = reference_runner.seasonal
CHECKPOINT = REFERENCE / "training" / "grpo_progress_selected.npz"
TRAINING_RECEIPT = CHECKPOINT.with_name("training_results.json")
ADAPTER = REFERENCE / "reproduce" / "native_grpo.py"
SUPPORT = STUDY.with_name("learned_safe_mark_policy") / "learned_mark_policy.json"
GUARD_SOURCE = Path(__file__).with_name("native_stale_guard.py")
GUARD = {
    "implementation": "RobustNativeEngine",
    "report_guard_uncertainty_multiplier": 2,
    "reserve_m": 3.0,
    "evaluation_envelope": "original full report and truth swept radii",
}


def scenario_parts(spec):
    engine, state, goals, start, cap = seasonal.scenario_parts(spec)
    robust = RobustNativeEngine(engine.environment, config=engine.config, dynamics=engine.dynamics)
    if robust.config.reserve_m != GUARD["reserve_m"]:
        raise ValueError("Native reserve differs from the fixed retest guard configuration.")
    return robust, state, goals, start, cap


def init_worker(output, method, checkpoint):
    reference_runner.init_worker(output, method, checkpoint)
    # Keep the episode execution and telemetry format from the first study.
    # Only the scenario factory in this worker supplies the hardened engine.
    reference_runner.seasonal = SimpleNamespace(
        scenario_parts=scenario_parts,
        finite_records=seasonal.finite_records,
        measured_margin=seasonal.measured_margin,
        json_value=seasonal.json_value,
    )


def load_reference_inputs(method):
    directory = REFERENCE / "native" / method
    path = directory / "inputs.json"
    receipt = json.loads(path.read_text())
    run = json.loads((directory / "run.json").read_text())
    if run["inputs_sha256"] != seasonal.digest(path) or run["episodes_sha256"] != seasonal.digest(directory / "episodes.csv"):
        raise ValueError(f"The original {method} inputs or episodes changed.")
    return receipt


def locked_protocol():
    manifest, forcing = seasonal.source_inputs(MANIFEST)
    scenarios = [spec for spec in manifest["scenarios"] if spec["split"] == "test"]
    scenario_ids = [spec["id"] for spec in scenarios]
    if manifest["roster"] != "expanded" or len(scenarios) != 12 or len(set(scenario_ids)) != 12:
        raise ValueError("Expected the full locked test roster of 12 cases.")
    provenance = seasonal.provenance(MANIFEST, forcing)
    for path in (Path(__file__), Path(reference_runner.__file__), GUARD_SOURCE, V2 / "world/learned_mark_policy.py"):
        provenance["source_sha256"][str(path.relative_to(V2))] = seasonal.digest(path)
    receipts = {method: load_reference_inputs(method) for method in METHODS}
    for method, old in receipts.items():
        if old["scenarios"] != scenario_ids or any(
            old[key] != provenance[key] for key in ("split_manifest_sha256", "forcing_sha256", "bathymetry_sha256")
        ):
            raise ValueError(f"The locked route, traffic, archive or chart differs from the original {method} arm.")
    training = json.loads(TRAINING_RECEIPT.read_text())
    original_grpo = receipts["grpo_progress"]
    if (training["selected_checkpoint_sha256"] != seasonal.digest(CHECKPOINT)
            or original_grpo["checkpoint_sha256"] != seasonal.digest(CHECKPOINT)
            or original_grpo["adapter_sha256"] != seasonal.digest(ADAPTER)
            or original_grpo["legacy_support_sha256"] != seasonal.digest(SUPPORT)
            or original_grpo["residual_strength"] != STRENGTH):
        raise ValueError("The GRPO checkpoint, adapter, support, or residual strength differs from the original test.")
    provenance["training_receipt_sha256"] = seasonal.digest(TRAINING_RECEIPT)
    provenance["model_sha256"] = seasonal.digest(CHECKPOINT)
    provenance["guard"] = GUARD
    return scenarios, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    output = (args.output or STUDY / "native" / args.method).resolve()
    checkpoint = CHECKPOINT if args.method == "grpo_progress" else None
    scenarios, provenance = locked_protocol()
    output.mkdir(parents=True, exist_ok=False)
    (output / "telemetry").mkdir()
    receipt = {
        **provenance, "method": args.method, "scenarios": [spec["id"] for spec in scenarios],
        "workers": args.workers, "platform": platform.platform(), "python": platform.python_version(),
        "package_versions": {package: version(package)
                             for package in ("numpy", "scipy", "xarray", "casadi", "netCDF4", "pyyaml")},
        "checkpoint": str(checkpoint.relative_to(V2)) if checkpoint else None,
        "checkpoint_sha256": seasonal.digest(checkpoint) if checkpoint else None,
        "adapter_sha256": seasonal.digest(ADAPTER) if checkpoint else None,
        "legacy_support_sha256": seasonal.digest(SUPPORT) if checkpoint else None,
        "residual_strength": STRENGTH if checkpoint else 0.,
    }
    seasonal.save_json(output / "inputs.json", receipt)
    init_args = (str(output), args.method, str(checkpoint) if checkpoint else None)
    rows = {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=init_worker, initargs=init_args) as pool:
        jobs = {pool.submit(reference_runner.execute, spec): spec["id"] for spec in scenarios}
        for job in as_completed(jobs):
            case = jobs[job]
            try:
                row = job.result()
            except Exception as error:
                raise RuntimeError(f"Hardened native run failed in {args.method}/{case}.") from error
            rows[case] = row
            print(f"{args.method:<13} {case} {row['capture_count']}/{row['expected_captures']} "
                  f"{row['status']} {row['duration_s']:.2f}s changes={row['learned_changes']}", flush=True)
    ordered_rows = [rows[case] for case in receipt["scenarios"]]
    episodes = output / "episodes.csv"
    seasonal.save_csv(episodes, ordered_rows)
    seasonal.save_json(output / "run.json",
                       {"method": args.method, "inputs_sha256": seasonal.digest(output / "inputs.json"),
                        "episodes_sha256": seasonal.digest(episodes), "summary": seasonal.summary(ordered_rows),
                        "trace_sha256es": {row["trajectory"]: seasonal.digest(output / row["trajectory"])
                                            for row in ordered_rows}})


if __name__ == "__main__":
    main()
