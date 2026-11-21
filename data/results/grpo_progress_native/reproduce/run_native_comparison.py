"""Run one arm on every locked autumn seasonal case using the existing native plant."""

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

STUDY = Path(__file__).resolve().parents[1]
V2 = STUDY.parents[2]
SEASONAL = STUDY.with_name("seasonal_native_training")
MANIFEST = SEASONAL / "manifest_expanded.json"
sys.path.insert(0, str(SEASONAL / "reproduce"))

import run_seasonal as seasonal


METHODS = ("physics", "grpo_progress")
STRENGTH = 1.0
POLICY = None
OUTPUT = None
METHOD = None


def init_worker(output, method, checkpoint):
    global POLICY, OUTPUT, METHOD
    seasonal.init(MANIFEST, None, False)
    METHOD = method
    OUTPUT = Path(output)
    POLICY = None
    if checkpoint is not None:
        from native_grpo import NativeGRPO

        POLICY = NativeGRPO.load(checkpoint)


def execute(spec):
    engine, state, goals, start, cap = seasonal.scenario_parts(spec)
    pending_selections = []

    def strategy(decision):
        selection = POLICY.select(decision, dynamics=engine.dynamics,
                                  remaining_s=cap - decision.elapsed_s,
                                  capture_radius_m=engine.config.capture_radius_m,
                                  strength=STRENGTH)
        pending_selections.append({
            "elapsed_s": decision.elapsed_s, "utc": decision.when, "goal": decision.goal.name,
            "physics_index": decision.physics_index, "requested_index": selection.index,
            "status": selection.status, "eligible_candidates": selection.eligible_candidates,
            "native_eta_regret_s": selection.native_eta_regret_s, "corrected_gain_s": selection.corrected_gain_s,
        })
        return selection.index

    outcome = engine.run(state, goals, start, strategy=strategy if POLICY is not None else None,
                         cap_seconds=cap, trace=True)
    trace = OUTPUT / "telemetry" / f"{spec['id']}__{METHOD}.json.gz"
    staged = trace.with_suffix(trace.suffix + ".part")
    with gzip.open(staged, "wt", encoding="utf-8", compresslevel=4) as stream:
        json.dump(
            {"states": outcome.trace, "controls": seasonal.finite_records(outcome.controls),
             "control_attempts": seasonal.finite_records(outcome.control_attempts),
             "decisions": outcome.decisions, "captures": outcome.captures,
             "strategy_evaluations": pending_selections},
            stream, default=seasonal.json_value, separators=(",", ":"), allow_nan=False,
        )
    staged.replace(trace)
    expected = spec["expected_captures"]
    captures = [{**actual, **planned} for actual, planned in zip(outcome.captures, expected)]
    if len(captures) != len(outcome.captures) or any(
        row["name"] != row["waypoint"] or row["distance_m"] > spec["capture_radius_m"] + 1e-4
        for row in captures
    ):
        raise ValueError(f"Invalid capture radius or order in {spec['id']}/{METHOD}.")
    completed = outcome.success and len(captures) == len(goals)
    decision_times = [row["complete_control_seconds"] for row in outcome.control_attempts]
    return {
        "scenario": spec["id"], "method": METHOD, "route": spec["route_gpx"],
        "family": spec["route_family"], "weather_block": spec["archive_block"],
        "traffic": spec["traffic_profile"], "departure_utc": spec["departure_utc"],
        "status": outcome.status, "completed": completed,
        "duration_s": captures[-1]["elapsed_s"] if completed else outcome.elapsed_s, "cap_s": cap,
        "expected_captures": len(goals), "captures": captures, "capture_count": len(captures),
        "min_report_margin_m": seasonal.measured_margin(outcome.report_minimum_m),
        "min_truth_margin_m": seasonal.measured_margin(outcome.truth_minimum_m),
        "maximum_report_age_s": max((row["max_report_age_s"] for row in outcome.controls), default=0.),
        "max_lost_epochs": max((row["dropped_report_epochs"] for row in outcome.controls), default=0),
        "learned_changes": sum(row["physics_index"] != row["requested_index"] for row in pending_selections),
        "strategy_calls": len(pending_selections), "shield_overrides": outcome.shield_overrides,
        "maximum_decision_s": max(decision_times, default=0.), "overruns_5s": sum(t >= 5. for t in decision_times),
        "trajectory": str(trace.relative_to(OUTPUT)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.method == "grpo_progress" and (args.checkpoint is None or not args.checkpoint.is_file()):
        parser.error("GRPO requires the frozen --checkpoint")
    if args.method == "physics" and args.checkpoint is not None:
        parser.error("Physics runs without a checkpoint")
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    output = (args.output or STUDY / "native" / args.method).resolve()

    manifest, forcing = seasonal.source_inputs(MANIFEST)
    registered = json.loads((SEASONAL / "independent_seasonal_inputs.json").read_text())
    if seasonal.digest(MANIFEST) != registered["manifests"][MANIFEST.name]["sha256"]:
        raise ValueError("The seasonal manifest differs from the registered evaluation roster.")
    if any(
        checksum != registered["blocks"][block]["sources"][field]["normalized_sha256"]
        for block, fields in forcing.items() for field, checksum in fields.items()
    ):
        raise ValueError("Seasonal forcing differs from the registered archive.")
    scenarios = [spec for spec in manifest["scenarios"] if spec["split"] == "test"]
    if manifest["roster"] != "expanded" or len(scenarios) != 12 or len({row["id"] for row in scenarios}) != 12:
        raise ValueError("Expected the full locked roster of 12 test cases.")
    provenance = seasonal.provenance(MANIFEST, forcing)
    provenance["source_sha256"]["grpo_progress_native/reproduce/run_native_comparison.py"] = seasonal.digest(__file__)
    provenance["source_sha256"]["world/learned_mark_policy.py"] = seasonal.digest(V2 / "world/learned_mark_policy.py")
    support = STUDY.with_name("learned_safe_mark_policy") / "learned_mark_policy.json"
    output.mkdir(parents=True, exist_ok=False)
    (output / "telemetry").mkdir()
    receipt = {
        **provenance, "method": args.method, "scenarios": [s["id"] for s in scenarios],
        "workers": args.workers, "platform": platform.platform(), "python": platform.python_version(),
        "package_versions": {pkg: version(pkg) for pkg in ("numpy", "scipy", "xarray", "casadi", "netCDF4", "pyyaml")},
        "checkpoint": str(checkpoint.relative_to(V2)) if checkpoint and checkpoint.is_relative_to(V2)
        else str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": seasonal.digest(checkpoint) if checkpoint else None,
        "adapter_sha256": seasonal.digest(Path(__file__).with_name("native_grpo.py")) if checkpoint else None,
        "legacy_support_sha256": seasonal.digest(support) if checkpoint else None,
        "residual_strength": STRENGTH if checkpoint else 0.,
    }
    seasonal.save_json(output / "inputs.json", receipt)
    init_args = (str(output), args.method, str(checkpoint) if checkpoint else None)
    rows = {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=init_worker, initargs=init_args) as pool:
        jobs = {pool.submit(execute, spec): spec["id"] for spec in scenarios}
        for job in as_completed(jobs):
            case = jobs[job]
            try:
                row = job.result()
            except Exception as error:
                raise RuntimeError(f"Native run failed in {args.method}/{case}.") from error
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
