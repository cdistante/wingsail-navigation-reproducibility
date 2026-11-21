"""Run the fixed Brindisi ridge benchmark with independent recovery interventions."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import gzip
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np


STUDY = Path(__file__).resolve().parents[1]
REFERENCE = STUDY.with_name("eta_ridge_closed_loop")
sys.path.insert(0, str(REFERENCE / "reproduce"))

import run_ridge_loop as reference
from ablation_ranker import select
from causal_weather import CausalEpisodeResult
from escape_maneuvers import EscapeNativeEngine, MAX_TRIMS, MIN_IMPROVEMENT_M, SWITCH_TIMES_S
from staged_departure import choose_departure


baseline = reference.baseline
seasonal = baseline.seasonal
METHODS = ("escape", "departure")


def worker_sources():
    return {**reference.worker_sources(), **{
        str(path.relative_to(STUDY.parent)): seasonal.digest(path)
        for path in (Path(__file__).resolve(), Path(__file__).with_name("staged_departure.py"),
                     Path(__file__).with_name("escape_maneuvers.py"))
    }}


def verified_protocol():
    manifest, inherited, checkpoint = reference.verified_protocol()
    directory = REFERENCE / "native" / "explicit"
    paths = {key: directory / key for key in ("inputs.json", "run.json", "episodes.csv")}
    recorded = json.loads(paths["inputs.json"].read_text())
    run = json.loads(paths["run.json"].read_text())
    sources = {key: seasonal.digest(path) for key, path in paths.items()}
    if (any(recorded[key] != value for key, value in inherited.items())
            or sources["inputs.json"] != run["inputs_sha256"]
            or sources["episodes.csv"] != run["episodes_sha256"]
            or len(run["trace_sha256es"]) != 12
            or any(seasonal.digest(directory / name) != check for name, check in run["trace_sha256es"].items())):
        raise ValueError("The archived compact ridge benchmark differs from its locked inputs.")
    protocol = {
        "reference_protocol": inherited, "frozen_ridge": sources,
        "worker_sources_sha256": worker_sources(),
        "test_ids": inherited["test_ids"], "model_sha256": inherited["checkpoint_sha256"],
        "escape_design": {"switch_times_s": list(SWITCH_TIMES_S), "trim_sample": MAX_TRIMS,
                          "minimum_forecast_improvement_m": MIN_IMPROVEMENT_M},
        "departure_design": {"protected_staging": True, "same_requested_ais_clock": True,
                             "wait_charged_to_mission_cap": True},
    }
    return manifest, protocol, checkpoint


def scenario_parts(spec, method="departure"):
    engine, state, goals, requested, cap = baseline.scenario_parts(spec)
    if method == "escape":
        engine = EscapeNativeEngine(engine.environment, config=engine.config, dynamics=engine.dynamics)
    return engine, state, goals, requested, cap


def init_worker(checkpoint, protocol):
    if worker_sources() != protocol["worker_sources_sha256"]:
        raise RuntimeError("Recovery worker sources differ from the pinned parent.")
    reference.init_worker(checkpoint, protocol["model_sha256"],
                          protocol["reference_protocol"]["worker_sources_sha256"])


def mission(job):
    spec, method, directory = job
    directory = Path(directory)
    engine, state, goals, requested, cap = scenario_parts(spec, method)
    start = requested
    staging = None
    wait = 0.
    if method == "departure":
        departure = choose_departure(engine, state, goals, requested, protected_staging=True)
        staging = asdict(departure)
        start = departure.launch_utc
        wait = departure.wait_seconds
    sailing_cap = cap - wait
    evaluations = []

    def strategy(decision):
        result = select("explicit", baseline.MODELS["explicit"], decision.features, baseline.forecasts(decision),
                        decision.allowed, decision.physics_index)
        evaluations.append({
            "elapsed_s": decision.elapsed_s, "utc": decision.when, "goal": decision.goal.name,
            "status": result.status, "eligible_candidates": result.scored_count,
            "physics_index": decision.physics_index, "requested_index": result.index,
            "corrected_gain_s": result.predicted_gain_s,
        })
        return result.index

    if start is None:
        outcome = CausalEpisodeResult(False, "staging_timeout", 0., state)
    else:
        # Protected waiting changes only launch UTC. The shared environment
        # still timestamps delivered reports from the original request.
        outcome = engine.run(state, goals, start, strategy=strategy, cap_seconds=sailing_cap, trace=True)
    expected = spec["expected_captures"]
    captures = outcome.captures
    if ([capture["name"] for capture in captures] != [entry["waypoint"] for entry in expected[:len(captures)]]
            or any(capture["distance_m"] > spec["capture_radius_m"] + 1e-4 for capture in captures)):
        raise ValueError(f"Invalid waypoint order or capture geometry for {spec['id']} / {method}.")
    applied_by_time = {row["elapsed_s"]: row["choice_id"] for row in outcome.controls}
    decisions = {row["elapsed_s"]: row for row in outcome.decisions}
    gate_count = 0
    for row in evaluations:
        logged = decisions[row["elapsed_s"]]
        changed = row["requested_index"] != row["physics_index"]
        gate_count += changed
        row["learned_applied"] = (changed and logged["selected_index"] == row["requested_index"]
                                  and applied_by_time.get(row["elapsed_s"]) == logged["choice_id"])
    path = directory / "telemetry" / f"{spec['id']}__{method}.json.gz"
    staged = path.with_suffix(".gz.part")
    escape_events = getattr(engine, "escape_events", [])
    payload = {
        "states": outcome.trace, "controls": seasonal.finite_records(outcome.controls),
        "control_attempts": seasonal.finite_records(outcome.control_attempts), "decisions": outcome.decisions,
        "captures": captures, "strategy_evaluations": evaluations, "anticipation_events": engine.anticipation_events,
        "escape_events": escape_events, "departure": staging, "requested_utc": requested, "launch_utc": start,
        "departure_wait_s": wait,
    }
    with gzip.open(staged, "wt", encoding="utf-8", compresslevel=4) as stream:
        json.dump(payload, stream, default=seasonal.json_value, separators=(",", ":"), allow_nan=False)
    staged.replace(path)
    timings = [row["complete_control_seconds"] for row in outcome.control_attempts]
    checks = [] if staging is None else staging["checks"]
    check_times = [row["wall_seconds"] for row in checks]
    minimum_guard = min((row["guarded_report_minimum_m"] for row in outcome.controls), default=np.inf)
    elapsed = captures[-1]["elapsed_s"] if outcome.success else outcome.elapsed_s
    return {
        "scenario": spec["id"], "method": method, "route": spec["route_gpx"], "family": spec["route_family"],
        "weather_block": spec["archive_block"], "traffic": spec["traffic_profile"],
        "departure_utc": spec["departure_utc"], "launch_utc": str(start) + "Z" if start is not None else None,
        "departure_wait_s": wait, "request_elapsed_s": wait + elapsed, "request_cap_s": cap,
        "departure_reason": None if staging is None else staging["reason"],
        "departure_checks": len(checks), "max_departure_check_s": max(check_times, default=0.),
        "departure_overruns_5s": sum(value >= 5. for value in check_times),
        "status": outcome.status, "completed": outcome.success, "duration_s": elapsed, "cap_s": sailing_cap,
        "expected_captures": len(goals), "capture_count": len(captures),
        "captures": [{**capture, **expected[i]} for i, capture in enumerate(captures)],
        "controls_applied": len(outcome.controls),
        "min_report_margin_m": seasonal.measured_margin(outcome.report_minimum_m),
        "min_truth_margin_m": seasonal.measured_margin(outcome.truth_minimum_m),
        "min_applied_guard_m": seasonal.measured_margin(minimum_guard),
        "maximum_report_age_s": max((row["max_report_age_s"] for row in outcome.controls), default=0.),
        "max_lost_epochs": max((row["dropped_report_epochs"] for row in outcome.controls), default=0),
        "strategy_calls": len(evaluations), "gate_crossings": gate_count,
        "scored_decisions": sum(row["corrected_gain_s"] is not None for row in evaluations),
        "learned_changes": sum(row["learned_applied"] for row in evaluations),
        "largest_predicted_gain_s": max((row["corrected_gain_s"] for row in evaluations
                                         if row["corrected_gain_s"] is not None), default=None),
        "maximum_decision_s": max(timings, default=0.), "overruns_5s": sum(t >= 5. for t in timings),
        "anticipation_events": len(engine.anticipation_events), "escape_searches": len(escape_events),
        "escape_selected": sum(event["selected"] for event in escape_events),
        "trajectory": str(path.relative_to(directory)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STUDY / "native")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--method", choices=(*METHODS, "both"), default="both")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Choose at least one worker.")
    manifest, protocol, checkpoint = verified_protocol()
    specs = [spec for spec in manifest["scenarios"] if spec["split"] == "test"]
    ids = [spec["id"] for spec in specs]
    if ids != protocol["test_ids"] or len(ids) != 12:
        raise ValueError("Expected the twelve fixed October cases.")
    methods = METHODS if args.method == "both" else (args.method,)
    out = args.output.resolve()
    workers = min(args.workers, len(specs) * len(methods))
    for method in methods:
        base = out / method
        (base / "telemetry").mkdir(parents=True, exist_ok=False)
        seasonal.save_json(base / "inputs.json", {
            **protocol, "method": method, "scenarios": ids, "workers": workers, "probe": False,
        })
    episodes = {method: {} for method in methods}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=init_worker, initargs=(str(checkpoint), protocol)) as pool:
        futures = [pool.submit(mission, (spec, method, str(out / method))) for method in methods for spec in specs]
        for future in as_completed(futures):
            row = future.result()
            episodes[row["method"]][row["scenario"]] = row
            print(f"{row['method']:<9} {row['scenario']} {row['capture_count']}/{row['expected_captures']} "
                  f"{row['status']} sail={row['duration_s']:.2f}s wait={row['departure_wait_s']:g}s "
                  f"escape={row['escape_selected']}", flush=True)
    if worker_sources() != protocol["worker_sources_sha256"]:
        raise ValueError("Recovery sources changed during the experiment.")
    for method in methods:
        base = out / method
        rows = [episodes[method][scenario] for scenario in ids]
        csv_path = base / "episodes.csv"
        seasonal.save_csv(csv_path, rows)
        seasonal.save_json(base / "run.json", {
            "method": method, "inputs_sha256": seasonal.digest(base / "inputs.json"),
            "episodes_sha256": seasonal.digest(csv_path), "summary": seasonal.summary(rows),
            "trace_sha256es": {row["trajectory"]: seasonal.digest(base / row["trajectory"]) for row in rows},
        })


if __name__ == "__main__":
    main()
