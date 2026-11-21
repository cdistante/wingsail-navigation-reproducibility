"""Replay protected Brindisi launches with causal trend wind in the tack schedule."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np


STUDY = Path(__file__).resolve().parents[1]
RECOVERY = STUDY.with_name("brindisi_escape_departure")
FORCING = STUDY.with_name("stage3_forcing_offline")
sys.path.insert(0, str(RECOVERY / "reproduce"))

import run_recovery as recovery
from ablation_ranker import select
from causal_weather import CausalEpisodeResult
from hybrid_planner import HybridWindEngine
from hybrid_wind import WindTrendProvider
from world._geo import normalise_time


METHOD = "hybrid"
baseline = recovery.baseline
seasonal = recovery.seasonal
PERSISTENCE = RECOVERY / "native" / "departure"
SETTINGS = FORCING / "selection_receipt.json"


def worker_sources():
    local = (
        Path(__file__).resolve(), Path(__file__).with_name("hybrid_planner.py"),
        Path(__file__).with_name("hybrid_wind.py"), SETTINGS,
    )
    return {**recovery.worker_sources(),
            **{str(path.relative_to(STUDY.parent)): seasonal.digest(path) for path in local}}


def verified_protocol():
    manifest, inherited, checkpoint = recovery.verified_protocol()
    paths = {name: PERSISTENCE / name for name in ("inputs.json", "run.json", "episodes.csv")}
    prior = json.loads(paths["inputs.json"].read_text())
    run = json.loads(paths["run.json"].read_text())
    hashes = {name: seasonal.digest(path) for name, path in paths.items()}
    if (any(prior[key] != value for key, value in inherited.items())
            or hashes["inputs.json"] != run["inputs_sha256"]
            or hashes["episodes.csv"] != run["episodes_sha256"]
            or prior["method"] != "departure" or prior["scenarios"] != inherited["test_ids"]
            or len(run["trace_sha256es"]) != 12):
        raise ValueError("The prior protected-departure comparator changed.")
    for relative, expected in run["trace_sha256es"].items():
        if seasonal.digest(PERSISTENCE / relative) != expected:
            raise ValueError(f"The archived protected departure trace changed: {relative}.")
    clearance = RECOVERY / "independent_clearance.json"
    assessed = json.loads(clearance.read_text())
    original = [x for x in assessed["cases"] if x["method"] == "departure"]
    if assessed["status"] != "verified" or len(original) != 12:
        raise ValueError("The independent protected-departure clearance record is missing cases.")
    settings = json.loads(SETTINGS.read_text())
    trend = settings["settings"]
    if (settings["manifest_sha256"] != inherited["reference_protocol"]["split_manifest_sha256"]
            or settings["prediction_horizon_s"] != 3600 or trend["trend_multiplier"] != .5):
        raise ValueError("Expected the trend configuration selected on the frozen April split.")
    return manifest, {
        "recovery_protocol": inherited, "worker_sources_sha256": worker_sources(),
        "frozen_departure": hashes, "frozen_departure_audit_sha256": seasonal.digest(clearance),
        "model_sha256": inherited["model_sha256"], "test_ids": inherited["test_ids"],
        "forcing_selection_sha256": seasonal.digest(SETTINGS),
        "wind_trend_multiplier": trend["trend_multiplier"], "wind_delta_cap_mps": trend["change_caps"][0],
        "shared_protected_launches": True, "future_current": "observed persistence",
        "planner_first_slot": "observed weather", "applied_weather": "unchanged interpolated simulation plant",
    }, checkpoint


def scenario_parts(spec, blend=1.):
    old, state, goals, requested, cap = baseline.scenario_parts(spec)
    if blend == 0.:
        return old, state, goals, requested, cap
    provider = WindTrendProvider.from_receipt(SETTINGS, blend=blend)
    engine = HybridWindEngine(old.environment, config=old.config, dynamics=old.dynamics, wind_provider=provider)
    return engine, state, goals, requested, cap


def init_worker(checkpoint, protocol):
    if worker_sources() != protocol["worker_sources_sha256"]:
        raise RuntimeError("The hybrid worker sources differ from their launch receipt.")
    recovery.init_worker(checkpoint, protocol["recovery_protocol"])


def mission(job):
    spec, out = job
    out = Path(out)
    engine, state, goals, requested, cap = scenario_parts(spec)
    original = PERSISTENCE / "telemetry" / f"{spec['id']}__departure.json.gz"
    with gzip.open(original, "rt", encoding="utf-8") as stream:
        prior = json.load(stream)
    staging = prior["departure"]
    launch = None if prior["launch_utc"] is None else normalise_time(prior["launch_utc"])
    wait = float(prior["departure_wait_s"])
    if (normalise_time(prior["requested_utc"]) != requested or staging["wait_seconds"] != wait
            or (launch is not None and launch != requested + np.timedelta64(round(wait * 1e9), "ns"))):
        raise ValueError("The frozen protected launch does not agree with the fixture clock.")
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

    if launch is None:
        outcome = CausalEpisodeResult(False, "staging_timeout", 0., state)
    else:
        outcome = engine.run(state, goals, launch, strategy=strategy, cap_seconds=sailing_cap, trace=True)
    expected = spec["expected_captures"]
    captures = outcome.captures
    if ([entry["name"] for entry in captures] != [item["waypoint"] for item in expected[:len(captures)]]
            or any(entry["distance_m"] > spec["capture_radius_m"] + 1e-4 for entry in captures)):
        raise ValueError(f"Invalid waypoint order or geometry for {spec['id']}.")
    applied_by_time = {row["elapsed_s"]: row["choice_id"] for row in outcome.controls}
    decisions = {row["elapsed_s"]: row for row in outcome.decisions}
    switches = 0
    for row in evaluations:
        stored = decisions[row["elapsed_s"]]
        changed = row["requested_index"] != row["physics_index"]
        switches += changed
        row["learned_applied"] = (changed and stored["selected_index"] == row["requested_index"]
                                  and applied_by_time.get(row["elapsed_s"]) == stored["choice_id"])

    path = out / "telemetry" / f"{spec['id']}__{METHOD}.json.gz"
    staged = path.with_suffix(".gz.part")
    weather = engine.wind_forecast_events
    payload = {
        "states": outcome.trace, "controls": seasonal.finite_records(outcome.controls),
        "control_attempts": seasonal.finite_records(outcome.control_attempts), "decisions": outcome.decisions,
        "captures": captures, "strategy_evaluations": evaluations, "anticipation_events": engine.anticipation_events,
        "escape_events": [], "departure": staging, "requested_utc": prior["requested_utc"],
        "launch_utc": prior["launch_utc"], "departure_wait_s": wait, "wind_forecast_events": weather,
    }
    with gzip.open(staged, "wt", encoding="utf-8", compresslevel=4) as stream:
        json.dump(payload, stream, default=seasonal.json_value, separators=(",", ":"), allow_nan=False)
    staged.replace(path)
    timings = [row["complete_control_seconds"] for row in outcome.control_attempts]
    waits = [row["wall_seconds"] for row in staging["checks"]]
    minimum_guard = min((row["guarded_report_minimum_m"] for row in outcome.controls), default=np.inf)
    elapsed = captures[-1]["elapsed_s"] if outcome.success else outcome.elapsed_s
    return {
        "scenario": spec["id"], "method": METHOD, "route": spec["route_gpx"], "family": spec["route_family"],
        "weather_block": spec["archive_block"], "traffic": spec["traffic_profile"],
        "departure_utc": spec["departure_utc"], "launch_utc": prior["launch_utc"],
        "departure_wait_s": wait, "request_elapsed_s": wait + elapsed, "request_cap_s": cap,
        "departure_reason": staging["reason"], "departure_checks": len(waits),
        "max_departure_check_s": max(waits, default=0.), "departure_overruns_5s": sum(t >= 5. for t in waits),
        "status": outcome.status, "completed": outcome.success, "duration_s": elapsed, "cap_s": sailing_cap,
        "expected_captures": len(goals), "capture_count": len(captures),
        "captures": [{**capture, **expected[i]} for i, capture in enumerate(captures)],
        "controls_applied": len(outcome.controls),
        "min_report_margin_m": seasonal.measured_margin(outcome.report_minimum_m),
        "min_truth_margin_m": seasonal.measured_margin(outcome.truth_minimum_m),
        "min_applied_guard_m": seasonal.measured_margin(minimum_guard),
        "maximum_report_age_s": max((row["max_report_age_s"] for row in outcome.controls), default=0.),
        "max_lost_epochs": max((row["dropped_report_epochs"] for row in outcome.controls), default=0),
        "strategy_calls": len(evaluations), "gate_crossings": switches,
        "scored_decisions": sum(row["corrected_gain_s"] is not None for row in evaluations),
        "learned_changes": sum(row["learned_applied"] for row in evaluations),
        "largest_predicted_gain_s": max((row["corrected_gain_s"] for row in evaluations
                                         if row["corrected_gain_s"] is not None), default=None),
        "maximum_decision_s": max(timings, default=0.), "overruns_5s": sum(t >= 5. for t in timings),
        "anticipation_events": len(engine.anticipation_events), "escape_searches": 0, "escape_selected": 0,
        "wind_forecast_events": len(weather), "trajectory": str(path.relative_to(out)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STUDY / "native")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Use at least one worker.")
    manifest, protocol, checkpoint = verified_protocol()
    specs = [row for row in manifest["scenarios"] if row["split"] == "test"]
    ids = [row["id"] for row in specs]
    if ids != protocol["test_ids"] or len(ids) != 12:
        raise ValueError("Expected the twelve registered October cases.")
    out = args.output.resolve() / METHOD
    (out / "telemetry").mkdir(parents=True, exist_ok=False)
    workers = min(args.workers, len(specs))
    inputs = out / "inputs.json"
    seasonal.save_json(inputs, {**protocol, "scenarios": ids, "method": METHOD, "probe": False, "workers": workers})
    outcomes = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=init_worker, initargs=(str(checkpoint), protocol)) as pool:
        futures = [pool.submit(mission, (spec, str(out))) for spec in specs]
        for future in as_completed(futures):
            row = future.result()
            outcomes[row["scenario"]] = row
            print(f"{row['scenario']} {row['capture_count']}/{row['expected_captures']} {row['status']} "
                  f"sail={row['duration_s']:.2f}s wait={row['departure_wait_s']:g}s "
                  f"forecast={row['wind_forecast_events']}", flush=True)
    if worker_sources() != protocol["worker_sources_sha256"]:
        raise ValueError("The hybrid controller sources changed during execution.")
    ordered = [outcomes[name] for name in ids]
    episodes = out / "episodes.csv"
    seasonal.save_csv(episodes, ordered)
    seasonal.save_json(out / "run.json", {
        "method": METHOD, "inputs_sha256": seasonal.digest(inputs),
        "episodes_sha256": seasonal.digest(episodes), "summary": seasonal.summary(ordered),
        "trace_sha256es": {row["trajectory"]: seasonal.digest(out / row["trajectory"]) for row in ordered},
    })


if __name__ == "__main__":
    main()
