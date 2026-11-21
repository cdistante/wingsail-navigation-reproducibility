"""Audit swept clearance and causal decisions in the shorter-detour AIS retest."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from math import isclose
import multiprocessing
from pathlib import Path
import sys

STUDY = Path(__file__).resolve().parents[1]
EARLIER = STUDY.with_name("grpo_progress_anticipatory_ais")
sys.path.insert(0, str(EARLIER / "reproduce"))

from audit_anticipatory_clearance import (
    MANIFEST, METHODS, PREVIOUS, ROBUST, STATE_KEYS, TARGET_SCENARIO, DecisionEnvironment,
    audit_episode, clearance_flag, first_control_change, historical_rejection,
    outcome, read_trace, require, seasonal, source_jobs,
)
from world._geo import normalise_time
from world.boat import BoatState


def replay_first_decision(spec, trace):
    """Repeat the first logged forecast decision from its exact observed state."""
    events = trace["anticipation_events"]
    if not events:
        return None

    from native_efficient_guard import EfficientNativeEngine

    event = events[0]
    when = normalise_time(event["utc"])
    base, state, goals, start, _ = seasonal.scenario_parts(spec)
    if when > start:
        before, = (row for row in trace["states"] if normalise_time(row["utc"]) == when)
        state = BoatState(**{key: before[key] for key in STATE_KEYS})
    goal, = (mark for mark in goals if mark.name == event["goal"])
    environment = DecisionEnvironment(base.environment, when)
    engine = EfficientNativeEngine(environment, config=base.config, dynamics=base.dynamics)
    decision = engine.decision(state, goal, when, event["elapsed_s"])
    repeated, = engine.anticipation_events
    require(repeated.keys() == event.keys() and all(
        isclose(value, repeated[key], rel_tol=0., abs_tol=1e-7)
        if type(value) is float else value == repeated[key]
        for key, value in event.items()
    ), f"{spec['id']}: logged forecast could not be repeated from information available then.")
    require(decision.choices[decision.physics_index].id == event["selected_choice"],
            f"{spec['id']}: causal replay returned a different native choice.")
    return {"event": event, "matched_causal_replay": True,
            "environment_requests": sorted(set(environment.requests)),
            "synthetic_truth_unavailable": True, "latest_allowed_environment_utc": event["utc"]}


def audit_efficient_episode(job):
    geometry = audit_episode(job)
    spec, _, path, _ = job
    geometry["first_forecast_causal_replay"] = replay_first_decision(spec, read_trace(path))
    return geometry


def target_comparison(scenarios, jobs, indexed):
    historical = {
        version: {row["method"]: (row, path, digest)
                  for _, row, path, digest in source_jobs(root / "native", scenarios)
                  if row["scenario"] == TARGET_SCENARIO}
        for version, root in (("original", PREVIOUS), ("robust_only", ROBUST), ("earlier_ais", EARLIER))
    }
    historical["efficient_ais"] = {row["method"]: (row, path, digest)
                                   for _, row, path, digest in jobs
                                   if row["scenario"] == TARGET_SCENARIO}
    comparison = {}
    for method in METHODS:
        traces, outcomes = {}, {}
        for version, arms in historical.items():
            row, path, digest = arms[method]
            require(seasonal.digest(path) == digest, f"Changed target trace: {path}.")
            traces[version] = read_trace(path)
            outcomes[version] = outcome(row, traces[version])
        earlier, efficient = outcomes["earlier_ais"], outcomes["efficient_ais"]
        audit = indexed[method, TARGET_SCENARIO]
        saved = earlier["elapsed_s"] - efficient["elapsed_s"] if earlier["completed"] and efficient["completed"] else None
        comparison[method] = {
            "outcomes": outcomes,
            "seconds_saved_vs_earlier_ais_on_joint_finish": saved,
            "first_actuator_difference_from_earlier_ais": first_control_change(
                traces["earlier_ais"]["controls"], traces["efficient_ais"]["controls"],
            ),
            "completion_retained_with_full_clearance": efficient["completed"]
                and clearance_flag(audit) == "clear",
            "independent_swept_minima": audit["minimum_clearance"],
            "first_forecast_causal_replay": audit["first_forecast_causal_replay"],
        }
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, default=STUDY / "native")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, default=STUDY / "independent_clearance.json")
    args = parser.parse_args()
    require(args.workers > 0, "At least one audit worker is required.")
    manifest = json.loads(MANIFEST.read_text())
    scenarios = {spec["id"]: spec for spec in manifest["scenarios"] if spec["split"] == "test"}
    require(len(scenarios) == 12, "The full locked native test roster must have twelve cases.")

    guard = Path(__file__).with_name("native_efficient_guard.py")
    guard_relative = str(guard.relative_to(STUDY.parents[2]))
    guard_digest = seasonal.digest(guard)
    manifest_digest = seasonal.digest(MANIFEST)
    for method in METHODS:
        directory = args.native / method
        inputs_path = directory / "inputs.json"
        inputs = json.loads(inputs_path.read_text())
        receipts = json.loads((directory / "run.json").read_text())
        require(receipts["inputs_sha256"] == seasonal.digest(inputs_path)
                and inputs["source_sha256"][guard_relative] == guard_digest
                and inputs["split_manifest_sha256"] == manifest_digest,
                f"{method}: input, guard or manifest digest does not match the run.")

    seasonal.init(MANIFEST, None, False)
    historical = historical_rejection(scenarios)
    jobs = list(source_jobs(args.native, scenarios))
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=seasonal.init, initargs=(MANIFEST, None, False)) as pool:
        results = list(pool.map(audit_efficient_episode, jobs, chunksize=1))
    indexed = {(record["method"], record["scenario"]): record for record in results}
    unsafe = [{key: record[key] for key in ("scenario", "method", "nonpositive_control_intervals")}
              for record in results if any(record["nonpositive_control_intervals"].values())]
    seasonal.save_json(args.output, {
        "status": "failed_clearance" if unsafe else "verified",
        "scope": "Independent held-forcing Fossen re-integration at 0.05 s, analytical targets, "
                 "derived report times, both original full envelopes and the expanded report guard. "
                 "First logged forecast decisions were replayed without target truth or later observations.",
        "manifest_sha256": manifest_digest,
        "auditor_sha256": seasonal.digest(__file__),
        "geometry_and_receiver_auditor_sha256": seasonal.digest(
            EARLIER / "reproduce" / "audit_anticipatory_clearance.py",
        ),
        "guard_sha256": guard_digest,
        "cases_per_arm": len(scenarios),
        "applied_controls": sum(record["applied_controls"] for record in results),
        "integration_substeps": sum(record["integration_substeps"] for record in results),
        "first_forecast_decisions_replayed": sum(record["first_forecast_causal_replay"] is not None
                                                for record in results),
        "historical_breach_replay": historical,
        "brindisi_003_overtake": target_comparison(scenarios, jobs, indexed),
        "paired_clearance_flags": {
            name: {method: clearance_flag(indexed[method, name]) for method in METHODS}
            for name in scenarios
        },
        "unsafe_applied_cases": unsafe,
        "cases": results,
    })
    require(not unsafe, f"Clearance or guard violations found. See {args.output}.")
    print(f"Verified {len(results)} efficient AIS traces and the full swept envelopes. {args.output}", flush=True)


if __name__ == "__main__":
    main()
