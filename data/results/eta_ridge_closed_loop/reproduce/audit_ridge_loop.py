"""Verify the matched Stage 1 ridge benchmark against frozen causal physics."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from math import isclose
import multiprocessing
from pathlib import Path
import sys

import numpy as np


STUDY = Path(__file__).resolve().parents[1]
CAUSAL = STUDY.with_name("temporal_embedding_native")
STAGE1 = STUDY.with_name("eta_stage1_expanded")
sys.path.insert(0, str(CAUSAL / "reproduce"))
sys.path.insert(0, str(STAGE1 / "reproduce"))

import ablation_report as audit
import run_temporal_ablation as baseline
from compact_features import feature_rows
from run_ridge_loop import verified_protocol


PHYSICS_RUN_SHA256 = "72a1789ccbbb43a6c78509c6a65bee165bbca7abd98f4740230ec5b2155e8ac1"
METHODS = ("physics", "explicit")
LABELS = {"physics": "Archived causal physics", "explicit": "Stage 1 compact explicit ridge"}
require = audit.require
digest = audit.seasonal.digest
CHECKPOINT = None


def verify_arm(arm, directory, scenarios, *, pinned_physics=False):
    inputs_path, run_path, episodes_path = (directory / name for name in ("inputs.json", "run.json", "episodes.csv"))
    run_sha = digest(run_path)
    if pinned_physics:
        require(run_sha == PHYSICS_RUN_SHA256, "The archived causal physics receipt changed.")
    inputs, run = audit.load_json(inputs_path), audit.load_json(run_path)
    inputs_sha, episodes_sha = digest(inputs_path), digest(episodes_path)
    require(inputs["method"] == run["method"] == arm and inputs["probe"] is False
            and inputs["scenarios"] == list(scenarios)
            and run["inputs_sha256"] == inputs_sha and run["episodes_sha256"] == episodes_sha,
            f"{arm}: unmatched roster or run receipt.")
    rows = audit.read_rows(episodes_path)
    require([row["scenario"] for row in rows] == list(scenarios), f"{arm}: missing or reordered cases.")
    audit.compare(audit.summarize(rows), run["summary"], arm)
    checksums = run["trace_sha256es"]
    require(len(checksums) == len(scenarios), f"{arm}: incorrect trajectory count.")
    jobs = []
    for row in rows:
        scenario = row["scenario"]
        name = f"telemetry/{scenario}__{arm}.json.gz"
        require(row["method"] == arm and row["trajectory"] == name and name in checksums,
                f"{arm}/{scenario}: unmatched trajectory.")
        jobs.append((scenarios[scenario], row, directory / name, checksums[name]))
    source = {"inputs_sha256": inputs_sha, "episodes_sha256": episodes_sha,
              "run_receipt_sha256": run_sha, "inputs": inputs}
    return inputs, jobs, source


def frozen_jobs(native):
    manifest, protocol, checkpoint = verified_protocol()
    protocol = json.loads(json.dumps(protocol, default=audit.seasonal.json_value))
    scenarios = {spec["id"]: spec for spec in manifest["scenarios"] if spec["split"] == "test"}
    require(len(scenarios) == 12 and all(spec["capture_radius_m"] == 50. for spec in scenarios.values()),
            "Expected exactly twelve October tests at 50 m.")
    original_inputs, physics, original = verify_arm("physics", CAUSAL / "native" / "physics", scenarios,
                                                    pinned_physics=True)
    actual_manifest, causal_protocol = baseline.locked_protocol()
    require(manifest == actual_manifest, "Method manifests disagree.")
    current_causal = json.loads(json.dumps(causal_protocol, default=audit.seasonal.json_value))
    old_causal = {key: original_inputs[key] for key in current_causal}
    reconciliation = audit.reconcile_protocol(old_causal, current_causal, CAUSAL)
    require(original_inputs["model_sha256"] is None, "Frozen physics declares a learned checkpoint.")

    new_inputs, ridge, new = verify_arm("explicit", native / "explicit", scenarios)
    require(all(new_inputs[key] == value for key, value in protocol.items()),
            "Stage 1 ridge receipt differs from the current pinned protocol.")
    require(new_inputs["model_sha256"] == digest(checkpoint), "Stage 1 compact ridge checkpoint changed.")
    return manifest, scenarios, physics + ridge, {"physics": original, "explicit": new}, checkpoint, reconciliation


def init_replay(checkpoint):
    global CHECKPOINT
    baseline.init_worker()
    with np.load(checkpoint, allow_pickle=False) as data:
        scales, coefficients = data["scales"], data["coefficients"]
        require(scales.shape == coefficients.shape == (1, 11) and np.isfinite(scales).all()
                and np.isfinite(coefficients).all() and np.all(scales > 0.), "Invalid compact ridge checkpoint.")
        CHECKPOINT = scales[0], coefficients[0]


def replay_gate(spec, trace):
    """Recompute every ridge gate from the saved state and then-available reports."""
    engine, initial_state, goals, start, _ = baseline.scenario_parts(spec)
    observed = audit.DecisionEnvironment(engine.environment, start)
    replay = audit.CausalEfficientNativeEngine(observed, config=engine.config, dynamics=engine.dynamics)
    states = {row["utc"]: row for row in trace["states"]}
    marks = {item.name: item for item in goals}
    scale, coefficients = CHECKPOINT
    counts = Counter()
    max_gain_error = 0.
    for event in trace["strategy_evaluations"]:
        utc = event["utc"]
        when = audit.normalise_time(utc)
        state = initial_state
        if when != start:
            row = states[utc]
            state = audit.BoatState(**{key: row[key] for key in audit.STATE_KEYS})
        observed.through = when
        observed.requests.clear()
        replay.anticipation_events.clear()
        decision = replay.decision(state, marks[event["goal"]], when, event["elapsed_s"])
        eta = baseline.forecasts(decision)
        incumbent = decision.physics_index
        require(incumbent >= 0 and incumbent == event["physics_index"] and decision.allowed[incumbent],
                f'{spec["id"]}/{utc}: different or inadmissible incumbent.')
        candidates = np.flatnonzero(
            decision.allowed & np.isfinite(eta) & (eta <= eta[incumbent] + baseline.TRUST_REGRET_S)
        ) if np.isfinite(eta[incumbent]) else np.array([], dtype=int)
        selected, gain, scored = incumbent, None, 0
        if len(candidates) >= 2:
            scores = eta[candidates] + np.clip(
                (feature_rows(decision.features[candidates]) / scale) @ coefficients,
                -baseline.RESIDUAL_CAP_S, baseline.RESIDUAL_CAP_S,
            )
            win = int(np.argmin(scores))
            baseline_score = scores[np.flatnonzero(candidates == incumbent)[0]]
            gain = float(baseline_score - scores[win])
            scored = len(candidates)
            if gain > baseline.MINIMUM_MODEL_GAIN_S:
                selected = int(candidates[win])
            error = abs(gain - event["corrected_gain_s"]) if event["corrected_gain_s"] is not None else float("inf")
            max_gain_error = max(max_gain_error, error)
        status = ("override" if selected != incumbent else
                  "below_or_equal_gain_threshold" if gain is not None and gain > 0. else
                  "incumbent_best" if gain is not None else
                  "no_eligible_alternative" if len(candidates) else "incumbent_has_no_forecast_capture")
        require(selected == event["requested_index"] and scored == event["eligible_candidates"]
                and status == event["status"]
                and (event["corrected_gain_s"] is None if gain is None else max_gain_error <= audit.EPS),
                f'{spec["id"]}/{utc}: live legality, Stage 1 ridge score or selected trim changed.')
        counts["strategy_calls"] += 1
        counts["scored_calls"] += int(scored > 0)
        counts["gate_crossings"] += int(selected != incumbent)
    return {**counts, "max_score_error_s": max_gain_error}


def verify_job(arguments):
    manifest, job = arguments
    spec, row, path, _ = job
    check = audit.verify_episode(manifest, job)
    geometry = audit.audit_episode(job)
    if row["method"] == "explicit":
        geometry["ridge_causal_gate_replay"] = replay_gate(spec, audit.read_trace(path))
    return check, geometry


def waypoint_deltas(ridge, physics):
    values = []
    for one, two in zip(ridge["captures_in_order"], physics["captures_in_order"]):
        require((one["name"], one["phase"]) == (two["name"], two["phase"]), "Paired capture order differs.")
        values.append({"waypoint": one["name"], "phase": one["phase"],
                       "ridge_minus_physics_s": one["elapsed_s"] - two["elapsed_s"]})
    return values


def paired_row(scenario, spec, outcome):
    row = {"scenario": scenario, "route": spec["route_gpx"], "departure_utc": spec["departure_utc"],
           "weather_block": spec["archive_block"], "traffic": spec["traffic_profile"],
           "expected_captures": len(spec["expected_captures"]), "capture_radius_m": spec["capture_radius_m"]}
    for arm in METHODS:
        label = "ridge" if arm == "explicit" else arm
        for key, value in outcome[arm].items():
            if key == "captures_in_order":
                row[f"{label}_{key}"] = " ".join(hit["name"] for hit in value)
            elif isinstance(value, dict):
                row.update({f"{label}_{key}_{item}": number for item, number in value.items()})
            else:
                row[f"{label}_{key}"] = value
    row["ridge_minus_physics_s_on_joint_finish"] = outcome["ridge_minus_physics_s_on_joint_finish"]
    row["shared_waypoint_deltas"] = outcome["shared_waypoint_deltas"]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, default=STUDY / "native")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    require(args.workers > 0, "Use a positive replay worker count.")
    manifest, scenarios, jobs, sources, checkpoint, reconciliation = frozen_jobs(args.native)
    by_key = {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=init_replay, initargs=(checkpoint,)) as pool:
        for check, geometry in pool.map(verify_job, ((manifest, job) for job in jobs), chunksize=1):
            key = geometry["method"], geometry["scenario"]
            by_key[key] = check, geometry
            print(f"Audited {key[0]} {key[1]} ({geometry['integration_substeps']} knots)", flush=True)
    runs = {arm: [row for _, row, _, _ in jobs if row["method"] == arm] for arm in METHODS}
    all_geometries = [geometry for _, geometry in by_key.values()]
    violations = [
        {"method": geom["method"], "scenario": geom["scenario"],
         "nonpositive_control_intervals": geom["nonpositive_control_intervals"],
         "applied_chart_violations": geom["applied_chart_violations"]}
        for geom in all_geometries
        if any(geom["nonpositive_control_intervals"].values()) or geom["applied_chart_violations"]
    ]
    status = "clearance_violation" if violations else "verified"
    scope = {"status": status, "cases_per_arm": len(scenarios), "capture_radius_m": 50.,
             "manifest_sha256": digest(baseline.MANIFEST), "auditor_sha256": digest(__file__),
             "geometry_auditor_sha256": digest(audit.__file__), "model_sha256": digest(checkpoint),
             "run_sources": sources, "physics_provenance_reconciliation": reconciliation}
    summaries = {}
    cases = {scenario: {} for scenario in scenarios}
    for arm in METHODS:
        arm_checks, gains, statuses = [], {}, Counter()
        chart_violations = 0
        for row in runs[arm]:
            scenario = row["scenario"]
            check, geometry = by_key[arm, scenario]
            arm_checks.append(check)
            statuses.update(check["strategy_status_counts"])
            item = audit.case_summary(row, check)
            item.update({"clearance_status": audit.clearance_flag(geometry),
                         "guard_margin_m": (geometry["minimum_clearance"]["guard"] or {}).get("margin_m"),
                         "applied_chart_violations": geometry["applied_chart_violations"]})
            chart_violations += geometry["applied_chart_violations"]
            cases[scenario][arm] = item
        for field in ("scored_gains_s", "gate_crossing_gains_s", "applied_override_gains_s"):
            gains[f"predicted_{field}"] = audit.gain_summary(
                [gain for check in arm_checks for gain in check[field]]
            )
        timing = audit.timing_summary([value for check in arm_checks for value in check["control_timings_s"]])
        summaries[arm] = {
            **audit.safety_summary(runs[arm]), **audit.summarize_clearance(all_geometries, arm), **gains,
            "strategy_status_counts": dict(statuses), "control_timing": timing,
            "five_second_budget_pass": timing["attempts_at_or_over_5s"] == 0,
            **{field: sum(case[arm][field] for case in cases.values())
               for field in ("strategy_calls", "scored_decisions", "gate_crossings", "applied_overrides")},
            "applied_chart_violations": chart_violations,
        }
    for outcome in cases.values():
        ridge, physics = outcome["explicit"], outcome["physics"]
        outcome["ridge_minus_physics_s_on_joint_finish"] = audit.joint_delta(outcome, "explicit", "physics")
        outcome["shared_waypoint_deltas"] = waypoint_deltas(ridge, physics)
    ridge_replays = [geom["ridge_causal_gate_replay"] for geom in all_geometries if geom["method"] == "explicit"]
    audit.seasonal.save_json(args.native.parent / "independent_clearance.json", {
        **scope, "summary": {arm: audit.summarize_clearance(all_geometries, arm) for arm in METHODS},
        "control_intervals_replayed": sum(geom["applied_controls"] for geom in all_geometries),
        "integration_substeps_replayed": sum(geom["integration_substeps"] for geom in all_geometries),
        "first_ais_previews_replayed": sum(geom["first_forecast_causal_replay"] is not None for geom in all_geometries),
        "ridge_strategy_calls_replayed": sum(replay.get("strategy_calls", 0) for replay in ridge_replays),
        "ridge_scored_calls_replayed": sum(replay.get("scored_calls", 0) for replay in ridge_replays),
        "ridge_max_score_error_s": max(replay["max_score_error_s"] for replay in ridge_replays),
        "unsafe_applied_cases": violations, "cases": all_geometries,
    })
    audit.seasonal.save_csv(args.native / "paired_episodes.csv",
                            [paired_row(scenario, spec, cases[scenario]) for scenario, spec in scenarios.items()])
    audit.seasonal.save_json(args.native / "independent_comparison.json", {
        **scope, "arm_labels": LABELS, "summary": summaries,
        "pairwise": {"ridge_vs_physics": audit.pair_summary(cases, "explicit", "physics")},
        "sailing_time_comparison": "Ridge minus physics. Mission duration compared only on joint completions.",
        "timing_scope": "Control and screening wall time. The physics measurements come from its archived run.",
        "cases": cases,
    })
    print(f"Wrote {len(scenarios)} matched cases and independently replayed {len(all_geometries)} runs.", flush=True)
    if violations:
        print(f"Detected {len(violations)} cases with applied AIS or chart violations.", flush=True)


if __name__ == "__main__":
    main()
