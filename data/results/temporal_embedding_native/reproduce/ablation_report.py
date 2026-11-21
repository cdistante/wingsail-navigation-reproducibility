"""Verify the matched causal-weather ablation and replay its swept AIS clearance."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
import gzip
from hashlib import sha256
import json
from math import isclose, isfinite
import multiprocessing
from pathlib import Path
from statistics import mean, median
import sys

import numpy as np

from run_temporal_ablation import METHODS, MANIFEST, SPLITS, init_worker, locked_protocol, scenario_parts, seasonal
from causal_weather import CausalEfficientNativeEngine
from ablation_ranker import MINIMUM_MODEL_GAIN_S


STUDY = Path(__file__).resolve().parents[1]
GROUP12 = STUDY.with_name("grpo_group12_native")
sys.path.insert(0, str(GROUP12 / "reproduce"))

from audit_group12_clearance import clearance_flag, summarize as summarize_clearance
from audit_anticipatory_clearance import DecisionEnvironment, normalise_time, read_trace
from audit_stale_clearance import observation_geometry
from recheck_clearance import (
    ENDPOINT_TOLERANCE_M, STATE_KEYS, STATE_TOLERANCE, BoatState, swept_margin, truth_track,
    verified_value, xy,
)
from verify_native_comparison import safety_summary
from verify_seasonal_heldout import (
    check_episode_metadata, close, compare, completed, delivered_age, load_json, number, read_rows, require,
    summarize, verify_trace,
)
from world.boat import BoatAction


EPS = 1e-7
MARGIN_FIELDS = {
    "report": "report_minimum_m", "truth": "truth_minimum_m", "guard": "guarded_report_minimum_m",
}
PAIRINGS = (
    ("explicit", "physics"),
    ("temporal", "physics"),
    ("snapshot", "physics"),
    ("temporal", "explicit"),
    ("snapshot", "explicit"),
    ("temporal", "snapshot"),
)


def reconcile_protocol(stored, current, run_root):
    """Authenticate the original protocol and the trace-only import fix."""
    if stored == current:
        return {"status": "exact_source_match"}

    source_name = "causal_weather.py"
    comparable = deepcopy(current)
    launch_hash = stored["study_sources"][source_name]
    active_hash = current["study_sources"][source_name]
    comparable["study_sources"][source_name] = launch_hash
    require(stored == comparable, "Frozen protocol differs beyond the one documented source fix.")

    launch = run_root / "provenance" / "causal_weather_at_launch.py"
    active = Path(__file__).with_name(source_name)
    require(launch.is_file() and seasonal.digest(launch) == launch_hash and seasonal.digest(active) == active_hash,
            "Could not authenticate both versions of the causal-weather source.")
    old, new = launch.read_bytes(), active.read_bytes()
    original_import = b"from dataclasses import dataclass\n"
    corrected_import = b"from dataclasses import asdict, dataclass\n"
    require(old.count(original_import) == 1 and old.replace(original_import, corrected_import, 1) == new,
            "The two causal-weather sources differ beyond the state trace import.")

    syntax = ast.parse(new)
    asdict_uses = [node for node in ast.walk(syntax) if isinstance(node, ast.Name) and node.id == "asdict"]
    trace_writes = [
        child for node in ast.walk(syntax)
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "trace"
        for entry in node.body for child in ast.walk(entry)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == "asdict"
        and len(child.args) == 1 and isinstance(child.args[0], ast.Name) and child.args[0].id == "final"
    ]
    require(len(asdict_uses) == 1 and len(trace_writes) == 1 and trace_writes[0].func is asdict_uses[0],
            "Could not confine the new import to completed-state trace serialization.")
    runner = ast.parse(Path(__file__).with_name("run_temporal_ablation.py").read_bytes())
    label_job, = (node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == "label_job")
    calls = [
        node for node in ast.walk(label_job)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"run", "branches"}
    ]
    require(sorted(node.func.attr for node in calls) == ["branches", "run"]
            and all(any(keyword.arg == "trace" and isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is False for keyword in node.keywords) for node in calls),
            "Could not verify that both label-generation paths disable state trace serialization.")
    return {
        "status": "exact_trace_only_import_fix",
        "frozen_launch_sha256": launch_hash,
        "active_replay_sha256": active_hash,
        "frozen_launch_snapshot": str(launch.relative_to(run_root)),
        "sole_change": "Import asdict from dataclasses for recorded BoatState serialization under if trace.",
        "scope": "Branches disable trace. Applied-motion reports require the corrected import. "
                 "The decision and plant logic are byte-identical and all original receipts remain pinned.",
    }


def verified_jobs(native):
    """Bind all four arms to matched forcing, vessel, chart and AIS inputs."""
    manifest, protocol = locked_protocol()
    protocol = json.loads(json.dumps(protocol, default=seasonal.json_value))
    scenarios = {spec["id"]: spec for spec in manifest["scenarios"] if spec["split"] == "test"}
    require(len(scenarios) == 12 and all(spec["capture_radius_m"] == 50. for spec in scenarios.values()),
            "Expected all twelve locked 50 m cases.")
    model_dir = native.parent / "models"
    fit_path = model_dir / "fit_receipt.json"
    fit = load_json(fit_path)
    models = {arm: seasonal.digest(model_dir / f"{arm}.npz") for arm in METHODS[1:]}
    require(fit["models_sha256"] == models, "Frozen model inputs changed after fitting.")
    provenance_reconciliation = reconcile_protocol(fit["protocol"], protocol, native.parent)
    protocol = fit["protocol"]
    for split in SPLITS:
        directory = native.parent / "branches"
        labels = directory / f"{split}.jsonl.gz"
        receipt = load_json(directory / f"{split}_receipt.json")
        ids = [spec["id"] for spec in manifest["scenarios"] if spec["split"] == split]
        require(receipt["protocol"] == protocol and receipt["split"] == split and receipt["scenario_ids"] == ids
                and fit["labels_sha256"][split] == receipt["labels_sha256"] == seasonal.digest(labels),
                f"{split}: native branch provenance or partition changed.")
        with gzip.open(labels, "rt", encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream]
        require(len(records) == receipt["states"]
                and all(record["split"] == split and record["scenario"] in ids for record in records),
                f"{split}: native decision states cross the locked partitions.")
    jobs, sources = [], {}
    for arm in METHODS:
        directory = native / arm
        inputs_path, run_path, episodes_path = (directory / name for name in ("inputs.json", "run.json", "episodes.csv"))
        inputs, run = load_json(inputs_path), load_json(run_path)
        inputs_digest, episodes_digest = seasonal.digest(inputs_path), seasonal.digest(episodes_path)
        require(inputs["method"] == run["method"] == arm and inputs["probe"] is False
                and inputs["scenarios"] == list(scenarios) and inputs["model_sha256"] == models.get(arm)
                and run["inputs_sha256"] == inputs_digest and run["episodes_sha256"] == episodes_digest,
                f"{arm}: invalid native inputs, model or receipt.")
        require(all(inputs[key] == value for key, value in protocol.items()),
                f"{arm}: native protocol differs from the current locked benchmark.")
        rows = read_rows(episodes_path)
        require([row["scenario"] for row in rows] == list(scenarios), f"{arm}: reordered or missing test cases.")
        compare(summarize(rows), run["summary"], arm)
        checksums = run["trace_sha256es"]
        require(len(checksums) == len(scenarios), f"{arm}: inconsistent trace count.")
        for row in rows:
            scenario = row["scenario"]
            name = f"telemetry/{scenario}__{arm}.json.gz"
            require(row["method"] == arm and row["trajectory"] == name and name in checksums,
                    f"{arm}/{scenario}: invalid trace path.")
            jobs.append((scenarios[scenario], row, directory / name, checksums[name]))
        sources[arm] = {"inputs_sha256": inputs_digest, "episodes_sha256": episodes_digest,
                        "run_receipt_sha256": seasonal.digest(run_path), "native_inputs": inputs}
    return manifest, scenarios, jobs, sources, seasonal.digest(fit_path), provenance_reconciliation


def decision_key(row):
    return row["elapsed_s"], row["utc"], row["goal"]


def verify_choices(row, trace):
    """Distinguish scoring, crossing the model threshold and applying the tack."""
    events = trace["strategy_evaluations"]
    require(len(events) == int(row["strategy_calls"]), "Strategy call count differs from trace.")
    if row["method"] == "physics":
        require(all(event["status"] == "physics_only" and event["corrected_gain_s"] is None for event in events),
                "The physics arm scored learned candidates.")
    decisions = {decision_key(item): item for item in trace["decisions"]}
    controls = {decision_key(item): item for item in trace["controls"]}
    gains, crossing_gains, applied_gains = [], [], []
    statuses = Counter()
    seen = set()
    for event in events:
        key = decision_key(event)
        require(key in decisions and key not in seen, f"Invalid or duplicated strategy key {key}.")
        seen.add(key)
        choice = decisions[key]
        physics_idx, request_idx = event["physics_index"], event["requested_index"]
        require((physics_idx, request_idx) == (choice["physics_index"], choice["requested_index"]),
                f"Selected candidate differs from strategy event at {key}.")
        statuses[event["status"]] += 1
        gain = event["corrected_gain_s"]
        if gain is not None:
            require(event["eligible_candidates"] >= 2 and isfinite(gain) and gain >= -EPS,
                    f"Invalid predicted gain at {key}.")
            gains.append(gain)
        changed = request_idx != physics_idx
        require(changed == (event["status"] == "override"), f"Unexpected model crossing status at {key}.")
        action = controls.get(key)
        applied = (changed and action is not None and action["choice_id"] == choice["choice_id"]
                   and choice["selected_index"] == request_idx)
        require(applied == event["learned_applied"], f"Applied override flag differs at {key}.")
        if changed:
            require(gain is not None and gain > MINIMUM_MODEL_GAIN_S,
                    f"Model crossing below the strict gain threshold at {key}.")
            crossing_gains.append(gain)
        if applied:
            applied_gains.append(gain)
    actual = (len(gains), len(crossing_gains), len(applied_gains))
    expected = tuple(int(row[key]) for key in ("scored_decisions", "gate_crossings", "learned_changes"))
    require(actual == expected, f"Scoring, gate crossings or applied changes differ. Replayed {actual}, saved {expected}.")
    require(close(max(gains, default=None), number(row, "largest_predicted_gain_s")),
            "Largest predicted gain disagrees with native summary.")
    return {"strategy_status_counts": dict(statuses), "scored_gains_s": gains,
            "gate_crossing_gains_s": crossing_gains, "applied_override_gains_s": applied_gains}


def verify_episode(manifest, job):
    spec, row, path, checksum = job
    require(seasonal.digest(path) == checksum, f"Changed trace: {path}.")
    captures = check_episode_metadata(row, spec)
    payload = read_trace(path)
    check = verify_trace(row, spec, manifest["routes"][spec["route_id"]],
                         manifest["traffic_profiles"][spec["traffic_profile"]], payload, captures)
    require(check["applied_intervals"] == int(row["controls_applied"]), "Applied-control count differs from summary.")
    motion = {
        "states": payload["states"], "captures": payload["captures"],
        "applied": [{key: value for key, value in control.items() if key != "complete_control_seconds"}
                    for control in payload["controls"]],
    }
    motion_hash = sha256(json.dumps(motion, sort_keys=True).encode()).hexdigest()
    return {**check, **verify_choices(row, payload), "captures_details": captures,
            "applied_motion_sha256_without_runtime": motion_hash,
            "control_timings_s": [step["complete_control_seconds"] for step in payload["control_attempts"]]}


def first_causal_forecast(engine, start_state, goals, start, trace, events):
    if not events:
        return None
    event = events[0]
    when = normalise_time(event["utc"])
    state = start_state
    if when > start:
        recorded, = (row for row in trace["states"] if normalise_time(row["utc"]) == when)
        state = BoatState(**{key: recorded[key] for key in STATE_KEYS})
    goal, = (item for item in goals if item.name == event["goal"])
    observed = DecisionEnvironment(engine.environment, when)
    replay = CausalEfficientNativeEngine(observed, config=engine.config, dynamics=engine.dynamics)
    decision = replay.decision(state, goal, when, event["elapsed_s"])
    repeated, = replay.anticipation_events
    require(event.keys() == repeated.keys() and all(
        isclose(value, repeated[key], rel_tol=0., abs_tol=EPS) if type(value) is float else value == repeated[key]
        for key, value in event.items()
    ), f"{goal.name}/{when}: native AIS preview could not be reconstructed from causal inputs.")
    require(decision.choices[decision.physics_index].id == event["selected_choice"],
            f"{goal.name}/{when}: causal replay returned a different native tack.")
    return {"event_utc": event["utc"], "selected_choice": event["selected_choice"],
            "matched_causal_replay": True, "environment_requests": sorted(set(observed.requests))}


def replay_interval(engine, state, start, control, recorded):
    """Hold the available-at-step report, integrate the interpolated physical forcing."""
    setting = engine.config
    count, elapsed = control["native_substeps"], control["elapsed_s"]
    duration = control["duration_s"]
    dt = setting.integration_dt_s
    when = start + np.timedelta64(round(elapsed * 1e9), "ns")
    require(dt == .05 and count > 0 and abs(count * dt - duration) < 1e-8
            and control["held_observation"] is True, "Invalid independent integration interval.")
    # Reconstruct AIS from delivered reports separately from the plant inputs.
    available = engine.environment.observe(state, when)
    plant = engine.environment.plant_observe(state, when)
    action = BoatAction(control["rudder_deg"], control["flap_target_deg"])
    final, path, _ = engine.integrate(state, action, plant, duration)
    require(len(path) == count + 1 and np.isfinite(path).all(), "Invalid re-integrated plant path.")
    position_error = float(np.linalg.norm(xy(final.latitude, final.longitude,
                                             (recorded["latitude"], recorded["longitude"]))))
    state_error = max(abs(getattr(final, key) - recorded[key]) for key in STATE_KEYS[2:])
    require(position_error <= ENDPOINT_TOLERANCE_M and state_error <= STATE_TOLERANCE,
            f"Re-integrated position difference {position_error:g} m, state difference {state_error:g}.")
    times = np.arange(count + 1) * dt
    origin = state.latitude, state.longitude
    own = xy(path[:, 0], path[:, 1], origin)
    report, fixed_radius, uncertainty, age, ids = observation_geometry(
        available, engine.environment, times, origin, setting,
    )
    require(abs(age - control["max_report_age_s"]) < EPS, "Delivered report age changed during replay.")
    truth = truth_track(engine.environment, elapsed + times, origin)
    domains = {"report": (report, fixed_radius + uncertainty), "truth": (truth, fixed_radius + uncertainty),
               "guard": (report, fixed_radius + 2. * uncertainty)}
    margins = {}
    for kind, (center, radius) in domains.items():
        margin, offset, index = swept_margin(times, center - own[:, None, :], radius, conservative=kind != "report")
        margins[kind] = {"margin_m": margin, "elapsed_s": elapsed + offset, "mmsi": ids[index],
                         "held_report_age_at_step_start_s": age}
    slack = float(np.min(uncertainty - np.linalg.norm(truth - report, axis=-1)))
    require(slack >= -EPS and margins["guard"]["margin_m"] <= margins["truth"]["margin_m"] + EPS,
            "Analytical contact position escaped the uncertainty bound.")
    chart_clear = bool(engine.environment.chart_path_is_clear(path, available))
    require(chart_clear == control["applied_chart_clear"], "Applied chart test did not reproduce.")
    return margins, position_error, state_error, slack, chart_clear


def audit_episode(job):
    spec, row, path, checksum = job
    require(seasonal.digest(path) == checksum, f"Changed trace before audit: {path}.")
    trace = read_trace(path)
    engine, state, goals, start, _ = scenario_parts(spec)
    events = trace["anticipation_events"]
    attempts = {control["utc"] for control in trace["control_attempts"]}
    profile = seasonal.INPUTS["traffic_profiles"][spec["traffic_profile"]]
    for event in events:
        when = normalise_time(event["utc"])
        elapsed = int((when - start).astype(np.int64))
        age, _ = delivered_age(profile, elapsed)
        require(event["utc"] in attempts and close(elapsed / 1e9, event["elapsed_s"], EPS)
                and close(age, event["report_age_s"], EPS), f"{path}: unexpected AIS preview clock or report age.")
    require(len(events) == int(row["anticipation_events"]), f"{path}: unexpected AIS preview count.")
    replayed_forecast = first_causal_forecast(engine, state, goals, start, trace, events)
    minima = dict.fromkeys(MARGIN_FIELDS)
    nonpositive = dict.fromkeys(MARGIN_FIELDS, 0)
    errors = dict.fromkeys(MARGIN_FIELDS, 0.)
    position_error = state_error = 0.
    slack = None
    chart_violations = 0
    for control, recorded in zip(trace["controls"], trace["states"], strict=True):
        context = f"{row['method']}/{spec['id']} at {control['elapsed_s']:g} s"
        try:
            checks, position_delta, state_delta, current_slack, chart_clear = replay_interval(
                engine, state, start, control, recorded,
            )
            require(control["screened_guard_margin_m"] > 0., "Admission record reports a nonpositive guard.")
            for kind, check in checks.items():
                margin = check["margin_m"]
                error = verified_value(margin, control[MARGIN_FIELDS[kind]], f"{context} {kind}")
                errors[kind] = max(errors[kind], error)
                nonpositive[kind] += int(margin <= 0.)
                if minima[kind] is None or margin < minima[kind]["margin_m"]:
                    minima[kind] = check
        except ValueError as error:
            raise ValueError(f"{context}: {error}") from error
        state = BoatState(**{key: recorded[key] for key in STATE_KEYS})
        position_error = max(position_error, position_delta)
        state_error = max(state_error, state_delta)
        slack = current_slack if slack is None else min(slack, current_slack)
        chart_violations += int(not chart_clear)
    for kind, field in {"truth": "min_truth_margin_m", "report": "min_report_margin_m", "guard": "min_applied_guard_m"}.items():
        saved = number(row, field)
        check = minima[kind]
        if check is None:
            require(saved is None, f"{path}: clearance exists without any applied step.")
        else:
            verified_value(check["margin_m"], saved, f"{path} aggregate {kind}")
    return {"scenario": spec["id"], "method": row["method"], "trace_sha256": checksum,
            "applied_controls": len(trace["controls"]),
            "integration_substeps": sum(control["native_substeps"] for control in trace["controls"]),
            "maximum_endpoint_error_m": position_error, "maximum_other_state_error": state_error,
            "maximum_clearance_error_m": errors, "minimum_center_uncertainty_slack_m": slack,
            "minimum_clearance": minima, "nonpositive_control_intervals": nonpositive,
            "applied_chart_violations": chart_violations, "ais_preview_events_checked": len(events),
            "first_forecast_causal_replay": replayed_forecast}


def gain_summary(values):
    return {"count": len(values), "positive_count": sum(value > EPS for value in values),
            "median_s": median(values) if values else None, "maximum_s": max(values, default=None)}


def timing_summary(values):
    return {
        "attempts": len(values), "median_s": median(values) if values else None,
        "p95_s": float(np.percentile(values, 95)) if values else None,
        "max_s": max(values, default=None), "attempts_at_or_over_5s": sum(value >= 5. for value in values),
    }


def case_summary(row, check):
    captures = check["captures_details"]
    return {
        "status": row["status"], "completed": completed(row), "duration_s": number(row, "duration_s"),
        "captures_in_order": [{"name": hit["name"], "phase": hit["phase"], "elapsed_s": hit["elapsed_s"],
                               "distance_m": hit["distance_m"]} for hit in captures],
        "applied_controls": check["applied_intervals"],
        "applied_motion_sha256_without_runtime": check["applied_motion_sha256_without_runtime"],
        "truth_full_margin_m": number(row, "min_truth_margin_m"),
        "report_full_margin_m": number(row, "min_report_margin_m"),
        "maximum_control_seconds": number(row, "maximum_decision_s"),
        "control_attempts_at_or_over_5s": int(row["overruns_5s"]),
        "control_timing": timing_summary(check["control_timings_s"]),
        "strategy_calls": int(row["strategy_calls"]), "scored_decisions": int(row["scored_decisions"]),
        "gate_crossings": int(row["gate_crossings"]), "applied_overrides": int(row["learned_changes"]),
        "predicted_gain_scored": gain_summary(check["scored_gains_s"]),
        "predicted_gain_crossing": gain_summary(check["gate_crossing_gains_s"]),
        "predicted_gain_applied": gain_summary(check["applied_override_gains_s"]),
    }


def joint_delta(case, method, baseline):
    arm, base = case[method], case[baseline]
    return arm["duration_s"] - base["duration_s"] if arm["completed"] and base["completed"] else None


def pair_summary(cases, method, baseline):
    deltas = [delta for case in cases.values() if (delta := joint_delta(case, method, baseline)) is not None]
    both_clear = [delta for case in cases.values() if (delta := joint_delta(case, method, baseline)) is not None
                  and all(case[arm]["clearance_status"] == "clear" for arm in (method, baseline))]
    return {
        "joint_finishes": len(deltas), "joint_clear_finishes": len(both_clear),
        "identical_trajectories_with_applied_motion": sum(
            case[method]["applied_controls"] > 0 and case[method]["applied_motion_sha256_without_runtime"]
            == case[baseline]["applied_motion_sha256_without_runtime"] for case in cases.values()
        ),
        "method_only_finishes": sum(case[method]["completed"] and not case[baseline]["completed"] for case in cases.values()),
        "baseline_only_finishes": sum(case[baseline]["completed"] and not case[method]["completed"] for case in cases.values()),
        "method_faster": sum(delta < -EPS for delta in deltas),
        "method_slower": sum(delta > EPS for delta in deltas),
        "ties": sum(abs(delta) <= EPS for delta in deltas),
        "median_method_minus_baseline_s": median(deltas) if deltas else None,
        "mean_method_minus_baseline_s": mean(deltas) if deltas else None,
        "median_method_minus_baseline_s_on_joint_clear_finish": median(both_clear) if both_clear else None,
    }


def flat_case(scenario, spec, cases):
    row = {"scenario": scenario, "route": spec["route_gpx"], "departure_utc": spec["departure_utc"],
           "weather_block": spec["archive_block"], "traffic": spec["traffic_profile"],
           "expected_captures": len(spec["expected_captures"]), "capture_radius_m": spec["capture_radius_m"]}
    for arm, outcome in cases.items():
        for key, value in outcome.items():
            if key == "captures_in_order":
                value = " ".join(capture["name"] for capture in value)
            if isinstance(value, dict):
                row.update({f"{arm}_{key}_{stat}": result for stat, result in value.items()})
            else:
                row[f"{arm}_{key}"] = value
    for arm, baseline in PAIRINGS:
        row[f"{arm}_minus_{baseline}_s_on_joint_finish"] = joint_delta(cases, arm, baseline)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, default=STUDY / "native")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    require(args.workers > 0, "Enable at least one independent replay worker.")
    manifest, scenarios, jobs, sources, fit_digest, provenance_reconciliation = verified_jobs(args.native)
    checks, rows = {}, {}
    for job in jobs:
        spec, row, _, _ = job
        key = row["method"], spec["id"]
        checks[key] = verify_episode(manifest, job)
        rows[key] = row
    print(f"Authenticated waypoint, timing and tack records for {len(jobs)} native traces.", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=init_worker) as pool:
        audited = []
        for record in pool.map(audit_episode, jobs, chunksize=1):
            audited.append(record)
            print(f"Replayed {record['method']} {record['scenario']} ({record['integration_substeps']} knots)", flush=True)
    audit_by_case = {(item["method"], item["scenario"]): item for item in audited}
    unsafe = [{"method": item["method"], "scenario": item["scenario"],
               "nonpositive_control_intervals": item["nonpositive_control_intervals"],
               "applied_chart_violations": item["applied_chart_violations"]} for item in audited
              if any(item["nonpositive_control_intervals"].values()) or item["applied_chart_violations"]]
    common = {"manifest_sha256": seasonal.digest(MANIFEST), "auditor_sha256": seasonal.digest(__file__),
              "run_sources": sources, "fit_receipt_sha256": fit_digest,
              "provenance_reconciliation": provenance_reconciliation}
    status = "failed_clearance" if unsafe else "verified"
    safety = {arm: summarize_clearance(audited, arm) for arm in METHODS}
    seasonal.save_json(args.native.parent / "independent_clearance.json", {
        **common, "status": status,
        "scope": "Fossen re-integration of every applied control at 0.05 s, with interpolated plant forcing, "
                 "causal AIS receiver, analytical target motion, separate B+U report and truth envelopes, "
                 "expanded B+2U guard, and chart replay at applied integration knots. "
                 "First native AIS preview replayed per episode where present, with capped environment access.",
        "applied_controls": sum(item["applied_controls"] for item in audited),
        "integration_substeps": sum(item["integration_substeps"] for item in audited),
        "first_forecast_decisions_replayed": sum(item["first_forecast_causal_replay"] is not None for item in audited),
        "summary": safety, "unsafe_applied_cases": unsafe, "cases": audited,
    })
    cases = {}
    for scenario in scenarios:
        cases[scenario] = {}
        for arm in METHODS:
            key = arm, scenario
            outcome = case_summary(rows[key], checks[key])
            audit = audit_by_case[key]
            outcome.update({"clearance_status": clearance_flag(audit),
                            "guard_margin_m": (audit["minimum_clearance"]["guard"] or {}).get("margin_m"),
                            "applied_chart_violations": audit["applied_chart_violations"]})
            cases[scenario][arm] = outcome
    summaries = {}
    for arm in METHODS:
        arm_checks = [checks[arm, scenario] for scenario in scenarios]
        counts = Counter()
        for check in arm_checks:
            counts.update(check["strategy_status_counts"])
        summaries[arm] = {
            **safety_summary([rows[arm, scenario] for scenario in scenarios]), **safety[arm],
            **{key: sum(case[arm][key] for case in cases.values()) for key in
               ("strategy_calls", "scored_decisions", "gate_crossings", "applied_overrides", "applied_chart_violations")},
            "strategy_status_counts": dict(counts),
            "control_timing": timing_summary([value for check in arm_checks for value in check["control_timings_s"]]),
            "five_second_budget_pass": all(case[arm]["control_attempts_at_or_over_5s"] == 0
                                            for case in cases.values()),
            **{f"predicted_{field}": gain_summary([gain for check in arm_checks for gain in check[field]])
               for field in ("scored_gains_s", "gate_crossing_gains_s", "applied_override_gains_s")},
        }
    seasonal.save_csv(args.native / "paired_episodes.csv",
                      [flat_case(scenario, spec, cases[scenario]) for scenario, spec in scenarios.items()])
    seasonal.save_json(args.native / "independent_comparison.json", {
        **common, "status": status, "arms": list(METHODS), "cases_per_arm": len(scenarios), "capture_radius_m": 50.,
        "sailing_time_comparison": "Method minus baseline, computed only on routes finished by both arms.",
        "timing_scope": "Native control decision and safety screening wall time, including learned inference. "
                        "Offline branch generation, model fitting, independent audit and plant advance are excluded.",
        "summary": summaries,
        "pairwise": {f"{arm}_vs_{baseline}": pair_summary(cases, arm, baseline) for arm, baseline in PAIRINGS},
        "cases": cases,
    })
    print(f"Wrote matched comparison and independent geometry for {len(audited)} native traces.", flush=True)
    if unsafe:
        print(f"Detected {len(unsafe)} cases with a swept AIS or chart violation.", flush=True)


if __name__ == "__main__":
    main()
