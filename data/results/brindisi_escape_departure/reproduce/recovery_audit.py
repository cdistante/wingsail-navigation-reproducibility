"""Independently verify the protected-departure and on-water recovery trials."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
import multiprocessing
from pathlib import Path
from statistics import mean, median
import sys

import numpy as np


STUDY = Path(__file__).resolve().parents[1]
RIDGE = STUDY.with_name("eta_ridge_closed_loop")
CAUSAL = STUDY.with_name("temporal_embedding_native")
sys.path.insert(0, str(CAUSAL / "reproduce"))
sys.path.insert(0, str(RIDGE / "reproduce"))

import ablation_report as audit
import audit_ridge_loop as ridge
import run_temporal_ablation as simulator


METHODS = ("escape", "departure")
REFERENCE = "ridge"
RIDGE_RUN_SHA256 = "5c10a902c03d82103221de743722728bde26cbe45693acc25c838722be8c8291"
RIDGE_AUDIT_SHA256 = "763d2da0d5137a18c2703518c016fa8a26f78d44d601f6aadcd05ff44686b525"
SUMMARY_MARGIN_FIELDS = {
    "truth": "min_truth_margin_m", "report": "min_report_margin_m", "guard": "min_applied_guard_m",
}
seasonal = audit.seasonal
digest = seasonal.digest
require = audit.require
EPS = audit.EPS


def normalized(data):
    return json.loads(json.dumps(data, default=seasonal.json_value))


def frozen_reference():
    manifest, scenarios, jobs, sources, _, _ = ridge.frozen_jobs(RIDGE / "native")
    require(sources["explicit"]["run_receipt_sha256"] == RIDGE_RUN_SHA256,
            "The frozen closed-loop compact ridge reference changed.")
    record = RIDGE / "independent_clearance.json"
    require(digest(record) == RIDGE_AUDIT_SHA256, "The frozen reference clearance audit changed.")
    verified = audit.load_json(record)
    old_geometry = {
        (item["method"], item["scenario"]): item for item in verified["cases"]
    }
    reference = {}
    for job in jobs:
        spec, row, _, checksum = job
        arm = row["method"]
        geometry = old_geometry[arm, spec["id"]]
        require(geometry["trace_sha256"] == checksum, f'{spec["id"]}: missing frozen {arm} geometry.')
        check = audit.verify_episode(manifest, job)
        reference[arm, spec["id"]] = row, check, geometry
    return manifest, scenarios, reference, sources


def shifted_receiver(profile, wait_s):
    """Rebase registered drop windows without resetting the AIS emission clock."""
    period = profile["report_interval_s"]
    require(abs(wait_s - round(wait_s / period) * period) <= EPS,
            "The launch wait does not align with the fixed AIS reporting phase.")
    return {**profile, "drop_windows_s": [(left - wait_s, right - wait_s)
                                         for left, right in profile["drop_windows_s"]]}


def check_episode(manifest, job, trace, launch, wait_s):
    spec, row, _, _ = job
    require(0. <= audit.number(row, "duration_s") <= audit.number(row, "cap_s") + EPS,
            "Sailing elapsed time exceeds the unspent mission cap.")
    captures = audit.check_episode_metadata({**row, "cap_s": row["request_cap_s"]}, spec)
    # The boat's elapsed clock starts when launched. The same receiver keeps
    # running from request, so shift its dropped epochs into the sailing clock.
    shifted_spec = {**spec, "departure_utc": launch}
    route = manifest["routes"][spec["route_id"]]
    profile = shifted_receiver(manifest["traffic_profiles"][spec["traffic_profile"]], wait_s)
    check = audit.verify_trace(row, shifted_spec, route, profile, trace, captures)
    require(check["applied_intervals"] == int(row["controls_applied"]),
            "Applied-control count differs from native episode receipt.")
    motion = {
        "states": trace["states"], "captures": trace["captures"],
        "applied": [{key: value for key, value in control.items() if key != "complete_control_seconds"}
                    for control in trace["controls"]],
    }
    motion_hash = sha256(json.dumps(motion, sort_keys=True).encode()).hexdigest()
    return {
        **check, **audit.verify_choices(row, trace), "captures_details": captures,
        "applied_motion_sha256_without_runtime": motion_hash,
        "control_timings_s": [item["complete_control_seconds"] for item in trace["control_attempts"]],
    }


def replay_geometry(manifest, job, trace, wait_s):
    spec, row, _, checksum = job
    engine, state, goals, requested, _ = simulator.scenario_parts(spec)
    actual = requested + np.timedelta64(round(wait_s * 1e9), "ns")
    profile = manifest["traffic_profiles"][spec["traffic_profile"]]
    anticipation = trace["anticipation_events"]
    attempts = {item["utc"] for item in trace["control_attempts"]}
    for event in anticipation:
        instant = audit.normalise_time(event["utc"])
        elapsed_ns = int((instant - requested).astype(np.int64))
        age, _ = audit.delivered_age(profile, elapsed_ns)
        require(event["utc"] in attempts and audit.close(elapsed_ns / 1e9, wait_s + event["elapsed_s"], EPS)
                and audit.close(age, event["report_age_s"], EPS), "AIS forecast used a shifted receiver clock.")
    require(len(anticipation) == int(row["anticipation_events"]), "AIS forecast event count changed.")
    first_forecast = audit.first_causal_forecast(engine, state, goals, actual, trace, anticipation)

    minima = dict.fromkeys(SUMMARY_MARGIN_FIELDS)
    nonpositive = dict.fromkeys(SUMMARY_MARGIN_FIELDS, 0)
    errors = dict.fromkeys(SUMMARY_MARGIN_FIELDS, 0.)
    max_position_error = max_state_error = 0.
    uncertainty_slack = None
    chart_violations = 0
    admission_violations = 0
    for control, endpoint in zip(trace["controls"], trace["states"], strict=True):
        # The independent integration helper computes analytic truth on its
        # origin clock. Pass absolute fixture time to it even after staging.
        absolute_epoch = control["elapsed_s"] + wait_s
        rebased = {**control, "elapsed_s": absolute_epoch}
        context = f'{row["method"]}/{spec["id"]} at request +{absolute_epoch:g} s'
        try:
            marks, pos_error, state_error, slack, chart = audit.replay_interval(
                engine, state, requested, rebased, endpoint,
            )
            for kind, mark in marks.items():
                margin = mark["margin_m"]
                errors[kind] = max(errors[kind], audit.verified_value(
                    margin, control[audit.MARGIN_FIELDS[kind]], f"{context} {kind}",
                ))
                nonpositive[kind] += int(margin <= 0.)
                if minima[kind] is None or margin < minima[kind]["margin_m"]:
                    minima[kind] = {**mark, "request_elapsed_s": mark["elapsed_s"],
                                    "sailing_elapsed_s": mark["elapsed_s"] - wait_s}
        except ValueError as error:
            raise ValueError(f"{context}: {error}") from error
        state = audit.BoatState(**{key: endpoint[key] for key in audit.STATE_KEYS})
        max_position_error = max(max_position_error, pos_error)
        max_state_error = max(max_state_error, state_error)
        uncertainty_slack = slack if uncertainty_slack is None else min(uncertainty_slack, slack)
        chart_violations += int(not chart)
        admission_violations += int(control["screened_guard_margin_m"] <= 0.)
    for kind, field in SUMMARY_MARGIN_FIELDS.items():
        recorded = audit.number(row, field)
        if minima[kind] is None:
            require(recorded is None, "No-motion episode declared a swept AIS margin.")
        else:
            audit.verified_value(minima[kind]["margin_m"], recorded, f'{spec["id"]} aggregate {kind}')
    return {
        "scenario": spec["id"], "method": row["method"], "trace_sha256": checksum,
        "applied_controls": len(trace["controls"]),
        "integration_substeps": sum(control["native_substeps"] for control in trace["controls"]),
        "maximum_endpoint_error_m": max_position_error, "maximum_other_state_error": max_state_error,
        "maximum_clearance_error_m": errors, "minimum_center_uncertainty_slack_m": uncertainty_slack,
        "minimum_clearance": minima, "nonpositive_control_intervals": nonpositive,
        "applied_chart_violations": chart_violations, "applied_admission_violations": admission_violations,
        "ais_preview_events_checked": len(anticipation), "first_forecast_causal_replay": first_forecast,
    }


def describe_case(row, check, geometry, requested, actual, wait_s, extras=None):
    item = audit.case_summary(row, check)
    item.update({
        "requested_departure_utc": requested, "actual_departure_utc": actual,
        "protected_wait_s": wait_s, "request_to_outcome_s": wait_s + item["duration_s"],
        "clearance_status": audit.clearance_flag(geometry),
        "guard_margin_m": (geometry["minimum_clearance"]["guard"] or {}).get("margin_m"),
        "applied_chart_violations": geometry["applied_chart_violations"],
        "applied_admission_violations": geometry.get("applied_admission_violations", 0),
    })
    if extras:
        item.update(extras)
    return item


def time_deltas(cases, winner, comparator, time_key):
    return [
        (scenario, case[winner][time_key] - case[comparator][time_key])
        for scenario, case in cases.items()
        if case[winner]["completed"] and case[comparator]["completed"]
    ]


def summarize_times(values):
    deltas = [delta for _, delta in values]
    return {
        "cases": len(deltas), "faster": sum(delta < -EPS for delta in deltas),
        "slower": sum(delta > EPS for delta in deltas), "ties": sum(abs(delta) <= EPS for delta in deltas),
        "mean_s": mean(deltas) if deltas else None, "median_s": median(deltas) if deltas else None,
        "by_scenario_s": dict(values),
    }


def summarize_arm(rows, checks, geometries, outcomes, arm):
    statuses = Counter()
    for check in checks:
        statuses.update(check["strategy_status_counts"])
    return {
        **audit.safety_summary(rows), **audit.summarize_clearance(geometries, arm),
        "control_timing": audit.timing_summary([value for check in checks for value in check["control_timings_s"]]),
        "failures_after_motion": sum(not case["completed"] and case["applied_controls"] > 0
                                     for case in outcomes.values()),
        "shield_stops_after_motion": sum(case["status"] == "no_safe_native_action"
                                         and case["applied_controls"] > 0 for case in outcomes.values()),
        "timeouts_after_motion": sum(case["status"] == "censored_cap"
                                     and case["applied_controls"] > 0 for case in outcomes.values()),
        "applied_chart_violations": sum(case["applied_chart_violations"] for case in outcomes.values()),
        "applied_admission_violations": sum(case["applied_admission_violations"] for case in outcomes.values()),
        "strategy_status_counts": dict(statuses),
        "protected_wait_s": {scenario: case["protected_wait_s"] for scenario, case in outcomes.items()},
        "scored_decisions": sum(case["scored_decisions"] for case in outcomes.values()),
        "gate_crossings": sum(case["gate_crossings"] for case in outcomes.values()),
        "applied_overrides": sum(case["applied_overrides"] for case in outcomes.values()),
    }


def summarize_pair(cases, arm, baseline):
    return {
        **audit.pair_summary(cases, arm, baseline),
        "sailing_time": summarize_times(time_deltas(cases, arm, baseline, "duration_s")),
        "request_to_finish_time": summarize_times(time_deltas(cases, arm, baseline, "request_to_outcome_s")),
        "added_completions": [scenario for scenario, case in cases.items()
                              if case[arm]["completed"] and not case[baseline]["completed"]],
        "lost_completions": [scenario for scenario, case in cases.items()
                             if case[baseline]["completed"] and not case[arm]["completed"]],
        "recovered_baseline_after_motion_failures": [
            scenario for scenario, case in cases.items() if case[arm]["completed"]
            and not case[baseline]["completed"] and case[baseline]["applied_controls"] > 0
        ],
    }


def paired_row(scenario, spec, outcomes):
    row = {
        "scenario": scenario, "route": spec["route_gpx"], "request_utc": spec["departure_utc"],
        "weather_block": spec["archive_block"], "traffic": spec["traffic_profile"],
        "capture_radius_m": spec["capture_radius_m"], "expected_captures": len(spec["expected_captures"]),
    }
    for arm, outcome in outcomes.items():
        for key, value in outcome.items():
            if key == "captures_in_order":
                row[f"{arm}_{key}"] = " ".join(hit["name"] for hit in value)
            elif isinstance(value, dict):
                row.update({f"{arm}_{key}_{item}": subvalue for item, subvalue in value.items()})
            else:
                row[f"{arm}_{key}"] = value
    for arm, baseline in (("escape", REFERENCE), ("departure", REFERENCE), ("departure", "escape")):
        row[f"{arm}_minus_{baseline}_sailing_s_on_joint_finish"] = audit.joint_delta(outcomes, arm, baseline)
        row[f"{arm}_minus_{baseline}_total_s_on_joint_finish"] = (
            outcomes[arm]["request_to_outcome_s"] - outcomes[baseline]["request_to_outcome_s"]
            if outcomes[arm]["completed"] and outcomes[baseline]["completed"] else None
        )
    return row


def outputs(native, cases, scenarios, summaries, geometries, scope):
    violations = [
        {key: geom[key] for key in ("method", "scenario", "nonpositive_control_intervals",
                                    "applied_chart_violations", "applied_admission_violations")}
        for geom in geometries
        if any(geom["nonpositive_control_intervals"].values()) or geom["applied_chart_violations"]
        or geom["applied_admission_violations"]
    ]
    shared = {**scope, "status": "clearance_violation" if violations else "verified",
              "cases_per_arm": len(cases), "capture_radius_m": 50.}
    seasonal.save_json(native.parent / "independent_clearance.json", {
        **shared,
        "scope": "Applied 0.05 s geometry audited from independently re-integrated controls. AIS motion and "
                 "report delivery follow request UTC, while navigation and waypoint clocks follow actual launch UTC. "
                 "Protected staging itself does not integrate an in-water boat trajectory.",
        "summary": {arm: audit.summarize_clearance(geometries, arm) for arm in METHODS},
        "control_intervals_replayed": sum(geom["applied_controls"] for geom in geometries),
        "integration_substeps_replayed": sum(geom["integration_substeps"] for geom in geometries),
        "first_ais_previews_replayed": sum(geom["first_forecast_causal_replay"] is not None for geom in geometries),
        "unsafe_applied_cases": violations, "cases": geometries,
    })
    seasonal.save_csv(native / "paired_episodes.csv",
                      [paired_row(case, spec, cases[case]) for case, spec in scenarios.items()])
    seasonal.save_json(native / "independent_comparison.json", {
        **shared, "summary": summaries,
        "pairwise": {f"{arm}_vs_{base}": summarize_pair(cases, arm, base)
                     for arm, base in (("escape", REFERENCE), ("departure", REFERENCE), ("departure", "escape"))},
        "time_comparison": "Intervention minus reference on joint finishes. Request-to-finish includes staging.",
        "cases": cases,
    })
    print(f"Independently verified {len(geometries)} new episode traces and wrote {len(cases)} paired cases.", flush=True)


def verify_protected_polls(spec, profile, result, *, poll_s, max_wait_s, max_report_age_s, startup_s):
    """Verify requested-clock polling and independently repeat release boundary checks."""
    from dataclasses import asdict

    from staged_departure import assess_launch

    checks = result["checks"]
    launch, wait = result["launch_utc"], result["wait_seconds"]
    requested = audit.normalise_time(spec["departure_utc"])
    require(checks and 0. <= wait <= max_wait_s
            and all(check["reason"] != "route_prefix_clear" for check in checks[:-1]),
            "Protected departure skipped a checked eligible launch.")
    expected_offsets = list(np.r_[np.arange(0., max_wait_s, poll_s), max_wait_s])
    require([item["elapsed_s"] for item in checks] == expected_offsets[:len(checks)]
            and audit.close(checks[-1]["elapsed_s"], wait),
            "The staged departure clock or poll coverage changed.")
    delays = []
    for check in checks:
        stamp = audit.normalise_time(check["utc"])
        offset = int((stamp - requested).astype(np.int64))
        age, _ = audit.delivered_age(profile, offset)
        require(audit.close(offset / 1e9, check["elapsed_s"]) and check["report_count"] == len(profile["targets"])
                and audit.close(age, check["maximum_report_age_s"]), "Staging report delivery or UTC differs.")
        require(np.isfinite(check["wall_seconds"]) and check["wall_seconds"] >= 0., "Invalid staging wall clock.")
        require(age <= max_report_age_s or check["reason"] == "ais_reports_too_old",
                "Staging admitted a report beyond the registered age threshold.")
        delays.append(check["wall_seconds"])
    if launch is None:
        require(wait == max_wait_s and result["reason"] == "protected_wait_limit"
                and checks[-1]["reason"] != "route_prefix_clear", "Unexpected release timeout.")
    else:
        effective = checks[-1]
        require(result["reason"] == effective["reason"] == "route_prefix_clear"
                and audit.normalise_time(launch) == audit.normalise_time(effective["utc"])
                and effective["maximum_report_age_s"] <= max_report_age_s
                and effective["chart_clear"] and effective["colreg_clear"]
                and effective["startup_margin_m"] > 0. and effective["preview_margin_m"] > 0.,
                "The actual release does not satisfy the protected startup criteria.")
    engine, boat, goals, _, _ = simulator.scenario_parts(spec)
    for candidate in checks[-2:]:
        when = audit.normalise_time(candidate["utc"])
        # No target truth or later weather/report call is available during
        # the release boundary replay.
        view = audit.DecisionEnvironment(engine.environment, when)
        replay = audit.CausalEfficientNativeEngine(view, config=engine.config, dynamics=engine.dynamics)
        repeated = normalized(asdict(assess_launch(
            replay, boat, goals[0], when, candidate["elapsed_s"],
            maximum_report_age_s=max_report_age_s, uncapped_startup_s=startup_s,
        )))
        require(candidate.keys() == repeated.keys() and all(
            a == repeated[key] if type(a) is not float else audit.close(a, repeated[key])
            for key, a in candidate.items() if key not in ("wall_seconds", "utc")
        ), "Could not repeat the protected release boundary using time-capped AIS and weather.")
    return {"staging_polls": len(checks), "staging_boundary_polls_replayed": min(2, len(checks)),
            "staging_timing": audit.timing_summary(delays), "staging_reason": result["reason"],
            "staging_first_command_guard_m": checks[-1]["startup_margin_m"],
            "staging_route_preview_guard_m": checks[-1]["preview_margin_m"]}


def verify_escape_events(trace, requested, wait_s):
    from escape_maneuvers import MIN_IMPROVEMENT_M, SWITCH_TIMES_S

    events = trace["escape_events"]
    controls = {control["utc"]: control for control in trace["controls"]}
    attempts = {item["utc"] for item in trace["control_attempts"]}
    selected = 0
    applied = set()
    for event in events:
        when = audit.normalise_time(event["utc"])
        absolute = float((when - requested) / np.timedelta64(1, "s"))
        require(event["utc"] in attempts and audit.close(absolute, wait_s + event["elapsed_s"])
                and event["constant_preview_margin_m"] < 0. and event["evaluated_escapes"] >= 0,
                "The escape preview is outside the recorded decision or was launched without a conflict.")
        predicted = event["forecast_margin_m"]
        threshold = event["constant_preview_margin_m"] + MIN_IMPROVEMENT_M
        require(event["selected"] == (predicted is not None and predicted > threshold),
                "An escape selection differs from its registered benefit threshold.")
        if event["selected"]:
            require(event["first_mode"] != event["second_mode"] and event["switch_seconds"] in SWITCH_TIMES_S,
                    "An escape intervention did not use the registered two-trim search.")
            selected += 1
            control = controls.get(event["utc"])
            if control and control["choice_id"] == event["first_mode"]:
                applied.add(event["utc"])
    return {"escape_search_events": len(events), "escape_selected_events": selected,
            "escape_applied_intervals": len(applied),
            "total_evaluated_escape_horizons": sum(item["evaluated_escapes"] for item in events)}


def check_clock_and_interventions(manifest, job, trace):
    from inspect import signature

    from staged_departure import choose_departure

    spec, row, _, _ = job
    requested = audit.normalise_time(spec["departure_utc"])
    actual = trace["launch_utc"]
    wait = trace["departure_wait_s"]
    limit = spec["max_steps"] * spec["control_dt_s"]
    require(audit.normalise_time(trace["requested_utc"]) == requested
            and row["departure_utc"] == spec["departure_utc"] and (row["launch_utc"] or None) == actual
            and audit.close(wait, audit.number(row, "departure_wait_s"))
            and audit.close(limit, audit.number(row, "request_cap_s"))
            and audit.close(limit - wait, audit.number(row, "cap_s"))
            and audit.close(wait + audit.number(row, "duration_s"), audit.number(row, "request_elapsed_s")),
            "Requested and sailing clocks or mission cap disagree.")
    clock = requested + np.timedelta64(round(wait * 1e9), "ns")
    clock_utc = str(clock) + "Z"
    if actual is None:
        require(row["status"] == "staging_timeout" and not any(
            trace[key] for key in ("controls", "control_attempts", "states", "captures", "decisions")
        ), "An unlaunched boat has in-water activity.")
    else:
        require(audit.normalise_time(actual) == clock and row["status"] != "staging_timeout",
                "Actual launch does not follow its declared wait.")
    staging = trace["departure"]
    extras = verify_escape_events(trace, requested, wait)
    require(extras["escape_search_events"] == int(row["escape_searches"])
            and extras["escape_selected_events"] == int(row["escape_selected"]),
            "Escape event counts differ from the native receipt.")
    if row["method"] == "escape":
        require(wait == 0. and staging is None and int(row["departure_checks"]) == 0,
                "The underway arm applied a protected departure intervention.")
    else:
        require(staging and (staging["launch_utc"], staging["wait_seconds"]) == (actual, wait)
                and staging["reason"] == row["departure_reason"] and extras["escape_search_events"] == 0,
                "The staged case contains mismatched decisions or underway escape.")
        profile = manifest["traffic_profiles"][spec["traffic_profile"]]
        config = signature(choose_departure).parameters
        verified = verify_protected_polls(
            spec, profile, staging, poll_s=config["poll_seconds"].default,
            max_wait_s=config["max_wait_s"].default, max_report_age_s=config["maximum_report_age_s"].default,
            startup_s=config["uncapped_startup_s"].default,
        )
        timing = verified["staging_timing"]
        require(verified["staging_polls"] == int(row["departure_checks"])
                and audit.close(timing["max_s"], audit.number(row, "max_departure_check_s"))
                and timing["attempts_at_or_over_5s"] == int(row["departure_overruns_5s"]),
                "Protected gate timing summary disagrees with the raw attempts.")
        extras.update(verified)
    return actual, clock_utc, wait, extras


def replay_job(arguments):
    manifest, job = arguments
    spec, row, path, checksum = job
    require(digest(path) == checksum, f"Changed native trace {path}.")
    trace = audit.read_trace(path)
    actual, clock_utc, wait, extras = check_clock_and_interventions(manifest, job, trace)
    check = check_episode(manifest, job, trace, clock_utc, wait)
    geometry = replay_geometry(manifest, job, trace, wait)
    item = describe_case(row, check, geometry, spec["departure_utc"], actual, wait, extras)
    return check, geometry, item


def main():
    from run_recovery import verified_protocol

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, default=STUDY / "native")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    require(args.workers > 0, "Choose a positive replay worker count.")
    manifest, scenarios, reference, archived_sources = frozen_reference()
    current_manifest, protocol, checkpoint = verified_protocol()
    require(manifest == current_manifest and list(scenarios) == protocol["test_ids"],
            "Recovery cases differ from the frozen benchmark.")
    protocol = normalized(protocol)
    jobs, sources = [], {}
    for arm in METHODS:
        inputs, work, source = ridge.verify_arm(arm, args.native / arm, scenarios)
        require(all(inputs[key] == value for key, value in protocol.items())
                and inputs["model_sha256"] == digest(checkpoint),
                f"{arm}: recovery receipt lost the common experimental protocol.")
        jobs.extend(work)
        sources[arm] = source
    by_key = {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=simulator.init_worker) as pool:
        for check, geometry, item in pool.map(replay_job, ((manifest, job) for job in jobs), chunksize=1):
            key = geometry["method"], geometry["scenario"]
            by_key[key] = check, geometry, item
            print(f"Audited {key[0]} {key[1]} ({geometry['integration_substeps']} knots)", flush=True)
    cases = {}
    for scenario, spec in scenarios.items():
        row, check, geometry = reference["explicit", scenario]
        comparison = describe_case(row, check, geometry, spec["departure_utc"], spec["departure_utc"], 0.)
        cases[scenario] = {REFERENCE: comparison, **{arm: by_key[arm, scenario][2] for arm in METHODS}}
    summaries = {REFERENCE: summarize_arm(
        [reference["explicit", scenario][0] for scenario in scenarios],
        [reference["explicit", scenario][1] for scenario in scenarios],
        [reference["explicit", scenario][2] for scenario in scenarios],
        {scenario: cases[scenario][REFERENCE] for scenario in scenarios}, "explicit",
    )}
    summaries["physics_archived_context"] = audit.safety_summary(
        [reference["physics", scenario][0] for scenario in scenarios]
    )
    geometries = [geometry for _, geometry, _ in by_key.values()]
    for arm in METHODS:
        outcomes = {scenario: cases[scenario][arm] for scenario in scenarios}
        summaries[arm] = summarize_arm(
            [row for _, row, _, _ in jobs if row["method"] == arm],
            [by_key[arm, scenario][0] for scenario in scenarios], geometries, outcomes, arm,
        )
    summaries["escape"].update({
        field: sum(case["escape"][field] for case in cases.values())
        for field in ("escape_search_events", "escape_selected_events", "escape_applied_intervals")
    })
    stage = [case["departure"] for case in cases.values()]
    summaries["departure"].update({
        "staging_polls": sum(item["staging_polls"] for item in stage),
        "staging_boundary_polls_replayed": sum(item["staging_boundary_polls_replayed"] for item in stage),
        "staging_overruns_5s": sum(item["staging_timing"]["attempts_at_or_over_5s"] for item in stage),
        "max_staging_check_s": max(item["staging_timing"]["max_s"] for item in stage),
    })
    outputs(args.native, cases, scenarios, summaries, geometries, {
        "manifest_sha256": digest(simulator.MANIFEST), "auditor_sha256": digest(__file__),
        "geometry_auditor_sha256": digest(audit.__file__), "run_sources": sources,
        "archived_reference": archived_sources, "archived_clearance_sha256": RIDGE_AUDIT_SHA256,
        "model_sha256": digest(checkpoint),
    })


if __name__ == "__main__":
    main()
