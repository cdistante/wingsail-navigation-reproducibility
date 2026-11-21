"""Independently audit hybrid wind planning against the frozen staged comparator."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import sys

import numpy as np


STUDY = Path(__file__).resolve().parents[1]
RECOVERY = STUDY.with_name("brindisi_escape_departure")
sys.path.insert(0, str(RECOVERY / "reproduce"))

import recovery_audit as prior
from world._geo import haversine_metres, local_east_north_metres


audit = prior.audit
simulator = prior.simulator
seasonal = prior.seasonal
digest = prior.digest
require = prior.require
METHODS = ("departure", "hybrid")
DEPARTURE_RUN_SHA256 = "ceeb396fd7d024ec706cc5bbfc1e18c4803a0ff9fadfdbb714a094acbbcb2daf"
DEPARTURE_AUDIT_SHA256 = "2f4eefac2a3122619311d9bcf0ae54ed239f361bd221b7dac72bcc41ad771b58"
STAGING_FIELDS = (
    "departure_utc", "launch_utc", "departure_wait_s", "departure_reason", "departure_checks",
    "max_departure_check_s", "departure_overruns_5s", "request_cap_s", "cap_s",
)


def frozen_comparator(manifest, scenarios):
    _, jobs, sources = prior.ridge.verify_arm("departure", RECOVERY / "native" / "departure", scenarios)
    path = RECOVERY / "independent_clearance.json"
    require(sources["run_receipt_sha256"] == DEPARTURE_RUN_SHA256 and digest(path) == DEPARTURE_AUDIT_SHA256,
            "The frozen staged comparator or its independent audit changed.")
    saved = audit.load_json(path)
    require(saved["status"] == "verified" and saved["cases_per_arm"] == len(scenarios),
            "The staged comparator lacks a valid clearance audit.")
    geometry_by_case = {item["scenario"]: item for item in saved["cases"] if item["method"] == "departure"}
    require(geometry_by_case.keys() == scenarios.keys(), "The frozen staged audit has an incomplete roster.")
    comparator = {}
    for job in jobs:
        spec, row, telemetry, checksum = job
        require(digest(telemetry) == checksum, f'Changed frozen trajectory {telemetry}.')
        geometry = geometry_by_case[spec["id"]]
        require(geometry["trace_sha256"] == checksum, "Frozen geometry binds a different staged trace.")
        trace = audit.read_trace(telemetry)
        launch, release_or_limit, wait, staging = prior.check_clock_and_interventions(manifest, job, trace)
        check = prior.check_episode(manifest, job, trace, release_or_limit, wait)
        item = prior.describe_case(row, check, geometry, spec["departure_utc"], launch, wait, staging)
        item["waypoint_002_approach"] = waypoint_002(manifest, spec, trace)
        comparator[spec["id"]] = row, check, geometry, item, trace["departure"]
    return comparator, sources


def waypoint_002(manifest, spec, trace):
    """Measure the first leg of the stored trajectory, including the exact capture if present."""
    if trace["launch_utc"] is None:
        return None
    route = manifest["routes"][spec["route_id"]]["waypoints"]
    origin, goal = route[:2]
    require(goal["waypoint"] == "002", f'{spec["id"]}: first goal is not waypoint 002.')
    initial = origin["latitude"], origin["longitude"]
    waypoint = goal["latitude"], goal["longitude"]
    capture = next((hit for hit in trace["captures"] if hit["name"] == "002"), None)
    limit = None if capture is None else capture.get("integrator_elapsed_s", capture["elapsed_s"])
    positions = [(0., initial)]
    for state in trace["states"]:
        if limit is None or state["elapsed_s"] < limit - prior.EPS:
            positions.append((state["elapsed_s"], (state["latitude"], state["longitude"])))
        elif capture is not None:
            break
    if capture is not None:
        positions.append((capture["elapsed_s"], (capture["latitude"], capture["longitude"])))
    times, points = zip(*positions, strict=True)
    distances = [haversine_metres(*point, *waypoint) for point in points]
    best = int(np.argmin(distances))
    easting, northing = local_east_north_metres(*points[-1], *points[0])
    waypoint_decisions = [item for item in trace["decisions"] if item["goal"] == "002"
                          and (capture is None or item["elapsed_s"] <= capture["elapsed_s"] + prior.EPS)]
    finite = [item for item in waypoint_decisions if item["predicted_eta_s"] is not None]
    return {
        "captured": capture is not None,
        "capture_sailing_s": capture["elapsed_s"] if capture is not None else None,
        "capture_request_s": trace["departure_wait_s"] + capture["elapsed_s"] if capture is not None else None,
        "sampled_minimum_distance_m": distances[best],
        "sampled_minimum_sailing_s": times[best], "last_distance_m": distances[-1], "start_distance_m": distances[0],
        "path_length_m": sum(haversine_metres(*a, *b) for a, b in zip(points, points[1:])),
        "east_displacement_m": float(easting), "north_displacement_m": float(northing),
        "finite_002_schedule_etas": len(finite), "002_schedule_decisions": len(waypoint_decisions),
        "first_finite_eta_sailing_s": finite[0]["elapsed_s"] if finite else None,
        "mode_switches": sum(a["choice_id"] != b["choice_id"] for a, b in zip(trace["controls"], trace["controls"][1:])
                             if a["goal"] == b["goal"] == "002"),
    }


def check_unchanged_departure(manifest, job, trace, old_row, old_staging):
    spec, row, _, _ = job
    require(all(row[key] == old_row[key] for key in STAGING_FIELDS)
            and trace["departure"] == old_staging, f'{spec["id"]}: hybrid changed protected departure.')
    require(trace["departure_wait_s"] == audit.number(old_row, "departure_wait_s")
            and (trace["launch_utc"] or "") == old_row["launch_utc"]
            and audit.normalise_time(trace["requested_utc"]) == audit.normalise_time(spec["departure_utc"]),
            f'{spec["id"]}: hybrid changed a fixture clock.')
    # The prior verifier checks the preserved departure boundary and reporting
    # phase. Present the hybrid as a staged arm only for those common checks.
    clock_job = (spec, {**row, "method": "departure"}, job[2], job[3])
    return prior.check_clock_and_interventions(manifest, clock_job, trace)


def geometry_only(manifest, job, trace, wait):
    # The applied plant and guard are unchanged. Do not feed hybrid previews
    # to the persistence-specific replay. The live hybrid replay checks them.
    physical_job = (job[0], {**job[1], "anticipation_events": "0"}, job[2], job[3])
    physical_trace = {**trace, "anticipation_events": []}
    return prior.replay_geometry(manifest, physical_job, physical_trace, wait)


def paired_row(scenario, spec, cases):
    row = {
        "scenario": scenario, "route": spec["route_gpx"], "request_utc": spec["departure_utc"],
        "weather_block": spec["archive_block"], "traffic": spec["traffic_profile"],
        "capture_radius_m": spec["capture_radius_m"], "expected_captures": len(spec["expected_captures"]),
    }
    for arm, outcome in cases.items():
        keys = ("status", "completed", "requested_departure_utc", "actual_departure_utc", "protected_wait_s",
                "duration_s", "request_to_outcome_s", "clearance_status", "truth_full_margin_m",
                "report_full_margin_m", "guard_margin_m", "maximum_control_seconds",
                "control_attempts_at_or_over_5s", "staging_polls", "scored_decisions", "applied_overrides")
        row.update({f"{arm}_{key}": outcome[key] for key in keys})
        row[f"{arm}_ordered_captures"] = " ".join(hit["name"] for hit in outcome["captures_in_order"])
        mark = outcome["waypoint_002_approach"] or {}
        for key in ("sampled_minimum_distance_m", "last_distance_m", "capture_sailing_s", "capture_request_s",
                    "path_length_m", "finite_002_schedule_etas", "002_schedule_decisions"):
            row[f"{arm}_002_{key}"] = mark.get(key)
    row["hybrid_minus_departure_sailing_s_on_joint_finish"] = audit.joint_delta(cases, "hybrid", "departure")
    row["hybrid_minus_departure_total_s_on_joint_finish"] = (
        cases["hybrid"]["request_to_outcome_s"] - cases["departure"]["request_to_outcome_s"]
        if cases["hybrid"]["completed"] and cases["departure"]["completed"] else None
    )
    return row


class CausalWindArchive:
    """Only expose wind samples whose valid time is at or before issue."""

    def __init__(self, environment):
        self.environment = environment

    def at(self, latitude, longitude, when):
        self.environment.check(when, "wind_archive_hour")
        return self.environment._environment.as_of_archive.at(latitude, longitude, when)


class CausalIssueEnvironment(audit.DecisionEnvironment):
    @property
    def as_of_archive(self):
        return CausalWindArchive(self)


def event_equal(saved, repeated):
    if isinstance(saved, dict):
        return isinstance(repeated, dict) and saved.keys() == repeated.keys() and all(
            event_equal(saved[key], repeated[key]) for key in saved
        )
    if isinstance(saved, list):
        return isinstance(repeated, list) and len(saved) == len(repeated) and all(
            event_equal(a, b) for a, b in zip(saved, repeated, strict=True)
        )
    if type(saved) is float:
        return audit.close(saved, repeated, prior.EPS)
    return saved == repeated


def verify_hybrid_forecast(manifest, spec, row, trace):
    from run_hybrid import scenario_parts

    wind_events = trace["wind_forecast_events"]
    ais_events = trace["anticipation_events"]
    attempts = {item["utc"]: item for item in trace["control_attempts"]}
    engine, first_state, goals, requested, _ = scenario_parts(spec)
    wait = trace["departure_wait_s"]
    profile = manifest["traffic_profiles"][spec["traffic_profile"]]
    require(len(wind_events) == int(row["wind_forecast_events"]) and len(ais_events) == int(row["anticipation_events"]),
            "Hybrid forecast event counts disagree with the native receipt.")
    fallback_slots = changed_count = 0
    for event in wind_events:
        when = audit.normalise_time(event["utc"])
        elapsed_ns = int((when - requested).astype(np.int64))
        require(event["utc"] in attempts and audit.close(elapsed_ns / 1e9, wait + event["elapsed_s"], prior.EPS),
                "Tactical forecast issue is outside its control update or sailing clock.")
        if event["status"] == "persistence_fallback":
            require(event["forecast_slots"] == event["changed_slots"] == 0, "Fallback unexpectedly changed a slot.")
            continue
        require(event["status"] == "forecast_slots", "Unknown hybrid forecast state.")
        offsets, vectors = event["slot_seconds"], event["wind_to_en_mps"]
        require(len(offsets) == len(vectors) == event["forecast_slots"] and offsets and offsets[0] == 0
                and all(a < b for a, b in zip(offsets, offsets[1:])) and offsets[-1] <= engine.config.forecast_horizon_s
                and np.isfinite(vectors).all() and np.isfinite(event["current_to_en_mps"]).all()
                and np.isfinite(event["wave_height_m"]), "Invalid forecast slots or observed environmental features.")
        require(0 <= event["slot_read_failures"] <= len(offsets)
                and event["changed_slots"] == sum(v != vectors[0] for v in vectors),
                "Hybrid wind slot changes disagree with recorded vectors.")
        fallback_slots += event["slot_read_failures"]
        changed_count += event["changed_slots"]
    for event in ais_events:
        when = audit.normalise_time(event["utc"])
        elapsed_ns = int((when - requested).astype(np.int64))
        age, _ = audit.delivered_age(profile, elapsed_ns)
        require(event["utc"] in attempts and audit.close(elapsed_ns / 1e9, wait + event["elapsed_s"], prior.EPS)
                and audit.close(age, event["report_age_s"], prior.EPS), "Hybrid AIS preview reset the report clock.")

    checked = []
    wanted = {event["utc"] for event in wind_events[:1] + ais_events[:1]}
    if wanted:
        launch = audit.normalise_time(trace["launch_utc"])
        poses = {item["utc"]: item for item in trace["states"]}
        marks = {goal.name: goal for goal in goals}
        for utc in sorted(wanted):
            when = audit.normalise_time(utc)
            event = next(item for item in wind_events if item["utc"] == utc)
            state = first_state if when == launch else audit.BoatState(
                **{key: poses[utc][key] for key in audit.STATE_KEYS}
            )
            observed = CausalIssueEnvironment(engine.environment, when)
            replay = type(engine)(observed, config=engine.config, dynamics=engine.dynamics,
                                  wind_provider=engine.wind_provider)
            decision = replay.decision(state, marks[event["goal"]], when, event["elapsed_s"])
            replayed_wind, = replay.wind_forecast_events
            require(event_equal(event, prior.normalized(replayed_wind)),
                    "Hybrid wind slots changed when regenerated using issue-limited weather.")
            if event["status"] == "forecast_slots":
                measured = observed.observe(state, when)
                require(event["wind_to_en_mps"][0] == [measured.wind.u, measured.wind.v]
                        and event["current_to_en_mps"] == [measured.current.u, measured.current.v]
                        and event["wave_height_m"] == measured.wave.height,
                        "The hybrid planner replaced its observed first slot, current, or wave.")
            ais = next((item for item in ais_events if item["utc"] == utc), None)
            if ais is not None:
                repeated_ais, = replay.anticipation_events
                require(event_equal(ais, prior.normalized(repeated_ais))
                        and decision.choices[decision.physics_index].id == ais["selected_choice"],
                        "The original AIS preview could not be reconstructed with hybrid tactical planning.")
            checked.append({"event_utc": utc, "forecast_matched": True, "ais_preview_matched": ais is not None,
                            "environment_requests": sorted(set(observed.requests))})
    return {"wind_forecast_events_checked": len(wind_events), "forecast_slots_changed": changed_count,
            "wind_fallback_decisions": sum(item["status"] == "persistence_fallback" for item in wind_events),
            "wind_fallback_slots": fallback_slots, "ais_preview_events_checked": len(ais_events),
            "causal_forecast_replays": checked}


def replay_job(arguments):
    manifest, job, old_row, old_staging = arguments
    spec, row, path, checksum = job
    require(digest(path) == checksum, f'Changed hybrid trajectory {path}.')
    trace = audit.read_trace(path)
    actual, clock_utc, wait, staging = check_unchanged_departure(manifest, job, trace, old_row, old_staging)
    check = prior.check_episode(manifest, job, trace, clock_utc, wait)
    geometry = geometry_only(manifest, job, trace, wait)
    geometry.update(verify_hybrid_forecast(manifest, spec, row, trace))
    item = prior.describe_case(row, check, geometry, spec["departure_utc"], actual, wait, staging)
    item["waypoint_002_approach"] = waypoint_002(manifest, spec, trace)
    return check, geometry, item


def write_reports(native, scenarios, cases, summaries, geometries, sources, model_sha):
    violations = [
        {key: geom[key] for key in ("method", "scenario", "nonpositive_control_intervals",
                                    "applied_chart_violations", "applied_admission_violations")}
        for geom in geometries
        if any(geom["nonpositive_control_intervals"].values()) or geom["applied_chart_violations"]
        or geom["applied_admission_violations"]
    ]
    shared = {
        "status": "clearance_violation" if violations else "verified",
        "cases_per_arm": len(cases), "capture_radius_m": 50., "identical_staging_decisions": len(cases),
        "manifest_sha256": digest(simulator.MANIFEST), "auditor_sha256": digest(__file__),
        "geometry_auditor_sha256": digest(prior.__file__), "run_sources": sources,
        "frozen_departure_audit_sha256": DEPARTURE_AUDIT_SHA256, "model_sha256": model_sha,
    }
    seasonal.save_json(native.parent / "independent_clearance.json", {
        **shared, "scope": "Applied safety replay uses the unchanged archived Fossen plant at 0.05 s. "
                           "AIS fixture and report delivery follow request UTC. Hybrid weather issues "
                           "and original AIS previews were spot-replayed with access capped at issue UTC.",
        "summary": audit.summarize_clearance(geometries, "hybrid"),
        "control_intervals_replayed": sum(item["applied_controls"] for item in geometries),
        "integration_substeps_replayed": sum(item["integration_substeps"] for item in geometries),
        "wind_forecast_events_checked": sum(item["wind_forecast_events_checked"] for item in geometries),
        "ais_preview_events_checked": sum(item["ais_preview_events_checked"] for item in geometries),
        "causal_forecast_replays": sum(len(item["causal_forecast_replays"]) for item in geometries),
        "unsafe_applied_cases": violations, "cases": geometries,
    })
    timeouts = {}
    for scenario, pair in cases.items():
        if pair["departure"]["status"] != "censored_cap":
            continue
        original, new = (pair[arm]["waypoint_002_approach"] for arm in METHODS)
        timeouts[scenario] = {
            "departure": original, "hybrid": new,
            "hybrid_minus_departure_closest_sample_m": (
                new["sampled_minimum_distance_m"] - original["sampled_minimum_distance_m"] if new else None
            ),
            "hybrid_minus_departure_last_distance_m": new["last_distance_m"] - original["last_distance_m"] if new else None,
        }
    seasonal.save_csv(native / "paired_episodes.csv",
                      [paired_row(scenario, spec, cases[scenario]) for scenario, spec in scenarios.items()])
    seasonal.save_json(native / "independent_comparison.json", {
        **shared, "summary": summaries, "pairwise": {"hybrid_vs_departure": prior.summarize_pair(cases, "hybrid", "departure")},
        "time_comparison": "Hybrid minus protected departure on jointly completed routes, with matched wait included in total.",
        "waypoint_002_distance_scope": "Start, applied 5 s endpoints and exact recorded 50 m entry, up to first capture.",
        "former_timeout_waypoint_002": timeouts, "cases": cases,
    })
    print(f"Replayed and paired {len(cases)} hybrid trials. Clearance violations: {len(violations)}.", flush=True)


def main():
    from run_hybrid import verified_protocol

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, default=STUDY / "native")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    require(args.workers > 0, "Choose a positive independent replay worker count.")
    manifest, protocol, checkpoint = verified_protocol()
    protocol = prior.normalized(protocol)
    scenarios = {spec["id"]: spec for spec in manifest["scenarios"] if spec["split"] == "test"}
    require(len(scenarios) == 12 and list(scenarios) == protocol["test_ids"]
            and all(spec["capture_radius_m"] == 50. for spec in scenarios.values()), "Unexpected October test roster.")
    simulator.init_worker()
    archived, old_sources = frozen_comparator(manifest, scenarios)
    inputs, jobs, hybrid = prior.ridge.verify_arm("hybrid", args.native / "hybrid", scenarios)
    require(all(inputs[key] == value for key, value in protocol.items())
            and inputs["model_sha256"] == digest(checkpoint), "Hybrid receipt differs from pinned dependencies.")
    tasks = [(manifest, job, archived[job[0]["id"]][0], archived[job[0]["id"]][4]) for job in jobs]
    indexed = {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=simulator.init_worker) as pool:
        for check, geometry, case in pool.map(replay_job, tasks, chunksize=1):
            scenario = geometry["scenario"]
            indexed[scenario] = check, geometry, case
            print(f"Audited {scenario} ({geometry['integration_substeps']} knots)", flush=True)
    cases = {scenario: {"departure": archived[scenario][3], "hybrid": indexed[scenario][2]} for scenario in scenarios}
    geometries = [indexed[scenario][1] for scenario in scenarios]
    summaries = {
        "departure": prior.summarize_arm(
            [archived[scenario][0] for scenario in scenarios],
            [archived[scenario][1] for scenario in scenarios],
            [archived[scenario][2] for scenario in scenarios],
            {scenario: cases[scenario]["departure"] for scenario in scenarios}, "departure",
        ),
        "hybrid": prior.summarize_arm(
            [job[1] for job in jobs], [indexed[scenario][0] for scenario in scenarios], geometries,
            {scenario: cases[scenario]["hybrid"] for scenario in scenarios}, "hybrid",
        ),
    }
    for arm in METHODS:
        stage = [outcomes[arm]["staging_timing"] for outcomes in cases.values()]
        summaries[arm].update({
            "staging_polls": sum(item["attempts"] for item in stage),
            "maximum_staging_check_s": max(item["max_s"] for item in stage),
            "staging_overruns_5s": sum(item["attempts_at_or_over_5s"] for item in stage),
        })
    write_reports(args.native, scenarios, cases, summaries, geometries,
                  {"departure": old_sources, "hybrid": hybrid}, digest(checkpoint))


if __name__ == "__main__":
    main()
