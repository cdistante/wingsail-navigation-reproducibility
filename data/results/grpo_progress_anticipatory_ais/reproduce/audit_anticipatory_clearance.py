"""Independently re-integrate the anticipatory AIS runs and audit the target encounter."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from math import isclose
import multiprocessing
from pathlib import Path
import sys

import numpy as np


STUDY = Path(__file__).resolve().parents[1]
ROBUST = STUDY.with_name("grpo_progress_stale_ais")
sys.path.insert(0, str(ROBUST / "reproduce"))

from audit_stale_clearance import (
    MANIFEST, METHODS, PREVIOUS, audit_job, clearance_flag, historical_rejection,
    require, seasonal, source_jobs,
)
from recheck_clearance import STATE_KEYS
from verify_seasonal_heldout import delivered_age
from world._geo import normalise_time
from world.boat import BoatState


TARGET_SCENARIO = "autumn__20261014T1200Z__brindisi_003__test_raking_overtake"
ACTUATOR_FIELDS = ("rudder_deg", "flap_target_deg")


class DecisionEnvironment:
    """Keep historical weather and received AIS, with no traffic truth API."""

    def __init__(self, environment, through):
        self._environment = environment
        self.through = through
        self.requests = []

    def check(self, when, source):
        instant = normalise_time(when)
        require(instant <= self.through, f"{source} queried after the first intervention.")
        self.requests.append(source)

    def observe(self, state, when):
        self.check(when, "observation")
        return self._environment.observe(state, when)

    def reports_at(self, when):
        self.check(when, "delivered_ais")
        return self._environment.reports_at(when)

    def weather_history(self, latitude, longitude, when):
        self.check(when, "weather_history")
        return self._environment.weather_history(latitude, longitude, when)

    def chart_path_is_clear(self, path, observation=None):
        return self._environment.chart_path_is_clear(path, observation)


def read_trace(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def audit_episode(job):
    spec, row, path, _ = job
    geometry = audit_job(job)
    profile = seasonal.INPUTS["traffic_profiles"][spec["traffic_profile"]]
    start = np.datetime64(spec["departure_utc"].removesuffix("Z"), "ns")
    trace = read_trace(path)
    maximum_age = 0.
    for control in trace["controls"]:
        at = np.datetime64(control["utc"].removesuffix("Z"), "ns")
        elapsed_ns = int((at - start).astype(np.int64))
        age, dropped = delivered_age(profile, elapsed_ns)
        require(isclose(age, control["max_report_age_s"], rel_tol=0., abs_tol=1e-7)
                and dropped == control["dropped_report_epochs"],
                f'{row["method"]}/{spec["id"]}: unexpected delivered report or dropout at {at}.')
        maximum_age = max(age, maximum_age)
    geometry["receiver_check"] = {
        "applied_intervals_checked": len(trace["controls"]),
        "maximum_verified_held_age_s": maximum_age if trace["controls"] else None,
    }
    events = trace["anticipation_events"]
    attempts_at_utc = {attempt["utc"] for attempt in trace["control_attempts"]}
    previous_when = start
    for event in events:
        when = normalise_time(event["utc"])
        elapsed_ns = int((when - start).astype(np.int64))
        age, _ = delivered_age(profile, elapsed_ns)
        require(when >= previous_when and event["utc"] in attempts_at_utc
                and isclose(elapsed_ns / 1e9, event["elapsed_s"], rel_tol=0., abs_tol=1e-7)
                and isclose(age, event["report_age_s"], rel_tol=0., abs_tol=1e-7),
                f'{row["method"]}/{spec["id"]}: invalid anticipation time or report age at {when}.')
        previous_when = when
    require(len(events) == int(row["anticipation_events"]),
            f'{row["method"]}/{spec["id"]}: anticipation event count differs from the episode receipt.')
    geometry["receiver_check"]["anticipation_events_checked"] = len(events)
    return geometry


def control_snapshot(control):
    return {key: control[key] for key in (
        "elapsed_s", "duration_s", *ACTUATOR_FIELDS, "bow_deg", "goal", "choice_id",
        "max_report_age_s", "dropped_report_epochs",
        "report_minimum_m", "truth_minimum_m",
    )}


def first_control_change(previous, new):
    for incumbent, altered in zip(previous, new):
        if any(not isclose(incumbent[key], altered[key], rel_tol=0., abs_tol=1e-9)
               for key in ("elapsed_s", "duration_s", *ACTUATOR_FIELDS)):
            return {"kind": "changed_command", "elapsed_s": altered["elapsed_s"],
                    "before": control_snapshot(incumbent), "after": control_snapshot(altered)}
    if len(previous) != len(new):
        last = min(len(previous), len(new))
        omitted = previous[last] if len(previous) > len(new) else new[last]
        return {"kind": "earlier_end" if len(previous) > len(new) else "extended_motion",
                "elapsed_s": omitted["elapsed_s"],
                "before": control_snapshot(omitted) if len(previous) > len(new) else None,
                "after": control_snapshot(omitted) if len(new) > len(previous) else None}
    return None


def outcome(row, trace):
    dt = row["duration_s"]
    return {
        "status": row["status"], "completed": row["completed"] == "True",
        "elapsed_s": float(dt) if dt else None,
        "ordered_captures": [{"name": mark["name"], "elapsed_s": mark["elapsed_s"],
                              "distance_m": mark["distance_m"]} for mark in trace["captures"]],
        "terminal_rejection": trace["control_attempts"][-1]
        if trace["control_attempts"] and not trace["control_attempts"][-1]["applied"] else None,
    }


def replay_first_intervention(spec, trace):
    events = trace["anticipation_events"]
    if not events:
        return None
    from native_anticipatory_guard import AnticipatoryNativeEngine

    event = events[0]
    when = normalise_time(event["utc"])
    base, initial, goals, start, _ = seasonal.scenario_parts(spec)
    if when > start:
        states = [fixed for fixed in trace["states"] if normalise_time(fixed["utc"]) == when]
        require(len(states) == 1, "Missing applied endpoint at the first anticipatory decision.")
        state = BoatState(**{key: states[0][key] for key in STATE_KEYS})
    else:
        state = initial
    goal, = (target for target in goals if target.name == event["goal"])
    environment = DecisionEnvironment(base.environment, when)
    engine = AnticipatoryNativeEngine(environment, config=base.config, dynamics=base.dynamics)
    choice = engine.decision(state, goal, when, event["elapsed_s"])
    repeated, = engine.anticipation_events
    require(repeated.keys() == event.keys() and all(
        isclose(value, repeated[key], rel_tol=0., abs_tol=1e-7)
        if type(value) is float else value == repeated[key]
        for key, value in event.items()
    ), "First anticipatory event could not be reproduced from the observation available then.")
    require(choice.choices[choice.physics_index].id == event["selected_choice"],
            "Causal replay returned a different immediate choice.")
    return {"event": event, "matched_causal_replay": True,
            "environment_requests": sorted(set(environment.requests)),
            "synthetic_truth_unavailable": True, "latest_allowed_environment_utc": event["utc"]}


def encounter_timeline(scenarios, jobs, indexed):
    histories = {
        name: {row["method"]: (row, path, checksum)
               for _, row, path, checksum in source_jobs(root / "native", scenarios)
               if row["scenario"] == TARGET_SCENARIO}
        for name, root in (("original", PREVIOUS), ("robust_only", ROBUST))
    }
    histories["anticipatory"] = {row["method"]: (row, path, checksum)
                                 for _, row, path, checksum in jobs
                                 if row["scenario"] == TARGET_SCENARIO}
    timeline = {}
    for method in METHODS:
        outcomes, traces = {}, {}
        for version, arms in histories.items():
            row, path, checksum = arms[method]
            require(seasonal.digest(path) == checksum, f"Changed encounter trace: {path}.")
            trace = read_trace(path)
            outcomes[version] = outcome(row, trace)
            traces[version] = trace
        previous = traces["robust_only"]["controls"]
        new = traces["anticipatory"]["controls"]
        first = first_control_change(previous, new)
        first_bow_change = next((
            {"elapsed_s": altered["elapsed_s"], "prior_bow_deg": incumbent["bow_deg"],
             "new_bow_deg": altered["bow_deg"]}
            for incumbent, altered in zip(previous, new)
            if not isclose(incumbent["bow_deg"], altered["bow_deg"], rel_tol=0., abs_tol=1e-9)
        ), None)
        previous_stop = outcomes["robust_only"]["elapsed_s"]
        audit = indexed[(method, TARGET_SCENARIO)]
        timeline[method] = {
            "outcomes": outcomes,
            "first_bow_target_difference_from_robust_only": first_bow_change,
            "first_control_difference_from_robust_only": first,
            "sailed_past_prior_guard_stop": audit["replayed_elapsed_s"] > previous_stop + 1e-7,
            "first_control_changed_before_prior_guard_stop":
                bool(first and first["kind"] == "changed_command" and first["elapsed_s"] < previous_stop),
            "completion_recovered_with_positive_full_clearance":
                outcomes["anticipatory"]["completed"] and not outcomes["robust_only"]["completed"]
                and clearance_flag(audit) == "clear",
            "first_intervention_causal_replay": replay_first_intervention(
                scenarios[TARGET_SCENARIO], traces["anticipatory"],
            ),
            "independent_swept_minima": audit["minimum_clearance"],
            "applied_commands_around_prior_guard_stop": [
                control_snapshot(control) for control in new
                if abs(control["elapsed_s"] - previous_stop) <= 10.
            ],
        }
    return timeline


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
    seasonal.init(MANIFEST, None, False)
    old_rejection = historical_rejection(scenarios)
    jobs = list(source_jobs(args.native, scenarios))
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                             initializer=seasonal.init, initargs=(MANIFEST, None, False)) as pool:
        results = list(pool.map(audit_episode, jobs, chunksize=1))
    indexed = {(result["method"], result["scenario"]): result for result in results}
    pairs = {name: {method: clearance_flag(indexed[method, name]) for method in METHODS}
             for name in scenarios}
    unsafe = [{key: row[key] for key in ("scenario", "method", "nonpositive_control_intervals")}
              for row in results if any(row["nonpositive_control_intervals"].values())]
    seasonal.save_json(args.output, {
        "status": "failed_clearance" if unsafe else "verified",
        "scope": "Held-forcing Fossen re-integration at 0.05 s, analytical targets, independently derived "
                 "report ages and dropout history, both original full clearance envelopes, the expanded "
                 "guard, and a first-intervention replay without target truth or later observations.",
        "manifest_sha256": seasonal.digest(MANIFEST),
        "auditor_sha256": seasonal.digest(__file__),
        "cases_per_arm": len(scenarios),
        "applied_controls": sum(row["applied_controls"] for row in results),
        "integration_substeps": sum(row["integration_substeps"] for row in results),
        "historical_breach_replay": old_rejection,
        "brindisi_003_overtake": encounter_timeline(scenarios, jobs, indexed),
        "paired_clearance_flags": pairs,
        "unsafe_applied_cases": unsafe,
        "cases": results,
    })
    require(not unsafe, f"Clearance or guard violations found. See {args.output}.")
    print(f"Verified {len(results)} anticipatory traces and the full AIS sweeps. {args.output}", flush=True)


if __name__ == "__main__":
    main()
