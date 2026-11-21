"""Independently replay the stale-AIS retest at the native 0.05 s step."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np


STUDY = Path(__file__).resolve().parents[1]
PREVIOUS = STUDY.with_name("grpo_progress_native")
sys.path.insert(0, str(PREVIOUS / "reproduce"))

from recheck_clearance import (
    ENDPOINT_TOLERANCE_M,
    MANIFEST,
    METHODS,
    STATE_KEYS,
    STATE_TOLERANCE,
    clearance_flag,
    course_velocity,
    require,
    seasonal,
    source_jobs,
    swept_margin,
    truth_track,
    verified_value,
    xy,
)
from world.ais_avoidance import DEFAULT_AIS_SAFETY_BUFFER_METRES, _OWN_HULL_RADIUS_METRES
from world.boat import BoatAction, BoatState


HISTORICAL_SCENARIO = "autumn__20261013T1800Z__brindisi_003__test_raking_overtake"
HISTORICAL_CONTROL_START_S = 165.
GUARD_GEOMETRY = "combined hulls + 25 m gap + 3 m reserve + twice the full report uncertainty"


def observation_geometry(observation, fixture, times, origin, config):
    """Rebuild constant-course report tracks and radii without calling the guard."""
    reports = observation.ais_targets
    target_ids = [target.mmsi for target in fixture.targets]
    by_id = {report.mmsi: report for report in reports}
    require(len(by_id) == len(reports) and by_id.keys() == set(target_ids) and bool(reports),
            "Locked fixture and delivered-report identifiers differ.")
    reports = [by_id[mmsi] for mmsi in target_ids]
    courses = np.array([report.course for report in reports], dtype=float)
    speeds = np.array([report.speed for report in reports], dtype=float)
    ages = np.array([report.report_age_seconds for report in reports], dtype=float)
    lengths = np.array([report.length_metres for report in reports], dtype=float)
    beams = np.array([report.beam_metres for report in reports], dtype=float)
    require(np.isfinite(np.stack((courses, speeds, ages, lengths, beams))).all() and
            np.all((courses >= 0.) & (courses < 360.)) and np.all(speeds >= 0.) and
            np.all(ages >= 0.) and np.all(lengths > 0.) and np.all(beams > 0.),
            "Independent guard audit requires the complete, valid locked reports.")
    velocity = course_velocity(courses, speeds)
    position = xy([report.latitude for report in reports], [report.longitude for report in reports], origin)
    projected_age = np.minimum(ages, 150.)
    center = position + (projected_age + times[:, None])[..., None] * velocity
    age_at_knot = ages + times[:, None]
    motion = np.maximum(.05 * age_at_knot, .5 * config.acceleration_bound_mps2 * age_at_knot**2)
    uncertainty = 5. + motion + (ages - projected_age) * speeds
    fixed_radius = (
        _OWN_HULL_RADIUS_METRES + .5 * np.hypot(lengths, beams)
        + DEFAULT_AIS_SAFETY_BUFFER_METRES + config.reserve_m
    )
    return center, fixed_radius, uncertainty, float(ages.max()), target_ids


def replay_control(engine, state, start, elapsed, control, recorded):
    step = control["duration_s"]
    knots = control["native_substeps"]
    dt = engine.config.integration_dt_s
    when = start + np.timedelta64(int(round(elapsed * 1e9)), "ns")
    require(knots > 0 and abs(knots * dt - step) < 1e-8 and control["held_observation"] is True,
            "Invalid applied duration or forcing sample.")
    require(abs(control["elapsed_s"] - elapsed) < 1e-6 and
            np.datetime64(control["utc"].removesuffix("Z"), "ns") == when and
            abs(recorded["elapsed_s"] - elapsed - step) < 1e-6, "Applied clocks disagree.")

    observation = engine.environment.observe(state, when)
    action = BoatAction(control["rudder_deg"], control["flap_target_deg"])
    final, path, _ = engine.integrate(state, action, observation, step)
    require(len(path) == knots + 1 and np.isfinite(path).all(), "Invalid re-integrated path.")
    endpoint_error = float(np.linalg.norm(xy(final.latitude, final.longitude,
                                              (recorded["latitude"], recorded["longitude"]))))
    state_error = max(abs(getattr(final, key) - recorded[key]) for key in STATE_KEYS[2:])
    require(endpoint_error <= ENDPOINT_TOLERANCE_M and state_error <= STATE_TOLERANCE,
            f"Re-integrated endpoint mismatch ({endpoint_error} m, state error {state_error}).")

    times = np.arange(knots + 1) * dt
    origin = (state.latitude, state.longitude)
    own = xy(path[:, 0], path[:, 1], origin)
    report, fixed_radius, uncertainty, age, ids = observation_geometry(
        observation, engine.environment, times, origin, engine.config,
    )
    require(abs(age - control["max_report_age_s"]) < 1e-8, "Replayed held report age differs.")
    truth = truth_track(engine.environment, elapsed + times, origin)
    centers_and_radii = {
        "report": (report, fixed_radius + uncertainty),
        "truth": (truth, fixed_radius + uncertainty),
        "guard": (report, fixed_radius + 2. * uncertainty),
    }
    margins = {}
    for kind, (center, radii) in centers_and_radii.items():
        margin, offset, index = swept_margin(
            times, center - own[:, None, :], radii, conservative=kind != "report",
        )
        margins[kind] = {"margin_m": margin, "elapsed_s": elapsed + offset, "mmsi": ids[index],
                         "held_report_age_at_step_start_s": age}
    approximation_slack = float(np.min(uncertainty - np.linalg.norm(truth - report, axis=-1)))
    require(approximation_slack >= -1e-7, "Analytical contact position escaped the report uncertainty bound.")
    require(margins["truth"]["margin_m"] >= margins["guard"]["margin_m"] - 1e-7,
            "Guard bound failed to lower-bound the true swept margin.")
    return margins, endpoint_error, state_error, approximation_slack


def audit_job(job):
    spec, row, trace_path, checksum = job
    require(seasonal.digest(trace_path) == checksum, f"Trace checksum changed: {trace_path}")
    with gzip.open(trace_path, "rt", encoding="utf-8") as stream:
        trace = json.load(stream)
    engine, state, _, start, _ = seasonal.scenario_parts(spec)
    controls, states = trace["controls"], trace["states"]
    require(len(controls) == len(states), f"Inconsistent control and state counts: {trace_path}")
    minima = {kind: None for kind in ("report", "truth", "guard")}
    nonpositive = dict.fromkeys(minima, 0)
    margin_errors = dict.fromkeys(("report", "truth"), 0.)
    elapsed = endpoint_error = state_error = 0.
    uncertainty_slack = None
    for control, recorded in zip(controls, states, strict=True):
        context = f"{row['method']}/{spec['id']} at {elapsed:g} s"
        try:
            margins, position_error, velocity_error, slack = replay_control(
                engine, state, start, elapsed, control, recorded,
            )
        except ValueError as error:
            raise ValueError(f"{context}: {error}") from error
        endpoint_error = max(endpoint_error, position_error)
        state_error = max(state_error, velocity_error)
        uncertainty_slack = slack if uncertainty_slack is None else min(uncertainty_slack, slack)
        for kind, check in margins.items():
            margin = check["margin_m"]
            if kind in margin_errors:
                error = verified_value(margin, control[f"{kind}_minimum_m"], f"{context} {kind}")
                margin_errors[kind] = max(margin_errors[kind], error)
            nonpositive[kind] += margin <= 0.
            if minima[kind] is None or margin < minima[kind]["margin_m"]:
                minima[kind] = check
        state = BoatState(**{key: recorded[key] for key in STATE_KEYS})
        elapsed = recorded["elapsed_s"]
    for kind in margin_errors:
        saved = row[f"min_{kind}_margin_m"]
        check = minima[kind]
        if check is None:
            require(not saved, f"Summary clearance without an applied path: {trace_path}")
        else:
            verified_value(check["margin_m"], float(saved), f"{trace_path} aggregate {kind}")
    return {"scenario": spec["id"], "method": row["method"], "applied_controls": len(controls),
            "integration_substeps": sum(control["native_substeps"] for control in controls),
            "replayed_elapsed_s": elapsed, "trace_sha256": checksum,
            "maximum_endpoint_error_m": endpoint_error, "maximum_other_state_error": state_error,
            "maximum_clearance_error_m": margin_errors, "nonpositive_control_intervals": nonpositive,
            "minimum_center_uncertainty_slack_m": uncertainty_slack, "minimum_clearance": minima}


def historical_rejection(scenarios):
    """Require the new geometry to reject the old accepted 165–170 s motion."""
    old = PREVIOUS / "native" / "physics" / "telemetry"
    path = old / f"{HISTORICAL_SCENARIO}__physics.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        trace = json.load(stream)
    engine, _, _, start, _ = seasonal.scenario_parts(scenarios[HISTORICAL_SCENARIO])
    index, = (index for index, step in enumerate(trace["controls"])
               if step["elapsed_s"] == HISTORICAL_CONTROL_START_S)
    before = trace["states"][index - 1]
    state = BoatState(**{key: before[key] for key in STATE_KEYS})
    control = trace["controls"][index]
    margins, _, _, slack = replay_control(
        engine, state, start, HISTORICAL_CONTROL_START_S, control, trace["states"][index],
    )
    for kind in ("report", "truth"):
        verified_value(margins[kind]["margin_m"], control[f"{kind}_minimum_m"], f"historical {kind}")
    require(margins["report"]["margin_m"] > 0. > margins["truth"]["margin_m"] >= margins["guard"]["margin_m"],
            "Historical regression failed to reproduce the old false clearance and new rejection.")
    return {"scenario": HISTORICAL_SCENARIO, "trace_sha256": seasonal.digest(path),
            "control_start_s": HISTORICAL_CONTROL_START_S,
            "control_end_s": HISTORICAL_CONTROL_START_S + control["duration_s"],
            "minimum_center_uncertainty_slack_m": slack, "would_be_rejected": True, "margins": margins}


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
        results = list(pool.map(audit_job, jobs, chunksize=1))
    indexed = {(result["method"], result["scenario"]): result for result in results}
    pairs = {name: {method: clearance_flag(indexed[method, name]) for method in METHODS}
             for name in scenarios}
    unsafe = [{key: row[key] for key in ("scenario", "method", "nonpositive_control_intervals")}
              for row in results if any(row["nonpositive_control_intervals"].values())]
    seasonal.save_json(args.output, {
        "status": "failed_clearance" if unsafe else "verified",
        "scope": "Held-forcing Fossen re-integration at 0.05 s, analytical targets, separately derived radii "
                 "and synchronized sweep around both original centers and the expanded report guard.",
        "guard_geometry": GUARD_GEOMETRY,
        "manifest_sha256": seasonal.digest(MANIFEST),
        "auditor_sha256": seasonal.digest(__file__),
        "cases_per_arm": len(scenarios),
        "applied_controls": sum(row["applied_controls"] for row in results),
        "integration_substeps": sum(row["integration_substeps"] for row in results),
        "historical_breach_replay": old_rejection,
        "paired_clearance_flags": pairs,
        "unsafe_applied_cases": unsafe,
        "cases": results,
    })
    require(not unsafe, f"Clearance or guard violations found. See {args.output}.")
    print(f"Verified {len(results)} traces, both full AIS margins and the robust applied guard. {args.output}",
          flush=True)


if __name__ == "__main__":
    main()
