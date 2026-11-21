"""Replay every applied native control and independently check both AIS margins."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import fields
import gzip
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np


STUDY = Path(__file__).resolve().parents[1]
SEASONAL = STUDY.with_name("seasonal_native_training")
MANIFEST = SEASONAL / "manifest_expanded.json"
sys.path.insert(0, str(SEASONAL / "reproduce"))

import run_seasonal as seasonal
from world._geo import local_east_north_metres, offset_latitude_longitude
from world.ais_avoidance import DEFAULT_AIS_SAFETY_BUFFER_METRES, _OWN_HULL_RADIUS_METRES
from world.boat import BoatAction, BoatState


METHODS = ("physics", "grpo_progress")
MARGIN_TOLERANCE_M = 2e-6
ENDPOINT_TOLERANCE_M = 1e-5
STATE_TOLERANCE = 1e-8
STATE_KEYS = tuple(field.name for field in fields(BoatState))


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def xy(latitude, longitude, origin):
    return np.stack(local_east_north_metres(latitude, longitude, *origin), axis=-1)


def swept_margin(times, relative, radius, *, conservative):
    """Minimize signed separation along each synchronized linear chord.

    The report sweep interpolates radius. The truth sweep uses the larger
    endpoint radius over each chord, matching the study's conservative bound.
    """
    start = relative[:-1]
    steps = np.diff(relative, axis=0)
    travel_squared = np.sum(steps**2, axis=-1)
    center_fraction = np.divide(-np.sum(start * steps, axis=-1), travel_squared,
                                out=np.zeros_like(travel_squared), where=travel_squared > 0.)
    if conservative:
        radii = np.maximum(radius[:-1], radius[1:])
        growth = np.zeros_like(radii)
        fraction = np.clip(center_fraction, 0., 1.)
    else:
        radii = radius[:-1]
        growth = np.diff(radius, axis=0)
        perpendicular = np.linalg.norm(start + center_fraction[..., None] * steps, axis=-1)
        has_stationary_point = travel_squared > growth**2
        denominator = np.sqrt(travel_squared * np.maximum(0., travel_squared - growth**2))
        adjustment = np.divide(growth * perpendicular, denominator, out=np.zeros_like(growth),
                               where=has_stationary_point)
        fraction = np.where(has_stationary_point, np.clip(center_fraction + adjustment, 0., 1.), growth > 0.)
    margins = np.linalg.norm(start + fraction[..., None] * steps, axis=-1) - radii - fraction * growth
    knot, target = np.unravel_index(np.argmin(margins), margins.shape)
    elapsed = times[knot] + fraction[knot, target] * (times[knot + 1] - times[knot])
    return float(margins[knot, target]), float(elapsed), int(target)


def course_velocity(course_deg, speed):
    course = np.radians(course_deg)
    return np.column_stack((np.sin(course), np.cos(course))) * np.asarray(speed)[:, None]


def truth_track(environment, times, origin):
    """Integrate the fixture's velocity ramps analytically at audit substeps."""
    targets = environment.targets
    initial = np.array([(target.initial_east_metres, target.initial_north_metres) for target in targets])
    start_speed = course_velocity([target.course_degrees for target in targets],
                                  [target.speed_metres_per_second for target in targets])
    points = initial + times[:, None, None] * start_speed
    for index, ramp in enumerate(environment.manoeuvres):
        if ramp is None:
            continue
        duration = ramp.finish_seconds - ramp.start_seconds
        turn_time = np.clip(times - ramp.start_seconds, 0., duration)
        after_turn = np.maximum(0., times - ramp.finish_seconds)
        final_speed = course_velocity([ramp.course_degrees], [ramp.speed_metres_per_second])[0]
        correction = turn_time**2 / (2. * duration) + after_turn
        points[:, index] += correction[:, None] * (final_speed - start_speed[index])
    latitude, longitude = offset_latitude_longitude(*environment.origin, points[..., 0], points[..., 1])
    return xy(latitude, longitude, origin)


def truth_radii(engine, prediction, radii, times):
    report_index = {target.mmsi: index for index, target in enumerate(prediction.targets)}
    columns = []
    growth = np.maximum(.05 * times, .5 * engine.config.acceleration_bound_mps2 * times**2)
    for target in engine.environment.targets:
        index = report_index.get(target.mmsi)
        if index is None:
            hull = .5 * np.hypot(target.length_metres, target.beam_metres) + _OWN_HULL_RADIUS_METRES
            columns.append(hull + DEFAULT_AIS_SAFETY_BUFFER_METRES + 5. + engine.config.reserve_m + growth)
        else:
            columns.append(radii[:, index])
    return np.column_stack(columns)


def verified_value(calculated, recorded, context):
    require(np.isfinite(calculated) and recorded is not None and np.isfinite(recorded),
            f"{context}: missing or nonfinite clearance.")
    error = abs(calculated - recorded)
    require(error <= MARGIN_TOLERANCE_M and (calculated > 0.) == (recorded > 0.),
            f"{context}: replay margin {calculated:.10f} m, recorded {recorded:.10f} m.")
    return error


def audit_job(data):
    spec, row, trace, checksum = data
    context = f"{row['method']}/{spec['id']}"
    require(seasonal.digest(trace) == checksum, f"{context}: trace checksum mismatch.")
    with gzip.open(trace, "rt", encoding="utf-8") as stream:
        replay = json.load(stream)
    engine, state, _, start, _ = seasonal.scenario_parts(spec)
    controls, states = replay["controls"], replay["states"]
    require(len(controls) == len(states), f"{context}: inconsistent control and state counts.")
    minima = {"report": None, "truth": None}
    nonpositive = {"report": 0, "truth": 0}
    margin_error = {"report": 0., "truth": 0.}
    elapsed = endpoint_error = state_error = 0.
    knots = 0
    for control, recorded in zip(controls, states, strict=True):
        step = control["duration_s"]
        n = control["native_substeps"]
        dt = engine.config.integration_dt_s
        when = start + np.timedelta64(int(round(elapsed * 1e9)), "ns")
        require(n > 0 and abs(n * dt - step) < 1e-8 and control.get("held_observation") is True,
                f"{context} {elapsed}: invalid applied duration or forcing sample.")
        require(abs(control["elapsed_s"] - elapsed) < 1e-6 and
                np.datetime64(control["utc"].removesuffix("Z"), "ns") == when and
                abs(recorded["elapsed_s"] - elapsed - step) < 1e-6,
                f"{context} {elapsed}: applied clocks disagree.")
        before = (state.latitude, state.longitude)
        observation = engine.environment.observe(state, when)
        action = BoatAction(control["rudder_deg"], control["flap_target_deg"])
        final, path, _ = engine.integrate(state, action, observation, step)
        require(len(path) == n + 1 and np.isfinite(path).all(), f"{context} {elapsed}: invalid re-integrated path.")
        error = float(np.linalg.norm(xy(final.latitude, final.longitude,
                                         (recorded["latitude"], recorded["longitude"]))))
        other_error = max(abs(getattr(final, key) - recorded[key]) for key in STATE_KEYS[2:])
        require(error <= ENDPOINT_TOLERANCE_M and other_error <= STATE_TOLERANCE,
                f"{context} {elapsed}: re-integrated endpoint mismatch ({error} m, state error {other_error}).")
        endpoint_error, state_error = max(endpoint_error, error), max(state_error, other_error)
        times = np.arange(n + 1) * dt
        own = xy(path[:, 0], path[:, 1], before)
        prediction = engine._prediction(observation)
        radii = prediction.radii_at(times)
        reported = prediction.centres_at(times)
        truth = truth_track(engine.environment, elapsed + times, before)
        bounds = {
            "report": (reported, radii, prediction.targets),
            "truth": (truth, truth_radii(engine, prediction, radii, times), engine.environment.targets),
        }
        for kind, (centers, envelope, targets) in bounds.items():
            require(bool(targets), f"{context} {elapsed}: no {kind} targets in a locked traffic fixture.")
            margin, offset, target = swept_margin(times, centers - own[:, None, :], envelope,
                                                   conservative=kind == "truth")
            error = verified_value(margin, control[f"{kind}_minimum_m"], f"{context} {elapsed} {kind}")
            margin_error[kind] = max(margin_error[kind], error)
            nonpositive[kind] += margin <= 0.
            if minima[kind] is None or margin < minima[kind]["margin_m"]:
                minima[kind] = {"margin_m": margin, "elapsed_s": elapsed + offset, "mmsi": targets[target].mmsi,
                                "held_report_age_at_step_start_s": control["max_report_age_s"]}
        # Start the next replay chord at the corresponding authenticated
        # endpoint to avoid accumulating floating-point roundoff across hours.
        state = BoatState(**{key: recorded[key] for key in STATE_KEYS})
        elapsed = recorded["elapsed_s"]
        knots += n
    for kind, minimum in minima.items():
        saved = row[f"min_{kind}_margin_m"]
        if minimum is None:
            require(not saved, f"{context}: summary clearance without an applied path.")
        else:
            verified_value(minimum["margin_m"], float(saved), f"{context} aggregate {kind}")
    return {"scenario": spec["id"], "method": row["method"], "applied_controls": len(controls),
            "integration_substeps": knots, "replayed_elapsed_s": elapsed, "trace_sha256": checksum,
            "maximum_endpoint_error_m": endpoint_error, "maximum_other_state_error": state_error,
            "maximum_clearance_error_m": margin_error, "nonpositive_control_intervals": nonpositive,
            "minimum_clearance": minima}


def source_jobs(native, scenarios):
    for method in METHODS:
        directory = native / method
        episodes = directory / "episodes.csv"
        run = json.loads((directory / "run.json").read_text())
        require(run["method"] == method and run["episodes_sha256"] == seasonal.digest(episodes),
                f"{method}: wrong input receipt or episode checksum.")
        with episodes.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        require(len(rows) == len(scenarios) and {row["scenario"] for row in rows} == set(scenarios),
                f"{method}: recorded routes are not the complete locked held-out set.")
        for row in rows:
            name = f"telemetry/{row['scenario']}__{method}.json.gz"
            require(row["method"] == method and row["trajectory"] == name,
                    f"{method}/{row['scenario']}: noncanonical trace.")
            yield scenarios[row["scenario"]], row, directory / name, run["trace_sha256es"][name]


def clearance_flag(audit):
    minima = audit["minimum_clearance"].values()
    if any(item is not None and item["margin_m"] <= 0. for item in minima):
        return "violated"
    return "clear" if all(item is not None for item in minima) else "no_applied_path"


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
    jobs = list(source_jobs(args.native, scenarios))
    context = multiprocessing.get_context("spawn")
    args_init = (MANIFEST, None, False)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                             initializer=seasonal.init, initargs=args_init) as pool:
        results = list(pool.map(audit_job, jobs, chunksize=1))
    indexed = {(result["method"], result["scenario"]): result for result in results}
    pairs = {name: {method: clearance_flag(indexed[method, name]) for method in METHODS}
             for name in scenarios}
    seasonal.save_json(args.output, {
        "status": "verified",
        "scope": "Full replay of applied Fossen controls at 0.05 s; analytical fixture truth and delivered-AIS "
                 "projection swept over synchronized chords with each full uncertainty envelope plus reserve.",
        "manifest_sha256": seasonal.digest(MANIFEST),
        "auditor_sha256": seasonal.digest(__file__),
        "cases_per_arm": len(scenarios),
        "applied_controls": sum(row["applied_controls"] for row in results),
        "integration_substeps": sum(row["integration_substeps"] for row in results),
        "paired_clearance_flags": pairs,
        "cases": results,
    })
    print(f"Replayed {len(results)} episodes and verified both AIS sweeps. {args.output}", flush=True)


if __name__ == "__main__":
    main()
