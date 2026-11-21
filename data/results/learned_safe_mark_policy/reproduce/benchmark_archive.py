"""Compare mark policies on paired, independently started archived GPX legs."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from ais_scenarios import ReducedAISPreview, available_cases
from environment import MAX_SECONDS, TIMEOUT_COST_SECONDS, active_forcing, enforce_first_entry, load_scenarios
from world.learned_mark_policy import FEATURE_NAMES, LearnedMarkPolicy, candidate_batch
from world.synthetic_ais import SyntheticAIS


STUDY = Path(__file__).resolve().parents[1]
METHODS = ("physics", "supervised", "fitted_q", "grpo")
PHYSICS_FEATURE = FEATURE_NAMES.index("physics_cost_h")
LANE_OFFSETS = {999000101: (140.0, 0.0), 999000102: (0.0, 230.0), 999000103: (0.0, -230.0)}
MODELS = {}
MANIFEST = {}


def initialize():
    global MODELS, MANIFEST
    MANIFEST = json.loads((STUDY / "archive_scenario_manifest.json").read_text())
    for method, filename in (
        ("supervised", "supervised_mark_policy.json"), ("fitted_q", "learned_mark_policy.json")
    ):
        MODELS[method] = LearnedMarkPolicy.load(STUDY / filename)
    details = json.loads((STUDY / "grpo_results.json").read_text())
    selected = min(details["GRPO"], key=lambda row: (row["trained_only_validation_objective_s"], row["seed"]))
    with np.load(STUDY / f"grpo_seed_{selected['seed']}.npz") as archive:
        weights = dict(archive)

    def grpo(features):
        hidden = np.tanh(features @ weights["network.0.weight"].T + weights["network.0.bias"])
        residual = np.tanh(hidden @ weights["network.2.weight"].T + weights["network.2.bias"])[..., 0]
        return 3600.0 * features[..., PHYSICS_FEATURE] - 1200.0 * residual

    MODELS["grpo"] = grpo


def apply_models(scenario, active, target, forcing, upcoming, until, dt, batch):
    costs = batch.physics_cost_s.copy()
    support = MODELS["fitted_q"].supported(np.linalg.norm(target, axis=1), scenario.radius,
                                            forcing.polar, upcoming, forcing.wind_speed,
                                            forcing.wave_height, dt, until)
    changed = np.zeros(len(active), dtype=bool)
    for index, policy_index in enumerate(active):
        method = METHODS[policy_index]
        if method == "physics" or not support[index]:
            continue
        model = MODELS[method] if method == "grpo" else MODELS[method].costs
        prediction = model(batch.features[index])
        if np.isfinite(prediction).all():
            costs[index] = prediction
            changed[index] = True
    return costs, changed


def episode(scenario, traffic, fixture_geometry, planning_reserve, stale_report_guard):
    leg_id = MANIFEST["rows"][scenario.identifier]["leg"]
    leg = MANIFEST["leg_blocks"][leg_id]
    group = MANIFEST["groups"][leg["group"]]
    results = [
        {"scenario": scenario.identifier, "route": group["gpx"], "block": group["archive_block"],
         "leg": leg["leg_index"], "origin_waypoint": leg["origin_waypoint"],
         "target_waypoint": leg["target_waypoint"],
         "wind_from_variant": MANIFEST["rows"][scenario.identifier]["wind_variant"],
         "radius_m": scenario.radius, "initial_range_m": leg["initial_leg_distance_m"],
         "target_chart_flag": leg["target_chart_preflight"]["cell_blocked"],
         "traffic": traffic, "fixture_geometry": fixture_geometry, "method": method,
         "additional_planning_reserve_m": planning_reserve,
         "stale_report_guard": stale_report_guard,
         "status": "timeout", "duration_s": MAX_SECONDS,
         "objective_s": MAX_SECONDS + TIMEOUT_COST_SECONDS, "decisions": 0,
         "learning_supported": 0, "ais_interventions": 0, "ais_candidates_screened": 0,
         "min_planned_report_margin_m": None, "min_applied_truth_margin_m": None,
         "min_applied_report_margin_m": None, "max_report_age_s": None, "dropped_reports": 0}
        for method in METHODS
    ]
    if leg["target_chart_preflight"]["complete_capture_disk_in_blocked_cell"]:
        for row in results:
            row.update(status="chart_preflight_blocked", duration_s=0.0)
        return results
    ais = None
    if traffic != "clear":
        bearing = np.degrees(np.arctan2(*scenario.target)) % 360.0
        fixture = traffic if traffic in SyntheticAIS.available_scenarios() else "mixed"
        offsets = {ship.mmsi: LANE_OFFSETS[ship.mmsi] for ship in SyntheticAIS._SCENARIOS[fixture]}
        fixture_options = {"track_offsets_EN": offsets} if fixture_geometry == "lanes" else {}
        ais = ReducedAISPreview(traffic, bearing, origin_latitude=leg["origin_lat_lon"][0],
                                origin_longitude=leg["origin_lat_lon"][1], **fixture_options)
    target = np.repeat(scenario.target[None, :], len(METHODS), axis=0)
    bow = np.full(len(METHODS), scenario.bow)
    active = np.arange(len(METHODS))
    elapsed = 0.0
    while len(active) and elapsed < MAX_SECONDS:
        forcing, future, until = active_forcing(scenario, elapsed)
        interval = 60.0 if elapsed < 900.0 else 300.0
        required_reserve = planning_reserve
        if stale_report_guard and ais is not None:
            age = ais.report_diagnostics(elapsed)["maximum_report_age_seconds"]
            if age > 10.0:
                # Replan at the original report cadence. Also budget possible
                # motion of the actual contact away from its reported course.
                interval = 5.0
                required_reserve += .5 * .030 * (age + interval)**2
        dt = min(interval, MAX_SECONDS - elapsed, until if future is not None else MAX_SECONDS)
        batch = candidate_batch(target[active], bow[active], scenario.radius, forcing.polar, future,
                                forcing.wind_speed, forcing.wave_height, dt, until, MAX_SECONDS - elapsed)
        assessment = None
        allowed = np.ones_like(batch.captured_after_s, dtype=bool)
        if ais is not None:
            assessment = ais.assess(elapsed, scenario.target - target[active], batch, dt)
            allowed &= assessment.report_margin_m > required_reserve
        scores, supported = apply_models(scenario, active, target[active], forcing, future, until, dt, batch)
        unmasked = np.argmin(enforce_first_entry(scores, batch, dt, np.ones_like(allowed)), axis=1)
        selected = np.argmin(enforce_first_entry(scores, batch, dt, allowed), axis=1)
        traveling = []
        for i, index in enumerate(active):
            row = results[index]
            row["decisions"] += 1
            row["learning_supported"] += int(supported[i])
            row["ais_candidates_screened"] += int((~allowed[i]).sum())
            if assessment is not None:
                row["max_report_age_s"] = max(row["max_report_age_s"] or 0.0,
                                              assessment.maximum_report_age_seconds)
                row["dropped_reports"] = max(row["dropped_reports"], assessment.dropped_reports_at_observation)
            if not allowed[i].any():
                row.update(status="ais_blocked", duration_s=elapsed)
                continue
            pick = selected[i]
            row["ais_interventions"] += int(unmasked[i] != pick)
            hit = float(batch.captured_after_s[i, pick])
            if assessment is not None:
                report = assessment.report_margin_m[i, pick]
                truth = assessment.truth_margin_m[i, pick]
                keep_min(row, "min_planned_report_margin_m", report)
                if np.isfinite(hit):
                    applied = ais.assess_applied(elapsed, scenario.target - target[index], batch, dt, i, pick, hit)
                    report, truth = applied
                keep_min(row, "min_applied_report_margin_m", report)
                keep_min(row, "min_applied_truth_margin_m", truth)
                if truth <= 0.0:
                    row.update(status="truth_clearance_unverified", duration_s=elapsed + min(dt, hit))
                    continue
            if np.isfinite(hit):
                row.update(status="captured", duration_s=elapsed + hit, objective_s=elapsed + hit)
                continue
            target[index] = batch.remaining_target[i, pick]
            bow[index] = batch.end_bow[i, pick]
            traveling.append(index)
        active = np.asarray(traveling, dtype=int)
        elapsed += dt
    return results


def keep_min(row, name, value):
    if np.isfinite(value):
        prior = row[name]
        row[name] = float(value) if prior is None else min(float(value), prior)


def run_job(task):
    return episode(*task)


def summarize(rows):
    groups = defaultdict(list)
    by_case = defaultdict(dict)
    for row in rows:
        groups[(row["traffic"], row["method"])].append(row)
        groups[("all", row["method"])].append(row)
        by_case[(row["scenario"], row["traffic"])][row["method"]] = row
    summary = []
    for (traffic, method), group in groups.items():
        hits = [row["duration_s"] for row in group if row["status"] == "captured"]
        counts = {key: sum(row["status"] == key for row in group) for key in sorted({row["status"] for row in group})}
        pairs = [(row, by_case[(row["scenario"], row["traffic"])]["physics"]) for row in group]
        paired = [a["duration_s"] - b["duration_s"] for a, b in pairs if a["status"] == b["status"] == "captured"]
        margins = [row["min_applied_truth_margin_m"] for row in group if row["min_applied_truth_margin_m"] is not None]
        report_margins = [row["min_applied_report_margin_m"] for row in group
                          if row["min_applied_report_margin_m"] is not None]
        ages = [row["max_report_age_s"] for row in group if row["max_report_age_s"] is not None]
        summary.append({"traffic": traffic, "method": method, "episodes": len(group), **counts,
                        "median_capture_s": float(np.median(hits)) if hits else None,
                        "mean_capped_objective_s": float(np.mean([row["objective_s"] for row in group])),
                        "joint_captures_vs_physics": len(paired),
                        "unique_captures_vs_physics": sum(a["status"] == "captured" and b["status"] != "captured"
                                                          for a, b in pairs),
                        "physics_only_captures": sum(b["status"] == "captured" and a["status"] != "captured"
                                                     for a, b in pairs),
                        "median_paired_delta_s": float(np.median(paired)) if paired else None,
                        "faster_joint_captures": sum(delta < -1e-6 for delta in paired),
                        "slower_joint_captures": sum(delta > 1e-6 for delta in paired),
                        "learning_supported_episodes": sum(row["learning_supported"] > 0 for row in group),
                        "learning_supported_decisions": sum(row["learning_supported"] for row in group),
                        "all_decisions": sum(row["decisions"] for row in group),
                        "ais_interventions": sum(row["ais_interventions"] for row in group),
                        "ais_candidates_screened": sum(row["ais_candidates_screened"] for row in group),
                        "min_applied_report_margin_m": min(report_margins, default=None),
                        "min_applied_truth_margin_m": min(margins, default=None),
                        "max_report_age_s": max(ages, default=None),
                        "episodes_observing_drops": sum(row["dropped_reports"] > 0 for row in group)})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--fixture-geometry", choices=("lanes", "native"), default="lanes")
    parser.add_argument("--additional-planning-reserve", type=float, default=0.0)
    parser.add_argument("--stale-report-guard", action="store_true")
    parser.add_argument("--traffic", nargs="+", choices=("clear", *available_cases()))
    arguments = parser.parse_args()
    if not np.isfinite(arguments.additional_planning_reserve) or arguments.additional_planning_reserve < 0.0:
        parser.error("Additional planning reserve must be finite and nonnegative.")
    scenarios = load_scenarios(STUDY / "archive_scenarios.json.gz")
    cases = ("clear", *available_cases()) if arguments.traffic is None else tuple(arguments.traffic)
    tasks = [(scenario, traffic, arguments.fixture_geometry,
              arguments.additional_planning_reserve, arguments.stale_report_guard)
             for scenario in scenarios[:arguments.case_limit] for traffic in cases]
    records = []
    if arguments.workers == 1:
        initialize()
        for task in tasks:
            records.extend(run_job(task))
    else:
        with ProcessPoolExecutor(max_workers=arguments.workers, mp_context=mp.get_context("spawn"),
                                 initializer=initialize) as executor:
            for index, result in enumerate(executor.map(run_job, tasks, chunksize=4), start=1):
                records.extend(result)
                if index % 100 == 0:
                    print(f"{index}/{len(tasks)} paired cases", flush=True)
    suffix = "_tight_stress" if arguments.fixture_geometry == "native" else ""
    if arguments.additional_planning_reserve:
        suffix += f"_extra_{arguments.additional_planning_reserve:g}m"
    if arguments.stale_report_guard:
        suffix += "_stale_guard"
    if arguments.case_limit is not None or arguments.traffic:
        suffix += "_smoke"
    destination = STUDY / f"archive_benchmark{suffix}.csv"
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metadata = {
        "archive_scenario_sha256": hashlib.sha256((STUDY / "archive_scenarios.json.gz").read_bytes()).hexdigest(),
        "ais_adapter_sha256": hashlib.sha256(Path(__file__).with_name("ais_scenarios.py").read_bytes()).hexdigest(),
        "coverage": {"route_files": len({row["route"] for row in records}),
                     "paired_scenarios": len(tasks), "archive_weather_cases": len({row["scenario"] for row in records}),
                     "policy_episodes": len(records), "traffic_cases": list(cases), "methods": list(METHODS)},
        "chart": "Existing GEBCO target disk preflight rejects entirely blocked marks. Other chart paths are unscreened.",
        "scope": "Independent six-hour reduced Fossen-equilibrium legs, using two archived weather snapshots per case.",
        "timing": "Common 60 s intervals for initial 900 s, followed by 300 s intervals in all treatments.",
        "ais": "Finite available actions screened over the entire proposed interval against reports. "
               "Truth margins audit only the applied motion, including exact waypoint entry. "
               "If no reduced action is admissible the episode stops. Native COLREG and fallback are separate.",
        "fixture_geometry": arguments.fixture_geometry,
        "additional_planning_reserve_m": arguments.additional_planning_reserve,
        "stale_report_guard": arguments.stale_report_guard,
        "track_offsets_in_fixture_east_aligned_frame_m": LANE_OFFSETS if arguments.fixture_geometry == "lanes" else {},
        "objective": "Time to first entry in seconds, or a fixed 43200 s for an uncaptured case.",
        "grpo_checkpoint": "The best post-update seed/checkpoint on synthetic validation, used for diagnostic transfer. "
                           "The main validation selection includes the physics initialization.",
        "methods": summarize(records),
    }
    (STUDY / f"archive_benchmark{suffix}_summary.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metadata["coverage"]), flush=True)


if __name__ == "__main__":
    main()
