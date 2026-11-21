"""Audit eight new-traffic weak-wind round trips, capture order and applied arbitration.

The independently reconstructed AIS sweep is consumed, never reproduced by
the controller's own margin routine.  Comparisons are paired within wind
bearing. Cached policy telemetry is counted separately from fresh model
evaluations and from decisions that survived actuator arbitration.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys
from xml.etree import ElementTree


HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[2]
sys.path[:0] = [str(HERE), str(ROOT)]
import audit_route_reference as reference


from launch import ARMS
CLEARANCE_FIELDS = ("truth_minimum_nominal_m", "truth_centered_minimum_uncertainty_m",
                    "configured_truth_centered_minimum_m", "model_minimum_uncertainty_m", "configured_model_minimum_m")


def write_csv(path, rows, fields=None):
    with path.open("w", newline="") as stream:
        fields = fields or list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize(args):
    skip_values = {"--acados-build-dir", "--output-dir", "--output-gpx", "--record-replay", "--temporal-recovery-model"}
    result, i = [], 0
    while i < len(args):
        if args[i] in skip_values:
            i += 2
        elif args[i] == "--confirm-temporal-release":
            i += 1
        else:
            result.append(args[i])
            i += 1
    return result


def verify_protocol(arm, folder, execution):
    assert execution["return_code"] == 0 and execution["source_unchanged"] and not execution["changed_inputs"], arm
    wind = execution["wind_from_degrees"]
    assert wind in (355, 5) and execution["wind_from"] == wind, arm
    assert reference.utc(execution["departure_utc"]) == reference.DEPARTURE, arm
    assert reference.utc(execution["encounter_utc"]) == reference.ENCOUNTER, arm
    assert execution["maximum_steps"] == reference.MAXIMUM_STEPS and execution["cap_s"] == 86400, arm
    assert execution["dt_s"] == 5 and execution["radius_m"] == 50, arm
    method = execution["method"]
    assert method in {"classical", "transformer"} and arm == f"{execution['scenario_id']}__otranto_{wind:03d}__{method}", arm
    learning = method == "transformer"
    log = (folder / "run.log").read_text().splitlines()
    command = reference.one_json(log, "Runner arguments: ")
    native_direction, = [float(line.split(":", 1)[1]) for line in log if line.startswith("Controlled wind FROM degrees:")]
    assert native_direction == wind, arm
    outer = execution["command"]
    assert outer.count("--wind-from") == 1 and float(reference.option(outer, "--wind-from")) == wind, arm
    protocol=json.loads((STUDY/"protocol.json").read_text())
    scenario=execution["scenario_id"]
    assert scenario in protocol["encounters"]
    fixture=STUDY/protocol["encounters"][scenario]["definition"]
    assert Path(reference.option(outer,"--traffic-fixture")).resolve()==fixture.resolve(), arm
    assert Path(execution["traffic_fixture"]).resolve()==fixture.resolve()
    assert reference.digest(fixture)==execution["traffic_fixture_sha256"]==protocol["encounters"][scenario]["definition_sha256"]
    assert reference.digest(STUDY/"protocol.json")==execution["protocol_sha256"]
    assert ("--temporal-recovery-model" in command) == ("--confirm-temporal-release" in command) == learning, arm
    checkpoint = ROOT / "data/results/otranto_temporal_hold_20261118/models/temporal_hold.pt"
    assert reference.digest(checkpoint) == execution["frozen_checkpoint_sha256"], arm
    if learning:
        assert Path(reference.option(command, "--temporal-recovery-model")).resolve() == checkpoint, arm
    assert set(("--forecast-tacks", "--forecast-tack-refinement", "--forecast-tracking-feedback",
                "--forecast-hold-tacks", "--return-to-start", "--limit-upwind-drift", "--ais-scenario",
                "--time-aware-ais", "--ais-maneuvering-sharp", "--colreg")).issubset(command), arm
    for flag, value in {
        "--forecast-tack-step": "60", "--forecast-refinement-phase": "return",
        "--forecast-gate-aim-fraction": "0.5", "--forecast-gate-aim-upwind-angle": "60",
        "--forecast-maneuver-commit-seconds": "60", "--ais-scenario": "mixed",
        "--ais-acceleration-bound": "0.026", "--ais-clearance-reserve": "3",
        "--ais-preview-age-budget": "65", "--ais-launch-lead": "40", "--ais-report-delay": "20",
    }.items():
        assert reference.option(command, flag) == value, (arm, flag)
    assert [[command[i + 1], command[i + 2]] for i, x in enumerate(command)
            if x == "--ais-drop-reports"] == [["85", "125"], ["145", "185"]], arm
    for name, checksum in execution["source_sha256"].items():
        assert reference.digest(ROOT / name) == checksum, (arm, name, "current input changed")
        assert reference.digest(folder / "source_snapshot" / name) == checksum, (arm, name, "snapshot mismatch")
    return command, {k: v for k, v in execution["source_sha256"].items() if k.startswith("world/")}


def audit_arbitration(arm, replay, learning, wind_from):
    counts, source_actions, reasons = Counter(), Counter(), Counter()
    accepted_windows, raw_windows, proposal_events = set(), set(), {}
    transitions, trim_intervals, decisions = [], [], []
    first_learned = first_difference = first_physical_release = previous = open_trim = None
    for frame in replay[1:]:
        nav = frame["navigation"]
        step, start_s = nav["step"], (nav["step"] - 1) * reference.STEP_SECONDS
        source, action = nav.get("temporal_recovery_source"), nav.get("temporal_recovery_applied_action")
        raw, classical = nav.get("temporal_recovery_raw_action"), nav.get("temporal_recovery_classical_action")
        reason, window = nav.get("temporal_recovery_fallback_reason"), nav.get("temporal_recovery_window_utc")
        decision_utc, probability = nav.get("temporal_recovery_decision_utc"), nav.get("temporal_recovery_p_hold")
        active, mode = nav.get("ais_colreg_active_contacts", 0), str(nav.get("ais_mode", "missing"))
        accepted = source in {"learned", "physical release"}
        actual_trim = nav.get("control_source") == "drift trim"
        if accepted:
            assert action in {"hold", "sail"} and active == 0, (arm, step, "COLREG/learned conflict")
            assert mode == "route" and nav.get("solver_fallback") is None, (arm, step, "Uncleared steering override")
            assert nav.get("nominal_guidance_suspended") is None, (arm, step, "Suspended policy marked applied")
            assert (action == "hold") == actual_trim, (arm, step, "Hold was not the applied trim")
            if source == "learned":
                assert raw == action, (arm, step, "Accepted model action differs from raw proposal")
            else:
                assert raw == "hold" and action == classical == "sail", (arm, step, "Invalid physical release provenance")
                assert nav.get("temporal_recovery_release_guard"), (arm, step, "Missing release evidence")
            if first_learned is None:
                first_learned = start_s
            if source == "physical release" and first_physical_release is None:
                first_physical_release = start_s
            if action != classical and first_difference is None:
                first_difference = start_s
            if window:
                accepted_windows.add(window)
        if probability is not None:
            assert learning and raw in {"hold", "sail"} and decision_utc and window, (arm, step)
            assert math.isfinite(probability) and 0 <= probability <= 1, (arm, step)
            assert (probability >= 0.5) == (raw == "hold"), (arm, step, "Raw probability/action mismatch")
            decision_time = reference.utc(decision_utc)
            action_time = reference.DEPARTURE + reference.timedelta(seconds=start_s)
            assert decision_time <= action_time and reference.utc(window) <= decision_time, (arm, step)
            assert decision_time < reference.utc(window) + reference.timedelta(seconds=300), (arm, step, "Decision/window mismatch")
            counts["raw_proposal_telemetry_updates"] += 1
            counts["fresh_model_evaluation_updates"] += decision_time == action_time
            counts["cached_model_telemetry_updates"] += decision_time < action_time
            counts["raw_hold_telemetry_updates"] += raw == "hold"
            raw_windows.add(window)
            key = (decision_utc, window)
            fingerprint = (raw, probability, classical)
            if key not in proposal_events:
                proposal_events[key] = fingerprint
                decisions.append({"arm": arm, "wind_from_degrees": wind_from, "decision_utc": decision_utc,
                                  "window_utc": window, "first_recorded_step": step, "first_recorded_action_start_s": start_s,
                                  "raw_action": raw, "p_hold": probability, "classical_action": classical,
                                  "first_applied_source": source, "first_applied_action": action, "fallback_reason": reason,
                                  "release_guard": nav.get("temporal_recovery_release_guard"),
                                  "gate_vmg_mps": nav.get("gate_vmg_mps"), "drift_mode": nav.get("drift_mode"),
                                  "phase": nav["phase"], "waypoint": nav["waypoint"]})
            else:
                assert proposal_events[key] == fingerprint, (arm, step, "Cached model proposal changed")
        source_actions[source, action] += 1
        reasons[reason or "none"] += 1
        counts["colreg_active_updates"] += active > 0
        counts["accepted_policy_updates"] += accepted
        counts["learned_source_updates"] += source == "learned"
        counts["learned_hold_updates"] += source == "learned" and action == "hold"
        counts["learned_sail_updates"] += source == "learned" and action == "sail"
        counts["accepted_hold_updates"] += accepted and action == "hold"
        counts["accepted_sail_updates"] += accepted and action == "sail"
        counts["physical_release_overrides"] += source == "physical release"
        counts["learned_applied_with_colreg"] += accepted and active > 0
        counts["accepted_action_differs_classical"] += accepted and action != classical
        counts["AIS_encounter_deferrals"] += reason == "AIS encounter"
        counts["AIS_steering_override_deferrals"] += reason == "AIS steering override"
        counts["AIS_action_fallback_deferrals"] += reason == "AIS action fallback"
        counts["actual_drift_trim_updates"] += actual_trim
        counts["holding_mode_updates"] += nav.get("drift_mode") == "holding"
        counts["recovering_mode_updates"] += nav.get("drift_mode") == "recovering"
        counts["forecast_hold_reference_updates"] += nav.get("forecast_hold_source") == "archive forecast"
        counts["forecast_reference_updates"] += nav.get("forecast_closest_m") is not None
        counts["AIS_buffer_maneuver_updates"] += mode == "buffer maneuver"
        counts["AIS_sustained_recovery_updates"] += bool(nav.get("ais_recovery_active", 0))
        counts["AIS_recovery_soft_colreg_override_updates"] += bool(nav.get("ais_colreg_soft_override", 0))
        counts["AIS_fallback_updates"] += mode.startswith("fallback")
        counts["native_NMPC_updates"] += mode == "route" and not actual_trim and nav.get("solver_fallback") is None
        trim_key = (nav["phase"], nav["waypoint"]) if actual_trim else None
        if open_trim is not None and trim_key != (open_trim["phase"], open_trim["waypoint"]):
            open_trim.update(end_s=start_s, duration_s=start_s - open_trim["start_s"], end_reason=reason or mode)
            trim_intervals.append(open_trim)
            open_trim = None
        if trim_key is not None and open_trim is None:
            open_trim = {"arm": arm, "phase": nav["phase"], "waypoint": nav["waypoint"], "start_s": start_s}
        state = (source, action, reason, active > 0, mode, nav.get("drift_mode"), actual_trim)
        if previous != state:
            transitions.append({"arm": arm, "step": step, "action_start_s": start_s, "source": source,
                                "action": action, "raw_action": raw, "reason": reason, "COLREG_active": active,
                                "ais_mode": mode, "drift_mode": nav.get("drift_mode"), "actual_drift_trim": actual_trim,
                                "phase": nav["phase"], "waypoint": nav["waypoint"]})
            previous = state
    if open_trim is not None:
        end = (len(replay) - 1) * reference.STEP_SECONDS
        open_trim.update(end_s=end, duration_s=end - open_trim["start_s"], end_reason="end of recording")
        trim_intervals.append(open_trim)
    assert sum(row["duration_s"] for row in trim_intervals) == counts["actual_drift_trim_updates"] * 5, arm
    if not learning:
        assert not accepted_windows and not proposal_events and counts["accepted_policy_updates"] == 0, arm
    counts["accepted_weather_windows"] = len(accepted_windows)
    counts["raw_model_weather_windows"] = len(raw_windows)
    counts["model_proposal_events"] = len(proposal_events)
    counts["raw_hold_proposal_events"] = sum(x[0] == "hold" for x in proposal_events.values())
    counts["raw_sail_proposal_events"] = sum(x[0] == "sail" for x in proposal_events.values())
    counts["physical_release_proposal_events"] = sum(row["first_applied_source"] == "physical release" for row in decisions)
    counts["classically_vetoed_proposal_events"] = sum(row["first_applied_source"] == "classical fallback" for row in decisions)
    counts["actual_hold_seconds"] = counts["actual_drift_trim_updates"] * 5
    return {"arbitration": dict(counts), "first_accepted_policy_start_s": first_learned,
            "first_accepted_difference_from_classical_s": first_difference,
            "first_physical_release_override_start_s": first_physical_release,
            "arbitration_source_actions": [{"source": s, "action": a, "updates": n} for (s, a), n in source_actions.items()],
            "arbitration_fallback_reasons": dict(reasons), "actual_hold_intervals": trim_intervals}, transitions, decisions


def recovery_diagnostics(replay):
    """Separate optimization cost and native astern motion from route outcomes."""
    applied = replay[1:]
    recovered = [frame for frame in applied if frame["navigation"].get("ais_recovery_active", 0)]
    calls = [float(frame["navigation"]["ais_maneuver_search_seconds"]) for frame in applied
             if frame["navigation"].get("ais_maneuver_search_seconds") is not None]
    ordered = sorted(calls)
    cached_rechecks, previous_remaining = [], None
    for frame in applied:
        nav = frame["navigation"]
        if nav.get("ais_recovery_active", 0):
            remaining = float(nav["ais_recovery_remaining_s"])
            if previous_remaining is not None and remaining < previous_remaining:
                cached_rechecks.append(float(nav["ais_maneuver_search_seconds"]))
            previous_remaining = remaining
        else:
            previous_remaining = None
    ordered_cached = sorted(cached_rechecks)
    margins = [float(frame["navigation"]["ais_recovery_reserve_margin_m"]) for frame in recovered]
    profiles = [{"step": frame["navigation"]["step"], **frame["navigation"]["ais_recovery_search_profile"]}
                for frame in applied if frame["navigation"].get("ais_recovery_search_profile") is not None]
    surges = [float(frame["state"]["surge_speed"]) for frame in recovered
              if frame["state"].get("surge_speed") is not None]
    return {
        "recovery_applied_updates": len(recovered),
        "recovery_first_action_start_s": None if not recovered else (recovered[0]["navigation"]["step"] - 1) * 5,
        "recovery_last_action_end_s": None if not recovered else recovered[-1]["navigation"]["step"] * 5,
        "minimum_recorded_recovery_preview_margin_m": min(margins, default=None),
        "minimum_recorded_recovery_surge_mps": min(surges, default=None),
        "search_calls": len(calls), "search_total_seconds": math.fsum(calls),
        "first_AIS_choose_seconds": calls[0] if calls else None,
        "search_max_seconds": max(calls, default=None),
        "search_p95_seconds": ordered[max(0, math.ceil(.95 * len(ordered)) - 1)] if ordered else None,
        "search_calls_exceeding_5s_interval": sum(value > 5 for value in calls),
        "recovery_cached_rechecks": len(cached_rechecks),
        "recovery_cached_recheck_max_seconds": max(cached_rechecks, default=None),
        "recovery_cached_recheck_p95_seconds": ordered_cached[max(0, math.ceil(.95 * len(ordered_cached)) - 1)] if ordered_cached else None,
        "recovery_backend_update_counts": dict(Counter(frame["navigation"].get("ais_recovery_backend", "unrecorded")
                                                       for frame in recovered)),
        "fresh_recovery_search_profiles": profiles,
        "timing_scope": "Wall time inside AISBufferPlanner.choose; includes cold construction and cached revalidation. Virtual simulation time does not advance during planning.",
        "astern_scope": "Signed native water-relative surge at saved endpoints during recovery, not an injected velocity or a validated reverse-hydrodynamic sea trial.",
    }


def audit_attempt_timings(folder, replay, diagnostics):
    prefix = "AIS decision timing: "
    attempts = [json.loads(line.removeprefix(prefix)) for line in (folder / "run.log").read_text().splitlines() if line.startswith(prefix)]
    applied = replay[1:]
    assert len(applied) <= len(attempts) <= len(applied) + 1, (folder.name,"Missing or duplicate attempt records")
    assert [row["step"] for row in attempts] == list(range(1,len(attempts)+1)), folder.name
    for frame, attempt in zip(applied, attempts):
        nav = frame["navigation"]
        assert nav["step"] == attempt["step"] and abs(nav["ais_maneuver_search_seconds"] - attempt["wall_seconds"]) < 1e-12
    values = [float(row["wall_seconds"]) for row in attempts]
    assert all(math.isfinite(value) and value >= 0 for value in values)
    ordered = sorted(values)
    diagnostics.update(applied_replay_search_calls=len(applied), unapplied_attempts=len(attempts)-len(applied),
        search_calls=len(attempts), search_total_seconds=math.fsum(values),
        first_AIS_choose_seconds=values[0] if values else None,
        search_max_seconds=max(values,default=None),
        search_p95_seconds=ordered[max(0,math.ceil(.95*len(ordered))-1)] if ordered else None,
        search_calls_exceeding_5s_interval=sum(value >= 5 for value in values),
        attempt_timing_scope="Every whole AIS choose attempt from finally-logged wall times, including unsuccessful/exceptional terminal attempts; applied-frame timers independently cross-checked.")
    return attempts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--arms", nargs="+", default=ARMS)
    args = parser.parse_args()
    study = args.study.resolve()
    independent = {row["case"]: row for row in json.loads((study / "independent_traffic_sweep.json").read_text())}
    providers = {name: cls(**reference.OTRANTO_BOUNDS).load() for name, cls in reference.ENVIRONMENT.items()}
    route = {node.findtext("{*}name"): reference.gpx_position(node)
             for node in ElementTree.parse(reference.ROUTE_PATH).findall(".//{*}rtept")}
    assert list(route) == ["001", "002", "003", "004", "005"]
    names = list(route)
    expected = [("outbound", x) for x in names[1:]] + [("return", x) for x in names[-2::-1]]
    runs, paired, commands, source_fingerprints, cached_frames = {}, {}, {}, {}, {}
    waypoint_rows, segment_rows, run_rows, transitions, decisions, paired_captures = [], [], [], [], [], []
    telemetry_dir = study / "telemetry_by_arm"
    telemetry_dir.mkdir(exist_ok=True)
    telemetry_headers = []
    for arm in args.arms:
        folder = study / "runs" / arm
        execution = json.loads((folder / "execution.json").read_text())
        command, source_fingerprints[arm] = verify_protocol(arm, folder, execution)
        commands[arm] = normalize(command)
        wind, method = execution["wind_from_degrees"], execution["method"]
        scenario = execution["scenario_id"]
        result, replay = reference.audit_run(arm, folder, route, expected, providers, wind)
        truth = independent[arm]
        assert truth["replay_sha256"] == reference.digest(folder / "navigation.jsonl.gz") and truth["frames"] == len(replay), arm
        result["scenario_id"] = scenario
        result["method"] = method
        result["independent_clearance"] = truth
        arbitration, arm_transitions, arm_decisions = audit_arbitration(arm, replay, method == "transformer", wind)
        result.update(arbitration)
        result["recovery_diagnostics"] = recovery_diagnostics(replay)
        attempts = audit_attempt_timings(folder, replay, result["recovery_diagnostics"])
        counts = arbitration["arbitration"]
        logged_colreg = sum(line.startswith("COLREG: ") for line in (folder / "run.log").read_text().splitlines())
        assert counts["colreg_active_updates"] <= logged_colreg <= counts["colreg_active_updates"] + len(attempts) - len(replay) + 1, arm
        result["unapplied_COLREG_assessments"] = logged_colreg - counts["colreg_active_updates"]
        for row in [*arm_transitions, *arm_decisions]: row["scenario_id"] = scenario
        transitions.extend(arm_transitions)
        decisions.extend(arm_decisions)
        previous_wp = "001"
        for capture in result["captures"]:
            waypoint_rows.append({"arm": arm, "scenario_id": scenario, "wind_from_degrees": wind, "method": method, **capture})
            segment_rows.append({"arm": arm, "scenario_id": scenario, "wind_from_degrees": wind, "method": method, "phase": capture["phase"],
                                 "from_waypoint": previous_wp, "to_waypoint": capture["waypoint"], "captured": True,
                                 "segment_s": capture["segment_s"], "track_m": capture["segment_distance_m"],
                                 "elapsed_s": capture["elapsed_s"], "capture_range_m": capture["distance_metres"]})
            previous_wp = capture["waypoint"]
        if result["unfinished_leg"]:
            unfinished = result["unfinished_leg"]
            segment_rows.append({"arm": arm, "scenario_id": scenario, "wind_from_degrees": wind, "method": method, "phase": unfinished["phase"],
                                 "from_waypoint": previous_wp, "to_waypoint": unfinished["waypoint"], "captured": False,
                                 "segment_s": unfinished["segment_s"], "track_m": unfinished["segment_distance_m"],
                                 "elapsed_s": result["elapsed_s"], "terminal_range_m": unfinished["terminal_range_m"],
                                 "minimum_range_m": unfinished["minimum_range_m"]})
        run_rows.append({"arm": arm, "scenario_id": scenario, "wind_from_degrees": wind, "method": method, "status": result["status"],
                         "captures": len(result["captures"]), "elapsed_s": result["elapsed_s"], "track_m": result["track_m"],
                         "first_accepted_policy_start_s": result["first_accepted_policy_start_s"], **counts,
                         "AIS_search_max_seconds": result["recovery_diagnostics"]["search_max_seconds"],
                         "AIS_search_p95_seconds": result["recovery_diagnostics"]["search_p95_seconds"],
                         "AIS_search_overrun_updates": result["recovery_diagnostics"]["search_calls_exceeding_5s_interval"],
                         "AIS_recovery_min_recorded_surge_mps": result["recovery_diagnostics"]["minimum_recorded_recovery_surge_mps"],
                         **{key: truth[key] for key in CLEARANCE_FIELDS}})
        telemetry = []
        for frame in replay:
            obs = frame["observation"]
            row = {"arm": arm, "scenario_id": scenario, "wind_from_degrees": wind, "method": method, "utc": obs["time"]}
            row.update({"state_" + key: value for key, value in frame["state"].items()})
            row.update(frame["navigation"])
            for layer in ("wind", "wave", "current"):
                row.update({layer + "_" + key: value for key, value in obs[layer].items()})
            telemetry.append(row)
        fields = list(dict.fromkeys(key for row in telemetry for key in row))
        telemetry_headers.extend(fields)
        write_csv(telemetry_dir / f"{arm}.csv", telemetry, fields)
        del telemetry
        runs[arm] = result
        cached_frames[arm] = replay
        pair = [f"{scenario}__otranto_{wind:03d}__{method}" for method in ("classical", "transformer")]
        pair_key = f"{scenario}__{wind:03d}"
        if all(name in cached_frames for name in pair):
            classical, transformer = pair
            assert commands[classical] == commands[transformer], (wind, "Treatment protocol mismatch")
            assert source_fingerprints[classical] == source_fingerprints[transformer], (wind, "Source mismatch")
            before, after = [cached_frames[name] for name in pair]
            assert before[0]["state"] == after[0]["state"] and before[0]["observation"] == after[0]["observation"], wind
            paired[pair_key] = {"classical_arm": classical, "transformer_arm": transformer,
                                **reference.compare_trajectories(before, after),
                                "launch_time_world_sources_identical": True,
                                "transformer_minus_classical_elapsed_s": runs[transformer]["elapsed_s"] - runs[classical]["elapsed_s"],
                                "both_completed": all(runs[name]["status"] == "completed" for name in pair)}
            capture_deltas = []
            for before_capture, after_capture in zip(runs[classical]["captures"], runs[transformer]["captures"]):
                assert (before_capture["phase"], before_capture["waypoint"]) == (after_capture["phase"], after_capture["waypoint"])
                capture_deltas.append({"scenario_id": scenario, "wind_from_degrees": wind, "phase": before_capture["phase"],
                                       "waypoint": before_capture["waypoint"],
                                       "classical_elapsed_s": before_capture["elapsed_s"],
                                       "transformer_elapsed_s": after_capture["elapsed_s"],
                                       "transformer_minus_classical_s": after_capture["elapsed_s"] - before_capture["elapsed_s"],
                                       "classical_capture_range_m": before_capture["distance_metres"],
                                       "transformer_capture_range_m": after_capture["distance_metres"],
                                       "classical_segment_s": before_capture["segment_s"],
                                       "transformer_segment_s": after_capture["segment_s"],
                                       "classical_segment_track_m": before_capture["segment_distance_m"],
                                       "transformer_segment_track_m": after_capture["segment_distance_m"]})
            paired[pair_key]["capture_deltas"] = capture_deltas
            paired_captures.extend(capture_deltas)
            for name in pair:
                del cached_frames[name]
        print(arm, result["status"], len(result["captures"]), result["elapsed_s"], flush=True)
    for stem, rows in (("waypoint_metrics", waypoint_rows), ("segment_metrics", segment_rows),
                        ("run_metrics", run_rows), ("arbitration_transitions", transitions),
                        ("model_decision_events", decisions), ("paired_waypoint_deltas", paired_captures)):
        empty_fields = {"waypoint_metrics": ["arm","scenario_id","wind_from_degrees","method","phase","waypoint","step","distance_metres","elapsed_s","segment_s","segment_distance_m","utc","latitude","longitude"],
                        "paired_waypoint_deltas": ["scenario_id","wind_from_degrees","phase","waypoint","classical_elapsed_s","transformer_elapsed_s","transformer_minus_classical_s"],
                        "model_decision_events": ["arm","scenario_id","wind_from_degrees","decision_utc","window_utc","raw_action","p_hold","first_applied_source","first_applied_action"]}
        write_csv(study / (stem + ".csv"), rows, empty_fields.get(stem) if not rows else None)
    fields = list(dict.fromkeys(telemetry_headers))
    with (study / "telemetry.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for arm in args.arms:
            with (telemetry_dir / f"{arm}.csv").open() as stream:
                writer.writerows(csv.DictReader(stream))
    result = {
        "departure_utc": reference.DEPARTURE.isoformat(), "encounter_utc": reference.ENCOUNTER.isoformat(),
        "radius_m": reference.RADIUS, "dt_s": reference.STEP_SECONDS, "maximum_steps": reference.MAXIMUM_STEPS,
        "wind_from_degrees": sorted({run["wind_from_degrees"] for run in runs.values()}),
        "scenario_ids": sorted({run["scenario_id"] for run in runs.values()}),
        "marine_bounds": reference.OTRANTO_BOUNDS, "runs": runs, "comparisons_by_wind": paired,
        "measurement": "First saved entry in each original 50 m GPX circle; geodesic chords between consecutive 5 s fixes. Incomplete tails are included in total distance.",
        "forcing_audit": "All saved positions and timestamps resampled from native archive; wind speed preserved with declared controlled FROM bearing.",
        "clearance_measurement": "Independent 0.05 s Fossen reconstruction against truth and report-projected full uncertainty with and without 3 m reserve.",
        "policy_counting": "Applied updates counted after actuator arbitration; fresh model decisions deduplicated by decision UTC plus weather-window UTC; cached telemetry is reported separately.",
        "classical_rule_diagnostic": "Each temporal classical_action is the rule evaluated at that decision's own state, retained with cached telemetry. It is distinct from the separate classical trajectory after paths diverge.",
    }
    (study / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Verified", study / "comparison.json", flush=True)


if __name__ == "__main__":
    main()
