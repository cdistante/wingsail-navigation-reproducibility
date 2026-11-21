"""Independently audit both Otranto priority runs and compare the matched off replays."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import sys
from time import sleep
from unittest.mock import patch


STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[2]
BASELINE = STUDY.with_name("otranto_closest_layline_ablation")
PRIOR_AUDIT_PATH = BASELINE / "reproduce" / "audit_laylines.py"
PRIOR_AUDIT_SHA256 = "36081ff6d53ca8c15b56cc780311a792903fc7fa97083c763ff3582d0865f49d"
WINDS = (5, 355)
FLAG = "--prioritize-waypoint-layline"
PRIORITY_SOURCE_EDITS = {"world/legacy_planner_adapter.py", "world/run_legacy_fossen.py"}
MODES = ("off", "priority")
ARMS = tuple(f"crossing_fan__otranto_{wind:03d}__layline_priority" for wind in WINDS)

sys.path.insert(0, str(PRIOR_AUDIT_PATH.parent))
import audit_laylines as prior

native = prior.native
read, save = prior.read, prior.save


def comparable_command(command):
    """Remove only the tested policy flag and run-local simulator output paths."""
    return prior.normalized([arg for arg in command if arg != FLAG], historical_command=True)


def baseline_for(wind):
    case = f"crossing_fan__otranto_{wind:03d}__layline_off"
    run = BASELINE / "runs" / case
    path = BASELINE / "independent_case_receipts" / f"{case}.json"
    receipt = read(path)
    execution_path = run / "execution.json"
    execution = read(execution_path)
    assert receipt["case"] == case and receipt["passed"]
    assert receipt["audit_source_sha256"] == PRIOR_AUDIT_SHA256
    assert receipt["input_verification"]["execution_sha256"] == native.digest(execution_path)
    for artifact, key in (("navigation.jsonl.gz", "replay_sha256"), ("run.log", "run_log_sha256")):
        assert native.digest(run / artifact) == receipt["clearance"][key], (case, artifact)
    assert execution["closest_waypoint_layline"] is False and execution["layline_mode"] == "off"
    assert receipt["layline_policy"]["applied_reference_steps"] == 0
    assert receipt["route"]["status"] == "completed" and len(receipt["route"]["captures"]) == 8
    assert receipt["clearance"]["configured_truth_centered_minimum_m"] > 0.
    assert receipt["clearance"]["configured_model_minimum_m"] > 0.
    assert receipt["native_control_timing"]["all_complete_native_control_attempts_below_5s"]
    timing_path = BASELINE / "independent_native_control_timing" / f"{case}.csv"
    assert native.digest(timing_path) == receipt["native_control_timing"]["timing_csv_sha256"]
    return run, execution, receipt, native.digest(path)


def verify_inputs(run):
    metadata = read(run / "experiment.json")
    execution = read(run / "execution.json")
    wind = int(metadata["wind_from_degrees"])
    baseline, base_execution, base_receipt, baseline_audit_hash = baseline_for(wind)
    mode = "priority"
    assert run.name == metadata["run_id"] == execution["run_id"] == f"crossing_fan__otranto_{wind:03d}__layline_{mode}"
    for row in (metadata, execution):
        assert row["prioritize_waypoint_layline"] is True and row["closest_waypoint_layline"] is True
        assert row["layline_mode"] == mode
    assert metadata["native_hybrid_enabled"] and metadata["native_schedule_veto"]
    assert metadata["radius_m"] == 50. and metadata["dt_s"] == 5.
    command = metadata["command"]
    other_command = base_execution["command"]
    assert command == execution["command"] == shlex.split((run / "command.txt").read_text())
    assert command.count(FLAG) == 1 and prior.FLAG not in command
    assert FLAG not in other_command and prior.FLAG not in other_command
    assert comparable_command(command) == comparable_command(other_command)
    # Maintain visibility of the priority treatment while comparing reference controls.
    safety_flags = ("--colreg", "--time-aware-ais", "--native-control-timing", "--ais-candidate-handoff")
    assert all(command.count(flag) == 1 for flag in safety_flags)
    for key in prior.MATCH_FIELDS:
        assert metadata[key] == execution[key] == base_execution[key], (run.name, key)
    for path_key, hash_key in prior.FIXTURE_FIELDS:
        expected = base_execution[hash_key]
        assert metadata[hash_key] == execution[hash_key] == native.digest(metadata[path_key]) == expected
        assert native.digest(base_execution[path_key]) == expected
    assert native.digest(STUDY / "archive_field_fingerprints.json") == metadata["archive_field_fingerprint_sha256"]
    assert metadata["source_sha256"] == execution["source_sha256"]
    assert execution["source_unchanged"] and not execution["changed_inputs"]
    previous_sources, fresh_sources = base_execution["source_sha256"], execution["source_sha256"]
    assert previous_sources.keys() <= fresh_sources.keys()
    differences = {name: {"baseline": checksum, "priority": fresh_sources[name]}
                   for name, checksum in previous_sources.items() if fresh_sources[name] != checksum}
    assert set(differences) == PRIORITY_SOURCE_EDITS
    assert differences == metadata["changes_from_layline_off_sources"] == execution["changes_from_layline_off_sources"]
    for key, checksum in execution["source_sha256"].items():
        assert native.digest(run / "source_snapshot" / key) == native.digest(ROOT / key) == checksum, (run.name, key)
    assert metadata["layline_off_execution_sha256"] == base_receipt["input_verification"]["execution_sha256"]
    assert metadata["layline_off_replay_sha256"] == base_receipt["clearance"]["replay_sha256"]
    for field, name in (("departure_gate_summary_sha256", "departure_gate.json"),
                        ("departure_gate_log_sha256", "departure_gate.jsonl")):
        assert execution[field] == native.digest(run / name)
        assert base_execution[field] == native.digest(baseline / name)
    gate = read(run / "departure_gate.json")
    base_gate = read(baseline / "departure_gate.json")
    for field in ("monitoring_start_utc", "traffic_epoch_utc", "parameters", "actual_departure_utc", "gate_wait_seconds"):
        assert gate[field] == base_gate[field], (run.name, field)
    first = next(prior.replay(run / "navigation.jsonl.gz"))
    expected_first = next(prior.replay(baseline / "navigation.jsonl.gz"))
    assert first["state"] == expected_first["state"] and first["observation"] == expected_first["observation"], run.name
    logged_args = prior.route_audit.one_json((run / "run.log").read_text().splitlines(), "Runner arguments: ")
    assert logged_args.count(FLAG) == 1 and all(logged_args.count(flag) == 1 for flag in safety_flags)
    return {"case": run.name, "mode": mode, "wind_from_degrees": wind,
            "execution_return_code": execution["return_code"],
            "same_invocation_as_verified_off_apart_from_priority_flag_and_run_paths": True,
            "same_forecast_archive_AIS_initial_state_and_launch_clock": True,
            "sources_bound_to_unchanged_execution_snapshot": True,
            "verified_baseline_case": baseline.name,
            "baseline_case_audit_sha256": baseline_audit_hash,
            "baseline_execution_sha256": base_receipt["input_verification"]["execution_sha256"],
            "baseline_replay_sha256": base_receipt["clearance"]["replay_sha256"],
            "experiment_sha256": native.digest(run / "experiment.json"),
            "execution_sha256": native.digest(run / "execution.json")}


def audit_policy(run, enabled):
    counts = Counter()
    first = last = None
    for frame in prior.replay(run / "navigation.jsonl.gz"):
        nav = frame["navigation"]
        source = nav.get("waypoint_layline_source")
        if source is None:
            assert "waypoint_layline_eta_s" not in nav and "waypoint_layline_policy" not in nav, (run.name, nav["step"])
            continue
        assert enabled and nav["waypoint_layline_policy"] == "priority"
        assert source in {"direct gate approach", "closest reachable tack"}
        assert nav["ais_mode"] == "route" and nav["ais_weather_candidate_admitted"] is True
        assert nav.get("forecast_closest_m") is None
        distance, eta = nav["waypoint_layline_distance_m"], nav["waypoint_layline_eta_s"]
        assert math.isfinite(distance) and distance >= 0. and 0. < eta <= 2880. + 1e-8
        assert (nav["waypoint_layline_second_bow_deg"] is None) == (source == "direct gate approach")
        if nav["step"] == 0:
            continue
        counts[(nav["phase"], nav["waypoint"], source)] += 1
        record = {field: nav[field] for field in (
            "step", "phase", "waypoint", "heading_reference_deg", "bow_reference_deg",
            "waypoint_layline_source", "waypoint_layline_distance_m", "waypoint_layline_eta_s",
            "waypoint_layline_second_bow_deg")}
        record["fix_utc"] = frame["observation"]["time"]
        if first is None:
            first = record
        last = record
    return {"applied_reference_steps": sum(counts.values()),
            "applied_reference_steps_by_leg_and_source": [
                {"phase": phase, "waypoint": mark, "source": source, "steps": steps}
                for (phase, mark, source), steps in counts.items()],
            "first_applied_reference": first, "last_applied_reference": last,
            "scope": "Priority layline references retained through AIS/controller admission. Held references count per step."}


def audit_case(run):
    # The pinned auditor reconstructs forcing and substep plant geometry independently.
    with patch.multiple(prior, STUDY=STUDY, verify_inputs=verify_inputs, audit_policy=audit_policy):
        result = prior.audit_case(run)
    result["audit_source_sha256"] = native.digest(__file__)
    return result


def pair_receipt(on):
    wind = on["input_verification"]["wind_from_degrees"]
    baseline, _, off, proof = baseline_for(wind)
    treatment = STUDY / "runs" / on["case"]
    parity = prior.compare_replays(baseline, treatment)
    assert parity["common_frames"] == 1 + min(off["route"]["steps"], on["route"]["steps"])
    row = {"wind_from_degrees": wind, "off_case": off["case"], "priority_case": on["case"],
           "off_case_receipt_sha256": proof, "baseline_is_reused_verified_run": True,
           "only_guidance_flag_differs": True, **parity}
    for mode, receipt in zip(MODES, (off, on), strict=True):
        route, clearance, timing = (receipt[key] for key in ("route", "clearance", "native_control_timing"))
        row.update({
            f"{mode}_status": route["status"],
            f"{mode}_captures": len(route["captures"]),
            f"{mode}_navigation_s": route["elapsed_s"],
            f"{mode}_total_from_monitoring_s": route["total_elapsed_s"],
            f"{mode}_wp002_s": next((mark["elapsed_s"] for mark in route["captures"]
                                    if (mark["phase"], mark["waypoint"]) == ("outbound", "002")), None),
            f"{mode}_full_truth_margin_m": clearance["configured_truth_centered_minimum_m"],
            f"{mode}_full_report_margin_m": clearance["configured_model_minimum_m"],
            f"{mode}_maximum_complete_control_s": timing["maximum_complete_native_control_decision_s"],
            f"{mode}_complete_control_attempts_at_or_above_5s": timing["attempts_at_or_above_5s"],
            f"{mode}_applied_layline_reference_steps": receipt["layline_policy"]["applied_reference_steps"],
        })
    row["priority_minus_off_navigation_s"] = (
        row["priority_navigation_s"] - row["off_navigation_s"] if row["priority_status"] == "completed" else None)
    row["priority_minus_off_wp002_s"] = (
        row["priority_wp002_s"] - row["off_wp002_s"] if row["priority_wp002_s"] is not None else None)
    return row


def aggregate():
    folder = STUDY / "independent_case_receipts"
    records = {case: read(folder / f"{case}.json") for case in ARMS if (folder / f"{case}.json").is_file()}
    clearance = [item["clearance"] for item in records.values()]
    gates = [item["gate"] for item in records.values()]
    save(STUDY / "independent_traffic_sweep.json", clearance)
    save(STUDY / "independent_departure_gate_check.json",
         {"all_available_gates_verified": bool(gates), "cases": gates,
          "all_gate_attempts_below_5s": bool(gates) and all(entry["all_gate_attempts_below_5s"] for entry in gates)})
    for section, filename in (("route", "route_verification"), ("native_control_timing", "native_control_timing"),
                              ("input_verification", "inputs_verification"), ("layline_policy", "layline_policy")):
        save(STUDY / f"independent_{filename}.json", {"cases": {k: v[section] for k, v in records.items()}})
    pairs = [pair_receipt(record) for record in records.values()]
    save(STUDY / "independent_comparison.json",
         {"sign_convention": "Priority minus off, with negative faster.", "pairs": pairs})
    prior.save_csv(STUDY / "independent_comparison.csv", pairs)
    complete = set(records) == set(ARMS)
    summary = {"expected_runs": len(ARMS), "closed_independently_audited_runs": len(records), "complete_matrix": complete,
               "all_acceptance_checks_passed": complete and all(item["passed"] for item in records.values()),
               "verified_captures": sum(len(item["route"]["captures"]) for item in records.values()),
               "applied_commands": sum(item["applied_actions"] for item in clearance),
               "native_0p05s_substeps": sum(item["applied_actions"] * 100 for item in clearance),
               "per_case_checks": {case: item["acceptance_checks"] for case, item in records.items()},
               "minimum_truth_full_plus_reserve_m": min((row["configured_truth_centered_minimum_m"] for row in clearance
                                                           if row["configured_truth_centered_minimum_m"] is not None), default=None),
               "minimum_report_full_plus_reserve_m": min((row["configured_model_minimum_m"] for row in clearance
                                                            if row["configured_model_minimum_m"] is not None), default=None),
               "maximum_complete_control_s": max(
                   (r["native_control_timing"]["maximum_complete_native_control_decision_s"] for r in records.values()
                    if r["native_control_timing"]["maximum_complete_native_control_decision_s"] is not None), default=None),
               "full_control_attempts_at_or_above_5s": sum(r["native_control_timing"]["attempts_at_or_above_5s"]
                                                          for r in records.values()),
               "audit_source_sha256": native.digest(__file__), "prior_audit_sha256": PRIOR_AUDIT_SHA256,
               "pinned_auditors": prior.PINS,
               "checked_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "Ordered first 50 m endpoint captures, archive and AIS parity, native substep clearance, all complete timed decisions."}
    if complete:
        with patch.object(sys, "argv", [prior.audit_exposure.__file__, "--study", str(STUDY)]):
            prior.audit_exposure.main()
        exposure = read(STUDY / "fault_exposure_check.json")
        summary["both_full40s_underway_lost_delivery_effects_experienced_in_all_runs"] = all(
            row["both_full40s_underway_lost_delivery_effects_experienced"] for row in exposure["cases"])
        summary["both_underway65s_peaks_experienced_in_all_runs"] = exposure["all_underway_peak_pairs_experienced"]
    save(STUDY / "independent_final_summary.json", summary)
    return summary


def current_receipt(run, folder):
    path = folder / f"{run.name}.json"
    if not path.is_file():
        return False
    receipt = read(path)
    return (receipt["audit_source_sha256"] == native.digest(__file__)
            and receipt["input_verification"]["execution_sha256"] == native.digest(run / "execution.json")
            and receipt["clearance"]["replay_sha256"] == native.digest(run / "navigation.jsonl.gz")
            and receipt["clearance"]["run_log_sha256"] == native.digest(run / "run.log"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=ARMS)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--watch", action="store_true", help="Audit each selected run once its process closes.")
    args = parser.parse_args()
    assert native.digest(PRIOR_AUDIT_PATH) == PRIOR_AUDIT_SHA256
    with patch.object(prior, "STUDY", STUDY):
        prior.verify_shared_files()
    for wind in WINDS:
        baseline_for(wind)
    folder = STUDY / "independent_case_receipts"
    pending = {STUDY / "runs" / case for case in (args.case or ARMS)}
    failures = []
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        active = {}
        while pending or active:
            for run in sorted(pending.copy()):
                if not prior.closed(run):
                    continue
                pending.remove(run)
                if not current_receipt(run, folder):
                    print(f"Auditing closed run {run.name}", flush=True)
                    active[pool.submit(audit_case, run)] = run
            if not args.watch:
                pending.clear()
            if not active and pending:
                sleep(10)
            finished, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
            for future in finished:
                run = active.pop(future)
                try:
                    receipt = future.result()
                except Exception as error:
                    failures.append({"case": run.name, "exception": repr(error)})
                    print(f"Audit failed: {run.name}: {error!r}", flush=True)
                    continue
                save(folder / f"{run.name}.json", receipt)
                print(f"Audited {run.name}: {receipt['route']['status']}, {len(receipt['route']['captures'])}/8 captures", flush=True)
                aggregate()
    if failures:
        save(STUDY / "independent_audit_failures.json", failures)
        raise RuntimeError(f"Independent audit failed for {len(failures)} cases; see failure receipt.")
    print(json.dumps(aggregate(), indent=2), flush=True)


if __name__ == "__main__":
    main()
