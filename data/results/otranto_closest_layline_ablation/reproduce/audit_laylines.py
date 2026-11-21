"""Audit the paired Otranto layline runs using the pinned native-veto auditors."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import csv
import gzip
import json
import math
from pathlib import Path
import shlex
import sys
from unittest.mock import patch
from xml.etree import ElementTree


STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[2]
PRIOR = STUDY.with_name("otranto_native_schedule_veto")
WINDS = (5, 355)
MODES = ("off", "on")
ARMS = tuple(f"crossing_fan__otranto_{wind:03d}__layline_{mode}"
             for wind in WINDS for mode in MODES)
FLAG = "--closest-waypoint-layline"
LOCAL_PATHS = ("--acados-build-dir", "--output-dir", "--output-gpx", "--record-replay")
FIXTURE_FIELDS = (
    ("traffic_fixture", "traffic_fixture_sha256"),
    ("truth_trace", "traffic_truth_trace_sha256"),
    ("underway_traffic_fixture", "underway_traffic_fixture_sha256"),
    ("underway_truth_trace_csv", "underway_truth_trace_sha256"),
)
MATCH_FIELDS = (
    "encounter_id", "scenario_id", "method", "wind_from_degrees", "wind_from", "reporting",
    "original_reporting", "dt_s", "radius_m", "maximum_steps", "cap_s", "return_to_start",
    "departure_utc", "encounter_utc", "requested_departure_utc", "fixed_encounter_utc",
    "fixed_drop_emission_utc_windows", "underway_first_emission_s", "underway_first_delivery_s",
    "underway_drop_emission_windows_s", "underway_frame_retention_original_epoch_s",
    "underway_reference_elapsed_s", "underway_anchor_utc", "native_AIS_candidate_handoff",
    "native_control_timing", "native_hybrid_enabled", "native_schedule_veto", "regatta_enabled",
    "native_hybrid_checkpoint_sha256", "native_hybrid_model_files_sha256",
    "archive_field_fingerprint_sha256", "thread_environment",
)
PINS = {
    "audit_independent.py": "2b4c281813ac4ed3f0852a92844cbf8b9cef4b10b26397d497f827775ee07d77",
    "audit_departure_gate.py": "daa0287b9bc8118cbb1faf3018ee6b517dac8812f7ffb2e4afbb6fbf62f1ce25",
    "audit_route_reference.py": "a2e6989155f544375f49b78c2598c40d68997aca10d887f834992be61501465d",
    "audit_exposure.py": "ce3e9b09ed1ab5e36630067bc9c3e64d5047313599d2d99bf59b304dea6d6b00",
    "audit_native_hybrid.py": "3d233127b48dc5796568110038b617392312c2b35aba30d33d45570df5db3721",
}

sys.path.insert(0, str(PRIOR / "reproduce"))
sys.path.insert(0, str(ROOT))
import audit_departure_gate as gate_audit
import audit_exposure
import audit_independent as native
import audit_native_hybrid as historical
import audit_route_reference as route_audit


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    native.save_json(path, data)


def save_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def replay(path):
    with gzip.open(path, "rt") as stream:
        next(stream)
        yield from (json.loads(row) for row in stream)


def normalized(command, *, historical_command=False):
    command = list(command)
    if historical_command:
        command[2] = Path(command[2]).name
    if FLAG in command:
        command.remove(FLAG)
    for option in LOCAL_PATHS:
        assert command.count(option) == 1, option
        command[command.index(option) + 1] = option
    return command


def reference_path(wind):
    return PRIOR / "runs" / f"crossing_fan__otranto_{wind:03d}__hybrid"


def verify_shared_files():
    for filename, expected in PINS.items():
        assert native.digest(PRIOR / "reproduce" / filename) == expected, filename
    for filename in ("protocol.json", "underway_config.json", "baseline_provenance.json",
                     "archive_field_fingerprints.json"):
        assert native.digest(STUDY / filename) == native.digest(PRIOR / filename), filename


def verify_inputs(run):
    metadata = read(run / "experiment.json")
    execution = read(run / "execution.json")
    wind = int(metadata["wind_from_degrees"])
    mode = metadata["layline_mode"]
    reference = reference_path(wind)
    old = read(reference / "execution.json")
    assert run.name == metadata["run_id"] == execution["run_id"] == f"crossing_fan__otranto_{wind:03d}__layline_{mode}"
    assert wind in WINDS and mode in MODES
    assert metadata["closest_waypoint_layline"] is (mode == "on")
    assert execution["closest_waypoint_layline"] is metadata["closest_waypoint_layline"]
    assert execution["layline_mode"] == mode
    assert metadata["native_hybrid_enabled"] and metadata["native_schedule_veto"]
    assert metadata["radius_m"] == 50 and metadata["dt_s"] == 5
    assert metadata["command"] == execution["command"] == shlex.split((run / "command.txt").read_text())
    command = metadata["command"]
    assert command.count(FLAG) == int(mode == "on")
    assert command.count("--colreg") == command.count("--time-aware-ais") == 1
    assert command.count("--native-control-timing") == command.count("--ais-candidate-handoff") == 1
    assert normalized(command, historical_command=True) == normalized(old["command"], historical_command=True)
    for field in MATCH_FIELDS:
        assert metadata[field] == execution[field] == old[field], (run.name, field)
    assert metadata["archive_field_fingerprint_sha256"] == native.digest(STUDY / "archive_field_fingerprints.json")
    for path_field, hash_field in FIXTURE_FIELDS:
        assert native.digest(metadata[path_field]) == metadata[hash_field] == execution[hash_field] == old[hash_field]
        assert native.digest(old[path_field]) == old[hash_field]
    assert metadata["source_sha256"] == execution["source_sha256"]
    assert execution["source_unchanged"] and not execution["changed_inputs"], run.name
    for key, checksum in execution["source_sha256"].items():
        assert native.digest(run / "source_snapshot" / key) == native.digest(ROOT / key) == checksum, (run.name, key)
    for field, basename in (("departure_gate_summary_sha256", "departure_gate.json"),
                            ("departure_gate_log_sha256", "departure_gate.jsonl")):
        assert execution[field] == native.digest(run / basename), (run.name, field)
    for field in ("monitoring_start_utc", "traffic_epoch_utc", "parameters", "actual_departure_utc", "gate_wait_seconds"):
        assert read(run / "departure_gate.json")[field] == read(reference / "departure_gate.json")[field], (run.name, field)
    first = next(replay(run / "navigation.jsonl.gz"))
    before = next(replay(reference / "navigation.jsonl.gz"))
    assert first["state"] == before["state"] and first["observation"] == before["observation"], run.name
    lines = (run / "run.log").read_text().splitlines()
    logged_args = route_audit.one_json(lines, "Runner arguments: ")
    assert logged_args.count(FLAG) == int(mode == "on")
    assert "--colreg" in logged_args and "--time-aware-ais" in logged_args
    return {"case": run.name, "mode": mode, "wind_from_degrees": wind,
            "execution_return_code": execution["return_code"],
            "same_invocation_as_native_hybrid_apart_from_layline_flag_and_output_paths": True,
            "same_archives_traffic_initial_state_and_launch_clock": True,
            "sources_bound_to_unchanged_execution_snapshot": True,
            "baseline_execution_sha256": native.digest(reference / "execution.json"),
            "baseline_replay_sha256": native.digest(reference / "navigation.jsonl.gz"),
            "experiment_sha256": native.digest(run / "experiment.json"),
            "execution_sha256": native.digest(run / "execution.json")}


def audit_policy(run, enabled):
    counts = Counter()
    first = last = None
    for frame in replay(run / "navigation.jsonl.gz"):
        nav = frame["navigation"]
        source = nav.get("waypoint_layline_source")
        if source is None:
            assert "waypoint_layline_eta_s" not in nav, (run.name, nav["step"])
            continue
        assert enabled and source in {"direct gate approach", "closest reachable tack"}
        assert nav["ais_mode"] == "route" and nav.get("ais_weather_candidate_admitted") is True
        assert nav.get("drift_mode") != "holding" and nav.get("forecast_closest_m") is None
        assert math.isfinite(nav["waypoint_layline_distance_m"]) and nav["waypoint_layline_distance_m"] >= 0
        assert 0 < nav["waypoint_layline_eta_s"] <= 2880. + 1e-8
        assert (nav["waypoint_layline_second_bow_deg"] is None) == (source == "direct gate approach")
        if not nav["step"]:
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
            "scope": "Replay references retained through AIS/controller admission. Repeated held references are included."}


def audit_case(run):
    run = Path(run)
    inputs = verify_inputs(run)
    metadata = read(run / "experiment.json")
    gate = gate_audit.audit_case(run)
    history = STUDY / "independent_traffic_history"
    history.mkdir(exist_ok=True)
    clearance = native.audit((str(run), metadata["traffic_fixture"], metadata["truth_trace"], str(history)))
    route = {point.findtext("{*}name"): route_audit.gpx_position(point)
             for point in ElementTree.parse(route_audit.ROUTE_PATH).findall(".//{*}rtept")}
    providers = {name: cls(**route_audit.OTRANTO_BOUNDS).load() for name, cls in route_audit.ENVIRONMENT.items()}
    try:
        with patch.object(route_audit, "STUDY", STUDY):
            route_receipt, _ = route_audit.audit_run(run.name, run, route, native.EXPECTED, providers,
                                                     inputs["wind_from_degrees"])
    finally:
        for provider in providers.values():
            provider.close()
    with patch.object(historical, "STUDY", STUDY):
        timing = historical.native_control_timing(run)
    policy = audit_policy(run, metadata["closest_waypoint_layline"])
    truth = clearance["configured_truth_centered_minimum_m"]
    report = clearance["configured_model_minimum_m"]
    assert clearance["frames"] == route_receipt["steps"] + 1 == timing["applied_attempts"] + 1
    checks = {"normal_execution": inputs["execution_return_code"] == 0,
              "complete_ordered_8captures_50m": route_receipt["status"] == "completed" and len(route_receipt["captures"]) == 8,
              "truth_full_plus_3m_positive": truth is not None and truth > 0.,
              "report_full_plus_3m_positive": report is not None and report > 0.,
              "no_negative_clearance_intervals": not any(clearance["negative_action_counts"].values()),
              "underway_acquisition_at420s": clearance["underway_first_acquisition_encounter_elapsed_s"] == 420.,
              "both_underway65s_peaks_verified": clearance["underway_both65s_age_peaks_verified"],
              "all_gate_attempts_below5s": gate["all_gate_attempts_below_5s"],
              "all_AIS_choose_attempts_below5s": clearance["all_AIS_choose_attempts_below_5s"],
              "all_AIS_accept_or_stop_attempts_below5s": clearance["all_AIS_applied_decision_attempts_below_5s"],
              "all_full_native_control_decisions_below5s": timing["all_complete_native_control_attempts_below_5s"]}
    return {"case": run.name, "input_verification": inputs, "gate": gate, "clearance": clearance,
            "route": route_receipt, "native_control_timing": timing, "layline_policy": policy,
            "audit_source_sha256": native.digest(__file__), "acceptance_checks": checks, "passed": all(checks.values())}


def compare_replays(first, second):
    state_step = command_step = reference_step = None
    common = 0
    a_path, b_path = first / "navigation.jsonl.gz", second / "navigation.jsonl.gz"
    control = ("rudder_command_deg", "flap_command_deg")
    nominal = ("heading_reference_deg", "bow_reference_deg")
    for step, (a, b) in enumerate(zip(replay(a_path), replay(b_path))):
        a_obs, b_obs = a["observation"], b["observation"]
        assert a_obs["time"] == b_obs["time"] and a_obs["ais_targets"] == b_obs["ais_targets"], (first.name, second.name, step)
        if step == 0:
            assert a["state"] == b["state"] and a_obs == b_obs
        if a["state"] == b["state"]:
            assert all(a_obs[layer] == b_obs[layer] for layer in route_audit.ENVIRONMENT), (first.name, second.name, step)
        if state_step is None and a["state"] != b["state"]:
            state_step = step
        if command_step is None and any(a["navigation"].get(key) != b["navigation"].get(key) for key in control):
            command_step = step
        if reference_step is None and any(a["navigation"].get(key) != b["navigation"].get(key) for key in nominal):
            reference_step = step
        common += 1
    return {"received_AIS_reports_identical_at_all_common_times": True, "common_frames": common,
            "first_nominal_reference_difference_step": reference_step, "first_applied_command_difference_step": command_step,
            "first_state_difference_step": state_step}


def pair_receipt(off, on):
    runs = [STUDY / "runs" / record["case"] for record in (off, on)]
    a, b = [read(run / "experiment.json") for run in runs]
    assert normalized(a["command"]) == normalized(b["command"])
    assert a["source_sha256"] == b["source_sha256"]
    for field in MATCH_FIELDS:
        assert a[field] == b[field], field
    parity = compare_replays(*runs)
    assert parity["common_frames"] == 1 + min(off["route"]["steps"], on["route"]["steps"])
    row = {"wind_from_degrees": a["wind_from_degrees"], "off_case": off["case"], "on_case": on["case"],
           "only_guidance_flag_differs": True, **parity}
    for mode, record in zip(MODES, (off, on), strict=True):
        route = record["route"]
        row[f"{mode}_status"] = route["status"]
        row[f"{mode}_navigation_s"] = route["elapsed_s"]
        row[f"{mode}_total_from_monitoring_s"] = route["total_elapsed_s"]
        row[f"{mode}_wp002_s"] = next((mark["elapsed_s"] for mark in route["captures"]
                                      if (mark["phase"], mark["waypoint"]) == ("outbound", "002")), None)
        row[f"{mode}_applied_layline_reference_steps"] = record["layline_policy"]["applied_reference_steps"]
    row["on_minus_off_navigation_s"] = row["on_navigation_s"] - row["off_navigation_s"]
    row["on_minus_off_wp002_s"] = (row["on_wp002_s"] - row["off_wp002_s"]
                                  if row["on_wp002_s"] is not None and row["off_wp002_s"] is not None else None)
    row["complete_physical_states_identical"] = (
        off["route"]["steps"] == on["route"]["steps"] and row["first_state_difference_step"] is None)
    row["complete_applied_commands_identical"] = (
        off["route"]["steps"] == on["route"]["steps"] and row["first_applied_command_difference_step"] is None)
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
    pairs = []
    for wind in WINDS:
        off, on = (records.get(f"crossing_fan__otranto_{wind:03d}__layline_{mode}") for mode in MODES)
        if off and on:
            pairs.append(pair_receipt(off, on))
    save(STUDY / "independent_comparison.json", {"sign_convention": "On minus off, with negative faster.", "pairs": pairs})
    save_csv(STUDY / "independent_comparison.csv", pairs)
    baselines = {}
    for case, item in records.items():
        if item["input_verification"]["mode"] == "off":
            reference = reference_path(item["input_verification"]["wind_from_degrees"])
            parity = compare_replays(reference, STUDY / "runs" / case)
            baseline = read(PRIOR / "independent_case_receipts" / f"{reference.name}.json")["route"]
            parity["historical_steps"] = baseline["steps"]
            parity["fresh_steps"] = item["route"]["steps"]
            parity["complete_off_baseline_reproduction"] = (
                parity["common_frames"] == 1 + baseline["steps"] == 1 + item["route"]["steps"]
                and parity["first_state_difference_step"] is None and parity["first_applied_command_difference_step"] is None)
            baselines[case] = parity
    save(STUDY / "independent_off_baseline_reproduction.json", {"cases": baselines})
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
               "maximum_complete_control_s": max((item["native_control_timing"]["maximum_complete_native_control_decision_s"]
                                                    for item in records.values()
                                                    if item["native_control_timing"]["maximum_complete_native_control_decision_s"] is not None), default=None),
               "full_control_attempts_at_or_above_5s": sum(item["native_control_timing"]["attempts_at_or_above_5s"]
                                                          for item in records.values()),
               "both_off_arms_reproduce_prior_native_trajectories": len(baselines) == 2 and all(
                   item["complete_off_baseline_reproduction"] for item in baselines.values()),
               "audit_source_sha256": native.digest(__file__), "pinned_auditors": PINS,
               "checked_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "Ordered first 50 m endpoint captures, archived forcing, matched AIS reports, native substep clearance, and all complete timed control attempts."}
    if complete:
        with patch.object(sys, "argv", [audit_exposure.__file__, "--study", str(STUDY)]):
            audit_exposure.main()
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
    previous = read(path)
    return (previous["audit_source_sha256"] == native.digest(__file__)
            and previous["input_verification"]["execution_sha256"] == native.digest(run / "execution.json")
            and previous["clearance"]["replay_sha256"] == native.digest(run / "navigation.jsonl.gz")
            and previous["clearance"]["run_log_sha256"] == native.digest(run / "run.log"))


def closed(run):
    path = run / "execution.json"
    if not path.is_file():
        return False
    try:
        return "return_code" in read(path)
    except json.JSONDecodeError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=ARMS)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--watch", action="store_true", help="Start an audit when each selected run closes.")
    args = parser.parse_args()
    verify_shared_files()
    folder = STUDY / "independent_case_receipts"
    pending = {STUDY / "runs" / case for case in (args.case or ARMS)}
    failures = []
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        active = {}
        while pending or active:
            for run in sorted(pending.copy()):
                if not closed(run):
                    continue
                pending.remove(run)
                if current_receipt(run, folder):
                    continue
                print(f"Auditing closed run {run.name}", flush=True)
                active[pool.submit(audit_case, run)] = run
            if not args.watch:
                pending.clear()
            finished, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
            if not active and pending:
                from time import sleep
                sleep(10)
            for future in finished:
                run = active.pop(future)
                try:
                    receipt = future.result()
                except Exception as error:
                    failures.append({"case": run.name, "exception": repr(error)})
                    print(f"Audit failed: {run.name}: {error!r}", flush=True)
                    continue
                save(folder / f"{run.name}.json", receipt)
                print(f"Audited {run.name}: {receipt['route']['status']}, "
                      f"{len(receipt['route']['captures'])}/8 captures", flush=True)
                aggregate()
    if failures:
        save(STUDY / "independent_audit_failures.json", failures)
        raise RuntimeError(f"Independent audit failed for {len(failures)} cases. See independent_audit_failures.json.")
    print(json.dumps(aggregate(), indent=2), flush=True)


if __name__ == "__main__":
    main()
