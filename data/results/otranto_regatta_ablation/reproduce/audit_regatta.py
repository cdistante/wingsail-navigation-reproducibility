"""Independently audit all fresh regatta ON/OFF trajectories and matched inputs."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime
from hashlib import file_digest
import gzip
from itertools import zip_longest
import json
from pathlib import Path
import shlex
import sys
import time
from xml.etree import ElementTree

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[2]
BASE = STUDY.with_name("otranto_underway_ais_retest")
sys.path.insert(0, str(ROOT))
import audit_independent as native
import audit_departure_gate as gate_audit
import audit_route_reference as reference


BASE_ARMS = tuple(
    f"{scenario}__otranto_{wind:03d}__{method}"
    for scenario in ("crossing_fan", "opposed_crossing")
    for wind in (355, 5)
    for method in ("classical", "transformer")
)
ARMS = tuple(f"{base}__regatta_{mode}" for base in BASE_ARMS for mode in ("on", "off"))
PATH_FLAGS = (
    "--acados-build-dir", "--output-dir", "--output-gpx", "--record-replay",
    "--traffic-fixture", "--underway-traffic-fixture",
)
PROTECTED_FIELDS = (
    "encounter_id", "scenario_id", "wind_from_degrees", "wind_from", "method",
    "reporting", "original_reporting", "maximum_steps", "cap_s", "dt_s", "radius_m",
    "departure_utc", "encounter_utc", "underway_first_emission_s", "underway_first_delivery_s",
    "underway_drop_emission_windows_s", "underway_frame_retention_original_epoch_s",
    "underway_reference_elapsed_s", "underway_anchor_utc", "frozen_checkpoint_sha256",
)


def digest(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    native.save_json(path, value)


def save_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalized(command):
    words = list(command)
    words[2] = Path(words[2]).name
    if "--regatta-time-to-mark" in words:
        assert words.count("--regatta-time-to-mark") == 1
        words.remove("--regatta-time-to-mark")
    for flag in PATH_FLAGS:
        assert words.count(flag) == 1, flag
        words[words.index(flag) + 1] = flag
    return words


def raw_frames(path):
    with gzip.open(path, "rt") as stream:
        next(stream)
        for line in stream:
            yield json.loads(line)


def verify_inputs(run):
    metadata, execution = load(run / "experiment.json"), load(run / "execution.json")
    base_arm, mode = metadata["base_arm"], metadata["regatta_mode"]
    assert base_arm in BASE_ARMS and mode in ("on", "off")
    assert metadata["run_id"] == run.name == f"{base_arm}__regatta_{mode}"
    old = load(BASE / "runs" / base_arm / "execution.json")
    assert execution["command"] == metadata["command"] == shlex.split((run / "command.txt").read_text())
    command = metadata["command"]
    assert metadata["regatta_enabled"] == (mode == "on")
    assert command.count("--regatta-time-to-mark") == int(mode == "on")
    assert normalized(command) == normalized(old["command"]), run.name
    for key in PROTECTED_FIELDS:
        assert metadata[key] == old[key], (run.name, key, metadata[key], old[key])
        assert execution[key] == metadata[key], (run.name, "closure changed " + key)
    assert execution["source_sha256"] == metadata["source_sha256"]
    assert execution["source_unchanged"] and not execution["changed_inputs"]
    for name, checksum in execution["source_sha256"].items():
        assert digest(ROOT / name) == checksum == digest(run / "source_snapshot" / name), (run.name, name)
        if name in old["source_sha256"] and (name == "config.py" or name.startswith("world/")):
            assert old["source_sha256"][name] == checksum, (run.name, "production baseline mismatch", name)
    frozen_paths = (
        ("traffic_fixture", "traffic_fixture_sha256"),
        ("truth_trace", "traffic_truth_trace_sha256"),
        ("underway_traffic_fixture", "underway_traffic_fixture_sha256"),
        ("underway_truth_trace_csv", "underway_truth_trace_sha256"),
    )
    for field, sha_key in frozen_paths:
        assert digest(metadata[field]) == metadata[sha_key] == old[sha_key] == digest(old[field])
    for name in ("protocol.json", "underway_config.json", "archive_field_fingerprints.json"):
        assert (STUDY / name).read_bytes() == (BASE / name).read_bytes(), name
    before = next(raw_frames(BASE / "runs" / base_arm / "navigation.jsonl.gz"))
    after = next(raw_frames(run / "navigation.jsonl.gz"))
    assert after["state"] == before["state"] and after["observation"] == before["observation"]
    gate = load(run / "departure_gate.json")
    old_gate = load(BASE / "runs" / base_arm / "departure_gate.json")
    assert gate["actual_departure_utc"] == old_gate["actual_departure_utc"]
    assert gate["gate_wait_seconds"] == old_gate["gate_wait_seconds"]
    for field in ("monitoring_start_utc", "traffic_epoch_utc", "parameters"):
        assert gate[field] == old_gate[field]
    assert reference.utc(after["observation"]["time"]) == reference.utc(gate["actual_departure_utc"])
    return {
        "case": run.name, "base_arm": base_arm, "regatta_mode": mode,
        "production_source_and_configuration_match_preserved_underway_run": True,
        "only_regatta_toggle_changes_behavior_arguments": True,
        "all_truth_fixture_and_fault_bytes_match": True,
        "initial_state_and_archived_forcing_match": True,
        "actual_launch_matches_preserved_underway_run": True,
        "actual_departure_utc": gate["actual_departure_utc"], "gate_wait_seconds": gate["gate_wait_seconds"],
        "frozen_source_files": len(metadata["source_sha256"]),
        "execution_return_code": execution.get("return_code", execution.get("exit_code")),
        "command_sha256": digest(run / "command.txt"),
        "experiment_sha256": digest(run / "experiment.json"),
        "execution_sha256": digest(run / "execution.json"),
    }


def validate_route(run, clearance):
    metadata = load(run / "experiment.json")
    route = {
        point.findtext("{*}name"): reference.gpx_position(point)
        for point in ElementTree.parse(reference.ROUTE_PATH).findall(".//{*}rtept")
    }
    providers = {name: cls(**reference.OTRANTO_BOUNDS).load() for name, cls in reference.ENVIRONMENT.items()}
    try:
        result, replay = reference.audit_run(
            run.name, run, route, native.EXPECTED, providers, metadata["wind_from_degrees"],
        )
    finally:
        for provider in providers.values():
            provider.close()
    assert clearance["frames"] == len(replay)
    assert clearance["replay_sha256"] == digest(run / "navigation.jsonl.gz")
    counters = Counter()
    for frame in replay[1:]:
        nav = frame["navigation"]
        counters["holding_mode_updates"] += nav.get("drift_mode") == "holding"
        counters["actual_drift_trim_updates"] += nav.get("control_source") == "drift trim"
        counters["colreg_active_updates"] += nav.get("ais_colreg_active_contacts", 0) > 0
        counters["race_guidance_updates"] += nav.get("race_guidance") is not None
        counters["learned_policy_applied_updates"] += nav.get("temporal_recovery_source") in ("learned", "physical release")
        if nav.get("temporal_recovery_source") in ("learned", "physical release"):
            assert nav.get("ais_mode") == "route" and nav.get("ais_colreg_active_contacts", 0) == 0
            assert nav.get("solver_fallback") is None and nav.get("nominal_guidance_suspended") is None
        if metadata["regatta_mode"] == "off":
            assert nav.get("race_guidance") is None and nav.get("race_forecast_search_wall_s") is None
    result.update(
        scenario_id=metadata["scenario_id"], method=metadata["method"], base_arm=metadata["base_arm"],
        regatta_mode=metadata["regatta_mode"], independent_clearance=clearance,
        arbitration=dict(counters),
    )
    return result


def audited_closed(task):
    run = Path(task)
    history = STUDY / "independent_traffic_history"
    history.mkdir(exist_ok=True)
    inputs = verify_inputs(run)
    gate = gate_audit.audit_case(run)
    metadata = load(run / "experiment.json")
    result = native.audit((str(run), metadata["traffic_fixture"], metadata["truth_trace"], str(history), None))
    route = validate_route(run, result)
    return {"case": run.name, "input_verification": inputs, "gate": gate, "clearance": result, "route": route}


def record_row(record):
    result, gate, route = record["clearance"], record["gate"], record["route"]
    captures = route["captures"]
    def captured(phase, target):
        return next((row["elapsed_s"] for row in captures if row["phase"] == phase and row["waypoint"] == target), None)
    checks = {
        "normal_execution": record["input_verification"]["execution_return_code"] == 0,
        "capture_order_verified": True,
        "complete_loop": route["status"] == "completed" and len(captures) == 8,
        "truth_full_uncertainty_positive": result["truth_centered_minimum_uncertainty_m"] is not None
            and result["truth_centered_minimum_uncertainty_m"] > 0,
        "truth_full_plus_reserve_positive": result["configured_truth_centered_minimum_m"] is not None
            and result["configured_truth_centered_minimum_m"] > 0,
        "report_full_uncertainty_positive": result["model_minimum_uncertainty_m"] is not None
            and result["model_minimum_uncertainty_m"] > 0,
        "report_full_plus_reserve_positive": result["configured_model_minimum_m"] is not None
            and result["configured_model_minimum_m"] > 0,
        "all_clearance_violations_zero": all(value == 0 for value in result["negative_action_counts"].values()),
        "both_65s_peaks_verified": result["underway_both65s_age_peaks_verified"],
        "new_contact_acquired_at_420s": result["underway_first_acquisition_encounter_elapsed_s"] == 420,
        "all_gate_attempts_below_5s": gate["all_gate_attempts_below_5s"],
        "all_choose_attempts_below_5s": result["all_AIS_choose_attempts_below_5s"],
        "all_complete_selection_attempts_below_5s": result["all_AIS_applied_decision_attempts_below_5s"],
        "startup_displacement_within_observed_gate_tube": result["observed_launch_displacement_tube_obeyed_first300s"],
    }
    row = {
        "arm": record["case"], "base_arm": route["base_arm"], "regatta_mode": route["regatta_mode"],
        "scenario_id": route["scenario_id"], "wind_from_degrees": route["wind_from_degrees"], "method": route["method"],
        "status": route["status"], "captures": len(captures), "capture_order_verified": True,
        "wp002_s": captured("outbound", "002"), "outbound_s": captured("outbound", "005"),
        "navigation_s": route["elapsed_s"], "gate_wait_s": gate["gate_wait_seconds"],
        "total_s": route["total_elapsed_s"], "actual_departure_utc": gate["actual_departure_utc"],
        "truth_hull_m": result["truth_minimum_hull_m"], "truth_nominal_m": result["truth_minimum_nominal_m"],
        "truth_full_m": result["truth_centered_minimum_uncertainty_m"],
        "truth_reserve_m": result["configured_truth_centered_minimum_m"],
        "report_full_m": result["model_minimum_uncertainty_m"], "report_reserve_m": result["configured_model_minimum_m"],
        "gate_max_s": gate["maximum_gate_decision_s"], "gate_attempts": gate["gate_attempts"],
        "choose_max_s": result["maximum_complete_AIS_choose_s"],
        "choose_overruns": result["AIS_choose_attempts_at_or_above_5s"],
        "selection_max_s": result["maximum_complete_AIS_applied_decision_s"],
        "selection_overruns": result["AIS_applied_decision_attempts_at_or_above_5s"],
        "unapplied_final_attempts": result["AIS_applied_decision_unapplied_final_attempts"],
        "maximum_held_age_s": result["maximum_applied_held_report_age_s"],
        "fault_peaks_verified": result["underway_both65s_age_peaks_verified"],
        "first_contact_acquisition_s": result["underway_first_acquisition_encounter_elapsed_s"],
        "underway_active_ais_or_colreg_steps": result["underway_active_ais_or_colreg_steps_420_to_900_s"],
        "maximum_native_pose_error_m": result["maximum_pose_reconstruction_error_m"],
        "maximum_archive_error": route["maximum_archive_value_difference"],
        "all_independent_acceptance_checks": all(checks.values()),
    }
    return row, {"case": record["case"], "passed": all(checks.values()), "checks": checks}


def pair_trajectory(on, off, prefix_s=900):
    first = second = None
    prefix_equal = True
    first_command = first_reference = first_state = None
    all_until = 0
    control = ("rudder_command_deg", "flap_command_deg")
    refs = ("heading_reference_deg", "bow_reference_deg", "forecast_closest_m", "forecast_tack_in_min")
    for step, (a, b) in enumerate(zip(raw_frames(on), raw_frames(off))):
        if step == 0:
            assert a["state"] == b["state"] and a["observation"] == b["observation"]
        nav_a, nav_b = a["navigation"], b["navigation"]
        command_a, command_b = [tuple(nav.get(key) for key in control) for nav in (nav_a, nav_b)]
        if first_command is None and command_a != command_b:
            first_command = step
        if first_state is None and a["state"] != b["state"]:
            first_state = step
        if first_reference is None and any(nav_a.get(key) != nav_b.get(key) for key in refs):
            first_reference = step
        if step * 5 <= prefix_s:
            all_until = step * 5
            prefix_equal &= a["state"] == b["state"] and a["observation"] == b["observation"] and command_a == command_b
    return {
        "prefix_900_s": all_until, "prefix_900_identical_physics_and_commands": prefix_equal and all_until == prefix_s,
        "first_applied_command_difference_step": first_command,
        "first_applied_command_difference_start_s": (first_command - 1) * 5 if first_command is not None else None,
        "first_reference_difference_step": first_reference, "first_state_difference_step": first_state,
    }


def compare_pair(base_arm, records):
    on_id, off_id = [f"{base_arm}__regatta_{mode}" for mode in ("on", "off")]
    on, off = [records[key] for key in (on_id, off_id)]
    row_on, check_on = record_row(on)
    row_off, check_off = record_row(off)
    meta_on, meta_off = [load(STUDY / "runs" / key / "experiment.json") for key in (on_id, off_id)]
    assert meta_on["source_sha256"] == meta_off["source_sha256"]
    assert normalized(meta_on["command"]) == normalized(meta_off["command"])
    for key in PROTECTED_FIELDS:
        assert meta_on[key] == meta_off[key]
    # Gate wall timers vary by execution, while the observed receiver clock and certificate stay fixed.
    for key in ("actual_departure_utc", "gate_wait_seconds"):
        assert on["gate"][key] == off["gate"][key]
    gate_on, gate_off = [load(STUDY / "runs" / key / "departure_gate.json") for key in (on_id, off_id)]
    assert gate_on["parameters"] == gate_off["parameters"]
    paired = {
        "base_arm": base_arm, "scenario_id": row_on["scenario_id"], "wind_from_degrees": row_on["wind_from_degrees"],
        "method": row_on["method"], "on_arm": on_id, "off_arm": off_id,
        **{f"{mode}_{key}": value for mode, row in (("on", row_on), ("off", row_off))
           for key, value in row.items() if key not in ("arm", "base_arm", "scenario_id", "wind_from_degrees", "method", "regatta_mode")},
        "on_minus_off_wp002_s": None if row_on["wp002_s"] is None or row_off["wp002_s"] is None else row_on["wp002_s"] - row_off["wp002_s"],
        "on_minus_off_outbound_s": None if row_on["outbound_s"] is None or row_off["outbound_s"] is None else row_on["outbound_s"] - row_off["outbound_s"],
        "on_minus_off_navigation_s": row_on["navigation_s"] - row_off["navigation_s"],
        "on_minus_off_total_s": row_on["total_s"] - row_off["total_s"],
        "identical_production_sources_and_behavior_arguments_except_flag": True,
        "identical_clock_forcing_and_actual_launch": True,
        **pair_trajectory(STUDY / "runs" / on_id / "navigation.jsonl.gz", STUDY / "runs" / off_id / "navigation.jsonl.gz"),
    }
    return paired


def publish(records, failures, final=False):
    rows, acceptance = zip(*(record_row(records[key]) for key in sorted(records))) if records else ([], [])
    save_csv(STUDY / "independent_arm_outcomes.csv", rows)
    save(STUDY / "independent_traffic_sweep.json", [records[key]["clearance"] for key in sorted(records)])
    gates = [records[key]["gate"] for key in sorted(records)]
    save(STUDY / "independent_departure_gate_check.json", {
        "all_available_gates_verified": True, "all_gate_attempts_below_5s": all(gate["all_gate_attempts_below_5s"] for gate in gates),
        "cases": gates,
    })
    comparisons = [
        compare_pair(base, records) for base in BASE_ARMS
        if all(f"{base}__regatta_{mode}" in records for mode in ("on", "off"))
    ]
    save_csv(STUDY / "independent_comparison.csv", comparisons)
    save(STUDY / "independent_comparison.json", {
        "sign_convention": "Regatta ON minus OFF. Negative time delta favors ON.",
        "closed_verified_runs": len(records), "expected_runs": len(ARMS), "paired_comparisons": comparisons,
        "complete_matrix": final and len(records) == len(ARMS), "audit_failures": failures,
    })
    save(STUDY / "independent_acceptance_check.json", {
        "all_outcomes_retained": True, "all_passed": len(records) == len(ARMS) and not failures
            and all(row["passed"] for row in acceptance),
        "cases": list(acceptance), "audit_failures": failures,
        "timing_scope": "Gate, complete AIS choose, and choice-through-action-acceptance timers checked separately against five seconds.",
    })
    save(STUDY / "comparison.json", {
        "radius_m": 50, "dt_s": 5, "maximum_steps": 17280,
        "runs": {key: records[key]["route"] for key in sorted(records)},
        "regatta_on_off_pairs": {row["base_arm"]: row for row in comparisons},
        "complete_matrix": final and len(records) == len(ARMS),
    })
    save(STUDY / "independent_inputs_verification.json", {
        "cases": [records[key]["input_verification"] for key in sorted(records)],
        "all_closed_verified": len(records) == len(ARMS) and not failures,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--release-actor-slots", action="store_true",
                        help="Use at most two audit workers until actor closures release extra CPU slots.")
    parser.add_argument("--poll-seconds", type=float, default=10)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    assert args.jobs > 0 and args.poll_seconds >= 1
    source = Path(__file__)
    for module in (native, reference, gate_audit):
        assert Path(module.__file__).read_bytes() == (BASE / "reproduce" / Path(module.__file__).name).read_bytes()
    snapshots = STUDY / "independent_source_snapshots"
    snapshots.mkdir(exist_ok=True)
    for script in (source, Path(native.__file__), Path(reference.__file__), Path(gate_audit.__file__)):
        snapshot = snapshots / (digest(script) + "_" + script.name)
        if snapshot.exists():
            assert snapshot.read_bytes() == script.read_bytes()
        else:
            snapshot.write_bytes(script.read_bytes())
    receipt_dir = STUDY / "independent_case_receipts"
    receipt_dir.mkdir(exist_ok=True)
    records, failures = {}, []
    for receipt in receipt_dir.glob("*.json"):
        assert args.reuse, "Use --reuse to keep previously completed independent audit receipts."
        record = load(receipt)
        run = STUDY / "runs" / record["case"]
        assert record["clearance"]["replay_sha256"] == digest(run / "navigation.jsonl.gz")
        assert record["clearance"]["run_log_sha256"] == digest(run / "run.log")
        records[record["case"]] = record
    submitted, pending = set(records), {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        while len(records) + len(failures) < len(ARMS):
            closed = sum((STUDY / "runs" / case / "execution.json").is_file() for case in ARMS)
            allowed = min(args.jobs, max(2, closed)) if args.release_actor_slots else args.jobs
            for case in ARMS:
                if len(pending) >= allowed:
                    break
                run = STUDY / "runs" / case
                if case in submitted or not (run / "execution.json").exists():
                    continue
                pending[pool.submit(audited_closed, str(run))] = case
                submitted.add(case)
                print("Started native and archive audit", case, flush=True)
            for future, case in list(pending.items()):
                if not future.done():
                    continue
                pending.pop(future)
                try:
                    record = future.result()
                    assert record["case"] == case
                    save(receipt_dir / f"{case}.json", record)
                    records[case] = record
                    publish(records, failures)
                    row, check = record_row(record)
                    print(f"Audited {case}: {row['status']}, {row['captures']}/8, "
                          f"WP002={row['wp002_s']}, full={row['truth_full_m']}, reserve={row['truth_reserve_m']}, "
                          f"selection={row['selection_max_s']}", flush=True)
                except Exception as error:
                    failure = {"case": case, "exception": repr(error), "source_sha256": digest(source)}
                    failures.append(failure)
                    save(STUDY / "independent_audit_failures.json", failures)
                    print("Audit failure retained", failure, flush=True)
            if not args.watch and not pending:
                break
            if len(records) + len(failures) < len(ARMS):
                time.sleep(args.poll_seconds)
    publish(records, failures, final=len(records) + len(failures) == len(ARMS))
    print(f"Independent audits closed: {len(records)} passed numerical reconstruction, {len(failures)} exceptions.", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
