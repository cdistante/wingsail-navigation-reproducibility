"""Independent native round-trip, forcing, traffic, timing and pairing audits.

The four numerical auditors beside this script are byte-identical copies of
the preceding native regatta audit. Failed/early-stop runs remain outcomes.
Structural audit exceptions are recorded and prevent an acceptance claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from hashlib import sha256
import gzip
from itertools import zip_longest
import json
from pathlib import Path
import shlex
import sys
from xml.etree import ElementTree

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[2]
sys.path.insert(0, str(ROOT))
import audit_independent as native
import audit_departure_gate as gate_audit
import audit_route_reference as route_audit

BASE = STUDY.with_name("otranto_regatta_ablation")
ARMS = tuple(f"crossing_fan__otranto_{wind:03d}__{mode}" for wind in (5, 355) for mode in ("physics", "hybrid"))
PROTECTED = (
    "encounter_id", "scenario_id", "wind_from_degrees", "wind_from", "method", "reporting",
    "original_reporting", "maximum_steps", "cap_s", "dt_s", "radius_m", "departure_utc", "encounter_utc",
    "underway_first_emission_s", "underway_first_delivery_s", "underway_drop_emission_windows_s",
    "underway_frame_retention_original_epoch_s", "underway_reference_elapsed_s", "underway_anchor_utc",
)
PINS = {
    "audit_independent.py": "2b4c281813ac4ed3f0852a92844cbf8b9cef4b10b26397d497f827775ee07d77",
    "audit_departure_gate.py": "daa0287b9bc8118cbb1faf3018ee6b517dac8812f7ffb2e4afbb6fbf62f1ce25",
    "audit_exposure.py": "ce3e9b09ed1ab5e36630067bc9c3e64d5047313599d2d99bf59b304dea6d6b00",
    "audit_route_reference.py": "a2e6989155f544375f49b78c2598c40d68997aca10d887f834992be61501465d",
}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    native.save_json(Path(path), value)


def save_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields); writer.writeheader(); writer.writerows(rows)


def frames(path):
    with gzip.open(path, "rt") as stream:
        next(stream)
        yield from (json.loads(line) for line in stream)


def normalized(command, *, remove_common=False):
    values = list(command)
    values[2] = "run_archive.py"
    if "--hybrid-forecast-tacks" in values:
        values.remove("--hybrid-forecast-tacks")
        index = values.index("--hybrid-model-dir"); del values[index:index+2]
    if remove_common:
        for flag in ("--native-control-timing", "--ais-candidate-handoff"):
            assert values.count(flag) == 1
            values.remove(flag)
    for flag in ("--acados-build-dir", "--output-dir", "--output-gpx", "--record-replay", "--traffic-fixture", "--underway-traffic-fixture"):
        assert values.count(flag) == 1
        values[values.index(flag)+1] = flag
    return values


def verify_inputs(run):
    metadata, execution = load(run/"experiment.json"), load(run/"execution.json")
    mode = metadata["native_hybrid_mode"]
    assert run.name in ARMS and mode in ("physics", "hybrid")
    assert metadata["run_id"] == run.name and metadata["native_hybrid_enabled"] == (mode == "hybrid")
    assert metadata["method"] == "classical" and metadata["regatta_enabled"] is False
    assert metadata["native_control_timing"] and metadata["native_AIS_candidate_handoff"]
    assert execution["command"] == metadata["command"] == shlex.split((run/"command.txt").read_text())
    assert metadata["command"].count("--hybrid-forecast-tacks") == int(mode == "hybrid")
    old_run = Path(metadata["baseline_directory"])
    old = load(old_run/"execution.json")
    assert digest(old_run/"execution.json") == metadata["baseline_execution_sha256"]
    assert digest(old_run/"navigation.jsonl.gz") == metadata["baseline_replay_sha256"]
    assert normalized(metadata["command"], remove_common=True) == normalized(old["command"])
    for key in PROTECTED:
        assert metadata[key] == old[key] == execution[key], (run.name, key)
    assert execution["source_sha256"] == metadata["source_sha256"]
    assert execution["source_unchanged"] and not execution["changed_inputs"]
    for key, expected in metadata["source_sha256"].items():
        assert digest(run/"source_snapshot"/key) == expected, (run.name, key, "snapshot")
        assert digest(ROOT/key) == expected, (run.name, key, "live source changed")
    for key, expected in metadata["native_common_source_sha256"].items():
        assert metadata["source_sha256"][key] == expected
    assert set(metadata["changes_from_original_sources"]) == {"world/run_legacy_fossen.py", "world/ais_buffer_maneuver.py"}
    for field, sha_key in (("traffic_fixture", "traffic_fixture_sha256"), ("truth_trace", "traffic_truth_trace_sha256"),
                           ("underway_traffic_fixture", "underway_traffic_fixture_sha256"),
                           ("underway_truth_trace_csv", "underway_truth_trace_sha256")):
        assert digest(metadata[field]) == metadata[sha_key] == old[sha_key] == digest(old[field])
    for name in ("protocol.json", "underway_config.json", "archive_field_fingerprints.json"):
        assert (STUDY/name).read_bytes() == (BASE/name).read_bytes()
    gate, prior_gate = load(run/"departure_gate.json"), load(old_run/"departure_gate.json")
    for key in ("monitoring_start_utc", "traffic_epoch_utc", "parameters", "actual_departure_utc", "gate_wait_seconds"):
        assert gate[key] == prior_gate[key], (run.name, key)
    first, prior = next(frames(run/"navigation.jsonl.gz")), next(frames(old_run/"navigation.jsonl.gz"))
    assert first["state"] == prior["state"] and first["observation"] == prior["observation"]
    return {"case": run.name, "mode": mode, "execution_return_code": execution.get("return_code", execution.get("exit_code")),
            "shared_protocol_archive_fault_bytes_verified": True, "initial_state_and_environment_match_baseline": True,
            "launch_and_monitoring_clock_match_baseline": True, "frozen_snapshot_and_live_source_match": True,
            "native_hybrid_checkpoint_sha256": metadata["native_hybrid_checkpoint_sha256"],
            "experiment_sha256": digest(run/"experiment.json"), "execution_sha256": digest(run/"execution.json")}


def native_control_timing(run):
    prefix = "Native control decision timing: "
    lines = (run/"run.log").read_text().splitlines()
    rows = [json.loads(line.removeprefix(prefix)) for line in lines if line.startswith(prefix)]
    ais = [json.loads(line.removeprefix("AIS applied decision timing: ")) for line in lines if line.startswith("AIS applied decision timing: ")]
    previous = None
    applied = 0
    for frame in frames(run/"navigation.jsonl.gz"):
        if previous is None:
            previous = frame; continue
        row = rows[applied]
        applied += 1
        nav = frame["navigation"]
        assert row["step"] == applied == nav["step"]
        assert row["applied"] is True
        assert row["phase"] == nav["phase"] and row["waypoint"] == nav["waypoint"]
        assert np.datetime64(row["observation_utc"], "ns") == np.datetime64(previous["observation"]["time"], "ns")
        assert abs(row["wall_seconds"]-nav["native_control_decision_seconds"]) < 1e-12
        previous = frame
    assert len(rows) in (applied, applied+1)
    assert [row["step"] for row in rows] == list(range(1, len(rows)+1))
    assert len(ais) == len(rows), "Every native decision must have an AIS accept/stop attempt."
    for row, counterpart in zip(rows, ais, strict=True):
        assert np.isfinite(row["wall_seconds"]) and row["wall_seconds"] >= 0.
        assert row["step"] == counterpart["step"] and row["applied"] == counterpart["applied"]
        assert row["phase"] == counterpart["phase"] and row["waypoint"] == counterpart["waypoint"]
        assert row["observation_utc"] == counterpart["observation_utc"]
        assert row["wall_seconds"] >= counterpart["wall_seconds"]
    if len(rows) > applied:
        assert rows[-1]["applied"] is False
    folder = STUDY/"independent_native_control_timing"; folder.mkdir(exist_ok=True)
    path = folder/(run.name+".csv"); save_csv(path, rows)
    values = [row["wall_seconds"] for row in rows]
    return {"case": run.name, "attempts": len(rows), "applied_attempts": applied,
            "unapplied_terminal_attempts": len(rows)-applied,
            "maximum_complete_native_control_decision_s": max(values, default=None),
            "attempts_at_or_above_5s": sum(value >= 5. for value in values),
            "all_complete_native_control_attempts_below_5s": bool(values) and all(value < 5. for value in values),
            "all_applied_timers_and_AIS_acceptance_attempts_bound": True,
            "overrun_or_unapplied_records": [row for row in rows if row["wall_seconds"] >= 5. or not row["applied"]],
            "timing_csv_sha256": digest(path), "run_log_sha256": digest(run/"run.log"),
            "replay_sha256": digest(run/"navigation.jsonl.gz"),
            "scope": "Before world.observe/local map and all upstream guidance, controller and AIS selection through command acceptance or strict stop; excludes plant integration and replay I/O."}


def audit_case(run_path):
    run = Path(run_path)
    inputs = verify_inputs(run)
    gate = gate_audit.audit_case(run)
    metadata = load(run/"experiment.json")
    history = STUDY/"independent_traffic_history"; history.mkdir(exist_ok=True)
    clearance = native.audit((str(run), metadata["traffic_fixture"], metadata["truth_trace"], str(history), None))
    route = {p.findtext("{*}name"): route_audit.gpx_position(p)
             for p in ElementTree.parse(route_audit.ROUTE_PATH).findall(".//{*}rtept")}
    providers = {name: cls(**route_audit.OTRANTO_BOUNDS).load() for name, cls in route_audit.ENVIRONMENT.items()}
    try:
        route_result, _ = route_audit.audit_run(run.name, run, route, native.EXPECTED, providers, metadata["wind_from_degrees"])
    finally:
        for provider in providers.values(): provider.close()
    timing = native_control_timing(run)
    unique_policy, cached_frames = {}, Counter()
    for frame in frames(run/"navigation.jsonl.gz"):
        nav = frame["navigation"]
        if nav.get("hybrid_tack_decision") is not None:
            cached_frames[nav["hybrid_tack_decision"]] += 1
        when = nav.get("hybrid_tack_decision_utc")
        if when is not None:
            details = {key: value for key, value in nav.items() if key.startswith("hybrid_tack_")}
            if when in unique_policy:
                assert unique_policy[when] == details, (run.name, when, "Cached policy decision changed")
            else:
                unique_policy[when] = details
            if details["hybrid_tack_decision"] == "accepted first trim":
                assert details["hybrid_tack_threshold_s"] == 180.
                learned = details["hybrid_tack_learned_eta_s"]
                physical = details["hybrid_tack_physics_eta_s"]
                assert learned is not None and (physical is None or physical-learned > 180.+1e-6)
    policy = {"unique_timestamped_decisions": len(unique_policy),
              "unique_decision_outcomes": dict(Counter(d["hybrid_tack_decision"] for d in unique_policy.values())),
              "frames_with_cached_decision_outcomes": dict(cached_frames),
              "unique_decisions": unique_policy,
              "scope": "Timestamped upstream tack-ranking decisions; cached frame counts are not independent calls or guarantees that AIS admitted the reference."}
    assert clearance["frames"] == route_result["steps"]+1
    checks = {"normal_execution": inputs["execution_return_code"] == 0,
              "complete_ordered_8captures_50m": route_result["status"] == "completed" and len(route_result["captures"]) == 8,
              "truth_full_plus_3m_positive": clearance["configured_truth_centered_minimum_m"] is not None and clearance["configured_truth_centered_minimum_m"] > 0.,
              "report_full_plus_3m_positive": clearance["configured_model_minimum_m"] is not None and clearance["configured_model_minimum_m"] > 0.,
              "no_negative_clearance_intervals": not any(clearance["negative_action_counts"].values()),
              "underway_acquisition_at420s": clearance["underway_first_acquisition_encounter_elapsed_s"] == 420.,
              "both_underway65s_peaks_verified": clearance["underway_both65s_age_peaks_verified"],
              "all_gate_attempts_below5s": gate["all_gate_attempts_below_5s"],
              "all_AIS_choose_attempts_below5s": clearance["all_AIS_choose_attempts_below_5s"],
              "all_AIS_accept_or_stop_attempts_below5s": clearance["all_AIS_applied_decision_attempts_below_5s"],
              "all_full_native_control_decisions_below5s": timing["all_complete_native_control_attempts_below_5s"]}
    return {"case": run.name, "input_verification": inputs, "gate": gate, "clearance": clearance,
            "route": route_result, "native_control_timing": timing, "hybrid_policy": policy,
            "audit_source_sha256": digest(__file__), "acceptance_checks": checks, "passed": all(checks.values())}


def pair_result(physics_run, hybrid_run, records):
    a, b = load(physics_run/"experiment.json"), load(hybrid_run/"experiment.json")
    assert normalized(a["command"]) == normalized(b["command"])
    assert a["native_common_source_sha256"] == b["native_common_source_sha256"]
    for field in PROTECTED: assert a[field] == b[field]
    first_command = first_state = None
    common = 0
    for index, (x, y) in enumerate(zip(frames(physics_run/"navigation.jsonl.gz"), frames(hybrid_run/"navigation.jsonl.gz"))):
        assert x["observation"]["time"] == y["observation"]["time"]
        assert x["observation"]["ais_targets"] == y["observation"]["ais_targets"], (physics_run.name, index, "Received report clock differs")
        if index == 0: assert x["state"] == y["state"] and x["observation"] == y["observation"]
        if first_command is None and any(x["navigation"].get(k) != y["navigation"].get(k) for k in ("rudder_command_deg", "flap_command_deg")): first_command = index
        if first_state is None and x["state"] != y["state"]: first_state = index
        common += 1
    p, h = records[physics_run.name]["route"], records[hybrid_run.name]["route"]
    def time(result, phase, name):
        return next((e["elapsed_s"] for e in result["captures"] if (e["phase"], e["waypoint"]) == (phase, name)), None)
    result = {"wind_from_degrees": a["wind_from_degrees"], "physics_case": physics_run.name, "hybrid_case": hybrid_run.name,
              "normalized_inputs_and_common_source_identical": True, "received_AIS_reports_identical_at_all_common_times": True,
              "common_frames": common, "first_applied_command_difference_step": first_command,
              "first_state_difference_step": first_state, "physics_status": p["status"], "hybrid_status": h["status"],
              "physics_navigation_s": p["elapsed_s"], "hybrid_navigation_s": h["elapsed_s"],
              "hybrid_minus_physics_navigation_s": h["elapsed_s"]-p["elapsed_s"],
              "physics_wp002_s": time(p, "outbound", "002"), "hybrid_wp002_s": time(h, "outbound", "002")}
    if result["physics_wp002_s"] is not None and result["hybrid_wp002_s"] is not None:
        result["hybrid_minus_physics_wp002_s"] = result["hybrid_wp002_s"]-result["physics_wp002_s"]
    return result


def aggregate():
    folder = STUDY/"independent_case_receipts"
    records = {p.stem: load(p) for p in folder.glob("*.json")}
    clearance = [r["clearance"] for r in records.values()]
    save(STUDY/"independent_traffic_sweep.json", clearance)
    save(STUDY/"independent_departure_gate_check.json", {"all_available_gates_verified": True,
         "all_gate_attempts_below_5s": all(r["gate"]["all_gate_attempts_below_5s"] for r in records.values()),
         "cases": [r["gate"] for r in records.values()]})
    save(STUDY/"independent_native_control_timing.json", {"scope": "All complete native control decisions, including unapplied terminal attempts.",
         "cases": {k: r["native_control_timing"] for k, r in records.items()}})
    save(STUDY/"independent_route_verification.json", {"cases": {k: r["route"] for k, r in records.items()},
         "scope": "All saved fixes, GPX timestamps, ordered first50m endpoint captures and every archived marine forcing sample."})
    save(STUDY/"independent_hybrid_policy_telemetry.json", {"cases": {k: r["hybrid_policy"] for k, r in records.items()}})
    save(STUDY/"independent_inputs_verification.json", {"cases": {k: r["input_verification"] for k, r in records.items()}})
    save(STUDY/"independent_acceptance_check.json", {"all_passed": bool(records) and all(r["passed"] for r in records.values()),
         "cases": [{"case": k, "passed": r["passed"], "checks": r["acceptance_checks"]} for k, r in records.items()]})
    runs = {k: {**r["route"], "native_hybrid_mode": r["input_verification"]["mode"], "independent_clearance": r["clearance"],
                "native_control_timing": r["native_control_timing"]} for k, r in records.items()}
    save(STUDY/"comparison.json", {"runs": runs})
    pairs = []
    for wind in (5, 355):
        p, h = (f"crossing_fan__otranto_{wind:03d}__{mode}" for mode in ("physics", "hybrid"))
        if p in records and h in records:
            pairs.append(pair_result(STUDY/"runs"/p, STUDY/"runs"/h, records))
    save(STUDY/"independent_comparison.json", {"sign_convention": "Hybrid minus physics; negative is faster.", "pairs": pairs})
    save_csv(STUDY/"independent_comparison.csv", pairs)
    summary = {"expected_runs": len(ARMS), "closed_independently_audited_runs": len(records),
               "complete_matrix": set(records) == set(ARMS), "all_acceptance_checks_passed": bool(records) and all(r["passed"] for r in records.values()),
               "verified_captures": sum(len(r["route"]["captures"]) for r in records.values()),
               "applied_commands": sum(r["applied_actions"] for r in clearance),
               "native_0p05s_substeps": sum(r["applied_actions"]*100 for r in clearance),
               "per_case_checks": {k: r["acceptance_checks"] for k, r in records.items()},
               "minimum_truth_full_plus_reserve_m": min((r["configured_truth_centered_minimum_m"] for r in clearance if r["configured_truth_centered_minimum_m"] is not None), default=None),
               "minimum_report_full_plus_reserve_m": min((r["configured_model_minimum_m"] for r in clearance if r["configured_model_minimum_m"] is not None), default=None),
               "audit_source_sha256": digest(__file__), "pinned_auditors": PINS,
               "timing_scope": "Common full control decision includes observation/map/guidance/controller/AIS through acceptance, excluding plant integration and replay I/O. Physics/hybrid host launches are staggered, so latency comparisons are contextual.",
               "preacquisition_scope": "Fourth contact truth is swept before acquisition with explicitly hypothetical65s allowance; no fabricated reports.",
               "scope": "Native 50m, eight-capture Otranto round trips; gate/forcing/traffic/control acceptance independently bound."}
    if set(records) == set(ARMS):
        import audit_exposure
        previous_argv = sys.argv
        try:
            sys.argv = [str(Path(audit_exposure.__file__)), "--study", str(STUDY)]
            audit_exposure.main()
        finally:
            sys.argv = previous_argv
        faults = load(STUDY/"fault_exposure_check.json")
        summary["both_underway65s_peaks_experienced_in_all_runs"] = faults["all_underway_peak_pairs_experienced"]
        summary["both_full40s_underway_lost_delivery_effects_experienced_in_all_runs"] = all(
            row["both_full40s_underway_lost_delivery_effects_experienced"] for row in faults["cases"])
        summary["fault_exposure_sha256"] = digest(STUDY/"fault_exposure_check.json")
    save(STUDY/"independent_final_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true", help="Rebind existing immutable numerical receipts and rebuild summaries.")
    args = parser.parse_args()
    snapshot = STUDY/"audit_source_snapshots"/(digest(__file__)+".py")
    snapshot.parent.mkdir(exist_ok=True)
    if snapshot.exists(): assert snapshot.read_bytes() == Path(__file__).read_bytes()
    else: snapshot.write_bytes(Path(__file__).read_bytes())
    for name, expected in PINS.items(): assert digest(Path(__file__).with_name(name)) == expected
    folder = STUDY/"independent_case_receipts"; folder.mkdir(exist_ok=True)
    if args.aggregate_only:
        for case in ARMS:
            run = STUDY/"runs"/case
            verify_inputs(run)
            receipt = load(folder/(case+".json"))
            snapshot = STUDY/"audit_source_snapshots"/(receipt["audit_source_sha256"]+".py")
            assert digest(snapshot) == receipt["audit_source_sha256"]
            for key, path in (("replay_sha256", run/"navigation.jsonl.gz"), ("run_log_sha256", run/"run.log"),
                              ("history_sha256", STUDY/"independent_traffic_history"/(case+".csv"))):
                assert digest(path) == receipt["clearance"][key]
            assert digest(STUDY/"independent_native_control_timing"/(case+".csv")) == receipt["native_control_timing"]["timing_csv_sha256"]
        print(json.dumps(aggregate(), indent=2), flush=True)
        return
    runs = [p.parent for p in (STUDY/"runs").glob("*/execution.json") if p.parent.name in ARMS]
    if args.case:
        runs = [p for p in runs if p.name in args.case]
        assert {p.name for p in runs} == set(args.case), "Selected actors must be closed."
    assert runs, "No closed native runs are available."
    tasks = []
    for run in runs:
        receipt = folder/(run.name+".json")
        prior = load(receipt) if args.reuse and receipt.exists() else None
        if prior and prior.get("audit_source_sha256") == digest(__file__) and prior["clearance"]["replay_sha256"] == digest(run/"navigation.jsonl.gz") and prior["clearance"]["run_log_sha256"] == digest(run/"run.log"):
            verify_inputs(run)
            continue
        tasks.append(str(run))
    failures = []
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        mapping = {pool.submit(audit_case, task): task for task in tasks}
        for future in as_completed(mapping):
            try:
                result = future.result(); save(folder/(result["case"]+".json"), result)
                print(f"Audited {result['case']}: {result['route']['status']}, {len(result['route']['captures'])}/8", flush=True)
            except Exception as error:
                failures.append({"case": Path(mapping[future]).name, "exception": repr(error)})
    if failures:
        save(STUDY/"independent_audit_failures.json", failures)
        raise RuntimeError(f"Independent audits failed: {failures}")
    report = aggregate()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
