"""Independently audit an Otranto rerun and compare it with its archived arm."""

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch
from xml.etree import ElementTree


RESULTS = Path("data/results")
ETA = RESULTS / "otranto_layline_eta"
OFF = RESULTS / "otranto_closest_layline_ablation"
NATIVE = RESULTS / "otranto_native_schedule_veto" / "reproduce"
LENGTH_TOLERANCE_METRES = 1e-5
TIME_TOLERANCE_SECONDS = 1e-9
BUDGET_SECONDS = 5.0
FIXTURE_KEYS = {
    "traffic_fixture": "traffic_fixture_sha256",
    "truth_trace": "traffic_truth_trace_sha256",
    "underway_traffic_fixture": "underway_traffic_fixture_sha256",
    "underway_truth_trace_csv": "underway_truth_trace_sha256",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def difference(value, archived, tolerance):
    if value is None:
        return {"fresh": None, "archived": archived, "delta": None, "within_tolerance": False}
    return {
        "fresh": value, "archived": archived, "delta": value - archived,
        "within_tolerance": math.isclose(value, archived, rel_tol=0., abs_tol=tolerance),
    }


def reference(eta, wind, arm):
    comparison = next(row for row in read_rows(eta / "independent_comparison.csv")
                      if int(row["wind_from_degrees"]) == wind)
    captures = []
    for row in read_rows(eta / "paired_layline_captures.csv"):
        if int(row["wind_from_deg"]) == wind:
            captures.append({
                "phase": row["phase"], "waypoint": row["to"],
                "elapsed_s": float(row[f"{arm}_capture_time_s"]),
                "distance_metres": float(row[f"{arm}_capture_distance_m"]),
            })
    return comparison, captures


def prepare_run(root, study, args, report):
    folder = root / (ETA if args.arm == "eta" else OFF) / "runs" / report["case"]
    metadata = read_json(folder / "experiment.json")
    if metadata["wind_from_degrees"] != args.wind or metadata["layline_mode"] != args.arm:
        raise ValueError("Archive receipt does not match the requested arm.")
    for field, expected in FIXTURE_KEYS.items():
        path = (root / metadata[field]).resolve()
        if not path.is_relative_to(root) or checksum(path) != metadata[expected]:
            raise ValueError(f"Released traffic fixture does not match the archived digest: {field}")
        metadata[field] = str(path)
    # Bring only inputs into the compatibility shim. Every outcome is read from
    # the supplied run files or recomputed by the independent audits.
    fields = (*FIXTURE_KEYS, "encounter_id", "wind_from_degrees", "departure_utc")
    run = study / "runs" / report["case"]
    run.mkdir(parents=True)
    log = args.run / "simulator.log"
    if not log.is_file():
        log = args.run / "run.log"
    inputs = {name: args.run / name for name in (
        "navigation.jsonl.gz", "departure_gate.json", "departure_gate.jsonl", "track.gpx",
    )}
    inputs["run.log"] = log
    for filename, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"Incomplete run. Missing {path}")
        (run / filename).symlink_to(path.resolve())
    save_json(run / "experiment.json", {field: metadata[field] for field in fields})
    report["run_log"] = str(log.resolve())
    return run


def verify_recorded_protocol(root, run, lines, arm):
    prefix = "Runner arguments: "
    arguments = [json.loads(line.removeprefix(prefix)) for line in lines if line.startswith(prefix)]
    if len(arguments) != 1:
        raise ValueError("Expected one recorded simulator invocation.")
    command, = arguments
    required = ("--return-to-start", "--forecast-tacks", "--time-aware-ais", "--colreg",
                "--native-control-timing", "--ais-candidate-handoff")
    values = {"--planner": "gamma", "--controller": "acados-direct", "--dynamics": "fossen",
              "--environment": "marine", "--arrival-radius": "50", "--dt": "5",
              "--ais-clearance-reserve": "3", "--ais-report-delay": "20"}
    if (not all(flag in command for flag in required)
            or not all(flag in command and command[command.index(flag) + 1] == value
                       for flag, value in values.items())
            or ("--compare-waypoint-layline-eta" in command) != (arm == "eta")):
        raise ValueError("Recorded controller and AIS settings differ from the archived protocol.")
    route = root / "current_route" / "otranto_short_test_loop.gpx"
    if (root / command[command.index("--route-gpx") + 1]).resolve() != route.resolve():
        raise ValueError("Recorded route differs from the published Otranto GPX.")
    expected = [(p.findtext("{*}name"), float(p.attrib["lat"]), float(p.attrib["lon"]))
                for p in ElementTree.parse(route).findall(".//{*}rtept")]
    with gzip.open(run / "navigation.jsonl.gz", "rt", encoding="utf-8") as stream:
        header = json.loads(stream.readline())
    actual = [(point["name"], point["latitude"], point["longitude"]) for point in header["route"]]
    if actual != expected:
        raise ValueError("Replay waypoint names and coordinates differ from the published GPX.")
    for prefix in ("Saved navigation replay: ", "Saved GPX track: "):
        if not any(line.startswith(prefix) for line in lines):
            raise ValueError("The simulator did not finish logging its output files.")


def compare(args, report, archive, archived_captures, clearance, gate, timing, dt):
    captures = clearance["goal_captures"]
    complete = clearance["mission_status"] == "completed"
    report["mission_status"] = clearance["mission_status"]
    report["captures"] = [
        {**item, "elapsed_s": item["step"] * dt} for item in captures
    ]
    report["departure_utc"] = gate["actual_departure_utc"]
    report["staging_wait_s"] = gate["gate_wait_seconds"]
    report["expected_capture_order"] = [
        f'{item["phase"]} {item["waypoint"]}' for item in archived_captures
    ]
    elapsed = clearance["applied_actions"] * dt
    report["elapsed_from_launch_s"] = elapsed
    report["elapsed_from_monitoring_start_s"] = gate["gate_wait_seconds"] + elapsed
    truth = clearance["configured_truth_centered_minimum_m"]
    projection = clearance["configured_model_minimum_m"]
    report["clearance"] = {
        "truth_full_plus_reserve_m": truth, "report_full_plus_reserve_m": projection,
        "native_substep_s": clearance["integration_substep_seconds"],
        "negative_intervals": clearance["negative_action_counts"],
        "first_underway_acquisition_encounter_s": clearance["underway_first_acquisition_encounter_elapsed_s"],
        "both_underway_65s_peaks_verified": clearance["underway_both65s_age_peaks_verified"],
    }
    report["budget"] = {
        "limit_s": BUDGET_SECONDS,
        "departure_gate_max_s": gate["maximum_gate_decision_s"],
        "ais_choose_max_s": clearance["maximum_complete_AIS_choose_s"],
        "ais_accept_or_stop_max_s": clearance["maximum_complete_AIS_applied_decision_s"],
        "full_control_max_s": timing["maximum_complete_native_control_decision_s"],
        "full_control_attempts_at_or_above_limit": timing["attempts_at_or_above_5s"],
    }
    report["archived_comparison"] = comparison = {}
    for name, value, column, tolerance in (
        ("capture_002_time_s", captures[0]["step"] * dt if captures else None,
         "wp002_s", TIME_TOLERANCE_SECONDS),
        ("completion_time_s", elapsed if complete else None, "navigation_s", TIME_TOLERANCE_SECONDS),
        ("elapsed_since_monitoring_s", report["elapsed_from_monitoring_start_s"],
         "total_from_monitoring_s", TIME_TOLERANCE_SECONDS),
        ("truth_full_plus_reserve_m", truth, "full_truth_margin_m", LENGTH_TOLERANCE_METRES),
        ("report_full_plus_reserve_m", projection, "full_report_margin_m", LENGTH_TOLERANCE_METRES),
    ):
        comparison[name] = difference(value, float(archive[f"{args.arm}_{column}"]), tolerance)
    comparison["captures"] = []
    for fresh, original in zip(report["captures"], archived_captures):
        comparison["captures"].append({
            "phase": fresh["phase"], "waypoint": fresh["waypoint"],
            "elapsed_s": difference(fresh["elapsed_s"], original["elapsed_s"], TIME_TOLERANCE_SECONDS),
            "distance_metres": difference(fresh["distance_metres"], original["distance_metres"],
                                         LENGTH_TOLERANCE_METRES),
        })
    comparison["full_control_max_s"] = {
        "fresh": timing["maximum_complete_native_control_decision_s"],
        "archived": float(archive[f"{args.arm}_maximum_complete_control_s"]),
        "scope": "The original and fresh measurements each use the 5 s budget.",
    }
    report["tolerances"] = {
        "time_s": TIME_TOLERANCE_SECONDS, "distance_and_margin_m": LENGTH_TOLERANCE_METRES,
    }
    report["checks"] = checks = {
        "completed_ordered_50m_waypoints": complete and [
            (item["phase"], item["waypoint"]) for item in captures
        ] == [(item["phase"], item["waypoint"]) for item in archived_captures],
        "truth_and_report_full_plus_reserve_positive": (
            truth is not None and truth > 0. and projection is not None and projection > 0.
        ),
        "no_negative_clearance_intervals": not any(clearance["negative_action_counts"].values()),
        "underway_acquisition_at_420s": clearance["underway_first_acquisition_encounter_elapsed_s"] == 420.,
        "both_underway_65s_held_age_peaks": clearance["underway_both65s_age_peaks_verified"],
        "departure_gate_below_5s": gate["all_gate_attempts_below_5s"],
        "ais_choose_below_5s": clearance["all_AIS_choose_attempts_below_5s"],
        "ais_accept_or_stop_below_5s": clearance["all_AIS_applied_decision_attempts_below_5s"],
        "full_native_control_below_5s": timing["all_complete_native_control_attempts_below_5s"],
    }
    if args.process_exit_code is not None:
        checks["process_exit_success"] = args.process_exit_code == 0
    report["benchmark_checks_pass"] = all(checks.values())
    report["matches_archive_within_tolerance"] = (
        checks["completed_ordered_50m_waypoints"]
        and all(metric["within_tolerance"] for metric in comparison.values()
                if "within_tolerance" in metric)
        and all(item[field]["within_tolerance"] for item in comparison["captures"]
                for field in ("elapsed_s", "distance_metres"))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="The fresh run directory.")
    parser.add_argument("--repository", type=Path, default=Path.cwd(),
                        help="Reproducibility checkout root. Defaults to the current directory.")
    parser.add_argument("--wind", choices=(5, 355), type=int, required=True)
    parser.add_argument("--arm", choices=("eta", "off"), required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--process-exit-code", type=int, help="Pass the rerun's actual process exit code.")
    parser.add_argument("--require-archived-match", action="store_true",
                        help="Also require the archived capture and clearance numbers to match.")
    args = parser.parse_args()
    args.run = args.run.resolve()
    root = args.repository.resolve()
    study = args.audit_dir.resolve()
    study.mkdir(parents=True, exist_ok=False)
    receipt = study / "audit.json"
    case = f"crossing_fan__otranto_{args.wind:03d}__layline_{args.arm}"
    report = {"case": case, "input_run": str(args.run), "process_exit_code": args.process_exit_code}
    stage = "adapt the run files and verify fixture hashes"
    try:
        run = prepare_run(root, study, args, report)
        sys.path[:0] = [str(root / NATIVE), str(root)]
        stage = "load the independent audit modules"
        import audit_departure_gate
        import audit_independent
        import audit_native_hybrid

        stage = "read the mission log"
        lines = (run / "run.log").read_text(encoding="utf-8").splitlines()
        prefix = "Mission outcome: "
        outcomes = [json.loads(line.removeprefix(prefix)) for line in lines if line.startswith(prefix)]
        if len(outcomes) != 1:
            raise ValueError("Expected one complete mission outcome in the simulator log.")
        report["mission_status"] = outcomes[0]["status"]
        stage = "validate the logged controller protocol and replay route"
        verify_recorded_protocol(root, run, lines, args.arm)
        stage = "independently check protected departure"
        gate = audit_departure_gate.audit_case(run)
        stage = "replay applied dynamics and sweep the full AIS envelope"
        metadata = read_json(run / "experiment.json")
        history = study / "independent_traffic_history"
        history.mkdir()
        clearance = audit_independent.audit(
            (str(run), metadata["traffic_fixture"], metadata["truth_trace"], str(history))
        )
        stage = "cross-check all complete control attempts against replay telemetry"
        with patch.object(audit_native_hybrid, "STUDY", study):
            timing = audit_native_hybrid.native_control_timing(run)
        save_json(study / "gate.json", gate)
        save_json(study / "clearance.json", clearance)
        save_json(study / "native_control_timing.json", timing)
        stage = "compare numerical outputs with the frozen benchmark"
        archive, captures = reference(root / ETA, args.wind, args.arm)
        compare(args, report, archive, captures, clearance, gate, timing, audit_independent.DT)
    except (AssertionError, OSError, ValueError, IndexError, KeyError, ImportError,
            EOFError, ElementTree.ParseError) as error:
        report.update(benchmark_checks_pass=False, matches_archive_within_tolerance=False,
                      failure_stage=stage, detail=str(error) or type(error).__name__)
    save_json(receipt, report)
    print(receipt)
    verified = report["benchmark_checks_pass"]
    if args.require_archived_match:
        verified = verified and report["matches_archive_within_tolerance"]
    return 0 if verified else 1


if __name__ == "__main__":
    sys.exit(main())
