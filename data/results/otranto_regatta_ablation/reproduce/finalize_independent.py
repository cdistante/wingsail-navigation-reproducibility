"""Bind the completed paired audits to immutable prefixes, events, and fault exposure."""

from __future__ import annotations

import csv
from hashlib import file_digest
import gzip
from itertools import zip_longest
from pathlib import Path
import json

from audit_regatta import ARMS, BASE_ARMS, BASE, STUDY, digest, load, raw_frames, save, save_csv
import audit_exposure as exposure


def main():
    records = {
        case: load(STUDY / "independent_case_receipts" / (case + ".json"))
        for case in ARMS
    }
    comparison = load(STUDY / "independent_comparison.json")
    assert comparison["complete_matrix"] and comparison["closed_verified_runs"] == 16
    assert not comparison["audit_failures"]
    assert len(comparison["paired_comparisons"]) == 8
    immutable = []
    reproduction = []
    rows = []
    for case, record in records.items():
        directory = STUDY / "runs" / case
        metadata = load(directory / "execution.json")
        native, route = record["clearance"], record["route"]
        assert digest(directory / "navigation.jsonl.gz") == native["replay_sha256"] == route["sha256"]["navigation.jsonl.gz"]
        assert digest(directory / "run.log") == native["run_log_sha256"] == route["sha256"]["run.log"]
        measured_prefix = STUDY / "independent_checkpoint_inputs" / case / "navigation_900s.jsonl.gz"
        with gzip.open(directory / "navigation.jsonl.gz", "rb") as stream:
            complete_prefix = b"".join(next(stream) for _ in range(182))
        assert gzip.decompress(measured_prefix.read_bytes()) == complete_prefix
        immutable.append({
            "case": case, "full_replay_matches_frozen900s_recording": True,
            "full_replay_sha256": native["replay_sha256"], "frozen_prefix_sha256": digest(measured_prefix),
        })
    for base in BASE_ARMS:
        fresh = STUDY / "runs" / f"{base}__regatta_on" / "navigation.jsonl.gz"
        preserved = BASE / "runs" / base / "navigation.jsonl.gz"
        first_difference = None
        fresh_count = old_count = 0
        for step, (a, b) in enumerate(zip_longest(raw_frames(fresh), raw_frames(preserved))):
            fresh_count += a is not None
            old_count += b is not None
            def physics(frame):
                if frame is None:
                    return None
                nav = frame["navigation"]
                return frame["state"], frame["observation"], nav.get("rudder_command_deg"), nav.get("flap_command_deg")
            if first_difference is None and physics(a) != physics(b):
                first_difference = step
        reproduction.append({
            "base_arm": base, "fresh_frames": fresh_count, "preserved_frames": old_count,
            "all_states_observations_and_applied_commands_identical": first_difference is None,
            "first_different_step": first_difference,
            "fresh_replay_sha256": digest(fresh), "preserved_replay_sha256": digest(preserved),
        })
        on, off = (records[f"{base}__regatta_{mode}"]["route"]["captures"] for mode in ("on", "off"))
        first = {(capture["phase"], capture["waypoint"]): capture for capture in on}
        second = {(capture["phase"], capture["waypoint"]): capture for capture in off}
        keys = list(dict.fromkeys([*first, *second]))
        for phase, waypoint in keys:
            a, b = first.get((phase, waypoint)), second.get((phase, waypoint))
            metadata = load(STUDY / "runs" / f"{base}__regatta_on" / "experiment.json")
            rows.append({
                "base_arm": base, "scenario_id": metadata["scenario_id"],
                "wind_from_degrees": metadata["wind_from_degrees"], "method": metadata["method"],
                "phase": phase, "waypoint": waypoint,
                "on_captured": a is not None, "off_captured": b is not None,
                "on_elapsed_s": a["elapsed_s"] if a else None,
                "off_elapsed_s": b["elapsed_s"] if b else None,
                "on_minus_off_elapsed_s": a["elapsed_s"] - b["elapsed_s"] if a and b else None,
                "on_capture_range_m": a["distance_metres"] if a else None,
                "off_capture_range_m": b["distance_metres"] if b else None,
                "on_segment_s": a["segment_s"] if a else None,
                "off_segment_s": b["segment_s"] if b else None,
                "on_minus_off_segment_s": a["segment_s"] - b["segment_s"] if a and b else None,
                "on_segment_track_m": a["segment_distance_m"] if a else None,
                "off_segment_track_m": b["segment_distance_m"] if b else None,
            })
    save_csv(STUDY / "independent_waypoint_deltas.csv", rows)
    save(STUDY / "independent_prefix_binding.json", {
        "cases": immutable, "all_frozen_prefixes_bound_to_full_replay": True,
        "first900s_pairing_receipt_sha256": digest(STUDY / "independent_prefix_pairs.json"),
    })
    save(STUDY / "independent_on_baseline_reproduction.json", {
        "cases": reproduction,
        "all_fresh_regatta_on_physical_trajectories_reproduce_preserved_baseline": all(
            row["all_states_observations_and_applied_commands_identical"] for row in reproduction
        ),
    })
    exposure.main()
    faults = load(STUDY / "fault_exposure_check.json")
    acceptance = load(STUDY / "independent_acceptance_check.json")
    audits = [record["clearance"] for record in records.values()]
    assert faults["all_cases_final"] and len(faults["cases"]) == 16
    summary = {
        "independent_cases": len(audits), "paired_treatments": 8, "sign_convention": "Regatta ON minus OFF.",
        "applied_commands": sum(a["applied_actions"] for a in audits),
        "native_0p05s_substeps": sum(a["applied_actions"] * 100 for a in audits),
        "verified_captures": sum(len(a["goal_captures"]) for a in audits),
        "completed_loops": sum(a["mission_status"] == "completed" for a in audits),
        "truth_hull_minimum_m": min(a["truth_minimum_hull_m"] for a in audits),
        "truth_nominal_minimum_m": min(a["truth_minimum_nominal_m"] for a in audits),
        "truth_full_uncertainty_minimum_m": min(a["truth_centered_minimum_uncertainty_m"] for a in audits),
        "truth_full_plus_reserve_minimum_m": min(a["configured_truth_centered_minimum_m"] for a in audits),
        "report_full_uncertainty_minimum_m": min(a["model_minimum_uncertainty_m"] for a in audits),
        "report_full_plus_reserve_minimum_m": min(a["configured_model_minimum_m"] for a in audits),
        "maximum_pose_reconstruction_error_m": max(a["maximum_pose_reconstruction_error_m"] for a in audits),
        "maximum_heading_reconstruction_error_deg": max(a["maximum_heading_reconstruction_error_deg"] for a in audits),
        "maximum_complete_AIS_choose_s": max(a["maximum_complete_AIS_choose_s"] for a in audits),
        "maximum_complete_AIS_selection_s": max(a["maximum_complete_AIS_applied_decision_s"] for a in audits),
        "maximum_gate_decision_s": max(record["gate"]["maximum_gate_decision_s"] for record in records.values()),
        "choose_overrun_attempts": sum(a["AIS_choose_attempts_at_or_above_5s"] for a in audits),
        "selection_overrun_attempts": sum(a["AIS_applied_decision_attempts_at_or_above_5s"] for a in audits),
        "all16_both_underway65s_peaks_experienced": faults["all_underway_peak_pairs_experienced"],
        "all16_both40s_delivery_loss_effects_experienced": all(
            row["both_full40s_underway_lost_delivery_effects_experienced"] for row in faults["cases"]
        ),
        "all16_gate_and_native_acceptance_passed": acceptance["all_passed"],
        "all8_physics_and_commands_equal_through900s": all(
            row["prefix_900_identical_physics_and_commands"] for row in comparison["paired_comparisons"]
        ),
        "upstream_regatta_timing_scope": (
            "Regatta forecast-search timing is recorded separately from AIS choice through action acceptance. "
            "Repeated cached telemetry is not an end-to-end guidance timing measurement."
        ),
        "preacquisition_scope": (
            "Before its first report, the fourth contact contributes physical truth with an explicitly hypothetical "
            "65-second allowance. No fabricated AIS report or projection is used."
        ),
        "evidence_sha256": {name: digest(STUDY / name) for name in (
            "independent_comparison.csv", "independent_comparison.json", "independent_arm_outcomes.csv",
            "independent_inputs_verification.json", "independent_acceptance_check.json",
            "independent_prefix_pairs.json", "independent_prefix_binding.json",
            "independent_on_baseline_reproduction.json",
            "independent_traffic_sweep.json", "fault_exposure_check.json", "independent_waypoint_deltas.csv",
        )},
    }
    save(STUDY / "independent_final_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
