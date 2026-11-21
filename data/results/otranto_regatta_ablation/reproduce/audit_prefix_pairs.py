"""Check matched traffic prefixes using the preserved 900 s independent recordings."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from audit_regatta import (
    ARMS, BASE_ARMS, STUDY, digest, gate_audit, load, pair_trajectory, raw_frames, reference, save,
)


def main():
    receipts = load(STUDY / "independent_traffic_prefix_sweep.json")
    assert {row["case"] for row in receipts} == set(ARMS)
    paths = {
        case: STUDY / "independent_checkpoint_inputs" / case / "navigation_900s.jsonl.gz"
        for case in ARMS
    }
    pairs = []
    for base in BASE_ARMS:
        on, off = (f"{base}__regatta_{mode}" for mode in ("on", "off"))
        pairs.append({
            "base_arm": base, **pair_trajectory(paths[on], paths[off]),
            "on_prefix_sha256": digest(paths[on]), "off_prefix_sha256": digest(paths[off]),
        })
    gates = [gate_audit.audit_case(STUDY / "runs" / case) for case in ARMS]
    save(STUDY / "independent_departure_gate_check.json", {
        "all_available_gates_verified": True, "all_gate_attempts_below_5s": all(g["all_gate_attempts_below_5s"] for g in gates),
        "cases": gates,
    })
    providers = {layer: cls(**reference.OTRANTO_BOUNDS).load() for layer, cls in reference.ENVIRONMENT.items()}
    forcing = []
    try:
        for case in ARMS:
            meta = load(STUDY / "runs" / case / "experiment.json")
            records = list(raw_frames(paths[case]))
            times = [row["observation"]["time"] for row in records]
            coords = np.array([[row["state"][key] for key in ("latitude", "longitude")] for row in records])
            max_error = 0.
            scalar_error = 0.
            for layer, provider in providers.items():
                values, missing = reference.archive_values(provider, layer, coords, times, meta["wind_from_degrees"])
                for key, expected in values.items():
                    error = np.max(np.abs(expected - np.array([row["observation"][layer][key] for row in records])))
                    max_error = max(max_error, float(error))
                    assert error <= 1e-8, (case, layer, key, error)
                for index in (0, 60, 120, 180):
                    scalar = asdict(provider.sample(*coords[index], times[index]))
                    if layer == "wind":
                        angle = np.deg2rad(meta["wind_from_degrees"])
                        scalar.update(u=-scalar["speed"] * np.sin(angle), v=-scalar["speed"] * np.cos(angle),
                                      direction=float(meta["wind_from_degrees"]))
                    for key, value in scalar.items():
                        error = abs(value - values[key][index])
                        scalar_error = max(scalar_error, float(error))
                        assert error <= 1e-8
            forcing.append({
                "case": case, "frames": len(records), "archive_queries_per_layer": len(records),
                "maximum_archive_value_difference": max_error,
                "maximum_scalar_archive_check_difference": scalar_error,
                "prefix_sha256": digest(paths[case]),
            })
    finally:
        for provider in providers.values():
            provider.close()
    result = {
        "cases": 16, "prefix_seconds": 900, "pairs": pairs,
        "all16_acquisition_at420s": all(r["underway_first_acquisition_encounter_elapsed_s"] == 420 for r in receipts),
        "all16_both65s_peaks_verified": all(r["underway_both65s_age_peaks_verified"] for r in receipts),
        "truth_full_plus_reserve_minimum_m": min(r["configured_truth_centered_minimum_m"] for r in receipts),
        "report_full_plus_reserve_minimum_m": min(r["configured_model_minimum_m"] for r in receipts),
        "all_native_endpoint_errors_zero": all(r["maximum_pose_reconstruction_error_m"] == 0 for r in receipts),
        "all_native_negative_clearance_counts_zero": all(all(v == 0 for v in r["negative_action_counts"].values()) for r in receipts),
        "whole_selection_max_s": max(r["maximum_complete_AIS_applied_decision_s"] for r in receipts),
        "all_complete_selection_attempts_below_5s": all(r["all_AIS_applied_decision_attempts_below_5s"] for r in receipts),
        "forcing": forcing,
    }
    save(STUDY / "independent_prefix_pairs.json", result)
    print(json.dumps({key: val for key, val in result.items() if key not in ("pairs", "forcing")}, indent=2))


if __name__ == "__main__":
    main()
