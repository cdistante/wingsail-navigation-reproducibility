"""Recompute the manuscript's headline comparisons from archived observations."""

import csv
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import mean, median


RESULTS = Path(__file__).resolve().parents[1] / "data" / "results"
SWEEP = RESULTS / "wind_direction_ablation_20261118"
OTRANTO = RESULTS / "otranto_layline_eta"
RIDGE = RESULTS / "eta_ridge_closed_loop"
DEPARTURE = RESULTS / "brindisi_escape_departure"
TREND = RESULTS / "brindisi_hybrid_forcing"
FORCING = RESULTS / "stage3_forcing_offline"
GRPO_6 = RESULTS / "grpo_dropout_native"
GRPO_12 = RESULTS / "grpo_group12_native"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def records(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def receipt(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_native_receipt(directory):
    run = receipt(directory / "run.json")
    checksums = {
        "inputs.json": run["inputs_sha256"],
        "episodes.csv": run["episodes_sha256"],
        **run["trace_sha256es"],
    }
    for name, expected in checksums.items():
        path = directory / name
        require(path.resolve().is_relative_to(directory.resolve()), f"Receipt path leaves its run directory: {name}")
        require(sha256(path.read_bytes()).hexdigest() == expected, f"Archived checksum differs for {path}")


def completed(row, method):
    return row[f"{method}_completed"] == "True"


def joint_deltas(rows, left, right, clock):
    return [
        float(row[f"{right}_{clock}"]) - float(row[f"{left}_{clock}"])
        for row in rows
        if completed(row, left) and completed(row, right)
    ]


def verify_sweep():
    rows = records(SWEEP / "run_metrics.csv")
    require(len({row["run_id"] for row in rows}) == len(rows) == 720, "Wind-sweep count differs")
    for method, finish_count, capture_count in (("gamma_direct", 29, 74), ("forecast", 37, 86)):
        compact = [row for row in rows if row["scope"] == "full_route" and row["method"] == method]
        completed_routes = sum(row["mission_completed"] == "True" for row in compact)
        captures = sum(int(row["waypoints_reached"]) for row in compact)
        required_marks = sum(int(row["waypoints_total"]) for row in compact)
        require(
            (len(compact), completed_routes, captures, required_marks) == (56, finish_count, capture_count, 128),
            f"Wind-sweep outcomes differ for {method}",
        )
        require(all(row["waypoint_order_verified"] == "True" for row in compact), "Sweep waypoint order differs")
    print("Wind sweep  720 runs, compact completions 29/56 to 37/56 and ordered captures 74/128 to 86/128")


def verify_otranto():
    rows = records(OTRANTO / "eta_comparison.csv")
    legs = records(OTRANTO / "paired_layline_captures.csv")
    audit = receipt(OTRANTO / "independent_final_summary.json")
    expected_savings = {355: 365, 5: 450}
    expected_order = ["002", "003", "004", "005", "004", "003", "002", "001"]
    require(len(rows) == 2 and len(legs) == 16, "Otranto run or gate count differs")
    require(audit["all_acceptance_checks_passed"] and audit["verified_captures"] == 16, "Otranto audit failed")
    require(audit["both_underway65s_peaks_experienced_in_all_runs"], "Otranto fault exposure differs")
    for row in rows:
        wind = int(row["wind_from_deg"])
        ordered = [leg for leg in legs if int(leg["wind_from_deg"]) == wind]
        require([leg["to"] for leg in ordered] == expected_order, f"Otranto capture sequence differs at {wind} degrees")
        for method in ("eta", "off"):
            require(row[f"{method}_status"] == "completed", f"Otranto {method} loop failed at {wind} degrees")
            times = [int(leg[f"{method}_capture_time_s"]) for leg in ordered]
            require(times == sorted(set(times)), "Otranto capture times are not strictly increasing")
            require(all(float(leg[f"{method}_capture_distance_m"]) <= 50 for leg in ordered), "Otranto gate miss")
            require(times[-1] == int(row[f"{method}_elapsed_to_loop_completion_s"]), "Otranto finish time differs")
            for center in ("truth", "report"):
                require(float(row[f"{method}_full_{center}_plus_3m_margin_m"]) > 0, "Otranto reserve incursion")
            require(float(row[f"{method}_maximum_full_control_s"]) < 5, "Otranto control budget exceeded")
        saved = int(row["off_elapsed_to_loop_completion_s"]) - int(row["eta_elapsed_to_loop_completion_s"])
        require(saved == expected_savings[wind], f"Otranto paired savings differ at {wind} degrees")
        print(f"Otranto {wind} degrees  8/8 ordered 50 m gates, {saved} s saved with ETA gating")


def verify_brindisi():
    ridge = records(RIDGE / "native" / "paired_episodes.csv")
    protected = records(DEPARTURE / "native" / "paired_episodes.csv")
    trend = records(TREND / "native" / "paired_episodes.csv")
    require(len(ridge) == len(protected) == len(trend) == 12, "Brindisi case count differs")
    case_ids = {row["scenario"] for row in ridge}
    require(
        len(case_ids) == 12 and all({row["scenario"] for row in arm} == case_ids for arm in (protected, trend)),
        "Brindisi case roster differs between comparisons",
    )
    improvement = joint_deltas(ridge, "physics", "ridge", "duration_s")
    require(len(improvement) == 5 and round(-mean(improvement), 2) == 36.03, "Brindisi ridge gain differs")
    require(sum(int(row["expected_captures"]) for row in protected) == 28, "Brindisi gate denominator differs")
    for arm, finishes, captures in (("ridge", 5, 12), ("departure", 8, 20)):
        require(sum(completed(row, arm) for row in protected) == finishes, f"Brindisi {arm} finishes differ")
        require(
            sum(len(row[f"{arm}_captures_in_order"].split()) for row in protected) == captures,
            f"Brindisi {arm} captures differ",
        )
    overlap = joint_deltas(trend, "departure", "hybrid", "duration_s")
    require(len(overlap) == 8 and round(median(overlap), 2) == 25.17, "Brindisi paired trend penalty differs")
    mission_deltas = joint_deltas(trend, "departure", "hybrid", "request_to_outcome_s")
    require(
        len(overlap) == len(mission_deltas)
        and all(math.isclose(sailing, mission, rel_tol=0, abs_tol=1e-8)
                for sailing, mission in zip(overlap, mission_deltas)),
        "Brindisi paired sailing and mission deltas differ",
    )
    for arm in ("departure", "hybrid"):
        moving = [row for row in trend if row[f"{arm}_truth_full_margin_m"]]
        require(len(moving) == 10, f"Brindisi {arm} applied-motion count differs")
        require(
            all(row[f"{arm}_clearance_status"] == "clear" for row in moving)
            and all(float(row[f"{arm}_{center}_full_margin_m"]) > 0 for row in moving for center in ("truth", "report")),
            f"Brindisi {arm} reserve incursion",
        )
        worst_truth = min(float(row[f"{arm}_truth_full_margin_m"]) for row in moving)
        require(round(worst_truth, 3) == 14.746, f"Brindisi {arm} minimum truth margin differs")
        require(
            sum(len(row[f"{arm}_ordered_captures"].split()) for row in trend) == 20,
            f"Brindisi {arm} trend comparison captures differ",
        )
        require(all(float(row[f"{arm}_maximum_control_seconds"]) < 5 for row in moving), "Brindisi control budget exceeded")
    print("Brindisi  ridge saves a mean 36.03 s over five joint finishes")
    print("Brindisi  protected departure completes 8/12 routes and 20/28 gates, from 5/12 and 12/28")
    print("Brindisi  trend adds 25.17 s paired median over eight joint finishes, both clear 10/10 with minimum +14.746 m")
    verify_native_receipt(RIDGE / "native" / "explicit")
    verify_native_receipt(DEPARTURE / "native" / "departure")
    verify_native_receipt(TREND / "native" / "hybrid")


def verify_forecast():
    test = [row for row in records(FORCING / "prediction_records.csv.gz") if row["split"] == "test"]
    require(len(test) == 1512, "Forecast test cohort differs")
    for method, expected in (("persistence", 0.672), ("causal_trend", 0.545)):
        squared_errors = [
            sum(
                (float(row[f"{method}_wind_{axis}_mps"]) - float(row[f"target_wind_{axis}_mps"])) ** 2
                for axis in ("e", "n")
            )
            for row in test
        ]
        rmse = math.sqrt(mean(squared_errors))
        require(round(rmse, 3) == expected, f"Forecast wind vector RMSE differs for {method}")
    print("Forecast  October next-hour wind vector RMSE 0.672 m/s for persistence and 0.545 m/s for trend")


def verify_grpo():
    diagnosis = receipt(GRPO_12 / "tack_diagnosis.json")
    for groups, directory, expected_changes, expected_gains in (
        (6, GRPO_6, (2, 0), (39.37, 3.21)),
        (12, GRPO_12, (0, 0), (8.07, 3.56)),
    ):
        rows = records(directory / "native" / "paired_episodes.csv")
        audit = receipt(directory / "native" / "independent_comparison.json")
        require(len(rows) == 12 and audit["audited_capture_order"], "GRPO case count or waypoint audit differs")
        observed_changes = []
        observed_gains = []
        for method in ("grpo_progress", "grpo_dropout"):
            verify_native_receipt(directory / "native" / method)
            finishes = [float(row[f"{method}_elapsed_s"]) for row in rows if completed(row, method)]
            require(len(finishes) == 5 and round(median(finishes), 2) == 3718.77, f"{method} finish outcomes differ")
            require(sum(len(row[f"{method}_ordered_captures"].split()) for row in rows) == 12, f"{method} capture count differs")
            movement = [row for row in rows if row[f"{method}_full_truth_margin_m"]]
            require(
                len(movement) == 10 and all(
                    float(row[f"{method}_full_{center}_margin_m"]) > 0
                    for row in movement for center in ("truth", "report")
                ),
                f"{method} reserve-inclusive clearance differs",
            )
            worst_truth = min(float(row[f"{method}_full_truth_margin_m"]) for row in movement)
            require(round(worst_truth, 3) == 16.324, f"{method} minimum truth margin differs")
            observed_changes.append(sum(int(row[f"{method}_learned_changes"]) for row in rows))
            gain = diagnosis[f"group{groups}"][method]["top_corrected_gain"]["corrected_gain_s"]
            observed_gains.append(round(gain, 2))
        require(tuple(observed_changes) == expected_changes, "GRPO selected tack counts differ")
        require(tuple(observed_gains) == expected_gains, "GRPO peak corrected ETA gains differ")
        print(f"GRPO group size {groups}  accepted overrides {observed_changes}, peak corrected ETA gains {observed_gains} s")
    print("GRPO retests  each arm completes 5/12, with median 3718.77 s and minimum full truth margin +16.324 m")


def main():
    verify_sweep()
    verify_otranto()
    verify_brindisi()
    verify_forecast()
    verify_grpo()
    print("All headline result checks passed")


if __name__ == "__main__":
    main()
