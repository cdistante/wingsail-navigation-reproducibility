"""Compare the audited Otranto ETA-gated guidance runs with laylines off."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from math import isfinite
import os
from pathlib import Path
import shlex
from statistics import median
import sys


STUDY = Path(__file__).resolve().parents[1]
BASELINE = STUDY.with_name("otranto_closest_layline_ablation")
PRIORITY = STUDY.with_name("otranto_layline_priority")
sys.path.insert(0, str(PRIORITY / "reproduce"))
from summarize_layline_priority import audit_receipts, hold_before_002, previous


FLAG = "--compare-waypoint-layline-eta"
MODES = ("off", "eta")
POLICY = "eta"
LAYLINE_ETA = "waypoint_layline_eta_s"
FORECAST_ETA = "waypoint_layline_forecast_eta_s"
GAIN = "waypoint_layline_eta_advantage_s"
EVENT_FIELDS = [
    "case", "step", "decision_utc", "utc_at_fix", "phase", "waypoint", "source", "layline_policy",
    "seconds_since_ais_admission", "wind_speed_mps", "current_speed_mps", "current_course_deg",
    "planned_first_distance_m", "planned_eta_s", "steady_layline_eta_s", "comparison_radius_m",
    "layline_comparison_eta_s", "forecast_comparison_eta_s",
    "predicted_gain_s", "heading_over_ground_reference_deg", "bow_reference_deg", "second_bow_deg",
    "rudder_command_deg", "flap_command_deg",
]


def name(wind, mode):
    return previous.arm_name(wind, mode)


def read_eta_events(replay, effects):
    """Read comparison telemetry and associate each applied command with its observation."""
    events = {row["step"]: row for row in effects}
    guidance_by_leg = {}
    decision_observation = None
    admission_start_step = None
    with gzip.open(replay, "rt", encoding="utf-8") as stream:
        next(stream)
        for raw in stream:
            row = json.loads(raw)
            nav = row["navigation"]
            step = nav["step"]
            if step:
                leg = f'{nav["phase"]} {nav["waypoint"]}'
                guidance_by_leg.setdefault(leg, Counter())[previous.guidance_context(nav)] += 1
            admitted = nav.get("ais_weather_candidate_admitted") is True
            if admitted and admission_start_step is None:
                admission_start_step = step
            elif not admitted:
                admission_start_step = None

            if step in events:
                if nav.get("waypoint_layline_policy") != POLICY or not admitted or decision_observation is None:
                    raise ValueError(f"ETA reference at fix {step} lacked policy or AIS admission evidence.")
                layline_eta, forecast_eta, gain = (nav.get(key) for key in (LAYLINE_ETA, FORECAST_ETA, GAIN))
                if (not all(isinstance(value, (int, float)) and isfinite(value)
                            for value in (layline_eta, forecast_eta, gain))
                        or not 0. <= layline_eta < forecast_eta
                        or abs(forecast_eta - layline_eta - gain) > 1.e-6
                        or nav.get("waypoint_layline_comparison_radius_m") != previous.RADIUS_M):
                    raise ValueError(f"Invalid applied ETA comparison at fix {step}.")
                events[step].update(
                    decision_utc=decision_observation["time"],
                    layline_policy=nav["waypoint_layline_policy"],
                    seconds_since_ais_admission=(step - admission_start_step) * previous.DT,
                    wind_speed_mps=decision_observation["wind"]["speed"],
                    current_speed_mps=decision_observation["current"]["speed"],
                    current_course_deg=decision_observation["current"]["direction"],
                    steady_layline_eta_s=nav.get("waypoint_layline_steady_eta_s"),
                    comparison_radius_m=nav["waypoint_layline_comparison_radius_m"],
                    layline_comparison_eta_s=layline_eta,
                    forecast_comparison_eta_s=forecast_eta,
                    predicted_gain_s=gain,
                )
            decision_observation = row["observation"]
    if any("decision_utc" not in event for event in effects):
        raise ValueError(f"Unmatched ETA event in {replay}.")
    return {leg: dict(counts) for leg, counts in guidance_by_leg.items()}


def combine(study, baseline):
    audited = {mode: audit_receipts(path) for mode, path in (("off", baseline), ("eta", study))}
    runs, pairs, legs, events = {}, {}, [], []
    for wind in previous.WINDS:
        case_ids = [name(wind, mode) for mode in MODES]
        folders = [baseline / "runs" / case_ids[0], study / "runs" / case_ids[1]]
        replays = [folder / "navigation.jsonl.gz" for folder in folders]
        trackers, alignment = previous.compare_replays(replays)
        for case, folder, mode, tracker, replay in zip(case_ids, folders, MODES, trackers, replays, strict=True):
            command = shlex.split((folder / "command.txt").read_text(encoding="utf-8"))
            if command.count(FLAG) != int(mode == "eta"):
                raise ValueError(f"Unexpected ETA mode in {folder}.")
            route, sweep, timing, gate = (collection[case] for collection in audited[mode])
            guidance = read_eta_events(replay, tracker.effects) if mode == "eta" else {}
            runs[case] = previous.audit_case(folder, wind, mode, route, sweep, timing, gate, tracker)
            runs[case].update(
                unfinished_leg=route["unfinished_leg"], track_m=route["track_m"],
                guidance_by_leg=guidance,
                applied_layline_references_by_source=dict(Counter(event["source"] for event in tracker.effects)),
            )
            if tracker.effects:
                gains = [event["predicted_gain_s"] for event in tracker.effects]
                first = tracker.effects[0]
                capture = next((row for row in route["captures"]
                                if (row["phase"], row["waypoint"]) == (first["phase"], first["waypoint"])), None)
                runs[case]["predicted_gain_s"] = {"minimum": min(gains), "median": median(gains), "maximum": max(gains)}
                runs[case]["observed_time_to_mark_from_first_eta_decision_s"] = (
                    capture["elapsed_s"] - (first["step"] - 1) * previous.DT if capture else None
                )
            events.extend({"case": case, **event} for event in tracker.effects)
        off, eta = (runs[case] for case in case_ids)
        if (off["actual_departure_utc"], off["gate_wait_s"]) != (eta["actual_departure_utc"], eta["gate_wait_s"]):
            raise ValueError(f"Paired departure clocks differ for {wind}°.")
        first = eta["first_applied_layline"]
        split = alignment["first_state_or_command_difference"]
        if first and split and first["step"] == split["step"]:
            mark = first["phase"], first["waypoint"]
            if mark == (split["off"]["phase"], split["off"]["waypoint"]):
                capture = next((row for row in off["captures"] if (row["phase"], row["waypoint"]) == mark), None)
                if capture:
                    eta["off_time_to_mark_from_first_eta_decision_s"] = (
                        capture["elapsed_s"] - (first["step"] - 1) * previous.DT
                    )
        legs.extend(
            {("eta_" + key[3:] if key.startswith("on_") else key): value for key, value in row.items()}
            for row in previous.paired_capture_rows(wind, off, eta)
        )
        pairs[str(wind)] = {
            "eta_minus_off": {key: previous.delta(eta[key], off[key]) for key in (
                "elapsed_to_outbound_002_s", "elapsed_to_loop_completion_s",
                "maximum_range_to_outbound_002_m", "maximum_southward_displacement_to_outbound_002_m",
            )},
            **alignment,
        }
    return runs, pairs, legs, events


def comparison_rows(runs, pairs):
    fields = [
        "status", "elapsed_to_outbound_002_s", "elapsed_to_loop_completion_s", "observed_underway_s",
        "capture_count", "ordered_capture_prefix", "all_captures_inside_50m",
        "full_truth_plus_3m_margin_m", "full_report_plus_3m_margin_m", "maximum_full_control_s",
        "passes_all_5s_checks", "full_control_attempts_over_or_at_5s",
        "applied_layline_frames", "applied_layline_frames_to_outbound_002",
        "maximum_range_to_outbound_002_m", "maximum_southward_displacement_to_outbound_002_m",
    ]
    for wind in previous.WINDS:
        row = {"wind_from_deg": wind}
        for mode in MODES:
            run = runs[name(wind, mode)]
            row.update({f"{mode}_{field}": run[field] for field in fields})
            row[f"{mode}_drift_hold_before_002_s"] = hold_before_002(run)
        row.update({f"delta_{field}": value for field, value in pairs[str(wind)]["eta_minus_off"].items()})
        yield row


def describe_run(wind, eta, alignment, legs):
    clock = previous.clock
    opening = eta["applied_layline_frames_to_outbound_002"]
    event = eta["first_applied_layline"]
    split = alignment["first_state_or_command_difference"]
    details = []
    if event:
        details.append(
            f'The first ETA-gated command was a {event["source"]} toward {event["phase"]} {event["waypoint"]} '
            f'at {event["decision_utc"][:19]} UTC, {clock(event["seconds_since_ais_admission"])} after continuous AIS '
            f'admission began. Predicted entry was {event["layline_comparison_eta_s"]:.1f} s for the layline and '
            f'{event["forecast_comparison_eta_s"]:.1f} s for forecast tacking, a gain of {event["predicted_gain_s"]:.1f} s.'
        )
        details.append(f'The run retained {eta["applied_layline_frames"]:,} ETA-gated references, '
                       f'including {opening:,} before outbound 002. The median predicted gain was '
                       f'{eta["predicted_gain_s"]["median"]:.1f} s.')
        elapsed = eta["observed_time_to_mark_from_first_eta_decision_s"]
        if elapsed is not None:
            baseline_elapsed = eta.get("off_time_to_mark_from_first_eta_decision_s")
            details.append(f'The first affected mark was captured {clock(elapsed)} after that decision.'
                           + (f' Layline-off took {clock(baseline_elapsed)} from the same decision time.'
                              if baseline_elapsed is not None else ""))
    else:
        details.append("The ETA gate supplied zero applied layline references.")
    if split:
        details.append(f'The first paired state or actuator change occurred at fix {split["step"]:,}.')
    elif alignment["all_states_and_commands_match"]:
        details.append("The paired states and actuator commands matched at every fix.")
    if eta["unfinished_leg"]:
        leg = eta["unfinished_leg"]
        details.append(f'The run ended after {clock(eta["observed_underway_s"])} on {leg["phase"]} {leg["waypoint"]}. '
                       f'The closest recorded approach on that leg was {leg["minimum_range_m"]:.2f} m.')
    matched = [row for row in legs if row["wind_from_deg"] == wind and row["delta_segment_s"] is not None]
    if matched:
        best = min(matched, key=lambda row: row["delta_segment_s"])
        worst = max(matched, key=lambda row: row["delta_segment_s"])
        if best["delta_segment_s"] < 0:
            details.append(f'The largest leg gain was {clock(-best["delta_segment_s"])} on '
                           f'{best["phase"]} {best["from"]} → {best["to"]}.')
        if worst["delta_segment_s"] > 0:
            details.append(f'The largest leg loss was {clock(worst["delta_segment_s"])} on '
                           f'{worst["phase"]} {worst["from"]} → {worst["to"]}.')
    return f"- **{wind}°**. " + " ".join(details)


def report(study, baseline, priority, runs, pairs, legs):
    clock, margin = previous.clock, previous.margin
    reference = runs[name(355, "eta")]
    lines = [
        "# Otranto ETA-gated closest-layline experiment", "",
        "The two ETA-gated runs are compared with the verified layline-off controls. Each pair uses the same "
        "archived wind, current and waves, Gamma guidance, Direct acados NMPC, Transformer schedule veto, "
        "AIS traffic and dropouts, and COLREG settings. "
        f'Times start at release at {reference["actual_departure_utc"]} after '
        f'{clock(reference["gate_wait_s"])} of protected staging.', "",
        "After AIS admits route guidance, the closest feasible layline can replace the forecast tack if its "
        "predicted first entry to the next waypoint is earlier. The layline planner aims at the inner 25 m circle. "
        "Both candidate paths are reprojected from the decision state using the same archived forecast speed, polar "
        "and turn assumptions. The comparison measures first entry to the original 50 m gate and requires a finite "
        "capture prediction from both candidates. Holds and forecast turns already underway keep priority. "
        "Each applied command still passes chart and AIS/COLREG checks. The full clearance margin includes "
        "ship hulls, the 25 m gap, uncertainty and a 3 m reserve.", "",
        "| Wind FROM | Guidance | To outbound 002 | Loop or stop | Ordered 50 m captures | "
        "Truth margin | Report margin | Slowest decision | 5 s checks | ETA references |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|:---|---:|",
    ]
    labels = {"off": "Laylines off", "eta": "ETA-gated closest layline"}
    for wind in previous.WINDS:
        for mode in MODES:
            run = runs[name(wind, mode)]
            duration = (clock(run["elapsed_to_loop_completion_s"]) if run["complete_ordered_loop_at_50m"]
                        else f'{clock(run["observed_underway_s"])} ({run["status"]})')
            valid_captures = run["ordered_capture_prefix"] and run["all_captures_inside_50m"]
            captures = f'{run["capture_count"]}/8' + ("" if valid_captures else " (check failed)")
            lines.append(
                f'| {wind}° | {labels[mode]} | {clock(run["elapsed_to_outbound_002_s"])} | {duration} | '
                f'{captures} | {margin(run["full_truth_plus_3m_margin_m"])} | {margin(run["full_report_plus_3m_margin_m"])} | '
                f'{run["maximum_full_control_s"]:.3f} s | {"Pass" if run["passes_all_5s_checks"] else "Fail"} | '
                f'{run["applied_layline_frames"]:,} |'
            )
    lines.extend([
        "",
        "| Wind FROM | Change to 002 | Change in loop time | Drift hold before 002, off → ETA | "
        "Furthest from 002, off → ETA | Furthest south before 002, off → ETA |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for wind in previous.WINDS:
        off, eta = (runs[name(wind, mode)] for mode in MODES)
        delta = pairs[str(wind)]["eta_minus_off"]
        key = "maximum_range_to_outbound_002_m"
        south = "maximum_southward_displacement_to_outbound_002_m"
        lines.append(
            f'| {wind}° | {clock(delta["elapsed_to_outbound_002_s"], signed=True)} | '
            f'{clock(delta["elapsed_to_loop_completion_s"], signed=True)} | '
            f'{clock(hold_before_002(off))} → {clock(hold_before_002(eta))} | {off[key]:,.0f} → {eta[key]:,.0f} m | '
            f'{off[south]:,.0f} → {eta[south]:,.0f} m |'
        )
    lines.extend(["", "Time changes are ETA-gated minus layline-off.", ""])
    lines.extend(describe_run(wind, runs[name(wind, "eta")], pairs[str(wind)], legs) for wind in previous.WINDS)
    earlier = priority / "priority_comparison.json"
    if earlier.is_file():
        priority_cases = previous.load(earlier)["runs"]
        lines.extend(["", "In the earlier priority-only trial, the closest feasible layline could take over without an ETA comparison.", ""])
        for wind in previous.WINDS:
            old = priority_cases[name(wind, "priority")]
            eta = runs[name(wind, "eta")]
            old_time, eta_time = (case["elapsed_to_loop_completion_s"] for case in (old, eta))
            old_loop = clock(old_time) if old_time is not None else f'{old["capture_count"]}/8 captures at {clock(old["observed_underway_s"])}'
            eta_loop = clock(eta_time) if eta_time is not None else f'{eta["capture_count"]}/8 captures at {clock(eta["observed_underway_s"])}'
            lines.append(f'- At {wind}°, the priority-only outcome was {old_loop}. The ETA-gated outcome was {eta_loop}.')
        lines[-1] += f' [Priority-only experiment]({os.path.relpath(priority / "README.md", study)})'
    lines.extend([
        "", "Capture order is outbound 002 → 003 → 004 → 005, then return 004 → 003 → 002 → 001. "
        "Decision timing includes observation, chart, guidance, Direct NMPC and action acceptance. "
        "Replay writing and integration follow the timed decision.", "",
        "[Run comparison](eta_comparison.csv) · [Per-leg times and distances](paired_layline_captures.csv) · "
        "[Applied ETA comparisons](applied_eta_frames.csv) · [Paired replay data](eta_comparison.json) · "
        "[Independent audit](independent_final_summary.json) · [Matched inputs](independent_inputs_verification.json)",
        "", "## Replays and labeled MP4s", "",
        "| Wind FROM | Guidance | Replay | MP4 |", "|---:|:---|:---|:---|",
    ])
    for wind in previous.WINDS:
        for mode in MODES:
            folder = (baseline if mode == "off" else study) / "runs" / name(wind, mode)
            path = os.path.relpath(folder, study)
            lines.append(f"| {wind}° | {labels[mode]} | [Replay]({path}/navigation.jsonl.gz) | [Labeled MP4]({path}/navigation.mp4) |")
    lines.extend(["", "[ETA-gated MP4 verification](native_video_verification.json)", ""])
    (study / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--priority", type=Path, default=PRIORITY)
    args = parser.parse_args()
    runs, pairs, legs, events = combine(args.study, args.baseline)
    rows = list(comparison_rows(runs, pairs))
    previous.save_csv(args.study / "eta_comparison.csv", rows, list(rows[0]))
    previous.save_csv(args.study / "paired_layline_captures.csv", legs, list(legs[0]))
    previous.save_csv(args.study / "applied_eta_frames.csv", events, EVENT_FIELDS)
    (args.study / "eta_comparison.json").write_text(
        json.dumps({"comparison_sign": "ETA-gated minus off", "runs": runs, "pairs": pairs}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report(args.study, args.baseline, args.priority, runs, pairs, legs)
    for wind, pair in pairs.items():
        delta = pair["eta_minus_off"]
        print(f'{wind}° · delta to 002 {previous.clock(delta["elapsed_to_outbound_002_s"], signed=True)} · '
              f'delta round trip {previous.clock(delta["elapsed_to_loop_completion_s"], signed=True)}')


if __name__ == "__main__":
    main()
