"""Compare audited Otranto priority runs with the audited layline-off controls."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import shlex
import sys


STUDY = Path(__file__).resolve().parents[1]
BASELINE = STUDY.with_name("otranto_closest_layline_ablation")
sys.path.insert(0, str(BASELINE / "reproduce"))
import summarize_layline_ablation as previous


FLAG = "--prioritize-waypoint-layline"
MODES = ("off", "priority")


def name(wind, mode):
    return previous.arm_name(wind, mode)


def audit_receipts(study):
    return (
        previous.load(study / "independent_route_verification.json")["cases"],
        {row["case"]: row for row in previous.load(study / "independent_traffic_sweep.json")},
        previous.load(study / "independent_native_control_timing.json")["cases"],
        {row["case"]: row for row in previous.load(study / "independent_departure_gate_check.json")["cases"]},
    )


def read_priority_events(replay, effects):
    """Bind retained priority references to the pre-action AIS decision clock."""
    events = {row["step"]: row for row in effects}
    guidance_by_leg = {}
    decision_observation = None
    admission_start_step = None
    with gzip.open(replay, "rt", encoding="utf-8") as stream:
        next(stream)
        for raw in stream:
            row = json.loads(raw)
            navigation = row["navigation"]
            step = navigation["step"]
            leg = f'{navigation["phase"]} {navigation["waypoint"]}'
            if step:
                guidance_by_leg.setdefault(leg, Counter())[previous.guidance_context(navigation)] += 1
            admitted = navigation.get("ais_weather_candidate_admitted") is True
            if admitted and admission_start_step is None:
                admission_start_step = step
            elif not admitted:
                admission_start_step = None
            if step in events:
                if navigation.get("waypoint_layline_policy") != "priority":
                    raise ValueError(f"Layline reference at fix {step} lacks the priority mode label.")
                if not admitted or decision_observation is None:
                    raise ValueError(f"Priority reference was retained without an admitted AIS handoff at fix {step}.")
                events[step].update(
                    decision_utc=decision_observation["time"],
                    layline_policy=navigation["waypoint_layline_policy"],
                    seconds_since_ais_admission=(step - admission_start_step) * previous.DT,
                    wind_speed_mps=decision_observation["wind"]["speed"],
                    current_speed_mps=decision_observation["current"]["speed"],
                    current_course_deg=decision_observation["current"]["direction"],
                )
            decision_observation = row["observation"]
    if any("decision_utc" not in event for event in effects):
        raise ValueError(f"Unmatched priority event in {replay}.")
    return {"guidance_context_by_active_leg": {leg: dict(counts) for leg, counts in guidance_by_leg.items()}}


def leg_rows(wind, baseline, treated):
    for row in previous.paired_capture_rows(wind, baseline, treated):
        yield {("priority_" + field[3:] if field.startswith("on_") else field): value
               for field, value in row.items()}


def combine(study, baseline):
    audited = {mode: audit_receipts(path) for mode, path in (("off", baseline), ("priority", study))}
    runs, pairs, legs, events = {}, {}, [], []
    for wind in previous.WINDS:
        case_ids = tuple(name(wind, mode) for mode in MODES)
        folders = (baseline / "runs" / case_ids[0], study / "runs" / case_ids[1])
        replay_paths = tuple(folder / "navigation.jsonl.gz" for folder in folders)
        trackers, track_comparison = previous.compare_replays(replay_paths)
        for case, folder, mode, tracker in zip(case_ids, folders, MODES, trackers, strict=True):
            flags = shlex.split((folder / "command.txt").read_text(encoding="utf-8"))
            if flags.count(FLAG) != int(mode == "priority"):
                raise ValueError(f"Unexpected mode in command for {case}.")
            route, sweep, timing, gate = (collection[case] for collection in audited[mode])
            diagnostics = read_priority_events(replay_paths[1], tracker.effects) if mode == "priority" else {}
            runs[case] = previous.audit_case(folder, wind, mode, route, sweep, timing, gate, tracker)
            runs[case]["unfinished_leg"] = route["unfinished_leg"]
            runs[case]["track_m"] = route["track_m"]
            runs[case].update(diagnostics)
            runs[case]["applied_layline_references_by_source"] = dict(Counter(
                event["source"] for event in tracker.effects
            ))
            runs[case]["applied_layline_references_by_source_before_002"] = dict(Counter(
                event["source"] for event in tracker.effects
                if (event["phase"], event["waypoint"]) == previous.SEQ[0]
            ))
            events.extend({"case": case, **event} for event in tracker.effects)
        off, priority = (runs[case] for case in case_ids)
        if (off["actual_departure_utc"], off["gate_wait_s"]) != (priority["actual_departure_utc"], priority["gate_wait_s"]):
            raise ValueError(f"Paired departure clocks differ at {wind}°.")
        legs.extend(leg_rows(wind, off, priority))
        delta_fields = ("elapsed_to_outbound_002_s", "elapsed_to_loop_completion_s",
                        "maximum_range_to_outbound_002_m", "maximum_southward_displacement_to_outbound_002_m")
        differences = {field: previous.delta(priority[field], off[field]) for field in delta_fields}
        pairs[str(wind)] = {"priority_minus_off": differences, **track_comparison}
    return runs, pairs, legs, events


def table(runs, pairs):
    for wind in previous.WINDS:
        off = runs[name(wind, "off")]
        priority = runs[name(wind, "priority")]
        delta = pairs[str(wind)]["priority_minus_off"]
        row = {"wind_from_deg": wind, "delta_to_002_s": delta["elapsed_to_outbound_002_s"],
               "delta_round_trip_s": delta["elapsed_to_loop_completion_s"]}
        for mode, run in (("off", off), ("priority", priority)):
            row.update({
                f"{mode}_status": run["status"],
                f"{mode}_elapsed_to_002_s": run["elapsed_to_outbound_002_s"],
                f"{mode}_elapsed_round_trip_s": run["elapsed_to_loop_completion_s"],
                f"{mode}_capture_count": run["capture_count"],
                f"{mode}_capture_order_and_50m_pass": run["complete_ordered_loop_at_50m"],
                f"{mode}_full_truth_plus_3m_margin_m": run["full_truth_plus_3m_margin_m"],
                f"{mode}_full_report_plus_3m_margin_m": run["full_report_plus_3m_margin_m"],
                f"{mode}_maximum_full_control_s": run["maximum_full_control_s"],
                f"{mode}_at_or_above_5s_attempts": run["full_control_attempts_over_or_at_5s"],
                f"{mode}_passes_all_5s_checks": run["passes_all_5s_checks"],
                f"{mode}_drift_hold_before_002_s": hold_before_002(run),
                f"{mode}_applied_layline_references_before_002": run["applied_layline_frames_to_outbound_002"],
                f"{mode}_applied_layline_references": run["applied_layline_frames"],
            })
        yield row


def hold_before_002(run):
    return run["guidance_context_applied_frames_to_outbound_002"].get("drift hold", 0) * previous.DT


def write_report(study, baseline, runs, pairs, legs):
    clock, margin = previous.clock, previous.margin
    launch = runs[name(355, "priority")]["actual_departure_utc"]
    wait = runs[name(355, "priority")]["gate_wait_s"]
    setup = (
        "The two priority runs are paired with the verified layline-off controls from the preceding Otranto study. "
        "Both arms use Gamma and Direct acados NMPC, the same Transformer schedule veto and archived wind speed, "
        "current and waves. Each pair shares a wind-from bearing, the crossing-fan AIS fixture, 20 s report delay, "
        f"the reported dropouts and COLREG. The runs launched at {launch} after {clock(wait)} of protected staging. "
        "Times start at release."
    )
    lines = [
        "# Otranto closest-layline priority experiment",
        "",
        setup,
        "",
        "With priority enabled, a chart-clear ground layline can supply the route reference as soon as AIS admits "
        "nominal guidance. The hold and forecast planners remain available when the layline search yields no candidate. "
        "Layline feasibility is tested with a 48-minute steady-polar lookahead into a 25 m planning circle. "
        "Waypoint capture uses the 50 m mission radius. Each action still passes the native AIS/COLREG checks. "
        "The full clearance margins include ship hulls, the 25 m nominal gap, model uncertainty and 3 m reserve.",
        "",
        "| Wind FROM | Guidance | To outbound 002 | Round trip | Ordered captures at 50 m | "
        "Truth margin | Report margin | Worst full decision | 5 s checks | Applied layline references |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|:---|---:|",
    ]
    for wind in previous.WINDS:
        for mode in MODES:
            run = runs[name(wind, mode)]
            decision = run["maximum_full_control_s"]
            valid_captures = run["ordered_capture_prefix"] and run["all_captures_inside_50m"]
            captures = f'{run["capture_count"]}/8' + ("" if valid_captures else " (check failed)")
            timing = "—" if decision is None else f"{decision:.3f} s"
            lines.append(
                f'| {wind}° | {"Laylines off" if mode == "off" else "Closest layline priority"} | '
                f'{clock(run["elapsed_to_outbound_002_s"])} | {clock(run["elapsed_to_loop_completion_s"])} | '
                f'{captures} | {margin(run["full_truth_plus_3m_margin_m"])} | {margin(run["full_report_plus_3m_margin_m"])} | '
                f'{timing} | {"Pass" if run["passes_all_5s_checks"] else "Fail"} | {run["applied_layline_frames"]} |'
            )
    lines += [
        "",
        "| Wind FROM | Change to 002 | Change in round-trip time | Drift hold before 002, off → priority | "
        "Greatest range from 002, off → priority | "
        "Greatest southward displacement to 002, off → priority |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for wind in previous.WINDS:
        off, treatment = (runs[name(wind, mode)] for mode in MODES)
        delta = pairs[str(wind)]["priority_minus_off"]
        range_field, south_field = "maximum_range_to_outbound_002_m", "maximum_southward_displacement_to_outbound_002_m"
        lines.append(
            f'| {wind}° | {clock(delta["elapsed_to_outbound_002_s"], signed=True)} | '
            f'{clock(delta["elapsed_to_loop_completion_s"], signed=True)} | '
            f'{clock(hold_before_002(off))} → {clock(hold_before_002(treatment))} | '
            f'{off[range_field]:,.0f} → {treatment[range_field]:,.0f} m | {off[south_field]:,.0f} → {treatment[south_field]:,.0f} m |'
        )
    lines.extend(["", "Time changes are priority minus off. A positive change means the priority run took longer.", ""])
    for wind in previous.WINDS:
        treated = runs[name(wind, "priority")]
        split = pairs[str(wind)]["first_state_or_command_difference"]
        event = treated["first_applied_layline"]
        if event:
            lines.append(
                f'- At {wind}°, the first priority command was a {event["source"]} toward {event["phase"]} '
                f'{event["waypoint"]} at {event["decision_utc"]} UTC. The planner commanded bow '
                f'{event["bow_reference_deg"]:.1f}° and ground course {event["heading_over_ground_reference_deg"]:.1f}°. '
                f'Wind was {event["wind_speed_mps"]:.2f} m/s and current was {event["current_speed_mps"]:.2f} m/s '
                f'toward {event["current_course_deg"]:.1f}°. Priority supplied '
                f'{treated["applied_layline_frames_to_outbound_002"]:,} steering references before outbound 002. '
                f'Nominal AIS guidance had been continuously admitted for {clock(event["seconds_since_ais_admission"])} '
                "at first use."
            )
        else:
            lines.append(f'- At {wind}°, the priority run retained zero closest-layline steering references.')
        if split is not None:
            lines[-1] += f' The first paired state or actuator difference appeared at fix {split["step"]:,}.'
        elif pairs[str(wind)]["all_states_and_commands_match"]:
            lines[-1] += " The paired tracks and actuator commands match at every fix."
        if treated["status"] != "completed":
            lines[-1] += (
                f' The observed run ended after {clock(treated["observed_underway_s"])} '
                f'with status `{treated["status"]}`. The missing time is shown as —.'
            )
            unfinished = treated["unfinished_leg"]
            if unfinished:
                missing = f'{unfinished["phase"]} {unfinished["waypoint"]}'
                context = treated["guidance_context_by_active_leg"].get(missing, {})
                held = clock(previous.DT * context.get("drift hold", 0))
                retained = context.get("direct gate approach", 0) + context.get("closest reachable tack", 0)
                lines[-1] += (
                    f' The closest recorded range to {missing} was {unfinished["minimum_range_m"]:.2f} m. '
                    f'That leg recorded {held} of drift hold and {retained:,} priority references.'
                )
        matched_legs = [row for row in legs if row["wind_from_deg"] == wind and row["delta_segment_s"] is not None]
        if matched_legs:
            gain = min(matched_legs, key=lambda row: row["delta_segment_s"])
            loss = max(matched_legs, key=lambda row: row["delta_segment_s"])
            if gain["delta_segment_s"] < 0:
                lines[-1] += (
                    f' The largest gain was {clock(abs(gain["delta_segment_s"]))} on '
                    f'{gain["phase"]} {gain["from"]} → {gain["to"]}.'
                )
            if loss["delta_segment_s"] > 0:
                lines[-1] += (
                    f' The largest loss was {clock(loss["delta_segment_s"])} on '
                    f'{loss["phase"]} {loss["from"]} → {loss["to"]}.'
                )
    lines += [
        "",
        "The prescribed capture order is outbound 002 → 003 → 004 → 005, then return 004 → 003 → 002 → 001. "
        "Full control timing includes observation, chart, guidance, Direct NMPC and action acceptance. "
        "Replay writing and integration run after the timed decision.",
        "",
        "[Run comparison](priority_comparison.csv) · [Per-leg times, distances and captures](paired_layline_captures.csv) · "
        "[Applied priority references](applied_priority_frames.csv) · [Paired replay comparison](priority_comparison.json) · "
        "[Independent audit](independent_final_summary.json) · [Matched inputs](independent_inputs_verification.json)",
        "",
        "## Replays and labeled MP4s",
        "",
        "| Wind FROM | Guidance | Replay | Labeled MP4 |",
        "|---:|:---|:---|:---|",
    ]
    for wind in previous.WINDS:
        for mode in MODES:
            root = baseline if mode == "off" else study
            directory = os.path.relpath(root / "runs" / name(wind, mode), study)
            lines.append(f'| {wind}° | {"Laylines off" if mode == "off" else "Closest layline priority"} | '
                         f'[Replay]({directory}/navigation.jsonl.gz) | [MP4]({directory}/navigation.mp4) |')
    lines.append("")
    lines.append("[Priority MP4 verification](native_video_verification.json)")
    lines.append("")
    (study / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args()
    runs, pairs, legs, events = combine(args.study, args.baseline)
    rows = list(table(runs, pairs))
    previous.save_csv(args.study / "priority_comparison.csv", rows, list(rows[0]))
    previous.save_csv(args.study / "paired_layline_captures.csv", legs, list(legs[0]))
    event_fields = ["case", "step", "decision_utc", "utc_at_fix", "phase", "waypoint", "source", "layline_policy",
                    "seconds_since_ais_admission", "wind_speed_mps", "current_speed_mps", "current_course_deg",
                    "planned_first_distance_m", "planned_eta_s", "heading_over_ground_reference_deg",
                    "bow_reference_deg", "second_bow_deg", "rudder_command_deg", "flap_command_deg"]
    previous.save_csv(args.study / "applied_priority_frames.csv", events, event_fields)
    summary = {"comparison_sign": "priority minus off", "runs": runs, "pairs": pairs}
    (args.study / "priority_comparison.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    write_report(args.study, args.baseline, runs, pairs, legs)
    for row in rows:
        wind = row["wind_from_deg"]
        print(f'{wind}° · delta to 002 {previous.clock(row["delta_to_002_s"], signed=True)} · '
              f'delta round trip {previous.clock(row["delta_round_trip_s"], signed=True)}')


if __name__ == "__main__":
    main()
