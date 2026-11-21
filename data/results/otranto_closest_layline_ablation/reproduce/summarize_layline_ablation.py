"""Summarize the four audited closest-layline runs and compare matched fixes."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass, field
import gzip
from hashlib import file_digest
from itertools import zip_longest
import json
from math import asin, cos, isfinite, radians, sin, sqrt
from pathlib import Path
import shlex


WINDS = (355, 5)
MODES = ("off", "on")
SEQ = (("outbound", "002"), ("outbound", "003"), ("outbound", "004"),
       ("outbound", "005"), ("return", "004"), ("return", "003"),
       ("return", "002"), ("return", "001"))
FLAG = "--closest-waypoint-layline"
DT = 5
EARTH_RADIUS_M = 6_371_008.8
RADIUS_M = 50.


def arm_name(wind, mode):
    return f"crossing_fan__otranto_{wind:03d}__layline_{mode}"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def save_csv(path, rows, columns):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def command(frame):
    nav = frame["navigation"]
    return nav.get("rudder_command_deg"), nav.get("flap_command_deg")


def distance_m(a, b):
    lat_a, lat_b = radians(a["latitude"]), radians(b["latitude"])
    dlon = radians(b["longitude"] - a["longitude"])
    h = sin((lat_b - lat_a) / 2.)**2 + cos(lat_a) * cos(lat_b) * sin(dlon / 2.)**2
    return 2. * EARTH_RADIUS_M * asin(sqrt(min(1., max(0., h))))


def guidance_context(nav):
    if nav.get("waypoint_layline_source"):
        return nav["waypoint_layline_source"]
    if nav.get("nominal_guidance_suspended"):
        return "nominal guidance suspended"
    if nav.get("drift_mode") == "holding":
        return "drift hold"
    forecast = nav.get("forecast_closest_m")
    if forecast is not None and forecast <= RADIUS_M:
        return "forecast predicts capture"
    if nav.get("forecast_tack_in_min") is not None or nav.get("forecast_maneuver_commit_active"):
        return "forecast tack scheduled or committed"
    return "other route guidance"


@dataclass
class ArmTrack:
    effects: list = field(default_factory=list)
    context: Counter = field(default_factory=Counter)
    opening_context: Counter = field(default_factory=Counter)
    opening_max_range_m: float = 0.
    opening_max_southward_m: float = 0.
    launch_latitude: float | None = None
    fixes: int = 0

    def observe(self, frame):
        nav = frame["navigation"]
        step = nav["step"]
        if step != self.fixes:
            raise ValueError(f"Expected replay fix {self.fixes}, got {step}.")
        self.fixes += 1
        lat = frame["state"]["latitude"]
        if step == 0:
            self.launch_latitude = lat
        opening = (nav["phase"], nav["waypoint"]) == SEQ[0]
        category = guidance_context(nav)
        if opening:
            self.opening_max_southward_m = max(self.opening_max_southward_m,
                                              EARTH_RADIUS_M * radians(self.launch_latitude - lat))
            distance = nav["waypoint_distance_m"]
            self.opening_max_range_m = max(self.opening_max_range_m, distance)
        if step == 0:
            return
        self.context[category] += 1
        if opening:
            self.opening_context[category] += 1
        if nav.get("waypoint_layline_source"):
            rudder, flap = command(frame)
            if (rudder is None or flap is None or nav.get("nominal_guidance_suspended")
                    or nav.get("ais_mode") != "route"):
                raise ValueError(f"Layline reference persisted through an overridden action at fix {step}.")
            self.effects.append({
                "step": step, "utc_at_fix": frame["observation"]["time"], "phase": nav["phase"],
                "waypoint": nav["waypoint"], "source": nav["waypoint_layline_source"],
                "planned_first_distance_m": nav["waypoint_layline_distance_m"],
                "planned_eta_s": nav["waypoint_layline_eta_s"],
                "heading_over_ground_reference_deg": nav["heading_reference_deg"],
                "bow_reference_deg": nav["bow_reference_deg"],
                "second_bow_deg": nav.get("waypoint_layline_second_bow_deg"),
                "rudder_command_deg": rudder, "flap_command_deg": flap,
            })

    def summary(self):
        return {
            "frames": self.fixes,
            "applied_layline_frames": len(self.effects),
            "applied_layline_frames_to_outbound_002": sum(
                (row["phase"], row["waypoint"]) == SEQ[0] for row in self.effects
            ),
            "first_applied_layline": self.effects[0] if self.effects else None,
            "guidance_context_applied_frames": dict(self.context),
            "guidance_context_applied_frames_to_outbound_002": dict(self.opening_context),
            "maximum_range_to_outbound_002_m": self.opening_max_range_m,
            "maximum_southward_displacement_to_outbound_002_m": self.opening_max_southward_m,
        }


def first_divergence(frame_off, frame_on, preceding_utc, gap):
    def reference(frame):
        nav = frame["navigation"]
        return {"phase": nav["phase"], "waypoint": nav["waypoint"], "rudder_flap_deg": command(frame),
                "bow_reference_deg": nav.get("bow_reference_deg"),
                "ground_course_reference_deg": nav.get("heading_reference_deg"),
                "layline_source": nav.get("waypoint_layline_source"), "context": guidance_context(nav)}
    return {"step": frame_on["navigation"]["step"], "decision_utc": preceding_utc,
            "resulting_fix_utc": frame_on["observation"]["time"], "resulting_track_separation_m": gap,
            "off": reference(frame_off), "on": reference(frame_on)}


def compare_replays(paths):
    tracks = (ArmTrack(), ArmTrack())
    compared, changed_commands, changed_states, changed_weather = 0, 0, 0, 0
    max_gap, end_gap, previous_utc, divergence = 0., None, None, None
    with (gzip.open(paths[0], "rt", encoding="utf-8") as off,
          gzip.open(paths[1], "rt", encoding="utf-8") as on):
        header_off, header_on = json.loads(next(off)), json.loads(next(on))
        if header_off["route"] != header_on["route"]:
            raise ValueError("Route mismatch between paired replays.")
        for raw_off, raw_on in zip_longest(off, on):
            frames = tuple(json.loads(raw) if raw is not None else None for raw in (raw_off, raw_on))
            for tracker, frame in zip(tracks, frames):
                if frame is not None:
                    tracker.observe(frame)
            a, b = frames
            if a is None or b is None:
                continue
            oa, ob = a["observation"], b["observation"]
            if (oa["time"], oa["ais_targets"]) != (ob["time"], ob["ais_targets"]):
                raise ValueError(f"Paired report clock or AIS reception differs at fix {compared}.")
            state_differs = a["state"] != b["state"]
            command_differs = command(a) != command(b)
            weather_differs = any(oa[name] != ob[name] for name in ("wind", "current", "wave"))
            gap = distance_m(a["state"], b["state"]) if state_differs else 0.
            if (state_differs or command_differs) and divergence is None:
                if compared == 0:
                    raise ValueError("Paired initial boat states do not match.")
                divergence = first_divergence(a, b, previous_utc, gap)
            if weather_differs and divergence is None:
                raise ValueError("Paired archived observations differ before track or command divergence.")
            compared += 1
            changed_commands += command_differs
            changed_states += state_differs
            changed_weather += weather_differs
            max_gap, end_gap, previous_utc = max(max_gap, gap), gap, ob["time"]
    if tracks[0].effects:
        raise ValueError("Layline reference appeared with the mode off.")
    return tracks, {
        "common_fixes_with_identical_report_receipts": compared,
        "changed_commands_on_shared_fixes": changed_commands, "changed_states_on_shared_fixes": changed_states,
        "changed_local_forcing_samples_on_shared_fixes": changed_weather,
        "maximum_track_separation_m": max_gap, "last_shared_track_separation_m": end_gap,
        "first_state_or_command_difference": divergence,
        "all_states_and_commands_match": divergence is None and tracks[0].fixes == tracks[1].fixes,
    }


def audit_case(directory, wind, mode, route, sweep, native, gate, tracker):
    cmd = shlex.split((directory / "command.txt").read_text(encoding="utf-8"))
    if cmd.count(FLAG) != int(mode == "on"):
        raise ValueError(f"Option mismatch in {directory.name}.")
    replay_sha = digest(directory / "navigation.jsonl.gz")
    log_sha = digest(directory / "run.log")
    if not (replay_sha == route["sha256"]["navigation.jsonl.gz"] == sweep["replay_sha256"] == native["replay_sha256"]
            and log_sha == route["sha256"]["run.log"] == sweep["run_log_sha256"] == native["run_log_sha256"]):
        raise ValueError(f"Audit hashes differ from the live run files for {directory.name}.")
    if not (tracker.fixes == route["steps"] + 1 == sweep["frames"]
            and route["steps"] == sweep["applied_actions"] == native["applied_attempts"]
            and route["elapsed_s"] == route["steps"] * DT and route["wind_from_degrees"] == wind):
        raise ValueError(f"Run counts or protocol disagree for {directory.name}.")
    captures = route["captures"]
    capture_order = tuple((row["phase"], row["waypoint"]) for row in captures)
    ordered_prefix = capture_order == SEQ[:len(capture_order)]
    inside_radius = all(0. <= row["distance_metres"] <= RADIUS_M for row in captures)
    complete = route["status"] == "completed" and capture_order == SEQ and inside_radius
    if route["status"] != sweep["mission_status"] or gate["actual_departure_utc"] != route["actual_departure_utc"]:
        raise ValueError(f"Route, clearance and gate audits disagree for {directory.name}.")
    to_002 = next((row["elapsed_s"] for row in captures if (row["phase"], row["waypoint"]) == SEQ[0]), None)
    truth_margin = sweep["configured_truth_centered_minimum_m"]
    report_margin = sweep["configured_model_minimum_m"]
    margin_positive = all(x is not None and isfinite(x) and x > 0. for x in (truth_margin, report_margin))
    return {
        "status": route["status"], "wind_from_deg": wind, "closest_laylines": mode,
        "elapsed_to_outbound_002_s": to_002, "elapsed_to_loop_completion_s": route["elapsed_s"] if complete else None,
        "observed_underway_s": route["elapsed_s"], "elapsed_from_monitoring_start_s": route["total_elapsed_s"],
        "actual_departure_utc": route["actual_departure_utc"], "gate_wait_s": route["gate_wait_seconds"],
        "capture_count": len(captures), "captures": captures, "ordered_capture_prefix": ordered_prefix,
        "all_captures_inside_50m": inside_radius, "complete_ordered_loop_at_50m": complete,
        "full_truth_plus_3m_margin_m": truth_margin, "full_report_plus_3m_margin_m": report_margin,
        "positive_full_margins": margin_positive, "negative_action_counts": sweep["negative_action_counts"],
        "staged_gate_max_s": gate["maximum_gate_decision_s"],
        "maximum_AIS_choose_s": sweep["maximum_complete_AIS_choose_s"],
        "maximum_AIS_accept_or_stop_s": sweep["maximum_complete_AIS_applied_decision_s"],
        "maximum_full_control_s": native["maximum_complete_native_control_decision_s"],
        "full_control_attempts_over_or_at_5s": native["attempts_at_or_above_5s"],
        "passes_all_5s_checks": (native["all_complete_native_control_attempts_below_5s"]
                                and sweep["all_AIS_choose_attempts_below_5s"]
                                and sweep["all_AIS_applied_decision_attempts_below_5s"]
                                and gate["all_gate_attempts_below_5s"]),
        "maximum_held_report_age_s": sweep["maximum_applied_held_report_age_s"],
        "both_underway_65s_age_peaks": sweep["underway_both65s_age_peaks_verified"],
        "replay_sha256": replay_sha, "run_log_sha256": log_sha, **tracker.summary(),
    }


def delta(after, before):
    return after - before if after is not None and before is not None else None


def paired_capture_rows(wind, off, on):
    indexed = [{(r["phase"], r["waypoint"]): r for r in run["captures"]} for run in (off, on)]
    fields = {"capture_time_s": "elapsed_s", "segment_s": "segment_s", "segment_m": "segment_distance_m",
              "capture_distance_m": "distance_metres"}
    prev = "001"
    for phase, goal in SEQ:
        row = {"wind_from_deg": wind, "phase": phase, "from": prev, "to": goal}
        for label, key in fields.items():
            a, b = [lookup.get((phase, goal), {}).get(key) for lookup in indexed]
            row.update({f"off_{label}": a, f"on_{label}": b, f"delta_{label}": delta(b, a)})
        yield row
        prev = goal


def clock(value, signed=False):
    if value is None:
        return "—"
    seconds = int(round(abs(value)))
    sign = "−" if value < 0. else "+" if signed and value else ""
    hour, remaining = divmod(seconds, 3600)
    minute, second = divmod(remaining, 60)
    return f"{sign}{hour}:{minute:02}:{second:02}"


def margin(value):
    return "—" if value is None else f"{value:+.2f} m"


def report_markdown(runs, pairs):
    launches = {(row["actual_departure_utc"], row["gate_wait_s"]) for row in runs.values()}
    launch_note = ""
    if len(launches) == 1:
        when, wait = next(iter(launches))
        launch_note = f" All four runs launched at {when} after {clock(wait)} of protected staging."
    lines = [
        "# Otranto closest-layline comparison",
        "",
        "The four runs pair closest laylines off and on at wind bearings FROM 355° and 5°. "
        "They use wind speed, current and waves from the same 30 September 2026 archive, with wind direction fixed "
        "within each pair. Both modes run Gamma with "
        "Direct acados NMPC and the same Transformer weather and AIS schedule veto. The crossing-fan AIS encounter, "
        f"20 s receiver delay, staged and underway dropouts, COLREG and the 3 m reserve are shared.{launch_note}",
        "",
        "Times start at actual release from protected staging. Total time measures return capture at 001. "
        "Clearance is the smallest measured excess over vessel hulls, the 25 m gap, full uncertainty and the 3 m "
        "reserve. Truth and report projection use separate independently swept checks. The worst full decision time "
        "includes observation, chart, guidance, Direct NMPC and action acceptance. Plant propagation and recording "
        "follow the timed decision.",
        "",
        "| Wind FROM | Closest laylines | Outbound 002 | Round trip | Ordered 50 m captures | "
        "Truth margin | Report margin | Max full decision | 5 s budget | Applied layline frames |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|:---|---:|",
    ]
    for wind in WINDS:
        for mode in MODES:
            r = runs[arm_name(wind, mode)]
            captures = f'{r["capture_count"]}/8'
            if not (r["ordered_capture_prefix"] and r["all_captures_inside_50m"]):
                captures += " failed check"
            decision = r["maximum_full_control_s"]
            timing = "—" if decision is None else f"{decision:.3f} s"
            lines.append(
                f'| {wind}° | {mode.capitalize()} | {clock(r["elapsed_to_outbound_002_s"])} | '
                f'{clock(r["elapsed_to_loop_completion_s"])} | {captures} | '
                f'{margin(r["full_truth_plus_3m_margin_m"])} | {margin(r["full_report_plus_3m_margin_m"])} | '
                f'{timing} | {"Pass" if r["passes_all_5s_checks"] else "Fail"} | {r["applied_layline_frames"]} |'
            )
    lines.extend(["", "| Wind FROM | Time to 002 on minus off | Total on minus off | "
                  "Maximum track separation at a shared fix |",
                  "|---:|---:|---:|---:|"])
    for wind in WINDS:
        pair = pairs[str(wind)]
        lines.append(f'| {wind}° | {clock(pair["delta_to_002_s"], signed=True)} | '
                     f'{clock(pair["delta_total_s"], signed=True)} | {pair["maximum_track_separation_m"]:.2f} m |')
    lines.append("")
    for wind in WINDS:
        pair, treated = pairs[str(wind)], runs[arm_name(wind, "on")]
        effect, split = treated["first_applied_layline"], pair["first_state_or_command_difference"]
        if effect is None:
            detail = f'The {wind}° run had zero controls with an applied closest-layline reference.'
        else:
            detail = (f'The {wind}° run applied {treated["applied_layline_frames"]} closest-layline references, '
                      f'with {treated["applied_layline_frames_to_outbound_002"]} before outbound 002. '
                      f'The first was {effect["source"]} toward {effect["phase"]} {effect["waypoint"]} '
                      f'at replay fix {effect["step"]} ({effect["utc_at_fix"]} UTC).')
        if pair["all_states_and_commands_match"]:
            detail += f' All {pair["common_fixes_with_identical_report_receipts"]:,} corresponding states and commands matched.'
        elif split is not None:
            detail += f' The first state or actuator difference was at fix {split["step"]}.'
        lines.append(f"- {detail}")
    if all(runs[arm_name(wind, "on")]["applied_layline_frames_to_outbound_002"] == 0 for wind in WINDS):
        lines += ["", "The enabled runs each recorded zero fallback controls before 002. Hold and forecast priorities "
                  "applied through the opening leg."]
        for wind in WINDS:
            treated = runs[arm_name(wind, "on")]
            observed_hold = DT * treated["guidance_context_applied_frames_to_outbound_002"].get("drift hold", 0)
            max_range = treated["maximum_range_to_outbound_002_m"]
            lines.append(f"The {wind}° run logged {clock(observed_hold)} of drift hold before 002, "
                         f"with a maximum recorded range to 002 of {max_range / 1000.:.2f} km.")
    lines += [
        "",
        "A chart-clear layline can supply a steering reference after the hold and AIS gates admit nominal guidance. "
        "A forecast capture, scheduled tack or committed maneuver retains priority. An enabled flag therefore "
        "measures the effect of this fallback in the integrated controller.",
        "",
        "The prescribed capture order is outbound 002 → 003 → 004 → 005, then return 004 → 003 → 002 → 001.",
        "",
        "[Per-leg times, distances and captures](paired_layline_captures.csv) · "
        "[Applied layline references](applied_layline_frames.csv) · [Paired replay comparison](layline_comparison.json) · "
        "[Independent acceptance summary](independent_final_summary.json) · "
        "[Matched inputs](independent_inputs_verification.json)",
        "",
        "## Replays and videos",
        "",
        "| Wind FROM | Closest laylines | Raw replay | Labeled video |",
        "|---:|:---|:---|:---|",
    ]
    for wind in WINDS:
        for mode in MODES:
            name = arm_name(wind, mode)
            lines.append(f"| {wind}° | {mode.capitalize()} | [Replay](runs/{name}/navigation.jsonl.gz) | "
                         f"[MP4](runs/{name}/navigation.mp4) |")
    lines += ["", "[Video verification](native_video_verification.json)", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    study = args.study
    routes = load(study / "independent_route_verification.json")["cases"]
    sweeps = {r["case"]: r for r in load(study / "independent_traffic_sweep.json")}
    native = load(study / "independent_native_control_timing.json")["cases"]
    gates = {r["case"]: r for r in load(study / "independent_departure_gate_check.json")["cases"]}
    expected = {arm_name(wind, mode) for wind in WINDS for mode in MODES}
    if any(set(data) != expected for data in (routes, sweeps, native, gates)):
        raise ValueError("Wait for all four audited runs before publishing the paired comparison.")
    runs, pairs, captures, effects = {}, {}, [], []
    for wind in WINDS:
        names = tuple(arm_name(wind, mode) for mode in MODES)
        dirs = tuple(study / "runs" / name for name in names)
        tracks, replay_comparison = compare_replays([directory / "navigation.jsonl.gz" for directory in dirs])
        for name, directory, mode, tracker in zip(names, dirs, MODES, tracks, strict=True):
            runs[name] = audit_case(directory, wind, mode, routes[name], sweeps[name], native[name], gates[name], tracker)
            effects.extend({"case": name, **row} for row in tracker.effects)
        off, on = (runs[name] for name in names)
        if (off["actual_departure_utc"], off["gate_wait_s"]) != (on["actual_departure_utc"], on["gate_wait_s"]):
            raise ValueError(f"Paired release differs at {wind}°.")
        pairs[str(wind)] = {
            "delta_to_002_s": delta(on["elapsed_to_outbound_002_s"], off["elapsed_to_outbound_002_s"]),
            "delta_total_s": delta(on["elapsed_to_loop_completion_s"], off["elapsed_to_loop_completion_s"]),
            "delta_maximum_opening_range_m": delta(on["maximum_range_to_outbound_002_m"],
                                                  off["maximum_range_to_outbound_002_m"]),
            **replay_comparison,
        }
        captures.extend(paired_capture_rows(wind, off, on))
    save_csv(study / "paired_layline_captures.csv", captures, list(captures[0]))
    effect_columns = ["case", "step", "utc_at_fix", "phase", "waypoint", "source", "planned_first_distance_m",
                      "planned_eta_s", "heading_over_ground_reference_deg", "bow_reference_deg", "second_bow_deg",
                      "rudder_command_deg", "flap_command_deg"]
    save_csv(study / "applied_layline_frames.csv", effects, effect_columns)
    summary = {"comparison_sign": "on minus off", "runs": runs, "pairs": pairs}
    (study / "layline_comparison.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (study / "README.md").write_text(report_markdown(runs, pairs), encoding="utf-8")
    for wind in WINDS:
        pa = pairs[str(wind)]
        print(f"{wind}° · delta to 002 {clock(pa['delta_to_002_s'], signed=True)} · "
              f"delta round trip {clock(pa['delta_total_s'], signed=True)} · "
              f"maximum separation {pa['maximum_track_separation_m']:.2f} m")


if __name__ == "__main__":
    main()
