"""Render AIS-gated departures with fixed traffic UTC and separate wait/sailed timing."""

from __future__ import annotations

import argparse
import csv
import gzip
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import math
import os
import textwrap
from pathlib import Path
import sys
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import numpy as np


STUDY = Path(__file__).resolve().parents[1]
WORKSPACE = STUDY.parents[2]
sys.path[:0] = [str(WORKSPACE), str(STUDY.parent / "wind_direction_ablation_20261118/reproduce")]

import render as reference
from world import navigation_replay as replay_io
from world._geo import local_east_north_metres
from world._geo import offset_latitude_longitude
from world.ais_avoidance import AISPrediction
from audit_independent import Truth


def baseline_manifest(study):
    return {}


def source_directory(study, run_id):
    return study / "runs" / run_id


def output_directory(study, run_id):
    return study / "runs" / run_id


def mission_label(status):
    if status=='no_full_clearance_action':
        return 'Stopped: no tested full-clearance action'
    if status=='predicted_hull_contact':
        return 'Stopped: predicted next-step hull contact'
    return reference.mission_label(status)


def short_event_camera(study,arms):
    """One fixed camera for all short stops, including actual report domains."""
    camera=reference.camera_for(study,tuple(arms))
    low=np.array([camera[0],camera[2]],dtype=float)
    high=np.array([camera[1],camera[3]],dtype=float)
    for arm in arms:
        replay=replay_io.load_navigation_replay(source_directory(study,arm)/'navigation.jsonl.gz')
        origin=replay.frames[0].state
        for frame in replay.frames:
            traffic=AISPrediction.from_observation(frame.observation,acceleration_bound_mps2=.026,
                                                   clearance_reserve_metres=3.)
            lat,lon=offset_latitude_longitude(traffic.origin_latitude,traffic.origin_longitude,
                                              *traffic.positions_metres.T)
            centers=np.column_stack(local_east_north_metres(lat,lon,origin.latitude,origin.longitude))
            radii=traffic.clearance_radii_metres[:,None]
            low=np.minimum(low,np.min(centers-radii,axis=0))
            high=np.maximum(high,np.max(centers+radii,axis=0))
    pad=max(60.,.055*float(max(high-low)))
    return [float(low[0]-pad),float(high[0]+pad),float(low[1]-pad),float(high[1]+pad)]


def wrapped_label(value):
    lines = textwrap.wrap(value, width=68)
    if len(lines) > 1 and len(lines[-1]) < 12:
        words = lines[-2].split()
        while words and len(lines[-1]) < 12:
            lines[-1] = words.pop() + " " + lines[-1]
        lines[-2] = " ".join(words)
    return lines


def method_lines(study, run_id):
    folder = source_directory(study, run_id)
    manifest_path = folder / "execution.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    learned = manifest["method"] == "transformer"
    description = (
        f"Encounter: {manifest['scenario_id'].replace('_',' ')}",
        "Frozen temporal Transformer hold/release + physical confirmation" if learned else "Classical hold/release guidance",
        "Gamma + Direct acados NMPC; forecast hold/tacks",
        "Refined forecast return; true 50m GPX gates",
        "Protected-staging AIS gate; strict receding-horizon fallback",
        "AIS/COLREG:20s delay;two 40s drops;preview age65s",
        "25m nominal AIS gap + full uncertainty +3m reserve",
    )
    return tuple(line for value in description for line in wrapped_label(value))


ORDER = ["002", "003", "004", "005", "004", "003", "002", "001"]
DEPARTURE = np.datetime64("2026-09-30T23:50:00")
UPDATE_SECONDS = 5
CAPTURE_RADIUS = 50


def clock(seconds: int) -> str:
    hours, tail = divmod(seconds, 3600)
    minutes, seconds = divmod(tail, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def displayed_frame(frame, step, capture, truth_margin=None, method=None):
    visual = reference.displayed_frame(frame, step, capture)
    raw = frame.navigation
    biased = raw.get("forecast_gate_aim_offset_m", 0.0) > 0.0
    navigation = {
        ("forecast_aim_closest_m" if biased and key == "forecast_closest_m" else key): value
        for key, value in visual.navigation.items()
    }
    if biased:
        navigation.update({
            "nominal_radius_m": raw["forecast_gate_aim_radius_m"],
            "aim_offset_m": raw["forecast_gate_aim_offset_m"],
            "true_capture_radius_m": raw["forecast_gate_true_radius_m"],
        })
    for raw_key, label in (
        ("forecast_maneuver_commit_active", "commit_active"),
        ("forecast_maneuver_commit_side", "commit_side"),
        ("forecast_maneuver_commit_remaining_s", "commit_remaining_s"),
        ("forecast_gate_aim_guard_reason", "gate_aim_guard"),
        ("forecast_gate_continuation_wind_angle_deg", "next_leg_wind_angle_deg"),
        ("forecast_gate_aim_upwind_angle_limit_deg", "gate_aim_angle_limit_deg"),
    ):
        if raw_key in raw:
            value = raw[raw_key]
            navigation[label] = "outside upwind cone" if value == "continuation outside upwind cone" else value
    if "forecast_gate_aim_guard_reason" in raw:
        navigation["gate_aim_active"] = int(biased)
    for key, label in (
        ("ais_mode", "AIS mode"), ("ais_colreg_active_contacts", "COLREG contacts"),
        ("ais_step_report_age_s", "Used report age s"),
        ("ais_step_reserve_margin_m", "Projected step margin m"),
        ("ais_control", "AIS control"),
        ("ais_recovery_backend", "AIS recovery backend"),
        ("ais_recovery_active", "AIS recovery active"),
        ("ais_recovery_reserve_margin_m", "Recovery forecast margin m"),
        ("ais_recovery_remaining_s", "Recovery remaining s"),
        ("ais_recovery_departure_turn_deg", "Recovery preview turn deg"),
        ("ais_colreg_soft_override", "COLREG soft override"),
        ("ais_maneuver_search_seconds", "AIS choose wall s"),
        ("ais_applied_decision_seconds", "AIS accepted/stop wall s"),
        ("temporal_recovery_applied_action", "Weather action"),
        ("temporal_recovery_source", "Weather source"),
        ("temporal_recovery_raw_action", "Model proposal"),
        ("temporal_recovery_classical_action", "Classical choice"),
        ("temporal_recovery_p_hold", "Model P hold"),
        ("temporal_recovery_release_guard", "Release guard"),
        ("drift_mode", "Drift mode"),
        ("gate_vmg_mps", "Gate VMG mps"),
    ):
        if raw.get(key) is not None:
            navigation[label] = raw[key]
    if truth_margin is not None:
        navigation["Truth step margin m"] = truth_margin
    if frame.state.surge_speed is not None:
        navigation["Water surge mps"] = frame.state.surge_speed
        if frame.state.surge_speed < -0.05:
            navigation["Water motion"] = "STERNWAY (AIS recovery)" if raw.get("ais_recovery_active") else "STERNWAY"
    mode = str(raw.get("ais_mode", ""))
    priority = (mode in {"buffer maneuver", "predictive fallback"} or mode.startswith("fallback")
                or bool(raw.get("ais_recovery_active"))
                or raw.get("ais_colreg_active_contacts", 0) > 0
                or raw.get("nominal_guidance_suspended")
                or raw.get("forecast_suspension_reason") == "AIS encounter")
    if priority:
        navigation["Weather action"] = "AIS priority"
        navigation["Weather source"] = "Classical" if method == "classical" else "suppressed"
    elif method == "classical":
        navigation["Weather action"] = (
            "hold" if raw.get("drift_mode") == "holding"
            else "sail" if raw.get("forecast_closest_m") is not None else "planner fallback"
        )
        navigation["Weather source"] = "Classical"
    elif navigation.get("Weather source") == "classical fallback":
        navigation["Weather source"] = "fallback"
    return replace(visual, navigation=navigation)


def shared_anchor_steps(study, arms):
    anchors = set()
    for arm in arms:
        with gzip.open(source_directory(study, arm) / "navigation.jsonl.gz", "rt") as stream:
            next(stream)
            previous = None
            for i, line in enumerate(stream):
                nav = json.loads(line)["navigation"]
                state = (nav.get("temporal_recovery_source"), nav.get("temporal_recovery_applied_action"),
                         nav.get("temporal_recovery_raw_action"), nav.get("temporal_recovery_classical_action"),
                         nav.get("control_source"), nav.get("drift_mode"),
                         bool(nav.get("ais_recovery_active")), nav.get("ais_control"), nav.get("ais_recovery_backend"),
                         bool(nav.get("ais_colreg_soft_override")),
                         nav.get("ais_mode"), bool(nav.get("ais_colreg_active_contacts", 0)))
                if state != previous:
                    anchors.update((max(0, i - 1), i, i + 1))
                    previous = state
    return sorted(anchors)


def round_trip_animation(camera, source_positions, source_indices, arrival_steps, aim_positions, aim_radii,
                         reported_positions, full_radii, nominal_radii, truth_positions,
                         launch_summary=None, launch_summary_frames=0):
    chart = reference.chart_animation(camera, CAPTURE_RADIUS)
    captures = np.asarray(arrival_steps, dtype=int)

    class RoundTripAnimation(chart):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # A fixed figure-space anchor keeps longer method labels inside
            # the canvas; constrained layout otherwise follows changing HUD
            # extents and can crop the first Transformer title line.
            self.figure.set_layout_engine(None)
            self.axis.set_position((.055, .13, .54, .77))
            self.readout.axes.set_position((.615, .04, .375, .92))
            self.readout.set_transform(self.figure.transFigure)
            self.readout.set_position((.62, .97))
            self.readout.set_fontsize(8.0)
            self.readout.set_linespacing(1.08)
            self.readout.set_in_layout(False)
            self.launch_summary=self.axis.text(.035,.97,launch_summary or "",transform=self.axis.transAxes,
                                              va="top",ha="left",fontsize=9.5,fontfamily="monospace",zorder=50,
                                              bbox={"facecolor":"white","edgecolor":"#69798b","alpha":.97,"pad":9})
            self.capture_points = self.axis.scatter([], [], s=23, c="#CB7A18", edgecolors="white",
                                                    linewidths=0.65, zorder=5, label="capture")
            has_aim = bool(np.isfinite(aim_radii).any())
            self.nominal_aim, = self.axis.plot([], [], marker="+", linestyle="none", color="#D97812",
                                               markersize=9, markeredgewidth=1.3, zorder=5.5,
                                               label="nominal aim" if has_aim else "_nolegend_")
            self.nominal_disk = Circle((0, 0), 1.0, facecolor="none", edgecolor="#D97812",
                                       linewidth=1.2, linestyle=":", zorder=3.5,
                                       label="nominal disk" if has_aim else "_nolegend_")
            self.axis.add_patch(self.nominal_disk)
            self.truth_centres = self.axis.scatter([], [], facecolors="none", edgecolors="#7345a0",
                                                 s=45, linewidths=1.0, zorder=6,
                                                 label="true AIS center")
            self.report_disks, self.nominal_ais_disks = [], []
            for target in range(3):
                disk = Circle((0, 0), 1, facecolor="none", edgecolor="#ba3333", linewidth=.9,
                              linestyle="--", alpha=.8, zorder=2.8,
                              label="AIS full+reserve" if target == 0 else "_nolegend_")
                nominal = Circle((0, 0), 1, facecolor="none", edgecolor="#777777", linewidth=.7,
                                 linestyle=":", alpha=.7, zorder=2.7,
                                 label="AIS nominal25m" if target == 0 else "_nolegend_")
                self.axis.add_patch(disk); self.axis.add_patch(nominal)
                self.report_disks.append(disk); self.nominal_ais_disks.append(nominal)

        def set_route(self, route):
            super().set_route(route)
            self.axis.add_patch(Circle(self.route_positions[0], CAPTURE_RADIUS, alpha=0.16,
                                       facecolor="#2477b8", edgecolor="none", zorder=1))
            # The open-water helper suppresses AIS legend entries. This
            # traffic study explicitly distinguishes reports, truth and domains.
            handles, labels = self.axis.get_legend_handles_labels()
            visible = []
            for handle, label in zip(handles, labels):
                if label.startswith("AIS predicted"):
                    continue
                if label == "AIS course/speed":
                    handle = Line2D([], [], marker=">", color="#ff4d4d", linestyle="none", markersize=6)
                visible.append((handle, label))
            self.readout.axes.legend(*zip(*visible), loc="lower left", ncol=2, fontsize=8)

        def update(self, index):
            artists = super().update(index)
            self.readout.set_text("\n".join(
                "\n".join(textwrap.wrap(line, width=68, break_long_words=False,
                                        break_on_hyphens=False)) if line else ""
                for line in self.readout.get_text().splitlines()
            ))
            self.launch_summary.set_visible(index<launch_summary_frames)
            step = source_indices[index]
            self.path.set_data(source_positions[:step + 1].T)
            self.capture_points.set_offsets(source_positions[captures[captures <= step]])
            has_aim = bool(np.isfinite(aim_radii[step]))
            self.nominal_aim.set_visible(has_aim)
            self.nominal_disk.set_visible(has_aim)
            if has_aim:
                center = aim_positions[step]
                self.nominal_aim.set_data([center[0]], [center[1]])
                self.nominal_disk.center = center
                self.nominal_disk.set_radius(float(aim_radii[step]))
            self.truth_centres.set_offsets(truth_positions[step])
            for target in range(3):
                self.report_disks[target].center = reported_positions[step, target]
                self.report_disks[target].set_radius(float(full_radii[step, target]))
                self.nominal_ais_disks[target].center = reported_positions[step, target]
                self.nominal_ais_disks[target].set_radius(float(nominal_radii[step, target]))
            return (*artists, self.capture_points, self.nominal_aim, self.nominal_disk, self.truth_centres,
                    *self.report_disks, *self.nominal_ais_disks)

    return RoundTripAnimation


class SingleThreadWriter(replay_io.FFMpegWriter):
    def __init__(self, *args, extra_args=None, **kwargs):
        super().__init__(*args, extra_args=[*(extra_args or ()), "-threads", "1"], **kwargs)


def render(task, preview_directory=None):
    study, run_id, camera, fps, frame_limit, *anchor_values = task
    anchors = anchor_values[0] if anchor_values else shared_anchor_steps(study, (run_id,))
    directory = source_directory(study, run_id)
    metadata = json.loads((directory / "execution.json").read_text())
    output = output_directory(study, run_id)
    output.mkdir(parents=True, exist_ok=True)
    replay_path = directory / "navigation.jsonl.gz"
    replay = replay_io.load_navigation_replay(replay_path)
    mission = reference.recorded_mission(directory, 0)
    count = len(replay.frames)
    independent, = [row for row in json.loads((study / "independent_traffic_sweep.json").read_text())
                     if row["case"] == run_id]
    if independent["replay_sha256"] != reference.digest(replay_path) or independent["frames"] != count:
        raise ValueError(f"{run_id}: independent clearance receipt differs from the replay")
    gate_path=directory/"departure_gate.json"
    gate=json.loads(gate_path.read_text())
    actual_utc=gate.get("actual_departure_utc")
    if not actual_utc:
        raise ValueError(f"{run_id}: no actual launch; this is a no-launch outcome, not a sailed replay")
    actual_departure=np.datetime64(str(actual_utc).removesuffix("Z"))
    waiting=float(gate["gate_wait_seconds"])
    gate_log_path=directory/"departure_gate.jsonl"
    if reference.digest(gate_log_path)!=gate["log_sha256"]:
        raise ValueError(f"{run_id}: gate log hash differs from summary")
    gate_rows=[json.loads(line) for line in gate_log_path.read_text().splitlines()]
    if len(gate_rows)!=gate["attempts"] or gate_rows[-1]!=gate["certificate"]:
        raise ValueError(f"{run_id}: gate attempt/certificate mismatch")
    certificate=gate["certificate"]
    prelaunch_age=max(float(target["report_age_seconds"]) for row in gate_rows
                      for target in row["received_reports"])
    release_age=max(float(target["report_age_seconds"]) for target in certificate["received_reports"])
    gate_text="\n".join((
        "RECEIVED-AIS LAUNCH SUMMARY",
        "Protected staging outside modeled open water",
        "No in-water waiting trajectory is assumed",
        f"Requested: 2026-09-30 23:50:00 UTC",
        f"Actual: {str(actual_departure).replace('T',' ')} UTC",
        f"Sheltered wait: {waiting:g}s; fresh renewals: {certificate['clear_streak']}",
        "Fixed traffic epoch: 23:50:40 UTC (Sep 30)",
        "Fixed emission losses: 23:52:05-45 / 23:53:05-45 UTC",
        f"Snapshot report age max: wait {prelaunch_age:g}s / release {release_age:g}s",
        f"Projected uncapped 120s launch tube: {certificate['min_uncapped_startup_margin_m']:+.2f}m",
        f"Projected 300s tube (65s age cap): {certificate['min_preview_margin_m']:+.2f}m",
        f"Robust receding speed lower bound: {certificate['min_robust_receding_mps']:+.2f}m/s",
        "Gate uses received reports; truth audit is postlaunch",
        "Plot beneath shows the actual launch fix only",
    ))
    if (actual_departure-DEPARTURE)/np.timedelta64(1,"s")!=waiting:
        raise ValueError(f"{run_id}: requested/actual departure and wait differ")
    if np.datetime64(str(gate["traffic_epoch_utc"]).removesuffix("Z"))!=np.datetime64("2026-09-30T23:50:40"):
        raise ValueError(f"{run_id}: original traffic epoch was shifted")
    arrivals = mission["waypoint_arrivals"]
    if (count != mission["steps"] + 1 or mission["arrival_radius_metres"] != CAPTURE_RADIUS
            or not mission["return_to_start"] or replay.frames[0].observation.time != actual_departure):
        raise ValueError(f"{run_id}: replay and mission settings differ")
    if (mission["status"] == "completed" and len(arrivals) != len(ORDER)
            or [arrival["waypoint"] for arrival in arrivals] != ORDER[:len(arrivals)]):
        raise ValueError(f"{run_id}: unexpected capture sequence")

    event = {point["step"]: f"{point['waypoint']} ({point['distance_metres']:.1f} m)" for point in arrivals}
    if not all(0 <= step < count and replay.frames[step].navigation.get("step") == step for step in event):
        raise ValueError(f"{run_id}: capture indices do not identify recorded fixes")
    # Preserve every five-second encounter update, then sample the long loop.
    retain = {0, count - 1, *range(min(count, 181))}
    retain.update(step for step in anchors if 0 <= step < count)
    for step in event:
        retain.update(range(max(0, step - 1), min(count, step + 2)))
    retain.update(i for i, frame in enumerate(replay.frames)
                  if frame.navigation.get("ais_colreg_active_contacts", 0) > 0)
    if len(retain) > frame_limit:
        chosen = sorted(retain)
    else:
        remaining = [i for i in range(count) if i not in retain]
        indexes = np.linspace(0, len(remaining) - 1, min(len(remaining), frame_limit - len(retain)), dtype=int)
        chosen = sorted(retain | {remaining[i] for i in indexes})
    held_frames = max(1, round(0.5 * fps))
    indices = [step for step in chosen for _ in range(held_frames if step in event else 1)]
    minimum_clip_frames=max(2,round(12*fps))
    if len(indices)<minimum_clip_frames:
        final_hold=max(1,round(2*fps))
        body_length=max(len(indices),minimum_clip_frames)
        quotient,remainder=divmod(body_length,len(indices))
        indices=[step for i,step in enumerate(indices) for _ in range(quotient+(i<remainder))]+[indices[-1]]*final_hold
    launch_summary_frames=max(1,round(4*fps))
    indices=[0]*launch_summary_frames+indices
    elapsed = UPDATE_SECONDS * mission["steps"]
    attempts=[json.loads(line.removeprefix("AIS decision timing: "))
              for line in (directory/'run.log').read_text().splitlines()
              if line.startswith("AIS decision timing: ")]
    applied_attempts=[json.loads(line.removeprefix("AIS applied decision timing: "))
                      for line in (directory/'run.log').read_text().splitlines()
                      if line.startswith("AIS applied decision timing: ")]
    applied_seconds=[float(row['wall_seconds']) for row in applied_attempts]
    if not applied_seconds or not all(math.isfinite(value) and value>=0 for value in applied_seconds):
        raise ValueError(f"{run_id}: missing/nonfinite complete applied-decision timings")
    applied_by_step={row['step']:row for row in applied_attempts}
    if len(applied_by_step)!=len(applied_attempts):
        raise ValueError(f"{run_id}: duplicate applied-decision attempt steps")
    choose_seconds=[float(row['wall_seconds']) for row in attempts]
    if not choose_seconds or not all(math.isfinite(value) and value>=0 for value in choose_seconds):
        raise ValueError(f"{run_id}: missing/nonfinite complete AIS attempt timings")
    by_step={row['step']:row for row in attempts}
    if len(by_step)!=len(attempts):raise ValueError(f"{run_id}: duplicate decision attempt steps")
    for frame in replay.frames[1:]:
        navigation=frame.navigation
        if (navigation['step'] not in by_step or
            abs(float(navigation['ais_maneuver_search_seconds'])-float(by_step[navigation['step']]['wall_seconds']))>1e-9):
            raise ValueError(f"{run_id}: applied replay/timing attempt mismatch")
        row=applied_by_step.get(navigation['step'])
        if (row is None or not row['applied'] or
            abs(float(navigation['ais_applied_decision_seconds'])-float(row['wall_seconds']))>1e-9):
            raise ValueError(f"{run_id}: applied replay/accepted-decision timer mismatch")
    if set(applied_by_step)!=set(by_step):
        raise ValueError(f"{run_id}: choose and applied-decision attempt steps differ")
    for step,row in applied_by_step.items():
        if row['wall_seconds']+1e-9<by_step[step]['wall_seconds']:
            raise ValueError(f"{run_id}: applied decision timer excludes choose work")
    accepted_latency={"calls":len(applied_seconds),
                      "applied_calls":sum(bool(row['applied']) for row in applied_attempts),
                      "unapplied_calls":sum(not row['applied'] for row in applied_attempts),
                      "first_seconds":applied_seconds[0],"maximum_seconds":max(applied_seconds),
                      "all_calls_below_5s":all(value<5 for value in applied_seconds),
                      "scope":"Choose start through strict action acceptance or stop, including the final unapplied attempt."}
    if accepted_latency['applied_calls']!=mission['steps']:
        raise ValueError(f"{run_id}: accepted attempt count differs from native applied states")
    latency = {"calls": len(choose_seconds), "applied_calls": mission['steps'],
               "unapplied_calls":len(choose_seconds)-mission['steps'],
               "first_seconds": choose_seconds[0], "maximum_seconds": max(choose_seconds),
               "all_calls_below_5s": all(value < 5.0 for value in choose_seconds),
               "scope":"All try/finally logged AIS choose attempts, including the final unapplied stop/exception attempt."}
    def metric(key):
        value=independent.get(key)
        return None if value is None else float(value)
    truth_margin=metric('configured_truth_centered_minimum_m')
    report_margin=metric('configured_model_minimum_m')
    clearance_available=truth_margin is not None and report_margin is not None
    clearance_label=('PASS' if min(truth_margin,report_margin)>=0 else 'FAIL') if clearance_available else 'NOT EVALUATED: no applied command'
    truth_text=f'{truth_margin:.2f} m' if truth_margin is not None else 'not measured'
    report_text=f'{report_margin:.2f} m' if report_margin is not None else 'not measured'
    held_age=independent.get('maximum_applied_held_report_age_s')
    exposure=(f"Postlaunch held age: {held_age:g}s;65s peak {'REACHED' if independent.get('longest_fault_peak_experienced') else 'NOT reached'}"
              if held_age is not None else 'Held report-age exposure: not measured')
    labels = (
        f"Otranto round trip · wind FROM {metadata['wind_from_degrees']:g}°",
        f"Actual launch {str(actual_departure).replace('T',' ')} UTC; protected wait {clock(int(waiting))}",
        *method_lines(study, run_id),
        "Fixed wind bearing; archived speed, waves and current",
        f"AIS choose max: {latency['maximum_seconds']:.3f} s; accept/stop max: {accepted_latency['maximum_seconds']:.3f} s",
        f"Complete accept/stop budget: {'PASS <5 s' if accepted_latency['all_calls_below_5s'] else 'FAIL deadline'}",
        f"AIS attempts/applied/unapplied: {latency['calls']}/{latency['applied_calls']}/{latency['unapplied_calls']}",
        exposure,
        f"AIS full+reserve: {clearance_label}; truth min {truth_text}",
        f"Report-centered min full+reserve: {report_text}",
        f"Final route outcome: {mission_label(mission['status'])} · {len(arrivals)}/8 captures · 50 m radius",
        f"{clock(elapsed)} sailed; {clock(int(waiting)+elapsed)} since request · 5 s control",
        "Boat icon enlarged; AIS domains apply to boat center",
    )
    span = max(camera[1] - camera[0], camera[3] - camera[2])
    short_stop=mission['status']!='completed' and elapsed<=600
    view = replace(replay.view, fixed_local_view=True, local_frame=False, boat_scale=4.0 if short_stop else 2.0)
    margin_file = study / "independent_traffic_history" / f"{run_id}.csv"
    margins = {}
    if margin_file.exists():
        with margin_file.open(newline="") as stream:
            for row in csv.DictReader(stream):
                step = int(row["step"]) + 1
                margins[step] = min(margins.get(step, np.inf), float(row["truth_with_reserve_m"]))
    if len(margins) != mission["steps"]:
        raise ValueError(f"{run_id}: missing independently audited applied-interval margins")
    visual = replace(
        replay, view=view, method_lines=labels,
        frames=tuple(displayed_frame(replay.frames[step], step, event.get(step), margins.get(step), metadata["method"]) for step in indices),
        grids=tuple(replay.grids[step] for step in indices),
    )
    origin = replay.frames[0].state
    source_positions = np.column_stack(local_east_north_metres(
        [frame.state.latitude for frame in replay.frames], [frame.state.longitude for frame in replay.frames],
        origin.latitude, origin.longitude,
    ))
    reported_positions = np.zeros((count, 3, 2))
    full_radii = np.zeros((count, 3)); nominal_radii = np.zeros((count, 3))
    for step, frame in enumerate(replay.frames):
        traffic = AISPrediction.from_observation(frame.observation, acceleration_bound_mps2=.026,
                                                clearance_reserve_metres=3.)
        lat, lon = offset_latitude_longitude(traffic.origin_latitude, traffic.origin_longitude,
                                           *traffic.positions_metres.T)
        reported_positions[step] = np.column_stack(local_east_north_metres(lat, lon, origin.latitude, origin.longitude))
        full_radii[step] = traffic.clearance_radii_metres
        nominal_radii[step] = traffic.hull_radii_metres + 25.
    truth = Truth(Path(metadata["traffic_fixture"]), Path(metadata["truth_trace"]))
    if abs(origin.latitude - truth.origin[0]) > 1e-10 or abs(origin.longitude - truth.origin[1]) > 1e-10:
        raise ValueError(f"{run_id}: traffic truth and boat origin differ")
    encounter = np.datetime64("2026-09-30T23:50:40")
    seconds = np.array([(frame.observation.time - encounter) / np.timedelta64(1, "s") for frame in replay.frames])
    truth_positions = truth.sample(seconds, (1, 2))
    aim_positions = np.full((count, 2), np.nan)
    aim_radii = np.full(count, np.nan)
    for step, frame in enumerate(replay.frames):
        nav = frame.navigation
        offset = nav.get("forecast_gate_aim_offset_m", 0.0)
        if offset <= 0.0:
            continue
        radius = float(nav["forecast_gate_aim_radius_m"])
        true_radius = float(nav["forecast_gate_true_radius_m"])
        if (nav.get("phase") != "return" or true_radius != CAPTURE_RADIUS
                or radius <= 0.0 or abs(radius + offset - true_radius) > 1e-8):
            raise ValueError(f"{run_id}: nominal gate is not contained in the original 50 m capture gate")
        aim_positions[step] = local_east_north_metres(
            nav["forecast_gate_aim_latitude"], nav["forecast_gate_aim_longitude"],
            origin.latitude, origin.longitude,
        )
        aim_radii[step] = radius
    video = output / "navigation.mp4"
    receipt = output / "video_verification.json"
    config = {
        "run_id": run_id, "source_directory": str(directory), "output_directory": str(output),
        "labels": list(labels), "mission_status": mission["status"],
        "arrivals": arrivals, "simulated_elapsed_s": elapsed, "camera_en_m": camera,
        "requested_departure_utc":str(DEPARTURE), "actual_departure_utc":actual_utc,
        "gate_wait_seconds":waiting,"total_requested_to_final_seconds":waiting+elapsed,
        "departure_gate_summary_sha256":reference.digest(gate_path),
        "departure_gate_log_sha256":gate["log_sha256"],
        "launch_summary_frames":launch_summary_frames,"launch_summary_text":gate_text,
        "maximum_prelaunch_snapshot_report_age_s":prelaunch_age,"release_held_report_age_s":release_age,
        "launch_summary_scope":"Four-second summary from recorded gate log over repeated actual launch state. No prelaunch ownship motion is depicted.",
        "maximum_gate_decision_s":gate["maximum_gate_decision_s"],
        "traffic_epoch_utc":gate["traffic_epoch_utc"],
        "prelaunch_wait_scope":"Sheltered monitoring outside modeled open water; no physical station keeping or prelaunch open-water clearance claim.",
        "fps": fps, "source_frame_indices": indices, "source_frame_count": count,
        "camera_group":"shared short-stop encounter view" if short_stop else "shared long full-track view",
        "display_boat_scale":view.boat_scale,
        "minimum_short_playback_seconds":12,
        "additional_short_final_hold_seconds":2,
        "short_playback_policy":"Repeat actual source indices; every recorded short state retained, plus at least2s final-state hold. No invented states or motion.",
        "maximum_applied_held_report_age_s":held_age,
        "longest_fault_peak_experienced":independent.get('longest_fault_peak_experienced'),
        "replay_sha256": reference.digest(replay_path), "gpx_sha256": reference.digest(directory / "track.gpx"),
        "render_script_sha256": reference.digest(Path(__file__)),
        "nominal_gate_overlay_frames": int(np.isfinite(aim_radii).sum()),
        "nominal_gate_display": "Orange plus and dotted nominal planning disk; blue disks remain true 50 m capture gates.",
        "AIS_display": "Age-projected reports; red full uncertainty plus3mreserve; gray nominal25m; purple true centers.",
        "AIS_recovery_display": "Recovery forecast margin uses the received-report preview age budget plus configured 3m reserve. Recovery horizon/turn/margin are recorded at command selection, at the start of the interval ending at the displayed state; step margins independently audit that applied interval.",
        "AIS_choose_latency": latency,
        "AIS_applied_decision_latency": accepted_latency,
        "AIS_applied_decision_timer_scope": accepted_latency["scope"],
        "retest_config_sha256":metadata["retest_config_sha256"],
        "controller_mode":metadata["controller_mode"],
        "AIS_choose_timer_scope": "Every try/finally logged AISBufferPlanner.choose attempt, including lazy setup/search/dense validation and any final unapplied stop or exception. Constructor-inclusive fresh-process benchmarks are separate; no universal hard-real-time claim.",
        "traffic_fixture_sha256":metadata['traffic_fixture_sha256'],
        "truth_trace_sha256":metadata['traffic_truth_trace_sha256'],
        "scenario_id":metadata['scenario_id'],
        "reporting":metadata['reporting'],
        "astern_display": "Bow glyph follows recorded heading; signed water surge below -0.05m/s displays Water motion: STERNWAY, with AIS recovery suffix only when that recovery is active. COG remains a separate measured ground-track arrow.",
        "COLREG_recovery_display": "Recovery preview turn is the first30s verified predicted ground displacement relative to the nominal course. Soft override records a clearance-first exception to advisory starboard preference; it is not a compliance certificate. AIS choose wall s is measured whole-decision computation time, distinct from the simulated control interval.",
        "independent_truth_margins": bool(margins),
        "shared_arbitration_anchor_steps": anchors,
        "weather_display": "Visual only: AIS priority if buffer/predictive fallback/COLREG/recorded suspension, with Transformer weather source shown suppressed. Classical weather outside AIS priority derives hold/sail/planner fallback from recorded guidance. Raw replay unchanged.",
    }
    if preview_directory is not None:
        preview_directory.mkdir(parents=True, exist_ok=True)
        animation = round_trip_animation(camera, source_positions, indices, list(event), aim_positions, aim_radii,
                                         reported_positions, full_radii, nominal_radii, truth_positions, gate_text, launch_summary_frames)
        panel = animation(list(visual.frames), method_lines=visual.method_lines,
                          **replay_io.asdict(visual.view))
        panel.set_route(visual.route)
        lengths = []
        for i in range(len(indices)):
            panel.update(i)
            lines = panel.readout.get_text().splitlines()
            lengths.append((len(lines), max(map(len, lines), default=0)))
        picks = {0, launch_summary_frames, len(indices) - 1, max(range(len(indices)), key=lambda i: lengths[i])}
        worst_detail=independent.get('minimum_details',{}).get('truth_with_reserve')
        worst_step=worst_detail.get('action_step') if worst_detail else None
        picks.update(indices.index(step) for step in (worst_step, *event) if step in indices)
        first_applied = next((i for i, step in enumerate(indices)
                              if replay.frames[step].navigation.get("temporal_recovery_source") in
                              {"learned", "physical release"}), None)
        if first_applied is not None:
            picks.add(first_applied)
        first_confirmed = next((i for i, step in enumerate(indices)
                                if replay.frames[step].navigation.get("temporal_recovery_source") == "physical release"), None)
        if first_confirmed is not None:
            picks.add(first_confirmed)
        for active_recovery in (False, True):
            sternway = [i for i, frame in enumerate(visual.frames)
                        if frame.state.surge_speed is not None and frame.state.surge_speed < -0.05
                        and bool(frame.navigation.get("AIS recovery active")) == active_recovery]
            if sternway:
                picks.add(min(sternway, key=lambda i: visual.frames[i].state.surge_speed))
        checks = []
        for i in sorted(picks):
            replay_io._render_replay_frame(panel, visual.grids[i], i)
            panel.figure.canvas.draw()
            renderer = panel.figure.canvas.get_renderer()
            bounds = panel.readout.get_window_extent(renderer)
            canvas = panel.figure.bbox
            legend = panel.readout.axes.get_legend().get_window_extent(renderer)
            assert bounds.x0 >= 0 and bounds.x1 <= canvas.x1 and bounds.y1 <= canvas.y1
            assert bounds.y0 >= legend.y1 + 6, (run_id, indices[i], bounds.bounds, legend.bounds)
            launch_bounds=None
            if i<launch_summary_frames:
                launch_bounds=panel.launch_summary.get_window_extent(renderer)
                axis_bounds=panel.axis.get_window_extent(renderer)
                assert (launch_bounds.x0>=axis_bounds.x0 and launch_bounds.x1<=axis_bounds.x1
                        and launch_bounds.y0>=axis_bounds.y0 and launch_bounds.y1<=axis_bounds.y1), (run_id,launch_bounds.bounds,axis_bounds.bounds)
            destination = preview_directory / (f"{run_id}_launch_summary.png" if i<launch_summary_frames
                                                 else f"{run_id}_step_{indices[i]:05d}.png")
            panel.figure.savefig(destination, dpi=panel.figure.dpi)
            checks.append({"source_step": indices[i], "video_frame_index":i,"text_bbox": list(bounds.bounds),
                           "launch_summary_visible":i<launch_summary_frames,
                           "launch_summary_bbox":list(launch_bounds.bounds) if launch_bounds is not None else None,
                           "legend_bbox": list(legend.bounds), "canvas_bbox": list(canvas.bounds),
                           "png": str(destination), "header": panel.readout.get_text().splitlines()[0]})
        replay_io.plt.close(panel.figure)
        reference.write_json(preview_directory / f"{run_id}_layout.json", checks)
        print(f"{run_id}: {len(checks)} preview/layout checks passed", flush=True)
        return checks
    if video.exists() or receipt.exists():
        if not (video.is_file() and receipt.is_file()):
            raise ValueError(f"{run_id}: existing clip or receipt is incomplete")
        original = json.loads(receipt.read_text())
        if any(original.get(key) != value for key, value in config.items()) or original["video_sha256"] != reference.digest(video):
            raise ValueError(f"{run_id}: existing video or source has changed")
    else:
        animation = round_trip_animation(camera, source_positions, indices, list(event), aim_positions, aim_radii,
                                         reported_positions, full_radii, nominal_radii, truth_positions, gate_text, launch_summary_frames)
        with (patch.object(replay_io, "SaildroneAnimation", animation),
              patch.object(replay_io, "FFMpegWriter", SingleThreadWriter)):
            replay_io.export_navigation_video(visual, video, fps=fps)
    reference.verify_decode(video, len(indices))
    record = {**config, "decoded_frames": len(indices), "video_sha256": reference.digest(video)}
    reference.write_json(receipt, record)
    print(f"{run_id}: verified {len(indices)} encoded frames from {count} replay frames", flush=True)
    return record


def write_index(study, results):
    lines = [
        "# Received-AIS departure Otranto playback", "",
        "All outcomes are retained, including partial routes and safety stops. Four opening seconds summarize the recorded gate certificate over the actual launch fix; no prelaunch in-water trajectory is invented. "
        "The common stack is Gamma + Direct acados NMPC, forecast hold/tacks and refined return guidance. "
        "The Transformer treatment adds the frozen weather hold/release model and physical confirmation. "
        "AIS intervention and nominal weather guidance are identified separately in every clip. Requested departure is 2026-09-30 23:50 UTC; each clip labels actual launch and sheltered wait separately. Original traffic and dropout UTC anchors remain fixed.", "",
        "Reports have 20 s delivery delay and two 40 s emission-drop bursts, with a 65 s planning age budget. "
        "Applied-step checks use uncapped actual report ages. The 25 m nominal gap, full uncertainty and 3 m reserve remain distinct checks. Postlaunch fault exposure is distinguished from prelaunch sheltered monitoring.", "",
        "Separate timers report complete choose work and choose-through-action-acceptance/stop work, including a final unapplied attempt. "
        "The 5 s budget is checked against the complete acceptance/stop timer. An early stop is not waypoint completion, even when applied intervals retained clearance.", "",
        "Short stopped cases share one fixed north-up camera showing the route and nearby opening traffic; "
        "long loops share a separate full-track camera. Exact bounds and enlarged boat scale are in each receipt. "
        "Clips show the entire recorded track line and true blue 50 m capture disks. "
        "Orange capture dots show actual entries. The yellow dot marks the bow and the purple arrow shows COG; "
        "signed surge labels distinguish sternway. Wind barbs show FROM and the green current arrow shows TO. "
        "Orange nominal gate aims, when present, do not change the true 50 m circle.", "",
        "Short stopped cases preserve every recorded state and last at least12s by repeating source indices, with at least2s final-state hold. "
        "No state or motion is invented. A strict-clearance stop records the absence of a tested acceptable action; it is a failed route and does not establish a physical safe-waiting policy. "
        "Raw replays retain every applied 5 s update. MP4s retain the opening encounter, COLREG and policy-source changes, "
        "capture neighborhoods and the final recorded state; only playback is downsampled.", "",
        "| Encounter | Wind FROM | Guidance | Captures | Route outcome | Sailed / sheltered wait | Max accept/stop decision | Max held age | Clearance | Files |",
        "|---|---:|---|---:|---|---:|---:|---:|---|---|",
    ]
    for result in sorted(results,key=lambda row:row['run_id']):
        run_id=result['run_id'];source=source_directory(study,run_id);output=output_directory(study,run_id)
        metadata=json.loads((source/'execution.json').read_text())
        method='Transformer + confirmation' if metadata['method']=='transformer' else 'Classical'
        outcome=mission_label(result['mission_status'])
        clearance=next(line for line in result['labels'] if line.startswith('AIS full+reserve:'))
        latency=result['AIS_applied_decision_latency']
        timing=f"{latency['maximum_seconds']:.3f}s ({'PASS' if latency['all_calls_below_5s'] else 'FAIL'})"
        relative=lambda path:Path(os.path.relpath(path,study)).as_posix()
        files=(f"[MP4]({relative(output/'navigation.mp4')}) · [Replay]({relative(source/'navigation.jsonl.gz')}) · "
               f"[GPX]({relative(source/'track.gpx')}) · [Receipt]({relative(output/'video_verification.json')}) · "
               f"[Gate summary]({relative(source/'departure_gate.json')}) · [Gate telemetry]({relative(source/'departure_gate.jsonl')})")
        held=result.get('maximum_applied_held_report_age_s')
        held_text=f'{held:g}s' if held is not None else 'not measured'
        lines.append(f"| {metadata['scenario_id']} | {metadata['wind_from_degrees']:g}° | {method} | "
                     f"{len(result['arrivals'])}/8 | {outcome} | {clock(result['simulated_elapsed_s'])} / {clock(int(result['gate_wait_seconds']))} | {timing} | {held_text} | {clearance} | {files} |")
    (study/'playback.md').write_text('\n'.join((*lines,'')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--frame-limit", type=int, default=420)
    parser.add_argument("--fps", type=float, default=12)
    parser.add_argument("--preview", type=Path, help="Check layout and save PNGs without encoding or writing video receipts.")
    parser.add_argument("--arms", nargs="+", help="Completed run directory names; defaults to every fresh completed arm.")
    parser.add_argument("--include-baselines", action="store_true", help="Render preserved controls into baseline_media without modifying their recordings.")
    args = parser.parse_args()
    if args.jobs < 1 or args.frame_limit < 26 or not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("Use positive jobs and FPS and at least 26 sampled frames")
    study = args.study.resolve()
    arms = args.arms or [folder.name for folder in sorted((study / "runs").iterdir())
                         if folder.is_dir() and (folder / "execution.json").is_file()]
    if args.include_baselines:
        arms = list(dict.fromkeys([*baseline_manifest(study), *arms]))
    if not arms:
        parser.error("No completed runs selected.")
    missions={run_id:reference.recorded_mission(source_directory(study,run_id),0) for run_id in arms}
    short=[name for name in arms if missions[name]['status']!='completed' and missions[name]['steps']*UPDATE_SECONDS<=600]
    long=[name for name in arms if name not in short]
    cameras={}
    with patch.object(reference,"run_folder",lambda root,name:source_directory(root,name)):
        if short:
            camera=short_event_camera(study,short)
            cameras.update({name:camera for name in short})
        if long:
            camera=reference.camera_for(study,tuple(long))
            cameras.update({name:camera for name in long})
    anchors=shared_anchor_steps(study,arms)
    tasks=[(study,run_id,cameras[run_id],args.fps,args.frame_limit,anchors) for run_id in arms]
    if args.preview is not None:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(render,tasks,[args.preview.resolve()]*len(tasks)))
        return
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(render, tasks))
    write_index(study, results)


if __name__ == "__main__":
    main()
