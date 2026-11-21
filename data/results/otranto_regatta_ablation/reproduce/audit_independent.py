"""Independently sweep every applied AIS-loop command at the plant's 0.05 s step.

Traffic truth comes from the preserved analytical fixture trace, rather than
SyntheticAIS/AISPrediction. Target positions after the recorded trace are
extended only after checking its explicitly constant post-ramp velocity.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from hashlib import sha256
import json
import gzip
from math import ceil, hypot
from pathlib import Path
import sys
import zlib

import numpy as np


STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[2]
sys.path.insert(0, str(ROOT))

from config import SAILDRONE_GEOMETRY_AND_HYDROSTATICS as OWN
from world._geo import haversine_metres, local_east_north_metres
from world.boat import BoatAction
from world.fossen import Fossen3DOFSaildroneDynamics
from world.colreg import ColregPolicy
from world.navigation_replay import (load_navigation_replay, NavigationReplay, ReplayView, RouteWaypoint, _decode_episode_frame)


FIXTURE = STUDY / "fixtures/crossing_fan"
TURN_BOUND = 0.026
RESERVE = 3.0
LEAD = 40.0
SAFETY_BUFFER = 25.0
INITIAL_UNCERTAINTY = 5.0
GROWTH = 0.05
DT = 5.0
START = np.datetime64("2026-09-30T23:50:00", "ns")
TRAFFIC_EPOCH = np.datetime64("2026-09-30T23:50:40", "ns")
EXPECTED = [("outbound", name) for name in ("002", "003", "004", "005")] + [
    ("return", name) for name in ("004", "003", "002", "001")
]


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Truth:
    def __init__(self, fixture, trace):
        configuration = json.loads(fixture.read_text())
        self.origin = configuration["origin_latitude_longitude"]
        targets = configuration["scenarios"]["mixed"]
        self.identities = [target["mmsi"] for target in targets]
        own_radius = 0.5 * hypot(OWN["hull_length_overall_m"], OWN["hull_beam_max_m"])
        self.hulls = np.array([
            own_radius + 0.5 * hypot(target["length_metres"], target["beam_metres"])
            for target in targets
        ])
        records = {identifier: [] for identifier in self.identities}
        fields = ("elapsed_s", "east_m", "north_m", "east_velocity_mps", "north_velocity_mps")
        with trace.open() as stream:
            for row in csv.DictReader(stream):
                identifier = int(row["mmsi"])
                if identifier in records:
                    records[identifier].append([float(row[field]) for field in fields])
        self.tracks = [np.array(records[identifier]) for identifier in self.identities]
        self.ramp_end = max(float(ramp["end_s"]) for ramp in configuration["velocity_ramps"])
        for track in self.tracks:
            intervals = np.diff(track[:, 0])
            assert np.all(intervals > 0)
            acceleration = np.linalg.norm(np.diff(track[:, 3:], axis=0), axis=1) / intervals
            assert acceleration.max() < TURN_BOUND
            tail = track[track[:, 0] >= self.ramp_end]
            assert len(tail) > 1
            np.testing.assert_allclose(tail[:, 3:], np.broadcast_to(tail[-1, 3:], tail[:, 3:].shape), atol=1e-10, rtol=0)
            np.testing.assert_allclose(tail[:, 1:3], tail[-1, 1:3] + (tail[:, 0] - tail[-1, 0])[:, None] * tail[-1, 3:], atol=1e-6, rtol=0)
        self.trace_end = min(float(track[-1, 0]) for track in self.tracks)

    def sample(self, times, columns):
        times = np.asarray(times)
        result = []
        for track in self.tracks:
            assert times.min() >= track[0, 0], "Truth trace does not cover the requested report generation."
            values = [np.interp(times, track[:, 0], track[:, column]) for column in columns]
            if columns == (1, 2):
                after = np.maximum(times - track[-1, 0], 0.0)
                values = [value + after * velocity for value, velocity in zip(values, track[-1, 3:])]
            result.append(np.column_stack(values))
        return np.stack(result, axis=1)


class MergedTruth:
    """Physical original fleet plus independently acquired underway contact."""
    def __init__(self, fixture, trace, underway_fixture, underway_trace):
        self.original = Truth(fixture, trace)
        self.underway = Truth(underway_fixture, underway_trace)
        np.testing.assert_allclose(self.original.origin, self.underway.origin, atol=1e-12, rtol=0)
        self.origin = self.original.origin
        self.identities = self.original.identities + self.underway.identities
        assert len(self.identities) == len(set(self.identities)) == 4
        self.hulls = np.r_[self.original.hulls, self.underway.hulls]
        self.trace_end = min(self.original.trace_end, self.underway.trace_end)
        self.first_emission = float(json.loads(underway_fixture.read_text())["first_emission_s"])
        assert self.first_emission == 400.0
        self.first_delivery = self.first_emission + 20.0

    def sample(self, times, columns):
        return np.concatenate((self.original.sample(times, columns), self.underway.sample(times, columns)), axis=1)

    def visible_columns(self, encounter_seconds):
        return np.arange(4 if encounter_seconds >= self.first_delivery else 3)


def swept_margin(relative, radius, dt):
    """Exact minimum of norm(relative chord) minus its linear radius chord."""
    start, delta = relative[:-1], np.diff(relative, axis=0)
    speed = np.linalg.norm(delta, axis=-1)
    norm = np.maximum(speed, 1e-12)
    along = np.sum(start * delta, axis=-1) / norm
    across = np.sqrt(np.maximum(np.sum(start * start, axis=-1) - along**2, 0.0))
    radius_delta = np.diff(radius, axis=0)
    slope = radius_delta / norm
    clamped = np.clip(slope, -1 + 1e-12, 1 - 1e-12)
    desired = clamped * across / np.sqrt(1 - clamped**2)
    at = np.clip((desired - along) / norm, 0, 1)
    at = np.where(slope >= 1, 1.0, np.where(slope <= -1, 0.0, at))
    at = np.where(speed > 1e-12, at, np.where(radius_delta > 0, 1.0, 0.0))
    margins = np.linalg.norm(start + at[..., None] * delta, axis=-1) - radius[:-1] - at * radius_delta
    index = np.argmin(margins, axis=0)
    columns = np.arange(relative.shape[1])
    return margins[index, columns], (index + at[index, columns]) * dt


def expected_report_age(encounter_seconds):
    generation = np.floor((encounter_seconds - 20.0) / DT) * DT
    while any(start <= generation < end for start, end in ((85.0, 125.0), (145.0, 185.0), (645.0, 685.0), (705.0, 745.0))):
        generation -= DT
    return encounter_seconds - generation


def audit_attempt_timings(run, replay, prefix_seconds=None, log_path=None):
    """Crosscheck both full attempts against every saved applied-action timer."""
    log_path = Path(log_path) if log_path is not None else run / "run.log"
    text = log_path.read_text()
    complete = text[:text.rfind("\n") + 1]
    applied = max(0, len(replay.frames) - 1)
    result = {}
    streams = (
        ("AIS decision timing: ", "choose", "ais_maneuver_search_seconds", "independent_decision_timing"),
        ("AIS applied decision timing: ", "applied_decision", "ais_applied_decision_seconds", "independent_applied_decision_timing"),
    )
    measured = {}
    for prefix, label, replay_key, folder_name in streams:
        rows = [json.loads(line.removeprefix(prefix)) for line in complete.splitlines() if line.startswith(prefix)]
        assert all(np.isfinite(row["wall_seconds"]) and row["wall_seconds"] >= 0 for row in rows), run.name
        assert [row["step"] for row in rows] == list(range(1, len(rows) + 1)), (run.name, label, "Missing or duplicate timing attempt")
        if prefix_seconds:
            rows = [row for row in rows if row["step"] <= applied]
        assert len(rows) >= applied, (run.name, label, "Applied controls missing whole-decision timing")
        if not prefix_seconds:
            assert len(rows) <= applied + 1, (run.name, label, "Multiple unapplied attempts")
        for row, frame in zip(rows[:applied], replay.frames[1:]):
            assert row["step"] == frame.navigation["step"]
            assert abs(row["wall_seconds"] - frame.navigation[replay_key]) < 1e-12, (run.name, label, row["step"])
            if label == "applied_decision":
                assert row["applied"] is True, (run.name, row["step"], "A saved applied action is flagged unapplied")
        values = [row["wall_seconds"] for row in rows]
        folder = run.parent.parent / (folder_name + "_checkpoint" if prefix_seconds else folder_name)
        folder.mkdir(exist_ok=True)
        timing_path = folder / f"{run.name}.csv"
        fields = list(dict.fromkeys([key for row in rows for key in row] + ["replay_applied"]))
        with timing_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows({**row, "replay_applied": row["step"] <= applied} for row in rows)
        stem = "AIS_" + label
        result.update({
            stem + "_attempts": len(rows),
            stem + "_applied_attempts": applied,
            stem + "_unapplied_final_attempts": max(0, len(rows) - applied),
            "maximum_complete_" + stem + "_s": max(values, default=None),
            stem + "_attempts_at_or_above_5s": sum(value >= 5 for value in values),
            "all_" + stem + "_attempts_below_5s": bool(values) and all(value < 5 for value in values),
            stem + "_timing_measurement_log": str(log_path),
            stem + "_timing_measurement_log_sha256": digest(log_path),
            stem + "_attempt_timing_csv": str(timing_path),
            stem + "_attempt_timing_csv_sha256": digest(timing_path),
            stem + "_overrun_or_unapplied_timing_records": [row for row in rows if row["wall_seconds"] >= 5 or row["step"] > applied],
        })
        measured[label] = rows
    choose, total = measured["choose"], measured["applied_decision"]
    assert len(choose) == len(total), (run.name, "Every choose requires a complete AIS-selection timer, including terminal attempts")
    for first, second in zip(choose, total):
        for key in ("step", "phase", "waypoint", "observation_utc", "nominal_course_deg", "prefer_starboard"):
            assert first[key] == second[key], (run.name, key, "Timing streams refer to different decisions")
        assert second["wall_seconds"] >= first["wall_seconds"], (run.name, first["step"], "Complete selection timer excludes part of choose")
    result["AIS_attempt_timing_scope"] = (
        "Every complete AISBufferPlanner.choose and entire AIS selection-through-native-assessment attempt, "
        "including the final unapplied stop or exception. Both streams are matched to all applied replay timers; "
        "the applied-decision scope also includes controller/actuator selection and any fallback."
    )
    return result


def velocity(course, speed):
    angle = np.deg2rad(course)
    return np.column_stack((np.sin(angle), np.cos(angle))) * np.asarray(speed)[:, None]


def captures(run, replay):
    prefix = "Mission outcome: "
    mission, = [json.loads(line.removeprefix(prefix)) for line in (run / "run.log").read_text().splitlines() if line.startswith(prefix)]
    assert mission["steps"] == len(replay.frames) - 1
    assert mission["arrival_radius_metres"] == 50.0 and mission["return_to_start"]
    logged = mission["waypoint_arrivals"]
    assert [(item["phase"], item["waypoint"]) for item in logged] == EXPECTED[:len(logged)]
    points = {point.name: point for point in replay.route}
    found = []
    previous = 0
    for phase, name in EXPECTED:
        point = points[name]
        first = next((index for index in range(previous, len(replay.frames)) if haversine_metres(
            replay.frames[index].state.latitude, replay.frames[index].state.longitude,
            point.latitude, point.longitude,
        ) <= 50.0), None)
        if first is None:
            break
        state = replay.frames[first].state
        found.append({"phase": phase, "waypoint": name, "step": first,
                      "distance_metres": haversine_metres(state.latitude, state.longitude, point.latitude, point.longitude)})
        previous = first
    assert [(row["phase"], row["waypoint"], row["step"]) for row in found] == [
        (row["phase"], row["waypoint"], row["step"]) for row in logged
    ], "First physical 50 m entries disagree with logged ordered captures."
    assert (mission["status"] == "completed") == (len(found) == len(EXPECTED))
    for physical, record in zip(found, logged):
        assert abs(physical["distance_metres"] - record["distance_metres"]) < 1e-6
    return mission, found


def load_live_prefix(path, seconds):
    raw = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(path.read_bytes())
    lines = raw[:raw.rfind(b"\n")].decode().splitlines()
    header = json.loads(lines[0])
    rows = [json.loads(line) for line in lines[1:int(seconds // DT) + 2]]
    assert len(rows) == int(seconds // DT) + 1, "The requested prefix is not flushed yet."
    frames = tuple(_decode_episode_frame(row) for row in rows)
    return NavigationReplay(tuple(RouteWaypoint(**p) for p in header["route"]), header["route_name"],
                            ReplayView(**header["view"]), frames, (None,) * len(frames),
                            tuple(header.get("method_lines", [])))


def load_complete_allow_empty(path):
    try:return load_navigation_replay(path)
    except ValueError as error:
        if "contains no frames" not in str(error):raise
        with gzip.open(path,"rt") as stream:header=json.loads(next(stream))
        return NavigationReplay(tuple(RouteWaypoint(**p) for p in header["route"]),header.get("route_name"),
            ReplayView(**header["view"]),(),(),tuple(header.get("method_lines",[])))


def no_applied_actions_result(run,replay,experiment,fixture,trace,history_dir,attempt_timing,gate_summary):
    history=history_dir/f"{run.name}.csv"
    history.write_text("step,scenario_seconds,mmsi,report_age_s,true_hull_m,true_nominal_m,truth_full_m,truth_with_reserve_m,projected_full_m,projected_with_reserve_m,max_report_extrapolation_error_m\n")
    lines=[json.loads(line.removeprefix("Mission outcome: ")) for line in (run/"run.log").read_text().splitlines() if line.startswith("Mission outcome: ")]
    mission=lines[0] if lines else {"status":"process_exception","waypoint_arrivals":[],"waypoint_distance_metres":None}
    assert not mission["waypoint_arrivals"]
    initial_clearance=None
    if replay.frames:
        truth=MergedTruth(fixture,trace,Path(experiment["underway_traffic_fixture"]),Path(experiment["underway_truth_trace_csv"]));frame=replay.frames[0];time=float((np.datetime64(gate_summary["actual_departure_utc"].removesuffix("Z"), "ns")-TRAFFIC_EPOCH)/np.timedelta64(1,"s"))
        own=np.array(local_east_north_metres(frame.state.latitude,frame.state.longitude,*truth.origin))
        targets=truth.sample([time],(1,2))[0]
        report_ages={target.mmsi:target.report_age_seconds for target in frame.observation.ais_targets}
        age=np.array([report_ages.get(identifier,65.0) for identifier in truth.identities])
        uncertainty=INITIAL_UNCERTAINTY+np.maximum(GROWTH*age,.5*TURN_BOUND*age**2)
        margins=np.linalg.norm(targets-own,axis=1)-truth.hulls-SAFETY_BUFFER-uncertainty-RESERVE
        initial_clearance=float(margins.min())
    result={"case":run.name,"encounter_id":experiment["encounter_id"],"scope":"no-applied-actions",**attempt_timing,
       "wind_from_degrees":experiment["wind_from_degrees"],"departure_utc":gate_summary["actual_departure_utc"],
       "monitoring_start_utc":gate_summary["monitoring_start_utc"],"traffic_epoch_utc":gate_summary["traffic_epoch_utc"],"gate_wait_seconds":gate_summary["gate_wait_seconds"],
       "applied_actions":0,"frames":len(replay.frames),"integration_substep_seconds":.05,
       "maximum_pose_reconstruction_error_m":None,"maximum_heading_reconstruction_error_deg":None,
       "maximum_runtime_margin_difference_m":None,"maximum_truth_outside_report_envelope_m":None,
       "initial_truth_full_plus_reserve_m":initial_clearance,"maximum_applied_held_report_age_s":None,
       "longest_fault_peak_experienced":False,"distinct_report_ages_s":[],"negative_action_counts":{},"ais_modes":{},
       "mission_status":mission["status"],"strict_clearance_stop":mission["status"]=="no_full_clearance_action","waypoint_arrivals":[],"goal_captures":[],"final_goal_distance_m":mission["waypoint_distance_metres"],
       "replay_sha256":digest(run/"navigation.jsonl.gz"),"run_log_sha256":digest(run/"run.log"),"history_sha256":digest(history),"audit_source_sha256":digest(__file__),
       "truth_fixture_sha256":digest(fixture),"truth_trace_sha256":digest(trace),
       "departure_gate_summary_sha256":digest(run/"departure_gate.json"),"departure_gate_log_sha256":digest(run/"departure_gate.jsonl"),
       "measurement":"No command was applied; interval minima/reconstruction errors are unmeasured, not zero. Initial clearance is reported separately when an initial frame exists."}
    for key in ("truth_minimum_hull_m","truth_minimum_nominal_m","truth_centered_minimum_uncertainty_m","configured_truth_centered_minimum_m","model_minimum_uncertainty_m","configured_model_minimum_m"):result[key]=None
    result["minimum_details"]={}
    extra=Path(experiment["underway_traffic_fixture"]);extra_trace=Path(experiment["underway_truth_trace_csv"])
    target=json.loads(extra.read_text())["scenarios"]["mixed"][0]["mmsi"]
    result.update(underway_truth_fixture_sha256=digest(extra),underway_truth_trace_sha256=digest(extra_trace),
        underway_target_mmsi=target,underway_first_acquisition_encounter_elapsed_s=None,
        underway_report_age_peak_intervals=[],underway_both65s_age_peaks_verified=False,
        underway_modes_420_to_900_s={},underway_active_ais_or_colreg_steps_420_to_900_s=0,
        maximum_first300s_launch_tube_excess_m=None,observed_launch_displacement_tube_obeyed_first300s=False,
        launch_tube_coverage_seconds=0.0)
    return result


def freeze_checkpoint_inputs(run,seconds,study):
    source=run/"navigation.jsonl.gz"
    raw=zlib.decompressobj(16+zlib.MAX_WBITS).decompress(source.read_bytes())
    lines=raw[:raw.rfind(b"\n")].splitlines()
    count=int(seconds//DT)+1
    assert len(lines)>=count+1,"The requested live prefix is not flushed yet."
    content=b"\n".join(lines[:count+1])+b"\n"
    folder=study/"independent_checkpoint_inputs"/run.name
    folder.mkdir(parents=True,exist_ok=True)
    recording=folder/f"navigation_{seconds:g}s.jsonl.gz"
    if recording.exists():assert gzip.decompress(recording.read_bytes())==content
    else:recording.write_bytes(gzip.compress(content,mtime=0))
    text=(run/"run.log").read_text();complete=text[:text.rfind("\n")+1]
    prefixes = ("AIS decision timing: ", "AIS applied decision timing: ")
    selected = []
    for prefix in prefixes:
        rows = [json.loads(line.removeprefix(prefix)) for line in complete.splitlines() if line.startswith(prefix)]
        rows = [row for row in rows if row["step"] <= count - 1]
        assert len(rows) == count - 1, (prefix, "A checkpoint action is missing a complete timing log.")
        selected.extend(prefix + json.dumps(row) + "\n" for row in rows)
    log=folder/f"decision_timing_{seconds:g}s.jsonl"
    log_content="".join(selected)
    if log.exists():assert log.read_text()==log_content
    else:log.write_text(log_content)
    experiment=folder/"experiment.json"
    if experiment.exists():assert experiment.read_bytes()==(run/"experiment.json").read_bytes()
    else:experiment.write_bytes((run/"experiment.json").read_bytes())
    return recording,log


def audit(task):
    run, fixture, trace, history_dir = map(Path, task[:4])
    prefix_seconds = task[4] if len(task) > 4 else None
    experiment = json.loads((run / "experiment.json").read_text())
    underway_fixture = Path(experiment["underway_traffic_fixture"])
    underway_trace = Path(experiment["underway_truth_trace_csv"])
    truth = MergedTruth(fixture, trace, underway_fixture, underway_trace)
    replay_path = run / "navigation.jsonl.gz"
    requested_prefix_seconds=prefix_seconds
    closed_short=False
    if prefix_seconds and (run/"execution.json").exists():
        lines=(run/"run.log").read_text().splitlines()
        outcomes=[json.loads(line.removeprefix("Mission outcome: ")) for line in lines if line.startswith("Mission outcome: ")]
        closed_short=bool(outcomes and outcomes[0]["steps"]*DT<=prefix_seconds)
        if not outcomes:closed_short=True  # Closed exceptional process: all saved actions must remain auditable.
    if closed_short:prefix_seconds=None
    recording_path,log_path=(freeze_checkpoint_inputs(run,prefix_seconds,history_dir.parent) if prefix_seconds else (replay_path,run/"run.log"))
    replay = load_complete_allow_empty(recording_path)
    experiment = json.loads((run / "experiment.json").read_text())
    gate_summary = json.loads((run / "departure_gate.json").read_text())
    launch_time = np.datetime64(gate_summary["actual_departure_utc"].removesuffix("Z"), "ns")
    launch_encounter_seconds = float((launch_time - TRAFFIC_EPOCH) / np.timedelta64(1, "s"))
    assert np.datetime64(gate_summary["monitoring_start_utc"].removesuffix("Z"), "ns") == START
    assert np.datetime64(gate_summary["traffic_epoch_utc"].removesuffix("Z"), "ns") == TRAFFIC_EPOCH
    assert launch_time >= START
    assert experiment["encounter_id"] == fixture.parent.name
    attempt_timing=audit_attempt_timings(run,replay,prefix_seconds,log_path)
    assert np.datetime64(experiment["departure_utc"].removesuffix("Z"), "ns") == START
    wind_from = float(experiment["wind_from_degrees"])
    assert wind_from in (355.0, 5.0)
    if len(replay.frames)<=1:
        result=no_applied_actions_result(run,replay,experiment,fixture,trace,history_dir,attempt_timing,gate_summary)
        result.update(requested_checkpoint_seconds=requested_prefix_seconds,closed_before_or_at_checkpoint=closed_short)
        return result
    initial = replay.frames[0]
    assert initial.observation.time == launch_time
    np.testing.assert_allclose((initial.state.latitude, initial.state.longitude), truth.origin, atol=1e-9, rtol=0)
    if prefix_seconds:
        mission = {"status": "live-prefix-only", "waypoint_arrivals": [], "waypoint_distance_metres": None}
        arrivals = []
    else:
        mission, arrivals = captures(run, replay)
    plant = Fossen3DOFSaildroneDynamics()
    n = ceil(DT / plant.parameters.max_integration_step)
    substep = DT / n
    assert substep <= 0.05
    offsets = np.arange(n + 1) * substep
    worst, negative, modes, ages = {}, Counter(), Counter(), set()
    maximum_endpoint = maximum_heading = maximum_runtime_margin_error = 0.0
    maximum_coverage = -np.inf
    maximum_held_report_age = 0.0
    first_underway_acquisition = None
    underway_peak_intervals = []
    regatta_search_cost_samples = []
    regatta_guidance_frames = Counter()
    underway_modes = Counter()
    underway_active_ais_or_colreg_steps = 0
    maximum_launch_tube_excess = -np.inf
    maximum_first300s_native_ground_speed = 0.0
    minimum_native_surge = np.inf
    astern_action_count = astern_boundary_count = 0
    maximum_colreg_velocity_error = maximum_cog_error = 0.0
    colreg_fossen_velocity_checks = colreg_legacy_velocity_checks = 0
    output_path = history_dir / f"{run.name}.csv"
    fields = ("step", "scenario_seconds", "mmsi", "report_age_s", "reported_visible", "allowance_age_s", "truth_allowance_scope", "true_hull_m", "true_nominal_m", "truth_full_m",
              "truth_with_reserve_m", "projected_full_m", "projected_with_reserve_m", "max_report_extrapolation_error_m")
    with output_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fields, lineterminator="\n")
        writer.writeheader()
        for k, (before, after) in enumerate(zip(replay.frames, replay.frames[1:])):
            observation, navigation = before.observation, after.navigation
            race_cost = navigation.get("race_forecast_search_wall_s")
            if race_cost is not None:
                assert np.isfinite(race_cost) and race_cost >= 0.0
                regatta_search_cost_samples.append(float(race_cost))
            if navigation.get("race_guidance") is not None:
                regatta_guidance_frames[navigation["race_guidance"]] += 1
            assert (after.observation.time - observation.time) / np.timedelta64(1, "s") == DT
            assert navigation["step"] == k + 1
            assert observation.wind and observation.current and observation.wave
            assert abs((observation.wind.direction - wind_from + 180) % 360 - 180) < 1e-8
            action = BoatAction(navigation["rudder_command_deg"], navigation["flap_command_deg"])
            states = [before.state]
            for _ in range(n):
                states.append(plant.step(states[-1], action, observation, substep))
            end = states[-1]
            water_surges = [state.surge_speed for state in states if state.surge_speed is not None]
            if water_surges:
                minimum_native_surge = min(minimum_native_surge, min(water_surges))
                astern_action_count += min(water_surges) < -1e-6
            boundary = before.state
            if boundary.surge_speed is not None:
                colreg_fossen_velocity_checks += 1
                astern_boundary_count += boundary.surge_speed < -1e-6
                heading = np.deg2rad(boundary.heading)
                u, v = boundary.surge_speed, boundary.sway_speed
                analytic = np.array((np.sin(heading) * u + np.cos(heading) * v,
                                     np.cos(heading) * u - np.sin(heading) * v))
                analytic += (observation.current.u, observation.current.v)
                colreg_velocity = np.asarray(ColregPolicy._own_ground_velocity(boundary, observation))
                velocity_error = float(np.linalg.norm(analytic - colreg_velocity))
                maximum_colreg_velocity_error = max(maximum_colreg_velocity_error, velocity_error)
                assert velocity_error < 1e-12
            else:
                colreg_legacy_velocity_checks += 1
                # Documented legacy fallback receives SOG, so it must not add
                # current a second time. The launch frame uses this branch.
                angle = np.deg2rad(boundary.heading)
                legacy = boundary.speed * np.array((np.sin(angle), np.cos(angle)))
                error = float(np.linalg.norm(legacy - np.asarray(ColregPolicy._own_ground_velocity(boundary, observation))))
                maximum_colreg_velocity_error = max(maximum_colreg_velocity_error, error)
                assert error < 1e-12
            # Independent compass bearing from successive recorded positions.
            dlon = (after.state.longitude - before.state.longitude + 180.0) % 360.0 - 180.0
            east = dlon * 111320.0 * np.cos(np.deg2rad(before.state.latitude))
            north = (after.state.latitude - before.state.latitude) * 111320.0
            physical_cog = after.state.heading if np.hypot(east, north) < 1e-6 else np.rad2deg(np.arctan2(east, north)) % 360.0
            if navigation.get("course_over_ground_deg") is not None:
                cog_error = abs((navigation["course_over_ground_deg"] - physical_cog + 180.0) % 360.0 - 180.0)
                maximum_cog_error = max(maximum_cog_error, cog_error)
                assert cog_error < 1e-7
            endpoint = hypot(*local_east_north_metres(end.latitude, end.longitude, after.state.latitude, after.state.longitude))
            heading_error = abs((end.heading - after.state.heading + 180) % 360 - 180)
            assert endpoint < 0.001 and heading_error < 1e-7, (run.name, k, endpoint, heading_error)
            maximum_endpoint, maximum_heading = max(maximum_endpoint, endpoint), max(maximum_heading, heading_error)
            own = np.column_stack(local_east_north_metres(
                [state.latitude for state in states], [state.longitude for state in states], *truth.origin,
            ))
            mission_times = k * DT + offsets
            in_startup = mission_times <= 300.0 + 1e-9
            if in_startup.any():
                tube = gate_summary["parameters"]["startup_room_m"] + gate_summary["parameters"]["startup_speed_bound_mps"] * mission_times[in_startup]
                maximum_launch_tube_excess = max(maximum_launch_tube_excess, float((np.linalg.norm(own[in_startup], axis=1) - tube).max()))
                intervals = mission_times[1:] <= 300.0 + 1e-9
                if intervals.any():
                    maximum_first300s_native_ground_speed = max(maximum_first300s_native_ground_speed, float((np.linalg.norm(np.diff(own, axis=0), axis=1)[intervals] / substep).max()))
            elapsed = k * DT + launch_encounter_seconds
            ships = truth.sample(elapsed + offsets, (1, 2))
            reports_by_id = {report.mmsi: report for report in observation.ais_targets}
            visible = truth.visible_columns(elapsed)
            visible_ids = [truth.identities[column] for column in visible]
            assert set(reports_by_id) == set(visible_ids), (run.name, k, set(reports_by_id), visible_ids)
            if len(visible) == 4 and first_underway_acquisition is None:
                first_underway_acquisition = elapsed
            reports = [reports_by_id[identifier] for identifier in visible_ids]
            age = np.array([report.report_age_seconds for report in reports])
            np.testing.assert_allclose(age, expected_report_age(elapsed), atol=1e-8, rtol=0)
            assert age.max() <= 150.0
            if navigation.get("ais_step_report_age_s") is not None:
                assert abs(navigation["ais_step_report_age_s"] - max(age)) < 1e-8
            ages.update(age.tolist())
            reported_positions = np.column_stack(local_east_north_metres(
                [report.latitude for report in reports], [report.longitude for report in reports], *truth.origin,
            ))
            reported_velocity = velocity([report.course for report in reports], [report.speed for report in reports])
            for observed_column, physical_column in enumerate(visible):
                generation = elapsed - age[observed_column]
                assert physical_column < 3 or generation >= truth.first_emission
                np.testing.assert_allclose(reported_positions[observed_column], truth.sample([generation], (1, 2))[0, physical_column], atol=0.001, rtol=0)
                np.testing.assert_allclose(reported_velocity[observed_column], truth.sample([generation], (3, 4))[0, physical_column], atol=1e-8, rtol=0)
            scale = np.array([np.cos(np.deg2rad(observation.boat.latitude)) / np.cos(np.deg2rad(truth.origin[0])), 1.0])
            actual = (ships - own[:, None, :]) * scale
            projected = (reported_positions - own[0]) * scale + (age + offsets[:, None])[..., None] * reported_velocity
            own_local = (own - own[0]) * scale
            modeled = projected - own_local[:, None, :]
            total_age = offsets[:, None] + age
            maximum_held_report_age = max(maximum_held_report_age, float(total_age.max()))
            for peak in (705.0, 765.0):
                if abs(elapsed + DT - peak) < 1e-8:
                    assert len(visible) == 4
                    assert abs(float(total_age.max()) - 65.0) < 1e-8
                    underway_peak_intervals.append({"encounter_interval_s":[elapsed,elapsed+DT],
                        "navigation_interval_s":[k*DT,(k+1)*DT],"maximum_actual_held_age_s":float(total_age.max()),
                        "all4_contacts_visible":True})
            reported_uncertainty = INITIAL_UNCERTAINTY + np.maximum(GROWTH * total_age, 0.5 * TURN_BOUND * total_age**2)
            # The unseen fourth ship has no fabricated report. Before first
            # acquisition, use an explicitly hypothetical constant65s allowance
            # around physical truth; only the visible subset has AIS projections.
            uncertainty = np.full((len(offsets), 4), INITIAL_UNCERTAINTY + .5 * TURN_BOUND * 65.0**2)
            uncertainty[:, visible] = reported_uncertainty
            hull = np.broadcast_to(truth.hulls, uncertainty.shape)
            nominal = hull + SAFETY_BUFFER
            true_hull, t_hull = swept_margin(actual, hull, substep)
            true_nominal, t_nominal = swept_margin(actual, nominal, substep)
            true_full, t_true = swept_margin(actual, nominal + uncertainty, substep)
            modeled_full, t_model = swept_margin(modeled, nominal[:, visible] + reported_uncertainty, substep)
            error = np.linalg.norm(actual[:, visible] - modeled, axis=-1)
            maximum_coverage = max(maximum_coverage, float((error - reported_uncertainty).max()))
            if navigation.get("ais_step_margin_m") is not None:
                discrepancy = abs(navigation["ais_step_margin_m"] - modeled_full.min())
                maximum_runtime_margin_error = max(maximum_runtime_margin_error, discrepancy)
                assert discrepancy < 1e-5, (run.name, k, discrepancy)
            if navigation.get("ais_step_reserve_margin_m") is not None:
                assert abs(navigation["ais_step_reserve_margin_m"] - (modeled_full.min() - RESERVE)) < 1e-5
            modes[navigation.get("ais_mode", "missing")] += 1
            if 420.0 <= elapsed < 900.0:
                underway_modes[navigation.get("ais_mode", "missing")] += 1
                underway_active_ais_or_colreg_steps += bool(navigation.get("ais_mode") not in (None,"route") or navigation.get("ais_colreg_active_contacts",0)>0)
            measures = (("true_hull", true_hull, t_hull), ("true_nominal_25m", true_nominal, t_nominal),
                        ("truth_full_uncertainty", true_full, t_true), ("projected_full_uncertainty", modeled_full, t_model),
                        ("truth_with_reserve", true_full - RESERVE, t_true), ("projected_with_reserve", modeled_full - RESERVE, t_model))
            for label, values, times in measures:
                column = int(np.argmin(values))
                if label not in worst or values[column] < worst[label]["margin_m"]:
                    worst[label] = {"margin_m": float(values[column]), "mmsi": (visible_ids[column] if label.startswith("projected") else truth.identities[column]),
                                    "elapsed_s": k * DT + float(times[column]), "action_step": k + 1,
                                    "scenario_seconds": elapsed + float(times[column]), "ais_mode": navigation.get("ais_mode")}
                negative[label] += bool(np.any(values < 0))
            observed_map = {int(column): i for i, column in enumerate(visible)}
            for column, identifier in enumerate(truth.identities):
                observed_column = observed_map.get(column)
                observed = observed_column is not None
                writer.writerow(dict(step=k, scenario_seconds=elapsed, mmsi=identifier,
                                     report_age_s=float(age[observed_column]) if observed else None,
                                     reported_visible=observed,
                                     allowance_age_s=float(age[observed_column]) if observed else 65.0,
                                     truth_allowance_scope="actual held report age" if observed else "unseen65s-equivalent physical diagnostic",
                                     true_hull_m=true_hull[column], true_nominal_m=true_nominal[column],
                                     truth_full_m=true_full[column], truth_with_reserve_m=true_full[column] - RESERVE,
                                     projected_full_m=float(modeled_full[observed_column]) if observed else None,
                                     projected_with_reserve_m=float(modeled_full[observed_column]-RESERVE) if observed else None,
                                     max_report_extrapolation_error_m=float(error[:, observed_column].max()) if observed else None))

    return {
        "case": run.name, "encounter_id":experiment["encounter_id"],
        "scope": "live-prefix-only" if prefix_seconds else "all-applied-actions",**attempt_timing,
        "requested_checkpoint_seconds":requested_prefix_seconds,"closed_before_or_at_checkpoint":closed_short,
        "wind_from_degrees": wind_from, "departure_utc": gate_summary["actual_departure_utc"],
        "monitoring_start_utc":gate_summary["monitoring_start_utc"],"traffic_epoch_utc":gate_summary["traffic_epoch_utc"],"gate_wait_seconds":gate_summary["gate_wait_seconds"],
        "applied_actions": len(replay.frames) - 1, "frames": len(replay.frames),
        "integration_substep_seconds": substep, "maximum_pose_reconstruction_error_m": maximum_endpoint,
        "maximum_heading_reconstruction_error_deg": maximum_heading,
        "minimum_native_integration_surge_mps": minimum_native_surge,
        "maximum_first300s_launch_tube_excess_m":maximum_launch_tube_excess,
        "observed_launch_displacement_tube_obeyed_first300s":maximum_launch_tube_excess<=1e-9,
        "launch_tube_coverage_seconds":min(300.0,(len(replay.frames)-1)*DT),
        "maximum_first300s_native_ground_speed_mps":maximum_first300s_native_ground_speed,
        "launch_tube_check_scope":"Observed native first300s displacement relative to launch origin satisfies the assumed30m+2m/s*t disk; this empirical check is not a general speed-bound proof.",
        "astern_action_count": astern_action_count,
        "astern_control_boundary_count": astern_boundary_count,
        "maximum_colreg_ground_velocity_error_mps": maximum_colreg_velocity_error,
        "colreg_fossen_velocity_checks": colreg_fossen_velocity_checks,
        "colreg_legacy_sog_velocity_checks": colreg_legacy_velocity_checks,
        "maximum_recorded_cog_error_deg": maximum_cog_error,
        "maximum_runtime_margin_difference_m": maximum_runtime_margin_error,
        "maximum_truth_outside_report_envelope_m": maximum_coverage, "distinct_report_ages_s": sorted(ages),
        "maximum_applied_held_report_age_s": maximum_held_report_age,
        "longest_fault_peak_experienced": maximum_held_report_age >= 65.-1e-8,
        "truth_minimum_hull_m": worst["true_hull"]["margin_m"],
        "truth_minimum_nominal_m": worst["true_nominal_25m"]["margin_m"],
        "truth_centered_minimum_uncertainty_m": worst["truth_full_uncertainty"]["margin_m"],
        "configured_truth_centered_minimum_m": worst["truth_with_reserve"]["margin_m"],
        "model_minimum_uncertainty_m": worst["projected_full_uncertainty"]["margin_m"],
        "configured_model_minimum_m": worst["projected_with_reserve"]["margin_m"],
        "minimum_details": worst, "negative_action_counts": dict(negative), "ais_modes": dict(modes),
        "verified_constant_velocity_truth_extended_after_s": truth.trace_end,
        "mission_status": mission["status"], "strict_clearance_stop": mission["status"] == "no_full_clearance_action", "waypoint_arrivals": mission["waypoint_arrivals"], "goal_captures": arrivals,
        "final_goal_distance_m": mission["waypoint_distance_metres"], "replay_sha256": digest(recording_path),
        "replay_measurement_path":str(recording_path),"run_log_measurement_path":str(log_path),
        "source_live_recording_path":str(replay_path) if prefix_seconds else None,
        "checkpoint_inputs_frozen":bool(prefix_seconds),
        "run_log_sha256":digest(log_path),
        "history_sha256": digest(output_path), "audit_source_sha256": digest(__file__),
        "truth_fixture_sha256": digest(fixture), "truth_trace_sha256": digest(trace),
        "underway_truth_fixture_sha256":digest(underway_fixture),"underway_truth_trace_sha256":digest(underway_trace),
        "underway_target_mmsi":truth.underway.identities[0],"underway_first_emission_s":truth.first_emission,"underway_first_delivery_s":truth.first_delivery,
        "underway_first_acquisition_encounter_elapsed_s":first_underway_acquisition,
        "underway_report_age_peak_intervals":underway_peak_intervals,
        "underway_both65s_age_peaks_verified":len(underway_peak_intervals)==2,
        "underway_modes_420_to_900_s":dict(underway_modes),
        "regatta_frames_with_reported_search_wall_s":len(regatta_search_cost_samples),
        "maximum_recorded_regatta_search_wall_s":max(regatta_search_cost_samples,default=None),
        "regatta_guidance_telemetry_frame_counts":dict(regatta_guidance_frames),
        "regatta_timing_scope":"Reported last upstream forecast-search cost, possibly repeated by cached references. It is separate from AIS selection and is not an end-to-end control-decision or unique-call count.",
        "underway_active_ais_or_colreg_steps_420_to_900_s":underway_active_ais_or_colreg_steps,
        "physical_clearance_scope":"All4physical contacts throughout. Before acquisition the fourth receives a labeled hypothetical65s allowance; report-centered clearance uses only actually received contacts.",
        "departure_gate_summary_sha256":digest(run/"departure_gate.json"),"departure_gate_log_sha256":digest(run/"departure_gate.jsonl"),
        "measurement": "Exact minima on relative-position and radius chords between 0.05 s plant integration samples; report- and truth-centered uncertainty are distinct.",
        "radius_m": {"nominal_clearance_beyond_combined_hulls": SAFETY_BUFFER, "initial_uncertainty": INITIAL_UNCERTAINTY,
                     "linear_growth_mps": GROWTH, "acceleration_bound_mps2": TURN_BOUND, "steering_reserve": RESERVE},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--fixture", type=Path, help="Override per-run fixture metadata for a single encounter only.")
    parser.add_argument("--trace", type=Path, help="Override per-run independent analytical truth trace.")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--case", action="append", default=[], help="Audit completed named cases only; repeatable.")
    parser.add_argument("--prefix-seconds", type=float, help="Preliminary audit of flushed live recording; does not assert waypoint completion.")
    parser.add_argument("--reuse", action="store_true", help="Reuse case receipts only when all input and source hashes match.")
    args = parser.parse_args()
    study = args.study.resolve()
    snapshot=STUDY/"audit_source_snapshots"/(digest(Path(__file__))+".py")
    snapshot.parent.mkdir(exist_ok=True)
    if snapshot.exists():assert snapshot.read_bytes()==Path(__file__).read_bytes()
    else:snapshot.write_bytes(Path(__file__).read_bytes())
    runs = sorted((study / "runs").glob("*")) if args.prefix_seconds else sorted(path.parent for path in (study / "runs").glob("*/execution.json"))
    if args.case:
        runs = [run for run in runs if run.name in args.case]
        assert {run.name for run in runs} == set(args.case), "Every selected case must have an execution receipt."
    if not runs:
        parser.error("No completed run receipts found.")
    # Failed/early-stop processes are outcomes, not exclusion criteria.
    for run in ([] if args.prefix_seconds else runs):
        manifest = json.loads((run / "execution.json").read_text())
        assert "return_code" in manifest or "exit_code" in manifest, (run.name, "Missing execution outcome")
    suffix = "_prefix" if args.prefix_seconds else ""
    output = study / f"independent_traffic{suffix}_history"
    output.mkdir(exist_ok=True)
    receipts = study / f"independent_traffic{suffix}_receipts"
    receipts.mkdir(exist_ok=True)
    tasks, results = [], []
    for run in runs:
        path = receipts / f"{run.name}.json"
        manifest=json.loads((run / "experiment.json").read_text())
        fixture=args.fixture or Path(manifest["traffic_fixture"])
        trace=args.trace or Path(manifest.get("truth_trace",manifest["traffic_truth_trace_csv"]))
        previous = json.loads(path.read_text()) if args.reuse and path.exists() else None
        measured_replay=Path(previous.get("replay_measurement_path",run/"navigation.jsonl.gz")) if previous else run/"navigation.jsonl.gz"
        measured_log=Path(previous.get("run_log_measurement_path",run/"run.log")) if previous else run/"run.log"
        reusable = previous and all(previous.get(key) == digest(source) for key, source in (
            ("replay_sha256", measured_replay),
            ("run_log_sha256",measured_log),
            ("audit_source_sha256", Path(__file__)),
            ("truth_fixture_sha256", fixture),
            ("truth_trace_sha256", trace),
            ("underway_truth_fixture_sha256",Path(manifest["underway_traffic_fixture"])),
            ("underway_truth_trace_sha256",Path(manifest["underway_truth_trace_csv"])),
            ("departure_gate_summary_sha256",run/"departure_gate.json"),
            ("departure_gate_log_sha256",run/"departure_gate.jsonl"),
            ("history_sha256", output / f"{run.name}.csv"),
        ))
        if reusable:
            for label in ("choose", "applied_decision"):
                timing_path = Path(previous["AIS_" + label + "_attempt_timing_csv"])
                reusable = reusable and timing_path.exists() and digest(timing_path) == previous["AIS_" + label + "_attempt_timing_csv_sha256"]
        if reusable:
            results.append(previous)
        else:
            tasks.append((str(run), str(fixture.resolve()), str(trace.resolve()), str(output), args.prefix_seconds))
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for result in pool.map(audit, tasks):
            save_json(receipts / f"{result['case']}.json", result)
            results.append(result)
            print(f"Audited {result['case']}: {result['mission_status']}, {len(result['goal_captures'])}/8", flush=True)
    aggregate_path = study / f"independent_traffic{suffix}_sweep.json"
    retained = json.loads(aggregate_path.read_text()) if aggregate_path.exists() else []
    merged = {result["case"]: result for result in retained}
    merged.update({result["case"]: result for result in results})
    save_json(aggregate_path, [merged[key] for key in sorted(merged)])
    for result in results:
        print(f"{result['case']}: {result['mission_status']}, {len(result['goal_captures'])}/8; "
              f"truth full+reserve={result['configured_truth_centered_minimum_m']}m; "
              f"report full+reserve={result['configured_model_minimum_m']}m", flush=True)


if __name__ == "__main__":
    main()
