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
    while any(start <= generation < end for start, end in ((85.0, 125.0), (145.0, 185.0))):
        generation -= DT
    return encounter_seconds - generation


def audit_attempt_timings(run,replay,prefix_seconds=None,log_path=None):
    prefix="AIS decision timing: "
    log_path=Path(log_path) if log_path is not None else run/"run.log"
    text=log_path.read_text()
    complete=text[:text.rfind("\n")+1]
    rows=[json.loads(line.removeprefix(prefix)) for line in complete.splitlines() if line.startswith(prefix)]
    assert all(np.isfinite(row["wall_seconds"]) and row["wall_seconds"]>=0 for row in rows),run.name
    assert [row["step"] for row in rows]==list(range(1,len(rows)+1)),(run.name,"Missing or duplicate timing attempt")
    applied=max(0,len(replay.frames)-1)
    if prefix_seconds:rows=[row for row in rows if row["step"]<=applied]
    assert len(rows)>=applied,(run.name,"Applied controls missing whole-decision timing")
    if not prefix_seconds:assert len(rows)<=applied+1,(run.name,"Multiple unapplied attempts")
    for row,frame in zip(rows[:applied],replay.frames[1:]):
        assert row["step"]==frame.navigation["step"]
        assert abs(row["wall_seconds"]-frame.navigation["ais_maneuver_search_seconds"])<1e-12,(run.name,row["step"])
    values=[row["wall_seconds"] for row in rows]
    folder=run.parent.parent/"independent_decision_timing"
    folder.mkdir(exist_ok=True)
    timing_path=folder/f"{run.name}.csv"
    fields=list(dict.fromkeys(key for row in rows for key in row))+["applied"]
    with timing_path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fields,lineterminator="\n");writer.writeheader()
        writer.writerows({**row,"applied":row["step"]<=applied} for row in rows)
    return {"AIS_choose_attempts":len(rows),"AIS_choose_applied_attempts":applied,
      "AIS_choose_unapplied_final_attempts":max(0,len(rows)-applied),
      "maximum_complete_AIS_choose_s":max(values,default=None),
      "AIS_choose_attempts_at_or_above_5s":sum(value>=5 for value in values),
      "all_AIS_choose_attempts_below_5s":bool(values) and all(value<5 for value in values),
      "AIS_choose_timing_measurement_log":str(log_path),
      "AIS_choose_timing_measurement_log_sha256":digest(log_path),
      "AIS_choose_attempt_timing_csv":str(timing_path),
      "AIS_choose_attempt_timing_csv_sha256":digest(timing_path),
      "AIS_choose_overrun_or_unapplied_timing_records":[row for row in rows if row["wall_seconds"]>=5 or row["step"]>applied],
      "AIS_attempt_timing_scope":"Every logged complete AISBufferPlanner.choose attempt, including the final unapplied unsafe-stop or exceptional attempt; compared to all applied replay timers."}


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


def no_applied_actions_result(run,replay,experiment,fixture,trace,history_dir,attempt_timing):
    history=history_dir/f"{run.name}.csv"
    history.write_text("step,scenario_seconds,mmsi,report_age_s,true_hull_m,true_nominal_m,truth_full_m,truth_with_reserve_m,projected_full_m,projected_with_reserve_m,max_report_extrapolation_error_m\n")
    lines=[json.loads(line.removeprefix("Mission outcome: ")) for line in (run/"run.log").read_text().splitlines() if line.startswith("Mission outcome: ")]
    mission=lines[0] if lines else {"status":"process_exception","waypoint_arrivals":[],"waypoint_distance_metres":None}
    assert not mission["waypoint_arrivals"]
    initial_clearance=None
    if replay.frames:
        truth=Truth(fixture,trace);frame=replay.frames[0];time=-LEAD
        own=np.array(local_east_north_metres(frame.state.latitude,frame.state.longitude,*truth.origin))
        targets=truth.sample([time],(1,2))[0]
        age=np.array([target.report_age_seconds for target in frame.observation.ais_targets])
        uncertainty=INITIAL_UNCERTAINTY+np.maximum(GROWTH*age,.5*TURN_BOUND*age**2)
        margins=np.linalg.norm(targets-own,axis=1)-truth.hulls-SAFETY_BUFFER-uncertainty-RESERVE
        initial_clearance=float(margins.min())
    result={"case":run.name,"encounter_id":experiment["encounter_id"],"scope":"no-applied-actions",**attempt_timing,
       "wind_from_degrees":experiment["wind_from_degrees"],"departure_utc":experiment["departure_utc"],
       "applied_actions":0,"frames":len(replay.frames),"integration_substep_seconds":.05,
       "maximum_pose_reconstruction_error_m":None,"maximum_heading_reconstruction_error_deg":None,
       "maximum_runtime_margin_difference_m":None,"maximum_truth_outside_report_envelope_m":None,
       "initial_truth_full_plus_reserve_m":initial_clearance,"maximum_applied_held_report_age_s":None,
       "longest_fault_peak_experienced":False,"distinct_report_ages_s":[],"negative_action_counts":{},"ais_modes":{},
       "mission_status":mission["status"],"waypoint_arrivals":[],"goal_captures":[],"final_goal_distance_m":mission["waypoint_distance_metres"],
       "replay_sha256":digest(run/"navigation.jsonl.gz"),"run_log_sha256":digest(run/"run.log"),"history_sha256":digest(history),"audit_source_sha256":digest(__file__),
       "truth_fixture_sha256":digest(fixture),"truth_trace_sha256":digest(trace),
       "measurement":"No command was applied; interval minima/reconstruction errors are unmeasured, not zero. Initial clearance is reported separately when an initial frame exists."}
    for key in ("truth_minimum_hull_m","truth_minimum_nominal_m","truth_centered_minimum_uncertainty_m","configured_truth_centered_minimum_m","model_minimum_uncertainty_m","configured_model_minimum_m"):result[key]=None
    result["minimum_details"]={}
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
    prefix="AIS decision timing: "
    rows=[json.loads(line.removeprefix(prefix)) for line in complete.splitlines() if line.startswith(prefix)]
    rows=[row for row in rows if row["step"]<=count-1]
    assert len(rows)==count-1,"A checkpoint action is missing a complete timing log."
    log=folder/f"decision_timing_{seconds:g}s.jsonl"
    log_content="".join(prefix+json.dumps(row)+"\n" for row in rows)
    if log.exists():assert log.read_text()==log_content
    else:log.write_text(log_content)
    experiment=folder/"experiment.json"
    if experiment.exists():assert experiment.read_bytes()==(run/"experiment.json").read_bytes()
    else:experiment.write_bytes((run/"experiment.json").read_bytes())
    return recording,log


def audit(task):
    run, fixture, trace, history_dir = map(Path, task[:4])
    prefix_seconds = task[4] if len(task) > 4 else None
    truth = Truth(fixture, trace)
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
    assert experiment["encounter_id"] == fixture.parent.name
    attempt_timing=audit_attempt_timings(run,replay,prefix_seconds,log_path)
    assert np.datetime64(experiment["departure_utc"].removesuffix("Z"), "ns") == START
    wind_from = float(experiment["wind_from_degrees"])
    assert wind_from in (355.0, 5.0)
    if len(replay.frames)<=1:
        result=no_applied_actions_result(run,replay,experiment,fixture,trace,history_dir,attempt_timing)
        result.update(requested_checkpoint_seconds=requested_prefix_seconds,closed_before_or_at_checkpoint=closed_short)
        return result
    initial = replay.frames[0]
    assert initial.observation.time == START
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
    minimum_native_surge = np.inf
    astern_action_count = astern_boundary_count = 0
    maximum_colreg_velocity_error = maximum_cog_error = 0.0
    colreg_fossen_velocity_checks = colreg_legacy_velocity_checks = 0
    output_path = history_dir / f"{run.name}.csv"
    fields = ("step", "scenario_seconds", "mmsi", "report_age_s", "true_hull_m", "true_nominal_m", "truth_full_m",
              "truth_with_reserve_m", "projected_full_m", "projected_with_reserve_m", "max_report_extrapolation_error_m")
    with output_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fields, lineterminator="\n")
        writer.writeheader()
        for k, (before, after) in enumerate(zip(replay.frames, replay.frames[1:])):
            observation, navigation = before.observation, after.navigation
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
            elapsed = k * DT - LEAD
            ships = truth.sample(elapsed + offsets, (1, 2))
            reports_by_id = {report.mmsi: report for report in observation.ais_targets}
            assert set(reports_by_id) == set(truth.identities), (run.name, k, set(reports_by_id))
            reports = [reports_by_id[identifier] for identifier in truth.identities]
            age = np.array([report.report_age_seconds for report in reports])
            np.testing.assert_allclose(age, expected_report_age(elapsed), atol=1e-8, rtol=0)
            assert age.max() <= 150.0, "This fixture audit does not assume extrapolation past the runtime age cap."
            if navigation.get("ais_step_report_age_s") is not None:
                assert abs(navigation["ais_step_report_age_s"] - max(age)) < 1e-8
            ages.update(age.tolist())
            reported_positions = np.column_stack(local_east_north_metres(
                [report.latitude for report in reports], [report.longitude for report in reports], *truth.origin,
            ))
            reported_velocity = velocity([report.course for report in reports], [report.speed for report in reports])
            for column in range(len(reports)):
                np.testing.assert_allclose(reported_positions[column], truth.sample([elapsed - age[column]], (1, 2))[0, column], atol=0.001, rtol=0)
                np.testing.assert_allclose(reported_velocity[column], truth.sample([elapsed - age[column]], (3, 4))[0, column], atol=1e-8, rtol=0)
            scale = np.array([np.cos(np.deg2rad(observation.boat.latitude)) / np.cos(np.deg2rad(truth.origin[0])), 1.0])
            actual = (ships - own[:, None, :]) * scale
            projected = (reported_positions - own[0]) * scale + (age + offsets[:, None])[..., None] * reported_velocity
            own_local = (own - own[0]) * scale
            modeled = projected - own_local[:, None, :]
            total_age = offsets[:, None] + age
            maximum_held_report_age = max(maximum_held_report_age, float(total_age.max()))
            uncertainty = INITIAL_UNCERTAINTY + np.maximum(GROWTH * total_age, 0.5 * TURN_BOUND * total_age**2)
            hull = np.broadcast_to(truth.hulls, uncertainty.shape)
            nominal = hull + SAFETY_BUFFER
            true_hull, t_hull = swept_margin(actual, hull, substep)
            true_nominal, t_nominal = swept_margin(actual, nominal, substep)
            true_full, t_true = swept_margin(actual, nominal + uncertainty, substep)
            modeled_full, t_model = swept_margin(modeled, nominal + uncertainty, substep)
            error = np.linalg.norm(actual - modeled, axis=-1)
            maximum_coverage = max(maximum_coverage, float((error - uncertainty).max()))
            if navigation.get("ais_step_margin_m") is not None:
                discrepancy = abs(navigation["ais_step_margin_m"] - modeled_full.min())
                maximum_runtime_margin_error = max(maximum_runtime_margin_error, discrepancy)
                assert discrepancy < 1e-5, (run.name, k, discrepancy)
            if navigation.get("ais_step_reserve_margin_m") is not None:
                assert abs(navigation["ais_step_reserve_margin_m"] - (modeled_full.min() - RESERVE)) < 1e-5
            modes[navigation.get("ais_mode", "missing")] += 1
            measures = (("true_hull", true_hull, t_hull), ("true_nominal_25m", true_nominal, t_nominal),
                        ("truth_full_uncertainty", true_full, t_true), ("projected_full_uncertainty", modeled_full, t_model),
                        ("truth_with_reserve", true_full - RESERVE, t_true), ("projected_with_reserve", modeled_full - RESERVE, t_model))
            for label, values, times in measures:
                column = int(np.argmin(values))
                if label not in worst or values[column] < worst[label]["margin_m"]:
                    worst[label] = {"margin_m": float(values[column]), "mmsi": truth.identities[column],
                                    "elapsed_s": k * DT + float(times[column]), "action_step": k + 1,
                                    "scenario_seconds": elapsed + float(times[column]), "ais_mode": navigation.get("ais_mode")}
                negative[label] += bool(np.any(values < 0))
            for column, identifier in enumerate(truth.identities):
                writer.writerow(dict(step=k, scenario_seconds=elapsed, mmsi=identifier, report_age_s=age[column],
                                     true_hull_m=true_hull[column], true_nominal_m=true_nominal[column],
                                     truth_full_m=true_full[column], truth_with_reserve_m=true_full[column] - RESERVE,
                                     projected_full_m=modeled_full[column], projected_with_reserve_m=modeled_full[column] - RESERVE,
                                     max_report_extrapolation_error_m=error[:, column].max()))
    return {
        "case": run.name, "encounter_id":experiment["encounter_id"],
        "scope": "live-prefix-only" if prefix_seconds else "all-applied-actions",**attempt_timing,
        "requested_checkpoint_seconds":requested_prefix_seconds,"closed_before_or_at_checkpoint":closed_short,
        "wind_from_degrees": wind_from, "departure_utc": experiment["departure_utc"],
        "applied_actions": len(replay.frames) - 1, "frames": len(replay.frames),
        "integration_substep_seconds": substep, "maximum_pose_reconstruction_error_m": maximum_endpoint,
        "maximum_heading_reconstruction_error_deg": maximum_heading,
        "minimum_native_integration_surge_mps": minimum_native_surge,
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
        "mission_status": mission["status"], "waypoint_arrivals": mission["waypoint_arrivals"], "goal_captures": arrivals,
        "final_goal_distance_m": mission["waypoint_distance_metres"], "replay_sha256": digest(recording_path),
        "replay_measurement_path":str(recording_path),"run_log_measurement_path":str(log_path),
        "source_live_recording_path":str(replay_path) if prefix_seconds else None,
        "checkpoint_inputs_frozen":bool(prefix_seconds),
        "run_log_sha256":digest(log_path),
        "history_sha256": digest(output_path), "audit_source_sha256": digest(__file__),
        "truth_fixture_sha256": digest(fixture), "truth_trace_sha256": digest(trace),
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
            ("history_sha256", output / f"{run.name}.csv"),
        ))
        if reusable:
            timing_path=Path(previous["AIS_choose_attempt_timing_csv"])
            reusable=timing_path.exists() and digest(timing_path)==previous["AIS_choose_attempt_timing_csv_sha256"]
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
