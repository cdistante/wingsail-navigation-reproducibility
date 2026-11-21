"""Build the matched native experiment's waypoint and timing tables."""

from __future__ import annotations

from collections import Counter
import csv
import gzip
import json
from pathlib import Path


STUDY = Path(__file__).resolve().parents[1]
WINDS = (5, 355)
METHODS = ("physics", "hybrid")
EXPECTED = [
    ("outbound", "002"), ("outbound", "003"), ("outbound", "004"), ("outbound", "005"),
    ("return", "004"), ("return", "003"), ("return", "002"), ("return", "001"),
]


def json_lines(path, prefix):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line.removeprefix(prefix)) for line in stream if line.startswith(prefix)]


def decision_events(path):
    """Count forecast decisions visible on applied replay rows, once per timestamp."""
    events = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        next(stream)
        for raw in stream:
            frame = json.loads(raw)
            nav = frame["navigation"]
            stamp = nav.get("hybrid_tack_decision_utc")
            if stamp:
                key = stamp, nav["phase"], nav["waypoint"]
                if key not in events:
                    events[key] = {name: value for name, value in nav.items() if name.startswith("hybrid_tack_")}
                    events[key].update(step=nav["step"], phase=nav["phase"], waypoint=nav["waypoint"])
    return list(events.values())


def summarize_arm(arm, audits, captures):
    folder = STUDY / "runs" / arm
    execution = json.loads((folder / "execution.json").read_text())
    result, = json_lines(folder / "run.log", "Mission outcome: ")
    timers = json_lines(folder / "run.log", "Native control decision timing: ")
    applied = [row for row in timers if row["applied"]]
    assert len(applied) == result["steps"]
    assert [row["step"] for row in applied] == list(range(1, result["steps"] + 1))
    dt = execution["dt_s"]
    sequence = [(row["phase"], row["waypoint"]) for row in result["waypoint_arrivals"]]
    within_radius = all(row["distance_metres"] <= 50. for row in result["waypoint_arrivals"])
    if result["arrival_radius_metres"] != 50. or execution["return_code"] or not execution["source_unchanged"]:
        raise ValueError(f"Invalid matched execution receipt for {arm}.")
    last_step = 0
    for row in result["waypoint_arrivals"]:
        captures.append({"wind_from_deg": execution["wind_from_degrees"], "method": arm.split("__")[-1],
                         "phase": row["phase"], "waypoint": row["waypoint"],
                         "elapsed_s": row["step"] * dt, "leg_time_s": (row["step"] - last_step) * dt,
                         "distance_at_capture_m": row["distance_metres"]})
        last_step = row["step"]
    sweep = audits[arm]
    events = decision_events(folder / "navigation.jsonl.gz")
    first = next((row["step"] * dt for row in result["waypoint_arrivals"]
                  if (row["phase"], row["waypoint"]) == EXPECTED[0]), None)
    aux = {key: value for key, value in sweep.items() if key in
           ("configured_truth_centered_minimum_m", "configured_model_minimum_m",
            "maximum_complete_AIS_choose_s", "maximum_complete_AIS_applied_decision_s",
            "maximum_applied_held_report_age_s")}
    return {"status": result["status"], "actual_launch_utc": execution["actual_departure_utc"],
            "time_to_002_s": first, "underway_time_s": result["steps"] * dt,
            "stopped_reason": result.get("termination_reason"),
            "capture_order_and_radius_pass": sequence == EXPECTED and within_radius,
            "capture_count": len(sequence), "capture_order": sequence,
            **aux, "maximum_complete_native_control_s": max(row["wall_seconds"] for row in timers),
            "complete_native_control_calls": len(timers),
            "native_control_at_or_above_5s": sum(row["wall_seconds"] >= 5. for row in timers),
            "replayed_forecast_classifications": Counter(row["hybrid_tack_decision"] for row in events),
            "replayed_accepted_first_trims": [row for row in events
                                              if row["hybrid_tack_decision"] == "accepted first trim"]}


def compare_replay_pair(wind):
    prefix = STUDY / "runs" / f"crossing_fan__otranto_{wind:03d}__physics"
    treated = prefix.with_name(prefix.name.removesuffix("physics") + "hybrid")
    controls = ("rudder_command_deg", "flap_command_deg")
    first_change = None
    compared = 0
    with (gzip.open(prefix / "navigation.jsonl.gz", "rt", encoding="utf-8") as a,
          gzip.open(treated / "navigation.jsonl.gz", "rt", encoding="utf-8") as b):
        route_a, route_b = json.loads(next(a))["route"], json.loads(next(b))["route"]
        assert route_a == route_b
        for raw_a, raw_b in zip(a, b):
            physics, hybrid = json.loads(raw_a), json.loads(raw_b)
            obs_a, obs_b = physics["observation"], hybrid["observation"]
            n_a, n_b = physics["navigation"], hybrid["navigation"]
            if (obs_a["time"], obs_a["ais_targets"], n_a["step"]) != (obs_b["time"], obs_b["ais_targets"], n_b["step"]):
                raise ValueError(f"Paired AIS reception differs at {wind} degrees, step {n_a['step']}.")
            compared += 1
            controls_a, controls_b = [tuple(nav.get(key) for key in controls) for nav in (n_a, n_b)]
            if first_change is None and (controls_a != controls_b or physics["state"] != hybrid["state"]):
                first_change = {"step": n_b["step"], "observation_utc": obs_b["time"],
                                "phase": n_b["phase"], "waypoint": n_b["waypoint"],
                                "physics_commands_deg": controls_a, "hybrid_commands_deg": controls_b,
                                "physics_bow_reference_deg": n_a.get("bow_reference_deg"),
                                "hybrid_bow_reference_deg": n_b.get("bow_reference_deg"),
                                "physics_native_schedule_closest_m": n_a.get("forecast_closest_m"),
                                "hybrid_native_schedule_closest_m": n_b.get("forecast_closest_m"),
                                "hybrid_forecast_decision": {key: value for key, value in n_b.items()
                                                             if key.startswith("hybrid_tack_")}}
    return {"matched_reception_fix_count": compared, "first_state_or_command_difference": first_change}


def main():
    sweeps = json.loads((STUDY / "independent_traffic_sweep.json").read_text())
    audits = {row["case"]: row for row in sweeps}
    captures = []
    outcomes = {}
    for wind in WINDS:
        for method in METHODS:
            arm = f"crossing_fan__otranto_{wind:03d}__{method}"
            outcomes[arm] = summarize_arm(arm, audits, captures)
    with (STUDY / "waypoint_captures.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(captures[0]))
        writer.writeheader()
        writer.writerows(captures)
    compare = {}
    for wind in WINDS:
        physics, hybrid = [outcomes[f"crossing_fan__otranto_{wind:03d}__{method}"] for method in METHODS]
        assert physics["actual_launch_utc"] == hybrid["actual_launch_utc"]
        compare[str(wind)] = {
            "hybrid_minus_physics_to_outbound_002_s": (hybrid["time_to_002_s"] - physics["time_to_002_s"]
                                                       if None not in (hybrid["time_to_002_s"],
                                                                       physics["time_to_002_s"]) else None),
            "hybrid_minus_physics_loop_s": (hybrid["underway_time_s"] - physics["underway_time_s"]
                                           if physics["capture_order_and_radius_pass"]
                                           and hybrid["capture_order_and_radius_pass"] else None),
            **compare_replay_pair(wind),
        }
    (STUDY / "native_comparison.json").write_text(json.dumps({"runs": outcomes, "comparisons": compare},
                                                             indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
