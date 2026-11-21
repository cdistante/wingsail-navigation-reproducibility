"""Decompose the paired elapsed times and describe tracks before outbound 002."""

import csv
import gzip
import json
from pathlib import Path


STUDY = Path(__file__).resolve().parents[1]
STEP_SECONDS = 5


def opening_leg(run, replay):
    mark = next((entry for entry in run["captures"]
                 if (entry["phase"], entry["waypoint"]) == ("outbound", "002")), None)
    last_step = mark["step"] if mark is not None else run["steps"]
    values = {
        "outbound_002_s": mark["elapsed_s"] if mark is not None else None,
        "outbound_002_track_m": mark.get("segment_distance_m") if mark is not None else None,
        "closing_distance_m": mark["distance_metres"] if mark is not None else None,
        "after_002_s": (run["elapsed_s"] - mark["elapsed_s"]
                        if mark is not None and run["status"] == "completed" else None),
        "applied_drift_trim_s": 0,
        "AIS_weather_deferral_s": 0,
        "recovery_mode_s": 0,
        "first_recovery_s": None,
        "successive_close_hauled_reference_side_changes": 0,
    }
    previous_side = None
    measured_step = 0
    with gzip.open(replay, "rt") as stream:
        next(stream)  # Schema header.
        initial = next(stream, None)
        if initial is None:
            assert last_step == 0, replay
            return values
        for expected, line in enumerate(stream, 1):
            if expected > last_step:
                break
            frame = json.loads(line)
            nav = frame["navigation"]
            assert nav["step"] == expected, replay
            measured_step = expected
            values["applied_drift_trim_s"] += STEP_SECONDS * (nav.get("control_source") == "drift trim")
            values["AIS_weather_deferral_s"] += STEP_SECONDS * (
                nav.get("forecast_hold_fallback_reason") == "AIS encounter")
            recovering = nav.get("drift_mode") == "recovering"
            values["recovery_mode_s"] += STEP_SECONDS * recovering
            if recovering and values["first_recovery_s"] is None:
                values["first_recovery_s"] = (expected - 1) * STEP_SECONDS
            side = None
            if nav.get("forecast_closest_m") is not None and nav.get("bow_reference_deg") is not None:
                wind_from = frame["observation"]["wind"]["direction"]
                angle = (nav["bow_reference_deg"] - wind_from + 180) % 360 - 180
                if 30 <= abs(angle) <= 75:
                    side = 1 if angle > 0 else -1
            if side is not None and previous_side is not None and side != previous_side:
                values["successive_close_hauled_reference_side_changes"] += 1
            previous_side = side
    assert measured_step == last_step, replay
    return values


def write_csv(name, rows):
    with (STUDY / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    runs = json.loads((STUDY / "comparison.json").read_text())["runs"]
    by_run = {}
    for run_id, run in runs.items():
        replay = STUDY / "runs" / run_id / "navigation.jsonl.gz"
        by_run[run_id] = opening_leg(run, replay)
    write_csv("regatta_opening_leg_metrics.csv",
              [{"run_id": run_id, **values} for run_id, values in by_run.items()])

    pairs = []
    for on_id, on in by_run.items():
        if not on_id.endswith("__regatta_on"):
            continue
        base = on_id.removesuffix("__regatta_on")
        off = by_run[base + "__regatta_off"]
        delta = {key: None if on[key] is None or off[key] is None else on[key] - off[key]
                 for key in on}
        pairs.append({"base_arm": base, **{"on_minus_off_" + key: value for key, value in delta.items()}})
    write_csv("paired_regatta_leg_deltas.csv", pairs)


if __name__ == "__main__":
    main()
