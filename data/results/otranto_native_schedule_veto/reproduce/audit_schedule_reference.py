"""Bind selected schedule switch/closest metadata to emitted nominal references."""
from datetime import datetime, timezone
from hashlib import sha256
import gzip
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def main():
    records = {}
    for wind in (5, 355):
        case = f"crossing_fan__otranto_{wind:03d}__hybrid"
        path = STUDY/"runs"/case/"navigation.jsonl.gz"
        count = 0
        closest_error = minutes_error = 0.
        with gzip.open(path, "rt") as stream:
            next(stream)
            for line in stream:
                frame = json.loads(line)
                nav = frame["navigation"]
                if nav.get("hybrid_tack_decision_utc") is None:
                    continue
                when = datetime.fromisoformat(frame["observation"]["time"])
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
                before_seconds = when.timestamp()-5.
                interval = nav["hybrid_tack_schedule_interval_s"]
                switch = nav["hybrid_tack_selected_switch_window"]
                expected = (switch*interval-before_seconds)/60. if switch > int(before_seconds//interval) else None
                actual = nav["forecast_tack_in_min"]
                assert (expected is None) == (actual is None), (case, nav["step"])
                if expected is not None:
                    error = abs(expected-actual)
                    assert error < 1e-9, (case, nav["step"], "switch clock")
                    minutes_error = max(minutes_error, error)
                error = abs(nav["forecast_closest_m"]-nav["hybrid_tack_selected_closest_m"])
                assert error < 1e-8, (case, nav["step"], "closest approach")
                closest_error = max(closest_error, error)
                count += 1
        records[case] = {"checked_reference_frames": count,
                         "maximum_forecast_closest_difference_m": closest_error,
                         "maximum_scheduled_tack_difference_minutes": minutes_error,
                         "replay_sha256": sha256(path.read_bytes()).hexdigest()}
    result = {"status": "PASS", "cases": records,
              "audit_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Selected native schedule closest approach and absolute switch window agree with each emitted timestamped/cached nominal reference. Scheduled/unavailable continuations are not new policy evaluations."}
    (STUDY/"independent_schedule_reference_check.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
