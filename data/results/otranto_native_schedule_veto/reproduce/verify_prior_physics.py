"""Check that each fresh physics run reproduces the previous native physics run."""

from hashlib import file_digest, sha256
from itertools import zip_longest
from pathlib import Path
import gzip
import json


STUDY = Path(__file__).resolve().parents[1]
COMPARED_FIELDS = ("state", "observation", "ais_course_forecasts", "wind_grid", "grid")
NAVIGATION_FIELDS = (
    "step", "phase", "waypoint", "waypoint_distance_m",
    "heading_reference_deg", "bow_reference_deg", "course_over_ground_deg",
    "rudder_command_deg", "flap_command_deg", "control_source",
    "forecast_closest_m", "forecast_tack_in_min", "drift_mode", "ais_mode",
    "ais_course_margin_m", "ais_step_margin_m", "ais_step_reserve_margin_m",
)


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def evidence(line):
    frame = json.loads(line)
    return {
        **{field: frame[field] for field in COMPARED_FIELDS},
        "navigation": {field: frame["navigation"].get(field) for field in NAVIGATION_FIELDS},
    }


def verify(wind):
    folder = STUDY / "runs" / f"crossing_fan__otranto_{wind:03d}__physics"
    receipt = json.loads((folder / "execution.json").read_text())
    if receipt["return_code"] or not receipt["source_unchanged"]:
        raise ValueError(f"{folder.name}: fresh physics execution was not preserved.")
    old_receipt = Path(receipt["prior_native_physics_receipt"])
    old_replay = old_receipt.with_name("navigation.jsonl.gz")
    new_replay = folder / "navigation.jsonl.gz"
    if (digest(old_receipt) != receipt["prior_native_physics_execution_sha256"]
            or digest(old_replay) != receipt["prior_native_physics_replay_sha256"]):
        raise ValueError(f"{folder.name}: the earlier native reference changed.")
    old = json.loads(old_receipt.read_text())
    if old["actual_departure_utc"] != receipt["actual_departure_utc"]:
        raise ValueError(f"{folder.name}: the actual simulated departure differs.")

    frames = 0
    canonical = sha256()
    with gzip.open(old_replay, "rt") as old_stream, gzip.open(new_replay, "rt") as new_stream:
        original, repeated = (json.loads(next(stream)) for stream in (old_stream, new_stream))
        for field in ("format", "version", "route", "route_name", "view"):
            if original[field] != repeated[field]:
                raise ValueError(f"{folder.name}: replay header field {field} differs.")
        for old_line, new_line in zip_longest(old_stream, new_stream):
            if old_line is None or new_line is None:
                raise ValueError(f"{folder.name}: replay length differs at frame {frames}.")
            original, repeated = evidence(old_line), evidence(new_line)
            if original != repeated:
                raise ValueError(f"{folder.name}: replay evidence differs at frame {frames}.")
            canonical.update(json.dumps(repeated, sort_keys=True, separators=(",", ":")).encode())
            frames += 1
    return {
        "run_id": folder.name,
        "reference_run": old_receipt.parent.name,
        "reference_execution": str(old_receipt),
        "actual_departure_utc": receipt["actual_departure_utc"],
        "matching_frames": frames,
        "canonical_physical_evidence_sha256": canonical.hexdigest(),
        "fresh_replay_sha256": digest(new_replay),
        "reference_replay_sha256": receipt["prior_native_physics_replay_sha256"],
    }


if __name__ == "__main__":
    rows = [verify(wind) for wind in (5, 355)]
    result = STUDY / "prior_native_physics_reproduction.json"
    staged = result.with_suffix(".json.tmp")
    staged.write_text(json.dumps({
        "compared_replay_fields": COMPARED_FIELDS,
        "compared_navigation_fields": NAVIGATION_FIELDS,
        "cases": rows,
    }, indent=2) + "\n")
    staged.replace(result)
    for row in rows:
        print(row["run_id"], row["matching_frames"], row["canonical_physical_evidence_sha256"])
