"""Register disjoint seasonal native sailing episodes before evaluating any policy."""
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
from math import atan2, cos, degrees, radians, sin
from pathlib import Path
from xml.etree import ElementTree as ET


STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[2]
RADIUS_M = 50.0
CONTROL_DT_S = 5
CAP_SECONDS = 6 * 3600
REPORT_INTERVAL_SECONDS = 5
FORECAST_BOUND_HOURS = 10
# Stable launch slots sample four dates and multiple hours without using outcomes.
FOUR_SLOTS = ((0, 6), (2, 12), (4, 18), (6, 0))
THREE_SLOTS = ((1, 6), (3, 12), (5, 18))
ADDITIONAL_TRAIN_SLOTS = ((1, 0), (3, 6), (5, 12), (6, 6))
ADDITIONAL_VALIDATION_SLOTS = ((0, 0), (2, 18), (4, 6))
ADDITIONAL_TEST_SLOTS = ((1, 18), (5, 12))
WEEKS = {
    "winter": ("2026-01-12", "train"),
    "spring": ("2026-04-13", "validation"),
    "summer": ("2026-07-13", "train"),
    "autumn": ("2026-10-12", "test"),
}
ROUTES = {
    "otranto_002": ("otranto", "current_route/otranto_001_002_test.gpx"),
    "otranto_003": ("otranto", "current_route/otranto_short_001_003_loop.gpx"),
    "brindisi_small": ("brindisi", "current_route/small route.gpx"),
    "brindisi_003": ("brindisi", "current_route/opencpn_001_003_test.gpx"),
    "brindisi_004": ("brindisi", "current_route/opencpn_001_004_test.gpx"),
}
# Otranto GPXs share the opening leg. The three Brindisi files are one geographic
# family. The longer Otranto file stays reserved with the Otranto family.
REGION_RESERVED_FILES = {"otranto": ("current_route/otranto_short_test_loop.gpx",)}
TARGET_FIELDS = (
    "name", "ship_type", "along_m", "starboard_m", "course_offset_deg",
    "speed_mps", "length_m", "beam_m",
)
# Local "along" points toward the first GPX mark. "Starboard" and course
# offsets are clockwise from along. All ships are synthetic.
TRAFFIC = {
    "train_crossing": {
        "split": "train", "delay_s": 10, "drop_windows_s": (),
        "targets": (
            ("crossing ferry", 60, 255, -375, 90, 2.5, 45, 10),
            ("outbound monitor", 52, 140, 335, 0, 1.9, 25, 8),
            ("distant inbound freighter", 70, 960, -460, 180, 2.7, 62, 12),
        ),
    },
    "train_opposed": {
        "split": "train", "delay_s": 10, "drop_windows_s": ((85, 105), (115, 135)),
        "targets": (
            ("parallel inbound", 70, 560, 220, 180, 2.2, 66, 14),
            ("parallel overtaking", 52, -510, -205, 0, 2.5, 23, 8),
            ("outer outbound ferry", 60, 85, -390, 18, 2.1, 48, 10),
        ),
    },
    "validation_slanted": {
        "split": "validation", "delay_s": 15, "drop_windows_s": ((145, 170),),
        "targets": (
            ("slanted eastbound ferry", 60, 295, -445, 72, 2.35, 50, 11),
            ("outer port freighter", 70, 830, -360, 169, 2.25, 58, 13),
            ("parallel tug", 52, -350, 285, 12, 2.55, 26, 9),
        ),
    },
    "test_oblique_fan": {
        "split": "test", "delay_s": 20, "drop_windows_s": ((120, 160), (190, 225)),
        "targets": (
            ("inbound oblique crossing", 60, 490, 465, -122, 2.4, 52, 12),
            ("offside joining tug", 52, -285, -235, 21, 2.65, 27, 9),
            ("distant eastbound freighter", 70, 850, 510, 169, 2.15, 61, 14),
        ),
        "velocity_ramps": (
            {"ship_index": 0, "start_s": 155, "end_s": 185, "final_offset_deg": -115, "final_speed_mps": 2.6},
        ),
    },
    "test_raking_overtake": {
        "split": "test", "delay_s": 20, "drop_windows_s": ((125, 165), (200, 235)),
        "targets": (
            ("raking overtaker", 52, -435, -160, 22, 2.75, 29, 9),
            ("starboard crossing ferry", 60, 410, 385, -75, 2.8, 51, 11),
            ("outer freight ship", 70, 820, -455, 172, 2.0, 60, 13),
        ),
        "velocity_ramps": (
            {"ship_index": 0, "start_s": 155, "end_s": 180, "final_offset_deg": 29, "final_speed_mps": 2.5},
        ),
    },
}
TRAIN_ROUTES = ("otranto_002", "otranto_003", "otranto_002", "otranto_003")
TRAIN_TRAFFIC = ("train_crossing", "train_opposed", "train_opposed", "train_crossing")
VALIDATION_ROUTES = ("otranto_002", "otranto_003", "otranto_003")
TEST_ROUTES = ("brindisi_small", "brindisi_003", "brindisi_004", "brindisi_small")
TEST_TRAFFIC = ("test_oblique_fan", "test_raking_overtake")


def sha(path):
    with path.open("rb") as stream:
        hashed = sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hashed.update(chunk)
    return hashed.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc(value):
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def local_bearing(start, goal):
    a, b = radians(start["latitude"]), radians(goal["latitude"])
    delta = radians(goal["longitude"] - start["longitude"])
    return degrees(atan2(sin(delta) * cos(b),
                         cos(a) * sin(b) - sin(a) * cos(b) * cos(delta))) % 360.


def load_route(file):
    point_elements = ET.parse(ROOT / file).findall(".//{*}rtept")
    points = [{
        "waypoint": point.findtext("{*}name"),
        "latitude": float(point.attrib["lat"]),
        "longitude": float(point.attrib["lon"]),
    } for point in point_elements]
    names = {point["waypoint"] for point in points}
    require(len(points) >= 2 and len(names) == len(points) and all(names),
            f"GPX route must identify every mark and a unique departure: {file}.")
    return points


def archive_weeks():
    blocks = {}
    for season, (isodate, split) in WEEKS.items():
        day = date.fromisoformat(isodate)
        first = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        end = first + timedelta(days=7)
        blocks[season] = {
            "split": split, "start_utc": utc(first), "end_utc_exclusive": utc(end),
            "requested_archive_start_utc": utc(first - timedelta(days=1)),
            "requested_archive_end_utc_exclusive": utc(end + timedelta(days=2)),
            "requested_hourly_samples": 240,
            "files": {kind: f"forcing/{season}/{kind}.nc" for kind in ("wind", "current", "wave")},
        }
    ordered = sorted(blocks.values(), key=lambda block: block["requested_archive_start_utc"])
    require(all(left["requested_archive_end_utc_exclusive"] <= right["requested_archive_start_utc"]
                for left, right in zip(ordered, ordered[1:])), "The buffered archive blocks overlap.")
    return blocks


def roster(split, season, expanded):
    if split == "train":
        slots = (*FOUR_SLOTS, *ADDITIONAL_TRAIN_SLOTS) if expanded else FOUR_SLOTS
        for index, slot in enumerate(slots):
            route_id = TRAIN_ROUTES[index % len(TRAIN_ROUTES)]
            profile = TRAIN_TRAFFIC[(index + int(season == "summer")) % len(TRAIN_TRAFFIC)]
            yield slot, route_id, (profile,)
    elif split == "validation":
        slots = (*THREE_SLOTS, *ADDITIONAL_VALIDATION_SLOTS) if expanded else THREE_SLOTS
        for index, slot in enumerate(slots):
            yield slot, VALIDATION_ROUTES[index % len(VALIDATION_ROUTES)], ("validation_slanted",)
    else:
        for slot, route_id in zip(FOUR_SLOTS, TEST_ROUTES, strict=True):
            yield slot, route_id, TEST_TRAFFIC
        if expanded:
            for slot, route_id in zip(ADDITIONAL_TEST_SLOTS, ("brindisi_003", "brindisi_004"), strict=True):
                yield slot, route_id, TEST_TRAFFIC


def scenario_rows(blocks, routes, expanded):
    for season, block in blocks.items():
        split = block["split"]
        first = datetime.fromisoformat(block["start_utc"].replace("Z", "+00:00"))
        stop = first + timedelta(days=7)
        for (day, hour), route_id, profiles in roster(split, season, expanded):
            when = first + timedelta(days=day, hours=hour)
            latest_forecast = when + timedelta(seconds=CAP_SECONDS, hours=FORECAST_BOUND_HOURS, minutes=15)
            require(latest_forecast < stop, "A registered episode would sample beyond its period.")
            route = routes[route_id]
            stamp = when.strftime("%Y%m%dT%H%MZ")
            for profile in profiles:
                require(TRAFFIC[profile]["split"] == split, "Traffic recipe assigned across its split boundary.")
                yield {
                    "id": f"{season}__{stamp}__{route_id}__{profile}",
                    "split": split, "archive_block": season, "departure_utc": utc(when),
                    "weather_coverage_through_utc": utc(latest_forecast),
                    "route_id": route_id, "route_family": route["family"], "route_gpx": route["route_gpx"],
                    "departure": route["waypoints"][0], "initial_heading_deg": route["first_leg_bearing_deg"],
                    "expected_captures": [{"phase": "outbound", "waypoint": point["waypoint"]}
                                          for point in route["waypoints"][1:]],
                    "traffic_profile": profile, "capture_radius_m": RADIUS_M,
                    "control_dt_s": CONTROL_DT_S, "max_steps": CAP_SECONDS // CONTROL_DT_S,
                    "return_to_start": False, "wind_from_override_deg": None,
                    "pairing": ["physics", "seasonal_learner"] if split == "test" else ["physics"],
                }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expanded", action="store_true", help="Use the additive 16/6/12 roster.")
    args = parser.parse_args()
    destination = args.output or STUDY / ("manifest_expanded.json" if args.expanded else "manifest.json")
    blocks = archive_weeks()
    routes = {}
    for name, (family, path) in ROUTES.items():
        points = load_route(path)
        routes[name] = {
            "route_gpx": path, "family": family, "sha256": sha(ROOT / path), "waypoints": points,
            "first_leg_bearing_deg": local_bearing(points[0], points[1]),
        }
    rows = list(scenario_rows(blocks, routes, args.expanded))
    counts = Counter(row["split"] for row in rows)
    expected = {"train": 16, "validation": 6, "test": 12} if args.expanded else {"train": 8, "validation": 3, "test": 8}
    require(counts == expected, "Incorrect scenario counts.")
    require(len({row["id"] for row in rows}) == len(rows), "Episode identity is not unique.")
    heldout = {row["route_family"] for row in rows if row["split"] == "test"}
    fit = {row["route_family"] for row in rows if row["split"] != "test"}
    require(fit.isdisjoint(heldout), "Held-out route geography overlaps training or validation.")
    profiles = {}
    for name, profile in TRAFFIC.items():
        records = [dict(zip(TARGET_FIELDS, target, strict=True)) for target in profile["targets"]]
        require(1 <= len(records) <= 3, "The registered fixture must contain at least one contact.")
        profiles[name] = {**profile, "report_interval_s": REPORT_INTERVAL_SECONDS, "targets": records}

    manifest = {
        "version": 1, "roster": "expanded" if args.expanded else "pilot",
        "archive_bounds_request": {"min_latitude": 39.85, "max_latitude": 40.85,
                                   "min_longitude": 17.65, "max_longitude": 19.15},
        "archive_blocks": blocks, "routes": routes, "traffic_profiles": profiles,
        "scenarios": rows, "scenario_counts": dict(counts),
        "paired_policy_arms": sum(len(row["pairing"]) for row in rows),
        "split_rules": {
            "weather": "Winter and summer fit; spring validation; autumn test. No dates overlap, including download buffers.",
            "geography": "Otranto fit and validation. Hold out the complete Brindisi family, including all three nearby or overlapping GPXs.",
            "traffic": "Fit, validation and test are assigned distinct synthetic traffic geometries and receipt timing profiles.",
            "independent_groups": "Two traffic geometry scenarios share each held-out route/departure group and are paired sensitivity trials.",
            "archive_provenance": "The block and roster are predeclared. Local archive coverage, checksums and provider provenance must be verified separately before launch.",
            "lookahead_scope": "Use archived valid-time fields under shared offline look-ahead. Validate chronology and accepted timestamps without implying operational forecast issuance.",
            "traffic_scope": "Synthetic native AIS traffic; recipe coordinates are relative to the first GPX leg, not reused global tracks.",
            "reserved": {family: [{"route_gpx": path, "sha256": sha(ROOT/path)} for path in paths]
                         for family, paths in REGION_RESERVED_FILES.items()},
        },
        "metrics": {
            "pair_unit": "Each test scenario ID pairs one frozen physics and one frozen learner arm.",
            "success": "Every listed waypoint captured in order within the actual 50 m gate; start mark is the departure.",
            "time": "Elapsed active-mark captures and mission completion from actual launch, with launch delay reported separately.",
            "clearance": "Minimum full-uncertainty clearance with extra 3 m reserve; report- and truth-centred audited minima recorded separately.",
            "timing": "Complete native decision time and overrun count over the 5 s control budget, including terminal attempts.",
            "attrition": "All declared episodes are retained. Mark nonavailable forcing/inputs distinctly. Record chart or AIS stops and timeouts without selecting on outcomes.",
        },
        "manifest_source_sha256": sha(Path(__file__).resolve()),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"scenarios": manifest["scenario_counts"], "native_policy_arms": manifest["paired_policy_arms"],
                      "test_route_time_groups": len({(row["route_id"], row["departure_utc"]) for row in rows
                                                     if row["split"] == "test"})}, indent=2))


if __name__ == "__main__":
    main()
