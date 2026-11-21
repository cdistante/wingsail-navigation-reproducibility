"""Freeze train, validation and joint transfer assignments before policy fitting."""
from __future__ import annotations

from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path


STUDY = Path(__file__).resolve().parents[1]
PRIOR = STUDY.with_name("learned_safe_mark_policy")
ROUTE_FAMILIES = {
    "brindisi": ("opencpn_001_003_test.gpx", "opencpn_001_004_test.gpx", "small route.gpx"),
    "otranto": ("otranto_001_002_test.gpx", "otranto_short_001_003_loop.gpx", "otranto_short_test_loop.gpx"),
    "costiera": ("costiera.gpx",),
    "corfu": ("rotta_corfu_nord.gpx",),
    "naples": ("naples_8_8nm_round_trip.gpx",),
}
FIT_TRAFFIC = ("crossing", "head_on", "mixed_dropout", "mixed_long_dropout")
TEST_TRAFFIC = ("overtaking", "mixed_sharp_dropout")


def digest(path):
    with path.open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def read_gzip(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def endpoint_pair(case):
    return tuple(sorted((tuple(case["origin_lat_lon"]), tuple(case["target_lat_lon"]))))


def archive_leg_key(case):
    return case["route_family"], case["gpx"], case["leg_index"]


def main():
    archive_manifest = PRIOR / "archive_scenario_manifest.json"
    archive_polars = PRIOR / "archive_scenarios.json.gz"
    synthetic_polars = PRIOR / "scenarios.json.gz"
    archive = json.loads(archive_manifest.read_text())
    assert digest(archive_polars) == archive["scenario_sha256"]
    archive_cases = {row["identifier"]: row for row in read_gzip(archive_polars)}
    synthetic = read_gzip(synthetic_polars)
    gpx_to_family = {gpx: family for family, routes in ROUTE_FAMILIES.items() for gpx in routes}
    assert set(gpx_to_family) == {group["gpx"] for group in archive["groups"].values()}
    assignments = {
        "training": {"synthetic_ids": sorted(row["identifier"] for row in synthetic if row["split"] == "train"),
                     "archive_ids": [], "traffic_profiles": list(FIT_TRAFFIC)},
        "validation": {"synthetic_ids": sorted(row["identifier"] for row in synthetic if row["split"] == "validation"),
                       "archive_ids": [], "traffic_profiles": list(FIT_TRAFFIC)},
        "test": {"synthetic_ids": [], "archive_ids": [], "traffic_profiles": list(TEST_TRAFFIC)},
    }
    ledger = {}
    unavailable = {}
    for identifier, row in sorted(archive["rows"].items()):
        group = archive["groups"][row["group"]]
        leg = archive["leg_blocks"][row["leg"]]
        gpx = group["gpx"]
        family = gpx_to_family[gpx]
        block = group["archive_block"]
        if family == "brindisi" and block == "july_overlap":
            split = "training"
        elif family in {"otranto", "costiera", "corfu"} and block == "original":
            split = "test"
        else:
            split = "excluded"
        case = {
            "split": split, "source_status": row["status"], "archive_block": block, "gpx": gpx,
            "route_family": family, "departure_utc": group["departure_utc"],
            "wind_variant": row["wind_variant"], "leg_index": leg["leg_index"],
            "origin_waypoint": leg["origin_waypoint"], "target_waypoint": leg["target_waypoint"],
            "origin_lat_lon": leg["origin_lat_lon"], "target_lat_lon": leg["target_lat_lon"],
            "initial_leg_distance_m": leg["initial_leg_distance_m"], "radius_m": group["declared_radius_m"],
            "entire_chart_disk_blocked": leg["target_chart_preflight"]["complete_capture_disk_in_blocked_cell"],
            "initial_range_outside_prior_synthetic_training": leg["range_outside_synthetic_support"],
            "radius_outside_prior_synthetic_training": leg["radius_outside_synthetic_support"],
        }
        ledger[identifier] = case
        if row["status"] != "generated":
            assert identifier not in archive_cases
            unavailable[identifier] = {"split": split, "status": row["status"], "detail": row.get("detail")}
        else:
            assert identifier in archive_cases
            if split in assignments:
                assignments[split]["archive_ids"].append(identifier)
    live = {split: [ledger[key] for key in spec["archive_ids"]] for split, spec in assignments.items()}
    assert len(live["training"]) == 63 and len(live["test"]) == 108
    assert set(assignments["training"]["synthetic_ids"]).isdisjoint(assignments["validation"]["synthetic_ids"])
    assert set(FIT_TRAFFIC).isdisjoint(TEST_TRAFFIC)
    assert {row["gpx"] for row in live["training"]}.isdisjoint(row["gpx"] for row in live["test"])
    assert {row["archive_block"] for row in live["training"]}.isdisjoint(row["archive_block"] for row in live["test"])
    assert {endpoint_pair(row) for row in live["training"]}.isdisjoint(endpoint_pair(row) for row in live["test"])
    coverage = {
        split: {
            "synthetic_scenarios": len(assignments[split]["synthetic_ids"]),
            "archive_scenarios": len(rows),
            "archive_scenarios_with_chart_disk_blocked": sum(row["entire_chart_disk_blocked"] for row in rows),
            "archive_blocks": sorted({row["archive_block"] for row in rows}),
            "archive_departure_times_utc": sorted({row["departure_utc"] for row in rows}),
            "archive_route_files": sorted({row["gpx"] for row in rows}),
            "archive_cases_per_route_family": dict(Counter(row["route_family"] for row in rows)),
            "archive_route_legs": len({archive_leg_key(row) for row in rows}),
            "distinct_archive_leg_endpoint_pairs": len({endpoint_pair(row) for row in rows}),
            "traffic_profiles": len(assignments[split]["traffic_profiles"]),
            "paired_episode_cells": (
                len(rows) + len(assignments[split]["synthetic_ids"])
            ) * len(assignments[split]["traffic_profiles"]),
        }
        for split, rows in live.items()
    }
    manifest = {
        "format_version": 1,
        **assignments,
        "coverage": coverage,
        "route_families": ROUTE_FAMILIES,
        "archive_cases": ledger,
        "declared_unavailable_cases": unavailable,
        "archive_holds": {
            "scope": "Test cells pair three simultaneously withheld axes: geographical route families, original archive block, and exact traffic profiles.",
            "geometry": "The three Brindisi tracks overlap spatially and stay together. The Otranto files share exact legs and all stay in test; repeated legs and wind variants are paired sensitivity cases.",
            "weather": "Fit uses July 29 weather at training routes and generated synthetic scenarios. Original September 30 through October 1 forcing is test-only. September/October wind-bearing perturbations retain their archived weather magnitudes and dates.",
            "validation": "Fit and validation synthetic scenario IDs are disjoint. Validation reuses training traffic profiles but has no archived route or weather row. Reserve the existing synthetic test IDs from this study.",
            "traffic": "The held-out exact profiles are single-ship overtaking and mixed sharp-dropout. Components of these fixtures overlap mixed training vessel templates. The standalone encounter and sharp course/speed schedule are unseen; the long dropout schedule itself is shared.",
            "exclusions": "Naples and cross-block route cases are unused. Corfu long legs stay in the ledger and test; chart disks entirely blocked by GEBCO stay as preflight outcomes. Nine unavailable July Corfu variants remain represented without rollouts.",
        },
        "files_sha256": {
            str(path.relative_to(STUDY.parents[2])): digest(path)
            for path in (archive_manifest, archive_polars, synthetic_polars, Path(__file__).resolve())
        },
    }
    output = STUDY / "splits.json"
    output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps(coverage, indent=2))


if __name__ == "__main__":
    main()
