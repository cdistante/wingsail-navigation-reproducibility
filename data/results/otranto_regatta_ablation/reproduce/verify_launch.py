"""Verify matched launch conditions and regatta-on physical reproducibility."""
from __future__ import annotations

import argparse
import gzip
from itertools import zip_longest
import json
from pathlib import Path

from launch import ARMS, FLAG, MODES, REFERENCE, ROOT, STUDY, digest, load, normalized, save


def physical(frame):
    nav = frame["navigation"]
    return (frame["state"], frame["observation"], nav["step"], nav["phase"], nav["waypoint"],
            nav.get("rudder_command_deg"), nav.get("flap_command_deg"))


def divergence_frame(frame):
    nav = frame["navigation"]
    return {
        "utc": frame["observation"]["time"],
        "latitude": frame["state"]["latitude"], "longitude": frame["state"]["longitude"],
        **{key: val for key, val in nav.items()
           if key.startswith("forecast_hold_") or key.startswith("race_") or key in (
               "step", "phase", "waypoint", "ais_mode", "heading_reference_deg", "rudder_command_deg", "flap_command_deg"
           )},
    }


def compare_recordings(path_a, path_b):
    count_a = count_b = 0
    first_step_difference = None
    first_frames = None
    prefix_600s_equal = True
    with gzip.open(path_a, "rt") as stream_a, gzip.open(path_b, "rt") as stream_b:
        header_a, header_b = json.loads(next(stream_a)), json.loads(next(stream_b))
        for step, (a, b) in enumerate(zip_longest(stream_a, stream_b)):
            count_a += a is not None
            count_b += b is not None
            frames = [json.loads(line) if line is not None else None for line in (a, b)]
            match = None not in frames and physical(frames[0]) == physical(frames[1])
            if not match:
                if first_step_difference is None:
                    first_step_difference = step
                    first_frames = [divergence_frame(frame) if frame is not None else None for frame in frames]
                if step <= 120:
                    prefix_600s_equal = False
    return {
        "frame_counts": [count_a, count_b],
        "route_identical": header_a["route"] == header_b["route"],
        "all_physical_frames_identical": first_step_difference is None,
        "first_physical_difference_step": first_step_difference,
        "first_physical_difference_frames": first_frames,
        "first_600s_physical_frames_identical": count_a >= 121 and count_b >= 121 and prefix_600s_equal,
        "replay_sha256": [digest(path_a), digest(path_b)],
    }


def gate_physics(path):
    with path.open() as stream:
        rows = [json.loads(line) for line in stream]
    for row in rows:
        row.pop("wall_seconds")
    return rows


def check_arm(arm, full):
    reference = REFERENCE / "runs" / arm
    ref_metadata = load(reference / "execution.json")
    ref_gate = gate_physics(reference / "departure_gate.jsonl")
    folders, entries = {}, {}
    for mode in MODES:
        folder = STUDY / "runs" / f"{arm}__regatta_{mode}"
        metadata = load(folder / ("execution.json" if full else "experiment.json"))
        claim = load(folder / "departure_gate.json")
        gate = gate_physics(folder / "departure_gate.jsonl")
        with (folder / "run.log").open() as log:
            runner = next(line.removeprefix("Runner arguments: ") for line in log
                          if line.startswith("Runner arguments: "))
        runner_command = json.loads(runner)
        expected_count = int(mode == "on")
        assert metadata["command"].count(FLAG) == runner_command.count(FLAG) == expected_count, (arm, mode, "regatta flag")
        assert metadata["regatta_enabled"] == bool(expected_count) and metadata["regatta_mode"] == mode, (arm, mode)
        assert runner_command[runner_command.index("--time") + 1] == claim["actual_departure_utc"], (arm, mode, "runtime launch")
        assert claim["actual_departure_utc"] == ref_metadata["actual_departure_utc"], (arm, mode, "baseline launch")
        assert gate == ref_gate, (arm, mode, "gate decisions excluding time to compute")
        assert normalized(metadata["command"], remove_regatta=True) == normalized(
            ref_metadata["command"], remove_regatta=True
        ), (arm, mode, "settings")
        if full:
            for name, sha in metadata["source_sha256"].items():
                assert digest(folder / "source_snapshot" / name) == digest(ROOT / name) == sha, (arm, mode, name)
        folders[mode] = folder
        entries[mode] = metadata
    assert entries["on"]["source_sha256"] == entries["off"]["source_sha256"], (arm, "pair source")
    result = {"pair": arm, "actual_departure_utc": ref_metadata["actual_departure_utc"],
              "matches_prior_gate_telemetry_except_wall_time": True, "native_runtime_flag_checked": True}
    if full:
        result.update(
            paired_physics=compare_recordings(folders["on"] / "navigation.jsonl.gz", folders["off"] / "navigation.jsonl.gz"),
            on_vs_prior_physics=compare_recordings(folders["on"] / "navigation.jsonl.gz", reference / "navigation.jsonl.gz"),
            execution_return_codes={mode: value["return_code"] for mode, value in entries.items()},
            execution_receipt_sha256={mode: digest(folder / "execution.json") for mode, folder in folders.items()},
            source_inputs_match=True,
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-only", action="store_true")
    args = parser.parse_args()
    results = [check_arm(arm, not args.gate_only) for arm in ARMS]
    report = {"all_pairs_pass_gate_and_runtime_option_checks": True, "cases": results}
    if not args.gate_only:
        report.update(
            old_regatta_on_physics_reproduced=all(row["on_vs_prior_physics"]["all_physical_frames_identical"] for row in results),
            all_on_off_pair_first_600s_physics_identical=all(row["paired_physics"]["first_600s_physical_frames_identical"] for row in results),
            comparison_scope="Recorded states, archived environment and received reports, route phase/target and applied rudder/flap commands. Runtime-only telemetry is excluded.",
        )
    output = STUDY / ("launch_gate_verification.json" if args.gate_only else "launch_physics_verification.json")
    save(output, report)
    print(f"Verified {len(results)} physical launch pairs. {output}")


if __name__ == "__main__":
    main()
