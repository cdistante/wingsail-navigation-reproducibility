"""Independently audit the ETA gate and matched Otranto route replays."""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import shlex
import sys
from unittest.mock import patch


STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[2]
PRIORITY_AUDIT = STUDY.with_name("otranto_layline_priority") / "reproduce" / "audit_priority.py"
PRIORITY_SHA256 = "0841994437ee4dec20073ffde1d5384ab7a1cb5a7ea5b8f1d64a212d948d68af"
FLAG = "--compare-waypoint-layline-eta"
WINDS = (5, 355)
ARMS = tuple(f"crossing_fan__otranto_{wind:03d}__layline_eta" for wind in WINDS)
SOURCE_CHANGES = {
    "world/forecast_tacks.py",
    "world/legacy_planner_adapter.py",
    "world/run_legacy_fossen.py",
}
ETA_FIELDS = (
    "waypoint_layline_eta_s",
    "waypoint_layline_forecast_eta_s",
    "waypoint_layline_eta_advantage_s",
    "waypoint_layline_comparison_radius_m",
    "waypoint_layline_steady_eta_s",
)

sys.path.insert(0, str(PRIORITY_AUDIT.parent))
import audit_priority as priority


prior = priority.prior
native = priority.native
read, save = priority.read, priority.save
previous_pair_receipt, previous_aggregate = priority.pair_receipt, priority.aggregate


def normalized(command):
    return prior.normalized([arg for arg in command if arg != FLAG], historical_command=True)


def verify_inputs(run):
    experiment_path, execution_path = (run / name for name in ("experiment.json", "execution.json"))
    metadata, execution = read(experiment_path), read(execution_path)
    wind = int(metadata["wind_from_degrees"])
    off, base_execution, base_receipt, baseline_audit_hash = priority.baseline_for(wind)
    assert run.name == metadata["run_id"] == execution["run_id"] == f"crossing_fan__otranto_{wind:03d}__layline_eta"
    for row in (metadata, execution):
        assert row["layline_mode"] == "eta" and row["compare_waypoint_layline_eta"] is True
        assert row["closest_waypoint_layline"] is True and not row.get("prioritize_waypoint_layline")
    assert metadata["native_hybrid_enabled"] and metadata["native_schedule_veto"]
    assert metadata["radius_m"] == 50. and metadata["dt_s"] == 5.
    command, base_command = metadata["command"], base_execution["command"]
    assert command == execution["command"] == shlex.split((run / "command.txt").read_text())
    assert command.count(FLAG) == 1 and FLAG not in base_command
    assert prior.FLAG not in command and priority.FLAG not in command
    assert normalized(command) == normalized(base_command), run.name
    safety_flags = ("--colreg", "--time-aware-ais", "--native-control-timing", "--ais-candidate-handoff")
    assert all(command.count(flag) == 1 for flag in safety_flags)
    logged_args = prior.route_audit.one_json((run / "run.log").read_text().splitlines(), "Runner arguments: ")
    assert logged_args.count(FLAG) == 1 and all(logged_args.count(flag) == 1 for flag in safety_flags)

    for key in prior.MATCH_FIELDS:
        assert metadata[key] == execution[key] == base_execution[key], (run.name, key)
    for path_key, hash_key in prior.FIXTURE_FIELDS:
        expected = base_execution[hash_key]
        assert metadata[hash_key] == execution[hash_key] == native.digest(metadata[path_key]) == expected
        assert native.digest(base_execution[path_key]) == expected
    assert native.digest(STUDY / "archive_field_fingerprints.json") == metadata["archive_field_fingerprint_sha256"]

    old_sources, sources = base_execution["source_sha256"], execution["source_sha256"]
    assert metadata["source_sha256"] == sources and old_sources.keys() <= sources.keys()
    assert execution["source_unchanged"] and not execution["changed_inputs"]
    assert {key for key in old_sources if old_sources[key] != sources[key]} == SOURCE_CHANGES
    for key, checksum in sources.items():
        assert native.digest(run / "source_snapshot" / key) == native.digest(ROOT / key) == checksum, (run.name, key)
    assert metadata["layline_off_execution_sha256"] == base_receipt["input_verification"]["execution_sha256"]
    assert metadata["layline_off_replay_sha256"] == base_receipt["clearance"]["replay_sha256"]
    for field, name in (("departure_gate_summary_sha256", "departure_gate.json"),
                        ("departure_gate_log_sha256", "departure_gate.jsonl")):
        assert execution[field] == native.digest(run / name)
        assert base_execution[field] == native.digest(off / name)
    gate, base_gate = (read(folder / "departure_gate.json") for folder in (run, off))
    for key in ("monitoring_start_utc", "traffic_epoch_utc", "parameters", "actual_departure_utc", "gate_wait_seconds"):
        assert gate[key] == base_gate[key], (run.name, key)
    first, baseline_first = (next(prior.replay(folder / "navigation.jsonl.gz")) for folder in (run, off))
    assert first["state"] == baseline_first["state"] and first["observation"] == baseline_first["observation"], run.name
    return {
        "case": run.name,
        "mode": "eta",
        "wind_from_degrees": wind,
        "execution_return_code": execution["return_code"],
        "same_invocation_as_verified_off_apart_from_eta_flag_and_run_paths": True,
        "same_forecast_archive_AIS_initial_state_and_launch_clock": True,
        "sources_bound_to_unchanged_execution_snapshot": True,
        "verified_baseline_case": off.name,
        "baseline_case_audit_sha256": baseline_audit_hash,
        "baseline_execution_sha256": base_receipt["input_verification"]["execution_sha256"],
        "baseline_replay_sha256": base_receipt["clearance"]["replay_sha256"],
        "experiment_sha256": native.digest(experiment_path),
        "execution_sha256": native.digest(execution_path),
    }


def audit_policy(run, enabled):
    counts = Counter()
    first = last = minimum_gain = maximum_gain = None
    for frame in prior.replay(run / "navigation.jsonl.gz"):
        nav = frame["navigation"]
        source = nav.get("waypoint_layline_source")
        context = run.name, nav["step"]
        if source is None:
            assert nav.get("waypoint_layline_policy") is None, context
            continue
        assert enabled and nav["waypoint_layline_policy"] == "eta", context
        assert source in {"direct gate approach", "closest reachable tack"}, context
        assert nav["ais_mode"] == "route" and nav["ais_weather_candidate_admitted"] is True, context
        assert nav.get("drift_mode") != "holding", context
        assert not nav.get("forecast_maneuver_commit_active"), context
        assert math.isfinite(nav["waypoint_layline_distance_m"]) and nav["waypoint_layline_distance_m"] >= 0., context
        eta, forecast_eta, gain, radius, steady_eta = (nav[key] for key in ETA_FIELDS)
        assert all(math.isfinite(value) for value in (eta, forecast_eta, gain, radius, steady_eta)), context
        assert radius == 50. and 0. <= eta < forecast_eta <= 2880. + 1e-8, context
        assert 0. < steady_eta <= 2880. + 1e-8, context
        assert gain > 0. and math.isclose(gain, forecast_eta - eta, rel_tol=1e-10, abs_tol=1e-8), context
        assert (nav["waypoint_layline_second_bow_deg"] is None) == (source == "direct gate approach"), context
        if nav["step"] == 0:
            continue
        counts[(nav["phase"], nav["waypoint"], source)] += 1
        record = {key: nav[key] for key in (
            "step", "phase", "waypoint", "heading_reference_deg", "bow_reference_deg",
            "waypoint_layline_source", "waypoint_layline_distance_m",
            "waypoint_layline_second_bow_deg", *ETA_FIELDS)}
        record["fix_utc"] = frame["observation"]["time"]
        if first is None:
            first = record
        last = record
        minimum_gain = gain if minimum_gain is None else min(minimum_gain, gain)
        maximum_gain = gain if maximum_gain is None else max(maximum_gain, gain)
    return {
        "applied_reference_steps": sum(counts.values()),
        "applied_reference_steps_by_leg_and_source": [
            {"phase": phase, "waypoint": mark, "source": source, "steps": steps}
            for (phase, mark, source), steps in counts.items()],
        "all_applied_laylines_beat_finite_forecast_eta_at_50m": True,
        "minimum_applied_eta_advantage_s": minimum_gain,
        "maximum_applied_eta_advantage_s": maximum_gain,
        "first_applied_reference": first,
        "last_applied_reference": last,
        "scope": "Retained replay commands after AIS admission, verifying reported ETA comparisons for the 50 m gate.",
    }


def audit_case(run):
    with patch.multiple(prior, STUDY=STUDY, verify_inputs=verify_inputs, audit_policy=audit_policy):
        receipt = prior.audit_case(run)
    receipt["audit_source_sha256"] = native.digest(__file__)
    return receipt


def pair_receipt(receipt):
    pair = previous_pair_receipt(receipt)
    result = {key.replace("priority_", "eta_", 1): value for key, value in pair.items()}
    summary = receipt["layline_policy"]
    result.update({
        "eta_all_applied_laylines_beat_finite_forecast_eta_at_50m":
            summary["all_applied_laylines_beat_finite_forecast_eta_at_50m"],
        "eta_minimum_applied_eta_advantage_s": summary["minimum_applied_eta_advantage_s"],
        "eta_maximum_applied_eta_advantage_s": summary["maximum_applied_eta_advantage_s"],
    })
    return result


def aggregate():
    summary = previous_aggregate()
    comparisons = read(STUDY / "independent_comparison.json")
    comparisons["sign_convention"] = "ETA-gated layline minus off, with negative faster."
    save(STUDY / "independent_comparison.json", comparisons)
    summary["pinned_priority_audit_sha256"] = PRIORITY_SHA256
    summary["applied_eta_layline_references"] = sum(
        row["eta_applied_layline_reference_steps"] for row in comparisons["pairs"])
    save(STUDY / "independent_final_summary.json", summary)
    return summary


def main():
    assert native.digest(PRIORITY_AUDIT) == PRIORITY_SHA256
    with patch.multiple(priority, __file__=__file__, __doc__=__doc__, STUDY=STUDY, ARMS=ARMS, audit_case=audit_case,
                        pair_receipt=pair_receipt, aggregate=aggregate):
        priority.main()


if __name__ == "__main__":
    main()
