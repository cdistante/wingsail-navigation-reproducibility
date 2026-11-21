"""Replay both dropout-comparison arms using the locked efficient AIS audit."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing
from pathlib import Path
import sys


STUDY = Path(__file__).resolve().parents[1]
V2 = STUDY.parents[2]
EFFICIENT = STUDY.with_name("grpo_progress_efficient_ais")
sys.path.insert(0, str(EFFICIENT / "reproduce"))

import run_efficient_retest as efficient
from audit_efficient_clearance import audit_efficient_episode, clearance_flag, historical_rejection, require


METHODS = ("grpo_progress", "grpo_dropout")
CHECKPOINTS = {
    "grpo_progress": efficient.CHECKPOINT,
    "grpo_dropout": STUDY / "training/grpo_dropout_selected.npz",
}


def locked_jobs(native):
    """Authenticate the matched run receipts against the previous guard protocol."""
    seasonal = efficient.seasonal
    roster, protocol = efficient.locked_protocol()
    scenarios = {spec["id"]: spec for spec in roster}
    require(len(scenarios) == len(roster) == 12, "Expected all twelve locked test cases.")
    original = EFFICIENT / "native/grpo_progress/inputs.json"
    baseline = json.loads(original.read_text())
    for key in ("guard", "split_manifest_sha256", "forcing_sha256", "bathymetry_sha256"):
        require(baseline[key] == protocol[key], f"Historical efficient AIS protocol changed: {key}.")

    jobs, sources = [], {}
    for method, checkpoint in CHECKPOINTS.items():
        directory = native / method
        inputs_path = directory / "inputs.json"
        episodes_path = directory / "episodes.csv"
        receipt_path = directory / "run.json"
        inputs = json.loads(inputs_path.read_text())
        receipt = json.loads(receipt_path.read_text())
        inputs_digest = seasonal.digest(inputs_path)
        episodes_digest = seasonal.digest(episodes_path)
        require(inputs["method"] == receipt["method"] == method
                and receipt["inputs_sha256"] == inputs_digest
                and receipt["episodes_sha256"] == episodes_digest
                and not inputs["probe"], f"{method}: invalid canonical receipt.")
        require(inputs["scenarios"] == list(scenarios), f"{method}: invalid locked case order.")
        for key in ("guard", "split_manifest_sha256", "forcing_sha256", "bathymetry_sha256",
                    "package_versions", "adapter_sha256", "legacy_support_sha256", "residual_strength"):
            require(inputs[key] == baseline[key], f"{method}: {key} differs from efficient AIS.")
        for path, digest in protocol["source_sha256"].items():
            require(inputs["source_sha256"][path] == digest == baseline["source_sha256"][path],
                    f"{method}: locked environment, simulator or guard source changed: {path}.")
        checkpoint_digest = seasonal.digest(checkpoint)
        training_path = checkpoint.with_name("training_results.json")
        training_digest = seasonal.digest(training_path)
        training = json.loads(training_path.read_text())
        require((V2 / inputs["checkpoint"]).resolve() == checkpoint.resolve()
                and inputs["checkpoint_sha256"] == training["selected_checkpoint_sha256"] == checkpoint_digest
                and inputs["training_receipt_sha256"] == training_digest
                and inputs["selected_seed"] == training["selected_seed"]
                and inputs["selected_update"] == training["selected_update"],
                f"{method}: checkpoint does not match the recorded validation selection.")

        with episodes_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        require([row["scenario"] for row in rows] == list(scenarios),
                f"{method}: recorded case set or order differs from the roster.")
        trace_digests = receipt["trace_sha256es"]
        require(len(trace_digests) == len(rows), f"{method}: wrong number of trace digests.")
        for row in rows:
            scenario = row["scenario"]
            name = f"telemetry/{scenario}__{method}.json.gz"
            require(row["method"] == method and row["trajectory"] == name,
                    f"{method}/{scenario}: noncanonical trace.")
            jobs.append((scenarios[scenario], row, directory / name, trace_digests[name]))
        sources[method] = {
            "inputs_sha256": inputs_digest, "episodes_sha256": episodes_digest,
            "run_receipt_sha256": seasonal.digest(receipt_path),
            "checkpoint": str(checkpoint.relative_to(V2)), "checkpoint_sha256": checkpoint_digest,
            "training_receipt_sha256": training_digest,
        }
    return scenarios, jobs, sources


def summarize(results, method):
    cases = [case for case in results if case["method"] == method]
    counts = Counter(clearance_flag(case) for case in cases)
    return {
        "clear_cases_with_motion": counts["clear"],
        "violated_cases_with_motion": counts["violated"],
        "cases_without_applied_motion": counts["no_applied_path"],
        "minimum_clearance_m": {
            kind: min((case["minimum_clearance"][kind]["margin_m"] for case in cases
                       if case["minimum_clearance"][kind] is not None), default=None)
            for kind in ("truth", "report", "guard")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, default=STUDY / "native")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path, default=STUDY / "independent_clearance.json")
    args = parser.parse_args()
    require(args.workers > 0, "At least one audit worker is required.")
    scenarios, jobs, sources = locked_jobs(args.native)
    seasonal, manifest = efficient.seasonal, efficient.MANIFEST
    seasonal.init(manifest, None, False)
    old_rejection = historical_rejection(scenarios)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=seasonal.init, initargs=(manifest, None, False)) as pool:
        results = list(pool.map(audit_efficient_episode, jobs, chunksize=1))
    indexed = {(record["method"], record["scenario"]): record for record in results}
    unsafe = [{key: record[key] for key in ("scenario", "method", "nonpositive_control_intervals")}
              for record in results if any(record["nonpositive_control_intervals"].values())]
    seasonal.save_json(args.output, {
        "status": "failed_clearance" if unsafe else "verified",
        "scope": "Independent held-forcing Fossen re-integration at 0.05 s, analytical targets, "
                 "derived AIS report timing, both original full envelopes and the expanded report guard. "
                 "Each first logged forecast decision is replayed without target truth or later observations.",
        "manifest_sha256": seasonal.digest(manifest),
        "auditor_sha256": seasonal.digest(__file__),
        "efficient_auditor_sha256": seasonal.digest(EFFICIENT / "reproduce/audit_efficient_clearance.py"),
        "guard_sha256": seasonal.digest(efficient.GUARD_SOURCE),
        "run_sources": sources,
        "cases_per_arm": len(scenarios),
        "applied_controls": sum(case["applied_controls"] for case in results),
        "integration_substeps": sum(case["integration_substeps"] for case in results),
        "first_forecast_decisions_replayed": sum(case["first_forecast_causal_replay"] is not None
                                                 for case in results),
        "historical_breach_replay": old_rejection,
        "summary": {method: summarize(results, method) for method in METHODS},
        "paired_clearance_flags": {
            scenario: {method: clearance_flag(indexed[method, scenario]) for method in METHODS}
            for scenario in scenarios
        },
        "unsafe_applied_cases": unsafe,
        "cases": results,
    })
    require(not unsafe, f"Clearance or guard violations found. See {args.output}.")
    print(f"Verified full swept clearance on {len(results)} native traces. {args.output}", flush=True)


if __name__ == "__main__":
    main()
