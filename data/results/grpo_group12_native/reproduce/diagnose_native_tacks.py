"""Check the accepted tack threshold and replay the original two changed decisions."""
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from dataclasses import asdict
import gzip
import json
from math import isclose
from pathlib import Path
import sys

import numpy as np


STUDY = Path(__file__).resolve().parents[1]
PRIOR = STUDY.with_name("grpo_dropout_native")
METHODS = ("grpo_progress", "grpo_dropout")
sys.path.insert(0, str(STUDY.with_name("grpo_progress_efficient_ais") / "reproduce"))

import run_efficient_retest as efficient
from native_grpo import MINIMUM_CORRECTED_GAIN_S, TRUST_REGRET_S, NativeGRPO


seasonal = efficient.seasonal


def traces(study, method):
    directory = study / "native" / method
    receipt = json.loads((directory / "run.json").read_text())
    for name, checksum in receipt["trace_sha256es"].items():
        path = directory / name
        if seasonal.digest(path) != checksum:
            raise ValueError(f"Trace changed: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            yield path.name.removesuffix(f"__{method}.json.gz"), json.load(stream)


def summarize(study, method):
    rows = [{**row, "scenario": scenario} for scenario, bundle in traces(study, method)
            for row in bundle["strategy_evaluations"]]
    scored = [row for row in rows if row["corrected_gain_s"] is not None]
    return {
        "decisions": len(rows), "by_status": dict(Counter(row["status"] for row in rows)),
        "scored_decisions": len(scored),
        "positive_gains": sum(row["corrected_gain_s"] > 0 for row in scored),
        "above_25s_gain": sum(row["corrected_gain_s"] > MINIMUM_CORRECTED_GAIN_S for row in scored),
        "top_corrected_gain": max(scored, key=lambda row: row["corrected_gain_s"], default=None),
    }


def replay_selected():
    seasonal.init(efficient.MANIFEST, None, False)
    cases = {case["id"]: case for case in seasonal.INPUTS["scenarios"]}
    dropout_path = PRIOR / "training" / "grpo_dropout_selected.npz"
    actors = {"grpo_progress": NativeGRPO.load(efficient.CHECKPOINT),
              "grpo_dropout": NativeGRPO.load(dropout_path)}
    results = []
    for scenario, trace in traces(PRIOR, "grpo_progress"):
        selected = [row for row in trace["strategy_evaluations"] if row["status"] == "selected"]
        if not selected:
            continue
        engine, initial, goals, _, cap = efficient.scenario_parts(cases[scenario])
        marks = {mark.name: mark for mark in goals}
        for event in selected:
            if event["elapsed_s"]:
                fixed, = (row for row in trace["states"]
                          if isclose(row["elapsed_s"], event["elapsed_s"], rel_tol=0., abs_tol=1e-8))
                state = type(initial)(**{key: fixed[key] for key in asdict(initial)})
            else:
                state = initial
            when = np.datetime64(event["utc"].removesuffix("Z"), "ns")
            elapsed = event["elapsed_s"]
            decision = engine.decision(state, marks[event["goal"]], when, elapsed)
            arms = {method: actor.select(decision, dynamics=engine.dynamics, remaining_s=cap - elapsed,
                                        capture_radius_m=engine.config.capture_radius_m,
                                        strength=efficient.STRENGTH)
                    for method, actor in actors.items()}
            original = arms["grpo_progress"]
            if not (
                decision.physics_index == event["physics_index"] and
                original.index == event["requested_index"] and original.status == event["status"] and
                isclose(original.corrected_gain_s, event["corrected_gain_s"], rel_tol=0., abs_tol=1e-6)
            ):
                raise ValueError(f"The original learned decision did not replay at {scenario}/{elapsed}.")
            results.append({"scenario": scenario, "elapsed_s": elapsed, "utc": event["utc"], "goal": event["goal"],
                            "same_state_scores": {method: asdict(arm) for method, arm in arms.items()}})
    return sorted(results, key=lambda row: row["utc"])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--prior-only", action="store_true")
    args = parser.parse_args()
    result = {"thresholds": {"minimum_gain_s_strict": MINIMUM_CORRECTED_GAIN_S,
                             "maximum_native_eta_regret_s": TRUST_REGRET_S},
              "group6": {method: summarize(PRIOR, method) for method in METHODS},
              "original_grpo_tack_changes_replayed_with_both_actors": replay_selected()}
    if not args.prior_only:
        result["group12"] = {method: summarize(STUDY, method) for method in METHODS}
    seasonal.save_json(STUDY / "tack_diagnosis.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
