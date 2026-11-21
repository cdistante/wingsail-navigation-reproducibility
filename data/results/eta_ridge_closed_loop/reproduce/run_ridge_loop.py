"""Replay the frozen Stage 1 compact ridge selector in the causal Fossen benchmark."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np


STUDY = Path(__file__).resolve().parents[1]
ABLATION = STUDY.with_name("temporal_embedding_native")
STAGE1 = STUDY.with_name("eta_stage1_expanded")
METHOD = "explicit"
DISPLAY_NAME = "Stage 1 compact explicit ridge"
sys.path.insert(0, str(ABLATION / "reproduce"))
sys.path.insert(0, str(STAGE1 / "reproduce"))

import run_temporal_ablation as baseline
from ablation_report import reconcile_protocol
from compact_features import EXPLICIT_NAMES, feature_rows
from seasonal_ranker import MINIMUM_MODEL_GAIN_S, RESIDUAL_CAP_S, TRUST_REGRET_S, Ranker

seasonal = baseline.seasonal


@dataclass(frozen=True)
class CompactRidge:
    model: Ranker

    def corrections(self, features):
        return self.model.corrections(feature_rows(features))


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as data:
        scale = data["scales"]
        beta = data["coefficients"]
    width = (1, len(EXPLICIT_NAMES))
    if (scale.shape != width or beta.shape != width or np.any(scale <= 0.)
            or not np.isfinite(scale).all() or not np.isfinite(beta).all()):
        raise ValueError("Invalid Stage 1 compact explicit ridge checkpoint.")
    return CompactRidge(Ranker(scale[0], beta[0]))


def worker_sources():
    sources = (
        Path(__file__).resolve(),
        STAGE1 / "reproduce" / "compact_features.py",
        ABLATION / "reproduce" / "run_temporal_ablation.py",
        ABLATION / "reproduce" / "causal_weather.py",
        ABLATION / "reproduce" / "ablation_ranker.py",
    )
    return {str(path.relative_to(STUDY.parent)): seasonal.digest(path) for path in sources}


def verified_protocol():
    """Authenticate the controller, simulator, and frozen physics reference."""
    manifest, current = baseline.locked_protocol()
    current = json.loads(json.dumps(current, default=seasonal.json_value))
    frozen_fit = json.loads((ABLATION / "models" / "fit_receipt.json").read_text())
    reconciliation = reconcile_protocol(frozen_fit["protocol"], current, ABLATION)
    physics = ABLATION / "native" / "physics"
    input_path, run_path, episodes_path = (physics / name for name in ("inputs.json", "run.json", "episodes.csv"))
    prior = json.loads(input_path.read_text())
    prior_run = json.loads(run_path.read_text())
    physics_hashes = {path.name: seasonal.digest(path) for path in (input_path, run_path, episodes_path)}
    if (prior["scenarios"] != current["test_ids"] or prior["method"] != "physics"
            or prior["model_sha256"] is not None or prior["probe"]
            or any(prior[key] != value for key, value in frozen_fit["protocol"].items())
            or prior_run["inputs_sha256"] != physics_hashes["inputs.json"]
            or prior_run["episodes_sha256"] != physics_hashes["episodes.csv"]):
        raise ValueError("The frozen physics run or its inputs have changed.")
    for relative, expected in prior_run["trace_sha256es"].items():
        if seasonal.digest(physics / relative) != expected:
            raise ValueError(f"Frozen physics trace changed: {relative}")
    if len(prior_run["trace_sha256es"]) != 12:
        raise ValueError("Expected all twelve archived physics traces.")

    stage_fit = STAGE1 / "fit"
    fit_path = stage_fit / "fit_receipt.json"
    results_path = stage_fit / "results.json"
    checkpoint = stage_fit / "explicit_ridge.npz"
    fit = json.loads(fit_path.read_text())
    results = json.loads(results_path.read_text())
    fixed_state = results["october"]["explicit_ridge_point"]
    checkpoint_hash = seasonal.digest(checkpoint)
    fit_hash = seasonal.digest(fit_path)
    if (
        fit["manifest_sha256"] != current["split_manifest_sha256"]
        or fit["protocol"]["explicit_names"] != list(EXPLICIT_NAMES)
        or fit["protocol"]["strict_gain_s"] != MINIMUM_MODEL_GAIN_S
        or fit["protocol"]["native_eta_regret_s"] != TRUST_REGRET_S
        or fit["protocol"]["residual_cap_s"] != RESIDUAL_CAP_S
        or fit["sources_sha256"]["compact_features.py"] != seasonal.digest(STAGE1 / "reproduce" / "compact_features.py")
        or fit["models"]["explicit"]["ridge"]["checkpoint_sha256"] != checkpoint_hash
        or results["fit_receipt_sha256"] != fit_hash
        or results["test_digest_sha256"] != frozen_fit["labels_sha256"]["test"]
        or (fixed_state["selected"], fixed_state["true_positives"], fixed_state["false_positives"]) != (3, 2, 1)
    ):
        raise ValueError("Expected the original explicit ridge model with two of three fixed-state wins.")
    load_checkpoint(checkpoint)
    return manifest, {
        **current, "display_name": DISPLAY_NAME, "checkpoint_sha256": checkpoint_hash,
        "fit_receipt_sha256": fit_hash, "fixed_state_results_sha256": seasonal.digest(results_path),
        "fixed_state_proposed_switches": 3, "fixed_state_safe_gains_over_25_s": 2,
        "frozen_physics": physics_hashes, "physics_provenance": reconciliation,
        "worker_sources_sha256": worker_sources(),
    }, checkpoint


def init_worker(checkpoint, expected_sha256, sources):
    if worker_sources() != sources or seasonal.digest(checkpoint) != expected_sha256:
        raise RuntimeError("The worker sources or the frozen ridge model changed.")
    baseline.init_worker()
    # Reuse the exact native explicit selector, gate, and telemetry path.
    baseline.MODELS[METHOD] = load_checkpoint(checkpoint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=STUDY / "native")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Choose at least one worker.")
    manifest, protocol, checkpoint = verified_protocol()
    specs = [row for row in manifest["scenarios"] if row["split"] == "test"]
    scenario_ids = [row["id"] for row in specs]
    if len(scenario_ids) != 12 or scenario_ids != protocol["test_ids"]:
        raise ValueError("The benchmark must include the frozen twelve October cases.")
    out = args.output.resolve()
    directory = out / METHOD
    (directory / "telemetry").mkdir(parents=True, exist_ok=False)
    receipt = {
        **protocol, "method": METHOD, "scenarios": scenario_ids, "probe": False,
        "model_sha256": protocol["checkpoint_sha256"], "workers": min(args.workers, len(specs)),
    }
    inputs = directory / "inputs.json"
    seasonal.save_json(inputs, receipt)
    initargs = (str(checkpoint), receipt["model_sha256"], receipt["worker_sources_sha256"])
    found = {}
    with ProcessPoolExecutor(max_workers=receipt["workers"], mp_context=multiprocessing.get_context("spawn"),
                             initializer=init_worker, initargs=initargs) as pool:
        futures = [pool.submit(baseline.mission_job, (spec, METHOD, str(out))) for spec in specs]
        for future in as_completed(futures):
            row = future.result()
            found[row["scenario"]] = row
            print(f"{row['scenario']} {row['capture_count']}/{row['expected_captures']} "
                  f"{row['status']} {row['duration_s']:.2f}s applied={row['learned_changes']}", flush=True)
    rows = [found[scenario] for scenario in scenario_ids]
    if worker_sources() != receipt["worker_sources_sha256"] or seasonal.digest(checkpoint) != receipt["model_sha256"]:
        raise ValueError("Sources or the checkpoint changed while episodes were running.")
    episodes = directory / "episodes.csv"
    seasonal.save_csv(episodes, rows)
    seasonal.save_json(directory / "run.json", {
        "method": METHOD, "inputs_sha256": seasonal.digest(inputs),
        "episodes_sha256": seasonal.digest(episodes), "summary": seasonal.summary(rows),
        "trace_sha256es": {row["trajectory"]: seasonal.digest(directory / row["trajectory"]) for row in rows},
    })


if __name__ == "__main__":
    main()
