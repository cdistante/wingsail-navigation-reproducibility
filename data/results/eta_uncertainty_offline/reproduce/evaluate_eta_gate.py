"""Fit and calibrate a conservative ETA acceptance rule on locked native branches."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize


STUDY = Path(__file__).resolve().parents[1]
ABLAT = STUDY.with_name("temporal_embedding_native")
SEASONAL = STUDY.with_name("seasonal_native_training")
sys.path.insert(0, str(SEASONAL / "reproduce"))
sys.path.insert(0, str(ABLAT / "reproduce"))

from ablation_ranker import eligible_mask
from seasonal_ranker import MINIMUM_MODEL_GAIN_S, RESIDUAL_CAP_S, RIDGE, Ranker


SEEDS = (7309, 8779, 10921, 13219, 16381)
HUBER_RADIUS_S = 60.
OVERPREDICTION_WEIGHT = 3.
SPREAD_FLOOR_S = 25.
NOMINAL_COVERAGE = .8


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_inputs():
    pin = json.loads((ABLAT / "models/fit_receipt.json").read_text())
    manifest = SEASONAL / "manifest_expanded.json"
    if digest(manifest) != pin["protocol"]["split_manifest_sha256"]:
        raise ValueError("The original seasonal split manifest changed.")
    cases = {case["id"]: case for case in json.loads(manifest.read_text())["scenarios"]}
    split_rows = {}
    for split in ("train", "validation", "test"):
        path = ABLAT / "branches" / f"{split}.jsonl.gz"
        if digest(path) != pin["labels_sha256"][split]:
            raise ValueError(f"Changed immutable {split} branches.")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            records = list(map(json.loads, stream))
        for row in records:
            case = cases[row["scenario"]]
            if row["split"] != split or case["split"] != split:
                raise ValueError("A branch was assigned to the wrong split.")
            row["group"] = f"{case['archive_block']}/{case['route_id']}/{case['departure_utc']}"
        split_rows[split] = records
    checkpoint = ABLAT / "models/explicit.npz"
    if digest(checkpoint) != pin["models_sha256"]["explicit"]:
        raise ValueError("Changed explicit baseline model.")
    return split_rows, Ranker.load(checkpoint), pin


def candidates(state):
    branches = state["branches"]
    features = np.asarray([b["features"] for b in branches], dtype=float)
    forecasts = np.asarray([b["forecast_s"] for b in branches], dtype=float)
    if features.shape != (len(branches), 34):
        raise ValueError("Expected the shared 34-feature native representation.")
    legal = eligible_mask(forecasts, np.ones(len(branches), dtype=bool), 0)
    if not legal.all() or len(branches) < 2:
        raise ValueError("A frozen branch start contains an ineligible comparison.")
    return branches, features, forecasts


def safe_capture(branch):
    return branch["matched_initial_action"] and branch["actual_capture"] and branch["audited_safe"]


def fit_huber(states, overprediction_weight):
    """Estimate candidate corrections with the specified one-sided loss weight."""
    incumbents, alternatives, targets, weights = [], [], [], []
    for state in states:
        branches, features, forecasts = candidates(state)
        incumbent = branches[0]
        if not safe_capture(incumbent):
            continue
        safe = [i for i in range(1, len(branches)) if safe_capture(branches[i])]
        for i in safe:
            actual = incumbent["duration_s"] - branches[i]["duration_s"]
            forecast = forecasts[0] - forecasts[i]
            incumbents.append(features[0])
            alternatives.append(features[i])
            targets.append(actual - forecast)
            weights.append(1. / len(safe))
    base = np.asarray(incumbents)
    alt = np.asarray(alternatives)
    target = np.asarray(targets)
    weights = np.asarray(weights)
    scale = np.maximum(np.std(np.vstack((base, alt)), axis=0), .05)
    design = (base - alt) / scale

    def value_and_gradient(beta):
        error = design @ beta - target
        absolute = np.abs(error)
        clipped = np.minimum(absolute, HUBER_RADIUS_S)
        side_weight = weights * np.where(error > 0., overprediction_weight, 1.)
        huber = clipped ** 2 / 2. + HUBER_RADIUS_S * (absolute - clipped)
        value = side_weight @ huber + RIDGE / 2. * (beta @ beta)
        derivative = design.T @ (side_weight * np.clip(error, -HUBER_RADIUS_S, HUBER_RADIUS_S))
        return value, derivative + RIDGE * beta

    result = minimize(value_and_gradient, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B",
                      options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-7})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Asymmetric residual fit failed: {result.message}")
    return Ranker(scale, result.x), len(targets)


def fit_ensemble(states, overprediction_weight):
    groups = defaultdict(list)
    for state in states:
        groups[state["group"]].append(state)
    keys = sorted(groups)
    models, receipts = [], []
    for seed in SEEDS:
        sampled = np.random.default_rng(seed).choice(keys, size=len(keys), replace=True)
        model, pairs = fit_huber([state for key in sampled for state in groups[key]], overprediction_weight)
        models.append(model)
        receipts.append({"seed": seed, "groups_drawn": sampled.tolist(), "safe_pairs_with_multiplicity": pairs})
    return models, receipts


def predicted_gains(state, models):
    branches, features, forecasts = candidates(state)
    scores = np.stack([forecasts + model.corrections(features) for model in models])
    return scores[:, :1] - scores[:, 1:]


def observed_gain(incumbent, alternative):
    if safe_capture(incumbent) and safe_capture(alternative):
        return incumbent["duration_s"] - alternative["duration_s"]
    return None


def estimates(state, models):
    members = predicted_gains(state, models)
    mean = members.mean(axis=0)
    if len(members) == 1:
        return mean, np.full_like(mean, SPREAD_FLOOR_S)
    return mean, np.maximum(members.std(axis=0, ddof=1), SPREAD_FLOOR_S)


def calibrate(states, models, *, include_failures):
    """Calibrate over all eligible alternatives with a maximum per departure."""
    group_scores = {}
    used = 0
    failure_targets = 0
    for state in states:
        incumbent, *alternatives = state["branches"]
        if not safe_capture(incumbent):
            raise ValueError("The calibration set needs safe incumbent captures for the restricted gain.")
        mean, spread = estimates(state, models)
        for alternative, prediction, scale in zip(alternatives, mean, spread, strict=True):
            observed = observed_gain(incumbent, alternative)
            if observed is None:
                if not include_failures:
                    continue
                # Restricted outcome assigns each failed alternative the branch
                # horizon. This is a finite-horizon utility, separate from ETA.
                observed = incumbent["duration_s"] - state["cap_s"]
                failure_targets += 1
            score = float((prediction - observed) / scale)
            group = state["group"]
            group_scores[group] = max(group_scores.get(group, -math.inf), score)
            used += 1
    n = len(group_scores)
    order = math.ceil((n + 1) * NOMINAL_COVERAGE)
    multiplier = max(0., sorted(group_scores.values())[order - 1]) if order <= n else None
    return {"multiplier": multiplier, "nominal_coverage": NOMINAL_COVERAGE, "order_statistic": order,
            "group_scores": group_scores, "calibration_pairs": used, "restricted_failure_targets": failure_targets}


def evaluate(states, models, multiplier):
    selected = []
    oracle_opportunities = 0
    group_coverage = defaultdict(list)
    errors = []
    for state in states:
        incumbent, *alternatives = state["branches"]
        actuals = [observed_gain(incumbent, alternative) for alternative in alternatives]
        oracle_opportunities += any(value is not None and value > MINIMUM_MODEL_GAIN_S for value in actuals)
        mean, spread = estimates(state, models)
        score = mean - multiplier * spread if multiplier is not None else np.full_like(mean, -np.inf)
        errors_at_start = [abs(prediction - actual) for prediction, actual in zip(mean, actuals, strict=True)
                           if actual is not None]
        if errors_at_start:
            errors.append(float(np.mean(errors_at_start)))
        if safe_capture(incumbent):
            for actual, lower in zip(actuals, score, strict=True):
                restricted = incumbent["duration_s"] - state["cap_s"] if actual is None else actual
                group_coverage[state["group"]].append(bool(lower <= restricted))
        index = int(np.argmax(score))
        if score[index] > MINIMUM_MODEL_GAIN_S:
            actual = actuals[index]
            selected.append({
                "state_id": state["state_id"], "group": state["group"],
                "candidate": alternatives[index]["action_id"], "mean_gain_s": float(mean[index]),
                "spread_s": float(spread[index]), "gate_score_s": float(score[index]), "safe_gain_s": actual,
                "good_switch": actual is not None and actual > MINIMUM_MODEL_GAIN_S,
            })
    true_positives = sum(row["good_switch"] for row in selected)
    return {"states": len(states), "eligible_groups": len({state["group"] for state in states}),
            "selected": len(selected), "true_positives": true_positives,
            "false_positives": len(selected) - true_positives,
            "precision": true_positives / len(selected) if selected else None,
            "oracle_states": oracle_opportunities, "recall_of_oracle_states": (
                true_positives / oracle_opportunities if oracle_opportunities else None),
            "point_gain_mae_s": float(np.mean(errors)) if errors else None,
            "covered_complete_groups": sum(all(group) for group in group_coverage.values()),
            "evaluable_coverage_groups": len(group_coverage),
            "selected_actions": selected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STUDY)
    args = parser.parse_args()
    branches, baseline, pin = load_inputs()
    trained, draws = fit_ensemble(branches["train"], OVERPREDICTION_WEIGHT)
    symmetric, symmetric_draws = fit_ensemble(branches["train"], 1.)
    if draws != symmetric_draws:
        raise ValueError("The Huber controls must use matched bootstrap draws.")
    calibration = calibrate(branches["validation"], trained, include_failures=True)
    capture_only = calibrate(branches["validation"], trained, include_failures=False)
    frozen = calibration["multiplier"]
    evaluations = {
        "symmetric_point": ([baseline], 0.),
        "symmetric_huber_point": (symmetric, 0.),
        "asymmetric_point": (trained, 0.),
        "asymmetric_lower_bound": (trained, frozen),
        "asymmetric_lower_bound_captures_only": (trained, capture_only["multiplier"]),
    }
    results = {
        "inputs": {"labels_sha256": pin["labels_sha256"], "baseline_sha256": pin["models_sha256"]["explicit"],
                   "source_sha256": digest(Path(__file__))},
        "protocol": {"huber_radius_s": HUBER_RADIUS_S, "overprediction_weight": OVERPREDICTION_WEIGHT,
                     "spread_floor_s": SPREAD_FLOOR_S, "strict_gate_s": MINIMUM_MODEL_GAIN_S,
                     "training_starts": len(branches["train"]),
                     "distinct_training_groups": len({state["group"] for state in branches["train"]})},
        "ensemble": draws, "calibration": calibration, "capture_only_calibration": capture_only,
        "april": {name: evaluate(branches["validation"], models, multiplier)
                  for name, (models, multiplier) in evaluations.items()},
        "october": {name: evaluate(branches["test"], models, multiplier)
                    for name, (models, multiplier) in evaluations.items()},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "results.json"
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    for name, summary in results["october"].items():
        print(f"{name} selected={summary['selected']} true={summary['true_positives']} "
              f"false={summary['false_positives']} precision={summary['precision']}")
    print(f"April group calibration multiplier = {frozen}")


if __name__ == "__main__":
    main()
