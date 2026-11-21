"""Fit compact ETA models and calibrate on April before evaluating October."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize


RESULTS = Path(__file__).resolve().parents[2]
V2 = RESULTS.parents[1]
for study in ("seasonal_native_training", "eta_uncertainty_offline", "learned_weather_ais_policy"):
    sys.path.insert(0, str(RESULTS / study / "reproduce"))
sys.path.insert(0, str(V2))

from compact_features import (
    EXPLICIT_NAMES, LATENT_NAMES, feature_rows, fit_decoder, latent_rows, raw_embeddings, weather_targets,
)
from evaluate_eta_gate import (
    HUBER_RADIUS_S, NOMINAL_COVERAGE, OVERPREDICTION_WEIGHT, SEEDS, SPREAD_FLOOR_S, digest,
)
from seasonal_ranker import MINIMUM_MODEL_GAIN_S, RESIDUAL_CAP_S, RIDGE, TRUST_REGRET_S, Ranker
from weather_embedding import WeatherEmbedding


FROZEN = RESULTS / "temporal_embedding_native"
WEATHER_ENCODER = RESULTS / "learned_weather_ais_policy" / "fresh_weather_encoder.pt"
MANIFEST = RESULTS / "seasonal_native_training" / "manifest_expanded.json"
REPRESENTATIONS = ("explicit", "explicit_weather")


@dataclass(frozen=True)
class Start:
    state_id: str
    scenario: str
    group: str
    cap_s: float
    branches: tuple
    forecasts: np.ndarray
    features: dict


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_rows(path, split, manifest, expected_hash=None):
    sha256 = digest(path)
    if expected_hash is not None and sha256 != expected_hash:
        raise ValueError(f"Unexpected {split} branch input checksum for {path}.")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = list(map(json.loads, stream))
    if not rows:
        raise ValueError(f"The {split} branch file is empty.")
    ids = set()
    allowed_splits = {"calibration", "validation"} if split == "calibration" else {split}
    for row in rows:
        if row["split"] not in allowed_splits or row["state_id"] in ids:
            raise ValueError(f"Wrong split or duplicate decision start: {row['state_id']}.")
        ids.add(row["state_id"])
        case = manifest.get(row["scenario"])
        if case is not None and case["split"] not in allowed_splits:
            raise ValueError(f"Manifest split conflict at {row['scenario']}.")
        if not row.get("group") and case is not None:
            row["group"] = "/".join(case[key] for key in ("archive_block", "route_id", "departure_utc"))
        if not isinstance(row.get("group"), str) or not row["group"]:
            raise ValueError(f"Missing canonical weather or departure group at {row['state_id']}.")
        if split == "test" and (case is None or case["archive_block"] != "autumn"):
            raise ValueError(f"Unregistered October benchmark row: {row['state_id']}.")
        accepted_blocks = {"winter", "summer"} if split == "train" else (
            {"spring"} if split == "calibration" else {"autumn"}
        )
        if row.get("block", case and case["archive_block"]) not in accepted_blocks:
            raise ValueError(f"Wrong weather block for {split}: {row['state_id']}.")
    if sha256 != digest(path):
        raise ValueError(f"The {split} file changed while it was being read.")
    return rows, sha256


def prepare(rows, embeddings, decoder):
    prepared = []
    for row, embedding in zip(rows, embeddings, strict=True):
        branches = tuple(row["branches"])
        if not branches:
            continue
        forecasts = np.array([float(b["forecast_s"]) if b["forecast_s"] is not None else np.inf for b in branches])
        eligible = np.array([b.get("allowed", True) for b in branches]) & np.isfinite(forecasts)
        if not eligible[0]:
            continue
        eligible &= forecasts <= forecasts[0] + TRUST_REGRET_S
        if np.count_nonzero(eligible) < 2:
            continue
        explicit = np.asarray(feature_rows(row), dtype=float)
        latent = np.asarray(latent_rows(row, embedding, decoder), dtype=float)
        if (explicit.shape != (len(branches), len(EXPLICIT_NAMES))
                or latent.shape != (len(branches), len(LATENT_NAMES))
                or not np.isfinite(explicit).all() or not np.isfinite(latent).all()):
            raise ValueError(f"Invalid compact candidate representation at {row['state_id']}.")
        cap = float(row["cap_s"])
        if not np.isfinite(cap) or cap <= 0.:
            raise ValueError(f"Invalid branch horizon at {row['state_id']}.")
        features = {"explicit": explicit[eligible], "explicit_weather": np.column_stack((explicit, latent))[eligible]}
        selected = tuple(branch for branch, kept in zip(branches, eligible, strict=True) if kept)
        prepared.append(Start(row["state_id"], row["scenario"], row["group"], cap,
                              selected, forecasts[eligible], features))
    return prepared


def outcome(branch):
    if not branch["matched_initial_action"]:
        return "unapplied"
    if branch.get("chart_violations", branch.get("applied_chart_violations", 0)):
        return "chart_violation"
    margins = [branch.get(name) for name in ("min_report_margin_m", "min_truth_margin_m")]
    if any(value is not None and value <= 0. for value in margins):
        return "ais_violation"
    if branch["actual_capture"]:
        duration = branch["duration_s"]
        margin_audited = all(value is not None and np.isfinite(value) for value in margins)
        if (branch["audited_safe"] and margin_audited and duration is not None
                and np.isfinite(duration) and duration >= 0.):
            return "safe_capture"
        return "unaudited_capture"
    return branch.get("terminal_status") or "other_failure"


def actual_gain(incumbent, alternative):
    if outcome(incumbent) == outcome(alternative) == "safe_capture":
        return float(incumbent["duration_s"] - alternative["duration_s"])
    return None


def learning_rows(starts, representation):
    incumbents, alternatives, targets, weights = [], [], [], []
    for start in starts:
        incumbent, *candidates = start.branches
        safe = [(i, gain) for i, candidate in enumerate(candidates, 1)
                if (gain := actual_gain(incumbent, candidate)) is not None]
        features = start.features[representation]
        for index, gain in safe:
            incumbents.append(features[0])
            alternatives.append(features[index])
            targets.append(gain - (start.forecasts[0] - start.forecasts[index]))
            weights.append(1. / len(safe))
    if not targets:
        raise ValueError("No jointly safe capture targets in the training sample.")
    base, alternative = np.asarray(incumbents), np.asarray(alternatives)
    scale = np.maximum(np.std(np.vstack((base, alternative)), axis=0), .05)
    return scale, (base - alternative) / scale, np.asarray(targets), np.asarray(weights)


def fit_ridge(starts, representation):
    scale, design, targets, weights = learning_rows(starts, representation)
    matrix = design * weights[:, None]
    capped = np.clip(targets, -2. * RESIDUAL_CAP_S, 2. * RESIDUAL_CAP_S)
    regularizer = RIDGE * np.eye(design.shape[1])
    beta = np.linalg.solve(design.T @ matrix + regularizer, matrix.T @ capped)
    return Ranker(scale, beta), {"safe_pairs": len(targets),
                                "targets_clipped": int(np.count_nonzero(capped != targets))}


def fit_huber(starts, representation, overprediction_weight):
    scale, design, targets, weights = learning_rows(starts, representation)

    def objective(beta):
        error = design @ beta - targets
        absolute = np.abs(error)
        clipped = np.minimum(absolute, HUBER_RADIUS_S)
        side_weight = weights * np.where(error > 0., overprediction_weight, 1.)
        loss = side_weight @ (clipped ** 2 / 2. + HUBER_RADIUS_S * (absolute - clipped))
        gradient = design.T @ (side_weight * np.clip(error, -HUBER_RADIUS_S, HUBER_RADIUS_S))
        return loss + RIDGE / 2. * (beta @ beta), gradient + RIDGE * beta

    fitted = minimize(objective, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B",
                      options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-7})
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError(f"Huber fit failed for {representation}: {fitted.message}")
    return Ranker(scale, fitted.x), {"safe_pairs_with_multiplicity": len(targets), "iterations": int(fitted.nit)}


def group_draws(starts):
    groups = defaultdict(list)
    for start in starts:
        groups[start.group].append(start)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError("At least two independent training groups are required.")
    for seed in SEEDS:
        draw = np.random.default_rng(seed).choice(keys, size=len(keys), replace=True).tolist()
        yield seed, draw, [start for group in draw for start in groups[group]]


def save_models(path, models):
    temporary = path.with_suffix(".part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, scales=np.stack([model.scale for model in models]),
                            coefficients=np.stack([model.coefficients for model in models]))
    temporary.replace(path)
    return digest(path)


def estimates(start, models, representation):
    features = start.features[representation]
    scores = np.stack([start.forecasts + model.corrections(features) for model in models])
    gains = scores[:, :1] - scores[:, 1:]
    spread = gains.std(axis=0, ddof=1) if len(models) > 1 else np.zeros(gains.shape[1])
    return gains.mean(axis=0), np.maximum(spread, SPREAD_FLOOR_S)


def calibrate(starts, models, representation, *, include_failures):
    scores = {}
    pairs, failures, excluded_incumbents = 0, 0, 0
    for start in starts:
        incumbent, *alternatives = start.branches
        if outcome(incumbent) != "safe_capture":
            excluded_incumbents += 1
            continue
        means, spreads = estimates(start, models, representation)
        for alternative, mean, spread in zip(alternatives, means, spreads, strict=True):
            actual = actual_gain(incumbent, alternative)
            if actual is None:
                if not include_failures:
                    continue
                actual = incumbent["duration_s"] - start.cap_s
                failures += 1
            score = float((mean - actual) / spread)
            scores[start.group] = max(score, scores.get(start.group, -math.inf))
            pairs += 1
    count = len(scores)
    order = math.ceil((count + 1) * NOMINAL_COVERAGE)
    multiplier = max(0., sorted(scores.values())[order - 1]) if order <= count else None
    return {"multiplier": multiplier, "calibration_pairs": pairs,
            "restricted_failure_targets": failures, "excluded_incumbent_starts": excluded_incumbents,
            "groups": count, "order_statistic": order, "group_scores": scores,
            "nominal_coverage": NOMINAL_COVERAGE}


def evaluate(starts, models, representation, multiplier):
    selected, errors, bias = [], [], []
    oracle_opportunities = 0
    coverage = defaultdict(list)
    for start in starts:
        incumbent, *alternatives = start.branches
        actual = [actual_gain(incumbent, alternative) for alternative in alternatives]
        oracle_opportunities += any(value is not None and value > MINIMUM_MODEL_GAIN_S for value in actual)
        mean, spread = estimates(start, models, representation)
        lower = mean - multiplier * spread if multiplier is not None else np.full_like(mean, -np.inf)
        mistakes = [predicted - observed for predicted, observed in zip(mean, actual, strict=True)
                    if observed is not None]
        if mistakes:
            errors.append(float(np.mean(np.abs(mistakes))))
            bias.append(float(np.mean(mistakes)))
        if outcome(incumbent) == "safe_capture":
            for gain, bound in zip(actual, lower, strict=True):
                restricted = incumbent["duration_s"] - start.cap_s if gain is None else gain
                coverage[start.group].append(bool(bound <= restricted))
        index = int(np.argmax(lower))
        if lower[index] > MINIMUM_MODEL_GAIN_S:
            gain = actual[index]
            selected.append({
                "state_id": start.state_id, "scenario": start.scenario, "group": start.group,
                "candidate": alternatives[index]["action_id"], "mean_gain_s": float(mean[index]),
                "spread_s": float(spread[index]), "gate_score_s": float(lower[index]), "safe_gain_s": gain,
                "incumbent_outcome": outcome(incumbent), "alternative_outcome": outcome(alternatives[index]),
                "true_positive": gain is not None and gain > MINIMUM_MODEL_GAIN_S,
            })
    wins = sum(row["true_positive"] for row in selected)
    evaluable = sum(row["safe_gain_s"] is not None for row in selected)
    return {
        "states": len(starts), "eligible_groups": len({start.group for start in starts}),
        "eligible_pairs": sum(len(start.branches) - 1 for start in starts),
        "selected": len(selected), "abstentions": len(starts) - len(selected),
        "true_positives": wins, "false_positives": len(selected) - wins,
        "safe_pair_false_positives": evaluable - wins, "without_paired_safe_eta": len(selected) - evaluable,
        "precision": wins / len(selected) if selected else None,
        "safe_pair_precision": wins / evaluable if evaluable else None, "oracle_states": oracle_opportunities,
        "recall_of_oracle_states": wins / oracle_opportunities if oracle_opportunities else None,
        "point_gain_mae_s": float(np.mean(errors)) if errors else None,
        "point_gain_bias_s": float(np.mean(bias)) if bias else None,
        "selected_alternative_outcomes": dict(Counter(row["alternative_outcome"] for row in selected)),
        "covered_complete_groups": sum(all(group) for group in coverage.values()),
        "evaluable_coverage_groups": len(coverage), "selected_actions": selected,
    }


def describe(rows, starts):
    return {
        "input_starts": len(rows), "eligible_starts": len(starts),
        "ineligible_starts": len(rows) - len(starts),
        "groups": sorted({row["group"] for row in rows}),
        "scenarios": sorted({row["scenario"] for row in rows}),
        "eligible_pairs": sum(len(start.branches) - 1 for start in starts),
        "incumbent_outcomes": dict(Counter(outcome(start.branches[0]) for start in starts)),
        "alternative_outcomes": dict(Counter(outcome(branch) for start in starts for branch in start.branches[1:])),
    }


def decoder_evaluation(rows, embeddings, decoder):
    targets = weather_targets(rows)
    groups = [row["group"] for row in rows]
    _, membership, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1. / counts[membership]
    errors = np.abs(decoder.predict(embeddings) - targets)
    mean_errors = np.abs(decoder.target_mean - targets)
    return {"group_weighted_mae": np.average(errors, weights=weights, axis=0).tolist(),
            "group_weighted_mean_baseline_mae": np.average(mean_errors, weights=weights, axis=0).tolist()}


def run(train_path, calibration_path, test_path, test_sha256, output, *, train_sha256=None, calibration_sha256=None):
    train_path, calibration_path, test_path, output = map(Path, (train_path, calibration_path, test_path, output))
    if len({path.resolve() for path in (train_path, calibration_path, test_path)}) != 3:
        raise ValueError("Training, calibration, and held-out data must use three different files.")
    pin = json.loads((FROZEN / "models" / "fit_receipt.json").read_text())
    if test_sha256 != pin["labels_sha256"]["test"]:
        raise ValueError("The nominated test input is different from the frozen October benchmark.")
    if digest(MANIFEST) != pin["protocol"]["split_manifest_sha256"]:
        raise ValueError("The original seasonal split manifest changed.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new fit output directory: {output}.")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {row["id"]: row for row in json.loads(MANIFEST.read_text())["scenarios"]}
    train_rows, train_sha = read_rows(train_path, "train", manifest, train_sha256)
    april_rows, april_sha = read_rows(calibration_path, "calibration", manifest, calibration_sha256)
    source_paths = {
        "fit_stage1.py": Path(__file__),
        "compact_features.py": Path(__file__).with_name("compact_features.py"),
        "evaluate_eta_gate.py": RESULTS / "eta_uncertainty_offline" / "reproduce" / "evaluate_eta_gate.py",
        "seasonal_ranker.py": RESULTS / "seasonal_native_training" / "reproduce" / "seasonal_ranker.py",
        "weather_embedding.py": WEATHER_ENCODER.parent / "reproduce" / "weather_embedding.py",
    }
    source_hashes = {name: digest(path) for name, path in source_paths.items()}
    weather_sha = digest(WEATHER_ENCODER)
    encoder = WeatherEmbedding(WEATHER_ENCODER)
    train_weather = raw_embeddings(train_rows, encoder)
    decoder, decoder_receipt = fit_decoder(train_rows, train_weather)
    train = prepare(train_rows, train_weather, decoder)
    april_weather = raw_embeddings(april_rows, encoder)
    april = prepare(april_rows, april_weather, decoder)
    descriptions = {"train": describe(train_rows, train), "calibration": describe(april_rows, april)}
    train_groups = set(descriptions["train"]["groups"])
    april_groups = set(descriptions["calibration"]["groups"])
    if train_groups & april_groups:
        raise ValueError("Training and calibration share a weather or departure group.")
    decoder_path = output / "weather_decoder.npz"
    decoder.save(decoder_path)
    draws = list(group_draws(train))
    comparisons, receipts, calibrations = {}, {}, {}
    for representation in REPRESENTATIONS:
        ridge, ridge_receipt = fit_ridge(train, representation)
        model_set = {"ridge": [ridge]}
        receipts[representation] = {"ridge": ridge_receipt}
        for name, weight in (("symmetric_huber", 1.), ("asymmetric_huber", OVERPREDICTION_WEIGHT)):
            models, confirmations = [], []
            for seed, sampled, starts in draws:
                model, confirmation = fit_huber(starts, representation, weight)
                models.append(model)
                confirmations.append({"seed": seed, "groups_drawn": sampled, **confirmation})
            model_set[name] = models
            receipts[representation][name] = {"overprediction_weight": weight, "draws": confirmations}
        for name, models in model_set.items():
            key = f"{representation}_{name}"
            model_path = output / f"{key}.npz"
            receipts[representation][name]["checkpoint_sha256"] = save_models(model_path, models)
            comparisons[f"{key}_point"] = (models, representation, 0.)
        asymmetric = model_set["asymmetric_huber"]
        restricted = calibrate(april, asymmetric, representation, include_failures=True)
        safe_only = calibrate(april, asymmetric, representation, include_failures=False)
        calibrations[representation] = {"restricted": restricted, "safe_pairs_only": safe_only}
        comparisons[f"{representation}_asymmetric_lower_bound"] = (asymmetric, representation, restricted["multiplier"])
    fit_path = output / "fit_receipt.json"
    write_json(fit_path, {
        "sources_sha256": source_hashes, "weather_encoder_sha256": weather_sha,
        "inputs_sha256": {"train": train_sha, "calibration": april_sha, "test_expected": test_sha256},
        "manifest_sha256": digest(MANIFEST), "training_and_calibration": descriptions,
        "software": {name: version(name) for name in ("numpy", "scipy", "torch")},
        "protocol": {"strict_gain_s": MINIMUM_MODEL_GAIN_S, "native_eta_regret_s": TRUST_REGRET_S,
                     "residual_cap_s": RESIDUAL_CAP_S, "ridge": RIDGE, "huber_radius_s": HUBER_RADIUS_S,
                     "bootstrap_seeds": SEEDS, "spread_floor_s": SPREAD_FLOOR_S,
                     "explicit_names": EXPLICIT_NAMES, "latent_names": LATENT_NAMES},
        "weather_decoder": {**decoder_receipt, "checkpoint_sha256": digest(decoder_path)},
        "models": receipts, "april_calibration": calibrations,
    })
    frozen_fit = digest(fit_path)

    test_rows, observed_test_sha = read_rows(test_path, "test", manifest, test_sha256)
    test_weather = raw_embeddings(test_rows, encoder)
    test = prepare(test_rows, test_weather, decoder)
    test_description = describe(test_rows, test)
    if set(test_description["groups"]) & (train_groups | april_groups):
        raise ValueError("The October benchmark shares a training or calibration weather or departure group.")
    results = {
        "fit_receipt_sha256": frozen_fit, "test_digest_sha256": observed_test_sha,
        "partitions": {**descriptions, "test": test_description},
        "decoder_diagnostics": {
            "target_names": decoder_receipt["target_names"],
            "april": decoder_evaluation(april_rows, april_weather, decoder),
            "october": decoder_evaluation(test_rows, test_weather, decoder),
        },
        "april": {name: evaluate(april, *args) for name, args in comparisons.items()},
        "october": {name: evaluate(test, *args) for name, args in comparisons.items()},
    }
    if (frozen_fit != digest(fit_path) or weather_sha != digest(WEATHER_ENCODER)
            or source_hashes != {name: digest(path) for name, path in source_paths.items()}):
        raise ValueError("A pinned source or the fit receipt changed during fitting/evaluation.")
    write_json(output / "results.json", results)
    print(f"Fit receipt {fit_path}, SHA-256 {frozen_fit}", flush=True)
    for name, row in results["october"].items():
        print(f"{name}: {row['selected']} selected, {row['true_positives']} true, "
              f"{row['false_positives']} false, precision={row['precision']}, "
              f"recall={row['recall_of_oracle_states']}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--test", type=Path, default=FROZEN / "branches" / "test.jsonl.gz")
    parser.add_argument("--test-sha256", required=True)
    parser.add_argument("--train-sha256")
    parser.add_argument("--calibration-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.train, args.calibration, args.test, args.test_sha256, args.output,
        train_sha256=args.train_sha256, calibration_sha256=args.calibration_sha256)


if __name__ == "__main__":
    main()
