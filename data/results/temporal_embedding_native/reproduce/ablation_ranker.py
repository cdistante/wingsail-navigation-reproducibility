"""Matched native ETA correction and fixed-state evaluation for the ablation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from seasonal_ranker import MINIMUM_MODEL_GAIN_S, RESIDUAL_CAP_S, TRUST_REGRET_S, Pair, Ranker, choose, fit


ARMS = ("physics", "explicit", "temporal", "snapshot")


@dataclass(frozen=True)
class Selection:
    index: int
    status: str
    scored_count: int
    predicted_gain_s: float | None


def _eta(values):
    return np.asarray([np.inf if value is None else float(value) for value in values])


def _eligible(eta, allowed, incumbent):
    available = np.asarray(allowed, dtype=bool)
    if available.shape != eta.shape:
        raise ValueError("The forecast and legality mask must cover the same candidates.")
    if incumbent < 0:
        return np.zeros_like(available)
    if incumbent >= len(eta) or not available[incumbent]:
        raise ValueError("The declared physics incumbent must be admissible.")
    if not np.isfinite(eta[incumbent]):
        return np.zeros_like(available)
    return available & np.isfinite(eta) & (eta <= eta[incumbent] + TRUST_REGRET_S)


def eligible_mask(forecast_eta, allowed, incumbent):
    """Return the common legal, finite-forecast candidate set, including physics."""
    return _eligible(_eta(forecast_eta), allowed, incumbent)


def _features(arm, explicit, embedding_history, embedding_snapshot):
    explicit = np.asarray(explicit, dtype=float)
    if arm == "explicit":
        return explicit
    embedding = embedding_history if arm == "temporal" else embedding_snapshot
    rows = np.asarray(embedding, dtype=float)
    if rows.ndim != 2 or rows.shape[0] != len(explicit):
        raise ValueError(f"The {arm} embedding must align with the candidate rows.")
    return np.column_stack((explicit, rows))


def select(arm, model: Ranker | None, explicit, forecast_eta, allowed, incumbent, *,
           embedding_history=None, embedding_snapshot=None) -> Selection:
    """Apply the shared deterministic gate to an already screened candidate set."""
    if arm not in ARMS:
        raise ValueError(f"Unknown ablation arm {arm!r}.")
    eta = _eta(forecast_eta)
    eligible = _eligible(eta, allowed, incumbent)
    count = int(eligible.sum())
    if incumbent < 0:
        return Selection(incumbent, "no_admissible_incumbent", 0, None)
    if arm == "physics":
        return Selection(incumbent, "physics_only", 0, None)
    if count == 0:
        return Selection(incumbent, "incumbent_has_no_forecast_capture", 0, None)
    if count < 2:
        return Selection(incumbent, "no_eligible_alternative", 0, None)
    if model is None:
        raise ValueError(f"The {arm} arm requires a fitted residual ranker.")

    indices = np.flatnonzero(eligible)
    features = _features(arm, explicit, embedding_history, embedding_snapshot)[indices]
    forecasts = eta[indices]
    corrected = forecasts + model.corrections(features)
    if not np.isfinite(corrected).all():
        raise ValueError(f"The {arm} arm produced a nonfinite eligible ETA.")
    baseline = int(np.count_nonzero(eligible[:incumbent]))
    gain = float(corrected[baseline] - np.min(corrected))
    selected = int(indices[choose(model, features, forecasts, np.ones(count, dtype=bool), baseline, strength=1.)])
    if selected != incumbent:
        status = "override"
    elif gain > 0.:
        status = "below_or_equal_gain_threshold"
    else:
        status = "incumbent_best"
    return Selection(selected, status, count, gain)


def _outcome(branch):
    if not branch["matched_initial_action"]:
        return "unapplied"
    margins = [branch.get(name) for name in ("min_report_margin_m", "min_truth_margin_m")]
    if any(value is not None and np.isfinite(value) and value <= 0. for value in margins):
        return "unsafe"
    if branch["actual_capture"]:
        if branch["audited_safe"]:
            return "safe_capture"
        return "unaudited_capture"
    status = branch.get("terminal_status")
    if status == "censored_cap":
        return "timeout"
    if status == "no_safe_native_action":
        return "no_safe_action"
    return "other_failure"


def _state_inputs(record):
    branches = record["branches"]
    if not branches:
        raise ValueError("Each fixed-state row must include its physics incumbent first.")
    explicit = np.asarray([branch["features"] for branch in branches], dtype=float)
    eta = _eta([branch["forecast_s"] for branch in branches])
    eligible = _eligible(eta, np.ones(len(branches), dtype=bool), 0)
    outcomes = [_outcome(branch) for branch in branches]
    return branches, explicit, eta, eligible, outcomes


def fit_arms(records):
    """Fit the three residual arms to the same observed safe-capture time pairs."""
    pairs = {arm: [] for arm in ARMS[1:]}
    for record in records:
        branches, explicit, eta, eligible, outcomes = _state_inputs(record)
        if outcomes[0] != "safe_capture":
            continue
        alternatives = [i for i in np.flatnonzero(eligible) if i and outcomes[i] == "safe_capture"]
        if not alternatives:
            continue
        for arm, rows in pairs.items():
            features = _features(arm, explicit, record.get("embedding_history"), record.get("embedding_snapshot"))
            for i in alternatives:
                rows.append(Pair(record["state_id"], features[0], features[i], eta[0], eta[i],
                                 branches[0]["duration_s"], branches[i]["duration_s"]))

    models = {}
    receipts = {}
    for arm, rows in pairs.items():
        models[arm], receipt = fit(rows)
        targets = np.asarray([(pair.alternative_outcome_s - pair.physics_outcome_s)
                              - (pair.alternative_forecast_s - pair.physics_forecast_s) for pair in rows])
        receipts[arm] = {**receipt, "feature_count": int(len(models[arm].scale)),
                         "minimum_unclipped_target_s": float(targets.min()),
                         "maximum_unclipped_target_s": float(targets.max()),
                         "targets_clipped": int(np.sum(np.abs(targets) > 2. * RESIDUAL_CAP_S))}
    return models, receipts


def _mean(values):
    return float(np.mean(values)) if values else None


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def evaluate_fixed_states(records, models):
    """Compare predicted and realized gains on the same native branch starts."""
    states = []
    incumbent_outcomes = Counter()
    alternative_outcomes = Counter()
    for record in records:
        branches, explicit, eta, eligible, outcomes = _state_inputs(record)
        alternatives = [int(i) for i in np.flatnonzero(eligible) if i]
        incumbent_outcomes.update([outcomes[0]])
        alternative_outcomes.update(outcomes[i] for i in alternatives)
        truth = {i: float(branches[0]["duration_s"] - branches[i]["duration_s"])
                 for i in alternatives if outcomes[0] == outcomes[i] == "safe_capture"}
        oracle = max((0., *truth.values())) if outcomes[0] == "safe_capture" else None
        row = {"state_id": record["state_id"], "scenario": record.get("scenario"),
               "block": record.get("block"), "eligible_alternatives": len(alternatives),
               "incumbent_outcome": outcomes[0], "oracle_safe_gain_s": oracle,
               "oracle_gain_gt_25": oracle is not None and oracle > MINIMUM_MODEL_GAIN_S,
               "actual_safe_gains_s": truth, "arms": {}}
        kwargs = {"embedding_history": record.get("embedding_history"),
                  "embedding_snapshot": record.get("embedding_snapshot")}
        for arm in ARMS:
            model = models.get(arm)
            choice = select(arm, model, explicit, eta, np.ones(len(eta), dtype=bool), 0, **kwargs)
            actual = 0. if choice.index == 0 and outcomes[0] == "safe_capture" else truth.get(choice.index)
            scores = eta[eligible]
            if arm != "physics" and truth:
                scores = scores + model.corrections(_features(arm, explicit, **kwargs)[eligible])
            residuals = {i: float(scores[0] - scores[position] - truth[i])
                         for position, i in enumerate(alternatives, start=1) if i in truth}
            row["arms"][arm] = {"status": choice.status, "scored_count": choice.scored_count,
                               "selected_index": choice.index, "selected_outcome": outcomes[choice.index],
                               "predicted_gain_s": choice.predicted_gain_s, "actual_selected_gain_s": actual,
                               "observed_oracle_regret_s": oracle - actual if actual is not None else None,
                               "paired_prediction_errors_s": residuals}
        states.append(row)

    oracle_opportunities = sum(row["oracle_gain_gt_25"] for row in states)
    summaries = {}
    for arm in ARMS:
        selected = [row["arms"][arm] for row in states]
        changes = [choice for choice in selected if choice["selected_index"] != 0]
        observed = [choice["actual_selected_gain_s"] for choice in changes
                    if choice["actual_selected_gain_s"] is not None]
        wins = sum(gain > MINIMUM_MODEL_GAIN_S for gain in observed)
        regrets = [choice["observed_oracle_regret_s"] for choice in selected
                   if choice["observed_oracle_regret_s"] is not None]
        errors = [list(choice["paired_prediction_errors_s"].values()) for choice in selected]
        errors = [row for row in errors if row]
        summaries[arm] = {"scored_states": sum(choice["scored_count"] > 0 for choice in selected),
                          "predicted_crossings": len(changes), "decidable_crossings": len(observed),
                          "observed_safe_gain_gt_25_crossings": wins,
                          "observed_safe_gain_le_25_crossings": len(observed) - wins,
                          "crossings_without_paired_safe_eta": len(changes) - len(observed),
                          "crossing_outcomes": dict(Counter(choice["selected_outcome"] for choice in changes)),
                          "selected_outcomes": dict(Counter(choice["selected_outcome"] for choice in selected)),
                          "gate_precision_on_safe_pairs": _ratio(wins, len(observed)),
                          "observed_safe_win_fraction_of_crossings": _ratio(wins, len(changes)),
                          "oracle_opportunity_recall": _ratio(wins, oracle_opportunities),
                          "mean_observed_crossing_gain_s": _mean(observed),
                          "observed_oracle_regret_states": len(regrets),
                          "mean_observed_oracle_regret_s": _mean(regrets),
                          "safe_eta_states": len(errors), "safe_eta_pairs": sum(map(len, errors)),
                          "state_weighted_pair_mae_s": _mean([np.mean(np.abs(row)) for row in errors]),
                          "state_weighted_pair_bias_s": _mean([np.mean(row) for row in errors])}

    return {"states": len(states), "eligible_states": sum(row["eligible_alternatives"] > 0 for row in states),
            "eligible_alternative_pairs": sum(row["eligible_alternatives"] for row in states),
            "states_with_oracle_safe_gain_gt_25": oracle_opportunities,
            "incumbent_outcomes": dict(incumbent_outcomes), "eligible_alternative_outcomes": dict(alternative_outcomes),
            "arms": summaries, "state_records": states}
