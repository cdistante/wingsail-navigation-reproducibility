"""Fit bounded schedule-time corrections from paired native tack outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


RIDGE = 2.0
RESIDUAL_CAP_S = 900.
TRUST_REGRET_S = 150.
MINIMUM_MODEL_GAIN_S = 25.


@dataclass(frozen=True)
class Pair:
    state_id: str
    physics_features: np.ndarray
    alternative_features: np.ndarray
    physics_forecast_s: float
    alternative_forecast_s: float
    physics_outcome_s: float
    alternative_outcome_s: float


@dataclass(frozen=True)
class Ranker:
    scale: np.ndarray
    coefficients: np.ndarray

    def corrections(self, features):
        projected = np.asarray(features, dtype=float) / self.scale
        return np.clip(projected @ self.coefficients, -RESIDUAL_CAP_S, RESIDUAL_CAP_S)

    def save(self, path: Path):
        np.savez_compressed(path, scale=self.scale, coefficients=self.coefficients)

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as values:
            scales, weights = values["scale"], values["coefficients"]
        if (scales.ndim != 1 or weights.shape != scales.shape or not np.isfinite(scales).all()
                or not np.isfinite(weights).all() or np.any(scales <= 0)):
            raise ValueError("Invalid seasonal learned coefficients.")
        return cls(scales, weights)


def fit(pairs):
    """Learn realized-versus-forecast differences within exact matched starts."""
    pairs = list(pairs)
    if not pairs:
        raise ValueError("Seasonal training needs at least one legal alternative paired to physics.")
    physics = np.stack([p.physics_features for p in pairs])
    alternatives = np.stack([p.alternative_features for p in pairs])
    samples = np.vstack((physics, alternatives))
    scale = np.maximum(np.std(samples, axis=0), .05)
    design = (alternatives - physics) / scale
    target = np.array([(p.alternative_outcome_s - p.physics_outcome_s)
                       - (p.alternative_forecast_s - p.physics_forecast_s) for p in pairs], dtype=float)
    states, inverse, counts = np.unique([p.state_id for p in pairs], return_inverse=True, return_counts=True)
    weight = (1. / counts[inverse]) ** .5
    if not np.isfinite(design).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite paired data.")
    target = np.clip(target, -2 * RESIDUAL_CAP_S, 2 * RESIDUAL_CAP_S)
    matrix = design * weight[:, None]
    beta = np.linalg.solve(matrix.T @ matrix + RIDGE * np.eye(matrix.shape[1]), matrix.T @ (weight * target))
    prediction = design @ beta
    receipt = {"pairs": len(pairs), "independent_start_states": len(states), "ridge": RIDGE,
               "rmse_s": float(np.sqrt(np.mean((prediction - target)**2))),
               "mean_zero_residual_rmse_s": float(np.sqrt(np.mean(target**2))),
               "training_target_clip_s": 2 * RESIDUAL_CAP_S}
    return Ranker(scale, beta), receipt


def choose(ranker, features, forecast_eta, allowed, incumbent, strength):
    """Rerank close capturing schedules. A foreseen miss cannot displace physics."""
    available = np.asarray(allowed, dtype=bool)
    if not available[incumbent]:
        raise ValueError("The schedule veto requires an admissible baseline action.")
    if strength == 0:
        return incumbent
    eta = np.asarray([float(x) if x is not None else np.inf for x in forecast_eta])
    baseline_eta = eta[incumbent]
    if not np.isfinite(baseline_eta):
        return incumbent
    eligible = available & np.isfinite(eta) & (eta <= baseline_eta + TRUST_REGRET_S)
    if eligible.sum() < 2:
        return incumbent
    corrected = eta + strength * ranker.corrections(features)
    proposed = int(np.argmin(np.where(eligible, corrected, np.inf)))
    return proposed if corrected[incumbent] - corrected[proposed] > MINIMUM_MODEL_GAIN_S else incumbent
