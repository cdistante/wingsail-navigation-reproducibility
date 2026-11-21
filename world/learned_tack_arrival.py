"""Predict capture and conditional arrival time for a proposed tack schedule."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .learned_tacks import FEATURE_NAMES as FIRST_LEG_FEATURE_NAMES
from .learned_tacks import tack_candidate_features
from .scenario import Waypoint
from .world import WorldObservation


_DIRECTION_FEATURE_NAMES = (
    "course_along_goal", "course_across_goal",
    "bow_along_wind_from", "bow_across_wind_from",
    "bow_along_boat", "bow_across_boat",
)
_DIRECTION_FEATURE_INDICES = np.array([
    FIRST_LEG_FEATURE_NAMES.index(name) for name in _DIRECTION_FEATURE_NAMES
])
FEATURE_NAMES = (
    *FIRST_LEG_FEATURE_NAMES,
    "has_transition", "first_leg_seconds",
    *(f"next_{name}" for name in _DIRECTION_FEATURE_NAMES),
    "arrival_radius_m", "horizon_seconds",
)


def tack_arrival_features(
    observation: WorldObservation, goal: Waypoint,
    first_course: float, first_bow: float, next_course: float | None, next_bow: float | None,
    switch_seconds: float | None, arrival_radius_m: float, horizon_seconds: float,
) -> np.ndarray:
    """Encode proposed first and continuation bearings in true degrees.

    A single leg uses absent continuation bearings and absent or zero switch
    seconds. A two-leg schedule switches strictly inside the rollout horizon.
    """
    if (not np.isfinite([arrival_radius_m, horizon_seconds]).all()
            or arrival_radius_m <= 0.0 or horizon_seconds <= 0.0):
        raise ValueError("Arrival radius and horizon must be positive and finite.")
    first = tack_candidate_features(observation, goal, first_course, first_bow)
    if next_course is None and next_bow is None and switch_seconds in (None, 0.0):
        has_transition, first_leg_seconds = 0.0, 0.0
        continuation = np.zeros(len(_DIRECTION_FEATURE_INDICES))
    else:
        if (next_course is None or next_bow is None or switch_seconds is None
                or not 0.0 < switch_seconds < horizon_seconds):
            raise ValueError("A transition needs both bearings and switch seconds inside the horizon.")
        has_transition, first_leg_seconds = 1.0, switch_seconds
        continuation = tack_candidate_features(
            observation, goal, next_course, next_bow,
        )[_DIRECTION_FEATURE_INDICES]
    return np.concatenate((
        first, [has_transition, first_leg_seconds], continuation, [arrival_radius_m, horizon_seconds],
    ))


@dataclass(frozen=True)
class TackArrivalPrediction:
    capture_probability: float
    arrival_seconds: float


@dataclass(frozen=True)
class LearnedTackArrivalSelector:
    """Predict capture by the recorded horizon and time conditional on capture."""

    means: np.ndarray
    scales: np.ndarray
    target_mean: float
    target_scale: float
    layers: tuple[tuple[np.ndarray, np.ndarray], ...]
    domain_bounds: tuple[tuple[int, float, float], ...]

    @classmethod
    def load(cls, path: Path | str) -> LearnedTackArrivalSelector:
        """Load a v2 JSON MLP with a capture logit and standardized seconds head."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") != 2 or data.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("Unsupported tack arrival model version or feature order.")
        try:
            means = np.asarray(data["means"], dtype=float)
            scales = np.asarray(data["scales"], dtype=float)
            target_mean, target_scale = float(data["target_mean"]), float(data["target_scale"])
            layers = tuple(
                (np.asarray(layer["weights"], dtype=float), np.asarray(layer["bias"], dtype=float))
                for layer in data["layers"]
            )
            domain_bounds = tuple(
                (FEATURE_NAMES.index(name), float(low), float(high))
                for name, (low, high) in data.get("domain_bounds", {}).items()
            )
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError("Malformed tack arrival model.") from error

        width = len(FEATURE_NAMES)
        if (means.shape != (width,) or scales.shape != (width,)
                or not np.isfinite(means).all() or not np.isfinite(scales).all()
                or np.any(scales <= 0.0) or not np.isfinite([target_mean, target_scale]).all()
                or target_scale <= 0.0):
            raise ValueError("Invalid tack arrival normalization.")
        for weights, bias in layers:
            if (weights.ndim != 2 or weights.shape[1] != width or weights.shape[0] == 0
                    or bias.shape != (weights.shape[0],)
                    or not np.isfinite(weights).all() or not np.isfinite(bias).all()):
                raise ValueError("Invalid tack arrival layer dimensions or values.")
            width = weights.shape[0]
        if not layers or width != 2:
            raise ValueError("The tack arrival model must output capture and conditional arrival time.")
        if any(not np.isfinite([low, high]).all() or low > high for _, low, high in domain_bounds):
            raise ValueError("Invalid tack arrival domain bound.")
        return cls(means, scales, target_mean, target_scale, layers, domain_bounds)

    def predict_arrival(
        self, observation: WorldObservation, goal: Waypoint,
        first_course: float, first_bow: float, next_course: float | None, next_bow: float | None,
        switch_seconds: float | None, arrival_radius_m: float, horizon_seconds: float,
    ) -> TackArrivalPrediction | None:
        """Return an in-domain estimate for a schedule within the supplied horizon."""
        try:
            features = tack_arrival_features(
                observation, goal, first_course, first_bow, next_course, next_bow,
                switch_seconds, arrival_radius_m, horizon_seconds,
            )
        except ValueError:
            return None
        if any(not low <= features[index] <= high for index, low, high in self.domain_bounds):
            return None

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            values = (features - self.means) / self.scales
            for weights, bias in self.layers[:-1]:
                values = np.maximum(weights @ values + bias, 0.0)
            weights, bias = self.layers[-1]
            logit, normalized_seconds = weights @ values + bias
            arrival_seconds = normalized_seconds * self.target_scale + self.target_mean
        if not np.isfinite(logit) or not 0.0 <= arrival_seconds <= horizon_seconds:
            return None
        probability = np.exp(-np.logaddexp(0.0, -logit))
        return TackArrivalPrediction(float(probability), float(arrival_seconds))
