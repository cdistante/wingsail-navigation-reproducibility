"""Score feasible tack candidates using a portable Fossen-rollout model."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import atan2, cos, hypot, radians, sin
from pathlib import Path

import numpy as np

from ._geo import local_east_north_metres
from .scenario import Waypoint
from .world import WorldObservation


FEATURE_NAMES = (
    "wind_speed_mps",
    "wind_from_along_goal",
    "wind_from_across_goal",
    "current_along_goal_mps",
    "current_across_goal_mps",
    "wave_height_m",
    "boat_speed_mps",
    "boat_surge_mps",
    "boat_sway_mps",
    "boat_yaw_rate_rad_s",
    "boat_rudder_deg",
    "boat_flap_deg",
    "course_along_goal",
    "course_across_goal",
    "bow_along_wind_from",
    "bow_across_wind_from",
    "bow_along_boat",
    "bow_across_boat",
    "goal_distance_m",
)


def tack_candidate_features(
    observation: WorldObservation, goal: Waypoint, course: float, bow: float,
) -> np.ndarray:
    """Build current-observation features for a candidate course and bow in true degrees.

    Bearings are encoded relative to the goal, reported wind source, or own
    bow. The cross-goal axis points to starboard of the goal bearing.
    """
    wind, current, wave = observation.wind, observation.current, observation.wave
    if wind is None or current is None or wave is None:
        raise ValueError("Tack scoring needs observed wind, current and wave height.")
    boat = observation.boat
    east, north = local_east_north_metres(
        goal.latitude, goal.longitude, boat.latitude, boat.longitude,
    )
    distance = hypot(east, north)
    inputs = (wind.speed, wind.direction, current.u, current.v, wave.height, boat.speed,
              boat.heading, course, bow, distance)
    if not np.isfinite(inputs).all() or distance <= 0.0:
        raise ValueError("Tack scoring needs finite inputs and a nonzero goal range.")

    goal_bearing = atan2(east, north)
    wind_bearing = radians(wind.direction)
    wind_relative = wind_bearing - goal_bearing
    course_relative = radians(course) - goal_bearing
    bow_wind_relative = radians(bow) - wind_bearing
    bow_boat_relative = radians(bow - boat.heading)
    east_fraction, north_fraction = east / distance, north / distance
    features = np.array((
        wind.speed, cos(wind_relative), sin(wind_relative),
        current.u * east_fraction + current.v * north_fraction,
        current.u * north_fraction - current.v * east_fraction,
        wave.height, boat.speed, boat.speed if boat.surge_speed is None else boat.surge_speed,
        boat.sway_speed, boat.yaw_rate, boat.rudder, boat.flap,
        cos(course_relative), sin(course_relative),
        cos(bow_wind_relative), sin(bow_wind_relative),
        cos(bow_boat_relative), sin(bow_boat_relative), distance,
    ), dtype=float)
    if not np.isfinite(features).all():
        raise ValueError("Tack features must be finite.")
    return features


@dataclass(frozen=True)
class LearnedTackSelector:
    """Predict Fossen-rollout progress for candidates provided by the planner."""

    means: np.ndarray
    scales: np.ndarray
    target_mean: float
    target_scale: float
    layers: tuple[tuple[np.ndarray, np.ndarray], ...]
    domain_bounds: tuple[tuple[int, float, float], ...]

    @classmethod
    def load(cls, path: Path | str) -> LearnedTackSelector:
        """Load a version 1 JSON MLP, with layer weights in output-by-input order."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") != 1 or data.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("Unsupported learned tack model version or feature order.")
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
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise ValueError("Malformed learned tack model.") from error

        width = len(FEATURE_NAMES)
        if (means.shape != (width,) or scales.shape != (width,)
                or not np.isfinite(means).all() or not np.isfinite(scales).all()
                or np.any(scales <= 0.0) or not np.isfinite([target_mean, target_scale]).all()
                or target_scale <= 0.0):
            raise ValueError("Invalid learned tack normalization.")
        for weights, bias in layers:
            if (weights.ndim != 2 or weights.shape[1] != width or weights.shape[0] == 0
                    or bias.shape != (weights.shape[0],)
                    or not np.isfinite(weights).all() or not np.isfinite(bias).all()):
                raise ValueError("Invalid learned tack layer dimensions or values.")
            width = weights.shape[0]
        if not layers or width != 1:
            raise ValueError("The learned tack model must output one progress estimate.")
        if any(not np.isfinite([low, high]).all() or low > high for _, low, high in domain_bounds):
            raise ValueError("Invalid learned tack domain bound.")
        return cls(means, scales, target_mean, target_scale, layers, domain_bounds)

    def predict_progress(
        self, observation: WorldObservation, goal: Waypoint, course: float, bow: float,
    ) -> float | None:
        """Return predicted displacement toward the goal, in metres, if in domain."""
        try:
            features = tack_candidate_features(observation, goal, course, bow)
        except ValueError:
            return None
        if any(not low <= features[index] <= high for index, low, high in self.domain_bounds):
            return None

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            values = (features - self.means) / self.scales
            for weights, bias in self.layers[:-1]:
                values = np.maximum(weights @ values + bias, 0.0)
            weights, bias = self.layers[-1]
            result = float((weights @ values + bias)[0] * self.target_scale + self.target_mean)
        return result if np.isfinite(result) else None
