"""Learned time-to-mark ranks for already generated sail-trim candidates.

The reduced motion used for learning advances a slow decision window with
Fossen equilibrium ground velocities and an explicit lost-motion turn interval.
The ranked candidates still require the runner's chart and AIS checks.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .world import WorldObservation


FEATURE_NAMES = (
    "range_km", "radius_100m", "north_goal_in_wind", "east_goal_in_wind",
    "live_wind_10mps", "wave_height_m", "flow_along", "flow_across",
    "future_flow_along", "future_flow_across", "has_future", "future_seconds_h",
    "live_gate_speed", "future_gate_speed", "decision_seconds_300",
    "remaining_seconds_h", "candidate_along", "candidate_across",
    "candidate_bow_sin_in_wind", "candidate_bow_cos_in_wind",
    "turn_seconds_180", "candidate_flap_15deg", "range_after_km",
    "gate_after_live", "gate_after_future", "physics_cost_h",
)
MAX_SECONDS = 21_600.0
TIMEOUT_COST_SECONDS = MAX_SECONDS
MODEL_VERSION = 1


@dataclass(frozen=True)
class PolarSnapshot:
    velocity: np.ndarray
    bow: np.ndarray
    flap: np.ndarray
    flow: np.ndarray
    wind_from: float
    hull: np.ndarray

    @classmethod
    def from_modes(cls, modes, flow, wind_from):
        velocity = np.asarray([mode.velocity for mode in modes], dtype=float).reshape(-1, 2)
        bow = np.asarray([mode.bow for mode in modes], dtype=float)
        flap = np.asarray([mode.flap for mode in modes], dtype=float)
        flow = np.asarray(flow, dtype=float)
        if not len(velocity) or flow.shape != (2,) or not all(
            np.isfinite(values).all() for values in (velocity, bow, flap, flow, wind_from)
        ):
            raise ValueError("A polar requires valid trim velocities and environmental vectors.")
        hull = np.unique(velocity, axis=0)
        if len(hull) > 2:
            try:
                hull = hull[ConvexHull(hull).vertices]
            except QhullError:
                # A collinear set needs only its extreme points.
                hull = hull[[0, -1]]
        return cls(velocity, bow, flap, flow, float(wind_from), hull)


@dataclass(frozen=True)
class CandidateBatch:
    features: np.ndarray
    displacement: np.ndarray
    remaining_target: np.ndarray
    end_bow: np.ndarray
    captured_after_s: np.ndarray
    physics_cost_s: np.ndarray
    turn_displacement: np.ndarray


@dataclass(frozen=True)
class MarkActionScore:
    candidate_index: int
    cost_seconds: float


def _signed_angle(heading, origin):
    return (heading - origin + 180.0) % 360.0 - 180.0


def _gate_speed(target: np.ndarray, polar: PolarSnapshot) -> np.ndarray:
    """Radial speed from the intersection of a velocity hull and the mark axis."""
    unit = target / np.maximum(np.linalg.norm(target, axis=-1, keepdims=True), 1e-6)
    normal = np.stack((unit[..., 1], -unit[..., 0]), axis=-1)
    along = unit @ polar.hull.T
    cross = normal @ polar.hull.T
    other_cross = np.roll(cross, 1, axis=-1)
    other_along = np.roll(along, 1, axis=-1)
    denominator = cross - other_cross
    weight = np.divide(cross, denominator, out=np.zeros_like(cross), where=np.abs(denominator) > 1e-9)
    edges = (cross * other_cross < 0) & (np.abs(denominator) > 1e-9)
    mixed = along + weight * (other_along - along)
    feasible = np.maximum(
        np.max(np.where(edges, mixed, -np.inf), axis=-1),
        np.max(np.where(np.abs(cross) < 1e-9, along, -np.inf), axis=-1),
    )
    return np.where(np.isfinite(feasible), feasible, -1.0)


def circle_entry(origin, displacement, target, radius):
    """Fraction of a ground segment before first circle entry, or infinity."""
    relative = origin - target
    radius_error = np.sum(relative * relative, axis=-1) - radius**2
    length = np.sum(displacement * displacement, axis=-1)
    projection = np.sum(relative * displacement, axis=-1)
    discriminant = projection**2 - length * radius_error
    with np.errstate(divide="ignore", invalid="ignore"):
        root = (-projection - np.sqrt(np.maximum(discriminant, 0.0))) / length
    enters = (length > 1e-10) & (discriminant >= 0.0) & (root >= 0.0) & (root <= 1.0)
    return np.where(radius_error <= 0.0, 0.0, np.where(enters, root, np.inf))


def candidate_batch(
    target, boat_bow, radius, live, future, wind_speed, wave_height,
    decision_seconds, until_future_seconds, remaining_seconds, *, choices=None,
) -> CandidateBatch:
    """Evaluate common candidate transitions and features for training/inference.

    Each row is a state. It can share the current trim set with other rows in
    the batch. `choices` specifies a per-row candidate subset when supplied.
    """
    target = np.asarray(target, dtype=float).reshape(-1, 2)
    count = len(target)
    broadcast = lambda x: np.broadcast_to(np.asarray(x, dtype=float), (count,))
    boat_bow = broadcast(boat_bow)
    radius = broadcast(radius)
    dt = broadcast(decision_seconds)
    horizon = broadcast(remaining_seconds)
    until = broadcast(until_future_seconds) if future is not None else np.zeros(count)
    if choices is None:
        choices = np.arange(len(live.bow))[None, :]
    velocity = live.velocity[choices]
    bows = live.bow[choices]
    flaps = live.flap[choices]
    width = velocity.shape[1]
    turn_degrees = _signed_angle(bows, boat_bow[:, None])
    wind_sides = (_signed_angle(bows, live.wind_from) < 0) != (
        _signed_angle(boat_bow[:, None], live.wind_from) < 0
    )
    turn_required = np.abs(turn_degrees) + 30.0 * wind_sides
    turning = np.minimum(dt[:, None], turn_required)
    turn_displacement = turning[..., None] * live.flow
    sailing = dt[:, None] - turning
    sailing_displacement = sailing[..., None] * velocity
    displacement = turn_displacement + sailing_displacement
    relative_target = target[:, None, :] - displacement
    turn_fraction = np.divide(turning, turn_required, out=np.ones_like(turning), where=turn_required > 0)
    end_bow = (boat_bow[:, None] + turn_fraction * turn_degrees) % 360.0
    hit_turn = circle_entry(np.zeros_like(displacement), turn_displacement, target[:, None, :], radius[:, None])
    hit_sail = circle_entry(turn_displacement, sailing_displacement, target[:, None, :], radius[:, None])
    with np.errstate(invalid="ignore"):
        captured = np.minimum(np.where(np.isfinite(hit_turn), hit_turn * turning, np.inf),
                              np.where(np.isfinite(hit_sail), turning + hit_sail * sailing, np.inf))

    ranges = np.linalg.norm(target, axis=-1)
    after_ranges = np.linalg.norm(relative_target, axis=-1)
    unit = target / np.maximum(ranges[:, None], 1e-6)
    normal = np.stack((unit[:, 1], -unit[:, 0]), axis=-1)
    goal_bearing = np.arctan2(unit[:, 0], unit[:, 1])
    future_polar = live if future is None else future
    after_live = _gate_speed(relative_target, live)
    after_future = _gate_speed(relative_target, future_polar)
    distance_left = np.maximum(0.0, after_ranges - radius[:, None])
    delay_left = np.maximum(0.0, until - dt)[:, None]
    # The non-learned comparator uses the best radial two-trim velocity.
    # A small floor ranks approximate progress even when an axis is unsailable.
    direct = distance_left / np.maximum(after_live, .02)
    if future is not None:
        finish_before_shift = (after_live > 0.02) & (direct <= delay_left)
        forecast_distance = np.maximum(0.0, distance_left - delay_left * np.maximum(after_live, -.4))
        after_shift = delay_left + forecast_distance / np.maximum(after_future, .02)
        direct = np.where(finish_before_shift, direct, after_shift)
    total = dt[:, None] + direct
    # Preserve variation among poor options instead of giving all timeout-bound
    # candidates identical supervised scores.
    physics = np.where(np.isfinite(captured), captured, total)
    capped_physics = np.minimum(physics, horizon[:, None] + TIMEOUT_COST_SECONDS)

    def repeat(value):
        return np.broadcast_to(np.broadcast_to(value, (count,))[:, None], (count, width))

    angle_to_wind = goal_bearing - np.radians(live.wind_from)
    angles = np.radians(_signed_angle(bows, live.wind_from))
    features = np.stack((
        repeat(ranges / 1000), repeat(radius / 100), repeat(np.cos(angle_to_wind)),
        repeat(np.sin(angle_to_wind)), repeat(np.asarray(wind_speed) / 10), repeat(wave_height),
        repeat(unit @ live.flow), repeat(normal @ live.flow),
        repeat(unit @ future_polar.flow), repeat(normal @ future_polar.flow),
        repeat(float(future is not None)), repeat(until / 3600),
        repeat(_gate_speed(target, live)), repeat(_gate_speed(target, future_polar)),
        repeat(dt / 300), repeat(horizon / 3600),
        np.sum(velocity * unit[:, None, :], axis=-1),
        np.sum(velocity * normal[:, None, :], axis=-1),
        np.broadcast_to(np.sin(angles), (count, width)),
        np.broadcast_to(np.cos(angles), (count, width)),
        turn_required / 180, np.broadcast_to(flaps / 15, (count, width)),
        after_ranges / 1000, after_live, after_future, capped_physics / 3600,
    ), axis=-1).astype(np.float32)
    return CandidateBatch(features, displacement, relative_target, end_bow, captured, physics, turn_displacement)


@dataclass(frozen=True)
class LearnedMarkPolicy:
    means: np.ndarray
    scales: np.ndarray
    layers: tuple[tuple[np.ndarray, np.ndarray], ...]
    domain: dict[str, tuple[float, float]]

    @classmethod
    def load(cls, path: Path | str) -> LearnedMarkPolicy:
        data = json.loads(Path(path).read_text())
        if data.get("version") != MODEL_VERSION or data.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("Incompatible learned mark policy.")
        means = np.asarray(data["means"], dtype=np.float32)
        scales = np.asarray(data["scales"], dtype=np.float32)
        width = len(FEATURE_NAMES)
        if (means.shape != (width,) or scales.shape != (width,)
                or not np.isfinite(means).all() or not np.isfinite(scales).all() or np.any(scales <= 0.0)):
            raise ValueError("Invalid learned mark normalization.")
        layers = []
        for layer in data["layers"]:
            weights = np.asarray(layer["weights"], dtype=np.float32)
            bias = np.asarray(layer["bias"], dtype=np.float32)
            if (weights.ndim != 2 or weights.shape[1] != width or weights.shape[0] == 0
                    or bias.shape != (weights.shape[0],) or not np.isfinite(weights).all()
                    or not np.isfinite(bias).all()):
                raise ValueError("Invalid learned mark layer.")
            layers.append((weights, bias))
            width = weights.shape[0]
        if not layers or width != 1:
            raise ValueError("The policy must predict one cost per candidate.")
        domain = {key: tuple(map(float, limits)) for key, limits in data["domain"].items()}
        domain_keys = {"range_m", "radius_m", "wind_mps", "current_mps", "wave_m",
                       "decision_s", "forecast_s", "ground_mps"}
        if set(domain) != domain_keys or any(len(bounds) != 2 or not np.isfinite(bounds).all()
                                             or bounds[0] > bounds[1] for bounds in domain.values()):
            raise ValueError("Invalid learned mark domain.")
        return cls(means, scales, tuple(layers), domain)

    def costs(self, features: np.ndarray) -> np.ndarray:
        values = (features - self.means) / self.scales
        for weights, bias in self.layers[:-1]:
            values = np.maximum(values @ weights.T + bias, 0.0)
        weights, bias = self.layers[-1]
        return (values @ weights.T + bias)[..., 0] * 3600.0

    def supported(
        self, distance, arrival_radius, live: PolarSnapshot, future: PolarSnapshot | None,
        wind_speed, wave_height, decision_seconds, forecast_after_seconds,
    ):
        """Vectorized applicability mask for the calibrated forecast domain."""
        polars = tuple(polar for polar in (live, future) if polar is not None)
        inputs = {
            "range_m": np.asarray(distance), "radius_m": np.asarray(arrival_radius),
            "wind_mps": np.asarray(wind_speed),
            "current_mps": max(np.linalg.norm(polar.flow) for polar in polars),
            "wave_m": np.asarray(wave_height), "decision_s": np.asarray(decision_seconds),
            "forecast_s": np.asarray(forecast_after_seconds),
            "ground_mps": max(np.max(np.linalg.norm(polar.velocity, axis=-1)) for polar in polars),
        }
        applicable = np.asarray(distance) > arrival_radius
        for key, value in inputs.items():
            lower, upper = self.domain[key]
            applicable = applicable & (lower <= value) & (value <= upper)
        if future is not None:
            applicable &= np.asarray(forecast_after_seconds) >= decision_seconds
        return applicable

    def rank(
        self, observation: WorldObservation, target, arrival_radius, modes,
        *, decision_seconds: float, future_modes=(), future_current=None,
        future_wind_from=None, forecast_after_seconds=None,
    ) -> tuple[MarkActionScore, ...] | None:
        """Return supported candidate indices, ordered by expected cost.

        Empty future modes mean a steady-weather search. Supply a complete
        forecast triple when offering future modes. A low cost does not confer
        authority to change the candidate or apply a maneuver.
        """
        wind, wave, current = observation.wind, observation.wave, observation.current
        target = np.asarray(target, dtype=float)
        modes = tuple(modes)
        future_modes = tuple(future_modes)
        if wind is None or current is None or wave is None or target.shape != (2,) or not modes:
            return None
        future_params = (future_current, future_wind_from, forecast_after_seconds)
        if (future_modes and any(value is None for value in future_params)
                or not future_modes and any(value is not None for value in future_params)):
            return None
        try:
            live = PolarSnapshot.from_modes(modes, (current.u, current.v), wind.direction)
            future = PolarSnapshot.from_modes(future_modes, future_current, future_wind_from) if future_modes else None
        except (TypeError, ValueError):
            return None
        forecast_after = float(forecast_after_seconds) if future_modes else 0.0
        distance = np.linalg.norm(target)
        if not np.isfinite(observation.boat.heading) or not self.supported(
            distance, arrival_radius, live, future, wind.speed, wave.height, decision_seconds, forecast_after
        ):
            return None
        batch = candidate_batch(
            target, observation.boat.heading, arrival_radius, live, future, wind.speed, wave.height,
            decision_seconds, forecast_after, MAX_SECONDS,
        )
        costs = self.costs(batch.features[0])
        if not np.isfinite(costs).all():
            return None
        costs = np.clip(costs, decision_seconds, MAX_SECONDS + TIMEOUT_COST_SECONDS)
        immediate = batch.captured_after_s[0]
        costs = np.where(np.isfinite(immediate), immediate, costs)
        return tuple(MarkActionScore(int(i), float(costs[i])) for i in np.argsort(costs, kind="stable"))
