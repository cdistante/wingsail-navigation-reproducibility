"""Compact candidate features and a frozen-encoder weather trend decoder."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from world._geo import local_east_north_metres


# Stable native row contract in seasonal_native_training/reproduce/native_branches.py.
NATIVE_WIDTH = 34
EMBEDDING_WIDTH = 48
WEATHER_SHAPE = (12, 5)
DECODER_RIDGE = 2.
MIN_WIND_SPEED_MPS = .5

EXPLICIT_NAMES = (
    "native_forecast_eta_s",
    "first_course_along_goal",
    "following_course_along_goal",
    "bow_along_wind_from",
    "turn_severity",
    "turn_severity_times_surge_mps",
    "current_along_first_course_mps",
    "turn_severity_times_wave_height_m",
    "switch_after_s",
    "absolute_flap_change_deg",
    "log1p_report_clearance_margin_m",
)

TARGET_NAMES = (
    "clockwise_wind_rotation_rad",
    "along_goal_current_change_mps",
    "starboard_of_goal_current_change_mps",
    "wave_height_change_m",
)

LATENT_NAMES = (
    "decoded_wind_rotation_times_crosswind_bow",
    "decoded_along_goal_current_change_times_along_goal_course",
    "decoded_across_goal_current_change_times_across_goal_course",
    "decoded_wave_height_change_times_turn_severity",
)


class _NativeColumn(IntEnum):
    CURRENT_ALONG = 3
    CURRENT_ACROSS = 4
    WAVE_HEIGHT = 5
    SURGE = 7
    FLAP = 11
    FIRST_COURSE_ALONG = 12
    FIRST_COURSE_ACROSS = 13
    BOW_ALONG_WIND = 14
    BOW_ACROSS_WIND = 15
    BOW_ALONG_BOAT = 16
    FORECAST_ETA = 20
    FOLLOWING_COURSE_ALONG = 23
    STEADY_FLAP = 27
    REPORT_CLEARANCE = 28
    SWITCH_AFTER = 33


def _native_rows(record_or_native_rows):
    if isinstance(record_or_native_rows, Mapping):
        record_or_native_rows = [branch["features"] for branch in record_or_native_rows["branches"]]
    rows = np.asarray(record_or_native_rows, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != NATIVE_WIDTH or not np.isfinite(rows).all():
        raise ValueError(f"Expected finite native candidate rows of width {NATIVE_WIDTH}.")
    return rows


def feature_rows(record_or_native_rows):
    """Build instantaneous schedule, maneuvering, forcing, and AIS features."""
    rows = _native_rows(record_or_native_rows)
    column = _NativeColumn
    course_along = rows[:, column.FIRST_COURSE_ALONG]
    course_across = rows[:, column.FIRST_COURSE_ACROSS]
    turn = 1. - rows[:, column.BOW_ALONG_BOAT]
    current_along_course = rows[:, column.CURRENT_ALONG] * course_along + rows[:, column.CURRENT_ACROSS] * course_across
    flap_change = np.abs(rows[:, column.STEADY_FLAP] - rows[:, column.FLAP])
    margin = np.log1p(np.maximum(rows[:, column.REPORT_CLEARANCE], 0.))
    return np.column_stack((rows[:, column.FORECAST_ETA], course_along, rows[:, column.FOLLOWING_COURSE_ALONG],
                            rows[:, column.BOW_ALONG_WIND], turn, turn * rows[:, column.SURGE],
                            current_along_course, turn * rows[:, column.WAVE_HEIGHT],
                            rows[:, column.SWITCH_AFTER], flap_change, margin))


def _history_and_goal(records):
    histories = np.asarray([record["weather_tokens"] for record in records], dtype=float)
    axes = np.asarray([local_east_north_metres(record["goal"]["latitude"], record["goal"]["longitude"],
                                              record["latitude"], record["longitude"])
                       for record in records], dtype=float)
    if histories.shape != (len(records), *WEATHER_SHAPE) or not np.isfinite(histories).all():
        raise ValueError("Expected twelve finite observed five-channel weather tokens per state.")
    if not np.isfinite(axes).all() or np.any(np.linalg.norm(axes, axis=1) <= 1e-8):
        raise ValueError("Weather features require a nonzero finite waypoint vector.")
    return histories, axes


def raw_embeddings(records, encoder):
    """Use saved frozen embeddings or encode the immutable causal weather tokens."""
    records = list(records)
    values = np.empty((len(records), EMBEDDING_WIDTH), dtype=float)
    encode = []
    for index, record in enumerate(records):
        embedding = record.get("weather_embedding")
        if embedding is None:
            encode.append(index)
        else:
            embedding = np.asarray(embedding, dtype=float)
            if embedding.shape != (EMBEDDING_WIDTH,):
                raise ValueError("A saved state weather embedding must have 48 components.")
            values[index] = embedding
    if encode:
        if encoder is None:
            raise ValueError("The frozen weather encoder is required for missing embeddings.")
        histories, axes = _history_and_goal([records[index] for index in encode])
        values[encode] = encoder.encode_histories(histories, axes)
    if not np.isfinite(values).all():
        raise ValueError("State weather embeddings must be finite.")
    return values


def weather_targets(records):
    """Observed oldest-to-latest changes in the 55-minute as-of history window."""
    records = list(records)
    if not records:
        return np.empty((0, len(TARGET_NAMES)), dtype=float)
    history, goal_axes = _history_and_goal(records)
    first, last = history[:, 0], history[:, -1]
    old_wind, new_wind = first[:, :2], last[:, :2]
    old_east, old_north = old_wind.T
    new_east, new_north = new_wind.T
    clockwise = np.arctan2(old_north * new_east - old_east * new_north,
                           np.sum(old_wind * new_wind, axis=1))
    # Wind rotation has no physical meaning if either endpoint is nearly calm.
    calm = np.minimum(np.linalg.norm(old_wind, axis=1), np.linalg.norm(new_wind, axis=1)) < MIN_WIND_SPEED_MPS
    clockwise[calm] = 0.
    along = goal_axes / np.linalg.norm(goal_axes, axis=1, keepdims=True)
    across = np.column_stack((along[:, 1], -along[:, 0]))
    current_change = last[:, 2:4] - first[:, 2:4]
    return np.column_stack((clockwise, np.sum(current_change * along, axis=1),
                            np.sum(current_change * across, axis=1), last[:, 4] - first[:, 4]))


@dataclass(frozen=True)
class WeatherDecoder:
    embedding_mean: np.ndarray
    embedding_scale: np.ndarray
    target_mean: np.ndarray
    coefficients: np.ndarray

    def predict(self, embedding):
        values = np.asarray(embedding, dtype=float)
        if values.ndim not in (1, 2) or values.shape[-1] != EMBEDDING_WIDTH or not np.isfinite(values).all():
            raise ValueError("Weather decoder inputs must contain finite 48-component embeddings.")
        normalized = (values - self.embedding_mean) / self.embedding_scale
        return normalized @ self.coefficients + self.target_mean

    def save(self, path):
        np.savez_compressed(path, embedding_mean=self.embedding_mean, embedding_scale=self.embedding_scale,
                            target_mean=self.target_mean, coefficients=self.coefficients)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as saved:
            decoder = cls(saved["embedding_mean"], saved["embedding_scale"], saved["target_mean"],
                          saved["coefficients"])
        shapes = ((EMBEDDING_WIDTH,), (EMBEDDING_WIDTH,), (len(TARGET_NAMES),),
                  (EMBEDDING_WIDTH, len(TARGET_NAMES)))
        arrays = (decoder.embedding_mean, decoder.embedding_scale, decoder.target_mean, decoder.coefficients)
        if any(value.shape != shape or not np.isfinite(value).all() for value, shape in zip(arrays, shapes)):
            raise ValueError("Invalid compact weather decoder arrays.")
        if np.any(decoder.embedding_scale <= 0.):
            raise ValueError("Weather decoder scales must be positive.")
        return decoder


def fit_decoder(training_records, training_embeddings):
    """Fit to past weather changes with equal weight per registered weather group."""
    records = list(training_records)
    if not records or any(record.get("split") != "train" for record in records):
        raise ValueError("Weather decoder fitting requires a nonempty training-only collection.")
    values = np.asarray(training_embeddings, dtype=float)
    if values.shape != (len(records), EMBEDDING_WIDTH) or not np.isfinite(values).all():
        raise ValueError("Provide one finite 48-component embedding per training state.")
    targets = weather_targets(records)
    groups = [str(record.get("group") or record.get("departure_group") or record["scenario"]) for record in records]
    _, membership, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weight = 1. / counts[membership]
    mean = np.average(values, weights=weight, axis=0)
    scale = np.maximum(np.sqrt(np.average((values - mean)**2, weights=weight, axis=0)), .05)
    target_mean = np.average(targets, weights=weight, axis=0)
    design = ((values - mean) / scale) * np.sqrt(weight[:, None])
    centered = (targets - target_mean) * np.sqrt(weight[:, None])
    coefficients = np.linalg.solve(design.T @ design + DECODER_RIDGE * np.eye(EMBEDDING_WIDTH), design.T @ centered)
    decoder = WeatherDecoder(mean, scale, target_mean, coefficients)
    error = np.abs(decoder.predict(values) - targets)
    return decoder, {"training_states": len(records), "training_departure_groups": len(counts),
                     "ridge": DECODER_RIDGE, "embedding_width": EMBEDDING_WIDTH,
                     "target_names": list(TARGET_NAMES),
                     "training_group_weighted_mae": np.average(error, weights=weight, axis=0).tolist(),
                     "training_group_weighted_mean_baseline_mae": np.average(np.abs(targets - target_mean),
                                                                               weights=weight, axis=0).tolist()}


def latent_rows(record_or_native_rows, embedding_48, decoder):
    """Project the four shared past-weather estimates onto each candidate maneuver."""
    rows = _native_rows(record_or_native_rows)
    trend = decoder.predict(embedding_48)
    if trend.shape != (len(TARGET_NAMES),):
        raise ValueError("A tactical state needs exactly one decoded weather vector.")
    column = _NativeColumn
    interactions = np.column_stack((rows[:, column.BOW_ACROSS_WIND], rows[:, column.FIRST_COURSE_ALONG],
                                    rows[:, column.FIRST_COURSE_ACROSS], 1. - rows[:, column.BOW_ALONG_BOAT]))
    return interactions * trend
