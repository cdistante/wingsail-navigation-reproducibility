"""Temporal Transformer data/model utilities for AIS course prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F


FEATURE_NAMES = (
    "east_from_present_km",
    "north_from_present_km",
    "speed_15mps",
    "course_sin",
    "course_cos",
)


@dataclass(frozen=True)
class AISSequenceDataset:
    """Leakage-safe temporal AIS samples and vessel-level split labels."""

    features: np.ndarray
    targets: np.ndarray
    mmsi: np.ndarray
    timestamps_ns: np.ndarray
    split: np.ndarray
    horizons_minutes: tuple[int, ...]
    history_minutes: int
    sample_interval_seconds: int

    def save(self, path: Path | str) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            features=self.features.astype(np.float32),
            targets=self.targets.astype(np.float32),
            mmsi=self.mmsi.astype(np.int64),
            timestamps_ns=self.timestamps_ns.astype(np.int64),
            split=self.split.astype("U5"),
            horizons_minutes=np.asarray(self.horizons_minutes, dtype=np.int16),
            history_minutes=np.asarray(self.history_minutes, dtype=np.int16),
            sample_interval_seconds=np.asarray(self.sample_interval_seconds, dtype=np.int16),
            feature_names=np.asarray(FEATURE_NAMES),
        )
        return destination

    @classmethod
    def load(cls, path: Path | str) -> "AISSequenceDataset":
        with np.load(path) as archive:
            feature_names = tuple(str(value) for value in archive["feature_names"])
            if feature_names != FEATURE_NAMES:
                raise ValueError(
                    f"Unsupported AIS feature schema {feature_names}; expected {FEATURE_NAMES}."
                )
            return cls(
                features=archive["features"],
                targets=archive["targets"],
                mmsi=archive["mmsi"],
                timestamps_ns=archive["timestamps_ns"],
                split=archive["split"].astype("U5"),
                horizons_minutes=tuple(int(value) for value in archive["horizons_minutes"]),
                history_minutes=int(archive["history_minutes"].item()),
                sample_interval_seconds=int(archive["sample_interval_seconds"].item()),
            )


def build_ais_sequence_dataset(
    csv_path: Path | str,
    *,
    history_minutes: int = 30,
    horizons_minutes: tuple[int, ...] = (5, 10, 15),
    sample_interval_seconds: int = 60,
    max_interpolation_gap_seconds: int = 900,
    minimum_future_displacement_metres: float = 25.0,
    window_stride_minutes: int = 1,
    seed: int = 42,
) -> AISSequenceDataset:
    """Build uniformly sampled vessel histories and future direction labels."""

    if history_minutes < 1:
        raise ValueError("history_minutes must be positive.")
    if not horizons_minutes or any(value < 1 for value in horizons_minutes):
        raise ValueError("horizons_minutes must contain positive values.")
    if tuple(sorted(set(horizons_minutes))) != horizons_minutes:
        raise ValueError("horizons_minutes must be unique and increasing.")
    if sample_interval_seconds < 1 or 60 % sample_interval_seconds != 0:
        raise ValueError("sample_interval_seconds must be a positive divisor of 60.")
    if max_interpolation_gap_seconds < sample_interval_seconds:
        raise ValueError("max_interpolation_gap_seconds is too short.")
    if minimum_future_displacement_metres < 0.0:
        raise ValueError("minimum_future_displacement_metres must be non-negative.")
    if window_stride_minutes < 1:
        raise ValueError("window_stride_minutes must be positive.")

    source = Path(csv_path)
    if not source.exists():
        raise RuntimeError(f"AIS CSV not found: {source}.")
    frame = pd.read_csv(
        source,
        usecols=["receiver_timestamp", "mmsi", "latitude", "longitude", "speed", "course"],
    )
    frame["receiver_timestamp"] = pd.to_datetime(
        frame["receiver_timestamp"], utc=True, errors="coerce"
    )
    for name in ("mmsi", "latitude", "longitude", "speed", "course"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame = frame.dropna(
        subset=["receiver_timestamp", "mmsi", "latitude", "longitude", "speed", "course"]
    )
    frame = frame[
        frame["latitude"].between(-90.0, 90.0)
        & frame["longitude"].between(-180.0, 180.0)
        & frame["speed"].ge(0.0)
    ].copy()
    if frame.empty:
        raise ValueError(f"AIS CSV contains no usable dynamic reports: {source}.")
    frame["mmsi"] = frame["mmsi"].astype(np.int64)
    frame["course"] %= 360.0
    frame = frame.sort_values(["mmsi", "receiver_timestamp"])

    history_steps = history_minutes * 60 // sample_interval_seconds + 1
    horizon_steps = tuple(
        minutes * 60 // sample_interval_seconds for minutes in horizons_minutes
    )
    stride_steps = window_stride_minutes * 60 // sample_interval_seconds

    feature_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    mmsi_batches: list[np.ndarray] = []
    time_batches: list[np.ndarray] = []
    for vessel_mmsi, track in frame.groupby("mmsi", sort=True):
        vessel = _resample_track(
            track,
            sample_interval_seconds=sample_interval_seconds,
            max_interpolation_gap_seconds=max_interpolation_gap_seconds,
        )
        if vessel is None:
            continue
        features, targets, times = _track_windows(
            vessel,
            history_steps=history_steps,
            horizon_steps=horizon_steps,
            stride_steps=stride_steps,
            minimum_future_displacement_metres=minimum_future_displacement_metres,
        )
        if not len(features):
            continue
        feature_batches.append(features)
        target_batches.append(targets)
        mmsi_batches.append(np.full(len(features), int(vessel_mmsi), dtype=np.int64))
        time_batches.append(times)

    if not feature_batches:
        raise ValueError("AIS data produced no complete history/future sequence windows.")
    features = np.concatenate(feature_batches)
    targets = np.concatenate(target_batches)
    mmsi = np.concatenate(mmsi_batches)
    timestamps_ns = np.concatenate(time_batches)
    split = vessel_level_split(mmsi, seed=seed)
    return AISSequenceDataset(
        features=features,
        targets=targets,
        mmsi=mmsi,
        timestamps_ns=timestamps_ns,
        split=split,
        horizons_minutes=horizons_minutes,
        history_minutes=history_minutes,
        sample_interval_seconds=sample_interval_seconds,
    )


def vessel_level_split(
    sample_mmsi: np.ndarray,
    *,
    seed: int = 42,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> np.ndarray:
    """Assign whole vessels to train/validation/test partitions."""

    if not 0.0 < train_ratio < 1.0 or not 0.0 < validation_ratio < 1.0:
        raise ValueError("Split ratios must lie in (0, 1).")
    if train_ratio + validation_ratio >= 1.0:
        raise ValueError("Train and validation ratios must leave a test partition.")
    vessels = np.unique(np.asarray(sample_mmsi, dtype=np.int64))
    if len(vessels) < 3:
        raise ValueError("At least three vessels are required for vessel-level splits.")
    shuffled = vessels.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    train_end = min(
        len(shuffled) - 2,
        max(1, int(round(len(shuffled) * train_ratio))),
    )
    validation_end = min(
        len(shuffled) - 1,
        max(
            train_end + 1,
            int(round(len(shuffled) * (train_ratio + validation_ratio))),
        ),
    )
    train_vessels = set(shuffled[:train_end])
    validation_vessels = set(shuffled[train_end:validation_end])
    return np.asarray(
        [
            "train"
            if int(value) in train_vessels
            else "val"
            if int(value) in validation_vessels
            else "test"
            for value in sample_mmsi
        ]
    )


def _resample_track(
    track: pd.DataFrame,
    *,
    sample_interval_seconds: int,
    max_interpolation_gap_seconds: int,
) -> dict[str, np.ndarray] | None:
    grouped = (
        track.assign(
            timestamp_seconds=track["receiver_timestamp"].map(
                lambda value: value.timestamp()
            )
        )
        .groupby("timestamp_seconds", as_index=False)
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            speed=("speed", "mean"),
            course_sin=("course", lambda values: np.sin(np.radians(values)).mean()),
            course_cos=("course", lambda values: np.cos(np.radians(values)).mean()),
        )
        .sort_values("timestamp_seconds")
    )
    source_seconds = grouped["timestamp_seconds"].to_numpy(dtype=float)
    if len(source_seconds) < 2:
        return None
    interval = float(sample_interval_seconds)
    start = np.ceil(source_seconds[0] / interval) * interval
    end = np.floor(source_seconds[-1] / interval) * interval
    if end < start:
        return None
    grid_seconds = np.arange(start, end + interval * 0.5, interval)
    right = np.searchsorted(source_seconds, grid_seconds, side="left")
    exact = (right < len(source_seconds)) & np.isclose(
        source_seconds[np.minimum(right, len(source_seconds) - 1)], grid_seconds
    )
    left = np.maximum(right - 1, 0)
    right_clipped = np.minimum(right, len(source_seconds) - 1)
    bracket_gap = source_seconds[right_clipped] - source_seconds[left]
    valid = exact | (
        (right > 0)
        & (right < len(source_seconds))
        & (bracket_gap <= max_interpolation_gap_seconds)
    )
    values = {}
    for name in ("latitude", "longitude", "speed", "course_sin", "course_cos"):
        values[name] = np.interp(
            grid_seconds,
            source_seconds,
            grouped[name].to_numpy(dtype=float),
        )
    course_norm = np.hypot(values["course_sin"], values["course_cos"])
    valid &= course_norm > 1e-6
    values["course_sin"] /= np.maximum(course_norm, 1e-6)
    values["course_cos"] /= np.maximum(course_norm, 1e-6)
    values["valid"] = valid
    values["timestamp_ns"] = np.rint(grid_seconds * 1e9).astype(np.int64)
    return values


def _track_windows(
    vessel: dict[str, np.ndarray],
    *,
    history_steps: int,
    horizon_steps: tuple[int, ...],
    stride_steps: int,
    minimum_future_displacement_metres: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    maximum_horizon = max(horizon_steps)
    sample_features = []
    sample_targets = []
    sample_times = []
    for current in range(history_steps - 1, len(vessel["valid"]) - maximum_horizon, stride_steps):
        history_slice = slice(current - history_steps + 1, current + 1)
        future_indices = np.asarray([current + value for value in horizon_steps])
        if not vessel["valid"][history_slice].all() or not vessel["valid"][future_indices].all():
            continue
        current_latitude = float(vessel["latitude"][current])
        current_longitude = float(vessel["longitude"][current])
        east, north = _local_offsets(
            vessel["latitude"][history_slice],
            vessel["longitude"][history_slice],
            current_latitude,
            current_longitude,
        )
        features = np.column_stack(
            (
                east / 1_000.0,
                north / 1_000.0,
                vessel["speed"][history_slice] / 15.0,
                vessel["course_sin"][history_slice],
                vessel["course_cos"][history_slice],
            )
        )
        future_east, future_north = _local_offsets(
            vessel["latitude"][future_indices],
            vessel["longitude"][future_indices],
            current_latitude,
            current_longitude,
        )
        magnitude = np.hypot(future_east, future_north)
        if np.any(magnitude < minimum_future_displacement_metres):
            continue
        targets = np.column_stack((future_east / magnitude, future_north / magnitude))
        sample_features.append(features.astype(np.float32))
        sample_targets.append(targets.astype(np.float32))
        sample_times.append(vessel["timestamp_ns"][current])
    if not sample_features:
        return (
            np.empty((0, history_steps, len(FEATURE_NAMES)), dtype=np.float32),
            np.empty((0, len(horizon_steps), 2), dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )
    return (
        np.stack(sample_features),
        np.stack(sample_targets),
        np.asarray(sample_times, dtype=np.int64),
    )


def _local_offsets(
    latitude: Iterable[float],
    longitude: Iterable[float],
    origin_latitude: float,
    origin_longitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    latitude = np.asarray(latitude, dtype=float)
    longitude = np.asarray(longitude, dtype=float)
    north = np.radians(latitude - origin_latitude) * 6_371_000.0
    east = (
        np.radians(longitude - origin_longitude)
        * 6_371_000.0
        * np.cos(np.radians(origin_latitude))
    )
    return east, north


class AISCourseTransformer(nn.Module):
    """Predict multiple future course vectors from a recent AIS sequence."""

    def __init__(
        self,
        *,
        input_features: int = len(FEATURE_NAMES),
        horizons: int = 3,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 192,
        dropout: float = 0.1,
        maximum_sequence_length: int = 128,
    ) -> None:
        super().__init__()
        self.horizons = int(horizons)
        self.maximum_sequence_length = int(maximum_sequence_length)
        self.input_projection = nn.Sequential(
            nn.Linear(input_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.maximum_sequence_length, d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.horizons * 2),
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("AIS features must have shape [batch, sequence, features].")
        sequence_length = features.shape[1]
        if sequence_length > self.maximum_sequence_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds {self.maximum_sequence_length}."
            )
        encoded = self.input_projection(features.float())
        encoded = encoded + self.position_embedding[:, :sequence_length]
        encoded = self.encoder(encoded)
        return self.head(encoded[:, -1]).reshape(-1, self.horizons, 2)


def direction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = F.normalize(prediction, dim=-1, eps=1e-8)
    target = F.normalize(target, dim=-1, eps=1e-8)
    cosine_loss = 1.0 - (prediction * target).sum(dim=-1)
    return cosine_loss.mean(), cosine_loss.detach()


def angular_errors_degrees(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = F.normalize(prediction, dim=-1, eps=1e-8)
    target = F.normalize(target, dim=-1, eps=1e-8)
    cosine = (prediction * target).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def direction_to_course_degrees(direction: np.ndarray) -> np.ndarray:
    """Convert [..., east/north] vectors to true course degrees."""

    direction = np.asarray(direction, dtype=float)
    if direction.shape[-1] != 2:
        raise ValueError("Direction vectors must end with east/north components.")
    return np.degrees(np.arctan2(direction[..., 0], direction[..., 1])) % 360.0


class AISCoursePredictor:
    """Load a trained checkpoint and predict future true courses."""

    def __init__(self, checkpoint_path: Path | str, *, device: str = "auto") -> None:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        self.device = torch.device(
            "cuda"
            if device == "cuda" or (device == "auto" and torch.cuda.is_available())
            else "cpu"
        )
        self.path = Path(checkpoint_path)
        if not self.path.exists():
            raise RuntimeError(f"AIS course checkpoint not found: {self.path}.")
        checkpoint = torch.load(self.path, map_location=self.device, weights_only=False)
        self.model = AISCourseTransformer(**checkpoint["model_config"]).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.horizons_minutes = tuple(int(value) for value in checkpoint["horizons_minutes"])
        self.history_minutes = int(checkpoint["history_minutes"])
        self.sample_interval_seconds = int(checkpoint["sample_interval_seconds"])
        self.feature_mean = np.asarray(checkpoint["feature_mean"], dtype=np.float32)
        self.feature_standard_deviation = np.asarray(
            checkpoint["feature_standard_deviation"], dtype=np.float32
        )

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return normalized east/north directions and true courses."""

        features = np.asarray(features, dtype=np.float32)
        expected_steps = self.history_minutes * 60 // self.sample_interval_seconds + 1
        if features.shape != (expected_steps, len(FEATURE_NAMES)):
            raise ValueError(
                f"Expected AIS history shape {(expected_steps, len(FEATURE_NAMES))}, "
                f"received {features.shape}."
            )
        directions, courses = self.predict_batch(features[np.newaxis, ...])
        return directions[0], courses[0]

    def predict_batch(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict a batch shaped ``[vessels, history, features]``."""

        features = np.asarray(features, dtype=np.float32)
        expected_steps = self.history_minutes * 60 // self.sample_interval_seconds + 1
        expected_shape = (expected_steps, len(FEATURE_NAMES))
        if features.ndim != 3 or features.shape[1:] != expected_shape:
            raise ValueError(
                f"Expected batched AIS history shape [vessels, {expected_shape[0]}, "
                f"{expected_shape[1]}], received {features.shape}."
            )
        normalized = (features - self.feature_mean) / self.feature_standard_deviation
        tensor = torch.from_numpy(normalized).to(self.device)
        with torch.no_grad():
            prediction = F.normalize(self.model(tensor), dim=-1).cpu().numpy()
        return prediction, direction_to_course_degrees(prediction)


@dataclass(frozen=True)
class AISCourseForecast:
    """Multi-horizon direction forecast for one AIS target."""

    mmsi: int
    history_end: pd.Timestamp
    horizons_minutes: tuple[int, ...]
    directions: np.ndarray
    courses_degrees: np.ndarray


class AISCourseReplayForecaster:
    """Batch forecasts from one recorded AIS CSV with lazy track resampling."""

    def __init__(
        self,
        checkpoint_path: Path | str,
        csv_path: Path | str,
        *,
        device: str = "auto",
        max_interpolation_gap_seconds: int = 900,
    ) -> None:
        self.predictor = AISCoursePredictor(checkpoint_path, device=device)
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise RuntimeError(f"AIS CSV not found: {self.csv_path}.")
        self.max_interpolation_gap_seconds = int(max_interpolation_gap_seconds)
        frame = pd.read_csv(
            self.csv_path,
            usecols=[
                "receiver_timestamp",
                "mmsi",
                "latitude",
                "longitude",
                "speed",
                "course",
            ],
        )
        frame["receiver_timestamp"] = pd.to_datetime(
            frame["receiver_timestamp"], utc=True, errors="coerce"
        )
        for name in ("mmsi", "latitude", "longitude", "speed", "course"):
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
        frame = frame.dropna(
            subset=[
                "receiver_timestamp",
                "mmsi",
                "latitude",
                "longitude",
                "speed",
                "course",
            ]
        )
        frame["mmsi"] = frame["mmsi"].astype(np.int64)
        self._tracks = {
            int(mmsi): track.sort_values("receiver_timestamp").reset_index(drop=True)
            for mmsi, track in frame.groupby("mmsi", sort=False)
        }
        self._resampled: dict[int, dict[str, np.ndarray] | None] = {}

    def predict_targets(
        self,
        mmsi_values: Iterable[int],
        when: str | pd.Timestamp | np.datetime64,
    ) -> tuple[dict[int, AISCourseForecast], dict[str, int]]:
        """Forecast all eligible MMSIs and report expected skip reasons."""

        query = pd.Timestamp(when)
        if query.tzinfo is None:
            query = query.tz_localize("UTC")
        else:
            query = query.tz_convert("UTC")
        requested_mmsi = tuple(dict.fromkeys(int(value) for value in mmsi_values))
        features = []
        eligible_mmsi = []
        history_ends = []
        missing = 0
        insufficient_history = 0
        reporting_gap = 0
        for raw_mmsi in requested_mmsi:
            if raw_mmsi not in self._tracks:
                missing += 1
                continue
            vessel = self._resampled_track(raw_mmsi)
            if vessel is None:
                insufficient_history += 1
                continue
            try:
                vessel_features, history_end = _history_features_from_resampled(
                    vessel,
                    query,
                    history_minutes=self.predictor.history_minutes,
                    sample_interval_seconds=self.predictor.sample_interval_seconds,
                    max_interpolation_gap_seconds=self.max_interpolation_gap_seconds,
                    mmsi=raw_mmsi,
                )
            except InsufficientAISTrackHistory:
                insufficient_history += 1
                continue
            except AISTrackReportingGap:
                reporting_gap += 1
                continue
            features.append(vessel_features)
            eligible_mmsi.append(raw_mmsi)
            history_ends.append(history_end)
        forecasts: dict[int, AISCourseForecast] = {}
        if features:
            directions, courses = self.predictor.predict_batch(np.stack(features))
            for index, vessel_mmsi in enumerate(eligible_mmsi):
                forecasts[vessel_mmsi] = AISCourseForecast(
                    mmsi=vessel_mmsi,
                    history_end=history_ends[index],
                    horizons_minutes=self.predictor.horizons_minutes,
                    directions=directions[index],
                    courses_degrees=courses[index],
                )
        return forecasts, {
            "requested": len(requested_mmsi),
            "forecasted": len(forecasts),
            "missing": missing,
            "insufficient_history": insufficient_history,
            "reporting_gap": reporting_gap,
        }

    def _resampled_track(self, mmsi: int) -> dict[str, np.ndarray] | None:
        if mmsi not in self._resampled:
            self._resampled[mmsi] = _resample_track(
                self._tracks[mmsi],
                sample_interval_seconds=self.predictor.sample_interval_seconds,
                max_interpolation_gap_seconds=self.max_interpolation_gap_seconds,
            )
        return self._resampled[mmsi]


class InsufficientAISTrackHistory(ValueError):
    """The requested target lacks a complete history window."""


class AISTrackReportingGap(ValueError):
    """The target history contains a reporting gap beyond the configured limit."""


def history_features_for_prediction(
    csv_path: Path | str,
    mmsi: int,
    when: str | pd.Timestamp,
    *,
    history_minutes: int,
    sample_interval_seconds: int,
    max_interpolation_gap_seconds: int = 900,
) -> tuple[np.ndarray, pd.Timestamp]:
    """Build the latest complete history ending at or before ``when``."""

    frame = pd.read_csv(
        csv_path,
        usecols=["receiver_timestamp", "mmsi", "latitude", "longitude", "speed", "course"],
    )
    frame["mmsi"] = pd.to_numeric(frame["mmsi"], errors="coerce")
    frame = frame[frame["mmsi"] == int(mmsi)].copy()
    if frame.empty:
        raise ValueError(f"MMSI {mmsi} is not present in AIS CSV: {csv_path}.")
    frame["receiver_timestamp"] = pd.to_datetime(
        frame["receiver_timestamp"], utc=True, errors="coerce"
    )
    for name in ("latitude", "longitude", "speed", "course"):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame = frame.dropna(
        subset=["receiver_timestamp", "latitude", "longitude", "speed", "course"]
    ).sort_values("receiver_timestamp")
    vessel = _resample_track(
        frame,
        sample_interval_seconds=sample_interval_seconds,
        max_interpolation_gap_seconds=max_interpolation_gap_seconds,
    )
    if vessel is None:
        raise ValueError(f"MMSI {mmsi} has insufficient reports for interpolation.")
    return _history_features_from_resampled(
        vessel,
        pd.Timestamp(when),
        history_minutes=history_minutes,
        sample_interval_seconds=sample_interval_seconds,
        max_interpolation_gap_seconds=max_interpolation_gap_seconds,
        mmsi=mmsi,
    )


def _history_features_from_resampled(
    vessel: dict[str, np.ndarray],
    when: pd.Timestamp,
    *,
    history_minutes: int,
    sample_interval_seconds: int,
    max_interpolation_gap_seconds: int,
    mmsi: int,
) -> tuple[np.ndarray, pd.Timestamp]:
    query = pd.Timestamp(when)
    if query.tzinfo is None:
        query = query.tz_localize("UTC")
    else:
        query = query.tz_convert("UTC")
    query_ns = query.as_unit("ns").value
    current = int(np.searchsorted(vessel["timestamp_ns"], query_ns, side="right") - 1)
    history_steps = history_minutes * 60 // sample_interval_seconds + 1
    if current < history_steps - 1:
        raise InsufficientAISTrackHistory(
            f"MMSI {mmsi} has insufficient history before {query.isoformat()}."
        )
    history_slice = slice(current - history_steps + 1, current + 1)
    if not vessel["valid"][history_slice].all():
        raise AISTrackReportingGap(
            f"MMSI {mmsi} history before {query.isoformat()} contains a reporting gap longer than "
            f"{max_interpolation_gap_seconds} seconds."
        )
    current_latitude = float(vessel["latitude"][current])
    current_longitude = float(vessel["longitude"][current])
    east, north = _local_offsets(
        vessel["latitude"][history_slice],
        vessel["longitude"][history_slice],
        current_latitude,
        current_longitude,
    )
    features = np.column_stack(
        (
            east / 1_000.0,
            north / 1_000.0,
            vessel["speed"][history_slice] / 15.0,
            vessel["course_sin"][history_slice],
            vessel["course_cos"][history_slice],
        )
    ).astype(np.float32)
    timestamp = pd.Timestamp(vessel["timestamp_ns"][current], unit="ns", tz="UTC")
    return features, timestamp
