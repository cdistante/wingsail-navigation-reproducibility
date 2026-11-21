"""Causal batched weather features for the reduced candidate-policy experiments."""
from __future__ import annotations

from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np
import torch

from world.mark_weather_embedding import EMBEDDING_DIMENSION, FrozenMarkWeatherEncoder
from world.temporal_recovery import HISTORY_FEATURES, HISTORY_STEPS, WEATHER_INTERVAL_SECONDS, RecoveryHoldModel, encode_hold_history


STUDY = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = STUDY / "fresh_weather_encoder.pt"
TRAINING_KIND = "synthetic_causal_weather_denoising_v1"


def forcing_vector(forcing) -> tuple[float, ...]:
    """Encode wind TO, current TO and significant wave height in earth axes."""
    bearing = np.radians(forcing.polar.wind_from)
    east, north = -forcing.wind_speed * np.array((np.sin(bearing), np.cos(bearing)))
    result = (east, north, *forcing.polar.flow, forcing.wave_height)
    if len(result) != HISTORY_FEATURES or not np.isfinite(result).all():
        raise ValueError("A weather stage needs five finite observation components.")
    return tuple(map(float, result))


@lru_cache(maxsize=8192)
def _window_history(first, future, switch_seconds, window):
    recent = np.maximum(0, window + np.arange(1 - HISTORY_STEPS, 1)) * WEATHER_INTERVAL_SECONDS
    history = np.broadcast_to(np.asarray(first), (HISTORY_STEPS, HISTORY_FEATURES)).copy()
    if future is not None:
        history[recent >= switch_seconds] = future
    history.setflags(write=False)
    return history


def causal_history(scenario, elapsed_seconds: float) -> np.ndarray:
    """Return the 12 observed windows through this simulated elapsed time.

    The scenario day is anchored at an aligned five-minute launch window.
    There is no pre-launch weather, so earlier slots repeat the first sample.
    Future-stage weather only becomes an observation when its first simulated
    sample window is reached.
    """
    elapsed_seconds = float(elapsed_seconds)
    if not np.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
        raise ValueError("Weather history requires finite nonnegative elapsed seconds.")
    window = int(elapsed_seconds // WEATHER_INTERVAL_SECONDS)
    first = forcing_vector(scenario.first)
    future = None
    switch = float(scenario.change_at)
    # Avoid reading the future stage at all before it can enter an observed
    # window. A change inside a window is first recorded at the next window.
    if scenario.future is not None:
        if not np.isfinite(switch) or switch < 0:
            raise ValueError("A weather transition needs a nonnegative finite clock.")
        if window * WEATHER_INTERVAL_SECONDS >= switch:
            future = forcing_vector(scenario.future)
    return _window_history(first, future, switch if future is not None else 0.0, window)


def encode_model(model, normalized_history: torch.Tensor) -> torch.Tensor:
    """Use the final-token representation of the original 48-value architecture."""
    return model.encoder(model.projection(normalized_history) + model.position)[:, -1]


def encoder_weights(model) -> dict[str, torch.Tensor]:
    """Exclude the old binary hold head from the representation artifact."""
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            if not name.startswith("head.")}


class WeatherEmbedding:
    """Frozen causal history features, independent of the AIS/policy inputs."""

    def __init__(self, checkpoint: Path | str = DEFAULT_CHECKPOINT, *,
                 legacy: bool = False, batch_size: int = 512):
        checkpoint = Path(checkpoint)
        if batch_size < 1:
            raise ValueError("Weather encoder batch size must be positive.")
        self.batch_size = batch_size
        self.checkpoint = checkpoint.resolve()
        self.checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if legacy:
            original = FrozenMarkWeatherEncoder(checkpoint)
            guidance = original._guidance
            self.model = guidance.model
            self.mean = guidance.scaling["history_mean"]
            self.scale = guidance.scaling["history_scale"]
            self.kind = "legacy_archive_pretraining"
        else:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if saved.get("kind") != TRAINING_KIND:
                raise ValueError("A fresh weather encoder checkpoint is required. Legacy transfer needs explicit opt-in.")
            self.model = RecoveryHoldModel()
            loaded = self.model.load_state_dict(saved["encoder_state"], strict=False)
            if (set(loaded.missing_keys) != {name for name in self.model.state_dict() if name.startswith("head.")}
                    or loaded.unexpected_keys):
                raise ValueError("Incomplete weather encoder checkpoint.")
            self.mean = torch.as_tensor(saved["history_mean"], dtype=torch.float32)
            self.scale = torch.as_tensor(saved["history_scale"], dtype=torch.float32)
            self.kind = TRAINING_KIND
        if (self.mean.shape != (HISTORY_FEATURES,) or self.scale.shape != (HISTORY_FEATURES,)
                or not torch.isfinite(self.mean).all() or not torch.isfinite(self.scale).all()
                or torch.any(self.scale <= 0)):
            raise ValueError("Weather encoder normalization must contain five valid history channels.")
        self.model.requires_grad_(False)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, scenario, elapsed_seconds: float, targets_en) -> np.ndarray:
        """Encode a batch of nonzero east/north target vectors as float32 N×48."""
        targets = np.asarray(targets_en, dtype=float)
        if targets.ndim != 2 or targets.shape[1] != 2:
            raise ValueError("Weather encoder target batch must have shape (N, 2).")
        if not len(targets):
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
        history = causal_history(scenario, elapsed_seconds)
        return self.encode_histories(history, targets)

    @torch.inference_mode()
    def encode_histories(self, history, targets_en) -> np.ndarray:
        """Project and batch already-observed histories through the checkpoint."""
        targets = np.asarray(targets_en, dtype=float)
        history = np.asarray(history, dtype=float)
        if targets.ndim != 2 or targets.shape[1] != 2:
            raise ValueError("Weather encoder target batch must have shape (N, 2).")
        if history.shape == (HISTORY_STEPS, HISTORY_FEATURES):
            history = np.broadcast_to(history, (len(targets), *history.shape))
        if history.shape != (len(targets), HISTORY_STEPS, HISTORY_FEATURES):
            raise ValueError("Weather encoder history batch must have shape (N, 12, 5).")
        if not len(targets):
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
        projected, _ = encode_hold_history(history, targets, False)
        normalized = (torch.as_tensor(projected) - self.mean) / self.scale
        hidden = torch.cat([encode_model(self.model, piece)
                            for piece in normalized.split(self.batch_size)]).numpy()
        if hidden.shape != (len(targets), EMBEDDING_DIMENSION) or not np.isfinite(hidden).all():
            raise ValueError("Weather encoder produced an invalid representation.")
        return hidden

    __call__ = encode
