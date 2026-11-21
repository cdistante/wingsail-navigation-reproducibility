"""Export the frozen hold Transformer's 48-value weather representation.

This is an observed-weather encoder, not a navigation policy. It uses the
original hold model's first valid observation in each UTC five-minute window,
12-window causal history, waypoint-axis projection and checkpoint scaling.
Earlier missing windows repeat the earliest observation; gaps carry the last
observation forward, exactly as in ``TemporalRecoveryGuidance``.

The vector contains wind TO, current TO and wave-height history in the active
waypoint frame. It depends on waypoint direction, but not waypoint range,
vessel motion or actuators, previous hold state, chart, AIS or a forecast.
Those must remain explicit inputs to a downstream candidate-scoring policy.
The original encoder attends over the whole *observed* window; no attention
mask is added because doing so would change its learned representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .temporal_recovery import (
    HISTORY_STEPS,
    WEATHER_INTERVAL_SECONDS,
    TemporalRecoveryGuidance,
    encode_hold_history,
)
from .world import WorldObservation


EMBEDDING_DIMENSION = 48


@dataclass(frozen=True)
class MarkWeatherEmbedding:
    """Frozen representation and the observed windows used to construct it."""

    values: np.ndarray
    decision_utc: str
    history_windows: tuple[int, ...]
    source_windows: tuple[int, ...]


class FrozenMarkWeatherEncoder:
    """One causal observation history per vessel/episode, with frozen weights.

    Call ``observe`` at every available control update, then ``encode`` when a
    downstream policy needs a vector. ``encode`` also observes its argument.
    Create a new encoder to reset an episode. Calls must be chronological;
    requests earlier than an already observed timestamp raise ``ValueError``
    rather than accidentally exposing future weather. A missing current
    weather window or invalid waypoint axis yields ``None``.
    """

    def __init__(self, checkpoint: Path | str) -> None:
        # Reuse the actual loader, history collection and fitted normalization.
        self._guidance = TemporalRecoveryGuidance(checkpoint)
        self._guidance.model.requires_grad_(False)
        self._last_observation_ns: int | None = None

    def observe(self, observation: WorldObservation) -> None:
        """Record weather available now, rejecting a backwards or invalid clock."""
        when = np.datetime64(observation.time, "ns")
        if np.isnat(when):
            raise ValueError("Weather embedding needs a valid observation time.")
        timestamp = int(when.astype(np.int64))
        if self._last_observation_ns is not None and timestamp < self._last_observation_ns:
            raise ValueError("Weather embedding observations must be chronological.")
        self._guidance.observe(observation)
        self._last_observation_ns = timestamp

    @property
    def observed_weather(self) -> tuple[tuple[int, tuple[float, ...]], ...]:
        """Return the same retained causal samples as the original guidance."""
        return self._guidance.observed_weather

    @torch.inference_mode()
    def encode(
        self, observation: WorldObservation, target_axis: np.ndarray,
    ) -> MarkWeatherEmbedding | None:
        """Return the original encoder's 48 values for a nonzero EN target axis."""
        self.observe(observation)
        window = self._guidance._window(observation)
        samples = self.observed_weather
        if not samples or samples[-1][0] != window:
            return None
        target_axis = np.asarray(target_axis, dtype=float)
        if target_axis.shape != (2,):
            return None
        available_windows, weather = zip(*samples, strict=True)
        recent = np.arange(window + 1 - HISTORY_STEPS, window + 1)
        offsets = np.maximum(0, np.searchsorted(available_windows, recent, side="right") - 1)
        try:
            # Previous hold state is an independent head input, not weather.
            projected, _ = encode_hold_history(np.asarray(weather)[offsets], target_axis, False)
        except ValueError:
            return None
        values = torch.from_numpy(projected[None])
        scaling = self._guidance.scaling
        values = (values - scaling["history_mean"]) / scaling["history_scale"]
        model = self._guidance.model
        embedding = model.encoder(model.projection(values) + model.position)[0, -1]
        vector = embedding.numpy().copy()
        if vector.shape != (EMBEDDING_DIMENSION,) or not np.isfinite(vector).all():
            return None
        vector.setflags(write=False)
        return MarkWeatherEmbedding(
            vector,
            str(observation.time.astype("datetime64[ms]")) + "Z",
            tuple(int(value) for value in recent),
            tuple(int(available_windows[offset]) for offset in offsets),
        )

    @torch.inference_mode()
    def original_hold_probability(
        self, embedding: MarkWeatherEmbedding, *, was_holding: bool,
    ) -> float:
        """Apply the unmodified old head for parity checks, not safe steering.

        Physical, confidence, chart and AIS guards are not part of this raw
        classifier probability. The separate hold-state input is normalized
        with the original checkpoint and concatenated after the embedding.
        """
        vector = np.asarray(embedding.values)
        if vector.shape != (EMBEDDING_DIMENSION,) or not np.isfinite(vector).all():
            raise ValueError("Expected a finite 48-value weather embedding.")
        features = torch.tensor(vector[None], dtype=torch.float32)
        context = torch.tensor([[float(was_holding)]], dtype=torch.float32)
        scaling = self._guidance.scaling
        context = (context - scaling["context_mean"]) / scaling["context_scale"]
        logits = self._guidance.model.head(torch.cat((features, context), dim=1))[0]
        return float(torch.softmax(logits / self._guidance.temperature, dim=0)[0])
