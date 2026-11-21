"""Observation-only temporal guidance for holding or resuming classical sailing."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .scenario import Waypoint
from .world import WorldObservation


HISTORY_STEPS = 12
WEATHER_INTERVAL_SECONDS = 300
HISTORY_FEATURES = 5
CONTEXT_FEATURES = 1


def encode_hold_history(env_history, target_axis, was_holding) -> tuple[np.ndarray, np.ndarray]:
    """Project observed wind-to and current EN velocities onto the waypoint frame."""
    env = np.asarray(env_history, dtype=float)
    target = np.asarray(target_axis, dtype=float)
    if env.ndim < 2 or env.shape[-1] != HISTORY_FEATURES or not env.shape[-2] or target.shape[-1:] != (2,):
        raise ValueError("Expected weather samples [wind_u, wind_v, current_u, current_v, wave_height] and a target vector.")
    length = np.linalg.norm(target, axis=-1)
    if not np.isfinite(env).all() or not np.isfinite(target).all() or np.any(length <= 1e-8):
        raise ValueError("Hold guidance needs finite weather and a nonzero target vector.")
    along = target / length[..., None]
    across = np.stack((along[..., 1], -along[..., 0]), axis=-1)
    env = env[..., -HISTORY_STEPS:, :]
    if env.shape[-2] < HISTORY_STEPS:
        pad = np.repeat(env[..., :1, :], HISTORY_STEPS - env.shape[-2], axis=-2)
        env = np.concatenate((pad, env), axis=-2)
    wind, current = env[..., :2], env[..., 2:4]
    history = np.stack((
        np.sum(wind * along[..., None, :], axis=-1), np.sum(wind * across[..., None, :], axis=-1),
        np.sum(current * along[..., None, :], axis=-1), np.sum(current * across[..., None, :], axis=-1),
        np.broadcast_to(env[..., 4], np.broadcast_shapes(env.shape[:-1], (*length.shape, 1))),
    ), axis=-1)
    context = np.broadcast_to(np.asarray(was_holding, dtype=float), history.shape[:-2])[..., None]
    return history.astype(np.float32), context.astype(np.float32)


class RecoveryHoldModel(nn.Module):
    """Encode recent observed weather and classify sail (0 is hold, 1 sail)."""

    def __init__(self, history_features: int = HISTORY_FEATURES, context_features: int = CONTEXT_FEATURES) -> None:
        super().__init__()
        width = 48
        self.projection = nn.Linear(history_features, width)
        self.position = nn.Parameter(torch.empty(HISTORY_STEPS, width))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            width, nhead=4, dim_feedforward=96, dropout=0.10,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=2, norm=nn.LayerNorm(width), enable_nested_tensor=False,
        )
        self.head = nn.Sequential(
            nn.Linear(width + context_features, 96), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(96, 48), nn.GELU(), nn.Linear(48, 2),
        )

    def forward(self, history: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        features = self.encoder(self.projection(history) + self.position)[:, -1]
        return self.head(torch.cat((features, context), dim=1))


@dataclass(frozen=True)
class TemporalHoldDecision:
    window_utc: str
    hold: bool
    classical_hold: bool
    p_hold: float | None = None
    raw_hold: bool | None = None
    fallback_reason: str | None = None
    decision_utc: str | None = None
    release_guard: str | None = None

    def telemetry(self) -> dict[str, str | float | None]:
        source = "learned"
        if self.release_guard is not None:
            source = "physical release"
        elif self.fallback_reason is not None:
            source = "classical fallback"
        return {
            "temporal_recovery_window_utc": self.window_utc,
            "temporal_recovery_decision_utc": self.decision_utc,
            "temporal_recovery_raw_action": None if self.raw_hold is None else "hold" if self.raw_hold else "sail",
            "temporal_recovery_classical_action": "hold" if self.classical_hold else "sail",
            "temporal_recovery_applied_action": "hold" if self.hold else "sail",
            "temporal_recovery_p_hold": self.p_hold,
            "temporal_recovery_source": source,
            "temporal_recovery_fallback_reason": self.fallback_reason,
            "temporal_recovery_release_guard": self.release_guard,
        }


class TemporalRecoveryGuidance:
    """Use a trained hold policy from observed weather with physical vetoes."""

    minimum_release_vmg_mps = 0.0
    maximum_hold_vmg_mps = 0.12

    def __init__(self, checkpoint: Path | str, *, confirm_release: bool = False) -> None:
        data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model = RecoveryHoldModel()
        self.model.load_state_dict(data["weights"])
        self.model.eval()
        self.scaling = data["scaling"]
        for name, size in (("history", HISTORY_FEATURES), ("context", CONTEXT_FEATURES)):
            mean = torch.as_tensor(self.scaling[f"{name}_mean"], dtype=torch.float32)
            scale = torch.as_tensor(self.scaling[f"{name}_scale"], dtype=torch.float32)
            if mean.shape != (size,) or scale.shape != (size,) or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or not (scale > 0).all():
                raise ValueError(f"Invalid temporal recovery {name} normalization.")
            self.scaling[f"{name}_mean"], self.scaling[f"{name}_scale"] = mean, scale
        self.temperature = float(data["temperature"])
        self.confidence_threshold = data["confidence_threshold"]
        if not np.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("Temporal recovery temperature must be finite and positive.")
        if self.confidence_threshold is not None and not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("Temporal recovery confidence threshold must be in [0, 1] or None.")
        self._history: deque[tuple[int, np.ndarray]] = deque(maxlen=HISTORY_STEPS)
        self.confirm_release = confirm_release
        self._release_goal: Waypoint | None = None
        self._release_ready_window: int | None = None
        self._release_confirmed = False

    @staticmethod
    def _window(observation: WorldObservation) -> int:
        epoch_s = int(observation.time.astype("datetime64[s]").astype(np.int64))
        return epoch_s // WEATHER_INTERVAL_SECONDS

    def observe(self, observation: WorldObservation) -> None:
        """Record the first usable weather observation in each five-minute window."""
        window = self._window(observation)
        if self._history and window <= self._history[-1][0]:
            return
        wind, current, wave = observation.wind, observation.current, observation.wave
        if wind is None or current is None or wave is None:
            return
        values = np.array([wind.u, wind.v, current.u, current.v, wave.height], dtype=float)
        if np.isfinite(values).all():
            self._history.append((window, values))

    @property
    def observed_weather(self) -> tuple[tuple[int, tuple[float, ...]], ...]:
        """Return the weather samples retained for the next decision."""
        return tuple((window, tuple(values)) for window, values in self._history)

    @torch.inference_mode()
    def _hold_probability(self, history: np.ndarray, context: np.ndarray) -> float:
        history_tensor, context_tensor = torch.from_numpy(history[None]), torch.from_numpy(context[None])
        weather = (history_tensor - self.scaling["history_mean"]) / self.scaling["history_scale"]
        state = (context_tensor - self.scaling["context_mean"]) / self.scaling["context_scale"]
        probabilities = torch.softmax(self.model(weather, state)[0] / self.temperature, dim=0)
        return float(probabilities[0])

    def reset_release_confirmation(self) -> None:
        """Require fresh evidence after traffic takes over the steering."""
        self._release_goal = None
        self._release_ready_window = None
        self._release_confirmed = False

    def _release_evidence(
        self, goal: Waypoint | None, window: int, was_holding: bool, classical_hold: bool,
    ) -> str | None:
        """Confirm the classical release across two consecutive weather windows."""
        if goal != self._release_goal or classical_hold:
            self._release_goal = goal
            self._release_ready_window = None
            self._release_confirmed = False
        if classical_hold:
            return None
        if self._release_confirmed:
            return "confirmed release remains feasible"
        if not was_holding:
            self._release_ready_window = None
            return None
        if self._release_ready_window == window - 1:
            self._release_confirmed = True
            return "sustained feasible release"
        self._release_ready_window = window
        return None

    def decide(
        self, observation: WorldObservation, target: np.ndarray, *, was_holding: bool,
        classical_hold: bool, gate_vmg: float | None, hold_available: bool, goal: Waypoint | None = None,
    ) -> TemporalHoldDecision:
        self.observe(observation)
        window = self._window(observation)
        window_utc = str(np.datetime64(window * WEATHER_INTERVAL_SECONDS, "s")) + "Z"
        decision_utc = str(observation.time.astype("datetime64[ms]")) + "Z"
        observed = bool(self._history and self._history[-1][0] == window)
        release = None
        if self.confirm_release:
            release = self._release_evidence(goal, window, was_holding, classical_hold or not observed)
        if not observed:
            return TemporalHoldDecision(window_utc, classical_hold, classical_hold, fallback_reason="missing observed weather",
                                        decision_utc=decision_utc)
        available_windows, samples = zip(*self._history, strict=True)
        recent_windows = np.arange(window + 1 - HISTORY_STEPS, window + 1)
        offsets = np.maximum(0, np.searchsorted(available_windows, recent_windows, side="right") - 1)
        try:
            history, context = encode_hold_history(np.asarray(samples)[offsets], target, was_holding)
        except ValueError:
            return TemporalHoldDecision(window_utc, classical_hold, classical_hold, fallback_reason="invalid target geometry",
                                        decision_utc=decision_utc)
        p_hold = self._hold_probability(history, context)
        if not np.isfinite(p_hold):
            return TemporalHoldDecision(window_utc, classical_hold, classical_hold, fallback_reason="invalid model probability",
                                        decision_utc=decision_utc)
        raw_hold = p_hold >= 0.5
        if raw_hold and release is not None:
            return TemporalHoldDecision(window_utc, False, classical_hold, p_hold, raw_hold,
                                        decision_utc=decision_utc, release_guard=release)
        confidence = max(p_hold, 1.0 - p_hold)
        reason = None
        if self.confidence_threshold is None or confidence < self.confidence_threshold:
            reason = "low confidence"
        elif raw_hold and not hold_available:
            reason = "no chart-clear hold trim"
        elif raw_hold and gate_vmg is not None and gate_vmg >= self.maximum_hold_vmg_mps:
            reason = "verified sailing progress"
        elif not raw_hold and (gate_vmg is None or gate_vmg < self.minimum_release_vmg_mps):
            reason = "no physically viable gate approach"
        return TemporalHoldDecision(window_utc, classical_hold if reason else raw_hold, classical_hold, p_hold, raw_hold, reason, decision_utc)
