"""Planner-conditioned Transformer that predicts saildrone actuators directly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn

from .boat import BoatAction, BoatState
from .harmonic_planner import HarmonicPlan
from .world import WorldObservation

PLANNER_MODES = ("standard", "gamma", "diffusion")
GLOBAL_FEATURE_NAMES = (
    "wind_u",
    "wind_v",
    "current_u",
    "current_v",
    "wave_height",
    "wave_sin",
    "wave_cos",
    "wave_period",
    "heading_sin",
    "heading_cos",
    "surge",
    "sway",
    "yaw_rate",
    "rudder",
    "flap",
    "no_go_angle",
)


def planner_feature_field(plan: HarmonicPlan) -> np.ndarray:
    """Return a finite field encoding local descent strength and direction."""

    potential = np.asarray(plan.potential, dtype=np.float32)
    finite = np.nan_to_num(potential, nan=1.0, posinf=1.0, neginf=0.0)
    row_gradient, column_gradient = np.gradient(finite)
    magnitude = np.hypot(row_gradient, column_gradient)
    scale = float(np.percentile(magnitude, 95.0))
    return np.clip(magnitude / max(scale, 1e-6), 0.0, 1.0).astype(np.float32)


def map_tensor(
    occupancy: np.ndarray,
    feature_field: np.ndarray,
    potential: np.ndarray,
    boat_cell: tuple[int, int],
    goal_cell: tuple[int, int],
) -> np.ndarray:
    """Build the five-channel spatial input shared by generation and runtime."""

    arrays = [
        np.asarray(occupancy, dtype=np.float32),
        np.asarray(feature_field, dtype=np.float32),
        np.nan_to_num(np.asarray(potential, dtype=np.float32), nan=1.0, posinf=1.0),
    ]
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("occupancy, feature field, and potential must share a shape.")
    boat = np.zeros_like(arrays[0])
    goal = np.zeros_like(arrays[0])
    boat[boat_cell] = 1.0
    goal[goal_cell] = 1.0
    return np.stack((*arrays, boat, goal)).astype(np.float32)


def global_features(
    observation: WorldObservation,
    state: BoatState,
    no_go_angle_degrees: float,
) -> np.ndarray:
    """Encode environmental and vessel state without redundant polar wind data."""

    wind_u = 0.0 if observation.wind is None else observation.wind.u
    wind_v = 0.0 if observation.wind is None else observation.wind.v
    current_u = 0.0 if observation.current is None else observation.current.u
    current_v = 0.0 if observation.current is None else observation.current.v
    wave_height = 0.0 if observation.wave is None else observation.wave.height
    wave_direction = 0.0 if observation.wave is None else observation.wave.direction
    wave_period = 0.0 if observation.wave is None else observation.wave.peak_period
    wave_radians = np.radians(wave_direction)
    heading_radians = np.radians(state.heading)
    return np.asarray(
        [
            wind_u,
            wind_v,
            current_u,
            current_v,
            wave_height,
            np.sin(wave_radians),
            np.cos(wave_radians),
            wave_period,
            np.sin(heading_radians),
            np.cos(heading_radians),
            state.speed if state.surge_speed is None else state.surge_speed,
            state.sway_speed,
            state.yaw_rate,
            state.rudder,
            state.flap,
            no_go_angle_degrees,
        ],
        dtype=np.float32,
    )


class PlannerConditionedMarineTransformer(nn.Module):
    """One model for STANDARD/GAMMA/DIFFUSION with a direct actuator head."""

    def __init__(
        self,
        grid_size: int = 41,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 192,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.config = {
            "grid_size": int(grid_size),
            "d_model": int(d_model),
            "nhead": int(nhead),
            "num_layers": int(num_layers),
            "dim_feedforward": int(dim_feedforward),
            "dropout": float(dropout),
        }
        self.grid_size = int(grid_size)
        self.patch = nn.Conv2d(5, d_model, kernel_size=3, stride=2, padding=1)
        token_side = (self.grid_size + 1) // 2
        self.position = nn.Parameter(torch.zeros(1, token_side * token_side, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            norm_first=True,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.planner_embedding = nn.Embedding(len(PLANNER_MODES), d_model)
        self.global_encoder = nn.Sequential(
            nn.Linear(len(GLOBAL_FEATURE_NAMES), d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )

    def forward(
        self, maps: torch.Tensor, global_state: torch.Tensor, planner_id: torch.Tensor
    ) -> torch.Tensor:
        if maps.ndim != 4 or maps.shape[1] != 5:
            raise ValueError("maps must have shape [batch, 5, height, width].")
        tokens = self.patch(maps.float()).flatten(2).transpose(1, 2)
        tokens = self.encoder(tokens + self.position[:, : tokens.shape[1]])
        pooled = tokens.mean(dim=1)
        context = self.global_encoder(global_state.float())
        planner = self.planner_embedding(planner_id.long())
        raw = self.head(torch.cat((pooled, context, planner), dim=1))
        rudder = 30.0 * torch.tanh(raw[:, 0])
        flap = 15.0 * torch.tanh(raw[:, 1])
        confidence = torch.sigmoid(raw[:, 2])
        return torch.stack((rudder, flap, confidence), dim=1)


@dataclass(frozen=True)
class ControllerDecision:
    action: BoatAction
    confidence: float
    accepted: bool
    reason: str


class MarineController:
    """Checkpoint inference plus bounded slew/confidence safety filtering."""

    def __init__(
        self,
        checkpoint: Path | str,
        *,
        device: str | None = None,
        confidence_threshold: float = 0.45,
    ) -> None:
        target = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(target)
        payload = torch.load(Path(checkpoint), map_location=self.device, weights_only=False)
        if payload.get("model_type") != "planner_conditioned_marine_transformer":
            raise ValueError("Checkpoint is not a combined marine controller.")
        self.model = PlannerConditionedMarineTransformer(**payload["model_config"])
        self.model.load_state_dict(payload["model_state"])
        self.model.to(self.device).eval()
        normalisation = payload["normalization"]
        self.mean = np.asarray(normalisation["global_mean"], dtype=np.float32)
        self.std = np.asarray(normalisation["global_std"], dtype=np.float32)
        if self.mean.shape != (len(GLOBAL_FEATURE_NAMES),) or self.std.shape != self.mean.shape:
            raise ValueError("Checkpoint normalization metadata has the wrong shape.")
        self.confidence_threshold = float(confidence_threshold)

    def decide(
        self,
        maps: np.ndarray,
        features: np.ndarray,
        planner_mode: str,
        state: BoatState,
        *,
        max_rudder_slew: float,
        max_flap_slew: float,
    ) -> ControllerDecision:
        mode = planner_mode.lower()
        if mode not in PLANNER_MODES:
            return ControllerDecision(BoatAction(state.rudder, state.flap), 0.0, False, "planner")
        if not np.all(np.isfinite(maps)) or not np.all(np.isfinite(features)):
            return ControllerDecision(BoatAction(state.rudder, state.flap), 0.0, False, "non_finite")
        normalized = (features.astype(np.float32) - self.mean) / np.maximum(self.std, 1e-6)
        with torch.inference_mode():
            output = self.model(
                torch.from_numpy(maps[None]).to(self.device),
                torch.from_numpy(normalized[None]).to(self.device),
                torch.tensor([PLANNER_MODES.index(mode)], device=self.device),
            )[0].cpu().numpy()
        confidence = float(output[2])
        if not np.all(np.isfinite(output)):
            return ControllerDecision(BoatAction(state.rudder, state.flap), 0.0, False, "non_finite")
        if confidence < self.confidence_threshold:
            return ControllerDecision(
                BoatAction(state.rudder, state.flap), confidence, False, "low_confidence"
            )
        rudder = float(np.clip(output[0], state.rudder - max_rudder_slew, state.rudder + max_rudder_slew))
        flap = float(np.clip(output[1], state.flap - max_flap_slew, state.flap + max_flap_slew))
        return ControllerDecision(
            BoatAction(float(np.clip(rudder, -30.0, 30.0)), float(np.clip(flap, -15.0, 15.0))),
            confidence,
            True,
            "accepted",
        )


def checkpoint_payload(
    model: PlannerConditionedMarineTransformer,
    normalization: Mapping[str, object],
    **metadata: object,
) -> dict[str, object]:
    return {
        "model_type": "planner_conditioned_marine_transformer",
        "model_config": model.config,
        "model_state": model.state_dict(),
        "normalization": dict(normalization),
        "planner_modes": PLANNER_MODES,
        "global_feature_names": GLOBAL_FEATURE_NAMES,
        **metadata,
    }
