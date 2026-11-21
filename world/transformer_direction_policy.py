"""Runtime wrapper for trained harmonic-planner direction Transformers."""

from __future__ import annotations

from math import cos, hypot, radians, sin
from pathlib import Path

import numpy as np
import torch

from simulator_harmonic.transformer_wind_policy import WindPolicyTransformer


class TransformerDirectionPolicy:
    """Load a trained policy and predict a normalized local-grid direction."""

    def __init__(self, checkpoint: Path, *, device: str | None = None) -> None:
        self.checkpoint = Path(checkpoint)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        saved = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        config = saved.get("config", {})
        self.grid_size = int(config.get("grid_size", 41))
        self.model = WindPolicyTransformer(
            grid_size=self.grid_size,
            d_model=int(config.get("d_model", 128)),
            nhead=int(config.get("nhead", 4)),
            num_layers=int(config.get("layers", 3)),
            dim_feedforward=int(config.get("ffn", 256)),
            dropout=float(config.get("dropout", 0.1)),
        )
        self.model.load_state_dict(saved["model_state"])
        self.model.to(self.device)
        self.model.eval()

    def predict_direction_with_diagnostics(
        self,
        occupancy: np.ndarray,
        feature_field: np.ndarray,
        potential: np.ndarray,
        boat_cell: tuple[int, int],
        goal_cell: tuple[int, int],
        wind_from_degrees: float,
        no_go_angle_degrees: float,
    ) -> tuple[float, float, float, float]:
        if occupancy.shape != (self.grid_size, self.grid_size):
            raise ValueError(
                f"Transformer expects a {self.grid_size}x{self.grid_size} grid, "
                f"received {occupancy.shape}."
            )
        tensors = [
            torch.from_numpy(np.asarray(values, dtype=np.float32))
            .unsqueeze(0)
            .to(self.device)
            for values in (occupancy, feature_field, potential)
        ]
        boat = torch.tensor([boat_cell], dtype=torch.float32, device=self.device)
        goal = torch.tensor([goal_cell], dtype=torch.float32, device=self.device)
        wind_radians = radians(float(wind_from_degrees))
        wind = torch.tensor(
            [[sin(wind_radians), cos(wind_radians)]],
            dtype=torch.float32,
            device=self.device,
        )
        no_go = torch.tensor(
            [[float(no_go_angle_degrees)]], dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            prediction = self.model(*tensors, boat, goal, wind, no_go)

        raw_row = float(prediction[0, 0].item())
        raw_column = float(prediction[0, 1].item())
        magnitude = hypot(raw_row, raw_column)
        if magnitude < 1e-9:
            direction = (0.0, 0.0)
        else:
            direction = (raw_row / magnitude, raw_column / magnitude)

        goal_row = float(goal_cell[0] - boat_cell[0])
        goal_column = float(goal_cell[1] - boat_cell[1])
        goal_magnitude = hypot(goal_row, goal_column)
        goal_alignment = (
            1.0
            if goal_magnitude < 1e-9
            else (
                direction[0] * goal_row / goal_magnitude
                + direction[1] * goal_column / goal_magnitude
            )
        )
        return direction[0], direction[1], magnitude, goal_alignment
