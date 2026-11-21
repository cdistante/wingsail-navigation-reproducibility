"""Transfer a trained GRPO residual to admitted native Fossen tack schedules.

The trained actor contributes only its bounded residual. The native
forecast supplies the baseline ETA, and the native engine continues to
own the plant, scheduled switches, chart, COLREG, and AIS decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from world._geo import local_east_north_metres
from world.drift_guidance import DriftMode
from world.forecast_tacks import steady_sailing_trim
from world.learned_mark_policy import FEATURE_NAMES, LearnedMarkPolicy, PolarSnapshot, candidate_batch


LEGACY_SUPPORT_PATH = (
    Path(__file__).resolve().parents[2] / "learned_safe_mark_policy" / "learned_mark_policy.json"
)
FEATURE_INTERVAL_S = 300.0
TRUST_REGRET_S = 150.0
MINIMUM_CORRECTED_GAIN_S = 25.0
RESIDUAL_SCALE_S = 1200.0


@dataclass(frozen=True)
class GRPOSelection:
    index: int
    status: str
    eligible_candidates: int
    native_eta_regret_s: float | None = None
    corrected_gain_s: float | None = None


@dataclass(frozen=True)
class NativeGRPO:
    hidden_weights: np.ndarray
    hidden_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: np.ndarray
    legacy_support: LearnedMarkPolicy

    @classmethod
    def load(cls, checkpoint, *, support_path=LEGACY_SUPPORT_PATH) -> NativeGRPO:
        # Match the checkpoint format of the reduced ResidualActor, with no
        # PyTorch runtime dependency and no deserialization of Python objects.
        names_and_shapes = (
            ("network.0.weight", (48, len(FEATURE_NAMES))),
            ("network.0.bias", (48,)),
            ("network.2.weight", (1, 48)),
            ("network.2.bias", (1,)),
        )
        with np.load(checkpoint, allow_pickle=False) as archive:
            arrays = []
            for name, shape in names_and_shapes:
                value = np.asarray(archive[name], dtype=np.float32)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError(f"Invalid GRPO checkpoint array {name}, expected shape {shape}.")
                value.setflags(write=False)
                arrays.append(value)
        return cls(*arrays, LearnedMarkPolicy.load(support_path))

    def cost_correction_seconds(self, features):
        hidden = np.tanh(features @ self.hidden_weights.T + self.hidden_bias)
        residual = np.tanh(hidden @ self.output_weights.T + self.output_bias)[..., 0]
        return -RESIDUAL_SCALE_S * residual

    def select(
        self, decision, *, dynamics, remaining_s: float, capture_radius_m: float, strength: float = 1.0,
    ) -> GRPOSelection:
        """Rerank nearby ETA choices, with the historical native trust rules.

        The 300-second feature projection matches the reduced training clock.
        It uses observed persistence, with no privileged weather or AIS truth.
        It does not change the native tack duration or the five-second helm.
        """
        incumbent = decision.physics_index
        if incumbent < 0 or not decision.allowed[incumbent]:
            raise ValueError("Native GRPO requires an admissible physics incumbent.")
        if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError("The learned residual strength must lie in [0, 1].")
        if strength == 0.0:
            return GRPOSelection(incumbent, "disabled", 0)

        eta = np.array([
            choice.predicted_eta_s if choice.predicted_eta_s is not None else np.inf
            for choice in decision.choices
        ])
        if not np.isfinite(eta[incumbent]):
            return GRPOSelection(incumbent, "no_incumbent_capture", 0)
        eligible = np.flatnonzero(
            decision.allowed & np.isfinite(eta) & (eta <= eta[incumbent] + TRUST_REGRET_S)
        )
        if len(eligible) < 2:
            return GRPOSelection(incumbent, "no_competing_capture", len(eligible))

        observation = decision.observation
        current = observation.current
        flow = np.array([current.u, current.v])
        modes = []
        # Recover the velocity at exactly the bow/flap already admitted by
        # the native planner. All modes belong in the feature polar even
        # when the report-based mask prevents them from being selected now.
        for choice in decision.choices:
            trim = steady_sailing_trim(observation, dynamics, choice.bow_deg, choice.steady_flap_deg)
            if trim is None:
                return GRPOSelection(incumbent, "unreconstructable_polar", len(eligible))
            water_velocity, rudder = trim
            velocity = dynamics._body_to_earth(choice.bow_deg) @ water_velocity + flow
            mode = DriftMode(velocity, choice.bow_deg, rudder, choice.steady_flap_deg)
            course_error = (mode.course - choice.course_deg + 180.0) % 360.0 - 180.0
            if abs(course_error) > 1e-6 or abs(rudder - choice.steady_rudder_deg) > 1e-6:
                return GRPOSelection(incumbent, "unreconstructable_polar", len(eligible))
            modes.append(mode)

        polar = PolarSnapshot.from_modes(modes, flow, observation.wind.direction)
        state = decision.state
        target = np.array(local_east_north_metres(
            decision.goal.latitude, decision.goal.longitude, state.latitude, state.longitude,
        ))
        projection_s = min(FEATURE_INTERVAL_S, remaining_s)
        if not self.legacy_support.supported(
            np.linalg.norm(target), capture_radius_m, polar, None,
            observation.wind.speed, observation.wave.height, projection_s, 0.0,
        ):
            return GRPOSelection(incumbent, "outside_legacy_support", len(eligible))
        feature_batch = candidate_batch(
            target, state.heading, capture_radius_m, polar, None,
            observation.wind.speed, observation.wave.height, projection_s, 0.0, remaining_s,
        )
        corrected = eta[eligible] + strength * self.cost_correction_seconds(feature_batch.features[0, eligible])
        if not np.isfinite(corrected).all():
            return GRPOSelection(incumbent, "nonfinite_score", len(eligible))
        best = int(np.argmin(corrected))
        incumbent_row = int(np.searchsorted(eligible, incumbent))
        gain = float(corrected[incumbent_row] - corrected[best])
        if gain <= MINIMUM_CORRECTED_GAIN_S:
            return GRPOSelection(incumbent, "incumbent_retained", len(eligible), 0.0, gain)
        index = int(eligible[best])
        return GRPOSelection(index, "selected", len(eligible), float(eta[index] - eta[incumbent]), gain)
