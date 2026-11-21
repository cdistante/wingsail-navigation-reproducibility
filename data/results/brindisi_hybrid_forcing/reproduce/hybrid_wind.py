"""Causal next-hour wind trend for native schedule planning."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from math import atan2, degrees, hypot, isfinite
from pathlib import Path

import numpy as np

from world._geo import normalise_time
from world.weather import Wind
from world.world import WorldObservation


STAGE3_RECEIPT = Path(__file__).resolve().parents[2] / "stage3_forcing_offline" / "selection_receipt.json"
HOUR = np.timedelta64(1, "h")


@dataclass(frozen=True)
class DecisionWindForecast:
    """An issue-local forecast with its available inputs frozen at issue time."""

    observation: WorldObservation
    next_hour: np.datetime64
    wind_change_en_mps: tuple[float, float]

    def __call__(self, when) -> WorldObservation:
        if self.wind_change_en_mps == (0., 0.):
            return self.observation
        time = normalise_time(when)
        issued = normalise_time(self.observation.time)
        if time <= issued:
            return self.observation
        fraction = min(1., float((time - issued) / (self.next_hour - issued)))
        known = self.observation.wind
        assert known is not None
        east = known.u + fraction * self.wind_change_en_mps[0]
        north = known.v + fraction * self.wind_change_en_mps[1]
        return replace(self.observation, time=time,
                       wind=Wind(east, north, hypot(east, north), degrees(atan2(-east, -north)) % 360.))


@dataclass(frozen=True)
class WindTrendProvider:
    """Fit-free provider using April's frozen trend setting and training cap."""

    trend_multiplier: float
    wind_change_cap_mps: float
    blend: float = 1.
    receipt_sha256: str | None = None

    def __post_init__(self):
        if not isfinite(self.trend_multiplier) or self.trend_multiplier < 0.:
            raise ValueError("The wind trend multiplier must be finite and nonnegative.")
        if not isfinite(self.wind_change_cap_mps) or self.wind_change_cap_mps <= 0.:
            raise ValueError("The hourly wind change cap must be finite and positive.")
        if not 0. <= self.blend <= 1.:
            raise ValueError("The forecast blend must be in [0, 1].")

    @classmethod
    def from_receipt(cls, path=STAGE3_RECEIPT, *, blend=1.):
        raw = Path(path).read_bytes()
        receipt = json.loads(raw)
        if (receipt["prediction_horizon_s"] != 3600 or receipt["training_blocks"] != ["winter", "summer"]
                or receipt["validation_block"] != "spring"):
            raise ValueError("Expected the staged January/July and April one-hour forcing receipt.")
        settings = receipt["settings"]
        selected = float(settings["trend_multiplier"])
        minimum = min(settings["validation_trend_candidates"], key=lambda row: (row["score"], row["alpha"]))
        if selected != float(minimum["alpha"]):
            raise ValueError("The saved wind trend does not match the April model selection.")
        return cls(selected, float(settings["change_caps"][0]), float(blend), sha256(raw).hexdigest())

    def at_decision(self, archive, observation: WorldObservation) -> DecisionWindForecast:
        issued = normalise_time(observation.time)
        current_hour = issued.astype("datetime64[h]")
        next_hour = current_hour + HOUR
        if self.blend == 0.:
            return DecisionWindForecast(observation, next_hour, (0., 0.))
        if observation.wind is None:
            raise ValueError("Forecast planning requires an observed wind vector.")
        latitude, longitude = observation.boat.latitude, observation.boat.longitude
        recent = archive.at(latitude, longitude, current_hour).wind
        previous = archive.at(latitude, longitude, current_hour - HOUR).wind
        east = self.trend_multiplier * (recent.u - previous.u)
        north = self.trend_multiplier * (recent.v - previous.v)
        length = hypot(east, north)
        if not isfinite(length):
            raise ValueError("The available hourly winds must have finite components.")
        factor = self.blend
        if length > self.wind_change_cap_mps:
            factor *= self.wind_change_cap_mps / length
        return DecisionWindForecast(observation, next_hour, (factor * east, factor * north))

    def forecast(self, archive, observation: WorldObservation, when) -> WorldObservation:
        return self.at_decision(archive, observation)(when)
