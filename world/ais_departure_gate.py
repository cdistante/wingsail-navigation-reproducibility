"""Screen a staged vessel's release against received AIS reports."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ais_avoidance import AISPrediction, _closest_fraction


@dataclass(frozen=True)
class DepartureGateSettings:
    maximum_report_age_s: float = 20.0
    uncapped_startup_seconds: float = 120.0
    preview_seconds: float = 300.0
    preview_age_budget_s: float = 65.0
    startup_speed_bound_mps: float = 2.0
    startup_room_m: float = 30.0
    required_consecutive_observations: int = 3
    maximum_wait_s: float = 1800.0


@dataclass(frozen=True)
class DepartureAssessment:
    fresh_reports: bool = False
    min_robust_receding_mps: float | None = None
    min_uncapped_startup_margin_m: float | None = None
    min_preview_margin_m: float | None = None

    @property
    def eligible(self) -> bool:
        checks = (self.min_robust_receding_mps, self.min_uncapped_startup_margin_m, self.min_preview_margin_m)
        return self.fresh_reports and all(value is not None and value > 0.0 for value in checks)


def _swept_tube_margin(prediction: AISPrediction, horizon: float, interval: float,
                       settings: DepartureGateSettings) -> float:
    """Minimum distance from a launch disk growing at the assumed speed bound."""
    times = np.r_[np.arange(0.0, horizon, interval), horizon]
    if prediction.preview_growth_seconds is not None:
        # Split at report-age caps so the radius chord is conservative on
        # each interval, including when a cap is between update knots.
        knots = prediction.preview_growth_seconds
        times = np.unique(np.r_[times, knots[(knots > 0.0) & (knots < horizon)]])
    centres = prediction.centres_at(times)
    radii = (prediction.radii_at(times) + settings.startup_room_m
             + settings.startup_speed_bound_mps * times[:, None])
    delta = np.diff(centres, axis=0)
    growth = np.diff(radii, axis=0)
    fraction = _closest_fraction(centres[:-1], delta, growth)
    margin = np.linalg.norm(centres[:-1] + fraction[..., None] * delta, axis=-1)
    return float(np.min(margin - radii[:-1] - fraction * growth))


def assess_departure(observation, settings: DepartureGateSettings, interval: float, *,
                     acceleration_bound_mps2: float, clearance_reserve_metres: float) -> DepartureAssessment:
    reports = observation.ais_targets
    identifiers = {target.mmsi for target in reports}
    if (not reports or None in identifiers or len(identifiers) != len(reports)
            or any(any(value is None or not np.isfinite(value) for value in
                       (report.course, report.speed, report.report_age_seconds,
                        report.length_metres, report.beam_metres))
                   or min(report.speed, report.report_age_seconds) < 0.0
                   or min(report.length_metres, report.beam_metres) <= 0.0
                   for report in reports)):
        return DepartureAssessment()
    prediction = AISPrediction.from_observation(
        observation, acceleration_bound_mps2=acceleration_bound_mps2,
        clearance_reserve_metres=clearance_reserve_metres,
    )
    age = prediction.report_ages_seconds
    distance = np.linalg.norm(prediction.positions_metres, axis=1)
    # If the true center can be at the origin, a receding bearing cannot be
    # certified. Otherwise bound the line-of-sight rotation and velocity error.
    sigma = prediction.clearance_radii_metres - prediction.hull_radii_metres - 25.0 - clearance_reserve_metres
    speed = np.linalg.norm(prediction.velocities_mps, axis=1)
    radial = np.einsum("ij,ij->i", prediction.positions_metres, prediction.velocities_mps)
    lower = (radial - 2.0 * speed * sigma) / np.maximum(distance, np.finfo(float).tiny)
    lower = np.where(distance > sigma, lower - acceleration_bound_mps2 * age, -1.0)
    preview = prediction.for_preview(settings.preview_age_budget_s, interval)
    return DepartureAssessment(
        fresh_reports=bool(np.all(age <= settings.maximum_report_age_s)),
        min_robust_receding_mps=float(np.min(lower)),
        min_uncapped_startup_margin_m=_swept_tube_margin(
            prediction, settings.uncapped_startup_seconds, interval, settings,
        ),
        min_preview_margin_m=_swept_tube_margin(preview, settings.preview_seconds, interval, settings),
    )
