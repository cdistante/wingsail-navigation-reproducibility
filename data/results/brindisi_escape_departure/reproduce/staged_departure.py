"""Protected launch timing from delivered AIS and a native forecast prefix."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from time import perf_counter

import numpy as np

from native_branches import _clock, _first_capture
from native_efficient_guard import FORECAST_REPORT_AGE_S, PREVIEW_HORIZON_S
from native_stale_guard import guarded_clearance
from world._geo import haversine_metres, local_east_north_metres, normalise_time, offset_latitude_longitude


@dataclass(frozen=True)
class DepartureCheck:
    utc: np.datetime64
    elapsed_s: float
    reason: str
    wall_seconds: float
    report_count: int
    maximum_report_age_s: float | None = None
    first_choice_id: str | None = None
    startup_margin_m: float | None = None
    preview_margin_m: float | None = None
    prefix_duration_s: float | None = None
    chart_clear: bool | None = None
    colreg_clear: bool | None = None

    @property
    def eligible(self):
        return self.reason == "route_prefix_clear"


@dataclass(frozen=True)
class DepartureDecision:
    launch_utc: np.datetime64 | None
    wait_seconds: float
    reason: str
    checks: tuple[DepartureCheck, ...]


def _course_allowed(engine, state, course, observation, prediction, elapsed):
    if not engine.config.colreg:
        return True
    relative = (course - state.heading + 180.) % 360. - 180.
    if not -engine.colreg.starboard_turn_prohibited_sector_degrees <= relative <= -1.:
        return True
    # These positions are forecasts from a single already-delivered snapshot.
    # COLREG consumes positions directly, while AIS uncertainty is swept below.
    centres = prediction.centres_at(elapsed)
    latitudes, longitudes = offset_latitude_longitude(
        prediction.origin_latitude, prediction.origin_longitude, centres[:, 0], centres[:, 1],
    )
    projected = tuple(
        replace(report, latitude=float(latitude), longitude=float(longitude))
        for report, latitude, longitude in zip(prediction.targets, latitudes, longitudes, strict=True)
    )
    forecast = replace(observation, boat=state, time=_clock(observation.time, elapsed), ais_targets=projected)
    return not any(item.active for item in engine.colreg.assess(state, forecast))


def _prefix(engine, decision, prediction):
    """Project the accepted first tack and its scheduled follow-up in fixed forcing."""
    first = decision.choices[decision.physics_index]
    following = min(decision.choices,
                    key=lambda choice: abs((choice.bow_deg - first.following_bow_deg + 180.) % 360. - 180.))
    state, observation, goal = decision.state, decision.observation, decision.goal
    path = [np.array([[state.latitude, state.longitude]])]
    dt = engine.config.control_dt_s
    colreg_clear = True
    for step in range(round(PREVIEW_HORIZON_S / dt)):
        elapsed = step * dt
        choice = first if elapsed < first.switch_after_s else following
        if step and colreg_clear:
            colreg_clear = _course_allowed(engine, state, choice.course_deg, observation, prediction, elapsed)
        action = engine.action(state, choice)
        state, segment, _ = engine.integrate(state, action, observation, dt)
        hit, _ = _first_capture(segment, goal, engine.config.capture_radius_m)
        if hit is not None:
            end = max(1, min(int(np.ceil(hit)), len(segment) - 1))
            path.append(segment[1:end + 1])
            break
        path.append(segment[1:])
    return np.concatenate(path), colreg_clear


def assess_launch(engine, state, goal, utc, elapsed, *, maximum_report_age_s=20., uncapped_startup_s=60.):
    """Evaluate one release instant using only the observation available then."""
    started = perf_counter()
    observation = engine.environment.observe(state, utc)
    reports = observation.ais_targets
    report_count = len(reports)
    if not reports:
        return DepartureCheck(utc, elapsed, "no_delivered_ais_reports", perf_counter() - started, report_count)
    ages = np.array([report.report_age_seconds for report in reports], dtype=float)
    if not np.isfinite(ages).all() or np.any(ages < 0.):
        return DepartureCheck(utc, elapsed, "invalid_report_age", perf_counter() - started, report_count)
    maximum_age = float(ages.max())
    if maximum_age > maximum_report_age_s:
        return DepartureCheck(utc, elapsed, "ais_reports_too_old", perf_counter() - started, report_count, maximum_age)

    decision = engine.decision(state, goal, utc, elapsed)
    if decision.physics_index < 0:
        return DepartureCheck(utc, elapsed, "no_startup_action", perf_counter() - started, report_count, maximum_age)
    prediction = engine._prediction(observation)
    path, colreg_clear = _prefix(engine, decision, prediction)
    chart_clear = bool(engine.environment.chart_path_is_clear(path, observation))
    east, north = local_east_north_metres(path[:, 0], path[:, 1], state.latitude, state.longitude)
    track = np.column_stack((east, north))
    times = np.arange(len(track)) * engine.config.integration_dt_s
    early = times <= uncapped_startup_s
    startup_margin = guarded_clearance(prediction, track[early], times[early])
    preview = prediction.for_preview(FORECAST_REPORT_AGE_S, engine.config.control_dt_s)
    preview_margin = guarded_clearance(preview, track, times)

    if not chart_clear:
        reason = "startup_path_intersects_chart"
    elif not colreg_clear:
        reason = "startup_on_projected_colreg_conflict"
    elif startup_margin <= 0.:
        reason = "startup_full_uncertainty_conflict"
    elif preview_margin <= 0.:
        reason = "route_prefix_predicted_conflict"
    else:
        reason = "route_prefix_clear"
    return DepartureCheck(utc, elapsed, reason, perf_counter() - started, report_count, maximum_age,
                          decision.choices[decision.physics_index].id, startup_margin, preview_margin,
                          float(times[-1]), chart_clear, colreg_clear)


def choose_departure(engine, initial_state, goals, request_utc, *, protected_staging,
                     max_wait_s=900., poll_seconds=5., maximum_report_age_s=20., uncapped_startup_s=60.):
    """Check the earliest admitted release while a protected start is available.

    Waiting only advances the absolute clock in the fixture. The ownship stays
    secured at the protected launch point before release and is not navigated
    through that interval. Once afloat, the normal uncapped control screen
    applies against each actually delivered report.
    """
    request = normalise_time(request_utc)
    goal = next(iter(goals), None)
    if goal is None:
        raise ValueError("A departure requires a next waypoint.")
    if request != engine.environment.start_utc:
        raise ValueError("The release gate must start at the original AIS fixture epoch.")
    distance = haversine_metres(initial_state.latitude, initial_state.longitude, *engine.environment.origin)
    if distance > 1e-6:
        raise ValueError("Protected staging is only defined at the fixture launch point.")
    if (not all(math.isfinite(value) for value in (
            poll_seconds, max_wait_s, maximum_report_age_s, uncapped_startup_s))
            or poll_seconds <= 0. or max_wait_s < 0. or maximum_report_age_s < 0.
            or not engine.config.integration_dt_s <= uncapped_startup_s <= PREVIEW_HORIZON_S):
        raise ValueError("Invalid launch polling duration, report age, or startup horizon.")
    end = max_wait_s if protected_staging else 0.
    waits = np.r_[np.arange(0., end, poll_seconds), end]
    # A separate engine prevents the readiness assessment from changing the
    # sailing run's anticipation telemetry or integration state.
    probe = type(engine)(engine.environment, config=engine.config, dynamics=engine.dynamics)
    checks = []
    for elapsed in waits:
        check = assess_launch(probe, initial_state, goal, _clock(request, float(elapsed)), float(elapsed),
                              maximum_report_age_s=maximum_report_age_s, uncapped_startup_s=uncapped_startup_s)
        checks.append(check)
        if check.eligible:
            return DepartureDecision(check.utc, check.elapsed_s, check.reason, tuple(checks))

    reason = "protected_wait_limit" if protected_staging else "protected_staging_unavailable"
    return DepartureDecision(None, float(end), reason, tuple(checks))
