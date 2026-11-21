"""Shared report-only guards for the joint weather/AIS reduced experiment.

Candidate motion is the earlier two-segment Fossen-equilibrium approximation.
Both policies receive the same candidates, clock and report-only guard. Ship
truth is queried only by ``assess_applied`` after an action has been selected.
This is not native Fossen control, chart coverage or a COLREG implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np

from data.results.learned_safe_mark_policy.reproduce.ais_scenarios import ReducedAISPreview, available_cases
from data.results.learned_safe_mark_policy.reproduce.environment import active_forcing
from world.boat import BoatState
from world.currents import Current
from world.learned_mark_policy import MAX_SECONDS, CandidateBatch, FEATURE_NAMES, candidate_batch, circle_entry
from world.waves import Wave
from world.weather import Wind
from world.world import WorldObservation


LANE_OFFSETS = {999000101: (140., 0.), 999000102: (0., 230.), 999000103: (0., -230.)}
REPORT_FEATURE_NAMES = (
    "report_margin_100m", "report_age_60s", "dropped_epochs_10",
    "allowed_fraction", "guard_required_clearance_100m", "has_ais",
)
_TURN_FEATURE = FEATURE_NAMES.index("turn_seconds_180")


@dataclass(frozen=True)
class JointContext:
    scenario: object
    profile: str
    ais: ReducedAISPreview | None
    guard_mode: str
    contingency_short_step: bool
    additional_planning_reserve_metres: float
    origin_latitude: float
    origin_longitude: float
    start_utc: np.datetime64


@dataclass(frozen=True)
class ReportAssessment:
    """Report-derived arrays only; analytical truth is deliberately absent."""
    allowed: np.ndarray
    report_margin_m: np.ndarray
    maximum_report_age_seconds: float
    maximum_committed_report_age_seconds: float
    dropped_reports_at_observation: int
    screened_seconds: np.ndarray
    used_far_lower_bound: bool


class JointEvaluation(NamedTuple):
    batch: CandidateBatch
    dt: float
    allowed: np.ndarray
    report_features: np.ndarray
    audit: dict
    assessment: ReportAssessment | None


def prepare(
    scenario, profile: str, *, origin_latitude: float = 40.225,
    origin_longitude: float = 18.75, guard_mode: str = "stale",
    contingency_short_step: bool = False, additional_planning_reserve_metres: float = 0.,
    start_utc="2000-01-01T00:00:00",
) -> JointContext:
    """Anchor one deterministic encounter; reuse its clock after every detour.

    The lane offsets are in the native eastbound fixture frame and are then
    rotated onto the initial mark bearing. The default stale guard exactly
    matches the preceding study: age >10 s requests a 5 s action plus an extra
    acceleration-reach margin. The 3 m static reserve remains inside the native
    AIS uncertainty radius and inside all subsequent truth audits.
    """
    if profile not in ("clear", *available_cases()):
        raise ValueError("Unknown joint AIS profile.")
    if guard_mode not in ("stale", "reach", "none"):
        raise ValueError("Guard mode must be stale, reach or none.")
    if not np.isfinite(additional_planning_reserve_metres) or additional_planning_reserve_metres < 0:
        raise ValueError("Additional planning reserve must be finite and nonnegative.")
    start = np.datetime64(str(start_utc).removesuffix("Z"), "ns")
    if np.isnat(start):
        raise ValueError("The encounter needs a valid UTC origin.")
    target = np.asarray(scenario.target, dtype=float)
    if target.shape != (2,) or not np.isfinite(target).all() or np.linalg.norm(target) <= 0:
        raise ValueError("The encounter needs a finite nonzero EN waypoint vector.")
    ais = None if profile == "clear" else ReducedAISPreview(
        profile, float(np.degrees(np.arctan2(*target)) % 360.),
        origin_latitude=origin_latitude, origin_longitude=origin_longitude,
        reserve_metres=3., track_offsets_EN=LANE_OFFSETS,
    )
    return JointContext(scenario, profile, ais, guard_mode, bool(contingency_short_step),
                        float(additional_planning_reserve_metres), float(origin_latitude),
                        float(origin_longitude), start)


def _nearest(start, finish):
    delta = finish-start
    length = np.sum(delta*delta, axis=-1)
    fraction = np.divide(-np.sum(start*delta, axis=-1), length,
                         out=np.zeros_like(length), where=length > 1e-16)
    return np.linalg.norm(start+np.clip(fraction, 0., 1.)[..., None]*delta, axis=-1)


def _report_assessment(ais, elapsed, own, batch, dt, required=0.) -> ReportAssessment:
    """Compute report geometry without invoking any analytical truth method."""
    snapshot = ais._snapshot(float(elapsed))
    prediction = snapshot.prediction
    rows, choices = batch.features.shape[:2]
    own = np.broadcast_to(own, (rows, 2))
    turn = np.minimum(dt, batch.features[..., _TURN_FEATURE]*180.)
    sail = batch.displacement-batch.turn_displacement
    length = np.linalg.norm(batch.turn_displacement, axis=-1)+np.linalg.norm(sail, axis=-1)
    radius_end = prediction.radii_at(dt)
    distance = np.linalg.norm(own[:, None]-prediction.positions_metres[None], axis=-1)
    lower = (distance[:, None]-length[..., None]
             -dt*np.linalg.norm(prediction.velocities_mps, axis=-1)-radius_end).min(axis=-1)
    # Each candidate independently chooses its cheap conservative bound.
    # Membership of other vessels/actions in a vectorized batch cannot change
    # its numeric features or mask. A bound below the guard threshold must
    # receive the tighter swept calculation before being rejected.
    far = lower > required
    if np.all(far):
        margin = lower
    else:
        count = max(1, int(np.ceil(dt/ais.chord_seconds)))
        regular = np.broadcast_to(np.linspace(0., dt, count+1), (rows, choices, count+1))
        times = np.sort(np.concatenate((regular, turn[..., None]), axis=-1), axis=-1)
        a = np.divide(np.minimum(times, turn[..., None]), turn[..., None],
                      out=np.zeros_like(times), where=turn[..., None] > 1e-12)
        b = np.divide(np.maximum(0., times-turn[..., None]), (dt-turn)[..., None],
                      out=np.zeros_like(times), where=(dt-turn)[..., None] > 1e-12)
        own_track = own[:, None, None]+a[..., None]*batch.turn_displacement[..., None, :]+b[..., None]*sail[..., None, :]
        relative = own_track[..., None, :]-prediction.centres_at(times)
        exact = (_nearest(relative[..., :-1, :, :], relative[..., 1:, :, :])
                 -prediction.radii_at(times[..., 1:])).min(axis=(-2, -1))
        margin = np.where(far, lower, exact)
    return ReportAssessment(margin > 0., margin, snapshot.maximum_age, snapshot.maximum_age+dt,
                            snapshot.lost_reports, np.full((rows, choices), dt), bool(np.any(far)))


def evaluate(context: JointContext, elapsed, targets, bows) -> JointEvaluation:
    """Return the common action set and report-only guard for batched vessels.

    ``targets`` is (rows,2), the EN vector from each vessel to the fixed mark;
    ``bows`` is scalar or (rows,). Their position in the anchored encounter is
    ``scenario.target - targets``. There is one shared scalar action duration.
    Returned feature arrays have shape (rows,candidates,6), in the public
    REPORT_FEATURE_NAMES order. All are finite, including clear-water cases.
    """
    elapsed = float(elapsed)
    targets = np.asarray(targets, dtype=float).reshape(-1, 2)
    bows = np.broadcast_to(np.asarray(bows, dtype=float), (len(targets),))
    if not np.isfinite(elapsed) or not 0 <= elapsed < MAX_SECONDS:
        raise ValueError("Decision time must be inside the six-hour episode cap.")
    if not np.isfinite(targets).all() or not np.isfinite(bows).all():
        raise ValueError("Candidate states must be finite.")
    scenario = context.scenario
    forcing, future, until = active_forcing(scenario, elapsed)
    age = lost = 0.
    if context.ais is not None:
        diagnostics = context.ais.report_diagnostics(elapsed)
        age = diagnostics["maximum_report_age_seconds"]
        lost = diagnostics["dropped_reports_at_observation"]
    interval = 60. if elapsed < 900. else 300.
    if context.guard_mode in ("stale", "reach") and age > 10.:
        interval = 5.
    dt = min(interval, MAX_SECONDS-elapsed, until if future is not None else MAX_SECONDS)

    def build(duration):
        batch = candidate_batch(targets, bows, scenario.radius, forcing.polar, future,
                                forcing.wind_speed, forcing.wave_height, duration, until, MAX_SECONDS-elapsed)
        allowed = np.ones(batch.captured_after_s.shape, dtype=bool)
        assessment = None
        required = context.additional_planning_reserve_metres
        if context.ais is not None:
            if context.guard_mode == "reach" or context.guard_mode == "stale" and age > 10.:
                required += .5*.030*(age+duration)**2
            assessment = _report_assessment(context.ais, elapsed, scenario.target-targets, batch, duration, required)
            allowed &= assessment.report_margin_m > required
        if scenario.obstacle is not None:
            center, radius = scenario.obstacle
            origin = (scenario.target-targets)[:, None, :]
            allowed &= ~np.isfinite(circle_entry(origin, batch.turn_displacement, center, radius))
            allowed &= ~np.isfinite(circle_entry(origin+batch.turn_displacement,
                                                 batch.displacement-batch.turn_displacement, center, radius))
        return batch, allowed, assessment, required

    batch, allowed, assessment, required = build(dt)
    contingency = False
    if context.contingency_short_step and dt > 5. and not allowed.all(axis=1).all() and np.any(~allowed.any(axis=1)):
        dt = 5.
        batch, allowed, assessment, required = build(dt)
        contingency = True
    margin = np.full(allowed.shape, 2000.) if assessment is None else assessment.report_margin_m
    features = np.stack((np.clip(margin/100., -20., 20.), np.full(allowed.shape, age/60.),
                         np.full(allowed.shape, lost/10.), np.broadcast_to(allowed.mean(axis=1)[:, None], allowed.shape),
                         np.full(allowed.shape, required/100.), np.full(allowed.shape, float(context.ais is not None))), axis=-1).astype(np.float32)
    audit = {"profile": context.profile, "guard_mode": context.guard_mode,
             "static_audit_reserve_m": 3., "required_additional_clearance_m": required,
             "maximum_report_age_seconds": age, "dropped_reports_at_observation": int(lost),
             "contingency_short_step_used": contingency,
             "truth_used_for_features_or_mask": False,
             "truth_audit": "Call assess_applied after selecting every applied action, including first hit.",
             "report_margin_scope": "Conservative lower bound for the whole proposed interval."}
    return JointEvaluation(batch, dt, allowed, features, audit, assessment)


def assess_applied(context, elapsed, targets, evaluation, row, candidate, hit=None):
    """Audit a selected action only, truncating turn/sail motion at first hit.

    ``hit`` is seconds after this decision, not absolute episode time. None
    audits the whole applied interval. Clear water returns two infinities.
    The truth result cannot alter any already returned feature or mask.
    """
    if context.ais is None:
        return float("inf"), float("inf")
    targets = np.asarray(targets, dtype=float).reshape(-1, 2)
    duration = evaluation.dt if hit is None else float(hit)
    return context.ais.assess_applied(elapsed, context.scenario.target-targets,
                                      evaluation.batch, evaluation.dt, row, candidate, duration)


def weather_observation(context: JointContext, elapsed, boat=None) -> WorldObservation:
    """Expose only currently observed reduced forcing, without AIS truth."""
    elapsed = float(elapsed)
    if not np.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Observed weather clock must be finite and nonnegative.")
    forcing, _, _ = active_forcing(context.scenario, elapsed)
    p = forcing.polar
    angle = np.radians(p.wind_from)
    wind = Wind(float(-forcing.wind_speed*np.sin(angle)), float(-forcing.wind_speed*np.cos(angle)),
                forcing.wind_speed, p.wind_from)
    flow = p.flow
    current = Current(float(flow[0]), float(flow[1]), float(np.linalg.norm(flow)),
                      float(np.degrees(np.arctan2(*flow)) % 360.), 0.)
    when = context.start_utc+np.timedelta64(int(round(elapsed*1e9)), "ns")
    boat = BoatState(context.origin_latitude, context.origin_longitude, context.scenario.bow, 0.) if boat is None else boat
    return WorldObservation(when, boat, wind, Wave(forcing.wave_height, p.wind_from, 5.),
                            current, None, None, None, ())


def observed_weather(context: JointContext, elapsed) -> np.ndarray:
    """Return 12 past/current five-minute EN observations, with launch padding.

    This reduced observer samples at UTC five-minute boundaries, independent
    of policy decisions. No negative prelaunch weather is fabricated: earlier
    slots repeat launch weather, matching the original encoder's padding.
    Its output is raw [wind_u,wind_v,current_u,current_v,wave_height]; use the
    pretrained checkpoint's waypoint projection and fitted normalization.
    """
    elapsed = float(elapsed)
    if not np.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Observed weather clock must be finite and nonnegative.")
    start_seconds = int(context.start_utc.astype("datetime64[s]").astype(np.int64))
    current_window = int(np.floor((start_seconds+elapsed)/300.))
    sample_times = np.maximum(0., np.arange(current_window-11, current_window+1)*300.-start_seconds)
    values = []
    for when in sample_times:
        obs = weather_observation(context, float(when))
        values.append((obs.wind.u, obs.wind.v, obs.current.u, obs.current.v, obs.wave.height))
    return np.asarray(values, dtype=np.float32)
