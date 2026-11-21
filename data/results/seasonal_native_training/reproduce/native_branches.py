"""Matched exact-state native Fossen tack branches for seasonal training.

The plant is the existing 0.05 s semi-implicit Fossen model, evaluated using
its CasADi transcription.  This is NOT acados MPC: the helm is the existing
slew-limited steady-trim feedback law in ``run_legacy_fossen.drift_trim_action``.
Forecasts persist the *present* observed weather, never future archive truth.
The same report-only guard and helm apply to every strategy and branch.

An unsafe proposed command is replaced by a screened candidate.  If none is
safe the episode terminates, right-censored, before unsafe integration.  That
termination is a simulation stop, not a physically stationary sailing action.
Traffic truth is queried only after selecting a command, for offline labels.
No shore/depth avoidance is implied when the facade has no chart validator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from math import atan2, degrees, hypot, radians
from time import perf_counter
from typing import Callable

import casadi as ca
import numpy as np

from world._geo import haversine_metres, local_east_north_metres, normalise_time
from world.acados_fossen import discrete_fossen
from world.ais_avoidance import AISPrediction, _OWN_HULL_RADIUS_METRES
from world.boat import BoatAction, BoatState
from world.colreg import ColregPolicy
from world.fossen import Fossen3DOFSaildroneDynamics
from world.forecast_tacks import ForecastTackPlanner, steady_sailing_trim
from world.learned_tacks import FEATURE_NAMES as BASE_FEATURE_NAMES, tack_candidate_features
from world.mpc import SaildroneMPC
from world.scenario import Waypoint


FEATURE_NAMES = BASE_FEATURE_NAMES + (
    "native_predicted_capture", "native_eta_or_horizon_s", "native_predicted_closest_m",
    "first_duration_s", "following_course_along_goal", "following_course_across_goal",
    "steady_rudder_deg", "steady_flap_deg", "report_margin_m_clipped",
    "max_report_age_s", "reported_contact_count", "allowed_fraction", "colreg_active",
    "native_schedule_class", "switch_after_s",
)


@dataclass(frozen=True)
class NativeConfig:
    control_dt_s: float = 5.
    integration_dt_s: float = .05
    commit_seconds: float = 120.
    forecast_horizon_s: int = 2880
    capture_radius_m: float = 50.
    no_go_angle_deg: float = 45.
    reserve_m: float = 3.
    acceleration_bound_mps2: float = .026
    cap_seconds: float = 7200.
    audit_truth: bool = True
    colreg: bool = True

    def __post_init__(self):
        if self.control_dt_s != 5. or self.integration_dt_s != .05:
            raise ValueError("The native experiment fixes 5 s helm and 0.05 s integration.")
        if self.commit_seconds < 5 or self.commit_seconds % 5:
            raise ValueError("Commitment must be a positive multiple of 5 s.")
        if not 120 <= self.forecast_horizon_s <= 2880 or self.forecast_horizon_s % 120:
            raise ValueError("Forecast horizon must be a multiple of 120 s in [120, 2880].")
        if self.capture_radius_m <= 0 or self.reserve_m < 0 or self.cap_seconds <= 0:
            raise ValueError("Invalid radius, reserve, or episode cap.")


@dataclass(frozen=True)
class NativeChoice:
    id: str
    course_deg: float
    bow_deg: float
    following_course_deg: float
    following_bow_deg: float
    switch_after_s: float
    first_duration_s: float
    schedule_class: int
    predicted_eta_s: float | None
    predicted_closest_m: float
    steady_rudder_deg: float
    steady_flap_deg: float
    report_margin_m: float = float("inf")
    colreg_allowed: bool = True
    score: tuple[float, ...] = ()


@dataclass(frozen=True)
class NativeDecision:
    observation: object
    state: BoatState
    goal: Waypoint
    when: np.datetime64
    elapsed_s: float
    choices: tuple[NativeChoice, ...]
    features: np.ndarray
    allowed: np.ndarray
    physics_index: int
    weather_history: np.ndarray | None = None


@dataclass(frozen=True)
class NativeSnapshot:
    state: BoatState
    goals: tuple[Waypoint, ...]
    when: np.datetime64
    elapsed_s: float
    remaining_cap_s: float
    # The visible pending switch is causal state, not future environmental data.
    pending_switch_bow_deg: float | None = None


@dataclass
class EpisodeResult:
    success: bool
    status: str
    elapsed_s: float
    final_state: BoatState
    captures: list[dict] = field(default_factory=list)
    report_minimum_m: float = float("inf")
    truth_minimum_m: float = float("inf")
    controls: list[dict] = field(default_factory=list)
    control_attempts: list[dict] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    first_choice_requested: str | None = None
    first_choice_applied: str | None = None
    first_commit_requested_s: float = 0.
    first_commit_applied_s: float = 0.
    first_commit_completed: bool = False
    shield_overrides: int = 0
    censored: bool = True
    closest_m: float = float("inf")

    def summary(self):
        """Small JSON-compatible receipt; large telemetry is deliberately separate."""
        values = asdict(self)
        for key in ("trace", "controls", "control_attempts", "decisions"):
            values[key + "_count"] = len(values.pop(key))
        return values


@dataclass(frozen=True)
class BranchResult:
    choice_index: int
    choice: NativeChoice
    features: np.ndarray
    outcome: EpisodeResult
    snapshot: NativeSnapshot


class _PersistentPlanner(ForecastTackPlanner):
    """Retain every first-mode schedule; use observed forcing at every window."""
    def __init__(self, observation, dynamics, config):
        super().__init__(None, dynamics, config.no_go_angle_deg, config.capture_radius_m,
                         arrival_selector=object())
        self.horizon_intervals = config.forecast_horizon_s // self.interval_seconds
        self.observation = observation
        self.schedules = {}
        self.baseline = None
        self.trims = {}
        self._constant_polar = None

    def _sample(self, observation, goal, when):
        return self.observation

    def _polar(self, observation):
        if self._constant_polar is None:
            headings = (observation.wind.direction + self.offsets) % 360.
            current = np.array((observation.current.u, observation.current.v))
            velocities = np.full((len(headings), 2), np.nan)
            p = self.dynamics.parameters
            for i, heading in enumerate(headings):
                best = .05
                for flap in (p.flap_max, p.flap_min, 0.):
                    trim = steady_sailing_trim(observation, self.dynamics, float(heading), flap,
                                               minimum_surge=best)
                    if trim is not None:
                        water, rudder = trim
                        best = float(water[0])
                        velocities[i] = self.dynamics._body_to_earth(heading) @ water + current
                        self.trims[i] = float(rudder), float(flap)
            self._constant_polar = velocities, headings, current
        return self._constant_polar

    def _choose_arrival_schedule(self, observation, goal, baseline, best_by_first,
                                 window, second, scheduled_switch=False):
        self.schedules = dict(best_by_first)
        self.baseline = baseline
        return baseline


def _clock(when, seconds):
    return normalise_time(when) + np.timedelta64(int(round(seconds * 1e9)), "ns")


def _name(goal):
    return getattr(goal, "name", "")


def _geographic_path(state, local):
    """Use native per-step latitude in the longitude update, not one tangent map."""
    delta = np.diff(local[:2], axis=1)
    lat = np.r_[state.latitude, state.latitude + np.cumsum(delta[1]) / 111320.]
    lon_delta = delta[0] / (111320. * np.cos(np.radians(lat[:-1])))
    lon = np.r_[state.longitude, state.longitude + np.cumsum(lon_delta)]
    lon = (lon + 180.) % 360. - 180.
    return np.column_stack((lat, lon))


def _first_capture(path, goal, radius):
    """Earliest great-circle gate entry on the native 0.05 s polyline."""
    lat, lon = np.radians(path[:, 0]), np.radians(path[:, 1])
    glat, glon = radians(goal.latitude), radians(goal.longitude)
    h = np.sin((lat - glat) / 2)**2 + np.cos(lat) * np.cos(glat) * np.sin((lon - glon) / 2)**2
    distance = 2. * 6371008.8 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
    east = 6371008.8 * (lon - glon) * np.cos(glat)
    north = 6371008.8 * (lat - glat)
    relative = np.column_stack((east, north))
    delta = np.diff(relative, axis=0)
    dot = np.einsum("ij,ij->i", relative[:-1], delta)
    norm = np.einsum("ij,ij->i", delta, delta)
    closest_fraction = np.clip(-dot / np.maximum(norm, 1e-30), 0, 1)
    approximate = np.linalg.norm(relative[:-1] + closest_fraction[:, None] * delta, axis=1)
    closest = min(float(distance.min()), float(approximate.min(initial=np.inf)))
    if distance[0] <= radius:
        return 0., closest
    # Include possible outside/outside grazing, then verify with haversine.
    candidates = np.flatnonzero((approximate <= radius + 1e-3) | (distance[1:] <= radius))
    for index in candidates:
        left, right = path[index], path[index + 1]
        f = float(closest_fraction[index]) if distance[index + 1] > radius else 1.
        near = left + f * (right - left)
        if haversine_metres(*near, goal.latitude, goal.longitude) > radius:
            continue
        low, high = 0., f
        for _ in range(32):
            middle = (low + high) / 2
            point = left + middle * (right - left)
            if haversine_metres(*point, goal.latitude, goal.longitude) <= radius:
                high = middle
            else:
                low = middle
        return float(index + high), closest
    return None, closest


class NativeEngine:
    def __init__(self, environment, *, config=None, dynamics=None):
        self.environment = environment
        self.config = config or NativeConfig()
        declared_acceleration = getattr(environment, "max_target_acceleration_mps2", 0.)
        if declared_acceleration > self.config.acceleration_bound_mps2 + 1e-12:
            raise ValueError("Declared contact acceleration exceeds the report guard's bound.")
        self.dynamics = dynamics or Fossen3DOFSaildroneDynamics()
        if self.dynamics.parameters.max_integration_step != .05:
            raise ValueError("Native plant parameters must specify 0.05 s integration.")
        self.mpc = SaildroneMPC()
        self.colreg = ColregPolicy()
        x, u, p = ca.SX.sym("native_x", 9), ca.SX.sym("native_u", 2), ca.SX.sym("native_p", 5)
        self._one = ca.Function("seasonal_native_step", [x, u, p],
                                [discrete_fossen(self.dynamics, x, u, p, .05)])
        self._mapped = {}

    def integrate(self, state, action, observation, duration_s=5.):
        """Return final BoatState, all 0.05 s geographic knots, and body states."""
        steps = int(round(duration_s / .05))
        if steps < 1 or abs(steps * .05 - duration_s) > 1e-8:
            raise ValueError("Native integration duration must be a positive multiple of 0.05 s.")
        if steps not in self._mapped:
            self._mapped[steps] = self._one.mapaccum(steps)
        nu = self.dynamics._relative_velocity_from_state(state)
        x = np.array((0., 0., radians(state.heading), *nu, radians(state.flap),
                      radians(state.rudder), radians(state.flap)))
        force = np.array((observation.wind.u, observation.wind.v, observation.current.u,
                          observation.current.v, observation.wave.height))
        u = np.radians((action.rudder, action.flap))
        generated = np.asarray(self._mapped[steps](x, np.tile(u[:, None], (1, steps)),
                                                  np.tile(force[:, None], (1, steps))))
        local = np.column_stack((x, generated))
        path = _geographic_path(state, local)
        final = self._state_at(local[:, -1], path[-1], action, observation,
                               previous=local[:, -2])
        return final, path, local

    def _state_at(self, x, position, action, observation, previous=None):
        heading = float(degrees(x[2]) % 360.)
        velocity = self.dynamics._body_to_earth(heading) @ x[3:5] + np.array((observation.current.u, observation.current.v))
        before = x if previous is None else previous
        tau = (self.dynamics._wing_load(before[3:6], degrees(before[2]) % 360., degrees(before[6]), observation)
               + self.dynamics._rudder_load(before[3:6], float(degrees(x[7]))))
        return BoatState(float(position[0]), float(position[1]), heading, hypot(*velocity),
                         float(degrees(x[7])), float(degrees(x[6])),
                         self.dynamics._heel_from_lateral_load(float(tau[1])),
                         float(x[3]), float(x[4]), float(x[5]))

    def action(self, state, choice):
        """Identical equation and slew limits to native drift_trim_action."""
        p = self.dynamics.parameters
        error = (state.heading - choice.bow_deg + 180.) % 360. - 180.
        rudder = choice.steady_rudder_deg + .9 * error + 2.5 * degrees(state.yaw_rate)
        return BoatAction(float(np.clip(rudder, max(p.rudder_min, state.rudder - self.mpc.max_rudder_change_per_update_degrees),
                                         min(p.rudder_max, state.rudder + self.mpc.max_rudder_change_per_update_degrees))),
                          float(np.clip(choice.steady_flap_deg, max(p.flap_min, state.flap - self.mpc.max_flap_change_per_update_degrees),
                                         min(p.flap_max, state.flap + self.mpc.max_flap_change_per_update_degrees))))

    def _prediction(self, observation):
        return AISPrediction.from_observation(observation,
            acceleration_bound_mps2=self.config.acceleration_bound_mps2,
            clearance_reserve_metres=self.config.reserve_m)

    def _dropped_epochs(self, when):
        receiver = getattr(self.environment, "reporting", None)
        clock = getattr(self.environment, "elapsed_seconds", None)
        if receiver is None or clock is None:
            return 0
        elapsed = clock(when)
        latest = receiver.latest_epoch(elapsed)
        available = elapsed - receiver.delay_seconds
        expected = np.floor(np.nextafter(available / receiver.interval_seconds, np.inf)) * receiver.interval_seconds
        return max(0, int(round((expected - latest) / receiver.interval_seconds)))

    def _screen(self, state, choice, observation, dt):
        action = self.action(state, choice)
        final, path, local = self.integrate(state, action, observation, dt)
        prediction = self._prediction(observation)
        times = np.arange(len(path)) * .05
        east, north = local_east_north_metres(path[:, 0], path[:, 1], state.latitude, state.longitude)
        assessment = prediction.assess_local_rollout(np.column_stack((east, north)), times,
                                                     compute_contact_times=False)
        active = self.config.colreg and any(item.active for item in self.colreg.assess(state, observation))
        relative = (choice.course_deg - state.heading + 180.) % 360. - 180.
        permitted = not (active and -self.colreg.starboard_turn_prohibited_sector_degrees <= relative <= -1.)
        validator = getattr(self.environment, "chart_path_is_clear", None)
        chart_clear = True if validator is None else bool(validator(path, observation))
        return assessment.minimum_clearance_metres, permitted and chart_clear, final, path, local, prediction, action

    def decision(self, state, goal, when, elapsed_s=0.):
        when = normalise_time(when)
        obs = self.environment.observe(state, when)
        if obs.wind is None or obs.current is None or obs.wave is None:
            raise ValueError("Native decisions require observed wind, current, and waves.")
        planner = _PersistentPlanner(obs, self.dynamics, self.config)
        planner._plan(obs, goal, 0, 0.)
        choices = []
        if planner.baseline is not None:
            velocities, bows, _ = planner._constant_polar
            for first, schedule in sorted(planner.schedules.items()):
                score, _, following, switch, closest = schedule
                course = degrees(atan2(*velocities[first])) % 360.
                after = degrees(atan2(*velocities[following])) % 360.
                switch_after = float(switch * 120) if first != following else float(self.config.forecast_horizon_s)
                rudder, flap = planner.trims[first]
                choice = NativeChoice(f"mode_{first:02d}", course, float(bows[first]), after,
                                      float(bows[following]), switch_after,
                                      min(self.config.commit_seconds, switch_after), int(score[0]),
                                      float(score[1]) if score[0] == 0 else None, float(closest), rudder, flap,
                                      score=tuple(map(float, score)))
                margin, permitted, *_ = self._screen(state, choice, obs, self.config.control_dt_s)
                choices.append(replace(choice, report_margin_m=margin, colreg_allowed=permitted))
        choices = tuple(choices)
        allowed = np.array([item.report_margin_m > 0 and item.colreg_allowed for item in choices], dtype=bool)
        legal = np.flatnonzero(allowed)
        physics_index = min(legal, key=lambda i: choices[i].score) if len(legal) else -1
        active = self.config.colreg and any(item.active for item in self.colreg.assess(state, obs))
        prediction = self._prediction(obs)
        max_age = float(prediction.report_ages_seconds.max(initial=0.))
        target = local_east_north_metres(goal.latitude, goal.longitude, state.latitude, state.longitude)
        bearing = atan2(*target)
        rows = []
        for item in choices:
            relative = radians(item.following_course_deg) - bearing
            rows.append(np.r_[tack_candidate_features(obs, goal, item.course_deg, item.bow_deg),
                float(item.schedule_class == 0), item.predicted_eta_s if item.predicted_eta_s is not None else self.config.forecast_horizon_s,
                item.predicted_closest_m, item.first_duration_s, np.cos(relative), np.sin(relative),
                item.steady_rudder_deg, item.steady_flap_deg, np.clip(item.report_margin_m, -10000., 10000.),
                max_age, len(obs.ais_targets), float(allowed.mean()) if len(allowed) else 0., float(active),
                item.schedule_class, item.switch_after_s])
        features = np.asarray(rows, dtype=float).reshape(len(rows), len(FEATURE_NAMES))
        history = getattr(self.environment, "weather_history", None)
        weather_history = None if history is None else np.asarray(history(state.latitude, state.longitude, when))
        return NativeDecision(obs, state, goal, when, float(elapsed_s), choices, features,
                              allowed, int(physics_index), weather_history)

    def _truth_margin(self, path, times, when, prediction):
        """Offline only: native 0.05 s contact truth, held report uncertainty +3m."""
        truth = getattr(self.environment, "truth_targets", None)
        if not self.config.audit_truth or truth is None:
            return float("inf")
        # All truth targets, including contacts not yet present in the receiver.
        frames = [truth(_clock(when, float(t))) for t in times]
        ids = {item.mmsi for frame in frames for item in frame}
        report_index = {item.mmsi: i for i, item in enumerate(prediction.targets)}
        radii = prediction.radii_at(times)
        minimum = float("inf")
        for mmsi in ids:
            contact = [next((item for item in frame if item.mmsi == mmsi), None) for frame in frames]
            if any(item is None for item in contact):
                raise ValueError("Truth target membership changed inside one native control interval.")
            east, north = local_east_north_metres(np.array([item.latitude for item in contact]),
                np.array([item.longitude for item in contact]), path[0, 0], path[0, 1])
            oe, on = local_east_north_metres(path[:, 0], path[:, 1], path[0, 0], path[0, 1])
            relative = np.column_stack((east - oe, north - on))
            if mmsi in report_index:
                radius = radii[:, report_index[mmsi]]
            else:
                # Unknown/undelivered reports do not enter the actor; audit the
                # same hull+25m+5m+3m envelope with at least elapsed growth.
                hull = .5 * hypot(contact[0].length_metres or 50., contact[0].beam_metres or 0.) + _OWN_HULL_RADIUS_METRES
                radius = hull + 25. + 5. + self.config.reserve_m + np.maximum(.05 * times, .5 * self.config.acceleration_bound_mps2 * times**2)
            delta = np.diff(relative, axis=0)
            norm = np.einsum("ij,ij->i", delta, delta)
            fraction = np.clip(-np.einsum("ij,ij->i", relative[:-1], delta) / np.maximum(norm, 1e-30), 0., 1.)
            center = np.linalg.norm(relative[:-1] + fraction[:, None] * delta, axis=1)
            # Using the larger endpoint radius is a conservative chord bound.
            candidate = center - np.maximum(radius[:-1], radius[1:])
            minimum = min(minimum, float(candidate.min(initial=np.inf)))
        return minimum

    def run(self, initial_state, goals, start_utc, *, strategy=None, forced_first=None,
            first_commit_seconds=None, cap_seconds=None, trace=True, on_decision=None,
            initial_elapsed_s=0., pending_switch_bow_deg=None):
        goals = tuple(goals)
        start = normalise_time(start_utc)
        cap = float(self.config.cap_seconds if cap_seconds is None else cap_seconds)
        if cap <= 0 or abs(cap / .05 - round(cap / .05)) > 1e-6:
            raise ValueError("Episode cap must be positive and a multiple of 0.05 s.")
        state, elapsed, goal_index = initial_state, 0., 0
        result = EpisodeResult(False, "censored_cap", 0., state)
        chosen, next_decision, decision_count = None, 0., 0
        requested_commit = self.config.commit_seconds if first_commit_seconds is None else float(first_commit_seconds)
        result.first_commit_requested_s = requested_commit
        pending = pending_switch_bow_deg
        first_commit_interrupted = False
        while elapsed < cap - 1e-8 and goal_index < len(goals):
            control_started = perf_counter()
            callback_seconds = 0.
            goal = goals[goal_index]
            when = _clock(start, elapsed)
            if haversine_metres(state.latitude, state.longitude, goal.latitude, goal.longitude) <= self.config.capture_radius_m:
                result.captures.append({"name": _name(goal), "index": goal_index, "elapsed_s": elapsed,
                    "utc": str(when) + "Z", "distance_m": haversine_metres(state.latitude, state.longitude, goal.latitude, goal.longitude),
                    "latitude": state.latitude, "longitude": state.longitude, "scope": "great-circle native polyline"})
                goal_index += 1
                chosen, next_decision, pending = None, elapsed, None
                continue
            if chosen is None or elapsed >= next_decision - 1e-8:
                decision_started = perf_counter()
                decision = self.decision(state, goal, when, initial_elapsed_s + elapsed)
                if decision.physics_index < 0:
                    result.status = "no_safe_native_action"
                    result.control_attempts.append({"elapsed_s": elapsed, "utc": str(when) + "Z",
                        "goal": _name(goal), "applied": False, "duration_s": 0.,
                        "complete_control_seconds": perf_counter() - control_started,
                        "reason": result.status})
                    break
                if decision_count == 0 and forced_first is not None:
                    index = int(forced_first)
                elif pending is not None:
                    legal = np.flatnonzero(decision.allowed)
                    index = int(min(legal, key=lambda i: abs((decision.choices[i].bow_deg - pending + 180) % 360 - 180)))
                else:
                    index = decision.physics_index if strategy is None else int(strategy(decision))
                if not 0 <= index < len(decision.choices):
                    raise ValueError("Strategy returned an invalid candidate index.")
                requested = index
                snapshot_pending = pending
                if not decision.allowed[index]:
                    index = decision.physics_index
                    result.shield_overrides += 1
                    if decision_count == 0:
                        first_commit_interrupted = True
                chosen = decision.choices[index]
                commit = requested_commit if decision_count == 0 else self.config.commit_seconds
                commit = min(commit, chosen.first_duration_s)
                next_decision = elapsed + commit
                pending = chosen.following_bow_deg if chosen.switch_after_s <= commit and abs((chosen.following_bow_deg - chosen.bow_deg + 180) % 360 - 180) > 1e-6 else None
                result.decisions.append({"elapsed_s": elapsed, "utc": str(when) + "Z", "goal": _name(goal),
                    "requested_index": requested, "selected_index": index, "physics_index": decision.physics_index,
                    "choice_id": chosen.id, "bow_deg": chosen.bow_deg, "commit_s": commit,
                    "predicted_eta_s": chosen.predicted_eta_s, "schedule_class": chosen.schedule_class,
                    "decision_seconds": perf_counter() - decision_started})
                if decision_count == 0:
                    result.first_choice_requested = decision.choices[requested].id
                    result.first_choice_applied = chosen.id
                if on_decision is not None:
                    snapshot = NativeSnapshot(state, goals[goal_index:], when, initial_elapsed_s + elapsed,
                                              cap - elapsed, snapshot_pending)
                    callback_started = perf_counter()
                    on_decision(snapshot, decision)
                    callback_seconds = perf_counter() - callback_started
                decision_count += 1
            obs = self.environment.observe(state, when)
            dt = round(min(self.config.control_dt_s, cap - elapsed) / .05) * .05
            if dt < .05 - 1e-8:
                break
            margin, permitted, final, path, local, prediction, action = self._screen(state, chosen, obs, dt)
            if margin <= 0. or not permitted:
                # Fresh 5 s report/helm screening can interrupt a tack commitment.
                fresh = self.decision(state, goal, when, initial_elapsed_s + elapsed)
                if fresh.physics_index < 0:
                    result.status = "no_safe_native_action"
                    result.control_attempts.append({"elapsed_s": elapsed, "utc": str(when) + "Z",
                        "goal": _name(goal), "applied": False, "duration_s": 0.,
                        "complete_control_seconds": perf_counter() - control_started - callback_seconds,
                        "reason": result.status})
                    break
                chosen = fresh.choices[fresh.physics_index]
                result.shield_overrides += 1
                if elapsed < requested_commit:
                    first_commit_interrupted = True
                next_decision, pending = elapsed + self.config.commit_seconds, None
                margin, permitted, final, path, local, prediction, action = self._screen(state, chosen, obs, dt)
                if margin <= 0. or not permitted:
                    raise RuntimeError("A freshly admitted native candidate failed deterministic re-screening.")
            complete_control_seconds = perf_counter() - control_started - callback_seconds
            hit, closest = _first_capture(path, goal, self.config.capture_radius_m)
            gate_entry = None
            applied_dt = dt
            if hit is not None:
                k = min(int(np.floor(hit)), len(path) - 2)
                f = float(hit - k)
                point = path[k] + f * (path[k + 1] - path[k])
                gate_entry = {"elapsed_s": elapsed + hit * .05,
                    "utc": str(_clock(when, hit * .05)) + "Z", "latitude": float(point[0]),
                    "longitude": float(point[1]), "distance_m": haversine_metres(*point, goal.latitude, goal.longitude)}
                # Continue/stop at an exact integrator knot. Gate-entry timing
                # is independently interpolated, never an interpolated plant reset.
                end = max(1, min(int(np.ceil(hit)), len(path) - 1))
                path = path[:end + 1]
                applied_dt = end * .05
                final = self._state_at(local[:, end], path[-1], action, obs, previous=local[:, end - 1])
                _, closest = _first_capture(path, goal, self.config.capture_radius_m)
            result.closest_m = min(result.closest_m, closest)
            times = np.arange(len(path)) * .05
            east, north = local_east_north_metres(path[:, 0], path[:, 1], state.latitude, state.longitude)
            applied_margin = prediction.assess_local_rollout(np.column_stack((east, north)), times,
                compute_contact_times=False).minimum_clearance_metres
            truth_margin = self._truth_margin(path, times, when, prediction)
            result.report_minimum_m = min(result.report_minimum_m, applied_margin)
            result.truth_minimum_m = min(result.truth_minimum_m, truth_margin)
            record = {"elapsed_s": elapsed, "utc": str(when) + "Z", "goal": _name(goal),
                "duration_s": applied_dt, "rudder_deg": action.rudder, "flap_target_deg": action.flap,
                "choice_id": chosen.id, "bow_deg": chosen.bow_deg, "report_minimum_m": applied_margin,
                "truth_minimum_m": truth_margin, "max_report_age_s": float(prediction.report_ages_seconds.max(initial=0.)),
                "native_substeps": len(path) - 1, "held_observation": True,
                "dropped_report_epochs": self._dropped_epochs(when),
                "complete_control_seconds": complete_control_seconds}
            result.controls.append(record)
            result.control_attempts.append({**record, "applied": True})
            if trace:
                result.trace.append({"elapsed_s": elapsed + applied_dt, "utc": str(_clock(when, applied_dt)) + "Z",
                                     **asdict(final)})
            if (not first_commit_interrupted and elapsed < requested_commit
                    and result.first_choice_applied == chosen.id):
                result.first_commit_applied_s += min(applied_dt, requested_commit - elapsed)
            state, elapsed = final, elapsed + applied_dt
            if hit is not None:
                result.captures.append({"name": _name(goal), "index": goal_index, **gate_entry,
                    "integrator_elapsed_s": elapsed, "scope": "great-circle native polyline"})
                goal_index += 1
                chosen, next_decision, pending = None, elapsed, None
        result.success = goal_index == len(goals)
        if result.success:
            result.status = "captured"
        result.censored = not result.success
        result.elapsed_s, result.final_state = elapsed, state
        result.first_commit_completed = result.first_commit_applied_s >= requested_commit - 1e-7
        return result

    def branches(self, snapshot, choice_indices, *, decision=None, cap_seconds=None,
                 first_commit_seconds=None, trace=False):
        """Fork identical exact state/UTC; labels end at the next real waypoint."""
        decision = decision or self.decision(snapshot.state, snapshot.goals[0], snapshot.when, snapshot.elapsed_s)
        cap = snapshot.remaining_cap_s if cap_seconds is None else min(cap_seconds, snapshot.remaining_cap_s)
        outcomes = []
        for index in choice_indices:
            index = int(index)
            outcome = self.run(snapshot.state, snapshot.goals[:1], snapshot.when,
                forced_first=index, first_commit_seconds=first_commit_seconds,
                cap_seconds=cap, trace=trace, initial_elapsed_s=snapshot.elapsed_s,
                pending_switch_bow_deg=snapshot.pending_switch_bow_deg)
            outcomes.append(BranchResult(index, decision.choices[index], decision.features[index].copy(), outcome, snapshot))
        return outcomes
