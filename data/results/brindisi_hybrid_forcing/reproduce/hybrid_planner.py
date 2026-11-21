"""Issue a causal wind forecast for the native tactical schedule only."""

from __future__ import annotations

from dataclasses import replace
from math import atan2, degrees, radians

import numpy as np

from causal_weather import CausalEfficientNativeEngine
from native_branches import FEATURE_NAMES, NativeChoice, NativeDecision, NativeEngine, _PersistentPlanner, _clock
from world._geo import local_east_north_metres, normalise_time
from world.forecast_tacks import steady_ground_polar
from world.learned_tacks import tack_candidate_features


class _WindSchedule(_PersistentPlanner):
    def __init__(self, observation, dynamics, config, wind_forecast):
        super().__init__(observation, dynamics, config)
        self.wind_forecast = wind_forecast
        self.commit_seconds = config.commit_seconds
        self.slot_wind = {}
        self.wind_polars = {}
        self.forecast_failures = 0

    def _sample(self, observation, goal, when):
        # Native planning numbers the slots relative to this decision at 0.
        offset_s = int(when.astype("datetime64[s]").astype(np.int64))
        if offset_s < self.interval_seconds:
            self.slot_wind[0] = observation.wind
            return observation
        valid = _clock(observation.time, offset_s)
        try:
            wind = self.wind_forecast(valid).wind
            if wind is None or not np.isfinite((wind.u, wind.v, wind.speed, wind.direction)).all():
                raise ValueError("Invalid forecast wind.")
        except ValueError:
            self.forecast_failures += 1
            wind = observation.wind
        self.slot_wind[offset_s] = wind
        if wind == observation.wind:
            return observation
        return replace(observation, time=valid, wind=wind)

    def _polar(self, observation):
        if observation is self.observation:
            return super()._polar(observation)
        wind = observation.wind
        key = (wind.u, wind.v, wind.speed, wind.direction)
        if key not in self.wind_polars:
            self.wind_polars[key] = steady_ground_polar(observation, self.dynamics, self.offsets)
        return self.wind_polars[key]

    def choices(self, state, screen, dt):
        if self.baseline is None:
            return ()
        velocities, bows, _ = self._constant_polar
        choices = []
        for first, schedule in sorted(self.schedules.items()):
            score, _, following, switch, closest = schedule
            course = degrees(atan2(*velocities[first])) % 360.
            switch_velocities, switch_bows, _ = self._polars[switch]
            after = degrees(atan2(*switch_velocities[following])) % 360.
            switch_after = float(switch * self.interval_seconds) if first != following else float(
                self.horizon_intervals * self.interval_seconds
            )
            rudder, flap = self.trims[first]
            choice = NativeChoice(
                f"mode_{first:02d}", course, float(bows[first]), after,
                float(switch_bows[following]), switch_after, min(self.commit_seconds, switch_after),
                int(score[0]), float(score[1]) if score[0] == 0 else None, float(closest),
                rudder, flap, score=tuple(map(float, score)),
            )
            margin, permitted, *_ = screen(state, choice, self.observation, dt)
            choices.append(replace(choice, report_margin_m=margin, colreg_allowed=permitted))
        return tuple(choices)


class _WindScheduleDecisions(NativeEngine):
    """Supply tactical candidates below the unmodified Efficient AIS decision."""

    def decision(self, state, goal, when, elapsed_s=0.):
        if self.wind_provider is None or self.wind_provider.blend == 0.:
            return super().decision(state, goal, when, elapsed_s)
        when = normalise_time(when)
        obs = self.environment.observe(state, when)
        if obs.wind is None or obs.current is None or obs.wave is None:
            raise ValueError("Native decisions require observed wind, current, and waves.")
        try:
            forecast = self.wind_provider.at_decision(self.environment.as_of_archive, obs)
        except ValueError:
            self.wind_forecast_events.append({
                "elapsed_s": float(elapsed_s), "utc": str(when) + "Z", "goal": goal.name,
                "status": "persistence_fallback", "forecast_slots": 0, "changed_slots": 0,
            })
            return super().decision(state, goal, when, elapsed_s)

        planner = _WindSchedule(obs, self.dynamics, self.config, forecast)
        planner._plan(obs, goal, 0, 0.)
        choices = planner.choices(state, self._screen, self.config.control_dt_s)
        allowed = np.array([item.report_margin_m > 0. and item.colreg_allowed for item in choices], dtype=bool)
        legal = np.flatnonzero(allowed)
        physics_index = min(legal, key=lambda index: choices[index].score) if len(legal) else -1

        active = self.config.colreg and any(item.active for item in self.colreg.assess(state, obs))
        max_age = float(self._prediction(obs).report_ages_seconds.max(initial=0.))
        target = local_east_north_metres(goal.latitude, goal.longitude, state.latitude, state.longitude)
        bearing = atan2(*target)
        rows = []
        # Match the original 34-column native feature contract. Live forcing
        # remains measured. Only schedule time and geometry can change.
        for item in choices:
            relative = radians(item.following_course_deg) - bearing
            eta = item.predicted_eta_s if item.predicted_eta_s is not None else self.config.forecast_horizon_s
            rows.append(np.r_[
                tack_candidate_features(obs, goal, item.course_deg, item.bow_deg),
                float(item.schedule_class == 0), eta, item.predicted_closest_m,
                item.first_duration_s, np.cos(relative), np.sin(relative),
                item.steady_rudder_deg, item.steady_flap_deg,
                np.clip(item.report_margin_m, -10000., 10000.), max_age, len(obs.ais_targets),
                float(allowed.mean()) if len(allowed) else 0., float(active),
                item.schedule_class, item.switch_after_s,
            ])
        features = np.asarray(rows, dtype=float).reshape(len(rows), len(FEATURE_NAMES))
        history = self.environment.weather_history(state.latitude, state.longitude, when)
        offsets = list(planner.slot_wind)
        winds = list(planner.slot_wind.values())
        self.wind_forecast_events.append({
            "elapsed_s": float(elapsed_s), "utc": str(when) + "Z", "goal": goal.name,
            "status": "forecast_slots", "forecast_slots": len(winds),
            "changed_slots": sum(wind != obs.wind for wind in winds),
            "slot_read_failures": planner.forecast_failures,
            "slot_seconds": offsets, "wind_to_en_mps": [[float(wind.u), float(wind.v)] for wind in winds],
            "current_to_en_mps": [float(obs.current.u), float(obs.current.v)],
            "wave_height_m": float(obs.wave.height),
        })
        return NativeDecision(obs, state, goal, when, float(elapsed_s), choices, features, allowed,
                              int(physics_index), np.asarray(history))


class HybridWindEngine(CausalEfficientNativeEngine, _WindScheduleDecisions):
    """Native protected test helm with wind-only tactical forecast slots.

    The inherited Efficient decision wraps the candidate supplier. It retains
    its observed-weather preview, applied uncapped screen and continuous
    swept audits. The isolated supplier avoids changing any archived source.
    """

    def __init__(self, environment, *, config=None, dynamics=None, wind_provider=None):
        super().__init__(environment, config=config, dynamics=dynamics)
        self.wind_provider = wind_provider
        self.wind_forecast_events = []
