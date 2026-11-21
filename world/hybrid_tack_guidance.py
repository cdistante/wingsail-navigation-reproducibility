"""Offer guarded first-tack changes to the native schedule planner."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import numpy as np
import torch

from ._geo import local_east_north_metres
from .ais_avoidance import AISPrediction
from .forecast_tacks import ForecastTackPlanner, steady_sailing_trim
from .learned_mark_policy import FEATURE_NAMES, PolarSnapshot, candidate_batch
from .temporal_recovery import HISTORY_STEPS, RecoveryHoldModel, encode_hold_history


WIDTH = len(FEATURE_NAMES) + 6 + 48
TURN_FEATURE = FEATURE_NAMES.index("turn_seconds_180")


class HybridTackGuidance:
    """Rank first tacks with the frozen critic and compare their native schedules.

    The native planner supplies its best committed schedule for each first
    tack, including its turn timing, tracking correction and waypoint gate.
    The reduced trim model supplies critic features. The native AIS/COLREG
    and actuator guards remain downstream of the proposal.
    """

    def __init__(self, model_dir: Path, *, gain_seconds: float, strength: float,
                 acceleration_bound: float, reserve: float, report_delay: float, control_seconds: float):
        torch.set_num_threads(1)
        checkpoint = torch.load(model_dir / "fresh_weather_encoder.pt", map_location="cpu", weights_only=True)
        if checkpoint.get("kind") != "synthetic_causal_weather_denoising_v1":
            raise ValueError("The hybrid requires the frozen synthetic weather encoder.")
        self.encoder = RecoveryHoldModel()
        loaded = self.encoder.load_state_dict(checkpoint["encoder_state"], strict=False)
        expected = {name for name in self.encoder.state_dict() if name.startswith("head.")}
        if set(loaded.missing_keys) != expected or loaded.unexpected_keys:
            raise ValueError("Mismatched weather encoder.")
        self.encoder.eval().requires_grad_(False)
        self.history_mean = torch.as_tensor(checkpoint["history_mean"], dtype=torch.float32)
        self.history_scale = torch.as_tensor(checkpoint["history_scale"], dtype=torch.float32)
        with np.load(model_dir / "joint_critic.npz") as values:
            self.mean, self.scale = values["mean"], values["scale"]
            self.layers = tuple(values[f"layer_{i}"] for i in range(6))
        expected_shapes = ((64, WIDTH), (64,), (32, 64), (32,), (1, 32), (1,))
        if (self.mean.shape != (WIDTH,) or self.scale.shape != (WIDTH,) or np.any(self.scale <= 0)
                or any(value.shape != shape or not np.isfinite(value).all()
                       for value, shape in zip(self.layers, expected_shapes, strict=True))
                or self.history_mean.shape != (5,) or self.history_scale.shape != (5,)
                or not torch.isfinite(self.history_mean).all() or not torch.isfinite(self.history_scale).all()
                or torch.any(self.history_scale <= 0)
                or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all()
                or not np.isfinite([gain_seconds, strength, report_delay, control_seconds]).all()
                or gain_seconds < 0 or strength <= 0 or report_delay < 0 or control_seconds <= 0):
            raise ValueError("Invalid hybrid model or control settings.")
        self.gain_seconds = gain_seconds
        self.strength = strength
        self.acceleration_bound = acceleration_bound
        self.reserve = reserve
        self.report_delay = report_delay
        self.control_seconds = control_seconds
        self.history = deque(maxlen=HISTORY_STEPS)
        self.chart_allows = None

    def observe(self, observation):
        wind, current, wave = observation.wind, observation.current, observation.wave
        window = int(observation.time.astype("datetime64[s]").astype(np.int64)) // 300
        if wind is None or current is None or wave is None or (self.history and window <= self.history[-1][0]):
            return
        values = np.array((wind.u, wind.v, current.u, current.v, wave.height))
        if np.isfinite(values).all():
            self.history.append((window, values))

    @torch.inference_mode()
    def weather_embedding(self, target):
        windows, weather = zip(*self.history, strict=True)
        recent = np.arange(windows[-1] + 1 - HISTORY_STEPS, windows[-1] + 1)
        indices = np.maximum(0, np.searchsorted(windows, recent, side="right") - 1)
        projected, _ = encode_hold_history(np.asarray(weather)[indices], target, False)
        history = (torch.from_numpy(projected[None]) - self.history_mean) / self.history_scale
        return self.encoder.encoder(self.encoder.projection(history) + self.encoder.position)[0, -1].numpy()

    def _corrections(self, features):
        w1, b1, w2, b2, w3, b3 = self.layers
        x = np.clip((features - self.mean) / self.scale, -8., 8.)
        x = np.tanh(x @ w1.T + b1)
        x = np.tanh(x @ w2.T + b2)
        return (self.strength * 7200. * np.tanh(x @ w3.T + b3))[:, 0]

    def _report_features(self, observation, batch, dt):
        prediction = AISPrediction.from_observation(observation, acceleration_bound_mps2=self.acceleration_bound,
                                                     clearance_reserve_metres=self.reserve)
        width = batch.features.shape[1]
        if not prediction.targets:
            return np.tile((20., 0., 0., 1., 0., 0.), (width, 1)), np.ones(width, dtype=bool)
        age = float(prediction.report_ages_seconds.max())
        lost = max(0., (age - self.report_delay) / self.control_seconds)
        required = .5 * .030 * (age + self.control_seconds)**2 if lost >= 1. else 0.
        turn = np.clip(batch.features[0, :, TURN_FEATURE].astype(float) * 180., 0., dt)
        displacements = batch.displacement[0]
        turning = batch.turn_displacement[0]
        lengths = np.linalg.norm(turning, axis=1) + np.linalg.norm(displacements-turning, axis=1)
        center_dist = np.linalg.norm(prediction.positions_metres, axis=1)
        ship_reach = dt * np.linalg.norm(prediction.velocities_mps, axis=1)
        margins = (center_dist[None] - ship_reach[None] - prediction.radii_at(dt)[None] - lengths[:, None]).min(axis=1)
        for index in np.flatnonzero(margins <= required):
            middle = turn[index]
            if 0. < middle < dt:
                vertices, times = ((0., 0.), turning[index], displacements[index]), (0., middle, dt)
            else:
                vertices, times = ((0., 0.), displacements[index]), (0., dt)
            margins[index] = prediction.assess_local_rollout(
                vertices, times, compute_contact_times=False
            ).minimum_clearance_metres
        allowed = margins > required
        features = np.column_stack((np.clip(margins / 100., -20., 20.),
                                    np.broadcast_to((age/60., lost/10., float(allowed.mean()), required/100., 1.),
                                                    (width, 5))))
        return features, allowed

    @staticmethod
    def _first_trims(targets, bows, radius, catalogue, durations):
        polar, wind, wave, _ = catalogue[0]
        future = catalogue[-1][0] if len(catalogue) > 1 else None
        remaining = durations.sum()
        return candidate_batch(targets, bows, radius, polar, future, wind, wave, durations[0],
                               remaining - durations[-1] if future is not None else 0., remaining)

    @staticmethod
    def _schedule_telemetry(label, schedule):
        score, first, following, switch, closest = schedule
        # Score item 1 is an arrival time only in the native capture class.
        return {
            f"hybrid_tack_{label}_schedule_class": int(score[0]),
            f"hybrid_tack_{label}_eta_s": float(score[1]) if score[0] == 0. else None,
            f"hybrid_tack_{label}_first_mode": int(first),
            f"hybrid_tack_{label}_following_mode": int(following),
            f"hybrid_tack_{label}_switch_window": int(switch),
            f"hybrid_tack_{label}_closest_m": float(closest),
        }

    def choose(self, observation, goal, baseline, alternatives, catalogue, durations, radius):
        began = perf_counter()
        details = {
            "hybrid_tack_decision": "physics fallback",
            "hybrid_tack_threshold_s": self.gain_seconds,
            "hybrid_tack_eta_source": "native committed schedule",
            **self._schedule_telemetry("physics", baseline),
        }
        target = np.asarray(local_east_north_metres(goal.latitude, goal.longitude,
                                                    observation.boat.latitude, observation.boat.longitude))
        distance = np.linalg.norm(target)
        if radius != 50. or not radius < distance <= 5500. or len(alternatives) < 2 or not self.history:
            details["hybrid_tack_decision"] = "outside policy support"
            return baseline, details
        first = self._first_trims(target[None], np.array([observation.boat.heading]), radius, catalogue, durations)
        slots = catalogue[0][3]
        first_index = int(np.flatnonzero(slots == baseline[1])[0])
        reports, allowed = self._report_features(observation, first, durations[0])
        candidates = [i for i, number in enumerate(slots) if number in alternatives and allowed[i]
                      and (self.chart_allows is None or self.chart_allows(catalogue[0][0].velocity[i]))]
        if first_index not in candidates or len(candidates) < 2:
            details["hybrid_tack_decision"] = "report or chart screen"
            return baseline, details
        # A legal, exact capture within the first commitment takes priority.
        physical_eta = details["hybrid_tack_physics_eta_s"]
        if physical_eta is not None and physical_eta <= durations[0]:
            details["hybrid_tack_decision"] = "physics capture in first interval"
            return baseline, details
        embedding = np.broadcast_to(self.weather_embedding(target), (len(slots), 48))
        features = np.column_stack((first.features[0], reports, embedding))
        costs = np.maximum(durations[0], first.physics_cost_s[0] + self._corrections(features))
        costs = np.where(np.isfinite(first.captured_after_s[0]), first.captured_after_s[0], costs)
        learned = int(candidates[np.argmin(costs[candidates])])
        polar = catalogue[0][0]
        proposed = alternatives[slots[learned]]
        details.update(hybrid_tack_physics_bow_deg=float(polar.bow[first_index]),
                       hybrid_tack_proposed_bow_deg=float(polar.bow[learned]),
                       **self._schedule_telemetry("learned", proposed))
        if learned == first_index:
            details["hybrid_tack_decision"] = "same first trim"
            return baseline, details
        learned_eta = details["hybrid_tack_learned_eta_s"]
        details["hybrid_tack_wall_seconds"] = perf_counter() - began
        if learned_eta is None:
            details["hybrid_tack_decision"] = "veto forecast miss"
        elif physical_eta is not None and physical_eta - learned_eta <= self.gain_seconds + 1e-6:
            details["hybrid_tack_decision"] = "veto insufficient gain"
        else:
            details["hybrid_tack_decision"] = "accepted first trim"
            return proposed, details
        return baseline, details


class HybridForecastTackPlanner(ForecastTackPlanner):
    """Use the native best-schedule-per-first-leg extension point."""

    def __init__(self, *args, hybrid: HybridTackGuidance, **kwargs):
        self.hybrid = hybrid
        self._hybrid_catalogue = {}
        self._hybrid_details = {}
        # The alternate schedule callback is overridden below with the joint critic.
        kwargs["arrival_selector"] = hybrid
        super().__init__(*args, **kwargs)

    def suspend(self, reason="nominal reference not applied"):
        super().suspend(reason)
        self._hybrid_catalogue.clear()
        self._hybrid_details = {}

    def reference(self, observation, goal):
        self.hybrid.observe(observation)
        reference = super().reference(observation, goal)
        if reference is not None:
            self.maneuver_telemetry.update(self._hybrid_details)
        return reference

    def _polar(self, observation):
        # Same native polar search, retaining the winning flap for the critic.
        wind, current = observation.wind, observation.current
        assert wind is not None and current is not None
        parameters = self.dynamics.parameters
        flow = np.array((current.u, current.v))
        headings = (wind.direction + self.offsets) % 360.
        velocities = np.full((len(headings), 2), np.nan)
        flaps = np.zeros(len(headings))
        for index, heading in enumerate(headings):
            rotation = self.dynamics._body_to_earth(heading)
            best_surge = .05
            for flap in (parameters.flap_max, parameters.flap_min, 0.):
                solved = steady_sailing_trim(observation, self.dynamics, heading, flap, minimum_surge=best_surge)
                if solved is not None:
                    water, _ = solved
                    best_surge = water[0]
                    velocities[index] = rotation @ water + flow
                    flaps[index] = flap
        modes = np.flatnonzero(np.isfinite(velocities).all(axis=1))
        slot = int(observation.time.astype("datetime64[s]").astype(np.int64)) // self.interval_seconds
        if len(modes):
            polar = PolarSnapshot.from_modes(
                tuple(PolarTrim(velocities[i], headings[i], flaps[i]) for i in modes), flow, wind.direction
            )
            self._hybrid_catalogue[slot] = (polar, wind.speed, observation.wave.height if observation.wave else 0., modes)
        else:
            self._hybrid_catalogue.pop(slot, None)
        return velocities, headings, flow

    def _choose_arrival_schedule(self, observation, goal, baseline, best_by_first, window, second, *, scheduled_switch):
        catalogue = []
        for slot in range(window, window + self.horizon_intervals):
            if slot not in self._hybrid_catalogue or slot not in self._polars:
                break
            catalogue.append(self._hybrid_catalogue[slot])
        self._hybrid_catalogue = {slot: polar for slot, polar in self._hybrid_catalogue.items() if slot >= window}
        if scheduled_switch or self.maneuver_commit_seconds and second < self._commit_until or not catalogue:
            self._hybrid_details = {"hybrid_tack_decision": "scheduled or unavailable"}
            return baseline
        durations = np.full(len(catalogue), float(self.interval_seconds))
        durations[0] -= second % self.interval_seconds
        selected, self._hybrid_details = self.hybrid.choose(
            observation, goal, baseline, best_by_first, catalogue, durations, self.arrival_radius
        )
        self._hybrid_details.update(self.hybrid._schedule_telemetry("selected", selected))
        self._hybrid_details["hybrid_tack_schedule_interval_s"] = self.interval_seconds
        self._hybrid_details["hybrid_tack_schedule_window"] = window
        self._hybrid_details["hybrid_tack_decision_utc"] = str(observation.time)
        return selected


class PolarTrim(NamedTuple):
    velocity: np.ndarray
    bow: float
    flap: float
