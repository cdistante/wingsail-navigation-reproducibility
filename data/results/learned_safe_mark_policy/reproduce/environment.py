"""Reduced receding-horizon environment for time-to-mark policy experiments."""
from __future__ import annotations

from dataclasses import dataclass, replace
import gzip
import json
from pathlib import Path
from typing import Callable

import numpy as np

from world.boat import BoatState
from world.currents import Current
from world.drift_guidance import DriftGuidance
from world.fossen import Fossen3DOFSaildroneDynamics
from world.learned_mark_policy import MAX_SECONDS, TIMEOUT_COST_SECONDS, PolarSnapshot, candidate_batch, circle_entry
from world.weather import Wind
from world.waves import Wave
from world.world import WorldObservation


INTERVAL_SECONDS = 300.0
SCENARIO_SEED = 4619
FAMILIES = ("opposed_weak", "cross_current", "steady_reach", "deteriorating")


@dataclass(frozen=True)
class Forcing:
    polar: PolarSnapshot
    wind_speed: float
    wave_height: float


@dataclass(frozen=True)
class Scenario:
    identifier: str
    family: str
    split: str
    target: np.ndarray
    bow: float
    radius: float
    first: Forcing
    future: Forcing | None
    change_at: float
    obstacle: tuple[np.ndarray, float] | None = None


def _unit(deg):
    angle = np.radians(deg)
    return np.array([np.sin(angle), np.cos(angle)])


def _make_forcing(sampler, wind_direction, wind_speed, current_direction, current_speed, wave_height):
    wind = Wind(*(-wind_speed * _unit(wind_direction)), wind_speed, wind_direction)
    current = Current(*(current_speed * _unit(current_direction)), current_speed, current_direction, 0.0)
    wave = Wave(wave_height, wind_direction, 5.0)
    observation = WorldObservation(np.datetime64("2000-01-01T00:00:00"), BoatState(0.0, 0.0, 0.0, 0.0),
                                   wind, wave, current, None, None, None, ())
    sampled = tuple(sampler._steady_modes(observation))
    # Keep the fastest two stable trims on each available heading, retaining
    # some slower trim alternatives while bounding the action set.
    selected = []
    for bow in sorted({mode.bow for mode in sampled}):
        choices = [mode for mode in sampled if mode.bow == bow]
        choices.sort(key=lambda mode: -np.linalg.norm(mode.velocity - [current.u, current.v]))
        selected.extend(choices[:2])
    if len(selected) < 4:
        raise ValueError("Fossen equilibrium produced too few alternatives.")
    return Forcing(PolarSnapshot.from_modes(tuple(selected), (current.u, current.v), wind_direction),
                   wind_speed, wave_height)


def generate_scenarios(counts=(96, 32, 64)):
    """Use disjoint RNG substreams and no shared forcing pairs between splits."""
    sampler = DriftGuidance(Fossen3DOFSaildroneDynamics(), 45.0)
    offsets = np.array([45.5, 52.0, 67.0, 85.0, 105.0, 128.0, 152.0, 175.0])
    sampler.offsets = np.r_[-offsets, offsets]
    sampler.flaps = sampler.flaps[[0, 3, 6]]
    for split, total, seed in zip(("train", "validation", "test"), counts,
                                  np.random.SeedSequence(SCENARIO_SEED).spawn(3), strict=True):
        rng = np.random.default_rng(seed)
        for i in range(total):
            family = FAMILIES[i % len(FAMILIES)]
            wind_from = rng.uniform(0.0, 360.0)
            if family == "opposed_weak":
                wind_speed = rng.uniform(2.4, 4.4)
                current_from = wind_from + 180.0 + rng.uniform(-45.0, 45.0)
                current_speed = rng.uniform(.23, .55)
                goal_from = wind_from + rng.uniform(-40.0, 40.0)
                future_wind = rng.uniform(4.9, 8.0)
                transition = float(300 * rng.integers(3, 57 if split == "test" else 51))
            elif family == "cross_current":
                wind_speed = rng.uniform(3.0, 7.0)
                current_from = wind_from + rng.choice((-90.0, 90.0)) + rng.uniform(-38.0, 38.0)
                current_speed = rng.uniform(.24, .70)
                goal_from = wind_from + rng.uniform(-110.0, 110.0)
                future_wind = rng.uniform(3.8, 8.5)
                transition = float(300 * rng.integers(2, 24))
            elif family == "steady_reach":
                wind_speed = rng.uniform(4.5, 9.0)
                current_from = rng.uniform(0.0, 360.0)
                current_speed = rng.uniform(.02, .45)
                goal_from = wind_from + rng.choice((-1.0, 1.0)) * rng.uniform(65.0, 178.0)
                future_wind, transition = wind_speed, 0.0
            else:
                wind_speed = rng.uniform(5.0, 9.0)
                current_from = wind_from + 180.0 + rng.uniform(-70.0, 70.0)
                current_speed = rng.uniform(.19, .63)
                goal_from = wind_from + rng.uniform(-103.0, 103.0)
                future_wind = rng.uniform(2.5, 5.0)
                transition = float(300 * rng.integers(2, 31))

            height = rng.uniform(0.1, 1.65)
            first = _make_forcing(sampler, wind_from, wind_speed, current_from % 360,
                                  current_speed, height)
            future = None
            if transition:
                future = _make_forcing(sampler, (wind_from + rng.uniform(-32.0, 32.0)) % 360,
                                       future_wind, (current_from + rng.uniform(-28.0, 28.0)) % 360,
                                       current_speed * rng.uniform(.64, 1.15), height * rng.uniform(.75, 1.15))
            target = rng.uniform(200.0, 2450.0) * _unit(goal_from)
            heading = rng.uniform(0, 360)
            yield Scenario(f"{split}-{i:03}", family, split, target, heading, 50.0, first, future, transition)


def _serialize_polar(forcing):
    if forcing is None:
        return None
    p = forcing.polar
    return {"velocity": p.velocity.tolist(), "bow": p.bow.tolist(), "flap": p.flap.tolist(),
            "flow": p.flow.tolist(), "wind_from": p.wind_from, "hull": p.hull.tolist(),
            "wind_speed": forcing.wind_speed, "wave_height": forcing.wave_height}


def _deserialize_polar(row):
    if row is None:
        return None
    polar = PolarSnapshot(*(np.asarray(row[key]) for key in ("velocity", "bow", "flap", "flow")),
                          row["wind_from"], np.asarray(row["hull"]))
    return Forcing(polar, row["wind_speed"], row["wave_height"])


def save_scenarios(path: Path, scenarios):
    rows = [{"identifier": x.identifier, "family": x.family, "split": x.split, "target": x.target.tolist(),
             "bow": x.bow, "radius": x.radius, "first": _serialize_polar(x.first),
             "future": _serialize_polar(x.future), "change_at": x.change_at} for x in scenarios]
    path.write_bytes(gzip.compress((json.dumps(rows, sort_keys=True) + "\n").encode(), mtime=0))


def load_scenarios(path: Path):
    with gzip.open(path, "rt") as stream:
        rows = json.load(stream)
    return [Scenario(x["identifier"], x["family"], x["split"], np.asarray(x["target"]),
                     x["bow"], x["radius"], _deserialize_polar(x["first"]),
                     _deserialize_polar(x["future"]), x["change_at"]) for x in rows]


def with_chart_obstacle(scenario: Scenario) -> Scenario:
    """Add a reproducible exclusion disk between the initial position and goal."""
    longitude = scenario.target / np.linalg.norm(scenario.target)
    normal = np.array([longitude[1], -longitude[0]])
    radius = min(95.0, np.linalg.norm(scenario.target) * .17)
    position = .53 * scenario.target + .30 * radius * normal
    return replace(scenario, obstacle=(position, radius))


def active_forcing(scenario: Scenario, elapsed_s: float):
    before_shift = scenario.future is not None and elapsed_s < scenario.change_at
    forcing = scenario.first if before_shift or scenario.future is None else scenario.future
    upcoming = scenario.future.polar if before_shift else None
    until = scenario.change_at - elapsed_s if before_shift else 0.0
    return forcing, upcoming, until


def evaluate(scenario: Scenario, elapsed_s: float, target, bow):
    """Evaluate the finite actions for states on a common experiment clock."""
    target = np.asarray(target, dtype=float).reshape(-1, 2)
    forcing, upcoming, until = active_forcing(scenario, elapsed_s)
    next_window = INTERVAL_SECONDS - elapsed_s % INTERVAL_SECONDS
    dt = min(next_window, MAX_SECONDS - elapsed_s, until if upcoming is not None else MAX_SECONDS)
    if dt <= 0.0:
        raise ValueError("The scenario has already reached its time cap.")
    batch = candidate_batch(target, bow, scenario.radius, forcing.polar, upcoming,
                            forcing.wind_speed, forcing.wave_height, dt, until, MAX_SECONDS - elapsed_s)
    if scenario.obstacle is None:
        return batch, dt, np.ones(batch.captured_after_s.shape, dtype=bool)
    center, radius = scenario.obstacle
    origin = (scenario.target - target)[:, None, :]
    after_turn = origin + batch.turn_displacement
    clear_turn = ~np.isfinite(circle_entry(origin, batch.turn_displacement, center, radius))
    clear_sail = ~np.isfinite(circle_entry(after_turn, batch.displacement - batch.turn_displacement, center, radius))
    return batch, dt, clear_turn & clear_sail


def enforce_first_entry(scores, batch, dt, allowed):
    """Use exact within-window hits and prevent blocked actions from winning."""
    bounded = np.maximum(np.asarray(scores), dt)
    bounded = np.where(np.isfinite(batch.captured_after_s), batch.captured_after_s, bounded)
    return np.where(allowed, bounded, np.inf)


def policy_costs(scenario: Scenario, elapsed_s, target, batch, dt, policy):
    """Use physics costs for out-of-domain states, then apply the external mask."""
    forcing, upcoming, until = active_forcing(scenario, elapsed_s)
    ranges = np.linalg.norm(np.asarray(target).reshape(-1, 2), axis=-1)
    supported = policy.supported(ranges, scenario.radius, forcing.polar, upcoming,
                                 forcing.wind_speed, forcing.wave_height, dt, until)
    costs = batch.physics_cost_s.copy()
    if supported.any():
        proposed = policy.costs(batch.features[supported])
        finite = np.isfinite(proposed).all(axis=1)
        indices = np.flatnonzero(supported)
        supported[indices[~finite]] = False
        costs[indices[finite]] = proposed[finite]
    return costs, supported


def rollout(scenario: Scenario, score: Callable | None) -> dict:
    target, bow, elapsed, interventions, tacks = scenario.target.copy(), scenario.bow, 0.0, 0, 0
    while elapsed < MAX_SECONDS:
        batch, dt, allowed = evaluate(scenario, elapsed, target, bow)
        if not allowed.any():
            return {"status": "blocked", "duration_s": elapsed, "objective_s": MAX_SECONDS + TIMEOUT_COST_SECONDS,
                    "chart_masks": interventions, "tacks": tacks}
        scores = batch.physics_cost_s if score is None else score(batch.features)
        unmasked = enforce_first_entry(scores, batch, dt, np.ones_like(allowed))
        masked = enforce_first_entry(scores, batch, dt, allowed)
        first = int(np.argmin(masked[0]))
        interventions += first != int(np.argmin(unmasked[0]))
        if np.isfinite(batch.captured_after_s[0, first]):
            hit = elapsed + float(batch.captured_after_s[0, first])
            return {"status": "captured", "duration_s": hit, "objective_s": hit,
                    "chart_masks": interventions, "tacks": tacks}
        updated = float(batch.end_bow[0, first])
        tacks += abs((updated - bow + 180.0) % 360.0 - 180.0) >= 30.0
        target, bow, elapsed = batch.remaining_target[0, first], updated, elapsed + dt
    return {"status": "timeout", "duration_s": elapsed, "objective_s": MAX_SECONDS + TIMEOUT_COST_SECONDS,
            "chart_masks": interventions, "tacks": tacks}
