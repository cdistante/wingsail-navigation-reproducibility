"""Receding-horizon escape search with a separate, uncapped applied-step check."""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from .ais_avoidance import AISPrediction, DEFAULT_AIS_SAFETY_BUFFER_METRES, _closest_fraction
from .ais_guidance import _blocked_segments, assess_action
from .boat import BoatAction
from .ais_clearance_refinement import refine_clearance

_HORIZON_SECONDS = 240.0


@dataclass(frozen=True)
class EscapeProposal:
    action: BoatAction
    continuation: np.ndarray
    preview_margin: float
    nominal_margin: float


@lru_cache(maxsize=4)
def _perturbations(count: int) -> np.ndarray:
    """Deterministic correlated exploration for multi-step actuator sequences."""
    rng = np.random.default_rng(721)
    knots = np.linspace(0, count - 1, min(12, count))
    values = rng.standard_normal((3, 96, 2, len(knots)))
    return np.asarray([[[np.interp(np.arange(count), knots, row) for row in pair]
                        for pair in generation] for generation in values]).transpose(0, 1, 3, 2)


def _sample_paths(step, initial, forcing, lower, upper, slew, desired, gains=None):
    """Integrate fitted or feedback-generated commands, with observed actuator slew."""
    count, candidates = desired.shape[:2]
    mapped = step.map(candidates)
    nodes = np.broadcast_to(initial[:, None], (9, candidates)).copy()
    environment = np.broadcast_to(forcing[:, None], (5, candidates)).copy()
    states = [nodes.T.copy()]
    commands = []
    for demand in desired:
        if gains is not None:
            error = (np.degrees(nodes[2]) - demand[:, 0] + 180.0) % 360.0 - 180.0
            rudder = gains * error + 3.0 * np.degrees(nodes[5])
            demand = np.column_stack((rudder, demand[:, 1]))
        previous = np.degrees(nodes[[7, 6]].T)
        control = np.clip(demand, np.maximum(lower, previous - slew), np.minimum(upper, previous + slew))
        nodes = np.asarray(mapped(nodes, np.radians(control).T, environment), dtype=float)
        states.append(nodes.T.copy())
        commands.append(control)
    return np.asarray(states), np.asarray(commands)


def _ranking_radii(traffic, times, dt):
    full = traffic.radii_at(times)
    weight = 0.80 + 0.20 * np.exp(-np.maximum(0.0, times - dt) / 40.0)
    return full[0] + weight[:, None] * (full - full[0])


def _score_paths(positions, traffic, dt):
    times = np.arange(len(positions)) * dt
    relative = positions[:, :, None, :] - traffic.centres_at(times)[:, None, :, :]
    delta = np.diff(relative, axis=0)
    full = traffic.radii_at(times)
    # Report renewal is possible during the lookahead. A discounted growth
    # term ranks uncertified tails, without reducing the applied-step buffer.
    ranking_radii = _ranking_radii(traffic, times, dt)

    def swept(radii):
        growth = np.diff(radii, axis=0)[:, None, :]
        fraction = _closest_fraction(relative[:-1], delta, growth)
        return (np.linalg.norm(relative[:-1] + fraction[..., None] * delta, axis=-1)
                - radii[:-1, None, :] - fraction * growth)

    margins = swept(full)
    fallback = swept(ranking_radii)
    worst_full = margins.min(axis=(0, 2))
    # A soft minimum considers several near-active encounter constraints.
    adjusted = fallback + (0.035 * times[:-1])[:, None, None]
    floor = adjusted.min(axis=(0, 2))
    soft = floor - 5.0 * np.log(np.exp((floor[None, :, None] - adjusted) / 5.0).sum(axis=(0, 2)))
    score = np.where(worst_full > 0.0, 1000.0 + worst_full, soft + 0.20 * worst_full)
    return np.where(margins[0].min(axis=1) > 0.0, score, -np.inf), worst_full


def search_escape(step, dynamics, initial, forcing, observation, grid, blocked,
                  traffic: AISPrediction, lower, upper, slew, dt, warm, prefer_starboard):
    """Rank finite trajectories, then independently check only the applied command.

    The previous predicted tail seeds the next search and is never applied
    without a new decision. No forecast report schedule or truth is used.
    """
    count = max(1, round(_HORIZON_SECONDS / dt))
    heading = observation.boat.heading
    wind = observation.wind.direction
    flap_limits = (lower[1], 0.0, upper[1])
    # Include both powered and depowered trims, and turn-then-settle modes.
    families = [(course, course, trim, trim, 0.0) for course in np.arange(0.0, 360.0, 30.0)
                for trim in flap_limits]
    families.extend((wind + first, wind + last, trim, later, switch)
                    for first in (-120., -60., 60., 120.) for last in (-150., -90., -60., 60., 90., 150., 180.)
                    for trim, later in ((lower[1], lower[1]), (upper[1], upper[1]),
                                        (lower[1], upper[1]), (upper[1], lower[1]))
                    for switch in (45., 85., 125.))
    families.extend((heading + turn, wind + offset, trim, later, pulse)
                    for turn in (-90.0, 90.0) for offset in (-120.0, -60.0, 60.0, 120.0, 180.0)
                    for trim, later in ((lower[1], lower[1]), (upper[1], upper[1]),
                                        (lower[1], upper[1]), (upper[1], lower[1]))
                    for pulse in (15.0, 35.0, 55.0))
    if prefer_starboard:
        families.sort(key=lambda row: ((row[0] - heading + 180.0) % 360.0 - 180.0) < 0.0)
    family = np.asarray(families)
    pulse = (np.arange(count) * dt)[:, None] < family[:, 4]
    demands = np.stack((np.where(pulse, family[:, 0], family[:, 1]),
                        np.where(pulse, family[:, 2], family[:, 3])), axis=2)
    paths, commands = _sample_paths(step, initial, forcing, lower, upper, slew, demands, .8)
    candidates = []

    def retain(states, controls):
        scores, margins = _score_paths(states[:, :, :2], traffic, dt)
        ranked = np.argsort(-scores, kind="stable")
        for index in ranked[:16]:
            if np.isfinite(scores[index]):
                candidates.append((float(scores[index]), float(margins[index]),
                                   states[:, index].copy(), controls[:, index].copy()))
        candidates.sort(key=lambda item: item[0], reverse=True)

    retain(paths, commands)
    if warm is not None and warm.shape == (count, 2):
        shifted = np.vstack((warm[1:], warm[-1:]))[:, None, :]
        retain(*_sample_paths(step, initial, forcing, lower, upper, slew, shifted))
    for noise, scale in zip(_perturbations(count), ((18., 10.), (11., 7.), (6., 4.))):
        if not candidates:
            break
        seeds = np.stack([candidates[i][3] for i in range(min(4, len(candidates)))])
        trial = seeds[np.arange(len(noise)) % len(seeds)] + noise * np.asarray(scale)
        retain(*_sample_paths(step, initial, forcing, lower, upper, slew, trial.transpose(1, 0, 2)))

    if candidates:
        _, margin, seed_states, seed_commands = candidates[0]
        times = np.arange(count * 4 + 1) * dt / 4
        radii = traffic.radii_at(times) if margin > 0.0 else _ranking_radii(traffic, times, dt)
        polished = refine_clearance(
            step, forcing, seed_states, np.radians(seed_commands),
            traffic.positions_metres, traffic.velocities_mps, radii,
            np.radians(lower), np.radians(upper), np.radians(slew), dt,
            max_iterations=12, cpu_seconds=None, keep_best_iterate=True,
        )
        if polished is not None:
            retain(*_sample_paths(step, initial, forcing, lower, upper, slew, np.degrees(polished)[:, None, :]))

    uncapped = replace(traffic, preview_growth_seconds=None)
    checked = set()
    for _, margin, states, commands in candidates:
        if _blocked_segments(grid, blocked, states[:, :2]):
            continue
        action = BoatAction(*map(float, commands[0]))
        if action in checked:
            continue
        checked.add(action)
        chart_clear, assessment = assess_action(dynamics, observation.boat, observation, action,
                                                dt, grid, blocked, uncapped)
        if chart_clear and assessment.safe:
            return EscapeProposal(action, commands, margin,
                                  assessment.minimum_hull_clearance_metres - DEFAULT_AIS_SAFETY_BUFFER_METRES)
    return None
