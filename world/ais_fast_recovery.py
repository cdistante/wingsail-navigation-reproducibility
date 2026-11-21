"""Generic, low-dimensional startup controls evaluated with the native plant.

No positions, times, vessel identities or recorded controls are embedded in
this search. Received AIS and the current native state determine its score.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np

from .ais_avoidance import _closest_fraction


@dataclass(frozen=True)
class FeedbackFamily:
    pulse_heading: float
    heading: float
    flap: float
    gain: float
    yaw_damping: float
    pulse_seconds: float
    landing_fraction: float | None = None


@dataclass(frozen=True)
class FastRecoverySeed:
    family: FeedbackFamily
    commands: np.ndarray
    states: np.ndarray
    margin: float


def _rollouts(step, initial, forcing, families, lower, upper, slew, dt, count):
    """Batch independent control families, preserving native actuator slews."""
    number = len(families)
    mapped = step.map(number)
    nodes = np.broadcast_to(initial[:, None], (9, number)).copy()
    environment = np.broadcast_to(forcing[:, None], (5, number)).copy()
    states = [nodes.T.copy()]
    commands = []
    pulse = np.array([x.pulse_heading for x in families])
    heading = np.array([x.heading for x in families])
    duration = np.array([x.pulse_seconds for x in families])
    flap = np.array([x.flap for x in families])
    gain = np.array([x.gain for x in families])
    damping = np.array([x.yaw_damping for x in families])
    landing = np.array([np.nan if x.landing_fraction is None else x.landing_fraction for x in families])
    initial_heading = np.degrees(initial[2])
    pulse_direction = np.sign((initial_heading - pulse + 180.) % 360. - 180.)
    for tick in range(count):
        seconds = tick * dt
        desired = np.where(seconds < duration, pulse, heading)
        error = (np.degrees(nodes[2]) - desired + 180.) % 360. - 180.
        rudder = gain * error + damping * np.degrees(nodes[5])
        settle = np.isfinite(landing) & (np.abs(seconds - duration) < 1e-9)
        rudder = np.where(settle, pulse_direction * upper[0] * landing, rudder)
        previous = np.degrees(nodes[[7, 8]].T)
        control = np.clip(np.column_stack((rudder, flap)),
                          np.maximum(lower, previous - slew), np.minimum(upper, previous + slew))
        nodes = np.asarray(mapped(nodes, np.radians(control).T, environment), dtype=float)
        states.append(nodes.T.copy())
        commands.append(np.radians(control))
    return np.asarray(states), np.asarray(commands)


def _margins(traffic, states, dt):
    times = np.arange(len(states)) * dt
    relative = states[:, :, None, :2] - traffic.centres_at(times)[:, None, :, :]
    delta = np.diff(relative, axis=0)
    radii = traffic.radii_at(times)
    growth = np.diff(radii, axis=0)
    fraction = _closest_fraction(relative[:-1], delta, growth[:, None, :])
    values = (np.linalg.norm(relative[:-1] + fraction[..., None] * delta, axis=-1)
              - radii[:-1, None, :] - fraction * growth[:, None, :])
    return values.min(axis=(0, 2))


def search_feedback(step, initial, forcing, traffic, lower, upper, slew, dt,
                    duration=300., *, local_evaluations=100):
    """Return a fresh native seed and diagnostics; all work occurs in this call."""
    started = perf_counter()
    from scipy.optimize import minimize

    initial = np.asarray(initial, dtype=float)
    count = round(duration / dt)
    bow = float(np.degrees(initial[2]))
    families = []
    for offset in (-20., -10., 0., 10., 20.):
        for flap in (float(lower[1]), float(upper[1])):
            for gain in (.5, .8, 1.2):
                for damping in (2.5, 5.):
                    families.append(FeedbackFamily(bow, bow + offset, flap, gain, damping, 0.))
                    for side in (-1., 1.):
                        for seconds in (10., 15., 20., 25.):
                            families.append(FeedbackFamily(bow + side * 90., bow + offset, flap,
                                                           gain, damping, seconds))
    states, controls = _rollouts(step, initial, forcing, families, lower, upper, slew, dt, count)
    margins = _margins(traffic, states, dt)
    order = np.argsort(-margins, kind="stable")
    ix = int(order[0])
    best = FastRecoverySeed(families[ix], controls[:, ix].copy(), states[:, ix].copy(), float(margins[ix]))
    batch_finished = perf_counter()
    evaluations = 0

    # Optimize four interpretable parameters around a seed found from these
    # inputs. No scenario-specific heading or control sequence is reused.
    for ix in order[:2]:
        ix = int(ix)
        seed = families[ix]
        landing_index = min(round(seed.pulse_seconds / dt), count - 1)
        side = np.sign((bow - seed.pulse_heading + 180.) % 360. - 180.)
        landing = 0. if side == 0 else float(controls[landing_index, ix, 0] / np.radians(upper[0]) / side)
        x0 = np.array((seed.heading, seed.gain, seed.yaw_damping, landing))

        def objective(values):
            nonlocal best, evaluations
            candidate = replace(seed, heading=float(values[0]), gain=float(values[1]),
                                yaw_damping=float(values[2]), landing_fraction=float(values[3]))
            result, actions = _rollouts(step, initial, forcing, [candidate], lower, upper, slew, dt, count)
            score = float(_margins(traffic, result, dt)[0])
            evaluations += 1
            if np.isfinite(score) and score > best.margin:
                best = FastRecoverySeed(candidate, actions[:, 0].copy(), result[:, 0].copy(), score)
            return -score

        simplex = np.vstack((x0, x0 + (5., 0., 0., 0.), x0 + (0., .15, 0., 0.),
                              x0 + (0., 0., .5, 0.), x0 + (0., 0., 0., .1)))
        minimize(objective, x0, method="Nelder-Mead",
                 bounds=((bow - 60., bow + 60.), (.1, 3.), (0., 20.), (-1., 1.)),
                 options={"maxfev": local_evaluations, "xatol": .005, "fatol": 1e-4,
                          "initial_simplex": simplex})
    return best, {"feedback_candidates": len(families), "feedback_local_evaluations": evaluations,
                  "feedback_batch_seconds": batch_finished - started,
                  "feedback_local_seconds": perf_counter() - batch_finished,
                  "feedback_total_seconds": perf_counter() - started,
                  "feedback_preview_margin_m": best.margin}
