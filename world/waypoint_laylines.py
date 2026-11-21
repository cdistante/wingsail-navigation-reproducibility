"""Construct waypoint approaches from feasible sailing velocities over ground."""

from __future__ import annotations

from dataclasses import dataclass
from math import copysign, hypot, sqrt

import numpy as np


@dataclass(frozen=True)
class LaylineCandidate:
    first: int
    second: int | None
    first_distance_m: float
    arrival_s: float
    vertices_en: np.ndarray


def _gate_entry(displacement: np.ndarray, unit: np.ndarray, radius: float) -> np.ndarray:
    """Distance along each ray to the first point in the waypoint disk."""
    along = np.sum(displacement * unit, axis=-1)
    across = np.abs(_cross(displacement, unit))
    half_chord = _half_chord(across, radius)
    tolerance = max(1e-10, radius * 1e-12)
    return np.where((across <= radius + tolerance) & (along + half_chord >= -tolerance),
                    np.maximum(0., along - half_chord), np.inf)


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _half_chord(across: np.ndarray, radius: float) -> np.ndarray:
    return np.sqrt(np.maximum(0., (radius - across) * (radius + across)))


def _limited_first_distance(
    lower: float, upper: float, *,
    across_zero: float, across_slope: float, along_zero: float, along_slope: float,
    first_speed: float, finish_speed: float, turns_s: float, radius: float, horizon: float,
) -> float | None:
    """Find the first feasible turn on one forward, outside-gate interval."""
    if lower > upper:
        return None

    def eta(distance: float) -> float:
        across = min(radius, abs(across_zero - distance * across_slope))
        remaining = along_zero - distance * along_slope - sqrt((radius - across) * (radius + across))
        return turns_s + distance / first_speed + remaining / finish_speed

    if eta(lower) <= horizon:
        return lower
    # ETA is linear in first distance minus the concave half-chord length.
    # Its stationary point bounds the decreasing interval for the search.
    slope = finish_speed / first_speed - along_slope
    if abs(across_slope) <= 1e-12:
        best = upper if slope < 0. else lower
    else:
        across_best = copysign(1., across_slope) * slope / hypot(slope, across_slope) * radius
        best = min(upper, max(lower, (across_zero - across_best) / across_slope))
    if eta(best) > horizon:
        return None
    for _ in range(48):
        middle = (lower + best) / 2.
        if eta(middle) <= horizon:
            best = middle
        else:
            lower = middle
    return best


def layline_candidates(
    target: np.ndarray,
    velocities: np.ndarray,
    bows: np.ndarray,
    wind_from_deg: float,
    boat_heading_deg: float,
    drift: np.ndarray,
    horizon_s: float,
    radius_m: float,
    turn_rate_deg_s: float,
    min_first_s: float = 30.,
) -> list[LaylineCandidate]:
    """Rank direct gate approaches, then nearby feasible opposite-tack laylines.

    Positions and velocities are east/north relative to the boat. Steady mode
    velocities already account for current. During either turn, propagate at
    the current velocity for the estimated maneuver duration. A two-tack
    layline turns at the earliest point whose forward finishing ray enters
    the capture disk. Project the gate entry onto that ray.
    """
    target, velocities, bows, drift = map(np.asarray, (target, velocities, bows, drift))
    if target.shape != (2,) or drift.shape != (2,) or bows.ndim != 1 or velocities.shape != (len(bows), 2):
        raise ValueError("Laylines require an east/north target and drift and one ground velocity for each bow.")
    limits = (wind_from_deg, boat_heading_deg, horizon_s, radius_m, turn_rate_deg_s, min_first_s)
    if not np.all(np.isfinite(np.r_[target, drift, limits])) or min(horizon_s, radius_m, turn_rate_deg_s, min_first_s) <= 0.:
        raise ValueError("Layline bearings must be finite and planning limits finite and positive.")

    speed = np.hypot(*velocities.T)
    usable = np.flatnonzero(np.isfinite(bows) & np.isfinite(speed) & (speed > 1e-8))
    if not len(usable) or np.hypot(*target) <= radius_m:
        return []
    b, velocity, speed = bows[usable] % 360., velocities[usable], speed[usable]
    unit = velocity / speed[:, None]
    origin = np.zeros(2)
    initial_turn = np.abs((b - boat_heading_deg % 360. + 180.) % 360. - 180.) / turn_rate_deg_s
    first_start = initial_turn[:, None] * drift
    direct_distance = _gate_entry(target - first_start, unit, radius_m)
    direct_time = initial_turn + direct_distance / speed
    direct = []
    for i in np.flatnonzero((direct_time > 0.) & (direct_time <= horizon_s)):
        entry = first_start[i] + direct_distance[i] * unit[i]
        direct.append(LaylineCandidate(int(usable[i]), None, float(direct_distance[i]),
                                       float(direct_time[i]), np.stack((origin, first_start[i], entry))))

    # At a first distance q, the finishing ray's cross-track offset is
    # cross(target_after_turns, finish_unit) - q * cross(first_unit, finish_unit).
    # Solve both disk tangents to obtain the full first-distance interval.
    side = np.sign((b - wind_from_deg % 360. + 180.) % 360. - 180.)
    first, second = np.nonzero(side[:, None] * side[None, :] < 0.)
    tack_time = np.abs((b[second] - b[first] + 180.) % 360. - 180.) / turn_rate_deg_s
    tack_drift = tack_time[:, None] * drift
    target_after_turns = target - first_start[first] - tack_drift
    first_unit, finish_unit = unit[first], unit[second]
    cross_at_zero = _cross(target_after_turns, finish_unit)
    cross_by_first = _cross(first_unit, finish_unit)
    nonparallel = np.abs(cross_by_first) > 1e-12
    edge_a = np.divide(cross_at_zero - radius_m, cross_by_first, where=nonparallel,
                       out=np.full(len(first), -np.inf))
    edge_b = np.divide(cross_at_zero + radius_m, cross_by_first, where=nonparallel,
                       out=np.full(len(first), np.inf))
    lower = np.maximum(np.minimum(edge_a, edge_b), speed[first] * min_first_s)
    turn_total = initial_turn[first] + tack_time
    upper = np.minimum(np.maximum(edge_a, edge_b), (horizon_s - turn_total) * speed[first])

    # The disk must be ahead of or abreast of the tack's finishing origin.
    along_at_zero = np.sum(target_after_turns * finish_unit, axis=1)
    along_by_first = np.sum(first_unit * finish_unit, axis=1)
    nonorthogonal = np.abs(along_by_first) > 1e-12
    forward_limit = np.divide(along_at_zero, along_by_first, where=nonorthogonal, out=np.zeros(len(first)))
    lower = np.where(nonorthogonal & (along_by_first < 0.), np.maximum(lower, forward_limit), lower)
    upper = np.where(nonorthogonal & (along_by_first > 0.), np.minimum(upper, forward_limit), upper)

    # The finishing ray must make its own forward entry. If it would start
    # inside the gate, move just beyond the other end of the disk along q.
    tolerance = max(1e-10, radius_m * 1e-12)
    own_across = np.abs(_cross(target_after_turns, first_unit))
    own_along = np.sum(target_after_turns * first_unit, axis=1)
    chord = _half_chord(own_across, radius_m)
    already_in_gate = ((own_across <= radius_m) & (lower >= own_along - chord - tolerance)
                       & (lower <= own_along + chord + tolerance))
    lower = np.where(already_in_gate, own_along + chord + tolerance, lower)
    upper = np.minimum(upper, direct_distance[first] - 2. * tolerance)
    feasible = ((nonparallel | (np.abs(cross_at_zero) <= radius_m + tolerance))
                & (nonorthogonal | (along_at_zero >= -tolerance))
                & (lower <= upper))
    kept = np.flatnonzero(feasible)
    finish_distance = _gate_entry(target_after_turns[kept] - lower[kept, None] * first_unit[kept],
                                 finish_unit[kept], radius_m)
    arrival = turn_total[kept] + lower[kept] / speed[first[kept]] + finish_distance / speed[second[kept]]
    for k in np.flatnonzero((finish_distance > 0.) & np.isfinite(arrival) & (arrival > horizon_s)):
        j = kept[k]
        bounds = [(lower[j], upper[j])]
        if own_across[j] <= radius_m and lower[j] < own_along[j] - chord[j]:
            # The outside portion of the finish-start disk can have two
            # disjoint intervals. Test the earlier one first.
            bounds = [(lower[j], min(upper[j], own_along[j] - chord[j] - tolerance)),
                      (max(lower[j], own_along[j] + chord[j] + tolerance), upper[j])]
        for lo, hi in bounds:
            revised = _limited_first_distance(
                lo, hi, across_zero=cross_at_zero[j], across_slope=cross_by_first[j],
                along_zero=along_at_zero[j], along_slope=along_by_first[j],
                first_speed=speed[first[j]], finish_speed=speed[second[j]], turns_s=turn_total[j],
                radius=radius_m, horizon=horizon_s,
            )
            if revised is not None:
                lower[j] = revised
                remaining = target_after_turns[j] - revised * first_unit[j]
                finish_distance[k] = _gate_entry(remaining, finish_unit[j], radius_m)
                arrival[k] = turn_total[j] + revised / speed[first[j]] + finish_distance[k] / speed[second[j]]
                break

    tack = first_start[first[kept]] + lower[kept, None] * first_unit[kept]
    finish_start = tack + tack_drift[kept]
    pairs = []
    for k in np.flatnonzero((finish_distance > 0.) & (arrival <= horizon_s + 1e-8)):
        index = kept[k]
        i, j = first[index], second[index]
        entry = finish_start[k] + finish_distance[k] * unit[j]
        pairs.append(LaylineCandidate(
            int(usable[i]), int(usable[j]), float(lower[index]), float(arrival[k]),
            np.stack((origin, first_start[i], tack[k], finish_start[k], entry)),
        ))
    return sorted(direct, key=lambda item: item.arrival_s) + sorted(
        pairs, key=lambda item: (item.first_distance_m, item.arrival_s),
    )
