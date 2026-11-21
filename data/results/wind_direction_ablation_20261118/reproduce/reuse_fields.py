"""Reuse identical harmonic field solves within an experiment process."""
from __future__ import annotations

from collections import Counter, OrderedDict
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np

from world.legacy_planners import harmonic_gamma
from world.legacy_planners.harmonic_diffusion import WindAwareHarmonicPlanner


GammaHarmonicPlanner = harmonic_gamma.GammaHarmonicPlanner
GAMMA_SOLVE = GammaHarmonicPlanner.solve_potential
DIFFUSION_SOLVE = WindAwareHarmonicPlanner.solve_potential


def input_key(value):
    """Retain exact array data and scalar values, including sign and dtype."""
    if isinstance(value, np.ndarray):
        return value.shape, value.dtype.str, value.tobytes()
    if isinstance(value, np.generic):
        return value.dtype.str, value.tobytes()
    if isinstance(value, float):
        return value.hex()
    if isinstance(value, (tuple, list)):
        return tuple(map(input_key, value))
    return value


def fields(cache, maxsize, counts, name, key, solver, outputs, solve):
    if key in cache:
        cache.move_to_end(key)
        for attribute, value in zip(outputs, cache[key], strict=True):
            setattr(solver, attribute, value.copy())
        counts[f"{name}_hits"] += 1
        return solver.potential

    result = solve()
    cache[key] = tuple(getattr(solver, name).copy() for name in outputs)
    if len(cache) > maxsize:
        cache.popitem(last=False)
    counts[f"{name}_misses"] += 1
    return result


@contextmanager
def install(maxsize: int = 128):
    """Memoize exact inputs and restore independent copies of output fields."""
    if maxsize < 1:
        raise ValueError("Harmonic caches require at least one entry.")
    gamma_cache = OrderedDict()
    diffusion_cache = OrderedDict()
    counts = Counter()

    def gamma(solver, obs_xy, goal_x, goal_y, boat_x, boat_y, dist_to_obst_xy=None):
        def solve():
            return GAMMA_SOLVE(solver, obs_xy, goal_x, goal_y, boat_x, boat_y, dist_to_obst_xy)

        if type(solver) is not GammaHarmonicPlanner:
            return solve()
        key = input_key((
            solver.nx, solver.ny, solver.width, solver.height, solver.res, solver.wind_dir,
            solver.no_go_angle, solver.gamma_min, solver.gamma_max, solver.transition_width,
            solver.iterations, solver.relax, harmonic_gamma._DEVICE, harmonic_gamma._ZONE_DANGER,
            harmonic_gamma._ZONE_CAUTION, harmonic_gamma._RAISE_DANGER, harmonic_gamma._RAISE_CAUTION,
            obs_xy, goal_x, goal_y, boat_x, boat_y, dist_to_obst_xy,
        ))
        solver._boat_x, solver._boat_y = boat_x, boat_y
        return fields(gamma_cache, maxsize, counts, "gamma", key, solver,
                      ("potential", "gamma_field"), solve)

    def diffusion(solver, iterations=1000, relax=1.0, dist_to_obst=None):
        def solve():
            return DIFFUSION_SOLVE(solver, iterations, relax, dist_to_obst)

        if type(solver) is not WindAwareHarmonicPlanner:
            return solve()
        key = input_key((
            solver.grid.grid_size, solver.grid.occupancy_grid, solver.goal_world,
            solver.wind_direction, solver.no_go_angle, solver.boundary, iterations, relax, dist_to_obst,
        ))
        return fields(diffusion_cache, maxsize, counts, "diffusion", key, solver,
                      ("potential", "diffusion_field"), solve)

    with (patch.object(GammaHarmonicPlanner, "solve_potential", gamma),
          patch.object(WindAwareHarmonicPlanner, "solve_potential", diffusion)):
        yield counts
