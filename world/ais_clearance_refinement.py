"""Refine a Fossen maneuver against the currently reported AIS motion."""

from __future__ import annotations

import numpy as np


_DISTANCE_SCALE_METRES = 100.0
_SAMPLES_PER_STEP = 4
_COMMAND_TOLERANCE_RADIANS = 1e-5


def refine_clearance(
    step,
    forcing: np.ndarray,
    seed_states: np.ndarray,
    seed_commands: np.ndarray,
    target_positions: np.ndarray,
    target_velocities: np.ndarray,
    clearance_radii: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    slew: np.ndarray,
    dt: float,
    *,
    max_iterations: int = 180,
    cpu_seconds: float | None = 8.0,
    keep_best_iterate: bool = False,
) -> np.ndarray | None:
    """Return a better sequence of rudder and flap commands, in radians.

    The native transition advances one control interval. Seed states and
    commands have shapes (N+1, 9) and (N, 2). The target arrays contain current
    local east/north centres and ground velocities. Clearance radii can be
    one value per contact or a time-varying array with shape (4*N+1, contacts),
    at quarter-control-interval samples. They include any configured reserve.
    A None CPU limit uses a deterministic iteration budget. The caller
    must check the resulting trajectory against chart and swept AIS constraints.
    """
    if step is None or not np.isfinite(dt) or dt <= 0:
        return None
    seed_states = np.asarray(seed_states, dtype=float)
    seed_commands = np.asarray(seed_commands, dtype=float)
    target_positions = np.asarray(target_positions, dtype=float)
    target_velocities = np.asarray(target_velocities, dtype=float)
    clearance_radii = np.asarray(clearance_radii, dtype=float)
    forcing, lower, upper, slew = (np.asarray(value, dtype=float) for value in (forcing, lower, upper, slew))
    count = len(seed_commands) if seed_commands.ndim == 2 else 0
    targets = len(target_positions) if target_positions.ndim == 2 else 0
    sample_count = count * _SAMPLES_PER_STEP + 1
    if (not count or not targets or seed_states.shape != (count + 1, 9)
            or seed_commands.shape != (count, 2) or target_positions.shape != (targets, 2)
            or target_velocities.shape != (targets, 2) or forcing.shape != (5,)
            or clearance_radii.shape not in ((targets,), (sample_count, targets))
            or not isinstance(max_iterations, int) or max_iterations <= 0
            or cpu_seconds is not None and (not np.isfinite(cpu_seconds) or cpu_seconds <= 0)
            or any(value.shape != (2,) for value in (lower, upper, slew))
            or not all(np.isfinite(value).all() for value in (
                seed_states, seed_commands, target_positions, target_velocities,
                clearance_radii, forcing, lower, upper, slew,
            )) or np.any(clearance_radii < 0.0) or np.any(slew < 0.0) or np.any(lower > upper)):
        return None

    try:
        import casadi as ca
    except ImportError:
        return None

    offsets = np.arange(count * _SAMPLES_PER_STEP + 1) / _SAMPLES_PER_STEP
    radii = np.broadcast_to(clearance_radii, (sample_count, targets))
    before = np.minimum(offsets.astype(int), count - 1)
    fraction = offsets - before
    centres = target_positions + (offsets * dt)[:, None, None] * target_velocities
    previous_command = seed_states[0, [7, 8]]

    def minimum_margin(states: np.ndarray) -> float:
        first = states[before, :2]
        points = first + fraction[:, None] * (states[before + 1, :2] - first)
        distances = np.linalg.norm(points[:, None, :] - centres, axis=-1)
        return float(np.min(distances - radii))

    seed_margin = minimum_margin(seed_states)
    best_margin, best_commands = seed_margin, None

    def retain_native_candidate(candidate):
        nonlocal best_margin, best_commands
        candidate = np.asarray(candidate, dtype=float).reshape(count, 2).copy()
        if not np.isfinite(candidate).all():
            return
        native = np.empty_like(seed_states)
        native[0] = seed_states[0]
        previous = seed_states[0, [7, 8]].copy()
        for index, control in enumerate(candidate):
            low, high = np.maximum(lower, previous - slew), np.minimum(upper, previous + slew)
            if np.any(control < low - _COMMAND_TOLERANCE_RADIANS) or np.any(control > high + _COMMAND_TOLERANCE_RADIANS):
                return
            candidate[index] = np.clip(control, low, high)
            previous = candidate[index]
            native[index + 1] = np.asarray(step(native[index], previous, forcing)).ravel()
        if np.isfinite(native).all():
            score = minimum_margin(native)
            if score > best_margin + 1e-3:
                best_margin, best_commands = score, candidate
    try:
        problem = ca.Opti()
        states = problem.variable(9, count + 1)
        commands = problem.variable(2, count)
        margin = problem.variable()
        environment = ca.DM(forcing)
        next_states = step.map(count)(states[:, :-1], commands, ca.repmat(environment, 1, count))
        problem.subject_to(states[:, 0] == seed_states[0])
        problem.subject_to(ca.vec(states[:, 1:] - next_states) == 0)
        problem.subject_to(problem.bounded(
            ca.vec(ca.repmat(lower, 1, count)), ca.vec(commands), ca.vec(ca.repmat(upper, 1, count)),
        ))
        changes = commands - ca.horzcat(previous_command, commands[:, :-1])
        limits = ca.vec(ca.repmat(slew, 1, count))
        problem.subject_to(problem.bounded(-limits, ca.vec(changes), limits))

        first = states[:2, before.tolist()]
        weights = ca.repmat(ca.DM(fraction).T, 2, 1)
        points = first + weights * (states[:2, (before + 1).tolist()] - first)
        for ix in range(targets):
            relative = (points - centres[:, ix].T) / _DISTANCE_SCALE_METRES
            distances = ca.sqrt(ca.sum1(relative**2) + 1e-12)
            problem.subject_to(ca.vec(distances - radii[:, ix][None, :] / _DISTANCE_SCALE_METRES) >= margin)
        regularization = 1e-4 * ca.sumsqr(commands - seed_commands.T) / count
        problem.minimize(-margin + regularization)
        problem.set_initial(states, seed_states.T)
        problem.set_initial(commands, seed_commands.T)
        problem.set_initial(margin, seed_margin / _DISTANCE_SCALE_METRES)
        options = {
            "print_level": 0,
            "sb": "yes",
            "max_iter": max_iterations,
            "hessian_approximation": "limited-memory",
            "acceptable_tol": 1e-4,
        }
        if cpu_seconds is not None:
            options["max_cpu_time"] = cpu_seconds
        problem.solver("ipopt", {"print_time": False}, options)
        if keep_best_iterate:
            def inspect_iteration(_iteration):
                try:
                    values = np.asarray(problem.debug.value(commands)).reshape(2, count).T
                    retain_native_candidate(values)
                except RuntimeError:
                    pass
            problem.callback(inspect_iteration)
        solution = problem.solve_limited()
        candidate = np.asarray(solution.value(commands)).reshape(2, count).T.copy()
    except RuntimeError:
        return best_commands

    if keep_best_iterate:
        retain_native_candidate(candidate)
        return best_commands

    if not np.isfinite(candidate).all():
        return None
    checked = np.empty_like(seed_states)
    checked[0] = seed_states[0]
    try:
        for index, command in enumerate(candidate):
            low = np.maximum(lower, previous_command - slew)
            high = np.minimum(upper, previous_command + slew)
            if (np.any(low > high) or np.any(command < low - _COMMAND_TOLERANCE_RADIANS)
                    or np.any(command > high + _COMMAND_TOLERANCE_RADIANS)):
                return None
            candidate[index] = np.clip(command, low, high)
            previous_command = candidate[index]
            checked[index + 1] = np.asarray(step(checked[index], previous_command, forcing)).ravel()
    except RuntimeError:
        return None
    if np.isfinite(checked).all() and minimum_margin(checked) > seed_margin + 1e-3:
        return candidate
    return None
