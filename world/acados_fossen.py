"""CasADi transcription of the simulator's semi-implicit Fossen step."""

from __future__ import annotations

import math

from .fossen import Fossen3DOFParameters, Fossen3DOFSaildroneDynamics


def discrete_fossen(
    dynamics: Fossen3DOFSaildroneDynamics,
    state,
    action,
    parameters,
    dt_seconds: float,
):
    """Return the next local EN state, with angles in radians.

    State order is east, north, compass heading, surge, sway, yaw rate, actual
    flap, previous rudder command, previous flap command. Action order is
    rudder, flap command. The first five parameters are the East/North wind
    and current components, followed by wave height. Heading stays unwrapped.
    """
    import casadi as ca

    if not math.isfinite(dt_seconds) or dt_seconds <= 0.0:
        raise ValueError("dt_seconds must be finite and positive.")

    coefficients = dynamics.parameters
    rudder = _clip(
        ca, action[0], math.radians(coefficients.rudder_min), math.radians(coefficients.rudder_max),
    )
    flap = _clip(
        ca, action[1], math.radians(coefficients.flap_min), math.radians(coefficients.flap_max),
    )
    inputs = ca.vertcat(rudder, flap, parameters[:5])
    substeps = math.ceil(dt_seconds / coefficients.max_integration_step)

    # Keep the fixed action and forcing in the accumulator across substeps.
    # CasADi's fold reuses the substep graph in generated dynamics code.
    accumulator = ca.SX.sym("fossen_accumulator", 14)
    next_state = _substep(ca, accumulator[:7], accumulator[7:], coefficients, dt_seconds / substeps)
    substep = ca.Function(
        "fossen_substep", [accumulator], [ca.vertcat(next_state, accumulator[7:])],
    )
    transition = substep.fold(substeps)
    integrated, = transition.call([ca.vertcat(state[:7], inputs)], False, True)
    return ca.vertcat(integrated[:7], rudder, flap)


def _clip(ca, value, lower, upper):
    return ca.fmin(ca.fmax(value, lower), upper)


def _body_to_earth(ca, heading):
    sine, cosine = ca.sin(heading), ca.cos(heading)
    return ca.vertcat(ca.horzcat(sine, cosine), ca.horzcat(cosine, -sine))


def _wing_load(ca, apparent_air, body_to_earth, flap, coefficients):
    lift = _clip(
        ca,
        coefficients.lift_slope_per_radian * coefficients.flap_effectiveness * flap,
        -coefficients.max_lift_coefficient,
        coefficients.max_lift_coefficient,
    )
    drag = coefficients.profile_drag_coefficient + coefficients.induced_drag_factor * lift**2
    speed_squared = ca.sumsqr(apparent_air)
    speed = ca.sqrt(ca.fmax(speed_squared, 1e-12))
    apparent_left = ca.vertcat(-apparent_air[1], apparent_air[0])
    pressure_area_over_speed = 0.5 * coefficients.air_density * coefficients.wing_sail_area * speed
    force_earth = pressure_area_over_speed * (lift * apparent_left + drag * apparent_air)
    force_body = ca.if_else(speed_squared < 1e-12, 0, body_to_earth.T @ force_earth)
    return ca.vertcat(force_body, coefficients.wing_sail_longitudinal_position * force_body[1])


def _rudder_load(ca, surge, sway, yaw_rate, rudder, coefficients):
    local_sway = sway + coefficients.rudder_longitudinal_position * yaw_rate
    speed_squared = surge**2 + local_sway**2
    inflow = ca.atan2(local_sway, ca.fmax(ca.fabs(surge), 0.05))
    lift = _clip(
        ca,
        coefficients.rudder_lift_slope * (rudder - inflow),
        -coefficients.rudder_max_lift_coefficient,
        coefficients.rudder_max_lift_coefficient,
    )
    drag = (
        coefficients.rudder_profile_drag_coefficient + coefficients.rudder_induced_drag_factor * lift**2
    )
    pressure_area = 0.5 * coefficients.water_density * coefficients.rudder_area * speed_squared
    return ca.if_else(
        ca.sqrt(speed_squared) < 0.05,
        0,
        pressure_area * ca.vertcat(-drag, lift, coefficients.rudder_longitudinal_position * lift),
    )


def _surge_damping(ca, surge, coefficients):
    speed = ca.fabs(surge)
    table = coefficients.surge_resistance_table
    if not table:
        return coefficients.linear_surge_damping + coefficients.quadratic_surge_damping * speed

    first_speed, first_resistance = table[0]
    last_speed, last_resistance = table[-1]
    resistance = last_resistance * (speed / last_speed)**2
    for (low, low_resistance), (high, high_resistance) in reversed(tuple(zip(table, table[1:]))):
        slope = (high_resistance - low_resistance) / (high - low)
        resistance = ca.if_else(speed <= high, low_resistance + slope * (speed - low), resistance)
    return ca.if_else(
        speed <= first_speed,
        first_resistance / first_speed,
        resistance / ca.fmax(speed, first_speed),
    )


def _substep(ca, state, inputs, coefficients: Fossen3DOFParameters, dt_seconds: float):
    position, heading, velocity, flap = state[:2], state[2], state[3:6], state[6]
    surge, sway, yaw_rate = velocity[0], velocity[1], velocity[2]
    rudder, target_flap, wind, current, wave_height = (
        inputs[0], inputs[1], inputs[2:4], inputs[4:6], inputs[6]
    )
    rotation = _body_to_earth(ca, heading)
    apparent_air = wind - (rotation @ velocity[:2] + current)
    loads = (
        _wing_load(ca, apparent_air, rotation, flap, coefficients)
        + _rudder_load(ca, surge, sway, yaw_rate, rudder, coefficients)
    )

    surge_mass = coefficients.mass + coefficients.added_mass_surge
    sway_mass = coefficients.mass + coefficients.added_mass_sway
    yaw_inertia = coefficients.yaw_inertia + coefficients.added_yaw_inertia
    coriolis = ca.vertcat(
        -sway_mass * sway * yaw_rate,
        surge_mass * surge * yaw_rate,
        (sway_mass - surge_mass) * surge * sway,
    )
    damping = (1.0 + coefficients.wave_damping_per_metre * wave_height) * ca.vertcat(
        _surge_damping(ca, surge, coefficients),
        coefficients.linear_sway_damping + coefficients.quadratic_sway_damping * ca.fabs(sway),
        coefficients.linear_yaw_damping + coefficients.quadratic_yaw_damping * ca.fabs(yaw_rate),
    )
    acceleration = (loads - coriolis - damping * velocity) / ca.DM([surge_mass, sway_mass, yaw_inertia])
    next_velocity = velocity + acceleration * dt_seconds
    next_heading = heading + next_velocity[2] * dt_seconds
    ground_velocity = _body_to_earth(ca, next_heading) @ next_velocity[:2] + current
    max_flap_delta = math.radians(coefficients.flap_rate) * dt_seconds
    next_flap = flap + _clip(ca, target_flap - flap, -max_flap_delta, max_flap_delta)
    return ca.vertcat(position + ground_velocity * dt_seconds, next_heading, next_velocity, next_flap)
