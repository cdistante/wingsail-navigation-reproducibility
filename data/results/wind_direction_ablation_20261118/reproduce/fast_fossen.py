"""Use the existing CasADi Fossen transcription for grid-MPC predictions."""

from __future__ import annotations

from functools import lru_cache
from math import ceil, pi, radians
from unittest.mock import patch

import casadi as ca

from world._geo import METRES_PER_DEGREE_LATITUDE
from world.acados_fossen import _body_to_earth, _clip, _rudder_load, _substep, _wing_load
from world.boat import BoatState
from world.fossen import Fossen3DOFSaildroneDynamics
from world.mpc import SaildroneMPC


@lru_cache(maxsize=16)
def transition(parameters, dt):
    """Retain the scalar plant's geographic integration and heel diagnostic."""
    count = ceil(dt / parameters.max_integration_step)
    pose = ca.SX.sym("pose", 15)
    heading, velocity, flap = pose[2], pose[3:6], pose[6]
    action = pose[7:9]
    wind, current = pose[9:11], pose[11:13]
    result = _substep(ca, ca.vertcat(0.0, 0.0, heading, velocity, flap), pose[7:14], parameters, dt / count)
    latitude = pose[0] + result[1] / METRES_PER_DEGREE_LATITUDE
    longitude = pose[1] + result[0] / (METRES_PER_DEGREE_LATITUDE * ca.cos(pose[0] * pi / 180.0))
    longitude += 180.0
    longitude = longitude - 360.0 * ca.floor(longitude / 360.0) - 180.0
    bow = result[2] - (2.0 * pi) * ca.floor(result[2] / (2.0 * pi))
    rotation = _body_to_earth(ca, heading)
    apparent = wind - (rotation @ velocity[:2] + current)
    load = _wing_load(ca, apparent, rotation, flap, parameters) + _rudder_load(
        ca, velocity[0], velocity[1], velocity[2], action[0], parameters,
    )
    heel = _clip(
        ca, load[1] * parameters.centre_of_effort_height / parameters.righting_moment_per_degree,
        -parameters.max_heel, parameters.max_heel,
    )
    substep = ca.Function(
        "prediction_substep", [pose], [ca.vertcat(latitude, longitude, bow, result[3:7], pose[7:14], heel)],
        {"cse": True},
    )
    final = substep.fold(count)(pose)
    ground = _body_to_earth(ca, final[2]) @ final[3:5] + current
    return ca.Function(
        "prediction", [pose],
        [ca.vertcat(final[:2], final[2] * 180.0 / pi, ca.norm_2(ground),
                    final[6] * 180.0 / pi, final[14], final[3:6])],
    )


class FastFossenPrediction:
    """Accelerate the prediction step while retaining the original applied-action plant."""

    def __init__(self, dynamics):
        self.dynamics = dynamics
        self.parameters = dynamics.parameters

    def step(self, state, action, environment, dt):
        if dt <= 0.0 or environment.wind is None:
            return self.dynamics.step(state, action, environment, dt)
        p = self.parameters
        rudder = min(max(action.rudder, p.rudder_min), p.rudder_max)
        flap = min(max(action.flap, p.flap_min), p.flap_max)
        surge = state.speed if state.surge_speed is None else state.surge_speed
        sway = 0.0 if state.surge_speed is None else state.sway_speed
        current, wind, wave = environment.current, environment.wind, environment.wave
        values = transition(p, dt)([
            state.latitude, state.longitude, radians(state.heading), surge, sway, state.yaw_rate,
            radians(state.flap), radians(rudder), radians(flap), wind.u, wind.v,
            0.0 if current is None else current.u, 0.0 if current is None else current.v,
            0.0 if wave is None else wave.height, 0.0,
        ]).nonzeros()
        latitude, longitude, heading, speed, actual_flap, heel, surge, sway, yaw = values
        return BoatState(latitude, longitude, heading, speed, rudder, actual_flap, heel, surge, sway, yaw)


@lru_cache(maxsize=8)
def _fast_dynamics(dynamics):
    return FastFossenPrediction(dynamics)


def install():
    """Temporarily accelerate both grid-MPC choice and action verification."""
    rollout_cost = SaildroneMPC._rollout_cost

    def fast_rollout(controller, dynamics, *args, **kwargs):
        if type(dynamics) is Fossen3DOFSaildroneDynamics:
            dynamics = _fast_dynamics(dynamics)
        return rollout_cost(controller, dynamics, *args, **kwargs)

    return patch.object(SaildroneMPC, "_rollout_cost", fast_rollout)
