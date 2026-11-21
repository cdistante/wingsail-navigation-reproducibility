"""Screen native controls against a position-robust full AIS envelope."""

from __future__ import annotations

import numpy as np

from native_branches import NativeEngine
from world._geo import local_east_north_metres
from world.ais_avoidance import DEFAULT_AIS_SAFETY_BUFFER_METRES


def guarded_clearance(prediction, own_east_north, times):
    """Lower bound the full clearance for any center inside the report error disk."""
    if not prediction.targets:
        return float("inf")

    full_radius = prediction.radii_at(times)
    fixed_radius = (prediction.hull_radii_metres + DEFAULT_AIS_SAFETY_BUFFER_METRES
                    + prediction.clearance_reserve_metres)
    position_uncertainty = full_radius - fixed_radius
    guard_radius = full_radius + position_uncertainty

    relative = own_east_north[:, None, :] - prediction.centres_at(times)
    travel = np.diff(relative, axis=0)
    distance_squared = np.sum(travel * travel, axis=-1)
    fractions = np.divide(-np.sum(relative[:-1] * travel, axis=-1), distance_squared,
                          out=np.zeros_like(distance_squared), where=distance_squared > 0.0)
    closest = relative[:-1] + np.clip(fractions, 0.0, 1.0)[..., None] * travel
    nearest = np.linalg.norm(closest, axis=-1)
    # The larger endpoint radius also covers every point between these knots.
    max_radius = np.maximum(guard_radius[:-1], guard_radius[1:])
    return float((nearest - max_radius).min())


class RobustNativeEngine(NativeEngine):
    """Use the robust margin for candidate and live control admission."""

    def _screen(self, state, choice, observation, dt):
        clearance, permitted, final, path, local, prediction, action = super()._screen(
            state, choice, observation, dt
        )
        if permitted and clearance > 0.0 and prediction.targets:
            east, north = local_east_north_metres(path[:, 0], path[:, 1], state.latitude, state.longitude)
            times = np.arange(len(path)) * self.config.integration_dt_s
            clearance = guarded_clearance(prediction, np.column_stack((east, north)), times)
        # The base prediction remains the full-envelope reference for comparable telemetry.
        return clearance, permitted, final, path, local, prediction, action
