"""Low-order waypoint guidance for interactive route demonstrations.

The policy in this module is deliberately small and deterministic.  It is a
useful bridge between an OpenCPN route and the later JEPA/MPC controller: it
selects a rudder demand towards the active waypoint but does not replace a
route-aware, constraint-aware MPC.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, degrees, radians, sin

import numpy as np

from .boat import BoatAction, BoatState
from .scenario import RouteWaypoint


def initial_bearing_degrees(
    latitude_a: float,
    longitude_a: float,
    latitude_b: float,
    longitude_b: float,
) -> float:
    """Return the initial true bearing from point A to point B, in degrees."""

    latitude_a_radians = radians(latitude_a)
    latitude_b_radians = radians(latitude_b)
    delta_longitude = radians(longitude_b - longitude_a)
    east = sin(delta_longitude) * cos(latitude_b_radians)
    north = (
        cos(latitude_a_radians) * sin(latitude_b_radians)
        - sin(latitude_a_radians) * cos(latitude_b_radians) * cos(delta_longitude)
    )
    return degrees(atan2(east, north)) % 360.0


def signed_heading_error(desired_heading: float, current_heading: float) -> float:
    """Return the shortest signed heading error in ``[-180, 180)`` degrees."""

    return (desired_heading - current_heading + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class WaypointGuidance:
    """Proportional course-to-waypoint controller for the two-control model.

    Positive rudder is a port command in :mod:`world.fossen`; consequently a
    waypoint to starboard (a positive course error) requires a negative rudder
    demand.  The yaw-rate term damps the turn rather than allowing the simple
    proportional controller to keep commanding rudder after the vessel starts
    rotating.
    """

    flap_degrees: float = -10.0
    rudder_gain_per_degree: float = 0.25
    yaw_damping_per_degree_per_second: float = 1.0
    rudder_limit_degrees: float = 30.0

    def __post_init__(self) -> None:
        if self.rudder_gain_per_degree < 0.0:
            raise ValueError("rudder_gain_per_degree must be non-negative.")
        if self.yaw_damping_per_degree_per_second < 0.0:
            raise ValueError("yaw_damping_per_degree_per_second must be non-negative.")
        if self.rudder_limit_degrees <= 0.0:
            raise ValueError("rudder_limit_degrees must be positive.")

    @staticmethod
    def desired_course(state: BoatState, goal: RouteWaypoint) -> float:
        """Return the true course from the current vessel position to ``goal``."""

        return initial_bearing_degrees(
            state.latitude,
            state.longitude,
            goal.latitude,
            goal.longitude,
        )

    def action(self, state: BoatState, goal: RouteWaypoint | None) -> BoatAction:
        """Return a bounded rudder/flap action towards the active waypoint."""

        if goal is None:
            return BoatAction(rudder=0.0, flap=0.0)
        return self.action_to_heading(state, self.desired_course(state, goal))

    def action_to_heading(self, state: BoatState, desired_heading: float) -> BoatAction:
        """Return an action that turns the vessel towards a true heading.

        This bridges a tactical-planner/MPC heading reference and the two
        available saildrone control channels.
        """

        error = signed_heading_error(desired_heading, state.heading)
        yaw_rate_degrees_per_second = degrees(state.yaw_rate)
        rudder = (
            -self.rudder_gain_per_degree * error
            + self.yaw_damping_per_degree_per_second * yaw_rate_degrees_per_second
        )
        return BoatAction(
            rudder=float(
                np.clip(rudder, -self.rudder_limit_degrees, self.rudder_limit_degrees)
            ),
            flap=self.flap_degrees,
        )
