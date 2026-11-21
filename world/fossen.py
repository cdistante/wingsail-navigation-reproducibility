"""Fossen-style 3-DOF surge--sway--yaw saildrone dynamics.

The model follows the horizontal-plane relative-motion formulation for a
starboard-port symmetric surface craft:

    M nu_dot_r + C(nu_r) nu_r + D(nu_r) nu_r = tau_wing + tau_rudder

where ``nu_r = [u_r, v_r, r]`` is body velocity relative to the water.
Currents enter the Earth-fixed kinematics, while true wind is converted to an
apparent-air velocity before calculating the wing-sail load.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import SAILDRONE, SAILDRONE_SURGE_RESISTANCE_TABLE_N

from ._geo import offset_latitude_longitude
from .boat import BoatAction, BoatState
from .world import WorldObservation


@dataclass(frozen=True)
class Fossen3DOFParameters:
    """Physical coefficients for :class:`Fossen3DOFSaildroneDynamics`.

    Values are read from ``config.SAILDRONE``.  Added-mass entries are stored
    as positive magnitudes, yielding a positive-definite effective inertia
    matrix when added to the rigid-body terms.
    """

    mass: float = float(SAILDRONE["mass_kg"])
    yaw_inertia: float = float(SAILDRONE["yaw_inertia_kg_m2"])
    added_mass_surge: float = float(SAILDRONE["added_mass_surge_kg"])
    added_mass_sway: float = float(SAILDRONE["added_mass_sway_kg"])
    added_yaw_inertia: float = float(SAILDRONE["added_yaw_inertia_kg_m2"])

    linear_surge_damping: float = float(SAILDRONE["linear_surge_damping_n_per_mps"])
    quadratic_surge_damping: float = float(SAILDRONE["quadratic_surge_damping_n_per_mps2"])
    surge_resistance_table: tuple[tuple[float, float], ...] = SAILDRONE_SURGE_RESISTANCE_TABLE_N
    linear_sway_damping: float = float(SAILDRONE["linear_sway_damping_n_per_mps"])
    quadratic_sway_damping: float = float(SAILDRONE["quadratic_sway_damping_n_per_mps2"])
    linear_yaw_damping: float = float(SAILDRONE["linear_yaw_damping_nm_per_radps"])
    quadratic_yaw_damping: float = float(SAILDRONE["quadratic_yaw_damping_nm_per_radps2"])
    wave_damping_per_metre: float = float(SAILDRONE["wave_drag_increase_per_metre"])

    air_density: float = float(SAILDRONE["air_density_kg_m3"])
    wing_sail_area: float = float(SAILDRONE["wing_sail_area_m2"])
    flap_effectiveness: float = float(SAILDRONE["flap_effectiveness"])
    lift_slope_per_radian: float = float(SAILDRONE["lift_curve_slope_per_rad"])
    max_lift_coefficient: float = float(SAILDRONE["max_lift_coefficient"])
    profile_drag_coefficient: float = float(SAILDRONE["profile_drag_coefficient"])
    induced_drag_factor: float = float(SAILDRONE["induced_drag_factor"])
    wing_sail_longitudinal_position: float = float(SAILDRONE["wing_sail_longitudinal_position_m"])

    water_density: float = float(SAILDRONE["water_density_kg_m3"])
    rudder_area: float = float(SAILDRONE["rudder_area_m2"])
    rudder_lift_slope: float = float(SAILDRONE["rudder_lift_slope_per_rad"])
    rudder_max_lift_coefficient: float = float(SAILDRONE["rudder_max_lift_coefficient"])
    rudder_profile_drag_coefficient: float = float(SAILDRONE["rudder_profile_drag_coefficient"])
    rudder_induced_drag_factor: float = float(SAILDRONE["rudder_induced_drag_factor"])
    rudder_longitudinal_position: float = float(SAILDRONE["rudder_longitudinal_position_m"])
    max_integration_step: float = float(SAILDRONE["fossen_max_integration_step_s"])

    rudder_min: float = float(SAILDRONE["rudder_min_degrees"])
    rudder_max: float = float(SAILDRONE["rudder_max_degrees"])
    flap_min: float = float(SAILDRONE["flap_min_degrees"])
    flap_max: float = float(SAILDRONE["flap_max_degrees"])
    flap_rate: float = float(SAILDRONE["flap_rate_degrees_per_s"])
    centre_of_effort_height: float = float(SAILDRONE["centre_of_effort_height_m"])
    righting_moment_per_degree: float = float(SAILDRONE["righting_moment_per_degree_nm"])
    max_heel: float = float(SAILDRONE["max_heel_degrees"])

    def __post_init__(self) -> None:
        """Fail early for coefficients that violate basic model assumptions."""

        positive = {
            "mass": self.mass,
            "yaw_inertia": self.yaw_inertia,
            "air_density": self.air_density,
            "wing_sail_area": self.wing_sail_area,
            "water_density": self.water_density,
            "rudder_area": self.rudder_area,
            "max_integration_step": self.max_integration_step,
            "righting_moment_per_degree": self.righting_moment_per_degree,
        }
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            raise ValueError(f"Fossen parameter(s) must be positive: {', '.join(invalid)}.")
        if self.rudder_min >= self.rudder_max or self.flap_min >= self.flap_max:
            raise ValueError("Rudder and flap lower bounds must be below their upper bounds.")
        previous_speed = 0.0
        for speed, resistance in self.surge_resistance_table:
            if speed <= previous_speed or resistance < 0.0:
                raise ValueError(
                    "surge_resistance_table must have strictly increasing positive speeds "
                    "and non-negative resistances."
                )
            previous_speed = speed

    @property
    def mass_matrix(self) -> np.ndarray:
        """Diagonal rigid-body plus added-mass matrix for the 3-DOF craft."""

        return np.diag(
            [
                self.mass + self.added_mass_surge,
                self.mass + self.added_mass_sway,
                self.yaw_inertia + self.added_yaw_inertia,
            ]
        )


class Fossen3DOFSaildroneDynamics:
    """Relative 3-DOF Fossen model with wing-sail and rudder control loads.

    The model is deliberately underactuated: the flap supplies wing-sail lift
    and drag, while the rudder produces stern sway/yaw load.  No direct surge
    propulsion is added.  ``BoatState`` carries the resulting through-water
    surge/sway velocities and yaw rate for deterministic JEPA/MPC rollouts.
    """

    def __init__(self, parameters: Fossen3DOFParameters | None = None) -> None:
        self.parameters = parameters or Fossen3DOFParameters()

    def step(
        self,
        state: BoatState,
        action: BoatAction,
        environment: WorldObservation,
        dt: float,
    ) -> BoatState:
        """Integrate one semi-implicit relative-motion transition."""

        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if environment.wind is None:
            raise ValueError("Fossen3DOFSaildroneDynamics requires a wind observation.")

        parameters = self.parameters
        rudder = float(np.clip(action.rudder, parameters.rudder_min, parameters.rudder_max))
        target_flap = float(np.clip(action.flap, parameters.flap_min, parameters.flap_max))
        flap = state.flap
        next_nu_relative = self._relative_velocity_from_state(state)
        heading = state.heading
        latitude, longitude = state.latitude, state.longitude
        substeps = max(1, int(np.ceil(dt / parameters.max_integration_step)))
        substep_dt = dt / substeps
        tau = np.zeros(3)
        current_earth = self._current_earth(environment)
        for _ in range(substeps):
            wing_load = self._wing_load(next_nu_relative, heading, flap, environment)
            rudder_load = self._rudder_load(next_nu_relative, rudder)
            tau = wing_load + rudder_load
            acceleration = np.linalg.solve(
                parameters.mass_matrix,
                tau
                - self._coriolis(next_nu_relative) @ next_nu_relative
                - self._damping(next_nu_relative, environment) @ next_nu_relative,
            )
            next_nu_relative = next_nu_relative + acceleration * substep_dt
            heading = self._mod360(heading + np.degrees(next_nu_relative[2] * substep_dt))
            ground_velocity = self._body_to_earth(heading) @ next_nu_relative[:2] + current_earth
            latitude, longitude = offset_latitude_longitude(
                latitude,
                longitude,
                ground_velocity[0] * substep_dt,
                ground_velocity[1] * substep_dt,
            )
            flap = self._advance_flap(flap, target_flap, substep_dt)

        ground_velocity = self._body_to_earth(heading) @ next_nu_relative[:2] + current_earth
        heel = self._heel_from_lateral_load(float(tau[1]))
        return BoatState(
            latitude=float(latitude),
            longitude=float(longitude),
            heading=heading,
            speed=float(np.hypot(*ground_velocity)),
            rudder=rudder,
            flap=flap,
            heel=heel,
            surge_speed=float(next_nu_relative[0]),
            sway_speed=float(next_nu_relative[1]),
            yaw_rate=float(next_nu_relative[2]),
        )

    def _relative_velocity_from_state(self, state: BoatState) -> np.ndarray:
        if state.surge_speed is not None:
            return np.array([state.surge_speed, state.sway_speed, state.yaw_rate], dtype=float)
        # A legacy/quasi-steady state has no body-frame velocity.  Treat its
        # scalar speed as initial surge through water so a model switch starts
        # from a physically meaningful free-drifting condition.
        return np.array([state.speed, 0.0, state.yaw_rate], dtype=float)

    def _coriolis(self, nu_relative: np.ndarray) -> np.ndarray:
        """Skew-symmetric 3-DOF rigid-body plus diagonal-added-mass matrix."""

        mass = self.parameters.mass + self.parameters.added_mass_surge
        sway_mass = self.parameters.mass + self.parameters.added_mass_sway
        surge, sway, _ = nu_relative
        return np.array(
            [
                [0.0, 0.0, -sway_mass * sway],
                [0.0, 0.0, mass * surge],
                [sway_mass * sway, -mass * surge, 0.0],
            ]
        )

    def _damping(self, nu_relative: np.ndarray, environment: WorldObservation) -> np.ndarray:
        """Positive hydrodynamic damping, with tabulated surge resistance."""

        parameters = self.parameters
        surge, sway, yaw_rate = nu_relative
        multiplier = 1.0
        if environment.wave is not None:
            multiplier += parameters.wave_damping_per_metre * environment.wave.height
        return multiplier * np.diag(
            [
                self._surge_damping_coefficient(surge),
                parameters.linear_sway_damping
                + parameters.quadratic_sway_damping * abs(sway),
                parameters.linear_yaw_damping
                + parameters.quadratic_yaw_damping * abs(yaw_rate),
            ]
        )

    def _surge_damping_coefficient(self, surge_speed: float) -> float:
        """Return ``D_u(|u|)`` such that ``D_u u`` matches calm-water drag.

        The default table is the supplied total-resistance envelope.  Its
        tabulated force is linearly interpolated inside the measured range;
        beyond the final point its local quadratic continuation avoids an
        unbounded or discontinuous extrapolation.  A custom empty table uses
        the conventional linear-plus-quadratic Fossen damping instead.
        """

        speed = abs(float(surge_speed))
        parameters = self.parameters
        if not parameters.surge_resistance_table:
            return (
                parameters.linear_surge_damping
                + parameters.quadratic_surge_damping * speed
            )
        if speed < 1e-9:
            first_speed, first_resistance = parameters.surge_resistance_table[0]
            return first_resistance / first_speed

        speeds, resistances = zip(*parameters.surge_resistance_table)
        if speed <= speeds[0]:
            resistance = resistances[0] * speed / speeds[0]
        elif speed <= speeds[-1]:
            resistance = float(np.interp(speed, speeds, resistances))
        else:
            resistance = resistances[-1] * (speed / speeds[-1]) ** 2
        return resistance / speed

    def _wing_load(
        self,
        nu_relative: np.ndarray,
        heading: float,
        flap: float,
        environment: WorldObservation,
    ) -> np.ndarray:
        """Return body-frame wing-sail surge, sway, and yaw load."""

        parameters = self.parameters
        body_to_earth = self._body_to_earth(heading)
        current_earth = self._current_earth(environment)
        ground_velocity = body_to_earth @ nu_relative[:2] + current_earth
        apparent_air_to = np.array([environment.wind.u, environment.wind.v]) - ground_velocity
        apparent_speed = float(np.hypot(*apparent_air_to))
        if apparent_speed < 1e-6:
            return np.zeros(3)

        effective_angle = parameters.flap_effectiveness * np.radians(flap)
        lift_coefficient = float(
            np.clip(
                parameters.lift_slope_per_radian * effective_angle,
                -parameters.max_lift_coefficient,
                parameters.max_lift_coefficient,
            )
        )
        drag_coefficient = (
            parameters.profile_drag_coefficient
            + parameters.induced_drag_factor * lift_coefficient**2
        )
        dynamic_pressure = 0.5 * parameters.air_density * apparent_speed**2
        apparent_to_unit = apparent_air_to / apparent_speed
        apparent_left = np.array([-apparent_to_unit[1], apparent_to_unit[0]])
        force_earth = (
            dynamic_pressure
            * parameters.wing_sail_area
            * (lift_coefficient * apparent_left + drag_coefficient * apparent_to_unit)
        )
        force_body = body_to_earth.T @ force_earth
        return np.array(
            [
                force_body[0],
                force_body[1],
                parameters.wing_sail_longitudinal_position * force_body[1],
            ]
        )

    def _rudder_load(self, nu_relative: np.ndarray, rudder_degrees: float) -> np.ndarray:
        """Return a stern-rudder load computed from local water-relative flow."""

        parameters = self.parameters
        surge, sway, yaw_rate = nu_relative
        local_sway = sway + parameters.rudder_longitudinal_position * yaw_rate
        inflow_speed = float(np.hypot(surge, local_sway))
        if inflow_speed < 0.05:
            return np.zeros(3)

        inflow_angle = float(np.arctan2(local_sway, max(abs(surge), 0.05)))
        angle_of_attack = np.radians(rudder_degrees) - inflow_angle
        lift_coefficient = float(
            np.clip(
                parameters.rudder_lift_slope * angle_of_attack,
                -parameters.rudder_max_lift_coefficient,
                parameters.rudder_max_lift_coefficient,
            )
        )
        drag_coefficient = (
            parameters.rudder_profile_drag_coefficient
            + parameters.rudder_induced_drag_factor * lift_coefficient**2
        )
        dynamic_pressure = 0.5 * parameters.water_density * inflow_speed**2
        sway_force = dynamic_pressure * parameters.rudder_area * lift_coefficient
        surge_force = -dynamic_pressure * parameters.rudder_area * drag_coefficient
        return np.array(
            [
                surge_force,
                sway_force,
                parameters.rudder_longitudinal_position * sway_force,
            ]
        )

    def _advance_flap(self, flap: float, target: float, dt: float) -> float:
        delta = float(
            np.clip(
                target - flap,
                -self.parameters.flap_rate * dt,
                self.parameters.flap_rate * dt,
            )
        )
        return flap + delta

    def _heel_from_lateral_load(self, lateral_load: float) -> float:
        moment = lateral_load * self.parameters.centre_of_effort_height
        heel = moment / self.parameters.righting_moment_per_degree
        return float(np.clip(heel, -self.parameters.max_heel, self.parameters.max_heel))

    @staticmethod
    def _body_to_earth(heading: float) -> np.ndarray:
        """Map body surge/starboard-sway components to East/North coordinates."""

        radians = np.radians(heading)
        return np.array(
            [
                [np.sin(radians), np.cos(radians)],
                [np.cos(radians), -np.sin(radians)],
            ]
        )

    @staticmethod
    def _current_earth(environment: WorldObservation) -> np.ndarray:
        if environment.current is None:
            return np.zeros(2)
        return np.array([environment.current.u, environment.current.v])

    @staticmethod
    def _mod360(angle: float) -> float:
        return float(angle % 360.0)
