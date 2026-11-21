"""Headless adapter of the supplied flap-and-rudder saildrone simulator.

The original ``simulator.py`` couples guidance, physics, Tkinter rendering,
and serial I/O in one module.  This file extracts only its ``calcSpeed`` and
``move`` behaviour so it can be driven by :class:`MaritimeWorld`, JEPA, or
MPC without importing a GUI or opening a serial connection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import SAILDRONE

from ._geo import offset_latitude_longitude
from .boat import BoatAction, BoatState
from .world import WorldObservation


@dataclass(frozen=True)
class SaildroneParameters:
    """Tunable values for the extracted saildrone dynamics.

    The default values describe a compact wing-sail drone using a quasi-steady
    force balance.  They are intentionally exposed because they must be
    calibrated against the selected hull and wing-sail before using this model
    for a real passage plan.

    ``legacy_visualisation`` preserves the supplied simulator.py speed curve
    exactly for regression checks and visual comparison.  It is not a physical
    hydrodynamic model.
    """

    # Legacy simulator.py speed-curve parameters.  These are used only when
    # ``use_legacy_sail_model`` is true.
    max_speed: float = 3.0
    reference_wind_speed: float = 10.0
    max_wind_factor: float = 1.5
    rudder_response: float = float(SAILDRONE["rudder_response_deg_per_s_per_deg_per_mps"])
    rudder_min: float = float(SAILDRONE["rudder_min_degrees"])
    rudder_max: float = float(SAILDRONE["rudder_max_degrees"])
    flap_min: float = float(SAILDRONE["flap_min_degrees"])
    flap_max: float = float(SAILDRONE["flap_max_degrees"])
    flap_rate: float = float(SAILDRONE["flap_rate_degrees_per_s"])
    wave_slowdown_per_metre: float = 0.12
    use_wind_speed_scaling: bool = True
    use_wave_slowdown: bool = True
    apply_current: bool = True
    use_legacy_geodesic: bool = False
    preserve_signed_speed: bool = False
    flap_step_per_call: float | None = None
    use_legacy_sail_model: bool = False

    # Quasi-steady physical model parameters (SI units).
    vessel_mass: float = float(SAILDRONE["mass_kg"])
    air_density: float = float(SAILDRONE["air_density_kg_m3"])
    wing_sail_area: float = float(SAILDRONE["wing_sail_area_m2"])
    flap_effectiveness: float = float(SAILDRONE["flap_effectiveness"])
    lift_slope_per_radian: float = float(SAILDRONE["lift_curve_slope_per_rad"])
    max_lift_coefficient: float = float(SAILDRONE["max_lift_coefficient"])
    parasite_drag_coefficient: float = float(SAILDRONE["profile_drag_coefficient"])
    induced_drag_factor: float = float(SAILDRONE["induced_drag_factor"])
    hull_drag_coefficient: float = float(SAILDRONE["hull_drag_coefficient_n_per_mps2"])
    wave_drag_per_metre: float = float(SAILDRONE["wave_drag_increase_per_metre"])
    centre_of_effort_height: float = float(SAILDRONE["centre_of_effort_height_m"])
    righting_moment_per_degree: float = float(SAILDRONE["righting_moment_per_degree_nm"])
    max_heel: float = float(SAILDRONE["max_heel_degrees"])

    def __post_init__(self) -> None:
        """Reject non-physical configuration values at simulator startup."""

        positive = {
            "vessel_mass": self.vessel_mass,
            "air_density": self.air_density,
            "wing_sail_area": self.wing_sail_area,
            "lift_slope_per_radian": self.lift_slope_per_radian,
            "max_lift_coefficient": self.max_lift_coefficient,
            "parasite_drag_coefficient": self.parasite_drag_coefficient,
            "hull_drag_coefficient": self.hull_drag_coefficient,
            "righting_moment_per_degree": self.righting_moment_per_degree,
            "max_heel": self.max_heel,
        }
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            raise ValueError(f"Saildrone parameter(s) must be positive: {', '.join(invalid)}.")
        if not 0.0 <= self.flap_effectiveness <= 1.0:
            raise ValueError("flap_effectiveness must be between 0 and 1.")
        if self.rudder_min >= self.rudder_max:
            raise ValueError("rudder_min must be lower than rudder_max.")
        if self.flap_min >= self.flap_max:
            raise ValueError("flap_min must be lower than flap_max.")

    @classmethod
    def legacy_visualisation(cls) -> "SaildroneParameters":
        """Return the original simulator's accelerated visualisation values."""

        return cls(
            max_speed=2_000_000.0,
            reference_wind_speed=10.0,
            max_wind_factor=1.0,
            rudder_response=0.00005,
            flap_rate=100.0,
            wave_slowdown_per_metre=0.0,
            use_wind_speed_scaling=False,
            use_wave_slowdown=False,
            apply_current=False,
            use_legacy_geodesic=True,
            preserve_signed_speed=True,
            flap_step_per_call=0.1,
            use_legacy_sail_model=True,
        )


class SaildroneDynamics:
    """Flap/rudder dynamics compatible with :class:`MaritimeWorld.step`.

    The main wing is treated as a passive weather-vane: with zero flap it
    aligns to the apparent wind and produces only profile drag.  Flap
    deflection creates an effective angle of attack, which produces lift and
    induced drag.  The component of the resulting aerodynamic force along the
    hull accelerates the boat against quadratic water resistance; the keel is
    assumed to balance the transverse component in this 2-D navigation model.

    The action is the desired rudder angle and final flap angle.  The flap is
    rate limited, while the world supplies spatial wind and surface-current
    disturbances at each transition.
    """

    def __init__(self, parameters: SaildroneParameters | None = None) -> None:
        self.parameters = parameters or SaildroneParameters()

    def step(
        self,
        state: BoatState,
        action: BoatAction,
        environment: WorldObservation,
        dt: float,
    ) -> BoatState:
        """Advance the saildrone by ``dt`` seconds using sampled environment."""

        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if environment.wind is None:
            raise ValueError("SaildroneDynamics requires a wind observation.")

        if self.parameters.use_legacy_sail_model:
            return self._legacy_step(state, action, environment, dt)
        return self._physical_step(state, action, environment, dt)

    def _legacy_step(
        self,
        state: BoatState,
        action: BoatAction,
        environment: WorldObservation,
        dt: float,
    ) -> BoatState:
        """Keep simulator.py behaviour available as an explicit compatibility mode."""

        parameters = self.parameters
        rudder = float(np.clip(action.rudder, parameters.rudder_min, parameters.rudder_max))
        target_flap = float(np.clip(action.flap, parameters.flap_min, parameters.flap_max))
        sail_angle = self._mod360(environment.wind.direction - state.heading - state.flap)
        speed_through_water = self._legacy_sail_speed(sail_angle, state.flap, environment)
        next_latitude, next_longitude = self._advance_position(
            state,
            speed_through_water,
            environment,
            dt,
        )
        heading = self._mod360(
            state.heading - parameters.rudder_response * rudder * speed_through_water * dt
        )
        flap = self._advance_flap(state.flap, target_flap, dt)
        return BoatState(
            latitude=next_latitude,
            longitude=next_longitude,
            heading=heading,
            speed=self._ground_speed(speed_through_water, state.heading, environment),
            rudder=rudder,
            flap=flap,
            heel=state.heel,
        )

    def _physical_step(
        self,
        state: BoatState,
        action: BoatAction,
        environment: WorldObservation,
        dt: float,
    ) -> BoatState:
        """Advance one force-balanced transition using true and apparent wind."""

        parameters = self.parameters
        rudder = float(np.clip(action.rudder, parameters.rudder_min, parameters.rudder_max))
        target_flap = float(np.clip(action.flap, parameters.flap_min, parameters.flap_max))
        through_water_speed = self._through_water_speed(state, environment)
        forward_force, lateral_force = self._aerodynamic_forces(
            state.heading,
            through_water_speed,
            state.flap,
            environment,
        )
        resistance = self._hull_resistance(through_water_speed, environment)
        acceleration = (forward_force - resistance) / parameters.vessel_mass
        next_through_water_speed = max(0.0, through_water_speed + acceleration * dt)

        # Rudder authority rises with water flow across it.  Its coefficient is
        # kept empirical until a hull-specific yaw/added-mass model is fitted.
        heading = self._mod360(
            state.heading
            - parameters.rudder_response * rudder * next_through_water_speed * dt
        )
        next_latitude, next_longitude = self._advance_position(
            state,
            next_through_water_speed,
            environment,
            dt,
            heading=heading,
        )
        flap = self._advance_flap(state.flap, target_flap, dt)
        ground_speed = self._ground_speed(next_through_water_speed, heading, environment)
        heel = self._heel_from_lateral_force(lateral_force)
        return BoatState(
            latitude=next_latitude,
            longitude=next_longitude,
            heading=heading,
            speed=ground_speed,
            rudder=rudder,
            flap=flap,
            heel=heel,
        )

    def _advance_flap(self, flap: float, target: float, dt: float) -> float:
        if self.parameters.flap_step_per_call is not None:
            # simulator.py::move() changes flap by FLAP_ITERATION once per
            # invocation.  Its equality behaviour deliberately oscillates
            # around flap_final, so preserve it in compatibility mode.
            step = self.parameters.flap_step_per_call
            return flap + step if flap < target else flap - step
        delta = float(np.clip(target - flap, -self.parameters.flap_rate * dt, self.parameters.flap_rate * dt))
        return flap + delta

    def _legacy_sail_speed(
        self,
        sail_angle: float,
        flap: float,
        environment: WorldObservation,
    ) -> float:
        """Exact speed curve from simulator.py::calcSpeed()."""

        wind_factor = 1.0
        if self.parameters.use_wind_speed_scaling:
            wind_factor = min(
                max(environment.wind.speed / self.parameters.reference_wind_speed, 0.0),
                self.parameters.max_wind_factor,
            )
        wave_factor = 1.0
        if self.parameters.use_wave_slowdown and environment.wave is not None:
            wave_factor = max(
                0.0,
                1.0 - self.parameters.wave_slowdown_per_metre * environment.wave.height,
            )
        speed = self.parameters.max_speed * wind_factor * wave_factor * abs(
            np.sin(np.radians(sail_angle))
        )
        if (sail_angle < 180.0 and flap > 0.0) or (sail_angle > 180.0 and flap < 0.0):
            speed = -speed
        return float(speed)

    def _through_water_speed(self, state: BoatState, environment: WorldObservation) -> float:
        """Estimate speed through water from the stored ground-speed state."""

        forward = self._bearing_vector(state.heading)
        ground_velocity = state.speed * forward
        if self.parameters.apply_current and environment.current is not None:
            ground_velocity -= np.array([environment.current.u, environment.current.v])
        return max(0.0, float(np.dot(ground_velocity, forward)))

    def _aerodynamic_forces(
        self,
        heading: float,
        through_water_speed: float,
        flap: float,
        environment: WorldObservation,
    ) -> tuple[float, float]:
        """Return longitudinal and lateral air loads in newtons.

        Wind observations are Earth-fixed air velocities directed *towards*
        ``(u, v)``.  Subtracting the vessel's ground velocity (through-water
        velocity plus current) gives the apparent air velocity at the wing. A
        zero-flap weather-vaning wing has zero lift; the controllable flap
        creates its effective angle of attack.
        """

        parameters = self.parameters
        forward = self._bearing_vector(heading)
        boat_ground_velocity = through_water_speed * forward
        if parameters.apply_current and environment.current is not None:
            boat_ground_velocity += np.array([environment.current.u, environment.current.v])
        apparent_velocity = np.array([environment.wind.u, environment.wind.v]) - boat_ground_velocity
        apparent_speed = float(np.hypot(*apparent_velocity))
        if apparent_speed < 1e-6:
            return 0.0, 0.0

        effective_angle = parameters.flap_effectiveness * np.radians(flap)
        lift_coefficient = float(
            np.clip(
                parameters.lift_slope_per_radian * effective_angle,
                -parameters.max_lift_coefficient,
                parameters.max_lift_coefficient,
            )
        )
        drag_coefficient = (
            parameters.parasite_drag_coefficient
            + parameters.induced_drag_factor * lift_coefficient**2
        )
        dynamic_pressure = 0.5 * parameters.air_density * apparent_speed**2
        lift = dynamic_pressure * parameters.wing_sail_area * lift_coefficient
        drag = dynamic_pressure * parameters.wing_sail_area * drag_coefficient
        apparent_to = apparent_velocity / apparent_speed
        # 90 degrees to port of the apparent airflow; lift sign follows flap
        # sign, so a negative flap under a southerly wind drives eastward.
        apparent_left = np.array([-apparent_to[1], apparent_to[0]])
        aerodynamic_force = lift * apparent_left + drag * apparent_to
        starboard = np.array([forward[1], -forward[0]])
        return (
            float(np.dot(aerodynamic_force, forward)),
            float(np.dot(aerodynamic_force, starboard)),
        )

    def _hull_resistance(self, through_water_speed: float, environment: WorldObservation) -> float:
        """Quadratic surge resistance, increased by significant wave height."""

        multiplier = 1.0
        if self.parameters.use_wave_slowdown and environment.wave is not None:
            multiplier += self.parameters.wave_drag_per_metre * environment.wave.height
        return self.parameters.hull_drag_coefficient * multiplier * through_water_speed**2

    def _heel_from_lateral_force(self, lateral_force: float) -> float:
        """Static heel estimate from lateral sail load and righting stiffness."""

        moment = lateral_force * self.parameters.centre_of_effort_height
        heel = moment / self.parameters.righting_moment_per_degree
        return float(np.clip(heel, -self.parameters.max_heel, self.parameters.max_heel))

    def _advance_position(
        self,
        state: BoatState,
        speed_through_water: float,
        environment: WorldObservation,
        dt: float,
        *,
        heading: float | None = None,
    ) -> tuple[float, float]:
        if self.parameters.use_legacy_geodesic:
            return self._legacy_destination(
                state.latitude,
                state.longitude,
                state.heading,
                speed_through_water * dt,
            )

        heading = state.heading if heading is None else heading
        heading_radians = np.radians(heading)
        east = speed_through_water * np.sin(heading_radians)
        north = speed_through_water * np.cos(heading_radians)
        if self.parameters.apply_current and environment.current is not None:
            east += environment.current.u
            north += environment.current.v
        latitude, longitude = offset_latitude_longitude(
            state.latitude,
            state.longitude,
            east * dt,
            north * dt,
        )
        return float(latitude), float(longitude)

    def _ground_speed(
        self,
        speed_through_water: float,
        heading: float,
        environment: WorldObservation,
    ) -> float:
        if self.parameters.preserve_signed_speed:
            return speed_through_water

        heading = np.radians(heading)
        east = speed_through_water * np.sin(heading)
        north = speed_through_water * np.cos(heading)
        if self.parameters.apply_current and environment.current is not None:
            east += environment.current.u
            north += environment.current.v
        return float(np.hypot(east, north))

    @staticmethod
    def _bearing_vector(direction: float) -> np.ndarray:
        """Return an east/north unit vector for a compass bearing."""

        radians = np.radians(direction)
        return np.array([np.sin(radians), np.cos(radians)])

    @staticmethod
    def _mod360(angle: float) -> float:
        return float(angle % 360.0)

    @staticmethod
    def _legacy_destination(
        latitude: float,
        longitude: float,
        bearing: float,
        distance: float,
    ) -> tuple[float, float]:
        """Exact equivalent of simulator.py::Destination()."""

        radius = 6_371_000.0
        latitude = np.radians(latitude)
        longitude = np.radians(longitude)
        bearing = np.radians(bearing)
        latitude_2 = np.arcsin(
            np.sin(latitude) * np.cos(distance / radius)
            + np.cos(latitude) * np.sin(distance / radius) * np.cos(bearing)
        )
        longitude_2 = longitude + np.arctan2(
            np.sin(bearing) * np.sin(distance / radius) * np.cos(latitude),
            np.cos(distance / radius) - np.sin(latitude) * np.sin(latitude_2),
        )
        return float(np.degrees(latitude_2)), float(np.degrees(longitude_2))
