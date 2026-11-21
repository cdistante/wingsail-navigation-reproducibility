"""Conservative AIS-informed COLREG encounter assessment for simulation.

The policy purposefully has a narrow role.  It evaluates reported AIS motion
with a constant-course/constant-speed CPA calculation and biases the local
harmonic planner towards a starboard avoidance path when the simulated craft
must give way.  It does not replace a visual/radar lookout, does not establish
legal vessel status or sailing tack, and is not a collision-avoidance system
for real-world navigation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import atan2, cos, degrees, hypot, radians, sin

import numpy as np

from config import COLREG_POLICY

from ._geo import local_east_north_metres
from .ais import AISTarget
from .boat import BoatState
from .occupancy import LocalGrid
from .world import WorldObservation


class ColregEncounter(str, Enum):
    """Encounter categories used by the good-visibility simulation policy."""

    CLEAR = "clear"
    HEAD_ON_GIVE_WAY = "head_on_give_way"
    CROSSING_GIVE_WAY = "crossing_give_way"
    CROSSING_STAND_ON = "crossing_stand_on"
    OVERTAKING_GIVE_WAY = "overtaking_give_way"
    EMERGENCY_AVOID = "emergency_avoid"
    UNKNOWN_RISK = "unknown_risk"


@dataclass(frozen=True)
class ColregDecision:
    """CPA-based assessment of one AIS target.

    Positive relative bearing is starboard; negative is port.  ``give_way``
    means this *simulation policy* asks the harmonic layer to reserve a
    port-forward manoeuvre sector and prefer a starboard alteration.
    """

    target: AISTarget
    encounter: ColregEncounter
    risk_of_collision: bool
    give_way: bool
    time_to_cpa_seconds: float | None
    distance_at_cpa_metres: float
    present_range_metres: float
    relative_bearing_degrees: float
    recommended_manoeuvre: str

    @property
    def active(self) -> bool:
        """Return whether this decision changes the tactical plan."""

        return self.risk_of_collision and self.give_way

    def summary(self) -> str:
        """Return a compact operator-facing explanation."""

        target_name = self.target.name or (
            f"MMSI {self.target.mmsi}" if self.target.mmsi is not None else "AIS target"
        )
        tcpa = "n/a" if self.time_to_cpa_seconds is None else f"{self.time_to_cpa_seconds:.0f}s"
        return (
            f"{target_name}: {self.encounter.value}, range {self.present_range_metres:.0f} m, "
            f"TCPA {tcpa}, DCPA {self.distance_at_cpa_metres:.0f} m, "
            f"{self.recommended_manoeuvre}"
        )


@dataclass(frozen=True)
class ColregPolicy:
    """Apply a bounded good-visibility COLREG-inspired tactical policy.

    The implementation uses Rules 13--17 geometry where AIS gives adequate
    COG/SOG information.  It intentionally treats ambiguous cases
    conservatively rather than attempting to infer sailing Rule-12 tack state
    from AIS data that do not contain it.
    """

    assessment_horizon_seconds: float = float(COLREG_POLICY["assessment_horizon_seconds"])
    risk_dcpa_metres: float = float(COLREG_POLICY["risk_dcpa_metres"])
    emergency_dcpa_metres: float = float(COLREG_POLICY["emergency_dcpa_metres"])
    emergency_tcpa_seconds: float = float(COLREG_POLICY["emergency_tcpa_seconds"])
    minimum_relative_speed_metres_per_second: float = float(
        COLREG_POLICY["minimum_relative_speed_metres_per_second"]
    )
    head_on_bearing_degrees: float = float(COLREG_POLICY["head_on_bearing_degrees"])
    head_on_course_tolerance_degrees: float = float(
        COLREG_POLICY["head_on_course_tolerance_degrees"]
    )
    overtaking_sector_degrees: float = float(COLREG_POLICY["overtaking_sector_degrees"])
    crossing_sector_degrees: float = float(COLREG_POLICY["crossing_sector_degrees"])
    starboard_turn_prohibited_sector_degrees: float = float(
        COLREG_POLICY["starboard_turn_prohibited_sector_degrees"]
    )
    starboard_turn_clearance_metres: float = float(
        COLREG_POLICY["starboard_turn_clearance_metres"]
    )

    def __post_init__(self) -> None:
        positive = {
            "assessment_horizon_seconds": self.assessment_horizon_seconds,
            "risk_dcpa_metres": self.risk_dcpa_metres,
            "emergency_dcpa_metres": self.emergency_dcpa_metres,
            "emergency_tcpa_seconds": self.emergency_tcpa_seconds,
            "minimum_relative_speed_metres_per_second": self.minimum_relative_speed_metres_per_second,
            "starboard_turn_clearance_metres": self.starboard_turn_clearance_metres,
        }
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            raise ValueError(f"COLREG policy values must be positive: {', '.join(invalid)}.")
        if self.emergency_dcpa_metres > self.risk_dcpa_metres:
            raise ValueError("emergency_dcpa_metres cannot exceed risk_dcpa_metres.")
        for name, value, upper in (
            ("head_on_bearing_degrees", self.head_on_bearing_degrees, 90.0),
            ("head_on_course_tolerance_degrees", self.head_on_course_tolerance_degrees, 90.0),
            ("overtaking_sector_degrees", self.overtaking_sector_degrees, 180.0),
            ("crossing_sector_degrees", self.crossing_sector_degrees, 180.0),
            (
                "starboard_turn_prohibited_sector_degrees",
                self.starboard_turn_prohibited_sector_degrees,
                180.0,
            ),
        ):
            if not 0.0 < value < upper:
                raise ValueError(f"{name} must be in (0, {upper}).")

    def assess(self, state: BoatState, observation: WorldObservation) -> tuple[ColregDecision, ...]:
        """Assess all visible AIS targets using their reported COG/SOG.

        The policy accepts the Fossen state when available, including current
        in the Earth-fixed velocity.  The legacy/quasi-steady state has only a
        SOG scalar, so its true-heading velocity is the documented fallback.
        """

        own_east, own_north = self._own_ground_velocity(state, observation)
        own_speed = hypot(own_east, own_north)
        own_course = (
            self._bearing_from_east_north(own_east, own_north)
            if own_speed >= self.minimum_relative_speed_metres_per_second
            else float(state.heading % 360.0)
        )
        decisions = [
            self._assess_target(
                state,
                target,
                own_east=own_east,
                own_north=own_north,
                own_course=own_course,
            )
            for target in observation.ais_targets
        ]
        return tuple(decisions)

    def extra_blocked_mask(
        self,
        grid: LocalGrid,
        state: BoatState,
        decisions: tuple[ColregDecision, ...],
    ) -> np.ndarray:
        """Reserve port-forward cells for active give-way encounters.

        The regular harmonic AIS corridor remains the primary collision
        envelope.  This small asymmetric extension only prevents the planner
        from solving a head-on/give-way situation with a port alteration.
        """

        mask = np.zeros((grid.cells, grid.cells), dtype=bool)
        if not any(decision.active for decision in decisions):
            return mask

        centre = (grid.cells - 1) / 2.0
        rows, columns = np.indices(mask.shape, dtype=float)
        east = (columns - centre) * grid.resolution
        north = (centre - rows) * grid.resolution
        distance = np.hypot(east, north)
        bearing = np.degrees(np.arctan2(east, north)) % 360.0
        relative = self._signed_angle(bearing - state.heading)

        # A negative relative bearing is the port side.  Keep a small forward
        # centreline gap so the departure cell itself stays available; the
        # harmonic planner additionally enforces this invariant.
        port_forward = (
            (distance <= self.starboard_turn_clearance_metres)
            & (distance >= grid.resolution)
            & (relative <= -1.0)
            & (relative >= -self.starboard_turn_prohibited_sector_degrees)
        )
        mask |= port_forward
        return mask

    def _assess_target(
        self,
        state: BoatState,
        target: AISTarget,
        *,
        own_east: float,
        own_north: float,
        own_course: float,
    ) -> ColregDecision:
        east, north = local_east_north_metres(
            target.latitude,
            target.longitude,
            state.latitude,
            state.longitude,
        )
        east = float(east)
        north = float(north)
        present_range = hypot(east, north)
        target_course = target.course if target.course is not None else target.heading
        target_speed = 0.0 if target.speed is None else max(0.0, float(target.speed))
        if target_course is None:
            target_east = target_north = 0.0
            target_course_value: float | None = None
        else:
            target_course_value = float(target_course % 360.0)
            target_east = target_speed * sin(radians(target_course_value))
            target_north = target_speed * cos(radians(target_course_value))

        relative_east = target_east - own_east
        relative_north = target_north - own_north
        relative_speed_squared = relative_east**2 + relative_north**2
        if relative_speed_squared < self.minimum_relative_speed_metres_per_second**2:
            tcpa: float | None = None
            dcpa = present_range
        else:
            raw_tcpa = -(east * relative_east + north * relative_north) / relative_speed_squared
            tcpa = max(0.0, float(raw_tcpa))
            dcpa = hypot(east + relative_east * tcpa, north + relative_north * tcpa)

        risk = (
            dcpa <= self.risk_dcpa_metres
            and (tcpa is None or tcpa <= self.assessment_horizon_seconds)
        )
        bearing = self._bearing_from_east_north(east, north) if present_range > 1e-6 else own_course
        relative_bearing = self._signed_angle(bearing - own_course)

        if not risk:
            encounter = ColregEncounter.CLEAR
            give_way = False
            manoeuvre = "monitor"
        elif present_range <= self.emergency_dcpa_metres or (
            dcpa <= self.emergency_dcpa_metres
            and tcpa is not None
            and tcpa <= self.emergency_tcpa_seconds
        ):
            encounter = ColregEncounter.EMERGENCY_AVOID
            give_way = True
            manoeuvre = "avoid immediately; prefer starboard where safe"
        else:
            encounter, give_way = self._classify_encounter(
                relative_bearing=relative_bearing,
                bearing_to_target=bearing,
                own_course=own_course,
                target_course=target_course_value,
                own_speed=hypot(own_east, own_north),
                target_speed=target_speed,
            )
            manoeuvre = (
                "give way; prefer a substantial starboard alteration"
                if give_way
                else "stand on while monitoring; avoid if risk increases"
                if encounter is ColregEncounter.CROSSING_STAND_ON
                else "avoid conservatively; encounter geometry is ambiguous"
            )

        return ColregDecision(
            target=target,
            encounter=encounter,
            risk_of_collision=risk,
            give_way=give_way,
            time_to_cpa_seconds=tcpa,
            distance_at_cpa_metres=float(dcpa),
            present_range_metres=float(present_range),
            relative_bearing_degrees=float(relative_bearing),
            recommended_manoeuvre=manoeuvre,
        )

    def _classify_encounter(
        self,
        *,
        relative_bearing: float,
        bearing_to_target: float,
        own_course: float,
        target_course: float | None,
        own_speed: float,
        target_speed: float,
    ) -> tuple[ColregEncounter, bool]:
        """Classify a risk encounter without inferring unreported sail tack."""

        if target_course is None:
            return ColregEncounter.UNKNOWN_RISK, True

        course_difference = abs(self._signed_angle(target_course - own_course))
        target_relative_bearing = self._signed_angle(
            (bearing_to_target + 180.0) - target_course
        )
        # Rule 13: a vessel approaching from more than 22.5° abaft the other
        # vessel's beam remains the give-way vessel even when the geometry
        # later resembles a crossing encounter.
        if (
            abs(target_relative_bearing) >= self.overtaking_sector_degrees
            and own_speed > target_speed + self.minimum_relative_speed_metres_per_second
        ):
            return ColregEncounter.OVERTAKING_GIVE_WAY, True
        if (
            abs(relative_bearing) <= self.head_on_bearing_degrees
            and abs(course_difference - 180.0) <= self.head_on_course_tolerance_degrees
        ):
            return ColregEncounter.HEAD_ON_GIVE_WAY, True
        if 0.0 < relative_bearing <= self.crossing_sector_degrees:
            return ColregEncounter.CROSSING_GIVE_WAY, True
        if -self.crossing_sector_degrees <= relative_bearing < 0.0:
            return ColregEncounter.CROSSING_STAND_ON, False
        return ColregEncounter.UNKNOWN_RISK, True

    @staticmethod
    def _signed_angle(angle_degrees: float | np.ndarray) -> float | np.ndarray:
        """Return the smallest signed difference in ``[-180, 180)`` degrees."""

        return (np.asarray(angle_degrees) + 180.0) % 360.0 - 180.0

    @staticmethod
    def _bearing_from_east_north(east: float, north: float) -> float:
        """Convert an Earth-fixed east/north vector into a true bearing."""

        return float(degrees(atan2(east, north)) % 360.0)

    @staticmethod
    def _own_ground_velocity(
        state: BoatState,
        observation: WorldObservation,
    ) -> tuple[float, float]:
        """Return simulated Earth-fixed ground velocity in east/north axes."""

        heading = radians(state.heading)
        if state.surge_speed is not None:
            # Same body-to-Earth convention as the Fossen kinematics.
            east = sin(heading) * state.surge_speed + cos(heading) * state.sway_speed
            north = cos(heading) * state.surge_speed - sin(heading) * state.sway_speed
            if observation.current is not None:
                east += observation.current.u
                north += observation.current.v
            return float(east), float(north)
        return (
            float(state.speed * sin(heading)),
            float(state.speed * cos(heading)),
        )
