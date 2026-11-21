"""Deterministic AIS traffic encounters for controller development."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

from ._geo import offset_latitude_longitude
from .ais import AISTarget


@dataclass(frozen=True)
class SyntheticAISManoeuvre:
    start_seconds: float
    finish_seconds: float
    course_degrees: float
    speed_metres_per_second: float


MIXED_AIS_MANOEUVRES = {
    999000101: SyntheticAISManoeuvre(115.0, 135.0, 358.5, 3.43),
    999000102: SyntheticAISManoeuvre(115.0, 135.0, 271.5, 2.62),
    999000103: SyntheticAISManoeuvre(85.0, 105.0, 88.5, 3.08),
}

SHARP_MIXED_AIS_MANOEUVRES = {
    999000101: SyntheticAISManoeuvre(115.0, 135.0, 354.0, 3.15),
    999000102: SyntheticAISManoeuvre(115.0, 135.0, 276.0, 2.35),
    999000103: SyntheticAISManoeuvre(85.0, 105.0, 84.0, 3.35),
}


@dataclass(frozen=True)
class SyntheticAISTarget:
    """A vessel with an initial constant-course, constant-speed local track."""

    mmsi: int
    name: str
    callsign: str
    ship_type: int
    initial_east_metres: float
    initial_north_metres: float
    speed_metres_per_second: float
    course_degrees: float
    length_metres: float
    beam_metres: float
    destination: str = "SIMULATION"
    heading_degrees: float | None = None

    def __post_init__(self) -> None:
        if not 1_000_000 <= self.mmsi <= 999_999_999:
            raise ValueError("Synthetic AIS MMSI must have 7 to 9 digits.")
        if not self.name.strip() or not self.callsign.strip():
            raise ValueError("Synthetic AIS target name and callsign must not be empty.")
        if not 0 <= self.ship_type <= 99:
            raise ValueError("AIS ship_type must be in [0, 99].")
        if self.speed_metres_per_second < 0.0 or not math.isfinite(self.speed_metres_per_second):
            raise ValueError("Synthetic AIS speed must be finite and non-negative.")
        if self.length_metres <= 0.0 or self.beam_metres <= 0.0:
            raise ValueError("Synthetic AIS dimensions must be positive.")

    def at(
        self,
        origin_latitude: float,
        origin_longitude: float,
        elapsed_seconds: float,
    ) -> AISTarget:
        """Return the target position after elapsed simulation time."""

        if elapsed_seconds < 0.0 or not math.isfinite(elapsed_seconds):
            raise ValueError("elapsed_seconds must be finite and non-negative.")
        return self._at_epoch(origin_latitude, origin_longitude, elapsed_seconds)

    def _at_epoch(self, origin_latitude: float, origin_longitude: float, elapsed_seconds: float,
                  manoeuvre: SyntheticAISManoeuvre | None = None) -> AISTarget:
        """The same trajectory can supply a report generated before launch."""
        course_radians = math.radians(self.course_degrees)
        travelled = self.speed_metres_per_second * elapsed_seconds
        east = self.initial_east_metres + travelled * math.sin(course_radians)
        north = self.initial_north_metres + travelled * math.cos(course_radians)
        speed = self.speed_metres_per_second
        course = self.course_degrees % 360.0
        if manoeuvre is not None and elapsed_seconds > manoeuvre.start_seconds:
            end_radians = math.radians(manoeuvre.course_degrees)
            end_speed = manoeuvre.speed_metres_per_second
            start_east = speed * math.sin(course_radians)
            start_north = speed * math.cos(course_radians)
            delta_east = end_speed * math.sin(end_radians) - start_east
            delta_north = end_speed * math.cos(end_radians) - start_north
            duration = manoeuvre.finish_seconds - manoeuvre.start_seconds
            turning = min(elapsed_seconds - manoeuvre.start_seconds, duration)
            after = max(0.0, elapsed_seconds - manoeuvre.finish_seconds)
            integrated_change = 0.5 * turning**2 / duration + after
            east += delta_east * integrated_change
            north += delta_north * integrated_change
            east_speed = start_east + delta_east * turning / duration
            north_speed = start_north + delta_north * turning / duration
            speed = math.hypot(east_speed, north_speed)
            course = math.degrees(math.atan2(east_speed, north_speed)) % 360.0
        latitude, longitude = offset_latitude_longitude(
            origin_latitude,
            origin_longitude,
            east,
            north,
        )
        return AISTarget(
            mmsi=self.mmsi,
            latitude=float(latitude),
            longitude=float(longitude),
            speed=speed,
            course=course,
            heading=course if self.heading_degrees is None else self.heading_degrees % 360.0,
            name=self.name,
            callsign=self.callsign,
            ship_type=self.ship_type,
            length_metres=self.length_metres,
            beam_metres=self.beam_metres,
            destination=self.destination,
        )


@dataclass(frozen=True)
class SyntheticAISReporting:
    """Define delivery timing for a deterministic synthetic AIS receiver."""

    interval_seconds: float
    delay_seconds: float = 0.0
    drop_windows: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if not math.isfinite(self.interval_seconds) or self.interval_seconds <= 0:
            raise ValueError("AIS report interval must be finite and positive.")
        if not math.isfinite(self.delay_seconds) or self.delay_seconds < 0:
            raise ValueError("AIS report delay must be finite and non-negative.")
        for start, stop in self.drop_windows:
            if not (math.isfinite(start) and math.isfinite(stop) and 0 <= start < stop):
                raise ValueError("AIS drop windows require finite times with 0 <= start < stop.")
        object.__setattr__(self, "drop_windows", tuple(sorted(self.drop_windows)))

    def latest_epoch(self, elapsed_seconds: float) -> float:
        """Return the generation time of the last delivered report."""
        available = elapsed_seconds - self.delay_seconds
        epoch = math.floor(math.nextafter(available / self.interval_seconds, math.inf)) * self.interval_seconds
        for start, stop in reversed(self.drop_windows):
            if start <= epoch < stop:
                index = math.ceil(start / self.interval_seconds) - 1
                epoch = index * self.interval_seconds
                if epoch >= start:
                    epoch = (index - 1) * self.interval_seconds
        return min(available, epoch)


class SyntheticAIS:
    """AIS layer that replays a named set of deterministic traffic encounters.

    Coordinates are anchored to the saildrone's initial position.  The targets
    use elapsed simulation time so that fast, slow, and paused runs produce the
    same encounter geometry.
    """

    _SCENARIOS: dict[str, tuple[SyntheticAISTarget, ...]] = {
        "crossing": (
            SyntheticAISTarget(
                mmsi=999000101,
                name="SIM CROSSING FERRY",
                callsign="SIM101",
                ship_type=60,
                initial_east_metres=90.0,
                initial_north_metres=-400.0,
                speed_metres_per_second=3.5,
                course_degrees=0.0,
                length_metres=55.0,
                beam_metres=12.0,
            ),
        ),
        "head_on": (
            SyntheticAISTarget(
                mmsi=999000102,
                name="SIM HEAD ON CARGO",
                callsign="SIM102",
                ship_type=70,
                initial_east_metres=350.0,
                initial_north_metres=0.0,
                speed_metres_per_second=2.7,
                course_degrees=270.0,
                length_metres=80.0,
                beam_metres=14.0,
            ),
        ),
        "overtaking": (
            SyntheticAISTarget(
                mmsi=999000103,
                name="SIM OVERTAKING TUG",
                callsign="SIM103",
                ship_type=52,
                initial_east_metres=-300.0,
                initial_north_metres=10.0,
                speed_metres_per_second=3.0,
                course_degrees=90.0,
                length_metres=24.0,
                beam_metres=8.0,
            ),
        ),
    }
    _SCENARIOS["mixed"] = (
        *_SCENARIOS["crossing"],
        *_SCENARIOS["head_on"],
        *_SCENARIOS["overtaking"],
    )

    def __init__(self, scenario: str, origin_latitude: float, origin_longitude: float, *,
                 reporting: SyntheticAISReporting | None = None, manoeuvring: bool = False,
                 sharp_manoeuvres: bool = False, launch_lead_seconds: float = 0.0) -> None:
        if scenario not in self._SCENARIOS:
            available = ", ".join(self.available_scenarios())
            raise ValueError(f"Unknown synthetic AIS scenario {scenario!r}; choose: {available}.")
        if (manoeuvring or sharp_manoeuvres) and scenario != "mixed":
            raise ValueError("The manoeuvring AIS fixture requires the mixed scenario.")
        if not math.isfinite(launch_lead_seconds) or launch_lead_seconds < 0.0:
            raise ValueError("The synthetic launch lead must be finite and non-negative.")
        self.scenario = scenario
        self.origin_latitude = float(origin_latitude)
        self.origin_longitude = float(origin_longitude)
        self.reporting = reporting
        self.launch_lead_seconds = launch_lead_seconds
        self.manoeuvring = manoeuvring or sharp_manoeuvres
        self._manoeuvres = SHARP_MIXED_AIS_MANOEUVRES if sharp_manoeuvres else (
            MIXED_AIS_MANOEUVRES if manoeuvring else {}
        )
        self._loaded = False

    @classmethod
    def available_scenarios(cls) -> tuple[str, ...]:
        """Return the names accepted by the OpenCPN stream command."""

        return tuple(cls._SCENARIOS)

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def time_mode(self) -> str:
        """Expose the same timing convention as elapsed-time CSV AIS replay."""

        return "elapsed_seconds"

    @property
    def reports_refreshed_each_step(self) -> bool:
        return self.reporting is None and not self.manoeuvring

    def load(self) -> "SyntheticAIS":
        """Mark the configured encounter set ready for sampling."""

        self._loaded = True
        return self

    def close(self) -> None:
        self._loaded = False

    def targets(self, when: float | int | None = None) -> list[AISTarget]:
        """Return all scenario vessels at the requested elapsed time."""

        if not self._loaded:
            raise RuntimeError("Synthetic AIS is not loaded. Call load() first.")
        elapsed_seconds = 0.0 if when is None else float(when)
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and non-negative.")
        scenario_time = elapsed_seconds - self.launch_lead_seconds
        epoch = scenario_time if self.reporting is None else self.reporting.latest_epoch(scenario_time)
        reports = [
            target._at_epoch(self.origin_latitude, self.origin_longitude, epoch, self._manoeuvres.get(target.mmsi))
            for target in self._SCENARIOS[self.scenario]
        ]
        if self.reporting is None:
            return reports
        return [replace(report, report_age_seconds=scenario_time - epoch) for report in reports]

    def sample(self, when: float | int | None = None) -> list[AISTarget]:
        """Alias for :meth:`targets`, matching the world-layer interface."""

        return self.targets(when)
