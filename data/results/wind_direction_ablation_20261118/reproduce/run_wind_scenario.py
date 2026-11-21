"""Run an archived navigation case with a controlled wind-from direction."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
from math import cos, isfinite, radians, sin
from pathlib import Path
import sys
from unittest.mock import patch


WORKSPACE = Path(__file__).resolve().parents[4]
ARCHIVE_WRAPPER = WORKSPACE / "data/results/learned_tack_arrival_20261118/reproduce"
sys.path[:0] = [str(WORKSPACE), str(ARCHIVE_WRAPPER)]
try:
    import run_archived_replay as archive
    from world import run_legacy_fossen as runner
    from world.navigation_replay import NavigationRecorder
    from world.scenario import Route
    from world.weather import Wind
finally:
    del sys.path[:2]


REGION_PADDING_DEGREES = 1.25
NATIVE_METHOD_LINES = runner.navigation_method_lines


@dataclass(frozen=True)
class WindScenario:
    base_from_deg: float
    offset_deg: float

    def __post_init__(self):
        if not all(map(isfinite, (self.base_from_deg, self.offset_deg))):
            raise ValueError("Wind angles must be finite.")

    @property
    def direction(self) -> float:
        return (self.base_from_deg % 360 + self.offset_deg % 360) % 360

    def wind(self, speed: float) -> Wind:
        angle = radians(self.direction)
        # Meteorological bearing is FROM. East/north components point TO.
        return Wind(-speed * sin(angle), -speed * cos(angle), speed, self.direction)

    @property
    def label(self) -> str:
        return f"Wind FROM {self.direction:g}° ({self.base_from_deg:g}° {self.offset_deg:+g}°)"


def archive_region(arguments) -> dict[str, tuple[float, float]]:
    """Crop native cells around the route, leaving space for drift and detours."""
    if arguments.route_gpx is None:
        latitudes = arguments.start_latitude, arguments.goal_latitude
        longitudes = arguments.start_longitude, arguments.goal_longitude
    else:
        route = Route.load_gpx(arguments.route_gpx, route_name=arguments.route_name)
        latitudes = [point.latitude for point in route.waypoints]
        longitudes = [point.longitude for point in route.waypoints]

    def bounds(coordinates):
        return min(coordinates) - REGION_PADDING_DEGREES, max(coordinates) + REGION_PADDING_DEGREES

    return {"latitude_bounds": bounds(latitudes), "longitude_bounds": bounds(longitudes)}


@contextmanager
def archived_wind_scenario(scenario: WindScenario, region: dict[str, tuple[float, float]]):
    """Apply one forcing scenario to the forecast planner, plant and replay."""
    class DirectedWeather(archive.FastWeather):
        def wind(self, latitude, longitude, when):
            native = super().wind(latitude, longitude, when)
            return scenario.wind(native.speed)

    def method_lines(arguments):
        return tuple(
            line if line != "Archived wind, waves and current"
            else f"{scenario.label} · Archived speed, waves, current"
            for line in NATIVE_METHOD_LINES(arguments)
        )

    with (patch.object(archive, "REGION", region),
          patch.object(runner, "Weather", DirectedWeather),
          patch.object(runner, "Waves", archive.FastWaves),
          patch.object(runner, "Currents", archive.FastCurrents),
          patch.object(runner, "MaritimeWorld", archive.CachedWorld),
          patch.object(runner, "navigation_method_lines", method_lines),
          patch.object(NavigationRecorder, "record", archive.record_local_wind)):
        yield


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--wind-from", type=float, required=True, help="Base meteorological FROM bearing in degrees.")
    parser.add_argument("--wind-offset", type=float, default=0.0, help="Offset from the base bearing in degrees.")
    if sys.argv[1:] == ["--help"]:
        parser.print_help()
        print("\nPass native navigation options after the wind scenario options.")
        runner.parser().print_help()
        return
    options, runner_args = parser.parse_known_args()
    try:
        scenario = WindScenario(options.wind_from, options.wind_offset)
    except ValueError as error:
        parser.error(str(error))
    arguments = runner.parser().parse_args(runner_args)
    if arguments.environment not in {"marine", "full"}:
        parser.error("Use --environment marine or full to retain archived wind speed, waves, and currents.")
    region = archive_region(arguments)
    metadata = {"wind_from_base_deg": scenario.base_from_deg, "wind_offset_deg": scenario.offset_deg,
                "wind_from_actual_deg": scenario.direction, "native_archive_bounds": region}
    print("Wind scenario: " + json.dumps(metadata, sort_keys=True), flush=True)
    with (archived_wind_scenario(scenario, region),
          patch.object(sys, "argv", [sys.argv[0], *runner_args])):
        runner.main()


if __name__ == "__main__":
    main()
