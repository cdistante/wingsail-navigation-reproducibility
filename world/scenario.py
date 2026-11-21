"""Reproducible mission/scenario definitions for simulation episodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ElementTree

import numpy as np
import yaml

from ._geo import TimeLike, haversine_metres, normalise_time


@dataclass(frozen=True)
class Waypoint:
    """A WGS-84 mission waypoint."""

    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError("Waypoint latitude must be in [-90, 90].")
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError("Waypoint longitude must be in [-180, 180].")


@dataclass(frozen=True)
class RouteWaypoint(Waypoint):
    """A named waypoint preserved from an imported OpenCPN route."""

    name: str = ""


@dataclass(frozen=True)
class Route:
    """An ordered OpenCPN/GPX route for a simulator mission or MPC reference."""

    name: str
    waypoints: tuple[RouteWaypoint, ...]
    source: Path
    planned_departure: np.datetime64 | None = None

    def __post_init__(self) -> None:
        if len(self.waypoints) < 2:
            raise ValueError("A route must contain at least two waypoints.")

    @property
    def departure(self) -> RouteWaypoint:
        """Return the first route point."""

        return self.waypoints[0]

    @property
    def destination(self) -> RouteWaypoint:
        """Return the final route point."""

        return self.waypoints[-1]

    @property
    def total_distance_metres(self) -> float:
        """Return the sum of great-circle leg lengths."""

        return float(
            sum(
                haversine_metres(start.latitude, start.longitude, end.latitude, end.longitude)
                for start, end in zip(self.waypoints, self.waypoints[1:])
            )
        )

    @classmethod
    def load_gpx(cls, path: Path | str, *, route_name: str | None = None) -> "Route":
        """Load one ordered ``<rte>`` from an OpenCPN GPX export.

        The route geometry is intentionally all that is imported.  Chart data,
        OpenCPN GUIDs, display style, and other vendor extensions remain in
        OpenCPN and are not needed by the simulator.
        """

        source = Path(path)
        if not source.exists():
            raise RuntimeError(f"GPX route file not found: {source}.")
        try:
            root = ElementTree.parse(source).getroot()
        except ElementTree.ParseError as error:
            raise ValueError(f"Invalid GPX/XML route file: {source}.") from error

        routes = [element for element in root.iter() if cls._local_name(element.tag) == "rte"]
        if not routes:
            raise ValueError(f"No GPX <rte> route was found in: {source}.")

        selected: ElementTree.Element | None = None
        for candidate in routes:
            candidate_name = cls._child_text(candidate, "name") or source.stem
            if route_name is None or candidate_name == route_name:
                selected = candidate
                break
        if selected is None:
            raise ValueError(f"Route named {route_name!r} was not found in: {source}.")

        waypoints = []
        for index, point in enumerate(selected):
            if cls._local_name(point.tag) != "rtept":
                continue
            try:
                latitude = float(point.attrib["lat"])
                longitude = float(point.attrib["lon"])
            except (KeyError, ValueError) as error:
                raise ValueError(f"Invalid route point {index + 1} in: {source}.") from error
            waypoints.append(
                RouteWaypoint(
                    latitude=latitude,
                    longitude=longitude,
                    name=cls._child_text(point, "name") or f"{len(waypoints) + 1:03d}",
                )
            )

        selected_name = cls._child_text(selected, "name") or source.stem
        planned_departure_text = cls._descendant_text(selected, "planned_departure")
        planned_departure = (
            normalise_time(planned_departure_text)
            if planned_departure_text is not None
            else None
        )
        return cls(
            name=selected_name,
            waypoints=tuple(waypoints),
            source=source,
            planned_departure=planned_departure,
        )

    @staticmethod
    def _local_name(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    @classmethod
    def _child_text(cls, parent: ElementTree.Element, name: str) -> str | None:
        for child in parent:
            if cls._local_name(child.tag) == name and child.text:
                return child.text.strip()
        return None

    @classmethod
    def _descendant_text(cls, parent: ElementTree.Element, name: str) -> str | None:
        for child in parent.iter():
            if cls._local_name(child.tag) == name and child.text:
                return child.text.strip()
        return None


@dataclass
class RouteMission:
    """Progress through a GPX route, exposing the current MPC waypoint target.

    This class deliberately provides a waypoint objective rather than steering
    commands.  The future JEPA/MPC controller must select safe flap/rudder
    actions that reach each target under the simulated weather and constraints.
    """

    route: Route
    arrival_radius_metres: float = 100.0
    active_index: int = 1

    def __post_init__(self) -> None:
        if self.arrival_radius_metres <= 0.0:
            raise ValueError("arrival_radius_metres must be positive.")
        if not 1 <= self.active_index <= len(self.route.waypoints):
            raise ValueError("active_index must identify a route point after departure.")

    @property
    def complete(self) -> bool:
        """Return whether the vessel has reached the final route point."""

        return self.active_index >= len(self.route.waypoints)

    @property
    def goal(self) -> RouteWaypoint | None:
        """Return the next route point, or ``None`` once the mission is complete."""

        return None if self.complete else self.route.waypoints[self.active_index]

    def update(self, position: Waypoint) -> bool:
        """Advance through every waypoint reached within the arrival radius."""

        advanced = False
        while self.goal is not None:
            goal = self.goal
            distance = haversine_metres(
                position.latitude,
                position.longitude,
                goal.latitude,
                goal.longitude,
            )
            if distance > self.arrival_radius_metres:
                break
            self.active_index += 1
            advanced = True
        return advanced


@dataclass(frozen=True)
class Scenario:
    """Inputs that make a simulation episode repeatable."""

    name: str
    departure: Waypoint
    destination: Waypoint
    start_time: np.datetime64
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str) -> "Scenario":
        """Load a YAML mission definition.

        Required fields are ``departure``, ``destination``, and ``start_time``.
        Each waypoint accepts either ``latitude/longitude`` or ``lat/lon``.
        Additional fields are retained under :attr:`metadata` for controller or
        event-engine configuration.
        """

        path = Path(path)
        if not path.exists():
            raise RuntimeError(f"Scenario file not found: {path}.")

        with path.open() as source:
            data = yaml.safe_load(source) or {}
        if not isinstance(data, dict):
            raise ValueError("Scenario YAML must contain a mapping at the root.")

        try:
            departure = cls._waypoint(data["departure"], "departure")
            destination = cls._waypoint(data.get("destination", data.get("arrival")), "destination")
            start_time = normalise_time(data["start_time"])
        except KeyError as error:
            raise ValueError(f"Scenario is missing required field: {error.args[0]}.") from error

        retained = {
            key: value
            for key, value in data.items()
            if key not in {"name", "departure", "destination", "arrival", "start_time"}
        }
        return cls(
            name=str(data.get("name", path.stem)),
            departure=departure,
            destination=destination,
            start_time=start_time,
            metadata=retained,
        )

    @classmethod
    def create(
        cls,
        *,
        name: str,
        departure: Waypoint,
        destination: Waypoint,
        start_time: TimeLike,
        **metadata: Any,
    ) -> "Scenario":
        """Create a scenario directly from Python values."""

        return cls(
            name=name,
            departure=departure,
            destination=destination,
            start_time=normalise_time(start_time),
            metadata=metadata,
        )

    @staticmethod
    def _waypoint(value: Any, name: str) -> Waypoint:
        if not isinstance(value, dict):
            raise ValueError(f"Scenario {name} must be a latitude/longitude mapping.")
        latitude = value.get("latitude", value.get("lat"))
        longitude = value.get("longitude", value.get("lon"))
        if latitude is None or longitude is None:
            raise ValueError(f"Scenario {name} must provide latitude and longitude.")
        return Waypoint(latitude=float(latitude), longitude=float(longitude))
