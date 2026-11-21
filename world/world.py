"""Composition root for the Maritime World Simulator layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._geo import TimeLike, normalise_time
from .ais import AIS, AISTarget
from .synthetic_ais import SyntheticAIS
from .bathymetry import Bathymetry, Seabed
from .boat import BoatAction, BoatDynamics, BoatState
from .coastline import Coast, Coastline
from .currents import Current, Currents
from .enc import ChartSample, ENCChart
from .waves import Wave, Waves
from .weather import Weather, Wind

if TYPE_CHECKING:
    from .occupancy import LocalGrid, OccupancyGrid
    from .scenario import Waypoint


@dataclass(frozen=True)
class WorldObservation:
    """All environment values available to dynamics, JEPA, and MPC at one step."""

    time: np.datetime64
    boat: BoatState
    wind: Wind | None
    wave: Wave | None
    current: Current | None
    seabed: Seabed | None
    coast: Coast | None
    chart: ChartSample | None
    ais_targets: tuple[AISTarget, ...]


@dataclass(frozen=True)
class WorldTransition:
    """A one-step transition directly suitable for a JEPA training dataset."""

    observation: WorldObservation
    action: BoatAction
    next_observation: WorldObservation
    grid: "LocalGrid | None" = None


class MaritimeWorld:
    """Compose optional environmental layers around the saildrone dynamics.

    The world owns no sail dynamics: pass the existing ``simulator.py`` model
    through :class:`world.boat.CallableBoatDynamics`.  This keeps the proven
    flap/rudder physics separate from data loading, map building, JEPA, and
    MPC experimentation.
    """

    def __init__(
        self,
        *,
        weather: Weather | None = None,
        waves: Waves | None = None,
        currents: Currents | None = None,
        bathymetry: Bathymetry | None = None,
        coastline: Coastline | None = None,
        ais: AIS | SyntheticAIS | None = None,
        enc: ENCChart | None = None,
    ) -> None:
        self.weather = weather
        self.waves = waves
        self.currents = currents
        self.bathymetry = bathymetry
        self.coastline = coastline
        self.ais = ais
        self.enc = enc

    def load(self) -> "MaritimeWorld":
        """Load each layer explicitly supplied to the world."""

        for layer in (
            self.weather,
            self.waves,
            self.currents,
            self.bathymetry,
            self.coastline,
            self.ais,
            self.enc,
        ):
            if layer is not None and not layer.loaded:
                layer.load()
        return self

    def observe(
        self,
        boat: BoatState,
        when: TimeLike,
        *,
        current_depth: float = 0.0,
        ais_elapsed_seconds: float | None = None,
    ) -> WorldObservation:
        """Sample all configured world layers at the boat state and UTC time."""

        time = normalise_time(when)
        wind = (
            self.weather.sample(boat.latitude, boat.longitude, time)
            if self.weather is not None
            else None
        )
        wave = (
            self.waves.sample(boat.latitude, boat.longitude, time)
            if self.waves is not None
            else None
        )
        current = (
            self.currents.sample(boat.latitude, boat.longitude, time, current_depth)
            if self.currents is not None
            else None
        )
        seabed = (
            self.bathymetry.sample(boat.latitude, boat.longitude)
            if self.bathymetry is not None
            else None
        )
        coast = (
            self.coastline.sample(boat.latitude, boat.longitude)
            if self.coastline is not None
            else None
        )
        chart = self.enc.sample(boat.latitude, boat.longitude) if self.enc is not None else None

        targets: tuple[AISTarget, ...] = ()
        if self.ais is not None:
            query = (
                ais_elapsed_seconds
                if self.ais.time_mode == "elapsed_seconds"
                else time
            )
            targets = tuple(self.ais.targets(query))

        return WorldObservation(
            time=time,
            boat=boat,
            wind=wind,
            wave=wave,
            current=current,
            seabed=seabed,
            coast=coast,
            chart=chart,
            ais_targets=targets,
        )

    def step(
        self,
        dynamics: BoatDynamics,
        state: BoatState,
        action: BoatAction,
        when: TimeLike,
        dt: float,
        *,
        grid_builder: "OccupancyGrid | None" = None,
        goal: "Waypoint | None" = None,
        current_depth: float = 0.0,
        ais_elapsed_seconds: float | None = None,
    ) -> WorldTransition:
        """Advance external boat dynamics through one fully sampled world step."""

        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        observation = self.observe(
            state,
            when,
            current_depth=current_depth,
            ais_elapsed_seconds=ais_elapsed_seconds,
        )
        next_state = dynamics.step(state, action, observation, dt)
        next_time = observation.time + np.timedelta64(int(round(dt * 1_000_000_000)), "ns")
        next_ais_time = (
            None if ais_elapsed_seconds is None else ais_elapsed_seconds + dt
        )
        next_observation = self.observe(
            next_state,
            next_time,
            current_depth=current_depth,
            ais_elapsed_seconds=next_ais_time,
        )
        grid = (
            grid_builder.build(
                state,
                observation,
                goal=goal,
                restricted=bool(observation.chart and observation.chart.is_restricted),
            )
            if grid_builder is not None
            else None
        )
        return WorldTransition(
            observation=observation,
            action=action,
            next_observation=next_observation,
            grid=grid,
        )

    def close(self) -> None:
        """Close every configured data source."""

        for layer in (
            self.weather,
            self.waves,
            self.currents,
            self.bathymetry,
            self.coastline,
            self.ais,
            self.enc,
        ):
            if layer is not None:
                layer.close()
