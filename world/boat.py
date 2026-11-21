"""Shared saildrone state and dynamics adapter interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from .world import WorldObservation


@dataclass(frozen=True)
class BoatState:
    """Navigation state exchanged between simulator and world layers.

    ``speed`` is speed over ground.  The optional body-frame ``surge_speed``
    and ``sway_speed`` are through-water velocities used by the Fossen 3-DOF
    model, while ``yaw_rate`` is in radians per second.  They remain optional
    so the legacy and quasi-steady models retain their compact state.
    """

    latitude: float
    longitude: float
    heading: float
    speed: float
    rudder: float = 0.0
    flap: float = 0.0
    heel: float = 0.0
    surge_speed: float | None = None
    sway_speed: float = 0.0
    yaw_rate: float = 0.0


@dataclass(frozen=True)
class BoatAction:
    """The two control channels available on the saildrone, in degrees."""

    rudder: float
    flap: float


class BoatDynamics(Protocol):
    """Protocol implemented by the existing saildrone simulator adapter."""

    def step(
        self,
        state: BoatState,
        action: BoatAction,
        environment: "WorldObservation",
        dt: float,
    ) -> BoatState:
        """Advance the boat state by ``dt`` seconds."""


class CallableBoatDynamics:
    """Wrap an existing ``simulator.py`` transition function as ``BoatDynamics``.

    This avoids duplicating the sail/wing and rudder model in the world
    package.  Supply a function that receives ``state``, ``action``, the
    sampled environment, and ``dt`` seconds, and returns a :class:`BoatState`.
    """

    def __init__(
        self,
        transition: Callable[[BoatState, BoatAction, "WorldObservation", float], BoatState],
    ) -> None:
        self._transition = transition

    def step(
        self,
        state: BoatState,
        action: BoatAction,
        environment: "WorldObservation",
        dt: float,
    ) -> BoatState:
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        next_state = self._transition(state, action, environment, dt)
        if not isinstance(next_state, BoatState):
            raise TypeError("Boat dynamics must return a BoatState.")
        return next_state
