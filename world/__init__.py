"""Public API for the Maritime World Simulator package."""

from .ais import AIS, AISTarget
from .bathymetry import Bathymetry, Seabed
from .boat import BoatAction, BoatDynamics, BoatState, CallableBoatDynamics
from .coastline import Coast, Coastline
from .colreg import ColregDecision, ColregEncounter, ColregPolicy
from .currents import Current, Currents
from .enc import ChartSample, ENCChart
from .fossen import Fossen3DOFParameters, Fossen3DOFSaildroneDynamics
from .guidance import WaypointGuidance, initial_bearing_degrees, signed_heading_error
from .harmonic_planner import HarmonicPlan, HarmonicTacticalPlanner
from .mpc import MPCResult, SaildroneMPC
from .nmea import (
    NMEA0183TCPClientOutput,
    NMEA0183TCPServerOutput,
    NMEA0183UDPOutput,
    aivdm_position_report,
    aivdm_static_data_sentences,
    hdt_sentence,
    nmea_sentence,
    rmc_sentence,
)
from .occupancy import LocalGrid, OccupancyGrid
from .scenario import Route, RouteMission, RouteWaypoint, Scenario, Waypoint
from .saildrone import SaildroneDynamics, SaildroneParameters
from .synthetic_ais import SyntheticAIS, SyntheticAISTarget
from .waves import Wave, Waves
from .weather import Weather, Wind
from .world import MaritimeWorld, WorldObservation, WorldTransition

__all__ = [
    "AIS",
    "AISTarget",
    "SyntheticAIS",
    "SyntheticAISTarget",
    "Bathymetry",
    "BoatAction",
    "BoatDynamics",
    "BoatState",
    "CallableBoatDynamics",
    "ChartSample",
    "Coast",
    "Coastline",
    "ColregDecision",
    "ColregEncounter",
    "ColregPolicy",
    "Current",
    "Currents",
    "ENCChart",
    "Fossen3DOFParameters",
    "Fossen3DOFSaildroneDynamics",
    "HarmonicPlan",
    "HarmonicTacticalPlanner",
    "MPCResult",
    "WaypointGuidance",
    "LocalGrid",
    "MaritimeWorld",
    "NMEA0183UDPOutput",
    "NMEA0183TCPClientOutput",
    "NMEA0183TCPServerOutput",
    "OccupancyGrid",
    "Route",
    "RouteMission",
    "RouteWaypoint",
    "Scenario",
    "SaildroneDynamics",
    "SaildroneMPC",
    "SaildroneParameters",
    "Seabed",
    "Wave",
    "Waves",
    "Waypoint",
    "Weather",
    "Wind",
    "WorldObservation",
    "WorldTransition",
    "hdt_sentence",
    "aivdm_position_report",
    "aivdm_static_data_sentences",
    "initial_bearing_degrees",
    "nmea_sentence",
    "rmc_sentence",
    "signed_heading_error",
    "GlobalOccupancyMap",
    "LegacyPlannerAdapter",
    "LegacyPlannerMode",
    "LegacyLocalMap",
    "LegacyLocalMapBuilder",
]

# Legacy-planner integration (planner/global/local/visualization replacement).
from .global_map import GlobalOccupancyMap
from .legacy_planner_adapter import LegacyPlannerAdapter, LegacyPlannerMode
from .local_map import LegacyLocalMap, LegacyLocalMapBuilder
