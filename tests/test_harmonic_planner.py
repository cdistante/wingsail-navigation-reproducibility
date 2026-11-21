"""Tests for wind-aware harmonic local tactical planning."""

from __future__ import annotations

import unittest

import numpy as np

from world import (
    AISTarget,
    BoatState,
    HarmonicTacticalPlanner,
    LocalGrid,
    Waypoint,
    WorldObservation,
)
from world._geo import offset_latitude_longitude
from world.synthetic_ais import SyntheticAIS
from world.weather import Wind


class HarmonicPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.latitude = 39.9
        self.longitude = 18.4
        self.grid = LocalGrid(
            data=np.zeros((10, 101, 101), dtype=np.float32),
            channels=(
                "ais_occupied",
                "depth",
                "coast_distance",
                "wind_u",
                "wind_v",
                "wave_height",
                "current_u",
                "current_v",
                "goal",
                "restricted",
            ),
            resolution=10.0,
            span=1_010.0,
            origin_latitude=self.latitude,
            origin_longitude=self.longitude,
        )
        self.boat = BoatState(self.latitude, self.longitude, heading=90.0, speed=1.0)

    def observation(
        self,
        *,
        wind_from: float | None = None,
        ais_targets: tuple[AISTarget, ...] = (),
    ) -> WorldObservation:
        wind = None if wind_from is None else Wind(0.0, 0.0, 8.0, wind_from)
        return WorldObservation(
            time=np.datetime64("2026-07-26T21:00:00"),
            boat=self.boat,
            wind=wind,
            wave=None,
            current=None,
            seabed=None,
            coast=None,
            chart=None,
            ais_targets=ais_targets,
        )

    def goal(self, east: float, north: float) -> Waypoint:
        latitude, longitude = offset_latitude_longitude(
            self.latitude, self.longitude, east, north
        )
        return Waypoint(float(latitude), float(longitude))

    def test_open_water_plan_points_towards_an_eastward_goal(self) -> None:
        plan = HarmonicTacticalPlanner().plan(
            self.grid,
            self.observation(),
            self.goal(300.0, 0.0),
        )
        self.assertTrue(plan.converged)
        self.assertTrue(plan.has_path)
        self.assertAlmostEqual(plan.desired_heading, 90.0, delta=1e-6)

    def test_clear_beam_reach_uses_the_direct_waypoint_line(self) -> None:
        plan = HarmonicTacticalPlanner().plan(
            self.grid,
            self.observation(wind_from=180.0),
            self.goal(300.0, 0.0),
        )
        self.assertTrue(plan.has_path)
        self.assertAlmostEqual(plan.desired_heading, 90.0, delta=1e-6)

    def test_upwind_goal_returns_a_tack_not_a_direct_upwind_heading(self) -> None:
        plan = HarmonicTacticalPlanner().plan(
            self.grid,
            self.observation(wind_from=90.0),
            self.goal(300.0, 0.0),
        )
        self.assertTrue(plan.has_path)
        difference = abs((plan.desired_heading - 90.0 + 180.0) % 360.0 - 180.0)
        self.assertGreaterEqual(difference, 45.0)

    def test_global_goal_becomes_a_boundary_exit_reference(self) -> None:
        plan = HarmonicTacticalPlanner().plan(
            self.grid,
            self.observation(),
            self.goal(5_000.0, 0.0),
        )
        self.assertTrue(plan.target_on_boundary)
        self.assertEqual(plan.target_cell[1], self.grid.cells - 1)

    def test_blocked_distant_goal_stays_on_a_free_boundary_exit(self) -> None:
        data = self.grid.data.copy()
        # Block the desired east-edge point, leaving a safe exit immediately
        # above it.  The temporary target must not jump to an interior cell.
        data[0, 50, 100] = 1.0
        grid = LocalGrid(
            data=data,
            channels=self.grid.channels,
            resolution=self.grid.resolution,
            span=self.grid.span,
            origin_latitude=self.latitude,
            origin_longitude=self.longitude,
        )
        plan = HarmonicTacticalPlanner().plan(
            grid,
            self.observation(),
            self.goal(5_000.0, 0.0),
        )
        row, column = plan.target_cell
        self.assertTrue(row in (0, grid.cells - 1) or column in (0, grid.cells - 1))
        self.assertNotEqual(plan.target_cell, (50, 100))

    def test_predicted_ais_track_is_added_to_the_obstacle_map(self) -> None:
        target_latitude, target_longitude = offset_latitude_longitude(
            self.latitude, self.longitude, 0.0, -300.0
        )
        target = AISTarget(
            mmsi=999000101,
            latitude=float(target_latitude),
            longitude=float(target_longitude),
            speed=3.0,
            course=0.0,
            heading=0.0,
            length_metres=40.0,
            beam_metres=10.0,
        )
        planner = HarmonicTacticalPlanner(
            ais_prediction_horizon_seconds=120.0,
            ais_prediction_interval_seconds=10.0,
        )
        blocked = planner.obstacle_mask(self.grid, self.observation(ais_targets=(target,)))
        self.assertTrue(blocked[80, 50])  # Present target position: 300 m south.
        self.assertTrue(blocked[50, 50])  # Predicted position after 100 seconds.

    def test_anisotropic_path_does_not_reverse_an_eastbound_mission(self) -> None:
        """A moving crossing target may cause a detour, never a route U-turn."""

        grid = LocalGrid(
            data=np.zeros((10, 81, 81), dtype=np.float32),
            channels=self.grid.channels,
            resolution=5.0,
            span=405.0,
            origin_latitude=self.latitude,
            origin_longitude=self.longitude,
        )
        target = SyntheticAIS("crossing", self.latitude, self.longitude).load().targets()[0]
        plan = HarmonicTacticalPlanner().plan(
            grid,
            self.observation(wind_from=178.0, ais_targets=(target,)),
            self.goal(5_000.0, 0.0),
        )
        self.assertTrue(plan.has_path)
        eastward_component = np.sin(np.radians(plan.desired_heading))
        self.assertGreater(eastward_component, 0.0)


if __name__ == "__main__":
    unittest.main()
