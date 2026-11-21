"""Regression checks for geographic guidance on a coarse occupancy grid."""

from dataclasses import replace
from math import cos, radians, sin
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from world import BoatState, Current, LegacyPlannerAdapter, LocalGrid, Waypoint, WorldObservation
from world._geo import offset_latitude_longitude
from world.local_map import LegacyLocalMap
from world.legacy_planners import HarmonicPlanner, WindAwareHarmonicPlanner
from world.weather import Wind


class WaypointHandoffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.boat = BoatState(40.225, 18.746, heading=90.0, speed=1.0)
        self.grid = LocalGrid(
            data=np.zeros((3, 7, 7), dtype=np.float32),
            channels=("ais_occupied", "depth", "restricted"),
            resolution=115.0,
            span=805.0,
            origin_latitude=self.boat.latitude,
            origin_longitude=self.boat.longitude,
        )
        self.observation = WorldObservation(
            time=np.datetime64("2026-09-30T03:00:00"),
            boat=self.boat,
            wind=Wind(0.0, 8.0, 8.0, 180.0),
            wave=None,
            current=None,
            seabed=None,
            coast=None,
            chart=None,
            ais_targets=(),
        )
        self.blocked = np.zeros((7, 7), dtype=bool)
        self.planner = LegacyPlannerAdapter(mode="standard")

    def goal(self, east: float, north: float = 0.0) -> Waypoint:
        latitude, longitude = offset_latitude_longitude(
            self.boat.latitude, self.boat.longitude, east, north
        )
        return Waypoint(float(latitude), float(longitude))

    def transformer(self, heading: float):
        angle = radians(heading)
        with patch("world.legacy_planner_adapter.TransformerDirectionPolicy") as factory:
            planner = LegacyPlannerAdapter(mode="transformer_diffusion", transformer_checkpoint=Path("model.pt"))
        predict = factory.return_value.predict_direction_with_diagnostics
        predict.return_value = (-cos(angle), sin(angle), 1.0, sin(angle))
        return planner, predict

    def with_wind(self, direction: float) -> WorldObservation:
        angle = radians(direction)
        return replace(self.observation, wind=Wind(-8.0 * sin(angle), -8.0 * cos(angle), 8.0, direction))

    def test_opposite_waypoints_in_the_same_cell_get_distinct_references(self) -> None:
        local_map = LegacyLocalMap(self.grid, self.blocked)
        for east, heading in ((40.0, 90.0), (-40.0, 270.0)):
            with self.subTest(east=east):
                goal = self.goal(east)
                plan = self.planner.plan(local_map, self.observation, goal)
                self.assertEqual(plan.target_cell, (3, 3))
                self.assertEqual(plan.navigation_target, goal)
                self.assertTrue(plan.has_path)
                self.assertAlmostEqual(plan.desired_heading, heading, places=5)

    def test_direct_reference_respects_an_intervening_blocked_cell(self) -> None:
        self.blocked[3, 4] = True
        plan = self.planner.plan(
            LegacyLocalMap(self.grid, self.blocked), self.observation, self.goal(230.0)
        )
        self.assertIsNone(plan.navigation_target)
        self.assertTrue(plan.blocked[3, 4])

    def test_direct_reference_does_not_point_into_the_wind(self) -> None:
        local_map = LegacyLocalMap(self.grid, self.blocked)
        observation = replace(self.observation, wind=Wind(-8.0, 0.0, 8.0, 90.0))
        goal = self.goal(230.0)
        plan = self.planner.plan(local_map, observation, goal)
        self.assertNotEqual(plan.navigation_target, goal)
        self.assertGreaterEqual(abs(plan.desired_heading - 90.0), self.planner.no_go_angle_degrees)

    def test_transformer_rejects_an_unsailable_emitted_heading(self) -> None:
        self.blocked[3, 4] = True
        local_map = LegacyLocalMap(self.grid, self.blocked)
        goal = self.goal(230.0)
        # The second proposal is 55 degrees from the wind before rounding.
        # Its first lattice step is due north, only 30 degrees from the wind.
        for wind, proposed_heading in ((0.0, 0.0), (330.0, 25.0)):
            with self.subTest(wind=wind, proposed_heading=proposed_heading):
                observation = self.with_wind(wind)
                planner, predict = self.transformer(proposed_heading)
                expected = LegacyPlannerAdapter(mode="diffusion").plan(local_map, observation, goal)
                plan = planner.plan(local_map, observation, goal)
                predict.assert_called_once()
                self.assertEqual(planner.transformer_acceptances, 0)
                self.assertEqual(planner.transformer_fallbacks, 1)
                self.assertEqual(plan.path_cells, expected.path_cells)
                self.assertEqual(plan.desired_heading, expected.desired_heading)
                wind_angle = abs((plan.desired_heading - wind + 180.0) % 360.0 - 180.0)
                self.assertGreaterEqual(wind_angle, planner.no_go_angle_degrees)

    def test_transformer_rejects_a_proposal_that_rounds_out_of_the_wind_cone(self) -> None:
        self.blocked[3, 5] = True
        planner, predict = self.transformer(50.0)
        # The proposal is 40 degrees from the wind. Its first lattice step
        # points to 45 degrees, which would pass a lattice-only guard.
        planner.plan(
            LegacyLocalMap(self.grid, self.blocked), self.with_wind(90.0), self.goal(340.0)
        )

        predict.assert_called_once()
        self.assertEqual(planner.transformer_acceptances, 0)
        self.assertEqual(planner.transformer_fallbacks, 1)

    def test_transformer_rejects_a_ray_touching_a_blocked_corner(self) -> None:
        # The NE ray visits free cells (2, 4), (1, 5), (0, 6),
        # but touches the NW corner of the occupied (3, 4) cell.
        self.blocked[3, 4] = True
        planner, predict = self.transformer(45.0)
        planner.plan(LegacyLocalMap(self.grid, self.blocked), self.observation, self.goal(230.0))

        predict.assert_called_once()
        self.assertEqual(planner.transformer_acceptances, 0)
        self.assertEqual(planner.transformer_fallbacks, 1)

    def test_transformer_can_supply_a_clear_sailable_detour(self) -> None:
        self.blocked[3, 5] = True
        planner, predict = self.transformer(45.0)
        plan = planner.plan(LegacyLocalMap(self.grid, self.blocked), self.observation, self.goal(340.0))

        predict.assert_called_once()
        self.assertEqual(planner.transformer_acceptances, 1)
        self.assertEqual(plan.path_cells[:2], ((3, 3), (2, 4)))
        self.assertEqual(plan.desired_heading, 45.0)

    def test_transformer_uses_the_diffusion_tack_on_a_clear_upwind_approach(self) -> None:
        local_map = LegacyLocalMap(self.grid, self.blocked)
        observation = self.with_wind(90.0)
        goal = self.goal(230.0)
        planner, predict = self.transformer(90.0)
        baseline = LegacyPlannerAdapter(mode="diffusion")
        expected = baseline.plan(local_map, observation, goal)
        plan = planner.plan(local_map, observation, goal)

        self.assertIsNotNone(plan.navigation_target)
        self.assertNotEqual(plan.navigation_target, goal)
        self.assertEqual(plan.navigation_target, expected.navigation_target)
        self.assertEqual(plan.desired_heading, expected.desired_heading)
        self.assertGreaterEqual(abs(plan.desired_heading - 90.0), planner.no_go_angle_degrees)

        yawed = replace(observation, boat=replace(observation.boat, heading=85.0))
        retained = planner.plan(local_map, yawed, goal)
        self.assertEqual(retained.desired_heading, plan.desired_heading)
        self.assertEqual(retained.navigation_target, plan.navigation_target)
        predict.assert_not_called()

    def test_signed_and_unsigned_westerly_headings_have_the_same_wind_clearance(self) -> None:
        wind_from = radians(330.0)
        standard = HarmonicPlanner(grid_size=7)
        standard.wind_direction = wind_from
        diffusion = WindAwareHarmonicPlanner(
            SimpleNamespace(grid_size=7), (3, 1), wind_from
        )
        for planner in (standard, diffusion):
            self.assertFalse(planner._is_no_go_heading(radians(-90.0)))
            self.assertFalse(planner._is_no_go_heading(radians(270.0)))

    def test_adverse_current_changes_the_opposite_ground_layline(self) -> None:
        observation = replace(
            self.observation,
            boat=replace(self.boat, heading=55.0, speed=0.25, surge_speed=0.30),
            wind=Wind(0.0, -5.0, 5.0, 0.0),
            current=Current(0.0, -0.15, 0.15, 180.0, 0.0),
        )
        # The boat makes north-west progress through water on the opposite
        # tack, but the southward flow makes its course over ground almost west.
        layline = self.planner._ground_layline(observation, 55.0)
        self.assertAlmostEqual(layline, 84.87, delta=0.1)
        self.assertEqual(self.planner._ground_layline(replace(observation, current=None), 55.0), 55.0)


if __name__ == "__main__":
    unittest.main()
