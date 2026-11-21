"""Checks for importing OpenCPN GPX routes into simulator missions."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from world import Route, RouteMission, Waypoint


GPX = """<?xml version=\"1.0\"?>
<gpx xmlns=\"http://www.topografix.com/GPX/1/1\" xmlns:opencpn=\"http://www.opencpn.org\">
  <rte>
    <name>test-route</name>
    <extensions><opencpn:planned_departure>2026-07-27T20:20:25Z</opencpn:planned_departure></extensions>
    <rtept lat=\"39.9\" lon=\"18.4\"><name>start</name></rtept>
    <rtept lat=\"39.9\" lon=\"18.5\"><name>middle</name></rtept>
    <rtept lat=\"39.8\" lon=\"18.5\"><name>finish</name></rtept>
  </rte>
</gpx>
"""


class RouteTests(unittest.TestCase):
    """Verify GPX parsing and active-waypoint progression."""

    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.path = Path(self.temporary_directory.name) / "route.gpx"
        self.path.write_text(GPX)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_loads_open_cpn_gpx_route(self) -> None:
        route = Route.load_gpx(self.path)

        self.assertEqual(route.name, "test-route")
        self.assertEqual([waypoint.name for waypoint in route.waypoints], ["start", "middle", "finish"])
        self.assertEqual(str(route.planned_departure), "2026-07-27T20:20:25.000000000")
        self.assertGreater(route.total_distance_metres, 15_000.0)

    def test_mission_advances_only_after_reaching_each_route_point(self) -> None:
        mission = RouteMission(Route.load_gpx(self.path), arrival_radius_metres=100.0)

        self.assertEqual(mission.goal.name if mission.goal else None, "middle")
        self.assertFalse(mission.update(Waypoint(39.9, 18.4)))
        self.assertTrue(mission.update(Waypoint(39.9, 18.5)))
        self.assertEqual(mission.goal.name if mission.goal else None, "finish")
        self.assertTrue(mission.update(Waypoint(39.8, 18.5)))
        self.assertTrue(mission.complete)

    def test_short_otranto_test_route_has_expected_geometry(self) -> None:
        path = Path(__file__).parents[1] / "current_route" / "otranto_short_test_loop.gpx"
        route = Route.load_gpx(path)

        self.assertEqual(route.name, "otranto_short_test_loop")
        self.assertEqual(len(route.waypoints), 5)
        self.assertAlmostEqual(route.total_distance_metres / 1852.0, 1.078, places=3)
