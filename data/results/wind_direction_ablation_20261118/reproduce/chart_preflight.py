"""Check saved route waypoints against the benchmark's nearest-cell chart."""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
from math import asin, cos, radians, sin
from pathlib import Path
import sys

import numpy as np

from run_study import CHART, ROUTES, WORKSPACE

sys.path.insert(0, str(WORKSPACE))
from world._geo import EARTH_RADIUS_M
from world.global_map import GlobalOccupancyMap
from world.scenario import Route
sys.path.pop(0)


MINIMUM_DEPTH_M = 3.0


@lru_cache(maxsize=4)
def load_chart(path):
    return GlobalOccupancyMap(path=path, minimum_depth_metres=MINIMUM_DEPTH_M).load()


class ChartPointSampler:
    """Sample the raw nearest-cell chart at a geographic position."""

    def __init__(self, path):
        chart = load_chart(path)
        self.occupancy = chart.occupancy
        self.latitude_origin = float(chart.latitudes[0])
        self.longitude_origin = float(chart.longitudes[0])
        self.latitude_step = float(np.median(np.diff(chart.latitudes)))
        self.longitude_step = float(np.median(np.diff(chart.longitudes)))
        self.rows, self.columns = chart.shape

    def blocked(self, latitude, longitude):
        """Return True for blocked, False for water or None outside the grid."""
        row = round((latitude - self.latitude_origin) / self.latitude_step)
        column = round((longitude - self.longitude_origin) / self.longitude_step)
        if 0 <= row < self.rows and 0 <= column < self.columns:
            return bool(self.occupancy[row, column])
        return None


def chart_context(workspace):
    """Identify circles completely inside one blocked nearest-neighbor cell."""
    chart = load_chart(workspace / CHART)
    rows = []
    for route_id, case in ROUTES.items():
        route_file = workspace / "current_route" / case.file
        for index, waypoint in enumerate(Route.load_gpx(route_file).waypoints):
            latitude, longitude = waypoint.latitude, waypoint.longitude
            chart_row, chart_col = (round(value) for value in chart.latitude_longitude_to_grid(latitude, longitude))
            if not 0 <= chart_row < chart.shape[0] or not 0 <= chart_col < chart.shape[1]:
                raise ValueError(f"{case.file}/{waypoint.name} outside GEBCO coverage")
            elevation = float(chart.elevation[chart_row, chart_col])
            south, west = chart.grid_to_latitude_longitude(chart_row - 0.5, chart_col - 0.5)
            north, east = chart.grid_to_latitude_longitude(chart_row + 0.5, chart_col + 0.5)
            # The nearest crossing of either longitude boundary lies no closer
            # than the perpendicular great-circle distance to its meridian.
            edge_range = EARTH_RADIUS_M * min(
                abs(radians(latitude - south)), abs(radians(latitude - north)),
                asin(abs(cos(radians(latitude)) * sin(radians(longitude - west)))),
                asin(abs(cos(radians(latitude)) * sin(radians(longitude - east)))),
            )
            blocked = bool(chart.occupancy[chart_row, chart_col])
            is_target = index > 0
            rows.append({
                "route": route_id, "route_file": case.file, "waypoint": waypoint.name, "is_capture_target": is_target,
                "latitude": latitude, "longitude": longitude, "chart": CHART,
                "cell_row": chart_row, "cell_column": chart_col,
                "cell_lat_min_deg": min(south, north), "cell_lat_max_deg": max(south, north),
                "cell_lon_min_deg": min(west, east), "cell_lon_max_deg": max(west, east),
                "chart_elevation_m": elevation, "chart_depth_positive_down_m": -elevation,
                "minimum_clear_depth_m": MINIMUM_DEPTH_M, "cell_blocked": blocked,
                "arrival_radius_m": case.radius, "distance_to_nearest_cell_boundary_lower_bound_m": edge_range,
                "complete_capture_disk_in_blocked_cell": is_target and blocked and edge_range > case.radius,
            })
    return rows


def write_chart_context(study, workspace):
    rows = chart_context(workspace)
    with (study / "route_chart_preflight.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    args = parser.parse_args()
    rows = write_chart_context(args.study, args.workspace)
    for row in rows:
        if row["complete_capture_disk_in_blocked_cell"]:
            print(f"{row['route']} {row['waypoint']}: {row['arrival_radius_m']} m disk within one blocked chart cell")


if __name__ == "__main__":
    main()
