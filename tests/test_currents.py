"""Checks for time-varying current interpolation."""

from __future__ import annotations

import unittest

import numpy as np
import xarray as xr

from world.currents import Currents


class CurrentTests(unittest.TestCase):
    def test_multiple_current_snapshots_are_interpolated_in_time(self) -> None:
        layer = Currents()
        layer._dataset = xr.Dataset(
            {
                "water_u": (("time", "latitude", "longitude"), [[[0.0]], [[2.0]]]),
                "water_v": (("time", "latitude", "longitude"), [[[1.0]], [[1.0]]]),
            },
            coords={
                "time": [
                    np.datetime64("2026-07-28T00:00", "ns"),
                    np.datetime64("2026-07-28T02:00", "ns"),
                ],
                "latitude": [40.0],
                "longitude": [18.0],
            },
        )

        current = layer.current(40.0, 18.0, "2026-07-28T01:00:00Z")

        self.assertAlmostEqual(current.u, 1.0)
        self.assertAlmostEqual(current.v, 1.0)

    def test_masked_coastal_interpolation_uses_nearest_valid_sea_cell(self) -> None:
        layer = Currents(coastal_fallback_max_distance_metres=2_000.0)
        layer._dataset = xr.Dataset(
            {
                "water_u": (("time", "latitude", "longitude"), [[[1.0, np.nan], [3.0, 4.0]]]),
                "water_v": (("time", "latitude", "longitude"), [[[2.0, np.nan], [4.0, 5.0]]]),
            },
            coords={
                "time": [np.datetime64("2026-07-28T00:00", "ns")],
                "latitude": [40.0, 40.01],
                "longitude": [18.0, 18.01],
            },
        )

        with self.assertWarnsRegex(RuntimeWarning, "masked coastal cell"):
            current = layer.current(40.004, 18.006, "2026-07-28T00:00:00Z")

        self.assertAlmostEqual(current.u, 1.0)
        self.assertAlmostEqual(current.v, 2.0)

    def test_masked_coastal_interpolation_still_fails_beyond_limit(self) -> None:
        layer = Currents(coastal_fallback_max_distance_metres=100.0)
        layer._dataset = xr.Dataset(
            {
                "water_u": (("time", "latitude", "longitude"), [[[1.0, np.nan], [3.0, 4.0]]]),
                "water_v": (("time", "latitude", "longitude"), [[[2.0, np.nan], [4.0, 5.0]]]),
            },
            coords={
                "time": [np.datetime64("2026-07-28T00:00", "ns")],
                "latitude": [40.0, 40.01],
                "longitude": [18.0, 18.01],
            },
        )

        with self.assertRaisesRegex(ValueError, "beyond the 100 m fallback limit"):
            layer.current(40.004, 18.006, "2026-07-28T00:00:00Z")
