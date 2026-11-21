"""Checks for masked coastal cells in wave interpolation."""

from __future__ import annotations

import unittest

import numpy as np
import xarray as xr

from world.waves import Waves


class WaveTests(unittest.TestCase):
    """Confirm nearby valid sea cells are used only for masked interpolation."""

    def test_masked_coastal_interpolation_uses_nearest_valid_sea_cell(self) -> None:
        layer = Waves(max_nearest_valid_distance_metres=20_000.0)
        values = np.array([[[1.0, np.nan], [np.nan, np.nan]]])
        layer._dataset = xr.Dataset(
            {
                "VHM0": (("time", "latitude", "longitude"), values),
                "VMDR": (("time", "latitude", "longitude"), values * 100.0),
                "VTPK": (("time", "latitude", "longitude"), values * 5.0),
            },
            coords={
                "time": [np.datetime64("2026-07-26T21:00:00")],
                "latitude": [39.9, 40.0],
                "longitude": [18.4, 18.5],
            },
        )

        wave = layer.wave(39.95, 18.45, "2026-07-26T21:00:00Z")

        self.assertEqual(wave.height, 1.0)
        self.assertEqual(wave.direction, 100.0)
        self.assertEqual(wave.peak_period, 5.0)
