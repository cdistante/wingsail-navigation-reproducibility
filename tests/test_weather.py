"""Checks for forecast-run selection in the weather layer."""

from __future__ import annotations

import unittest

import numpy as np
import xarray as xr

from world.weather import Weather


def forecast(valid_time: str, wind_u: float) -> xr.Dataset:
    return xr.Dataset(
        {
            "u10": (("valid_time", "latitude", "longitude"), [[[wind_u]]]),
            "v10": (("valid_time", "latitude", "longitude"), [[[0.0]]]),
        },
        coords={
            "valid_time": [np.datetime64(valid_time)],
            "latitude": [40.0],
            "longitude": [18.0],
        },
    )


class WeatherRunSelectionTests(unittest.TestCase):
    def test_newest_gfs_run_wins_for_an_overlapping_valid_time(self) -> None:
        selected = Weather._latest_forecast_per_valid_time(
            [
                (np.datetime64("2026-07-28T00:00"), forecast("2026-07-29T00:00", 2.0)),
                (np.datetime64("2026-07-28T06:00"), forecast("2026-07-29T00:00", 8.0)),
            ]
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(float(selected[0]["u10"].item()), 8.0)
