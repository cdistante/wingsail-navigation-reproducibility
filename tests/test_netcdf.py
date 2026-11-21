"""Checks for safe selection of refreshed environmental files."""

from __future__ import annotations

import unittest

import numpy as np
import xarray as xr

from world._netcdf import _latest_time_slices


def field(value: float) -> xr.Dataset:
    return xr.Dataset(
        {"value": (("time", "latitude", "longitude"), [[[value]]])},
        coords={
            "time": [np.datetime64("2026-07-28T00:00")],
            "latitude": [40.0],
            "longitude": [18.0],
        },
    )


class NetcdfRefreshTests(unittest.TestCase):
    def test_newer_retrieval_replaces_an_overlapping_time_slice(self) -> None:
        selected = _latest_time_slices([(1.0, field(1.0)), (2.0, field(2.0))])
        self.assertEqual(len(selected), 1)
        self.assertEqual(float(selected[0]["value"].item()), 2.0)
