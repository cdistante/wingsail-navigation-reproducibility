"""Check hourly trend parity, causality, and the persistence branch."""

from __future__ import annotations

import csv
import gzip
from math import atan2, degrees, hypot
from types import SimpleNamespace
from unittest import TestCase, main

import numpy as np

from hybrid_wind import HOUR, STAGE3_RECEIPT, WindTrendProvider
from world._geo import normalise_time
from world.boat import BoatState
from world.currents import Current
from world.waves import Wave
from world.weather import Wind
from world.world import WorldObservation


def wind(east, north):
    return Wind(east, north, hypot(east, north), degrees(atan2(-east, -north)) % 360.)


class RecordingArchive:
    def __init__(self, values):
        self.values = values
        self.reads = []

    def at(self, latitude, longitude, when):
        self.reads.append((latitude, longitude, when))
        return SimpleNamespace(wind=self.values[when.astype("datetime64[h]")])


def observed(when, measured, lat=40.6, lon=18.2):
    boat = BoatState(lat, lon, 120., .5)
    current = Current(.2, 0., .2, 90., 0.)
    wave = Wave(.5, 330., 4.)
    return WorldObservation(normalise_time(when), boat, measured, wave, current, None, None, None, ())


class CausalWindTest(TestCase):
    def test_future_fields_do_not_influence_frozen_forecast(self):
        hour = np.datetime64("2026-10-12T09", "h")
        issued = hour + np.timedelta64(15, "m")
        present_wind = wind(2., 0.)
        obs = observed(issued, present_wind)
        prefix = {hour - HOUR: wind(-2., 0.), hour: present_wind}
        first = RecordingArchive({**prefix, hour + HOUR: wind(30., 60.)})
        second = RecordingArchive({**prefix, hour + HOUR: wind(-30., -60.)})
        provider = WindTrendProvider.from_receipt()
        forecast_a = provider.at_decision(first, obs)
        forecast_b = provider.at_decision(second, obs)
        slots = (issued, issued + np.timedelta64(15, "m"), hour + HOUR, hour + 2 * HOUR)
        self.assertEqual([forecast_a(slot) for slot in slots], [forecast_b(slot) for slot in slots])
        self.assertIs(forecast_a(issued), obs)
        self.assertIs(forecast_a(issued + np.timedelta64(1, "m")).current, obs.current)
        self.assertIs(forecast_a(hour + HOUR).wave, obs.wave)
        self.assertAlmostEqual(forecast_a(slots[1]).wind.u, 2. + provider.wind_change_cap_mps / 3.)
        self.assertAlmostEqual(forecast_a(slots[2]).wind.u, 2. + provider.wind_change_cap_mps)
        self.assertEqual(forecast_a(slots[2]).wind, forecast_a(slots[3]).wind)
        self.assertEqual(forecast_a(slots[2]).wind.direction, 270.)
        self.assertEqual(first.reads, [(obs.boat.latitude, obs.boat.longitude, t) for t in (hour, hour - HOUR)])
        self.assertEqual(first.reads, second.reads)

    def test_zero_blend_is_exact_persistence_without_archive_reads(self):
        hour = np.datetime64("2026-10-12T09", "h")
        obs = observed(hour, wind(2., 1.))
        archive = RecordingArchive({})
        forecast = WindTrendProvider.from_receipt(blend=0.).at_decision(archive, obs)
        self.assertIs(forecast(hour), obs)
        self.assertIs(forecast(hour + HOUR), obs)
        self.assertEqual(archive.reads, [])

    def test_hourly_endpoint_matches_saved_stage3_predictions(self):
        path = STAGE3_RECEIPT.with_name("prediction_records.csv.gz")
        with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if row["block"] == "autumn"]
        provider = WindTrendProvider.from_receipt()
        self.assertEqual(provider.trend_multiplier, .5)
        self.assertTrue(rows)
        for row in rows:
            hour = normalise_time(row["issue_utc"]).astype("datetime64[h]")
            recent = wind(float(row["issue_wind_e_mps"]), float(row["issue_wind_n_mps"]))
            previous = wind(float(row["previous_hour_wind_e_mps"]), float(row["previous_hour_wind_n_mps"]))
            archive = RecordingArchive({hour: recent, hour - HOUR: previous})
            obs = observed(hour, recent, float(row["latitude"]), float(row["longitude"]))
            result = provider.forecast(archive, obs, hour + HOUR).wind
            self.assertAlmostEqual(result.u, float(row["causal_trend_wind_e_mps"]), places=12)
            self.assertAlmostEqual(result.v, float(row["causal_trend_wind_n_mps"]), places=12)


if __name__ == "__main__":
    main()
