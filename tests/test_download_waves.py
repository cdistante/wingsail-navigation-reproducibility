"""Tests for wave download time selection."""

from __future__ import annotations

import unittest

from download.download_waves import parse_utc


class DownloadWavesTests(unittest.TestCase):
    def test_parse_utc_accepts_z_timestamp(self) -> None:
        parsed = parse_utc("2026-09-30T00:00:00Z")

        self.assertEqual(parsed.isoformat(), "2026-09-30T00:00:00+00:00")

    def test_parse_utc_treats_naive_timestamp_as_utc(self) -> None:
        parsed = parse_utc("2026-09-30T00:00:00")

        self.assertEqual(parsed.isoformat(), "2026-09-30T00:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
