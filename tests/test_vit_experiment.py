"""Tests for the route/time-separated environmental ViT experiment runner."""

from __future__ import annotations

import unittest

from world.vit_experiment import _allocate, _intervals_overlap


class ViTExperimentTests(unittest.TestCase):
    def test_window_sample_allocation_is_exact_and_balanced(self) -> None:
        self.assertEqual(_allocate(10, 3), [4, 3, 3])
        self.assertEqual(_allocate(9, 3), [3, 3, 3])

    def test_time_overlap_uses_half_open_intervals(self) -> None:
        first = {"start": "2026-07-29T00:00:00Z", "end": "2026-07-30T00:00:00Z"}
        touching = {"start": "2026-07-30T00:00:00Z", "end": "2026-07-31T00:00:00Z"}
        overlapping = {"start": "2026-07-29T23:00:00Z", "end": "2026-07-30T01:00:00Z"}
        self.assertFalse(_intervals_overlap(first, touching))
        self.assertTrue(_intervals_overlap(first, overlapping))


if __name__ == "__main__":
    unittest.main()
