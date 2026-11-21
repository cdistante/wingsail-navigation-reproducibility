"""Ensure synthetic traffic reaches the JEPA/MPC occupancy tensor."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from world import AIS, BoatState, MaritimeWorld, OccupancyGrid, SyntheticAIS


class OccupancyAISTests(unittest.TestCase):
    def test_synthetic_target_marks_the_ais_channel(self) -> None:
        latitude, longitude = 39.902416747, 18.414363207
        world = MaritimeWorld(ais=SyntheticAIS("crossing", latitude, longitude)).load()
        try:
            observation = world.observe(
                BoatState(latitude, longitude, heading=90.0, speed=0.0),
                "2026-07-26T21:00:00Z",
                ais_elapsed_seconds=0.0,
            )
            grid = OccupancyGrid(span=1_000.0, resolution=10.0).build(
                observation.boat,
                observation,
            )
        finally:
            world.close()

        self.assertEqual(len(observation.ais_targets), 1)
        self.assertGreater(int(grid.channel("ais_occupied").sum()), 0)

    def test_recorded_replay_can_filter_to_one_mmsi(self) -> None:
        content = "\n".join(
            (
                "receiver_timestamp,mmsi,latitude,longitude,speed,course,heading",
                "2026-09-28T12:40:00Z,247487500,40.62,14.59,7.1,221,221",
                "2026-09-28T12:40:00Z,123456789,40.63,14.60,4.0,180,180",
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recorded_ais.csv"
            path.write_text(content + "\n", encoding="utf-8")
            source = AIS(path, mmsi=247487500).load()
            try:
                targets = source.targets("2026-09-28T12:40:00Z")
            finally:
                source.close()

        self.assertEqual([target.mmsi for target in targets], [247487500])


if __name__ == "__main__":
    unittest.main()
