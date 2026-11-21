"""Generate sharded planner-conditioned actuator imitation data around Otranto."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

from ._geo import offset_latitude_longitude
from .boat import BoatState
from .currents import Currents
from .fossen import Fossen3DOFSaildroneDynamics
from .legacy_planner_adapter import LegacyPlannerAdapter
from .local_map import LegacyLocalMap
from .marine_controller import (
    MarineController,
    PLANNER_MODES,
    global_features,
    map_tensor,
    planner_feature_field,
)
from .mpc import SaildroneMPC
from .occupancy import LocalGrid
from .scenario import Waypoint
from .waves import Waves
from .weather import Weather
from .world import MaritimeWorld


def _occupancy(rng: np.random.Generator, cells: int, snapshot: Path | None) -> np.ndarray:
    if snapshot is not None:
        with np.load(snapshot) as data:
            source = np.asarray(data["occ"], dtype=bool)
        if source.shape != (cells, cells):
            raise ValueError(f"Snapshot occupancy is {source.shape}, expected {(cells, cells)}.")
        result = source.copy()
        for _ in range(int(rng.integers(0, 4))):
            row, column = rng.integers(2, cells - 2, size=2)
            result[row, column] = True
        return binary_dilation(result, iterations=int(rng.integers(0, 2)))
    result = np.zeros((cells, cells), dtype=bool)
    for _ in range(int(rng.integers(3, 9))):
        row, column = rng.integers(1, cells - 5, size=2)
        height, width = rng.integers(1, 6, size=2)
        result[row : row + height, column : column + width] = True
    result[[0, -1], :] = True
    result[:, [0, -1]] = True
    return result


def _local_map(observation, occupancy: np.ndarray, resolution: float) -> LegacyLocalMap:
    cells = occupancy.shape[0]
    data = np.zeros((10, cells, cells), dtype=np.float32)
    return LegacyLocalMap(
        tensor=LocalGrid(
            data=data,
            channels=(
                "ais_occupied", "depth", "coast_distance", "wind_u", "wind_v",
                "wave_height", "current_u", "current_v", "goal", "restricted",
            ),
            resolution=resolution,
            span=cells * resolution,
            origin_latitude=observation.boat.latitude,
            origin_longitude=observation.boat.longitude,
        ),
        static_blocked=occupancy,
    )


def generate_dataset(
    output_dir: Path,
    *,
    episodes: int = 3,
    samples_per_episode: int = 3,
    shard_size: int = 32,
    seed: int = 42,
    grid_size: int = 41,
    resolution_metres: float = 115.0,
    snapshot: Path | None = None,
    weather_dir: Path | None = None,
    waves_dir: Path | None = None,
    currents_dir: Path | None = None,
    dagger_checkpoint: Path | None = None,
    dagger_device: str | None = None,
    rollout_dt_seconds: float = 5.0,
) -> dict[str, object]:
    """Generate deterministic, balanced planner/MPC examples from real marine layers."""

    if episodes < 1 or samples_per_episode < 1 or shard_size < 1:
        raise ValueError("episodes, samples_per_episode, and shard_size must be positive.")
    if rollout_dt_seconds <= 0.0:
        raise ValueError("rollout_dt_seconds must be positive.")
    episode_duration_seconds = (
        (samples_per_episode - 1) * rollout_dt_seconds
        if dagger_checkpoint is not None
        else (samples_per_episode - 1) * 30.0 * 60.0
    )
    if episode_duration_seconds >= 3 * 24 * 60 * 60:
        raise ValueError("samples_per_episode exceeds the three-day marine-data window.")
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    world = MaritimeWorld(
        weather=Weather(weather_dir),
        waves=Waves(waves_dir),
        currents=Currents(currents_dir),
    ).load()
    dynamics = Fossen3DOFSaildroneDynamics()
    mpc = SaildroneMPC()
    dagger_controller = (
        MarineController(
            dagger_checkpoint,
            device=dagger_device,
            confidence_threshold=0.0,
        )
        if dagger_checkpoint is not None
        else None
    )
    bucket: dict[str, list[np.ndarray]] = {}
    shard_rows: list[tuple[str, int]] = []
    planner_counts = {mode: 0 for mode in PLANNER_MODES}
    total = 0
    shard_number = 0

    def flush() -> None:
        nonlocal shard_number
        if not bucket:
            return
        name = f"marine_controller_{shard_number:04d}.npz"
        payload = {key: np.stack(values) for key, values in bucket.items()}
        np.savez_compressed(output_dir / name, **payload)
        shard_rows.append((name, len(payload["maps"])))
        bucket.clear()
        shard_number += 1

    try:
        for episode in range(episodes):
            latest_start_second = int(3 * 24 * 60 * 60 - 1 - episode_duration_seconds)
            scenario_time = np.datetime64("2026-09-29T00:00:00") + np.timedelta64(
                int(rng.integers(0, latest_start_second + 1)), "s"
            )
            # Stay around the offshore short-test route and verify both ends of
            # the episode against the real gridded marine products. HYCOM masks
            # coastal cells, so geographic bounding boxes alone are insufficient.
            for location_attempt in range(20):
                latitude = float(rng.uniform(40.20, 40.26))
                longitude = float(rng.uniform(18.70, 18.82))
                probe = BoatState(latitude, longitude, heading=0.0, speed=0.0)
                try:
                    world.observe(probe, scenario_time)
                    world.observe(
                        probe,
                        scenario_time
                        + np.timedelta64(int(round(episode_duration_seconds)), "s"),
                    )
                except ValueError:
                    continue
                break
            else:
                raise RuntimeError(
                    "Could not sample a valid offshore Otranto marine location "
                    f"for episode {episode} after {location_attempt + 1} attempts."
                )
            occupancy = _occupancy(rng, grid_size, snapshot)
            centre = grid_size // 2
            occupancy[centre - 1 : centre + 2, centre - 1 : centre + 2] = False
            state = BoatState(
                latitude=latitude,
                longitude=longitude,
                heading=float(rng.uniform(0.0, 360.0)),
                speed=float(rng.uniform(0.2, 3.0)),
                surge_speed=float(rng.uniform(0.2, 3.0)),
                sway_speed=float(rng.uniform(-0.5, 0.5)),
                yaw_rate=float(rng.uniform(-0.08, 0.08)),
                rudder=float(rng.uniform(-20.0, 20.0)),
                flap=float(rng.uniform(-12.0, 12.0)),
            )
            dagger_mode = PLANNER_MODES[episode % len(PLANNER_MODES)]
            dagger_goal_row = int(rng.integers(2, grid_size - 2))
            dagger_goal_column = int(rng.integers(2, grid_size - 2))
            if np.hypot(dagger_goal_row - centre, dagger_goal_column - centre) < 8:
                dagger_goal_row = 2
            occupancy[dagger_goal_row, dagger_goal_column] = False
            dagger_east = (dagger_goal_column - centre) * resolution_metres
            dagger_north = (centre - dagger_goal_row) * resolution_metres
            dagger_goal_lat, dagger_goal_lon = offset_latitude_longitude(
                latitude, longitude, dagger_east, dagger_north
            )
            dagger_goal = Waypoint(float(dagger_goal_lat), float(dagger_goal_lon))
            for sample_index in range(samples_per_episode):
                mode = (
                    dagger_mode
                    if dagger_controller is not None
                    else PLANNER_MODES[(episode * samples_per_episode + sample_index) % 3]
                )
                if dagger_controller is None and sample_index:
                    state = BoatState(
                        latitude=latitude,
                        longitude=longitude,
                        heading=float(rng.uniform(0.0, 360.0)),
                        speed=float(rng.uniform(0.2, 3.0)),
                        surge_speed=float(rng.uniform(0.2, 3.0)),
                        sway_speed=float(rng.uniform(-0.5, 0.5)),
                        yaw_rate=float(rng.uniform(-0.08, 0.08)),
                        rudder=float(rng.uniform(-20.0, 20.0)),
                        flap=float(rng.uniform(-12.0, 12.0)),
                    )
                when = scenario_time + np.timedelta64(
                    int(round(
                        sample_index
                        * (rollout_dt_seconds if dagger_controller is not None else 1800.0)
                    )),
                    "s",
                )
                observation = world.observe(state, when)
                if dagger_controller is not None:
                    goal = dagger_goal
                else:
                    goal_row = int(rng.integers(2, grid_size - 2))
                    goal_column = int(rng.integers(2, grid_size - 2))
                    if np.hypot(goal_row - centre, goal_column - centre) < 8:
                        goal_row = 2
                    occupancy[goal_row, goal_column] = False
                    east = (goal_column - centre) * resolution_metres
                    north = (centre - goal_row) * resolution_metres
                    goal_lat, goal_lon = offset_latitude_longitude(
                        state.latitude, state.longitude, east, north
                    )
                    goal = Waypoint(float(goal_lat), float(goal_lon))
                local = _local_map(observation, occupancy, resolution_metres)
                plan = LegacyPlannerAdapter(mode=mode).plan(local, observation, goal)
                if plan.desired_heading is None:
                    raise RuntimeError(f"{mode} planner produced no heading for episode {episode}.")
                expert = mpc.choose_action(
                    dynamics,
                    state,
                    observation,
                    plan.desired_heading,
                    local_grid=local.tensor,
                    harmonic_plan=plan,
                ).action
                maps = map_tensor(
                    plan.blocked,
                    planner_feature_field(plan),
                    plan.potential,
                    (centre, centre),
                    plan.target_cell,
                )
                features = global_features(observation, state, 45.0)
                confidence_target = 1.0
                behavior_action = expert
                if dagger_controller is not None:
                    decision = dagger_controller.decide(
                        maps,
                        features,
                        mode,
                        state,
                        max_rudder_slew=mpc.max_rudder_change_per_update_degrees,
                        max_flap_slew=mpc.max_flap_change_per_update_degrees,
                    )
                    if decision.accepted:
                        behavior_action = decision.action
                    confidence_target = float(
                        abs(behavior_action.rudder - expert.rudder) <= 3.0
                        and abs(behavior_action.flap - expert.flap) <= 3.0
                    )
                sample = {
                    "maps": maps,
                    "global_state": features,
                    "planner_id": np.asarray(PLANNER_MODES.index(mode), dtype=np.int64),
                    "actions": np.asarray((expert.rudder, expert.flap), dtype=np.float32),
                    "confidence_target": np.asarray(confidence_target, dtype=np.float32),
                    "episode_id": np.asarray(episode, dtype=np.int64),
                    "time_ns": np.asarray(when.astype("datetime64[ns]").astype(np.int64)),
                    "latitude_longitude": np.asarray(
                        (state.latitude, state.longitude), dtype=np.float64
                    ),
                }
                for key, value in sample.items():
                    bucket.setdefault(key, []).append(value)
                planner_counts[mode] += 1
                total += 1
                if len(bucket["maps"]) >= shard_size:
                    flush()
                if dagger_controller is not None:
                    state = dynamics.step(
                        state,
                        behavior_action,
                        observation,
                        rollout_dt_seconds,
                    )
        flush()
    finally:
        world.close()

    with (output_dir / "index.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("shard", "samples"))
        writer.writerows(shard_rows)
    summary = {
        "samples": total,
        "episodes": episodes,
        "seed": seed,
        "planner_counts": planner_counts,
        "time_range": ["2026-09-29", "2026-10-01"],
        "marine_sources": {"weather": True, "waves": True, "currents": True},
        "grid_size": grid_size,
        "dataset_kind": "dagger" if dagger_controller is not None else "expert",
        "dagger_checkpoint": (
            None if dagger_checkpoint is None else str(dagger_checkpoint)
        ),
        "rollout_dt_seconds": rollout_dt_seconds,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-dir", type=Path, default=Path("data/training/marine_controller"))
    result.add_argument("--episodes", type=int, default=30)
    result.add_argument("--samples-per-episode", type=int, default=3)
    result.add_argument("--shard-size", type=int, default=128)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--grid-size", type=int, default=41)
    result.add_argument("--resolution-metres", type=float, default=115.0)
    result.add_argument("--live-snapshot", type=Path)
    result.add_argument("--weather-dir", type=Path)
    result.add_argument("--waves-dir", type=Path)
    result.add_argument("--currents-dir", type=Path)
    result.add_argument("--dagger-checkpoint", type=Path)
    result.add_argument("--dagger-device", default=None)
    result.add_argument("--rollout-dt-seconds", type=float, default=5.0)
    result.add_argument("--smoke", action="store_true", help="Generate 3 balanced examples only.")
    return result


def main() -> None:
    args = parser().parse_args()
    episodes = 1 if args.smoke else args.episodes
    samples = 3 if args.smoke else args.samples_per_episode
    summary = generate_dataset(
        args.output_dir,
        episodes=episodes,
        samples_per_episode=samples,
        shard_size=args.shard_size,
        seed=args.seed,
        grid_size=args.grid_size,
        resolution_metres=args.resolution_metres,
        snapshot=args.live_snapshot,
        weather_dir=args.weather_dir,
        waves_dir=args.waves_dir,
        currents_dir=args.currents_dir,
        dagger_checkpoint=args.dagger_checkpoint,
        dagger_device=args.dagger_device,
        rollout_dt_seconds=args.rollout_dt_seconds,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
