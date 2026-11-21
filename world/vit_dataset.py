"""Environmental occupancy-grid datasets for Vision Transformer imitation learning.

The harmonic tactical planner is the teacher during this pre-JEPA stage.  Each
sample contains the same local grid presented to the future JEPA/MPC stack,
plus global environmental and vessel-state features.  The target is a
continuous Earth-frame unit direction vector ``[east, north]`` corresponding
to the teacher's desired heading.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._geo import haversine_metres, normalise_time
from .ais import AIS
from .bathymetry import Bathymetry
from .boat import BoatState
from .coastline import Coastline
from .currents import Currents
from .guidance import initial_bearing_degrees
from .harmonic_planner import HarmonicTacticalPlanner
from .occupancy import OccupancyGrid
from .scenario import Route, Waypoint
from .synthetic_ais import SyntheticAIS
from .waves import Waves
from .weather import Weather
from .world import MaritimeWorld, WorldObservation


GLOBAL_FEATURE_NAMES = (
    "wind_u_mps",
    "wind_v_mps",
    "wind_speed_mps",
    "wave_height_m",
    "wave_peak_period_s",
    "current_u_mps",
    "current_v_mps",
    "boat_speed_mps",
    "boat_heading_sin",
    "boat_heading_cos",
    "rudder_degrees",
    "flap_degrees",
    "goal_bearing_sin",
    "goal_bearing_cos",
    "goal_distance_km",
)


@dataclass(frozen=True)
class AISTrafficSource:
    """One reproducible AIS source used while generating ViT samples.

    ``synthetic`` sources create an encounter relative to the sampled
    saildrone position.  ``recorded`` sources replay a processed AIS CSV at
    the sampled UTC time (or at a sampled elapsed time for legacy logs).
    The weights are converted to an exact per-dataset allocation, so an AIS
    balance claimed in an experiment configuration is retained in the archive.
    """

    name: str
    kind: str
    weight: float = 1.0
    scenario: str | None = None
    path: Path | str | None = None
    elapsed_seconds_max: float = 3_600.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("AIS traffic-source names must not be empty.")
        if self.kind not in {"none", "synthetic", "recorded"}:
            raise ValueError("AIS traffic-source kind must be none, synthetic, or recorded.")
        if not np.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("AIS traffic-source weight must be finite and positive.")
        if self.kind == "synthetic":
            if self.scenario not in SyntheticAIS.available_scenarios():
                available = ", ".join(SyntheticAIS.available_scenarios())
                raise ValueError(
                    f"Synthetic AIS scenario must be one of {available}; got {self.scenario!r}."
                )
        if self.kind == "recorded" and self.path is None:
            raise ValueError("Recorded AIS traffic sources require a processed CSV path.")
        if self.elapsed_seconds_max < 0.0:
            raise ValueError("AIS elapsed_seconds_max must be non-negative.")

    @property
    def source_path(self) -> Path | None:
        return None if self.path is None else Path(self.path)


def direction_vector(heading_degrees: float) -> np.ndarray:
    """Convert a true heading into an Earth-frame ``[east, north]`` vector."""

    heading = np.radians(float(heading_degrees))
    return np.asarray([np.sin(heading), np.cos(heading)], dtype=np.float32)


def heading_from_direction(vector: np.ndarray) -> float:
    """Convert an Earth-frame direction vector to a true heading in degrees."""

    vector = np.asarray(vector, dtype=float)
    if vector.shape[-1] != 2 or not np.all(np.isfinite(vector)):
        raise ValueError("Direction vector must contain finite east and north components.")
    if np.hypot(vector[0], vector[1]) < 1e-12:
        raise ValueError("Direction vector must be non-zero.")
    return float(np.degrees(np.arctan2(vector[0], vector[1])) % 360.0)


def global_environment_features(
    state: BoatState,
    observation: WorldObservation,
    goal: Waypoint,
) -> np.ndarray:
    """Return the non-spatial features fused with the ViT grid embedding."""

    wind = observation.wind
    wave = observation.wave
    current = observation.current
    goal_bearing = initial_bearing_degrees(
        state.latitude,
        state.longitude,
        goal.latitude,
        goal.longitude,
    )
    goal_distance_km = min(
        200.0,
        haversine_metres(state.latitude, state.longitude, goal.latitude, goal.longitude)
        / 1_000.0,
    )
    heading = np.radians(state.heading)
    bearing = np.radians(goal_bearing)
    return np.asarray(
        [
            0.0 if wind is None else wind.u,
            0.0 if wind is None else wind.v,
            0.0 if wind is None else wind.speed,
            0.0 if wave is None else wave.height,
            0.0 if wave is None else wave.peak_period,
            0.0 if current is None else current.u,
            0.0 if current is None else current.v,
            state.speed,
            np.sin(heading),
            np.cos(heading),
            state.rudder,
            state.flap,
            np.sin(bearing),
            np.cos(bearing),
            goal_distance_km,
        ],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class ViTDataset:
    """Arrays and provenance for supervised environmental ViT training."""

    grids: np.ndarray
    global_features: np.ndarray
    targets: np.ndarray
    split: np.ndarray
    teacher_headings: np.ndarray
    traffic_sources: np.ndarray
    partitions: np.ndarray
    metadata: dict[str, object]

    def save(self, path: Path | str) -> Path:
        """Write a compressed, self-describing dataset archive."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            grids=self.grids.astype(np.float32),
            global_features=self.global_features.astype(np.float32),
            targets=self.targets.astype(np.float32),
            split=self.split.astype(np.int8),
            teacher_headings=self.teacher_headings.astype(np.float32),
            traffic_sources=np.asarray(self.traffic_sources, dtype="U"),
            partitions=np.asarray(self.partitions, dtype="U"),
            metadata_json=np.asarray(json.dumps(self.metadata, sort_keys=True)),
        )
        return destination


class EnvironmentalViTDatasetBuilder:
    """Sample forecast-replayed routes and label their local grids with harmonic plans."""

    def __init__(
        self,
        *,
        grid_span_metres: float = 400.0,
        grid_resolution_metres: float = 10.0,
        environmental_margin_degrees: float = 1.0,
        maximum_attempt_multiplier: int = 20,
        seed: int = 7,
    ) -> None:
        if environmental_margin_degrees <= 0.0:
            raise ValueError("environmental_margin_degrees must be positive.")
        if maximum_attempt_multiplier < 1:
            raise ValueError("maximum_attempt_multiplier must be positive.")
        self.grid = OccupancyGrid(span=grid_span_metres, resolution=grid_resolution_metres)
        self.planner = HarmonicTacticalPlanner()
        self.environmental_margin_degrees = float(environmental_margin_degrees)
        self.maximum_attempt_multiplier = int(maximum_attempt_multiplier)
        self.seed = int(seed)

    def build(
        self,
        routes: tuple[Route, ...],
        *,
        samples: int,
        samples_per_episode: int = 4,
        time_start: str | np.datetime64 | None = None,
        time_end: str | np.datetime64 | None = None,
        traffic_sources: tuple[AISTrafficSource, ...] | None = None,
        partition_name: str = "unassigned",
    ) -> ViTDataset:
        """Build samples using all available world environmental layers.

        The ordinary command line mode assigns its split by episode—not by
        individual grid—so closely related samples do not leak across the
        train/validation/test boundary.  ``world.vit_experiment`` passes a
        fixed ``partition_name`` and overwrites the split only after complete,
        route/time-separated partitions have been built.
        """

        if not routes:
            raise ValueError("At least one GPX route is required for ViT dataset generation.")
        if samples < 3:
            raise ValueError("At least three samples are required for train/validation/test splits.")
        if samples_per_episode < 1:
            raise ValueError("samples_per_episode must be positive.")
        if not partition_name.strip():
            raise ValueError("partition_name must not be empty.")

        sources = traffic_sources or (AISTrafficSource("none", "none"),)
        if len({source.name for source in sources}) != len(sources):
            raise ValueError("AIS traffic-source names must be unique.")

        latitude_bounds, longitude_bounds = self._environmental_bounds(routes)
        world = MaritimeWorld(
            weather=Weather(
                latitude_bounds=latitude_bounds,
                longitude_bounds=longitude_bounds,
            ),
            waves=Waves(
                latitude_bounds=latitude_bounds,
                longitude_bounds=longitude_bounds,
            ),
            currents=Currents(
                latitude_bounds=latitude_bounds,
                longitude_bounds=longitude_bounds,
            ),
            bathymetry=Bathymetry(),
            coastline=Coastline(),
        ).load()
        try:
            lower, upper = self._common_time_bounds(world)
            if time_start is not None:
                lower = max(lower, normalise_time(time_start))
            if time_end is not None:
                upper = min(upper, normalise_time(time_end))
            if lower >= upper:
                raise ValueError("Requested ViT training interval has no common environmental coverage.")

            rng = np.random.default_rng(self.seed)
            traffic_schedule = self._traffic_schedule(sources, samples, rng)
            recorded_ais = {
                source.name: AIS(source.source_path).load()
                for source in sources
                if source.kind == "recorded"
            }
            grids: list[np.ndarray] = []
            features: list[np.ndarray] = []
            targets: list[np.ndarray] = []
            headings: list[float] = []
            episode_ids: list[int] = []
            traffic_labels: list[str] = []
            attempted = 0
            maximum_attempts = samples * self.maximum_attempt_multiplier
            while len(grids) < samples and attempted < maximum_attempts:
                attempted += 1
                traffic_source = traffic_schedule[len(grids)]
                episode_id = len(grids) // samples_per_episode
                route = routes[episode_id % len(routes)]
                latitude, longitude, goal = self._sample_route_state(route, rng)
                state = BoatState(
                    latitude=latitude,
                    longitude=longitude,
                    heading=float(rng.uniform(0.0, 360.0)),
                    speed=float(rng.uniform(0.2, 2.0)),
                    rudder=float(rng.uniform(-7.5, 7.5)),
                    flap=float(rng.uniform(-15.0, 15.0)),
                )
                when = self._sample_time(lower, upper, rng)
                try:
                    observation = world.observe(state, when)
                    ais_targets = self._ais_targets(
                        traffic_source,
                        recorded_ais,
                        state,
                        when,
                        rng,
                    )
                    if ais_targets:
                        observation = replace(observation, ais_targets=ais_targets)
                    local_grid = self.grid.build(
                        state,
                        observation,
                        goal=goal,
                        restricted=bool(observation.chart and observation.chart.is_restricted),
                    )
                    plan = self.planner.plan(local_grid, observation, goal)
                except (RuntimeError, ValueError):
                    # Coastal data masks and genuinely infeasible local plans
                    # are excluded instead of fabricating a training target.
                    continue
                if not plan.has_path or plan.desired_heading is None:
                    continue
                grids.append(local_grid.data)
                features.append(global_environment_features(state, observation, goal))
                targets.append(direction_vector(plan.desired_heading))
                headings.append(plan.desired_heading)
                episode_ids.append(episode_id)
                traffic_labels.append(traffic_source.name)

            if len(grids) < samples:
                raise RuntimeError(
                    f"Generated only {len(grids)} valid samples after {attempted} attempts; "
                    "expand the environmental time range or move the route farther offshore."
                )
            split = self._episode_split(np.asarray(episode_ids, dtype=int))
            metadata = {
                "schema_version": 1,
                "teacher": "HarmonicTacticalPlanner",
                "routes": [str(route.source) for route in routes],
                "samples": len(grids),
                "samples_per_episode": samples_per_episode,
                "seed": self.seed,
                "time_start": np.datetime_as_string(lower, unit="s"),
                "time_end": np.datetime_as_string(upper, unit="s"),
                "channels": list(self.grid.CHANNELS),
                "global_feature_names": list(GLOBAL_FEATURE_NAMES),
                "grid_span_metres": self.grid.span,
                "grid_resolution_metres": self.grid.resolution,
                "environmental_bounds": {
                    "latitude": list(latitude_bounds),
                    "longitude": list(longitude_bounds),
                },
                "traffic_sources": [
                    {
                        "name": source.name,
                        "kind": source.kind,
                        "weight": source.weight,
                        "scenario": source.scenario,
                        "path": None if source.source_path is None else str(source.source_path),
                        "elapsed_seconds_max": source.elapsed_seconds_max,
                    }
                    for source in sources
                ],
                "traffic_sample_counts": dict(sorted(Counter(traffic_labels).items())),
                "split_names": {"0": "train", "1": "validation", "2": "test"},
            }
            return ViTDataset(
                grids=np.stack(grids).astype(np.float32),
                global_features=np.stack(features).astype(np.float32),
                targets=np.stack(targets).astype(np.float32),
                split=split,
                teacher_headings=np.asarray(headings, dtype=np.float32),
                traffic_sources=np.asarray(traffic_labels, dtype="U"),
                partitions=np.full(
                    len(grids), partition_name, dtype=f"<U{max(1, len(partition_name))}"
                ),
                metadata=metadata,
            )
        finally:
            for source in locals().get("recorded_ais", {}).values():
                source.close()
            world.close()

    @staticmethod
    def _traffic_schedule(
        sources: tuple[AISTrafficSource, ...],
        samples: int,
        rng: np.random.Generator,
    ) -> tuple[AISTrafficSource, ...]:
        """Create a shuffled schedule with exact largest-remainder allocation."""

        weights = np.asarray([source.weight for source in sources], dtype=float)
        requested = samples * weights / weights.sum()
        counts = np.floor(requested).astype(int)
        remaining = samples - int(counts.sum())
        ranking = sorted(
            range(len(sources)),
            key=lambda index: (-(requested[index] - counts[index]), sources[index].name),
        )
        for index in ranking[:remaining]:
            counts[index] += 1
        schedule = np.repeat(np.arange(len(sources), dtype=int), counts)
        rng.shuffle(schedule)
        return tuple(sources[int(index)] for index in schedule)

    @staticmethod
    def _ais_targets(
        source: AISTrafficSource,
        recorded_ais: dict[str, AIS],
        state: BoatState,
        when: np.datetime64,
        rng: np.random.Generator,
    ) -> tuple:
        """Sample targets without coupling traffic origin to the world object."""

        if source.kind == "none":
            return ()
        if source.kind == "synthetic":
            return tuple(
                SyntheticAIS(source.scenario or "mixed", state.latitude, state.longitude)
                .load()
                .targets(0.0)
            )
        replay = recorded_ais[source.name]
        if replay.time_mode == "utc_datetime":
            query = when
        elif replay.time_mode == "elapsed_seconds":
            query = float(rng.uniform(0.0, source.elapsed_seconds_max))
        else:
            query = None
        return tuple(replay.targets(query))

    @staticmethod
    def _common_time_bounds(world: MaritimeWorld) -> tuple[np.datetime64, np.datetime64]:
        """Return the interval shared by wind, wave, and current forecasts."""

        assert world.weather is not None and world.weather.dataset is not None
        assert world.waves is not None and world.waves._dataset is not None
        assert world.currents is not None and world.currents._dataset is not None
        axes = (
            np.asarray(world.weather.dataset.valid_time.values).astype("datetime64[ns]"),
            np.asarray(world.waves._dataset.time.values).astype("datetime64[ns]"),
            np.asarray(world.currents._dataset.time.values).astype("datetime64[ns]"),
        )
        return max(axis.min() for axis in axes), min(axis.max() for axis in axes)

    @staticmethod
    def _sample_time(
        lower: np.datetime64,
        upper: np.datetime64,
        rng: np.random.Generator,
    ) -> np.datetime64:
        start = lower.astype("datetime64[ns]").astype(np.int64)
        end = upper.astype("datetime64[ns]").astype(np.int64)
        return np.datetime64(int(rng.integers(start, end + 1)), "ns")

    def _environmental_bounds(
        self,
        routes: tuple[Route, ...],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return a padded regional domain for memory-efficient forecast replay."""

        latitudes = [waypoint.latitude for route in routes for waypoint in route.waypoints]
        longitudes = [waypoint.longitude for route in routes for waypoint in route.waypoints]
        if not latitudes or not longitudes:
            raise ValueError("ViT routes must contain at least one waypoint.")
        margin = self.environmental_margin_degrees
        return (
            (max(-90.0, min(latitudes) - margin), min(90.0, max(latitudes) + margin)),
            (min(longitudes) - margin, max(longitudes) + margin),
        )

    @staticmethod
    def _sample_route_state(
        route: Route,
        rng: np.random.Generator,
    ) -> tuple[float, float, Waypoint]:
        """Draw a position on a route leg and retain that leg's endpoint as goal."""

        legs = np.asarray(
            [
                haversine_metres(start.latitude, start.longitude, end.latitude, end.longitude)
                for start, end in zip(route.waypoints, route.waypoints[1:])
            ],
            dtype=float,
        )
        if not np.any(legs > 0.0):
            raise ValueError(f"Route {route.name!r} has no non-zero legs.")
        index = int(rng.choice(len(legs), p=legs / legs.sum()))
        start, goal = route.waypoints[index], route.waypoints[index + 1]
        # Do not sample immediately on either route point: the active goal
        # must produce a non-zero local direction label.
        fraction = float(rng.uniform(0.05, 0.90))
        latitude = start.latitude + fraction * (goal.latitude - start.latitude)
        longitude = start.longitude + fraction * (goal.longitude - start.longitude)
        return float(latitude), float(longitude), goal

    @staticmethod
    def _episode_split(episode_ids: np.ndarray) -> np.ndarray:
        """Assign 70/15/15 train/validation/test partitions at episode level."""

        unique = np.unique(episode_ids)
        if unique.size < 3:
            # Tiny smoke-test datasets still expose each partition.
            ranks = np.arange(episode_ids.size)
            return np.where(ranks % 10 < 7, 0, np.where(ranks % 10 < 9, 1, 2)).astype(np.int8)
        train_stop = max(1, int(np.floor(unique.size * 0.70)))
        validation_stop = max(train_stop + 1, int(np.floor(unique.size * 0.85)))
        validation_stop = min(validation_stop, unique.size - 1)
        split_by_episode = {
            episode: 0 if rank < train_stop else 1 if rank < validation_stop else 2
            for rank, episode in enumerate(unique)
        }
        return np.asarray([split_by_episode[episode] for episode in episode_ids], dtype=np.int8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a forecast-replayed occupancy-grid dataset for Vision Transformer training."
    )
    parser.add_argument(
        "--route-gpx",
        action="append",
        required=True,
        help="OpenCPN GPX route; repeat the option to combine routes.",
    )
    parser.add_argument("--output", required=True, help="Output .npz dataset path.")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--samples-per-episode", type=int, default=4)
    parser.add_argument("--time-start", help="Optional UTC start constrained to local forecast coverage.")
    parser.add_argument("--time-end", help="Optional UTC end constrained to local forecast coverage.")
    parser.add_argument("--grid-span", type=float, default=400.0)
    parser.add_argument("--grid-resolution", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=7)
    arguments = parser.parse_args()
    if arguments.samples < 3:
        parser.error("--samples must be at least 3.")

    routes = tuple(Route.load_gpx(path) for path in arguments.route_gpx)
    builder = EnvironmentalViTDatasetBuilder(
        grid_span_metres=arguments.grid_span,
        grid_resolution_metres=arguments.grid_resolution,
        seed=arguments.seed,
    )
    dataset = builder.build(
        routes,
        samples=arguments.samples,
        samples_per_episode=arguments.samples_per_episode,
        time_start=arguments.time_start,
        time_end=arguments.time_end,
    )
    output = dataset.save(arguments.output)
    split_counts = np.bincount(dataset.split, minlength=3)
    print(
        f"Saved {len(dataset.grids)} environmental ViT samples to {output}. "
        f"train/validation/test={tuple(int(value) for value in split_counts)}; "
        f"grid={dataset.grids.shape[1:]}; global_features={dataset.global_features.shape[1]}."
    )


if __name__ == "__main__":
    main()
