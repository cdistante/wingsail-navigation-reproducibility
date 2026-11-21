"""Build a route/time-separated environmental ViT experiment from YAML.

The simple ``world.vit_dataset`` command remains useful for a quick local
baseline.  This module is the reproducible experiment entry point: it builds
named partitions with explicit route and weather-window separation, balances
synthetic and recorded AIS sources exactly, and stores input hashes alongside
the dataset.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from config import BATHYMETRY, COASTLINE, CURRENTS, WAVES, WEATHER

from ._geo import normalise_time
from .scenario import Route
from .vit_dataset import (
    AISTrafficSource,
    EnvironmentalViTDatasetBuilder,
    GLOBAL_FEATURE_NAMES,
    ViTDataset,
)


SPLIT_CODES = {"train": 0, "validation": 1, "test": 2}


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def input_manifest(
    root: Path,
    *,
    route_paths: list[Path],
    recorded_ais_paths: list[Path],
    config_path: Path,
    include_hashes: bool,
) -> list[dict[str, object]]:
    """Capture the exact data/route/config inputs used to build an archive."""

    files = [config_path, *route_paths, *recorded_ais_paths]
    for directory, patterns in (
        (WEATHER, ("*.nc",)),
        (WAVES, ("*.nc",)),
        (CURRENTS, ("*.nc",)),
        (BATHYMETRY, ("*.tif", "*.tiff")),
        (COASTLINE, ("*.geojson",)),
    ):
        for pattern in patterns:
            files.extend(sorted(Path(directory).glob(pattern)))
    unique = sorted({path.resolve() for path in files})
    missing = [path for path in unique if not path.exists()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise RuntimeError(f"Experiment inputs are missing: {names}.")
    return [
        {
            "path": _relative(path, root),
            "bytes": path.stat().st_size,
            **({"sha256": _sha256(path)} if include_hashes else {}),
        }
        for path in unique
    ]


def _allocate(total: int, groups: int) -> list[int]:
    """Split an integer count as evenly as possible over a fixed number of groups."""

    if total < 1 or groups < 1:
        raise ValueError("Sample allocation requires positive counts.")
    base, remainder = divmod(total, groups)
    return [base + (1 if index < remainder else 0) for index in range(groups)]


def _windows(value: Any, name: str) -> tuple[dict[str, str], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Partition {name!r} must define a non-empty time_windows list.")
    result = []
    for window in value:
        if not isinstance(window, dict) or "start" not in window or "end" not in window:
            raise ValueError(f"Every {name!r} time window needs start and end UTC timestamps.")
        start, end = normalise_time(window["start"]), normalise_time(window["end"])
        if start >= end:
            raise ValueError(f"Partition {name!r} has a non-positive time window.")
        result.append({"start": str(window["start"]), "end": str(window["end"])})
    return tuple(result)


def _intervals_overlap(first: dict[str, str], second: dict[str, str]) -> bool:
    return normalise_time(first["start"]) < normalise_time(second["end"]) and normalise_time(
        second["start"]
    ) < normalise_time(first["end"])


def _traffic_sources(value: Any, root: Path, partition: str) -> tuple[AISTrafficSource, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Partition {partition!r} must define non-empty traffic_sources.")
    sources = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"Traffic sources in {partition!r} must be mappings.")
        kind = str(item.get("kind", ""))
        path = item.get("path")
        if path is not None:
            path = root / str(path)
            if not path.exists():
                raise RuntimeError(
                    f"Recorded AIS source for partition {partition!r} is missing: {path}. "
                    "Place a processed CSV in data/ais/processed and update the experiment YAML."
                )
        sources.append(
            AISTrafficSource(
                name=str(item.get("name", "")),
                kind=kind,
                weight=float(item.get("weight", 1.0)),
                scenario=item.get("scenario"),
                path=path,
                elapsed_seconds_max=float(item.get("elapsed_seconds_max", 3_600.0)),
            )
        )
    return tuple(sources)


def _partition_routes(value: Any, root: Path, partition: str) -> tuple[Route, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Partition {partition!r} must define one or more route_gpx paths.")
    paths = [root / str(path) for path in value]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Partition {partition!r} route file is missing: {missing[0]}.")
    return tuple(Route.load_gpx(path) for path in paths)


def _validate_separation(partitions: dict[str, dict[str, Any]], root: Path) -> None:
    """Fail early when a declared route/time holdout leaks training inputs."""

    training = [item for item in partitions.values() if item["split"] == "train"]
    if not training:
        raise ValueError("An experiment requires at least one training partition.")
    train_routes = {
        str((root / route).resolve())
        for item in training
        for route in item["route_gpx"]
    }
    train_windows = [window for item in training for window in item["time_windows"]]
    for name, item in partitions.items():
        declared = set(item.get("separation", ()))
        if not declared:
            continue
        if not declared.issubset({"route", "time"}):
            raise ValueError(f"Partition {name!r} separation accepts only route and/or time.")
        if "route" in declared:
            routes = {str((root / route).resolve()) for route in item["route_gpx"]}
            if routes & train_routes:
                raise ValueError(f"Partition {name!r} declares route separation but shares a train route.")
        if "time" in declared:
            if any(
                _intervals_overlap(window, train_window)
                for window in item["time_windows"]
                for train_window in train_windows
            ):
                raise ValueError(f"Partition {name!r} declares time separation but overlaps train time.")


def build_experiment(config_path: Path | str, *, include_hashes: bool = True) -> Path:
    """Build and save one fully provenance-tracked route/time/AIS ViT dataset."""

    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise RuntimeError(f"Experiment configuration does not exist: {config_path}.")
    root = Path.cwd().resolve()
    with config_path.open() as source:
        raw = yaml.safe_load(source) or {}
    if not isinstance(raw, dict):
        raise ValueError("The experiment YAML root must be a mapping.")
    if int(raw.get("schema_version", 0)) != 1:
        raise ValueError("This runner supports experiment schema_version: 1 only.")
    dataset_config = raw.get("dataset")
    partition_config = raw.get("partitions")
    if not isinstance(dataset_config, dict) or not isinstance(partition_config, dict):
        raise ValueError("Experiment YAML requires dataset and partitions mappings.")
    if not partition_config:
        raise ValueError("Experiment YAML must define at least one partition.")

    normalised: dict[str, dict[str, Any]] = {}
    for name, item in partition_config.items():
        if not isinstance(item, dict):
            raise ValueError(f"Partition {name!r} must be a mapping.")
        split = str(item.get("split", ""))
        if split not in SPLIT_CODES:
            raise ValueError(f"Partition {name!r} must choose split train, validation, or test.")
        normalised[str(name)] = {
            "split": split,
            "samples": int(item.get("samples", 0)),
            "route_gpx": list(item.get("route_gpx", ())),
            "time_windows": _windows(item.get("time_windows"), str(name)),
            "traffic_sources": item.get("traffic_sources"),
            "separation": tuple(item.get("separation", ())),
        }
        if normalised[str(name)]["samples"] < 1:
            raise ValueError(f"Partition {name!r} samples must be positive.")
    _validate_separation(normalised, root)

    grid_span = float(dataset_config.get("grid_span_metres", 400.0))
    grid_resolution = float(dataset_config.get("grid_resolution_metres", 10.0))
    samples_per_episode = int(dataset_config.get("samples_per_episode", 8))
    max_attempt_multiplier = int(dataset_config.get("maximum_attempt_multiplier", 40))
    environmental_margin = float(dataset_config.get("environmental_margin_degrees", 1.0))
    seed = int(raw.get("seed", 7))
    output = root / str(dataset_config.get("output", "data/training/vit_experiment.npz"))

    grids = []
    features = []
    targets = []
    splits = []
    headings = []
    traffic_labels = []
    partition_labels = []
    source_metadata = []
    used_routes: list[Path] = []
    used_recordings: list[Path] = []
    bounds: list[dict[str, object]] = []
    for partition_index, (name, item) in enumerate(normalised.items()):
        routes = _partition_routes(item["route_gpx"], root, name)
        traffic = _traffic_sources(item["traffic_sources"], root, name)
        used_routes.extend(route.source for route in routes)
        used_recordings.extend(
            source.source_path for source in traffic if source.source_path is not None
        )
        allocation = _allocate(item["samples"], len(item["time_windows"]))
        for window_index, (window, samples) in enumerate(zip(item["time_windows"], allocation)):
            builder = EnvironmentalViTDatasetBuilder(
                grid_span_metres=grid_span,
                grid_resolution_metres=grid_resolution,
                environmental_margin_degrees=environmental_margin,
                maximum_attempt_multiplier=max_attempt_multiplier,
                seed=seed + partition_index * 10_000 + window_index,
            )
            generated = builder.build(
                routes,
                samples=samples,
                samples_per_episode=samples_per_episode,
                time_start=window["start"],
                time_end=window["end"],
                traffic_sources=traffic,
                partition_name=name,
            )
            grids.append(generated.grids)
            features.append(generated.global_features)
            targets.append(generated.targets)
            splits.append(np.full(len(generated.grids), SPLIT_CODES[item["split"]], dtype=np.int8))
            headings.append(generated.teacher_headings)
            traffic_labels.append(generated.traffic_sources)
            partition_labels.append(generated.partitions)
            bounds.append(generated.metadata["environmental_bounds"])
        source_metadata.append(
            {
                "name": name,
                "split": item["split"],
                "samples": item["samples"],
                "route_gpx": item["route_gpx"],
                "time_windows": item["time_windows"],
                "separation": list(item["separation"]),
                "traffic_sources": [
                    {
                        "name": source.name,
                        "kind": source.kind,
                        "weight": source.weight,
                        "scenario": source.scenario,
                        "path": None if source.source_path is None else _relative(source.source_path, root),
                    }
                    for source in traffic
                ],
            }
        )

    merged_bounds = {
        "latitude": [
            min(float(bound["latitude"][0]) for bound in bounds),
            max(float(bound["latitude"][1]) for bound in bounds),
        ],
        "longitude": [
            min(float(bound["longitude"][0]) for bound in bounds),
            max(float(bound["longitude"][1]) for bound in bounds),
        ],
    }
    traffic_array = np.concatenate(traffic_labels).astype("U")
    partition_array = np.concatenate(partition_labels).astype("U")
    metadata = {
        "schema_version": 2,
        "experiment_name": str(raw.get("name", config_path.stem)),
        "experiment_config": raw,
        "teacher": "HarmonicTacticalPlanner",
        "channels": list(EnvironmentalViTDatasetBuilder().grid.CHANNELS),
        "global_feature_names": list(GLOBAL_FEATURE_NAMES),
        "grid_span_metres": grid_span,
        "grid_resolution_metres": grid_resolution,
        "environmental_bounds": merged_bounds,
        "partitions": source_metadata,
        "split_names": {"0": "train", "1": "validation", "2": "test"},
        "traffic_sample_counts": dict(sorted(Counter(traffic_array.tolist()).items())),
        "partition_sample_counts": dict(sorted(Counter(partition_array.tolist()).items())),
        "input_manifest": input_manifest(
            root,
            route_paths=used_routes,
            recorded_ais_paths=used_recordings,
            config_path=config_path,
            include_hashes=include_hashes,
        ),
    }
    dataset = ViTDataset(
        grids=np.concatenate(grids).astype(np.float32),
        global_features=np.concatenate(features).astype(np.float32),
        targets=np.concatenate(targets).astype(np.float32),
        split=np.concatenate(splits).astype(np.int8),
        teacher_headings=np.concatenate(headings).astype(np.float32),
        traffic_sources=traffic_array,
        partitions=partition_array,
        metadata=metadata,
    )
    return dataset.save(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a route/time-separated, AIS-balanced environmental ViT dataset."
    )
    parser.add_argument("--config", required=True, help="Experiment YAML, interpreted from the repository root.")
    parser.add_argument(
        "--skip-input-hashes",
        action="store_true",
        help="Skip SHA-256 digests in the provenance manifest (faster, less auditable).",
    )
    arguments = parser.parse_args()
    output = build_experiment(arguments.config, include_hashes=not arguments.skip_input_hashes)
    with np.load(output, allow_pickle=False) as archive:
        split = np.asarray(archive["split"], dtype=np.int8)
        partitions = np.asarray(archive["partitions"], dtype=str)
        traffic = np.asarray(archive["traffic_sources"], dtype=str)
    print(
        "Saved {0} samples to {1}; train/validation/test={2}; partitions={3}; traffic={4}.".format(
            len(split),
            output,
            tuple(int(value) for value in np.bincount(split, minlength=3)),
            dict(sorted(Counter(partitions.tolist()).items())),
            dict(sorted(Counter(traffic.tolist()).items())),
        )
    )


if __name__ == "__main__":
    main()
