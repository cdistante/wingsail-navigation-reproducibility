"""Compare next-hour forecasts at the same causal archive knots and route legs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gzip
from hashlib import file_digest
from io import TextIOWrapper
import json
from pathlib import Path
import sys

import numpy as np


STUDY = Path(__file__).resolve().parents[1]
REPOSITORY = STUDY.parents[2]
SEASONAL = STUDY.with_name("seasonal_native_training")
ENCODER_STUDY = STUDY.with_name("learned_weather_ais_policy")
MANIFEST = SEASONAL / "manifest_expanded.json"
ACQUISITIONS = SEASONAL / "forcing" / "acquisition_index.json"
ENCODER_SHA256 = "cf467bcf85a08af99c9819f13b04e65b57f34006ff3957cc42646bba04bc0972"
sys.path[:0] = [str(REPOSITORY), str(SEASONAL / "reproduce"), str(ENCODER_STUDY / "reproduce"),
                str(STUDY.with_name("learned_safe_mark_policy") / "reproduce")]

from seasonal_environment import SeasonalArchive
from weather_embedding import WeatherEmbedding
from world._geo import local_east_north_metres


TRAIN_BLOCKS = ("winter", "summer")
VALIDATION_BLOCK = "spring"
TEST_BLOCK = "autumn"
HOUR_NS = 3_600_000_000_000
HISTORY_STEPS = 12
CHANNELS = ("wind_e_mps", "wind_n_mps", "current_e_mps", "current_n_mps", "wave_height_m")
METHODS = ("persistence", "causal_trend", "transformer")
TREND_MULTIPLIERS = (0., .25, .5, .75, 1.)
RIDGES = (.1, 1., 10., 100., 1000.)
CHANGE_CLIP_QUANTILE = .98
OPPOSING_CURRENT_MPS = -.15
WEAK_WIND_MPS = 4.


@dataclass(frozen=True)
class Leg:
    name: str
    family: str
    origin: tuple[float, float]
    goal: tuple[float, float]
    axis: tuple[float, float]


@dataclass(frozen=True)
class Rows:
    legs: tuple[Leg, ...]
    issue_ns: np.ndarray
    leg_index: np.ndarray
    history: np.ndarray
    target: np.ndarray

    @property
    def anchor(self):
        return self.history[:, -1]

    @property
    def lag(self):
        return self.history[:, 0]

    @property
    def axes(self):
        return np.asarray([leg.axis for leg in self.legs])[self.leg_index]


def digest(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def json_file(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def registered_legs(manifest):
    legs = {}
    for name, route in manifest["routes"].items():
        if digest(REPOSITORY / route["route_gpx"]) != route["sha256"]:
            raise ValueError(f"Registered route changed on disk: {name}.")
        for first, next_point in zip(route["waypoints"], route["waypoints"][1:]):
            origin = first["latitude"], first["longitude"]
            goal = next_point["latitude"], next_point["longitude"]
            key = origin, goal
            if key in legs:
                continue
            axis = local_east_north_metres(*goal, *origin)
            legs[key] = Leg(f"{name}:{first['waypoint']}–{next_point['waypoint']}", route["family"],
                            origin, goal, axis)
    return tuple(legs.values())


def utc_ns(stamp):
    return int(np.datetime64(stamp.removesuffix("Z"), "ns").astype(np.int64))


def utc_string(stamp):
    return str(np.datetime64(int(stamp), "ns").astype("datetime64[s]")) + "Z"


def read_block(manifest, acquisitions, block, legs):
    spec = manifest["archive_blocks"][block]
    files = {kind: SEASONAL / path for kind, path in spec["files"].items()}
    if len({path.parent for path in files.values()}) != 1:
        raise ValueError("One block must supply a common forcing archive directory.")
    archived = acquisitions["seasons"][block]
    receipt_path = SEASONAL / archived["path"]
    if digest(receipt_path) != archived["sha256"]:
        raise ValueError(f"The pinned acquisition receipt changed for {block}.")
    archived_files = json.loads(receipt_path.read_text(encoding="utf-8"))["data"]
    hashes = {name: digest(path) for name, path in files.items()}
    if any(value != archived_files[name]["sha256"] for name, value in hashes.items()):
        raise ValueError(f"The pinned weather fields changed for {block}.")
    archive = SeasonalArchive(files["wind"].parent)
    grids = (archive._wind, archive._current, archive._wave)
    times = np.arange(utc_ns(spec["start_utc"]), utc_ns(spec["end_utc_exclusive"]), HOUR_NS, dtype=np.int64)
    knots = np.r_[times[0] - HOUR_NS, times, times[-1] + HOUR_NS]
    if any(not np.isin(knots, grid.times).all() for grid in grids):
        raise ValueError(f"A product lacks a required common hourly knot in {block}.")
    by_leg = [np.stack([archive.sample(*leg.origin, int(stamp)).vector() for stamp in knots]) for leg in legs]
    fields = np.stack(by_leg)
    if not np.isfinite(fields).all():
        raise ValueError(f"Nonfinite forcing at a registered route-leg origin in {block}.")
    # At an issue exactly on an hourly knot, previous five-minute tokens all
    # hold the preceding knot. Only the last token sees the current knot.
    lag = np.broadcast_to(fields[:, :-2, None, :], (len(legs), len(times), HISTORY_STEPS - 1, len(CHANNELS)))
    current = fields[:, 1:-1, None, :]
    history = np.concatenate((lag, current), axis=2).reshape(-1, HISTORY_STEPS, len(CHANNELS))
    targets = fields[:, 2:].reshape(-1, len(CHANNELS))
    rows = Rows(legs, np.tile(times, len(legs)), np.repeat(np.arange(len(legs)), len(times)), history, targets)
    receipt = {"count": int(len(targets)), "day_count": int(len(times) // 24), "coastal_fallbacks": archive.coastal_fallbacks(),
               "issue_first_utc": utc_string(times[0]), "issue_last_utc": utc_string(times[-1]),
               "target_last_utc": utc_string(times[-1] + HOUR_NS),
               "files_sha256": hashes}
    return rows, receipt


def leg_basis(axes):
    along = axes / np.linalg.norm(axes, axis=1, keepdims=True)
    across = np.column_stack((along[:, 1], -along[:, 0]))
    return np.stack((along, across), axis=2)


def into_leg(values, axes):
    result = values.copy()
    basis = leg_basis(axes)
    for interval in (slice(0, 2), slice(2, 4)):
        result[:, interval] = np.einsum("ni,nij->nj", values[:, interval], basis)
    return result


def into_en(values, axes):
    result = values.copy()
    basis = leg_basis(axes)
    for interval in (slice(0, 2), slice(2, 4)):
        result[:, interval] = np.einsum("nj,nij->ni", values[:, interval], basis)
    return result


def cap_changes(delta, caps):
    result = delta.copy()
    for interval, limit in zip((slice(0, 2), slice(2, 4)), caps[:2]):
        length = np.linalg.norm(delta[:, interval], axis=1)
        factor = np.minimum(1., limit / np.maximum(length, 1e-12))
        result[:, interval] *= factor[:, None]
    result[:, 4] = np.clip(delta[:, 4], -caps[2], caps[2])
    return result


def add_changes(anchor, delta, caps):
    result = anchor + cap_changes(delta, caps)
    result[:, 4] = np.maximum(0., result[:, 4])
    return result


def score(expected, predicted, scales):
    error = expected - predicted
    wind = np.mean(np.sum(error[:, :2]**2, axis=1)) / scales[0]**2
    current = np.mean(np.sum(error[:, 2:4]**2, axis=1)) / scales[1]**2
    return float((wind + current) / 2.)


def design(matrix, mean, scale):
    return np.column_stack((np.ones(len(matrix)), (matrix - mean) / scale))


def select_models(training, validation, encoder):
    train_history = np.concatenate([row.history for row in training])
    train_axes = np.concatenate([row.axes for row in training])
    train_delta = np.concatenate([row.target - row.anchor for row in training])
    magnitudes = np.column_stack((np.linalg.norm(train_delta[:, :2], axis=1),
                                  np.linalg.norm(train_delta[:, 2:4], axis=1), np.abs(train_delta[:, 4])))
    caps = np.quantile(magnitudes, CHANGE_CLIP_QUANTILE, axis=0)
    scales = np.sqrt(np.mean(magnitudes[:, :2]**2, axis=0))
    if np.any(scales <= 0.0):
        raise ValueError("Training persistence error must be nonzero for both vector targets.")

    trend_rates = []
    for multiplier in TREND_MULTIPLIERS:
        estimate = add_changes(validation.anchor, multiplier * (validation.anchor - validation.lag), caps)
        trend_rates.append((score(validation.target, estimate, scales), multiplier))
    _, multiplier = min(trend_rates)

    training_z = encoder.encode_histories(train_history, train_axes).astype(float)
    validation_z = encoder.encode_histories(validation.history, validation.axes).astype(float)
    mean = training_z.mean(axis=0)
    scale = np.maximum(training_z.std(axis=0), .05)
    fit = design(training_z, mean, scale)
    valid = design(validation_z, mean, scale)
    change = into_leg(train_delta, train_axes)
    penalty = np.eye(fit.shape[1])
    penalty[0, 0] = 0.
    candidates = []
    for ridge in RIDGES:
        weights = np.linalg.solve(fit.T @ fit + ridge * penalty, fit.T @ change)
        vector = into_en(valid @ weights, validation.axes)
        estimate = add_changes(validation.anchor, vector, caps)
        candidates.append((score(validation.target, estimate, scales), ridge, weights))
    _, ridge, weights = min(candidates, key=lambda item: item[:2])

    settings = {"trend_multiplier": multiplier, "ridge": ridge, "change_caps": caps.tolist(),
                "training_persistence_vector_rms_scales": scales.tolist(),
                "validation_objective": "Mean wind and current squared vector error, each normalized by January/July persistence MSE.",
                "validation_persistence_score": score(validation.target, validation.anchor, scales),
                "validation_trend_candidates": [{"alpha": value, "score": loss} for loss, value in trend_rates],
                "validation_ridge_candidates": [{"penalty": value, "score": loss} for loss, value, _ in candidates]}
    return settings, {"embedding_mean": mean, "embedding_scale": scale, "weights": weights}


def predictions(rows, encoder, settings, head):
    bias = (rows.anchor - rows.lag) * settings["trend_multiplier"]
    embedding = encoder.encode_histories(rows.history, rows.axes).astype(float)
    learned = design(embedding, head["embedding_mean"], head["embedding_scale"]) @ head["weights"]
    caps = settings["change_caps"]
    return {"persistence": rows.anchor, "causal_trend": add_changes(rows.anchor, bias, caps),
            "transformer": add_changes(rows.anchor, into_en(learned, rows.axes), caps)}


def row_regimes(rows):
    truth = into_leg(rows.target, rows.axes)
    speed = np.linalg.norm(rows.target[:, :2], axis=1)
    opposing = truth[:, 2] <= OPPOSING_CURRENT_MPS
    return {"all": np.ones(len(rows.target), dtype=bool), "opposing_current": opposing,
            "headwind": (truth[:, 0] <= -.5 * speed) & (speed > 0.),
            "weak_wind_opposing_current": (speed < WEAK_WIND_MPS) & opposing}


def error_metrics(rows, predicted, mask):
    count = int(mask.sum())
    if not count:
        return {"count": 0}
    diff = predicted[mask] - rows.target[mask]
    along_cross = into_leg(diff, rows.axes[mask])
    result = {"count": count}
    fields = (("wind", slice(0, 2)), ("current", slice(2, 4)))
    for name, interval in fields:
        vector = diff[:, interval]
        goal = along_cross[:, interval]
        result[f"{name}_en_vector_rmse_mps"] = float(np.sqrt(np.mean(np.sum(vector**2, axis=1))))
        result[f"{name}_along_leg_mae_mps"] = float(np.mean(np.abs(goal[:, 0])))
        result[f"{name}_along_leg_bias_mps"] = float(goal[:, 0].mean())
        result[f"{name}_cross_leg_mae_mps"] = float(np.mean(np.abs(goal[:, 1])))
    result["wave_height_mae_m"] = float(np.mean(np.abs(diff[:, 4])))
    return result


def summary(rows, estimates, block):
    regimes = row_regimes(rows)
    groups = {**regimes}
    family = np.asarray([leg.family for leg in rows.legs])[rows.leg_index]
    groups.update({name: family == name for name in sorted(set(family))})
    report = {name: {m: error_metrics(rows, p, mask) for m, p in estimates.items()} for name, mask in groups.items()}
    days = rows.issue_ns.astype("datetime64[ns]").astype("datetime64[D]")
    daily = []
    for day in np.unique(days):
        for name, regime in regimes.items():
            mask = (days == day) & regime
            for method, predicted in estimates.items():
                daily.append({"block": block, "issue_day_utc": str(day), "regime": name, "method": method,
                              **error_metrics(rows, predicted, mask)})
    return report, daily


def write_records(path, datasets):
    base_fields = ["block", "split", "issue_utc", "target_utc", "leg", "family", "latitude", "longitude",
                   "along_axis_east", "along_axis_north", "opposing_current", "headwind", "weak_wind_opposing_current"]
    prefixes = ("previous_hour", "issue", "target", *METHODS)
    fields = base_fields + [f"{prefix}_{key}" for prefix in prefixes for key in CHANNELS]
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
        with TextIOWrapper(zipped, newline="", encoding="utf-8") as stream:
            sink = csv.writer(stream)
            sink.writerow(fields)
            for block, split, rows, estimates in datasets:
                regimes = row_regimes(rows)
                along = leg_basis(rows.axes)[:, :, 0]
                for i, stamp in enumerate(rows.issue_ns):
                    leg = rows.legs[rows.leg_index[i]]
                    vectors = (rows.lag[i], rows.anchor[i], rows.target[i], *(estimates[arm][i] for arm in METHODS))
                    sink.writerow((block, split, utc_string(stamp), utc_string(stamp + HOUR_NS), leg.name, leg.family,
                                   *leg.origin, *along[i], *(int(regimes[name][i]) for name in base_fields[-3:]),
                                   *np.concatenate(vectors)))


def run(output):
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    acquisitions = json.loads(ACQUISITIONS.read_text(encoding="utf-8"))
    specs = manifest["archive_blocks"]
    assert all(specs[block]["split"] == "train" for block in TRAIN_BLOCKS)
    assert specs[VALIDATION_BLOCK]["split"] == "validation" and specs[TEST_BLOCK]["split"] == "test"
    legs = registered_legs(manifest)
    encoder = WeatherEmbedding()
    if encoder.checkpoint_sha256 != ENCODER_SHA256:
        raise ValueError("The pinned frozen weather encoder changed.")
    loaded = {block: read_block(manifest, acquisitions, block, legs) for block in (*TRAIN_BLOCKS, VALIDATION_BLOCK)}
    settings, head = select_models([loaded[block][0] for block in TRAIN_BLOCKS], loaded[VALIDATION_BLOCK][0], encoder)
    output.mkdir(parents=True, exist_ok=True)
    json_file(output / "selection_receipt.json", {"manifest_sha256": digest(MANIFEST), "acquisition_index_sha256": digest(ACQUISITIONS),
              "encoder_sha256": encoder.checkpoint_sha256, "prediction_horizon_s": 3600,
              "history_steps": HISTORY_STEPS, "history_cadence_s": 300, "change_clip_quantile": CHANGE_CLIP_QUANTILE,
              "training_blocks": TRAIN_BLOCKS, "validation_block": VALIDATION_BLOCK, "settings": settings,
              "pretest_blocks": {block: data[1] for block, data in loaded.items()}})
    np.savez_compressed(output / "forcing_ridge_head.npz", **head)

    # Read the test fields only after selecting and saving all learned settings.
    loaded[TEST_BLOCK] = read_block(manifest, acquisitions, TEST_BLOCK, legs)
    records, aggregated, daily = [], {}, []
    for block, (rows, receipt) in loaded.items():
        estimates = predictions(rows, encoder, settings, head)
        records.append((block, specs[block]["split"], rows, estimates))
        metrics, day_rows = summary(rows, estimates, block)
        aggregated[block] = {"archive": receipt, "metrics": metrics}
        daily.extend(day_rows)
    daily_metrics = list(next(row for row in daily if row["count"] > 0))
    with (output / "daily_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        sink = csv.DictWriter(stream, fieldnames=daily_metrics)
        sink.writeheader()
        sink.writerows(daily)
    write_records(output / "prediction_records.csv.gz", records)
    json_file(output / "metrics.json", {"leg_count": len(legs), "leg_positions": [leg.__dict__ for leg in legs],
              "thresholds": {"opposing_current_mps": OPPOSING_CURRENT_MPS, "weak_wind_mps": WEAK_WIND_MPS,
                             "headwind_to_bearing_cosine_at_most": -.5}, "blocks": aggregated})


def main():
    import torch

    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--output", type=Path, default=STUDY)
    args = command.parse_args()
    torch.set_num_threads(1)
    run(args.output)


if __name__ == "__main__":
    main()
