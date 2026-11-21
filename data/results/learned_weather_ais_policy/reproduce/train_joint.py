"""Train and test report-guarded time-to-mark residual policies on disjoint splits."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
from typing import NamedTuple

import numpy as np

from environment import MAX_SECONDS, TIMEOUT_COST_SECONDS, enforce_first_entry, load_scenarios
from environment_joint import REPORT_FEATURE_NAMES, assess_applied, evaluate, prepare
from world.learned_mark_policy import FEATURE_NAMES


STUDY = Path(__file__).resolve().parents[1]
PREVIOUS = STUDY.with_name("learned_safe_mark_policy")
FAILURE_COST = MAX_SECONDS + TIMEOUT_COST_SECONDS
EMBEDDING_WIDTH = 48
REPORT_WIDTH = len(REPORT_FEATURE_NAMES)
EXPLICIT_WIDTH = len(FEATURE_NAMES)
WIDTH = EXPLICIT_WIDTH + REPORT_WIDTH + EMBEDDING_WIDTH
SCALE_S = 1200.0
MAX_CORRECTION = 6.0
BRANCHES = 6
SAMPLE_TIMES = (0., 60., 120., 180., 300., 600., 1800., 5400., 10800., 11100.)
ARMS = {"explicit": (False, False), "weather": (False, True),
        "ais": (True, False), "joint": (True, True)}
STRENGTHS = (0., .025, .075, .15, .30, .60)
SEED = 331019
CASE_INDEX = {}
ENCODER = None
MODELS = {}


class EpisodeKey(NamedTuple):
    scenario: str
    route: str
    weather_block: str
    traffic: str
    blocked_target: bool
    origin: tuple[float, float] | None


@dataclass(frozen=True)
class Model:
    mean: np.ndarray
    scale: np.ndarray
    layers: tuple

    def residual(self, features):
        x = np.clip((features - self.mean) / self.scale, -8., 8.)
        w1, b1, w2, b2, w3, b3 = self.layers
        x = np.tanh(x @ w1.T + b1)
        x = np.tanh(x @ w2.T + b2)
        return MAX_CORRECTION * np.tanh((x @ w3.T + b3)[..., 0])


def inputs():
    split = json.loads((STUDY / "splits.json").read_text())
    for filename, checksum in split["files_sha256"].items():
        if digest(STUDY.parents[2] / filename) != checksum:
            raise ValueError(f"Dataset provenance check failed for {filename}.")
    archive = json.loads((PREVIOUS / "archive_scenario_manifest.json").read_text())
    return split, archive


def episode_keys(subset, manifest):
    result = [EpisodeKey(case, "synthetic", "synthetic", profile, False, None)
              for case in subset["synthetic_ids"] for profile in subset["traffic_profiles"]]
    for case in subset["archive_ids"]:
        row = manifest["rows"][case]
        leg = manifest["leg_blocks"][row["leg"]]
        group = manifest["groups"][row["group"]]
        result.extend(EpisodeKey(case, group["gpx"], group["archive_block"], profile,
                                 leg["target_chart_preflight"]["complete_capture_disk_in_blocked_cell"],
                                 tuple(leg["origin_lat_lon"]))
                      for profile in subset["traffic_profiles"])
    return result


def init_workers(method_names=()):
    global CASE_INDEX, ENCODER, MODELS
    cases = (load_scenarios(PREVIOUS / "scenarios.json.gz")
             + load_scenarios(PREVIOUS / "archive_scenarios.json.gz"))
    CASE_INDEX = {case.identifier: case for case in cases}
    if method_names:
        import torch
        from weather_embedding import WeatherEmbedding

        torch.set_num_threads(1)
        ENCODER = WeatherEmbedding()
        MODELS = {}
        for name in method_names:
            with np.load(STUDY / f"{name}_critic.npz") as checkpoint:
                layers = tuple(checkpoint[f"layer_{i}"] for i in range(6))
                MODELS[name] = Model(checkpoint["mean"], checkpoint["scale"], layers)


def work(function, jobs, workers, *, method_names=()):
    if workers == 1:
        init_workers(method_names)
        yield from map(function, jobs)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=init_workers, initargs=(method_names,)) as executor:
        yield from executor.map(function, jobs, chunksize=3)


def setup(key):
    options = {} if key.origin is None else {"origin_latitude": key.origin[0], "origin_longitude": key.origin[1]}
    return prepare(CASE_INDEX[key.scenario], key.traffic, **options)


def eligible(case, targets):
    distance = np.linalg.norm(targets, axis=1)
    return (case.radius == 50.) & (distance > case.radius) & (distance <= 5500.)


def choose_actions(assessment, scores=None):
    raw = assessment.batch.physics_cost_s if scores is None else scores
    return np.argmin(enforce_first_entry(raw, assessment.batch, assessment.dt, assessment.allowed), axis=1)


def candidate_features(assessment):
    features = np.concatenate((assessment.batch.features, assessment.report_features), axis=-1)
    if features.shape[-1] != EXPLICIT_WIDTH + REPORT_WIDTH or not np.isfinite(features).all():
        raise ValueError("Joint candidate features have incorrect shape or nonfinite values.")
    return features


def alternatives(assessment):
    batch = assessment.batch
    valid = np.flatnonzero(assessment.allowed[0])
    scores = enforce_first_entry(batch.physics_cost_s, batch, assessment.dt, assessment.allowed)[0]
    ranked = valid[np.argsort(scores[valid], kind="stable")]
    # Collapse trims that give the same finite-window motion and heading.
    motion = np.column_stack((batch.displacement[0, ranked], batch.end_bow[0, ranked]))
    _, first = np.unique(np.round(motion, 5), axis=0, return_index=True)
    ranked = ranked[np.sort(first)]
    if len(ranked) < BRANCHES:
        return ranked
    # Keep nearby choices as well as alternatives with different continuation.
    positions = [0, 1, 2, len(ranked) // 3, 2 * len(ranked) // 3, len(ranked) - 1]
    return ranked[np.unique(positions)]


def branch_outcomes(context, first, indices, elapsed):
    """Force one legal first action then roll out the shared physics continuation."""
    mapped = len(indices)
    returned = np.full(mapped, FAILURE_COST)
    remaining = first.batch.remaining_target[0, indices].copy()
    bows = first.batch.end_bow[0, indices].copy()
    hit = first.batch.captured_after_s[0, indices]
    finite = np.isfinite(hit)
    returned[finite] = elapsed + hit[finite]
    alive = np.flatnonzero(~finite)
    now = elapsed + first.dt
    while alive.size and now < MAX_SECONDS:
        step = evaluate(context, now, remaining[alive], bows[alive])
        ok = np.flatnonzero(step.allowed.any(axis=1))
        choices = choose_actions(step)[ok]
        hit = step.batch.captured_after_s[ok, choices]
        finite = np.isfinite(hit)
        returned[alive[ok[finite]]] = now + hit[finite]
        alive = alive[ok[~finite]]
        remaining[alive] = step.batch.remaining_target[ok[~finite], choices[~finite]]
        bows[alive] = step.batch.end_bow[ok[~finite], choices[~finite]]
        now += step.dt
    returned[returned < FAILURE_COST] -= elapsed
    return returned


def collect(key):
    case = CASE_INDEX[key.scenario]
    context = setup(key)
    target, bow, elapsed = case.target.copy(), case.bow, 0.
    sample = 0
    states = []
    while elapsed < MAX_SECONDS and sample < len(SAMPLE_TIMES):
        step = evaluate(context, elapsed, target[None, :], np.array([bow]))
        if not step.allowed.any():
            break
        first = int(choose_actions(step)[0])
        if eligible(case, target[None, :])[0] and elapsed >= SAMPLE_TIMES[sample]:
            sample = int(np.searchsorted(SAMPLE_TIMES, elapsed, side="right"))
            indices = alternatives(step)
            if len(indices) > 1:
                nominal = np.clip(step.batch.physics_cost_s[0, indices], 0., FAILURE_COST)
                actual = branch_outcomes(context, step, indices, elapsed)
                states.append((elapsed, target.copy(), candidate_features(step)[0, indices], nominal, actual))
        if np.isfinite(step.batch.captured_after_s[0, first]):
            break
        target = step.batch.remaining_target[0, first]
        bow = float(step.batch.end_bow[0, first])
        elapsed += step.dt
    return key, states


def fit_data(results):
    from weather_embedding import WeatherEmbedding

    encoder = WeatherEmbedding()
    grouped = defaultdict(list)
    for key, states in results:
        for item in states:
            grouped[key.scenario, item[0]].append((key.traffic, item))
    features, returns, nominal, masks, identifiers, profiles, times, targets = [], [], [], [], [], [], [], []
    for (scenario_id, elapsed), items in grouped.items():
        state_vectors = np.stack([item[1] for _, item in items])
        weather = encoder.encode(CASE_INDEX[scenario_id], elapsed, state_vectors)
        for (profile, (_, target, values, physics, actual)), vector in zip(items, weather, strict=True):
            width = len(values)
            feature = np.zeros((BRANCHES, WIDTH), dtype=np.float32)
            feature[:width] = np.column_stack((values, np.broadcast_to(vector, (width, EMBEDDING_WIDTH))))
            mask = np.arange(BRANCHES) < width
            features.append(feature)
            returns.append(np.pad(actual, (0, BRANCHES - width)))
            nominal.append(np.pad(physics, (0, BRANCHES - width)))
            masks.append(mask)
            identifiers.append(scenario_id)
            profiles.append(profile)
            times.append(elapsed)
            targets.append(target)
    if not features:
        raise RuntimeError("No training counterfactuals were collected.")
    dataset = {name: np.asarray(value) for name, value in
               (("features", features), ("returns_s", returns), ("physics_s", nominal), ("valid", masks),
                ("scenarios", identifiers), ("traffic", profiles), ("state_elapsed_s", times), ("targets_en", targets))}
    np.savez_compressed(STUDY / "training_counterfactuals.npz", **dataset)
    return dataset


def mask_features(features, arm):
    use_ais, use_weather = ARMS[arm]
    values = features.copy()
    if not use_ais:
        values[..., EXPLICIT_WIDTH:EXPLICIT_WIDTH + REPORT_WIDTH] = 0.
    if not use_weather:
        values[..., EXPLICIT_WIDTH + REPORT_WIDTH:] = 0.
    return values


def fit_arm(data, arm):
    import torch
    from torch import nn

    torch.manual_seed(SEED)
    generator = np.random.default_rng(SEED)
    rows = mask_features(data["features"], arm)
    valid = data["valid"]
    # Contrast one-step counterfactual values against the physics action from
    # the same start to avoid regressing a common, action-invariant delay.
    delta = data["returns_s"] - data["physics_s"]
    relative = np.clip((delta - delta[:, :1]) / SCALE_S, -MAX_CORRECTION, MAX_CORRECTION)
    mean = rows[valid].mean(axis=0)
    scale = rows[valid].std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.)
    x = torch.from_numpy(np.clip((rows - mean) / scale, -8., 8.))
    y = torch.from_numpy(relative.astype(np.float32))
    comparisons = valid.copy()
    comparisons[:, 0] = False
    network = nn.Sequential(nn.Linear(WIDTH, 64), nn.Tanh(), nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1))
    nn.init.zeros_(network[-1].weight)
    nn.init.zeros_(network[-1].bias)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3, weight_decay=.004)
    curve = []
    for _ in range(28):
        total, count = 0., 0
        for sample in np.array_split(generator.permutation(len(x)), max(1, len(x)//256)):
            output = MAX_CORRECTION * torch.tanh(network(x[sample])[..., 0])
            mask = comparisons[sample]
            residual = (output - output[:, :1])[mask]
            target = y[sample][mask]
            loss = nn.functional.smooth_l1_loss(residual, target, beta=.25)
            loss += .008 * output[torch.from_numpy(valid[sample])].square().mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 2.)
            optimizer.step()
            total += float(loss.detach()) * int(mask.sum())
            count += int(mask.sum())
        curve.append(total / count)
    weights = {f"layer_{i}": item.detach().numpy() for i, item in enumerate(network.parameters())}
    np.savez_compressed(STUDY / f"{arm}_critic.npz", mean=mean, scale=scale, **weights)
    return curve


def minimum(old, value):
    if not np.isfinite(value):
        return old
    return min(old, float(value)) if old is not None else float(value)


def row_for(key, case, method, strength):
    return {**key._asdict(), "method": method, "strength": strength, "radius_m": case.radius,
            "leg_length_m": float(np.linalg.norm(case.target)), "status": "timeout", "duration_s": MAX_SECONDS,
            "capped_time_s": FAILURE_COST, "capture_distance_m": None, "min_report_margin_m": None,
            "min_truth_margin_m": None, "min_truth_margin_step_s": None,
            "max_report_age_s": 0., "max_dropped_epochs": 0, "stale_steps": 0, "decisions": 0,
            "eligible_decisions": 0, "changed_decisions": 0, "rejected_actions": 0,
            "clearance_confirmed": None, "distance_sailed_m": 0.}


def grade(job):
    key, methods, trace = job
    case = CASE_INDEX[key.scenario]
    rows = [row_for(key, case, name, strength) for name, _, strength in methods]
    if key.blocked_target:
        for row in rows:
            row.update(status="chart_preflight_blocked", duration_s=0.)
        return rows, ()
    context = setup(key)
    targets = np.repeat(case.target[None, :], len(rows), axis=0)
    bows = np.full(len(rows), case.bow)
    alive = np.arange(len(rows))
    elapsed = 0.
    traces = []
    while alive.size and elapsed < MAX_SECONDS:
        step = evaluate(context, elapsed, targets[alive], bows[alive])
        scores = step.batch.physics_cost_s.copy()
        baseline = choose_actions(step)
        applies = eligible(case, targets[alive]) & step.allowed.any(axis=1)
        learned_rows = np.array([applies[i] and methods[number][2] > 0.
                                 for i, number in enumerate(alive)], dtype=bool)
        if learned_rows.any():
            state_features = candidate_features(step)
            use_weather = any(ARMS[methods[number][1]][1] for number in alive[learned_rows])
            weather = ENCODER.encode(case, elapsed, targets[alive[learned_rows]]) if use_weather else None
            width = scores.shape[1]
            for k, i in enumerate(np.flatnonzero(learned_rows)):
                _, arm, strength = methods[alive[i]]
                vector = weather[k] if weather is not None else np.zeros(EMBEDDING_WIDTH, dtype=np.float32)
                full = np.column_stack((state_features[i], np.broadcast_to(vector, (width, EMBEDDING_WIDTH))))
                correction = MODELS[arm].residual(mask_features(full, arm))
                scores[i] += strength * SCALE_S * correction
        choices = choose_actions(step, scores)
        left = []
        for i, idx in enumerate(alive):
            row = rows[idx]
            row["decisions"] += 1
            row["eligible_decisions"] += int(applies[i])
            row["rejected_actions"] += int((~step.allowed[i]).sum())
            if step.assessment is not None:
                age = float(step.assessment.maximum_report_age_seconds)
                row["max_report_age_s"] = max(row["max_report_age_s"], age)
                row["max_dropped_epochs"] = max(row["max_dropped_epochs"], step.assessment.dropped_reports_at_observation)
                row["stale_steps"] += int(age > 10.)
            if not step.allowed[i].any():
                row.update(status="ais_blocked", duration_s=elapsed)
                continue
            selected = int(choices[i])
            row["changed_decisions"] += int(selected != baseline[i])
            hit = float(step.batch.captured_after_s[i, selected])
            duration = min(step.dt, hit)
            # Recover the actual swept endpoint if capture truncates the trim.
            turn = step.batch.turn_displacement[i, selected]
            sailing = step.batch.displacement[i, selected] - turn
            turning = min(step.dt, float(step.batch.features[i, selected, FEATURE_NAMES.index("turn_seconds_180")]) * 180.)
            turn_fraction = min(duration / turning, 1.) if turning > 0. else 0.
            sail_fraction = max(0., duration - turning) / (step.dt - turning) if turning < step.dt else 0.
            turn_part, sail_part = turn_fraction * turn, sail_fraction * sailing
            end_target = targets[idx] - turn_part - sail_part
            row["distance_sailed_m"] += float(np.linalg.norm(turn_part) + np.linalg.norm(sail_part))
            report = truth = None
            if trace and step.assessment is not None:
                report, truth = assess_applied(context, elapsed, targets[alive], step, i, selected, duration)
                prior = row["min_truth_margin_m"]
                row["min_report_margin_m"] = minimum(row["min_report_margin_m"], report)
                row["min_truth_margin_m"] = minimum(prior, truth)
                if prior != row["min_truth_margin_m"]:
                    row["min_truth_margin_step_s"] = elapsed
                row["clearance_confirmed"] = bool(row["min_truth_margin_m"] > 0. and row["min_report_margin_m"] > 0.)
            if trace:
                traces.append((row["method"], elapsed, duration, *map(float, case.target - targets[idx]),
                               *map(float, case.target - end_target), float(bows[idx]),
                               float(step.batch.end_bow[i, selected]), selected, int(baseline[i]), report, truth,
                               None if step.assessment is None else age))
            if np.isfinite(hit):
                row.update(status="captured", duration_s=elapsed + hit, capped_time_s=elapsed + hit,
                           capture_distance_m=float(np.linalg.norm(end_target)))
                continue
            targets[idx] = step.batch.remaining_target[i, selected]
            bows[idx] = step.batch.end_bow[i, selected]
            left.append(idx)
        alive = np.asarray(left, dtype=int)
        elapsed += step.dt
    return rows, traces


def summarize(rows):
    captures = [row["duration_s"] for row in rows if row["status"] == "captured"]
    truth = [row["min_truth_margin_m"] for row in rows if row["min_truth_margin_m"] is not None]
    report = [row["min_report_margin_m"] for row in rows if row["min_report_margin_m"] is not None]
    counts = {status: sum(row["status"] == status for row in rows)
              for status in ("captured", "timeout", "ais_blocked", "chart_preflight_blocked")}
    return {"episodes": len(rows), **counts, "median_capture_time_s": float(np.median(captures)) if captures else None,
            "mean_capped_time_s": float(np.mean([row["capped_time_s"] for row in rows])),
            "min_applied_truth_margin_m": min(truth, default=None),
            "min_applied_report_margin_m": min(report, default=None),
            "applied_episodes_clearance_unconfirmed": sum(row["clearance_confirmed"] is False for row in rows),
            "applied_episodes_audited": len(truth),
            "captured_with_clearance_confirmed": sum(row["status"] == "captured"
                                                    and row["clearance_confirmed"] is True for row in rows),
            "max_report_age_s": max(row["max_report_age_s"] for row in rows),
            "episodes_observing_loss": sum(row["max_dropped_epochs"] > 0 for row in rows),
            "episodes_scored_or_eligible": sum(row["eligible_decisions"] > 0 for row in rows),
            "changed_decisions": sum(row["changed_decisions"] for row in rows)}


def csv_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train(workers):
    split, archive = inputs()
    keys = [key for key in episode_keys(split["training"], archive) if not key.blocked_target]
    results = []
    for index, result in enumerate(work(collect, keys, workers), start=1):
        results.append(result)
        if index % 40 == 0:
            print(f"Collected physics continuations for {index}/{len(keys)} training episodes", flush=True)
    import torch

    torch.set_num_threads(1)
    init_workers()
    data = fit_data(results)
    for arm in ARMS:
        curves = fit_arm(data, arm)
        print(f"Fitted {arm}: residual Huber loss {curves[0]:.4f} to {curves[-1]:.4f}", flush=True)
    validation_keys = episode_keys(split["validation"], archive)
    validation_methods = (("physics", None, 0.),) + tuple(
        (f"{arm}_{strength:g}", arm, strength) for arm in ARMS for strength in STRENGTHS if strength > 0.
    )
    validation = defaultdict(list)
    for index, (rows, _) in enumerate(work(grade, ((key, validation_methods, False) for key in validation_keys),
                                                workers, method_names=tuple(ARMS)), start=1):
        for row in rows:
            validation[row["method"]].append(row)
        if index % 40 == 0:
            print(f"Validated {index}/{len(validation_keys)} episodes", flush=True)
    csv_rows(STUDY / "validation_episodes.csv", [row for rows in validation.values() for row in rows])
    statistics = {name: summarize(rows) for name, rows in validation.items()}
    order = lambda name: (-statistics[name]["captured"], statistics[name]["mean_capped_time_s"])
    strengths = {}
    for arm in ARMS:
        positive = min((strength for strength in STRENGTHS if strength > 0.),
                       key=lambda strength: (*order(f"{arm}_{strength:g}"), strength))
        deploy = positive if order(f"{arm}_{positive:g}") < order("physics") else 0.
        strengths[arm] = {"best_nonzero": positive, "physics_allowed": deploy}
    receipt = {"split_sha256": digest(STUDY / "splits.json"),
               "weather_encoder_sha256": digest(STUDY / "fresh_weather_encoder.pt"),
               "learner_source_sha256": digest(Path(__file__)),
               "guard_source_sha256": digest(Path(__file__).with_name("environment_joint.py")),
               "embedding_source_sha256": digest(Path(__file__).with_name("weather_embedding.py")),
               "feature_schema": (*FEATURE_NAMES, *REPORT_FEATURE_NAMES,
                                  *(f"weather_embedding_{i}" for i in range(EMBEDDING_WIDTH))),
               "models_sha256": {arm: digest(STUDY / f"{arm}_critic.npz") for arm in ARMS},
               "training_branches_sha256": digest(STUDY / "training_counterfactuals.npz"),
               "training_states": int(len(data["valid"])), "training_branches": int(data["valid"].sum()),
               "training_episode_cells": len(keys), "informative_terminal_groups": int(sum(
                   len(np.unique(returns[valid])) > 1 for returns, valid in zip(data["returns_s"], data["valid"]))),
               "validation": statistics, "strengths": strengths}
    (STUDY / "training_receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps(strengths), flush=True)


def paired(rows):
    groups = defaultdict(dict)
    for row in rows:
        groups[row["scenario"], row["traffic"]][row["method"]] = row
    summary = {}
    for method in sorted({row["method"] for row in rows} - {"physics"}):
        pairs = [(values[method], values["physics"]) for values in groups.values()]
        jointly = [a["duration_s"] - b["duration_s"] for a, b in pairs if a["status"] == b["status"] == "captured"]
        summary[method] = {"joint_captures": len(jointly),
                           "median_delta_s": float(np.median(jointly)) if jointly else None,
                           "faster": sum(value < -1e-6 for value in jointly),
                           "slower": sum(value > 1e-6 for value in jointly),
                           "tied": sum(abs(value) <= 1e-6 for value in jointly),
                           "learned_only": sum(a["status"] == "captured" and b["status"] != "captured" for a, b in pairs),
                           "physics_only": sum(b["status"] == "captured" and a["status"] != "captured" for a, b in pairs)}
    return summary


def test(workers):
    split, archive = inputs()
    receipt = json.loads((STUDY / "training_receipt.json").read_text())
    if (digest(STUDY / "splits.json") != receipt["split_sha256"]
            or digest(STUDY / "fresh_weather_encoder.pt") != receipt["weather_encoder_sha256"]
            or digest(Path(__file__)) != receipt["learner_source_sha256"]
            or digest(Path(__file__).with_name("environment_joint.py")) != receipt["guard_source_sha256"]
            or digest(Path(__file__).with_name("weather_embedding.py")) != receipt["embedding_source_sha256"]
            or any(digest(STUDY / f"{arm}_critic.npz") != receipt["models_sha256"][arm] for arm in ARMS)):
        raise ValueError("A frozen training input or model has changed.")
    keys = episode_keys(split["test"], archive)
    methods = (("physics", None, 0.),) + tuple(
        (arm, arm, receipt["strengths"][arm]["best_nonzero"]) for arm in ARMS
    )
    selected = receipt["strengths"]["joint"]["physics_allowed"]
    if selected != receipt["strengths"]["joint"]["best_nonzero"]:
        methods += (("joint_validation_selected", "joint", selected),)
    episodes = []
    columns = ("scenario", "profile", "method", "elapsed_s", "applied_s", "start_east", "start_north",
               "end_east", "end_north", "initial_bow", "nominal_end_bow", "selected_action",
               "physics_action_at_same_state", "report_margin_m", "truth_margin_m", "age_s")
    with gzip.open(STUDY / "test_leg_trajectories.csv.gz", "wt", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for index, (key, result) in enumerate(zip(
            keys, work(grade, ((key, methods, True) for key in keys), workers, method_names=tuple(ARMS)), strict=True
        ), start=1):
            rows, trace = result
            episodes.extend(rows)
            writer.writerows((key.scenario, key.traffic, *frame) for frame in trace)
            if index % 40 == 0:
                print(f"Tested {index}/{len(keys)} unseen episode cells", flush=True)
    csv_rows(STUDY / "test_episodes.csv", episodes)
    levels = {
        "all": episodes,
        "eligible_at_any_decision": [row for row in episodes if row["eligible_decisions"]],
    }
    for traffic in sorted({row["traffic"] for row in episodes}):
        levels[traffic] = [row for row in episodes if row["traffic"] == traffic]
    for route in sorted({row["route"] for row in episodes}):
        levels[route] = [row for row in episodes if row["route"] == route]
    summary = {key: {method: summarize([row for row in rows if row["method"] == method])
                     for method in sorted({row["method"] for row in rows})}
               for key, rows in levels.items() if rows}
    result = {"scope": "Independent held-out GPX legs, two-snapshot weather, synthetic AIS traffic by leg.",
              "heldout_episode_cells": len(keys), "episode_rows": len(episodes),
              "training_receipt_sha256": digest(STUDY / "training_receipt.json"),
              "test_csv_sha256": digest(STUDY / "test_episodes.csv"),
              "test_trajectories_sha256": digest(STUDY / "test_leg_trajectories.csv.gz"),
              "method_strengths": methods, "strata": summary, "paired_all": paired(episodes)}
    (STUDY / "test_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"number_of_episodes": len(episodes), "summary": summary["all"], "paired": result["paired_all"]}),
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "test"))
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    (train if arguments.stage == "train" else test)(arguments.workers)


if __name__ == "__main__":
    main()
