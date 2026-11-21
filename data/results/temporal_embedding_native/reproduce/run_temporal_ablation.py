"""Train paired residuals and replay the four-arm native Fossen ablation."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import gzip
from importlib.metadata import version
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np

STUDY = Path(__file__).resolve().parents[1]
V2 = STUDY.parents[2]
sys.path.insert(0, str(V2))
sys.path.insert(0, str(STUDY.with_name("grpo_progress_efficient_ais") / "reproduce"))
sys.path.insert(0, str(STUDY.with_name("learned_weather_ais_policy") / "reproduce"))

import run_efficient_retest as efficient
from causal_weather import AsOfSeasonalArchive, CausalEfficientNativeEngine, CausalScenarioEnvironment
from seasonal_ranker import MINIMUM_MODEL_GAIN_S, RESIDUAL_CAP_S, RIDGE, TRUST_REGRET_S, Ranker
from world._geo import local_east_north_metres


seasonal = efficient.seasonal
MANIFEST = efficient.MANIFEST
METHODS = ("physics", "explicit", "temporal", "snapshot")
SPLITS = ("train", "validation", "test")
MAX_STATES_PER_SCENARIO = 8
MIN_START_INTERVAL_S = 180.
MAX_BRANCH_DURATION_S = 4800.
ENCODER_NAME = "fresh_weather_encoder.pt"
ENCODER_PATH = STUDY.with_name("learned_weather_ais_policy") / ENCODER_NAME
AS_OF_ARCHIVES = {}
MODELS = {}
ENCODER = None


def init_worker(model_directory=None, with_encoder=False):
    global MODELS, ENCODER
    seasonal.init(MANIFEST, None, False)
    AS_OF_ARCHIVES.clear()
    MODELS = ({arm: Ranker.load(Path(model_directory) / f"{arm}.npz") for arm in METHODS[1:]}
              if model_directory is not None else {})
    ENCODER = None
    if with_encoder:
        import torch
        from weather_embedding import WeatherEmbedding

        torch.set_num_threads(1)
        ENCODER = WeatherEmbedding(ENCODER_PATH)


def scenario_parts(spec):
    base, state, goals, start, cap = seasonal.scenario_parts(spec)
    block = spec["archive_block"]
    if block not in AS_OF_ARCHIVES:
        AS_OF_ARCHIVES[block] = AsOfSeasonalArchive(base.environment.archive)
    environment = CausalScenarioEnvironment(base.environment, as_of_archive=AS_OF_ARCHIVES[block])
    engine = CausalEfficientNativeEngine(environment, config=base.config, dynamics=base.dynamics)
    return engine, state, goals, start, cap


def forecasts(decision):
    return np.array([c.predicted_eta_s if c.predicted_eta_s is not None else np.inf
                     for c in decision.choices])


def gate_indices(decision, eta):
    incumbent = decision.physics_index
    if incumbent < 0 or not decision.allowed[incumbent] or not np.isfinite(eta[incumbent]):
        return np.empty(0, dtype=int)
    return np.flatnonzero(decision.allowed & np.isfinite(eta)
                          & (eta <= eta[incumbent] + TRUST_REGRET_S))


def history_features(decision, arm=None):
    """Return candidate interactions with the actual and repeated-latest sequence."""
    obs = decision.observation
    target = np.array(local_east_north_metres(
        decision.goal.latitude, decision.goal.longitude, obs.boat.latitude, obs.boat.longitude,
    ))
    history = decision.weather_history
    if history is None or history.shape != (12, 5):
        raise ValueError("A native decision needs twelve as-of weather tokens.")
    latest = np.broadcast_to(history[-1], history.shape)
    if arm is None:
        histories = np.stack((history, latest))
    else:
        histories = (history if arm == "temporal" else latest)[None]
    vectors = ENCODER.encode_histories(histories, np.repeat(target[None], len(histories), axis=0))
    bows = np.array([choice.bow_deg for choice in decision.choices])
    wind_relative = np.radians(bows - obs.wind.direction)
    turn = np.radians(bows - obs.boat.heading)
    interactions = np.column_stack((np.sin(wind_relative), np.cos(wind_relative), np.sin(turn)))
    return tuple((interactions[..., None] * vector).reshape(len(bows), -1) for vector in vectors)


def label_job(spec):
    engine, state, goals, departure, cap = scenario_parts(spec)
    starts = []
    last = {}

    def collect(snapshot, decision):
        if snapshot.pending_switch_bow_deg is not None or snapshot.remaining_cap_s < 120:
            return
        goal_key = decision.goal.latitude, decision.goal.longitude
        if (len(starts) >= MAX_STATES_PER_SCENARIO
                or decision.elapsed_s - last.get(goal_key, -np.inf) < MIN_START_INTERVAL_S):
            return
        eta = forecasts(decision)
        indices = gate_indices(decision, eta)
        if len(indices) >= 2:
            incumbent = decision.physics_index
            starts.append((snapshot, decision, [incumbent, *(int(i) for i in indices if i != incumbent)], eta))
            last[goal_key] = decision.elapsed_s

    initial = engine.run(state, goals, departure, on_decision=collect, cap_seconds=cap, trace=False)
    cases = []
    for snapshot, decision, indices, eta in starts:
        label_cap = min(snapshot.remaining_cap_s, MAX_BRANCH_DURATION_S)
        embedding_history, embedding_snapshot = history_features(decision)
        branches = []
        for branch in engine.branches(snapshot, indices, decision=decision, cap_seconds=label_cap, trace=False):
            outcome = branch.outcome
            action = branch.choice
            report = seasonal.measured_margin(outcome.report_minimum_m)
            truth = seasonal.measured_margin(outcome.truth_minimum_m)
            matched = (outcome.first_choice_requested == action.id == outcome.first_choice_applied
                       and outcome.first_commit_applied_s > 0.)
            branches.append({
                "choice": branch.choice_index, "action_id": action.id, "bow_deg": action.bow_deg,
                "forecast_s": eta[branch.choice_index], "features": decision.features[branch.choice_index],
                "matched_initial_action": matched, "first_commit_applied_s": outcome.first_commit_applied_s,
                "terminal_status": outcome.status, "actual_capture": outcome.success,
                "duration_s": outcome.captures[-1]["elapsed_s"] if outcome.success else outcome.elapsed_s,
                "min_report_margin_m": report, "min_truth_margin_m": truth,
                "audited_safe": report is not None and truth is not None and min(report, truth) > 0.,
            })
        cases.append({
            "state_id": f"{spec['id']}/{decision.elapsed_s:g}", "scenario": spec["id"],
            "block": spec["archive_block"], "split": spec["split"], "decision_utc": decision.when,
            "clock_s": decision.elapsed_s, "cap_s": label_cap,
            "latitude": decision.state.latitude, "longitude": decision.state.longitude,
            "bow_deg": decision.state.heading, "goal": asdict(decision.goal),
            "report_ages_s": [item.report_age_seconds for item in decision.observation.ais_targets],
            "weather_tokens": decision.weather_history, "embedding_history": embedding_history[indices],
            "embedding_snapshot": embedding_snapshot[indices], "branches": branches,
        })
    return spec["id"], initial.status, cases


def mission_job(job):
    from ablation_ranker import select

    spec, method, output = job
    output = Path(output)
    engine, state, goals, departure, cap = scenario_parts(spec)
    evaluations = []

    def strategy(decision):
        eta = forecasts(decision)
        embedding_args = {}
        if method in ("temporal", "snapshot") and len(gate_indices(decision, eta)) >= 2:
            embedding, = history_features(decision, method)
            embedding_args["embedding_history" if method == "temporal" else "embedding_snapshot"] = embedding
        selected = select(method, MODELS.get(method), decision.features, eta, decision.allowed,
                          decision.physics_index, **embedding_args)
        evaluations.append({
            "elapsed_s": decision.elapsed_s, "utc": decision.when, "goal": decision.goal.name,
            "status": selected.status, "eligible_candidates": selected.scored_count,
            "physics_index": decision.physics_index, "requested_index": selected.index,
            "corrected_gain_s": selected.predicted_gain_s,
        })
        return selected.index

    outcome = engine.run(state, goals, departure, strategy=strategy, cap_seconds=cap, trace=True)
    captures = outcome.captures
    expected = spec["expected_captures"]
    if ([capture["name"] for capture in captures] != [item["waypoint"] for item in expected[:len(captures)]]
            or any(row["distance_m"] > spec["capture_radius_m"] + 1e-4 for row in captures)):
        raise ValueError(f"Capture order or geometry failed for {spec['id']} / {method}.")
    applied_by_start = {row["elapsed_s"]: row["choice_id"] for row in outcome.controls}
    decisions = {row["elapsed_s"]: row for row in outcome.decisions}
    gates = [row for row in evaluations if row["requested_index"] != row["physics_index"]]
    for row in evaluations:
        stored = decisions[row["elapsed_s"]]
        row["learned_applied"] = (row["requested_index"] != row["physics_index"]
                                  and stored["selected_index"] == row["requested_index"]
                                  and applied_by_start.get(row["elapsed_s"]) == stored["choice_id"])
    path = output / method / "telemetry" / f"{spec['id']}__{method}.json.gz"
    payload = {
        "states": outcome.trace, "controls": seasonal.finite_records(outcome.controls),
        "control_attempts": seasonal.finite_records(outcome.control_attempts), "decisions": outcome.decisions,
        "captures": captures, "strategy_evaluations": evaluations, "anticipation_events": engine.anticipation_events,
    }
    staged = path.with_suffix(".gz.part")
    with gzip.open(staged, "wt", encoding="utf-8", compresslevel=4) as stream:
        json.dump(payload, stream, default=seasonal.json_value, separators=(",", ":"), allow_nan=False)
    staged.replace(path)
    timings = [row["complete_control_seconds"] for row in outcome.control_attempts]
    minimum_guard = min((row.get("guarded_report_minimum_m", np.inf) for row in outcome.controls),
                        default=np.inf)
    return {
        "scenario": spec["id"], "method": method, "route": spec["route_gpx"],
        "family": spec["route_family"], "weather_block": spec["archive_block"],
        "traffic": spec["traffic_profile"], "departure_utc": spec["departure_utc"],
        "status": outcome.status, "completed": outcome.success,
        "duration_s": captures[-1]["elapsed_s"] if outcome.success else outcome.elapsed_s, "cap_s": cap,
        "expected_captures": len(goals), "capture_count": len(captures),
        "captures": [{**capture, **expected[i]} for i, capture in enumerate(captures)],
        "controls_applied": len(outcome.controls),
        "min_report_margin_m": seasonal.measured_margin(outcome.report_minimum_m),
        "min_truth_margin_m": seasonal.measured_margin(outcome.truth_minimum_m),
        "min_applied_guard_m": seasonal.measured_margin(minimum_guard),
        "maximum_report_age_s": max((row["max_report_age_s"] for row in outcome.controls), default=0.),
        "max_lost_epochs": max((row["dropped_report_epochs"] for row in outcome.controls), default=0),
        "strategy_calls": len(evaluations), "scored_decisions": sum(
            row["corrected_gain_s"] is not None for row in evaluations),
        "gate_crossings": len(gates), "learned_changes": sum(row["learned_applied"] for row in evaluations),
        "largest_predicted_gain_s": max((row["corrected_gain_s"] for row in evaluations
                                         if row["corrected_gain_s"] is not None), default=None),
        "maximum_decision_s": max(timings, default=0.), "overruns_5s": sum(t >= 5. for t in timings),
        "anticipation_events": len(engine.anticipation_events), "trajectory": str(path.relative_to(output / method)),
    }


def jobs(work, items, workers, *, model_directory=None, with_encoder=False):
    args = (model_directory, with_encoder)
    if workers == 1:
        init_worker(*args)
        yield from map(work, items)
        return
    with ProcessPoolExecutor(max_workers=min(workers, len(items)), mp_context=multiprocessing.get_context("spawn"),
                             initializer=init_worker, initargs=args) as pool:
        futures = [pool.submit(work, item) for item in items]
        for future in as_completed(futures):
            yield future.result()


def locked_protocol():
    roster, inherited = efficient.locked_protocol()
    manifest, facts = seasonal.source_inputs(MANIFEST)
    ids = [spec["id"] for spec in manifest["scenarios"] if spec["split"] == "test"]
    if len(ids) != 12 or ids != [spec["id"] for spec in roster]:
        raise ValueError("The October roster differs from the original twelve cases.")
    protocol = {
        **seasonal.provenance(MANIFEST, facts),
        "reference_protocol": inherited,
        "test_ids": ids, "arms": METHODS,
        "minimum_gain_s_strict": MINIMUM_MODEL_GAIN_S, "maximum_forecast_regret_s": TRUST_REGRET_S,
        "ridge": RIDGE, "residual_cap_s": RESIDUAL_CAP_S, "residual_strength": 1.,
        "branch_limit_per_episode": MAX_STATES_PER_SCENARIO, "min_start_interval_s": MIN_START_INTERVAL_S,
        "max_branch_duration_s": MAX_BRANCH_DURATION_S,
        "controller_weather": "Most recent grid valid time at or before each observation or history token.",
        "plant_weather": "Interpolated archive forcing, held for each native control interval.",
        "study_sources": {name: seasonal.digest(Path(__file__).with_name(name)) for name in
                          ("run_temporal_ablation.py", "ablation_ranker.py", "causal_weather.py")},
        "packages": {name: version(name) for name in ("numpy", "scipy", "xarray", "netCDF4", "casadi", "torch")},
    }
    return manifest, protocol


def label_all(out, manifest, protocol, workers):
    directory = out / "branches"
    directory.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        cases = [spec for spec in manifest["scenarios"] if spec["split"] == split]
        summaries, records = [], []
        for name, status, states in jobs(label_job, cases, workers, with_encoder=True):
            records.extend(states)
            count = sum(len(row["branches"]) for row in states)
            summaries.append({"scenario": name, "physics_status": status, "states": len(states), "branches": count})
            print(f"{split:<10} {name} {status}: {len(states)} starts, {count} branches", flush=True)
        path = directory / f"{split}.jsonl.gz"
        staged = path.with_suffix(".gz.part")
        seasonal.save_labels(staged, sorted(records, key=lambda row: (row["scenario"], row["clock_s"])))
        staged.replace(path)
        seasonal.save_json(directory / f"{split}_receipt.json", {
            "protocol": protocol, "split": split, "scenario_ids": [spec["id"] for spec in cases],
            "states": len(records), "labels_sha256": seasonal.digest(path),
            "cases": sorted(summaries, key=lambda row: row["scenario"]),
        })


def fit_models(out, protocol):
    from ablation_ranker import evaluate_fixed_states, fit_arms

    directory = out / "models"
    directory.mkdir(exist_ok=True)
    labels = {}
    hashes = {}
    for split in SPLITS:
        path = out / "branches" / f"{split}.jsonl.gz"
        receipt = json.loads(path.with_name(f"{split}_receipt.json").read_text())
        hashes[split] = seasonal.digest(path)
        if receipt["labels_sha256"] != hashes[split] or receipt["protocol"] != protocol:
            raise ValueError(f"The {split} labels do not match the declared inputs or study.")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            labels[split] = [json.loads(line) for line in stream]
        if any(record["split"] != split for record in labels[split]):
            raise ValueError("Labels are in the wrong partition.")
    models, fit_receipts = fit_arms(labels["train"])
    for method, model in models.items():
        model.save(directory / f"{method}.npz")
    model_hashes = {method: seasonal.digest(directory / f"{method}.npz") for method in models}
    seasonal.save_json(directory / "fit_receipt.json", {
        "protocol": protocol, "fit": fit_receipts, "labels_sha256": hashes, "models_sha256": model_hashes,
    })
    for split in SPLITS:
        evaluation = evaluate_fixed_states(labels[split], models)
        seasonal.save_json(out / f"fixed_states_{split}.json", evaluation)
    print(f"Fitted three residual models and wrote fixed-state metrics from {len(labels['train'])} training starts.",
          flush=True)


def run_missions(out, manifest, protocol, workers):
    directory = out / "models"
    fit_receipt = json.loads((directory / "fit_receipt.json").read_text())
    models = {arm: seasonal.digest(directory / f"{arm}.npz") for arm in METHODS[1:]}
    if fit_receipt["models_sha256"] != models or fit_receipt["protocol"] != protocol:
        raise ValueError("Model or training provenance changed before the locked test.")
    native = out / "native"
    specs = [spec for spec in manifest["scenarios"] if spec["split"] == "test"]
    ids = [spec["id"] for spec in specs]
    for method in METHODS:
        output = native / method
        (output / "telemetry").mkdir(parents=True, exist_ok=True)
        seasonal.save_json(output / "inputs.json", {
            **protocol, "method": method, "scenarios": ids, "probe": False,
            "model_sha256": models.get(method), "workers": min(workers, len(specs) * len(METHODS)),
        })
    tasks = [(spec, method, str(native)) for method in METHODS for spec in specs]
    rows = {method: {} for method in METHODS}
    for row in jobs(mission_job, tasks, workers, model_directory=directory, with_encoder=True):
        method, case = row["method"], row["scenario"]
        rows[method][case] = row
        print(f"{method:<9} {case} {row['capture_count']}/{row['expected_captures']} {row['status']} "
              f"{row['duration_s']:.2f}s applied={row['learned_changes']}", flush=True)
    for method in METHODS:
        output = native / method
        episodes = output / "episodes.csv"
        records = [rows[method][case] for case in ids]
        seasonal.save_csv(episodes, records)
        seasonal.save_json(output / "run.json", {
            "method": method, "inputs_sha256": seasonal.digest(output / "inputs.json"),
            "episodes_sha256": seasonal.digest(episodes), "summary": seasonal.summary(records),
            "trace_sha256es": {row["trajectory"]: seasonal.digest(output / row["trajectory"]) for row in records},
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("all", "branches", "fit", "missions"))
    parser.add_argument("--output", type=Path, default=STUDY)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    manifest, protocol = locked_protocol()
    # JSON normalization keeps protocol equality stable across stage invocations.
    protocol = json.loads(json.dumps(protocol, default=seasonal.json_value))
    out = args.output.resolve()
    if args.stage in ("all", "branches"):
        label_all(out, manifest, protocol, args.workers)
    if args.stage in ("all", "fit"):
        fit_models(out, protocol)
    if args.stage in ("all", "missions"):
        run_missions(out, manifest, protocol, args.workers)


if __name__ == "__main__":
    main()
