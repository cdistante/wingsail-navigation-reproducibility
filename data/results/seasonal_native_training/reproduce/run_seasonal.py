"""Generate matched Fossen labels, fit a residual, and compare on locked splits."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, is_dataclass
import csv
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np

from seasonal_environment import SeasonalArchive
from seasonal_ranker import Pair, Ranker, choose, fit
from world.boat import BoatState
from world._geo import local_east_north_metres
from world.scenario import RouteWaypoint


STUDY = Path(__file__).resolve().parents[1]
V2 = STUDY.parents[2]
STRATEGY_STRENGTHS = (0., .3, 1.)
MAX_LABEL_DURATION_S = 4800.
MAX_LABEL_STATES_PER_SCENARIO = 5
MAX_ALTERNATIVES_PER_STATE = 5
INPUTS = {}
ARCHIVES = {}
ENCODER = None
RANKER = None


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_value(item):
    if isinstance(item, np.datetime64):
        return str(item) + "Z"
    if isinstance(item, np.generic):
        return item.item()
    if isinstance(item, np.ndarray):
        return item.tolist()
    if is_dataclass(item):
        return asdict(item)
    raise TypeError(f"Cannot serialize {type(item)!r}.")


def finite_records(rows):
    return [{key: None if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value
             for key, value in row.items()} for row in rows]


def save_json(path, data):
    staged = path.with_suffix(path.suffix + ".part")
    staged.write_text(json.dumps(data, indent=2, default=json_value, allow_nan=False) + "\n")
    staged.replace(path)


def save_csv(path, records):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(value, default=json_value) if isinstance(value, (tuple, list, dict))
                             else value for key, value in row.items()})


def source_inputs(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    for item in manifest["routes"].values():
        if digest(V2 / item["route_gpx"]) != item["sha256"]:
            raise ValueError(f"The frozen GPX changed: {item['route_gpx']}.")
    facts = {}
    for name, week in manifest["archive_blocks"].items():
        entry = {}
        for kind, reference in week["files"].items():
            path = STUDY / reference
            if not path.is_file():
                raise ValueError(f"Missing verified {name} forcing: {path}.")
            entry[kind] = digest(path)
        facts[name] = entry
    return manifest, facts


def init(manifest_path, model_path, with_features):
    global INPUTS, ARCHIVES, ENCODER, RANKER

    INPUTS = json.loads(Path(manifest_path).read_text())
    ARCHIVES = {}
    ENCODER = None
    RANKER = Ranker.load(Path(model_path)) if model_path else None
    if with_features:
        import torch
        from weather_embedding import WeatherEmbedding

        torch.set_num_threads(1)
        ENCODER = WeatherEmbedding(STUDY.with_name("learned_weather_ais_policy") / "fresh_weather_encoder.pt")


def run_jobs(job, inputs, workers, *, manifest_path, model=None, with_features=True):
    arguments = (str(manifest_path), str(model) if model else None, with_features)
    if workers == 1:
        init(*arguments)
        yield from map(job, inputs)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=init, initargs=arguments) as pool:
        yield from pool.map(job, inputs, chunksize=1)


def scenario_parts(spec):
    from native_branches import NativeEngine, NativeConfig
    from scenario_environment import ScenarioEnvironment

    name = spec["archive_block"]
    if name not in ARCHIVES:
        ARCHIVES[name] = SeasonalArchive(STUDY / "forcing" / name)
    route = INPUTS["routes"][spec["route_id"]]["waypoints"]
    goals = tuple(RouteWaypoint(point["latitude"], point["longitude"], point["waypoint"]) for point in route[1:])
    env = ScenarioEnvironment.from_manifest(ARCHIVES[name], INPUTS, spec)
    engine = NativeEngine(env, config=NativeConfig())
    initial = BoatState(route[0]["latitude"], route[0]["longitude"], spec["initial_heading_deg"], 0.,
                        flap=-10., surge_speed=0.)
    departure = np.datetime64(spec["departure_utc"].removesuffix("Z"), "ns")
    cap = spec["max_steps"] * spec["control_dt_s"]
    return engine, initial, goals, departure, cap


def encode_features(decision, environment):
    """The weather vector interacts with the tack, so it changes relative scores."""
    obs = decision.observation
    target = np.array(local_east_north_metres(decision.goal.latitude, decision.goal.longitude,
                                              obs.boat.latitude, obs.boat.longitude))
    history = decision.weather_history
    if history is None:
        history = environment.weather_history(obs.boat.latitude, obs.boat.longitude, obs.time)
    vector = ENCODER.encode_histories(history, target[None])[0]
    bows = np.array([choice.bow_deg for choice in decision.choices])
    wind_angle = np.radians(bows - obs.wind.direction)
    turn_angle = np.radians(bows - obs.boat.heading)
    interactions = np.column_stack((np.sin(wind_angle), np.cos(wind_angle), np.sin(turn_angle)))
    embedded = (interactions[..., None] * vector[None, None, :]).reshape(len(bows), -1)
    return np.column_stack((decision.features, embedded))


def forecast_times(decision):
    return [choice.predicted_eta_s for choice in decision.choices]


def runnable_choices(decision):
    incumbent = decision.physics_index
    if incumbent < 0:
        return []
    finite = [int(i) for i in np.flatnonzero(decision.allowed)
              if decision.choices[int(i)].predicted_eta_s is not None]
    if incumbent not in finite:
        return []
    others = sorted((i for i in finite if i != incumbent),
                    key=lambda index: decision.choices[index].predicted_eta_s)
    # Keep close alternatives and an occasional far scheduled capture.
    if len(others) >= MAX_ALTERNATIVES_PER_STATE:
        others = others[:MAX_ALTERNATIVES_PER_STATE - 1] + others[-1:]
    return [incumbent, *others]


def branch_pairs(gathered):
    for state in gathered:
        branches = state["branches"]
        physics = branches[0]
        if not physics["matched_initial_action"]:
            continue
        for alt in branches[1:]:
            if alt["matched_initial_action"]:
                yield Pair(state["state_id"], np.asarray(physics["features"]), np.asarray(alt["features"]),
                           physics["forecast_s"], alt["forecast_s"], physics["cost_s"], alt["cost_s"])


def s_time(captures, elapsed, cap):
    return elapsed if captures else 2 * cap


def measured_margin(value):
    return float(value) if value is not None and np.isfinite(value) else None


def summary(rows):
    completed = [r["duration_s"] for r in rows if r["completed"]]
    margins = [min(r["min_report_margin_m"], r["min_truth_margin_m"]) for r in rows
               if r["min_report_margin_m"] is not None and r["min_truth_margin_m"] is not None]
    return {"cases": len(rows), "completed": len(completed),
            "status": dict(Counter(r["status"] for r in rows)),
            "fully_audited_cases": len(margins),
            "nonpositive_clearance_cases": sum(value <= 0. for value in margins),
            "minimum_full_margin_m": min(margins, default=None),
            "mean_penalized_s": float(np.mean([s_time(r["completed"], r["duration_s"], r["cap_s"]) for r in rows])),
            "median_completed_s": float(np.median(completed)) if completed else None,
            "learned_changes": sum(r["learned_changes"] for r in rows),
            "maximum_decision_s": max(r["maximum_decision_s"] for r in rows),
            "overruns_5s": sum(r["overruns_5s"] for r in rows)}


def paired(rows):
    physics = {row["scenario"]: row for row in rows if row["method"] == "physics"}
    learner = {row["scenario"]: row for row in rows if row["method"] == "hybrid"}
    if set(physics) != set(learner):
        raise ValueError("The held-out rows are not paired.")
    common = [(physics[key], learner[key]) for key in physics]
    changes = [b["duration_s"] - a["duration_s"] for a, b in common if a["completed"] and b["completed"]]
    return {"joint_captures": len(changes), "median_hybrid_minus_physics_s": float(np.median(changes)) if changes else None,
            "faster": sum(x < -1e-6 for x in changes), "slower": sum(x > 1e-6 for x in changes),
            "ties": sum(abs(x) <= 1e-6 for x in changes),
            "hybrid_only": sum(b["completed"] and not a["completed"] for a, b in common),
            "physics_only": sum(a["completed"] and not b["completed"] for a, b in common)}


def label_job(spec):
    engine, state, goals, start, cap = scenario_parts(spec)
    starts = []
    last = {}

    def record(snapshot, decision):
        if snapshot.pending_switch_bow_deg is not None:
            return
        choices = runnable_choices(decision)
        goal = (decision.goal.latitude, decision.goal.longitude)
        if (len(starts) < MAX_LABEL_STATES_PER_SCENARIO and len(choices) >= 2
                and decision.elapsed_s - last.get(goal, -np.inf) >= 120.
                and sum(row[1].goal == decision.goal for row in starts) < 3):
            starts.append((snapshot, decision, choices))
            last[goal] = decision.elapsed_s

    baseline = engine.run(state, goals, start, on_decision=record, cap_seconds=cap, trace=False)
    states = []
    for snapshot, decision, indices in starts:
        features = encode_features(decision, engine.environment)
        label_cap = min(snapshot.remaining_cap_s, MAX_LABEL_DURATION_S)
        if label_cap < 120.:
            continue
        branches = engine.branches(snapshot, indices, decision=decision, cap_seconds=label_cap, trace=False)
        candidates = []
        for branch in branches:
            i, action, outcome = branch.choice_index, branch.choice, branch.outcome
            captured = outcome.success
            report = measured_margin(outcome.report_minimum_m)
            truth = measured_margin(outcome.truth_minimum_m)
            safe = report is not None and truth is not None and min(report, truth) > 0.
            matched = (outcome.first_choice_requested == action.id == outcome.first_choice_applied
                       and outcome.first_commit_applied_s > 0.)
            duration = outcome.captures[-1]["elapsed_s"] if captured else outcome.elapsed_s
            candidates.append({"choice": i, "action_id": action.id, "bow_deg": action.bow_deg,
                               "forecast_s": action.predicted_eta_s,
                               "features": features[i], "matched_initial_action": matched,
                               "first_commit_applied_s": outcome.first_commit_applied_s,
                               "terminal_status": outcome.status, "duration_s": duration,
                               "cost_s": s_time(captured and safe, duration, label_cap),
                               "min_report_margin_m": report, "min_truth_margin_m": truth,
                               "actual_capture": captured, "audited_safe": safe})
        states.append({"state_id": f"{spec['id']}/{decision.elapsed_s:g}",
                       "scenario": spec["id"], "block": spec["archive_block"],
                       "clock_s": decision.elapsed_s, "decision_utc": decision.when,
                       "latitude": decision.state.latitude, "longitude": decision.state.longitude,
                       "bow_deg": decision.state.heading, "goal": asdict(decision.goal),
                       "report_ids": [item.mmsi for item in decision.observation.ais_targets],
                       "report_ages_s": [item.report_age_seconds for item in decision.observation.ais_targets],
                       "cap_s": label_cap, "branches": candidates})
    return spec["id"], baseline.status, states


def mission_job(job):
    spec, method, strength, stage = job
    engine, state, goals, start, cap = scenario_parts(spec)
    learned_changes = 0

    def strategy(decision):
        nonlocal learned_changes

        if not strength or len(runnable_choices(decision)) < 2:
            return decision.physics_index
        features = encode_features(decision, engine.environment)
        selected = choose(RANKER, features, forecast_times(decision), decision.allowed,
                          decision.physics_index, strength)
        learned_changes += selected != decision.physics_index
        return selected

    outcome = engine.run(state, goals, start, strategy=strategy, cap_seconds=cap, trace=stage == "test")
    trace_path = None
    if stage == "test":
        directory = STUDY / "telemetry"
        directory.mkdir(exist_ok=True)
        trace_path = directory / f"{spec['id']}__{method}.json.gz"
        with gzip.open(trace_path, "wt", encoding="utf-8", compresslevel=4) as stream:
            bundle = {"states": outcome.trace, "controls": finite_records(outcome.controls),
                      "control_attempts": finite_records(outcome.control_attempts),
                      "decisions": outcome.decisions, "captures": outcome.captures}
            json.dump(bundle, stream, default=json_value, separators=(",", ":"), allow_nan=False)
    captures = list(outcome.captures)
    if not all(capture["distance_m"] <= spec["capture_radius_m"] + 1e-4 for capture in captures):
        raise ValueError("A route capture lies outside the declared GPX radius.")
    expected = spec["expected_captures"]
    if [capture["name"] for capture in captures] != [item["waypoint"] for item in expected[:len(captures)]]:
        raise ValueError("The simulator returned captures outside the declared GPX order.")
    placed = [{**capture, **expected[i]} for i, capture in enumerate(captures)]
    completed = outcome.success and len(captures) == len(goals)
    duration = captures[-1]["elapsed_s"] if completed else outcome.elapsed_s
    decision_times = [row["complete_control_seconds"] for row in outcome.control_attempts]
    return {"scenario": spec["id"], "route": spec["route_gpx"], "family": spec["route_family"],
            "weather_block": spec["archive_block"], "traffic": spec["traffic_profile"],
            "departure_utc": spec["departure_utc"], "method": method, "strength": strength,
            "status": outcome.status, "completed": completed, "duration_s": duration, "cap_s": cap,
            "expected_captures": len(goals), "captures": placed, "capture_count": len(placed),
            "min_report_margin_m": measured_margin(outcome.report_minimum_m),
            "min_truth_margin_m": measured_margin(outcome.truth_minimum_m),
            "maximum_report_age_s": max((row["max_report_age_s"] for row in outcome.controls), default=0.),
            "max_lost_epochs": max((row["dropped_report_epochs"] for row in outcome.controls), default=0),
            "learned_changes": learned_changes, "maximum_decision_s": max(decision_times, default=0.),
            "overruns_5s": sum(t >= 5. for t in decision_times),
            "trajectory": str(trace_path.relative_to(STUDY)) if trace_path else None}


def save_labels(path, rows):
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=4) as stream:
        for entry in rows:
            stream.write(json.dumps(entry, default=json_value, allow_nan=False) + "\n")


def provenance(manifest_path, facts):
    from scenario_environment import CHART_PATH

    sources = {name: digest(Path(__file__).with_name(name)) for name in
               ("run_seasonal.py", "native_branches.py", "seasonal_environment.py", "scenario_environment.py",
                "seasonal_ranker.py")}
    for filename in ("fossen.py", "acados_fossen.py", "forecast_tacks.py", "ais_avoidance.py",
                     "synthetic_ais.py", "colreg.py", "global_map.py", "learned_tacks.py"):
        sources[f"world/{filename}"] = digest(V2 / "world" / filename)
    encoder = STUDY.with_name("learned_weather_ais_policy") / "fresh_weather_encoder.pt"
    sources["weather_embedding.py"] = digest(encoder.parent / "reproduce" / "weather_embedding.py")
    return {"split_manifest_sha256": digest(manifest_path), "forcing_sha256": facts, "source_sha256": sources,
            "frozen_encoder_sha256": digest(encoder), "bathymetry_sha256": digest(CHART_PATH)}


def train(manifest_path, manifest, facts, workers):
    entries = [spec for spec in manifest["scenarios"] if spec["split"] == "train"]
    labelled = []
    behavior = []
    for case, status, states in run_jobs(label_job, entries, workers, manifest_path=manifest_path):
        labelled.extend(states)
        behavior.append({"scenario": case, "physics_status": status, "decision_states": len(states),
                         "native_branches": sum(len(row["branches"]) for row in states)})
        print(f"Training {case} {status} ({len(states)} branch starts)", flush=True)
    labels_path = STUDY / "native_tack_examples.jsonl.gz"
    save_labels(labels_path, labelled)
    fitted, receipt = fit(branch_pairs(labelled))
    model_path = STUDY / "seasonal_hybrid.npz"
    fitted.save(model_path)
    save_json(STUDY / "training_receipt.json",
              {**provenance(manifest_path, facts), **receipt, "model_sha256": digest(model_path),
               "labels_sha256": digest(labels_path), "training_cases": behavior,
               "training_examples": sum(len(row["branches"]) for row in labelled),
               "closed_loop_forecast": "Causal persistence with latest observed forcing at the decision.",
               "label_scope": "Native integration and AIS from matched starts through first waypoint entry or censoring."})


def verify_training(manifest_path, facts):
    receipt = json.loads((STUDY / "training_receipt.json").read_text())
    if any(receipt[key] != value for key, value in provenance(manifest_path, facts).items()):
        raise ValueError("The pinned seasonal training inputs have changed.")
    model_path = STUDY / "seasonal_hybrid.npz"
    if receipt["model_sha256"] != digest(model_path):
        raise ValueError("The pinned trained model has changed.")
    return model_path


def validate(manifest_path, manifest, facts, workers):
    model = verify_training(manifest_path, facts)
    entries = [spec for spec in manifest["scenarios"] if spec["split"] == "validation"]
    jobs = [(spec, "physics" if strength == 0. else f"hybrid_{strength:g}", strength, "validation")
            for spec in entries for strength in STRATEGY_STRENGTHS]
    rows = []
    for outcome in run_jobs(mission_job, jobs, workers, manifest_path=manifest_path, model=model):
        rows.append(outcome)
        print(f"Validation {outcome['scenario']} {outcome['method']} {outcome['status']}", flush=True)
    stats = {strength: summary([row for row in rows if row["strength"] == strength])
             for strength in STRATEGY_STRENGTHS}
    metric = lambda strength: (stats[strength]["nonpositive_clearance_cases"],
                               -stats[strength]["completed"], stats[strength]["mean_penalized_s"], strength)
    best_active = min((strength for strength in STRATEGY_STRENGTHS if strength > 0), key=metric)
    selected = min(STRATEGY_STRENGTHS, key=metric)
    save_csv(STUDY / "validation_episodes.csv", rows)
    save_json(STUDY / "validation_selection.json",
              {"selected_strength": selected, "best_nonzero_strength": best_active,
               "training_receipt_sha256": digest(STUDY / "training_receipt.json"),
               "validation_cases": len(entries), "metrics": stats})


def test(manifest_path, manifest, facts, workers):
    model = verify_training(manifest_path, facts)
    selection = json.loads((STUDY / "validation_selection.json").read_text())
    if selection["training_receipt_sha256"] != digest(STUDY / "training_receipt.json"):
        raise ValueError("The validation receipt belongs to a different model.")
    entries = [spec for spec in manifest["scenarios"] if spec["split"] == "test"]
    strength = selection["best_nonzero_strength"]
    jobs = [(spec, method, mix, "test") for spec in entries
            for method, mix in (("physics", 0.), ("hybrid", strength))]
    rows = []
    for outcome in run_jobs(mission_job, jobs, workers, manifest_path=manifest_path, model=model):
        rows.append(outcome)
        print(f"Held out {outcome['scenario']} {outcome['method']} {outcome['status']}", flush=True)
    save_csv(STUDY / "heldout_episodes.csv", rows)
    save_json(STUDY / "heldout_results.json", {
        "summary": {method: summary([row for row in rows if row["method"] == method])
                    for method in ("physics", "hybrid")},
        "paired": paired(rows),
        "hybrid_strength_tested": strength,
        "validation_preferred_strength": selection["selected_strength"],
        "validation_selection_sha256": digest(STUDY / "validation_selection.json"),
        "episodes_sha256": digest(STUDY / "heldout_episodes.csv"),
        "trace_sha256es": {row["trajectory"]: digest(STUDY / row["trajectory"]) for row in rows},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "validate", "test"))
    parser.add_argument("--manifest", type=Path, default=STUDY / "manifest_expanded.json")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifest, facts = source_inputs(args.manifest)
    {"train": train, "validate": validate, "test": test}[args.stage](
        args.manifest, manifest, facts, args.workers
    )


if __name__ == "__main__":
    main()
