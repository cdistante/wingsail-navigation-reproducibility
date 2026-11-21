"""Collect reproducible matched native Fossen branch labels at hard starts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
import gzip
import hashlib
from itertools import islice
import json
import multiprocessing
from pathlib import Path
import random
import sys
from time import monotonic

import numpy as np


STUDY = Path(__file__).resolve().parents[1]
ABLAT = STUDY.with_name("temporal_embedding_native")
sys.path.insert(0, str(ABLAT / "reproduce"))
import run_temporal_ablation as baseline
from hard_starts import MAX_EXPANSION, discover, make_roster, prepare


seasonal = baseline.seasonal
SPLITS = ("train", "calibration")
TARGETS = {"train": 3000, "calibration": 480}
MAX_CANDIDATES = 6
MAX_BRANCH_DURATION_S = 4800.
SEED = 20261120
SOURCE_FILES = (
    Path(__file__).resolve(), Path(__file__).with_name("hard_starts.py"),
    ABLAT / "reproduce" / "causal_weather.py",
    ABLAT / "reproduce" / "run_temporal_ablation.py",
)


def source_hashes():
    return {str(path.relative_to(STUDY.parent)): seasonal.digest(path) for path in SOURCE_FILES}


def init_worker(pinned):
    if source_hashes() != pinned:
        raise RuntimeError("Stage 1 simulator differs from the pinned parent sources.")
    baseline.init_worker()


def gather(pool, function, rows, window=64):
    """Bound outstanding futures so an interruption has bounded in-flight work."""
    remaining = iter(rows)
    pending = {pool.submit(function, row) for row in islice(remaining, window)}
    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        pending.update(pool.submit(function, row) for row in islice(remaining, len(done)))
        for future in done:
            yield future.result()


def json_key(data):
    return json.dumps(data, default=seasonal.json_value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def selected_candidates(decision, indices):
    """Keep the best ETA options with at least one other wind side when available."""
    incumbent, *alternatives = (int(i) for i in indices)
    choices = decision.choices
    alternatives.sort(key=lambda i: (choices[i].predicted_eta_s, i))
    if len(alternatives) < MAX_CANDIDATES:
        return [incumbent, *alternatives]
    wind = decision.observation.wind.direction
    side = lambda i: (choices[i].bow_deg - wind + 180.) % 360. - 180. >= 0.
    other = next((i for i in alternatives if side(i) != side(incumbent)), None)
    kept = alternatives[:MAX_CANDIDATES - 1]
    if other is not None and other not in kept:
        kept[-1] = other
    return [incumbent, *kept]


def label_job(task):
    spec, discovered = task
    engine, snapshot, decision, indices = prepare(spec)
    if list(map(int, indices)) != discovered["eligible_indices"]:
        raise RuntimeError(f"Candidate eligibility changed after discovery for {spec['id']}.")
    chosen = selected_candidates(decision, indices)
    cap = min(snapshot.remaining_cap_s, MAX_BRANCH_DURATION_S)
    branches = []
    for branch in engine.branches(snapshot, chosen, decision=decision, cap_seconds=cap, trace=False):
        action, outcome = branch.choice, branch.outcome
        report = seasonal.measured_margin(outcome.report_minimum_m)
        truth = seasonal.measured_margin(outcome.truth_minimum_m)
        guarded = seasonal.measured_margin(outcome.guarded_report_minimum_m)
        matched = (outcome.first_choice_requested == action.id == outcome.first_choice_applied
                   and outcome.first_commit_applied_s > 0.)
        audited = (report is not None and truth is not None and min(report, truth) > 0.
                   and outcome.applied_chart_violations == 0)
        branches.append({
            "choice": branch.choice_index, "action_id": action.id, "bow_deg": action.bow_deg,
            "forecast_s": action.predicted_eta_s, "features": branch.features,
            "matched_initial_action": matched, "first_commit_applied_s": outcome.first_commit_applied_s,
            "terminal_status": outcome.status, "actual_capture": outcome.success, "audited_safe": audited,
            "duration_s": outcome.captures[-1]["elapsed_s"] if outcome.success else outcome.elapsed_s,
            "min_report_margin_m": report, "min_truth_margin_m": truth, "min_guard_margin_m": guarded,
            "chart_violations": outcome.applied_chart_violations,
        })
    row = {
        "state_id": spec["id"], "scenario": spec["id"], "block": spec["archive_block"],
        "split": spec["split"], "group": spec["cluster"], "route_id": spec["route_id"],
        "decision_utc": decision.when, "clock_s": decision.elapsed_s, "cap_s": cap,
        "latitude": decision.state.latitude, "longitude": decision.state.longitude,
        "bow_deg": decision.state.heading, "goal": asdict(decision.goal),
        "report_ages_s": [item.report_age_seconds for item in decision.observation.ais_targets],
        "weather_tokens": np.asarray(decision.weather_history),
        "hard_start": discovered["regime"], "geometry": spec["geometry"],
        "initial_boat_state": asdict(decision.state),
        "eligible_before_sampling": len(indices), "branches": branches,
    }
    return row


def stratified_order(plans):
    rng = random.Random(SEED)
    by_stratum = defaultdict(lambda: defaultdict(list))
    for plan in plans:
        spec, discovered = plan
        by_stratum[(spec["archive_block"], str(discovered["regime"]))][spec["cluster"]].append(plan)
    strata = []
    for key in sorted(by_stratum):
        clusters = list(by_stratum[key].values())
        rng.shuffle(clusters)
        for group in clusters:
            rng.shuffle(group)
        strata.append(deque(deque(group) for group in clusters))
    ordered = []
    while any(strata):
        for clusters in strata:
            if clusters:
                cluster = clusters.popleft()
                ordered.append(cluster.popleft())
                if cluster:
                    clusters.append(cluster)
    return ordered


def plan_labels(out, split, pool, target, expansion_limit):
    receipts = []
    eligible = []
    known = set()
    for expansion in range(expansion_limit + 1):
        specs = [spec for spec in make_roster(split, expansion=expansion) if spec["id"] not in known]
        if not specs:
            continue
        if len({spec["id"] for spec in specs}) != len(specs):
            raise ValueError("The prospective roster contains duplicate starts.")
        indexed = {spec["id"]: spec for spec in specs}
        known.update(indexed)
        records = list(gather(pool, discover, specs))
        available = [row for row in records if len(row["eligible_indices"]) >= 3]
        eligible.extend((indexed[row["id"]], row) for row in sorted(available, key=lambda r: r["id"]))
        potential = sum(min(len(row["eligible_indices"]), MAX_CANDIDATES) for _, row in eligible)
        counts = dict(Counter(row["status"] for row in records))
        receipts.append({"expansion": expansion, "prospective_starts": len(specs), "status": counts,
                         "eligible_starts": len(available), "potential_accumulated_branches": potential})
        print(f"{split} discovery {expansion}: {len(specs)} starts screened, {len(available)} eligible; "
              f"{potential} potential branches", flush=True)
        if potential >= target:
            break
    ordered = stratified_order(eligible)
    selected = []
    planned = 0
    for pair in ordered:
        if planned >= target:
            break
        selected.append(pair)
        planned += min(len(pair[1]["eligible_indices"]), MAX_CANDIDATES)
    if planned < target:
        raise RuntimeError(f"Only {planned} of {target} {split} branches available in the prospective roster.")
    result = {"split": split, "target": target, "planned_branches": planned, "discoveries": receipts,
              "selected": [{"spec": spec, "decision": decision} for spec, decision in selected]}
    seasonal.save_json(out / f"{split}_plan.json", result)
    return result


def counts(rows):
    branches = [branch for row in rows for branch in row["branches"]]
    status = Counter(branch["terminal_status"] for branch in branches)
    good = lambda branch: (branch["actual_capture"] and branch["audited_safe"]
                           and branch["matched_initial_action"])
    safe_pairs = 0
    gain_pairs = 0
    gain_states = 0
    for row in rows:
        incumbent, *alternatives = row["branches"]
        if good(incumbent):
            used = [branch for branch in alternatives if good(branch)]
            wins = sum(incumbent["duration_s"] - branch["duration_s"] > 25. for branch in used)
            safe_pairs += len(used)
            gain_pairs += wins
            gain_states += bool(wins)
    return {"starts": len(rows), "branches": len(branches), "terminal_status": dict(status),
            "groups": len({row["group"] for row in rows}), "safe_capture_pairs": safe_pairs,
            "realized_safe_gains_over_25s": gain_pairs, "starts_with_realized_safe_gain_over_25s": gain_states,
            "start_regimes": dict(Counter(str(row["hard_start"]) for row in rows)),
            "branches_with_nonpositive_full_margin": sum(
                any(branch[key] is not None and branch[key] <= 0.
                    for key in ("min_report_margin_m", "min_truth_margin_m")) for branch in branches),
            "branches_with_chart_violations": sum(branch["chart_violations"] > 0 for branch in branches),
            "unapplied_first_actions": sum(not branch["matched_initial_action"] for branch in branches)}


def label_plan(out, split, pool, plan, protocol):
    checkpoint = out / f"{split}_checkpoint.jsonl"
    binding = out / f"{split}_checkpoint_receipt.json"
    identity = hashlib.sha256(json_key({"protocol": protocol, "plan": plan}).encode()).hexdigest()
    existing = {}
    if binding.exists():
        if json.loads(binding.read_text())["sha256"] != identity:
            raise ValueError("The saved checkpoint belongs to a different source or design.")
        if checkpoint.exists():
            with checkpoint.open(encoding="utf-8") as stream:
                existing = {row["state_id"]: row for row in map(json.loads, stream)}
    else:
        seasonal.save_json(binding, {"sha256": identity})
    all_ids = [row["spec"]["id"] for row in plan["selected"]]
    tasks = [(row["spec"], row["decision"]) for row in plan["selected"] if row["spec"]["id"] not in existing]
    completed = sum(len(row["branches"]) for row in existing.values())
    last = monotonic()
    with checkpoint.open("a", encoding="utf-8", buffering=1) as stream:
        for row in gather(pool, label_job, tasks, window=64):
            line = json_key(row)
            stream.write(line + "\n")
            existing[row["state_id"]] = json.loads(line)
            completed += len(row["branches"])
            if monotonic() - last >= 25 or completed == plan["planned_branches"]:
                print(f"{split}: {completed}/{plan['planned_branches']} native branches "
                      f"at {len(existing)}/{len(all_ids)} starts", flush=True)
                last = monotonic()
    if set(existing) != set(all_ids) or completed != plan["planned_branches"]:
        raise RuntimeError("Incomplete or mismatched native branch results.")
    rows = [existing[state_id] for state_id in all_ids]
    file = out / f"{split}.jsonl.gz"
    temporary = file.with_suffix(".gz.part")
    seasonal.save_labels(temporary, rows)
    temporary.replace(file)
    receipt = {**counts(rows), "labels_sha256": seasonal.digest(file),
               "plan_sha256": seasonal.digest(out / f"{split}_plan.json"), "protocol": protocol}
    seasonal.save_json(out / f"{split}_receipt.json", receipt)
    print(f"{split}: {receipt['branches']} branches, {receipt['safe_capture_pairs']} exact safe pairs, "
          f"{receipt['groups']} groups", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STUDY / "branches")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--train-target", type=int, default=TARGETS["train"])
    parser.add_argument("--calibration-target", type=int, default=TARGETS["calibration"])
    parser.add_argument("--expansion-limit", type=int, default=MAX_EXPANSION)
    parser.add_argument("--split", choices=(*SPLITS, "both"), default="both")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if min(args.workers, args.train_target, args.calibration_target) < 1 or not 0 <= args.expansion_limit <= MAX_EXPANSION:
        parser.error("Choose positive workers and branch targets and a supported expansion tier.")
    hashes = source_hashes()
    _, forcing = seasonal.source_inputs(baseline.MANIFEST)
    _, inherited = baseline.efficient.locked_protocol()
    protocol = {
        "source_sha256": hashes, "manifest_sha256": seasonal.digest(baseline.MANIFEST),
        "forcing_sha256": forcing, "guard": inherited["guard"],
        "geometry": "Prospectively filtered by upwind geometry or opposing current, with at least two alternatives.",
        "acceptance": "Legal finite schedule ETAs within 150 s of incumbent, at most six branches per start.",
        "max_branch_duration_s": MAX_BRANCH_DURATION_S, "capture_radius_m": 50.,
        "population": {"train": ["winter", "summer"], "calibration": ["spring"]},
    }
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=init_worker, initargs=(hashes,)) as pool:
        targets = {"train": args.train_target, "calibration": args.calibration_target}
        for split in SPLITS if args.split == "both" else (args.split,):
            plan_path = out / f"{split}_plan.json"
            if plan_path.exists():
                plan = json.loads(plan_path.read_text())
                if plan["target"] != targets[split]:
                    raise ValueError("Use a separate output directory for a new branch budget.")
            else:
                plan = plan_labels(out, split, pool, targets[split], args.expansion_limit)
            if not args.plan_only:
                label_plan(out, split, pool, plan, protocol)
    if source_hashes() != hashes:
        raise RuntimeError("Stage 1 simulator source changed before completion.")


if __name__ == "__main__":
    main()
