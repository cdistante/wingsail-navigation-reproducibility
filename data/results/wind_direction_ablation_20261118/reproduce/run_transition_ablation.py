"""Run and audit the same-label removal of scheduled tack inputs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import gzip
from hashlib import sha256
from itertools import product, zip_longest
import json
import os
from pathlib import Path
import shlex
import sys

import run_study as study


METHOD = "arrival_no_transition"
MODEL = study.STUDY / "arrival_without_transition.json"
CASES = list(product(study.ROUTES, study.WINDS, study.OFFSETS))
ALLOWED_DIFFERENCES = {"--output-dir", "--output-gpx", "--record-replay", "--learned-arrival-model"}


def provenance(build_path: Path, main: Path):
    reproduction = Path(__file__).resolve().parent
    solver = build_path / "direct/c_generated_code"
    libraries = sorted(file for file in solver.glob("*acados_ocp_solver*")
                       if file.suffix in {".dylib", ".so", ".dll"})
    if not libraries or not (solver / "ocp.json").is_file():
        raise FileNotFoundError("The parent study must build Direct acados before the supplement.")
    paths = [
        main / "input_sha256.json", MODEL, study.WORKSPACE / study.ARRIVAL,
        study.STUDY / "offline_metrics_without_transition.json",
        reproduction / "run_study.py", reproduction / "train_arrival_without_transition.py",
        Path(__file__).resolve(), solver / "ocp.json", *libraries,
    ]
    return {str(path): sha256(path.read_bytes()).hexdigest() for path in paths}


def launch(output: Path, main: Path, build_path: Path, jobs: int):
    hashes = provenance(build_path, main)
    manifest = output / "input_sha256.json"
    if manifest.exists() and hashes != json.loads(manifest.read_text()):
        raise ValueError("A supplemental input changed. Use a new output directory.")
    study.write_json(manifest, hashes)
    study.METHODS[METHOD] = replace(
        study.METHODS["arrival"], model=("--learned-arrival-model", str(MODEL)),
        label="Gamma · Direct acados · Arrival time without transition inputs",
    )
    env = {
        **os.environ, "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "PYTHONPATH": os.pathsep.join(filter(None, (str(study.WORKSPACE), os.environ.get("PYTHONPATH")))),
    }
    print(f"Running {len(CASES)} matched schedule-input ablation cases on {jobs} workers.", flush=True)
    failed = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        pending = [pool.submit(study.run_one, output, build_path, route, METHOD, base, offset,
                               env, None, False) for route, base, offset in CASES]
        for index, future in enumerate(as_completed(pending), 1):
            run_id, status, elapsed = future.result()
            print(f"{index}/{len(CASES)} {run_id} exit={status} wall={elapsed:.1f}s", flush=True)
            if status:
                failed.append(run_id)
    if provenance(build_path, main) != hashes:
        raise RuntimeError("A supplemental input changed during the study.")
    if failed:
        raise RuntimeError("Failed cases: " + ", ".join(failed))


def command_parts(folder: Path):
    invariant, varying = [], {}
    tokens = iter(shlex.split((folder / "command.txt").read_text()))
    for token in tokens:
        if token in ALLOWED_DIFFERENCES:
            varying[token] = next(tokens)
        else:
            invariant.append(token)
    return invariant, varying


def state_and_applied_commands(folder: Path):
    with gzip.open(folder / "navigation.jsonl.gz", "rt") as stream:
        next(stream)
        for line in stream:
            frame = json.loads(line)
            nav = frame["navigation"]
            yield frame["state"], nav.get("rudder_command_deg"), nav.get("flap_command_deg")


def audit_pair(control_dir: Path, masked_dir: Path):
    control, ablated = command_parts(control_dir), command_parts(masked_dir)
    if (control[0] != ablated[0]
            or Path(control[1]["--learned-arrival-model"]) != study.WORKSPACE / study.ARRIVAL
            or Path(ablated[1]["--learned-arrival-model"]) != MODEL):
        raise ValueError(f"Paired launch inputs disagree for {masked_dir.name}.")
    first_difference = next((step for step, (a, b) in enumerate(zip_longest(
        state_and_applied_commands(control_dir), state_and_applied_commands(masked_dir),
    )) if a != b), None)
    return first_difference


def report(output: Path, main: Path):
    import summarize as audit

    audit.METHODS = (*audit.METHODS, METHOD)
    audit.CONTRASTS = ((METHOD, "arrival"),)
    controls, variants, legs, manifests, differences = [], [], [], [], {}
    cases = []
    for route, base, offset in CASES:
        prefix = f"{route}__{base:03d}{offset:+03d}__"
        control_dir = main / "runs" / (prefix + "arrival")
        masked_dir = output / "runs" / (prefix + METHOD)
        cases.append((control_dir, masked_dir))
    missing = [folder.name for pair in cases for folder in pair if not (folder / "execution.json").is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} matched run receipts are still missing. First missing: {missing[0]}.")
    for control_dir, masked_dir in cases:
        control, _, _ = audit.audited_run(control_dir / "experiment.json", main, study.WORKSPACE)
        variant, waypoint_rows, evidence = audit.audited_run(
            masked_dir / "experiment.json", output, study.WORKSPACE,
        )
        controls.append(control)
        variants.append(variant)
        legs.extend(waypoint_rows)
        manifests.append(evidence)
        differences[masked_dir.name] = audit_pair(control_dir, masked_dir)
    rows = list(audit.matched_ablation([*controls, *variants]))
    aggregates = list(audit.summarize_pairs(rows))
    by_run = {row["run_id"]: row for row in (*controls, *variants)}
    for row in rows:
        first_difference = differences[row["variant_run"]]
        row["different_state_or_action"] = first_difference is not None
        row["first_different_state_or_action_step"] = first_difference
        row["full_overrides"] = by_run[row["reference_run"]]["learned_overrides"]
        row["ablated_overrides"] = by_run[row["variant_run"]]["learned_overrides"]
    for summary in aggregates:
        matching = [row for row in rows if row["scope"] == summary["scope"]]
        summary.update(
            full_completed=sum(row["reference_completed"] for row in matching),
            ablated_completed=sum(row["variant_completed"] for row in matching),
            different_state_or_action=sum(row["different_state_or_action"] for row in matching),
            full_overrides=sum(row["full_overrides"] for row in matching),
            ablated_overrides=sum(row["ablated_overrides"] for row in matching),
        )
    for filename, table in (("comparison.csv", rows), ("paper_transition_ablation.csv", aggregates),
                            ("run_metrics.csv", variants), ("waypoint_metrics.csv", legs)):
        audit.save_csv(output / filename, table)
    study.write_json(output / "output_verification.json", manifests)
    table = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Same-data ablation of scheduled-continuation inputs. Both models use identical",
        r"transition-aware waypoint arrival labels. Completion time differences use jointly completed",
        r"runs and are reported as ablated minus full. Long-route trials have a three-hour cap.}",
        r"\label{tab:schedule-input-ablation}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Route scope & Pairs & Full complete & Ablated complete & Changed tracks & Median $\Delta t$ (s) & Overrides full/ablated \\",
        r"\midrule",
    ]
    for summary in aggregates:
        scope = {"full_route": "Short saved routes", "bounded_progress": "Bounded long routes"}[summary["scope"]]
        delta = summary["median_completion_time_delta_s"]
        formatted_time = "--" if delta is None else f"{delta:.1f}"
        values = (scope, summary["paired_cells"], summary["full_completed"], summary["ablated_completed"],
                  summary["different_state_or_action"], formatted_time,
                  f"{summary['full_overrides']}/{summary['ablated_overrides']}")
        table.append(" & ".join(map(str, values)) + r" \\")
    (output / "paper_transition_ablation.tex").write_text(
        "\n".join([*table, r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]), encoding="utf-8",
    )
    lines = [
        "# Scheduled-continuation feature ablation replays", "",
        "| Route | Wind FROM | Full arrival model | Ablated arrival model |",
        "|---|---:|---|---|",
    ]
    for a, b in zip(controls, variants, strict=True):
        reference = os.path.relpath(main / a["replay"], output)
        lines.append(f'| {b["route"]} | {b["wind_from"]:g}° | [Replay]({reference}) | [Replay]({b["replay"]}) |')
    (output / "replays.md").write_text("\n".join([*lines, ""]), encoding="utf-8")
    print(json.dumps(aggregates, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-study", type=Path, default=study.STUDY)
    parser.add_argument("--output", type=Path, default=study.STUDY / "transition_ablation")
    parser.add_argument("--acados-build-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=3)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run-only", action="store_true")
    mode.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1 or not args.summarize_only and args.acados_build_dir is None:
        parser.error("Choose at least one worker and supply the parent acados build when running.")
    output, main_study = args.output.resolve(), args.main_study.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        launch(output, main_study, args.acados_build_dir.resolve(), args.jobs)
    if not args.run_only:
        report(output, main_study)


if __name__ == "__main__":
    main()
