"""Launch matched saved-route cases and keep every replay, track and run log."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from time import monotonic

STUDY = Path(__file__).resolve().parents[1]
WORKSPACE = STUDY.parents[2]
CHART = "data/bathymetry/gebco_2026_n44.0_s37.0_w14.0_e22.0.nc"
POLICY = "simulator_harmonic/transformer_runs_diff_e30/run_diffusion_g41_n3578_e30/best.pt"
ACTUATOR = "data/models/marine_controller_combined_dagger_30e/best.pt"
PROGRESS = "data/results/learned_tack_selection_20261118/tack_model.json"
ARRIVAL = "data/results/learned_tack_arrival_20261118/arrival_model.json"
WINDS = (0, 90, 180, 270)
OFFSETS = (-5, 5)
DT = 5


@dataclass(frozen=True)
class RouteCase:
    file: str
    departure: str
    cap_s: int = 6000
    radius: int = 50
    scope: str = "full_route"


ROUTES = {
    "brindisi_short": RouteCase("small route.gpx", "2026-10-01T13:00:00Z"),
    "brindisi_003": RouteCase("opencpn_001_003_test.gpx", "2026-10-01T13:00:00Z"),
    "brindisi_004": RouteCase("opencpn_001_004_test.gpx", "2026-10-01T13:00:00Z"),
    "costiera": RouteCase("costiera.gpx", "2026-09-30T03:00:00Z"),
    "otranto_gate": RouteCase("otranto_001_002_test.gpx", "2026-09-30T03:00:00Z"),
    "otranto_003": RouteCase("otranto_short_001_003_loop.gpx", "2026-09-30T02:40:00Z"),
    "otranto_005": RouteCase("otranto_short_test_loop.gpx", "2026-09-30T23:50:00Z"),
    "naples": RouteCase("naples_8_8nm_round_trip.gpx", "2026-09-30T13:00:00Z", 10800, 100, "bounded_progress"),
    "corfu": RouteCase("rotta_corfu_nord.gpx", "2026-09-30T00:00:00Z", 10800, 100, "bounded_progress"),
}


@dataclass(frozen=True)
class Method:
    label: str
    planner: str = "gamma"
    controller: str = "grid-mpc"
    forecast: bool = False
    model: tuple[str, str] | None = None
    device: str | None = None


METHODS = {
    "standard_grid": Method("Standard harmonic · grid MPC", "standard"),
    "gamma_grid": Method("Gamma harmonic · grid MPC"),
    "diffusion_grid": Method("Anisotropic Diffusion · grid MPC", "diffusion"),
    "transformer_grid": Method("Guarded Diffusion Transformer · grid MPC", "transformer_diffusion",
                               model=("--transformer-diffusion-checkpoint", POLICY),
                               device="--transformer-diffusion-device"),
    "gamma_los": Method("Gamma · acados LOS", controller="acados-los"),
    "gamma_direct": Method("Gamma · Direct acados", controller="acados-direct"),
    "forecast": Method("Gamma · Direct acados · Classical forecast tacks", controller="acados-direct",
                       forecast=True),
    "progress": Method("Gamma · Direct acados · Learned 120 s progress", controller="acados-direct",
                       forecast=True, model=("--learned-tack-model", PROGRESS)),
    "arrival": Method("Gamma · Direct acados · Learned waypoint arrival time", controller="acados-direct",
                      forecast=True, model=("--learned-arrival-model", ARRIVAL)),
    "actuator": Method("Gamma · Guarded DAgger actuator Transformer", model=("--combined-controller-checkpoint", ACTUATOR),
                       device="--combined-controller-device"),
}


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def fingerprint():
    source = sorted((WORKSPACE / "world").glob("*.py"))
    source.extend(sorted((WORKSPACE / "world/legacy_planners").glob("*.py")))
    source.extend(Path(__file__).with_name(name) for name in
                  ("run_study.py", "run_worker.py", "run_wind_scenario.py", "fast_fossen.py", "reuse_fields.py"))
    archives = [p for field in ("weather", "waves", "currents")
                for p in sorted((WORKSPACE / "data" / field).glob("*.nc"))]
    paths = [WORKSPACE / file for file in ("config.py", CHART, POLICY, ACTUATOR, PROGRESS, ARRIVAL)]
    paths.extend(WORKSPACE / "current_route" / case.file for case in ROUTES.values())
    archive_wrappers = WORKSPACE / "data/results/learned_tack_arrival_20261118/reproduce"
    paths.extend(archive_wrappers / file for file in ("run_archived_replay.py", "run_local_wind_replay.py"))
    manifest = {}
    for path in [*paths, *source, *archives]:
        with path.open("rb") as stream:
            manifest[str(path.relative_to(WORKSPACE))] = sha256_stream(stream)
    return manifest


def sha256_stream(stream):
    digest = sha256()
    for block in iter(lambda: stream.read(4 * 1024**2), b""):
        digest.update(block)
    return digest.hexdigest()


def prepare_solver(build_dir, controllers, env):
    build_dir.mkdir(parents=True, exist_ok=True)
    for controller in sorted(controllers - {"grid-mpc"}):
        mode = controller.removeprefix("acados-")
        code = (
            "import sys; from world.acados_nmpc import AcadosNMPC; "
            "from world.fossen import Fossen3DOFSaildroneDynamics; "
            "AcadosNMPC(dynamics=Fossen3DOFSaildroneDynamics(), mode=sys.argv[1], "
            "dt_seconds=float(sys.argv[2]), no_go_angle_degrees=45.0, build_directory=sys.argv[3])"
        )
        subprocess.run([sys.executable, "-B", "-c", code, mode, str(DT), str(build_dir)],
                       cwd=WORKSPACE, env=env, check=True)


def run_one(study, build_dir, route, method, base, offset, env, smoke_steps, reference):
    case, strategy = ROUTES[route], METHODS[method]
    cap_s = case.cap_s if smoke_steps is None else DT * smoke_steps
    run_id = f"{route}__{base:03d}{offset:+03d}__{method}"
    folder = study / ("pilots" if smoke_steps is not None else "runs") / run_id
    folder.mkdir(parents=True, exist_ok=True)
    route_path = WORKSPACE / "current_route" / case.file
    command = [
        sys.executable, "-B", str(Path(__file__).with_name("run_worker.py")),
        "--wind-from", str(base), "--wind-offset", str(offset),
        "--environment", "marine", "--dynamics", "fossen",
        "--planner", strategy.planner, "--controller", strategy.controller,
        "--route-gpx", str(route_path), "--global-map", str(WORKSPACE / CHART),
        "--time", case.departure, "--dt", str(DT), "--steps", str(cap_s // DT),
        "--arrival-radius", str(case.radius), "--heading", "45", "--flap=-10",
        "--plot-every", "0", "--live-plot-frame", "fixed", "--live-plot-boat-scale", "3",
        "--output-dir", str(folder), "--output-gpx", str(folder / "track.gpx"),
        "--record-replay", str(folder / "navigation.jsonl.gz"),
    ]
    if strategy.controller != "grid-mpc":
        command += ["--acados-build-dir", str(build_dir)]
    if strategy.forecast:
        command.append("--forecast-tacks")
    if strategy.model is not None:
        flag, checkpoint = strategy.model
        command += [flag, str(WORKSPACE / checkpoint)]
    if strategy.device:
        command += [strategy.device, "cpu"]
    if reference:
        command.append("--reference-kernels")
    metadata = {
        "run_id": run_id, "route": route, "route_file": str(route_path), "scope": case.scope,
        "method": method, "method_label": strategy.label, "base_wind": base, "offset": offset,
        "wind_from": (base + offset) % 360, "cap_s": cap_s, "radius_m": case.radius, "dt_s": DT,
        "departure": case.departure, "chart": str(WORKSPACE / CHART),
        "reference_kernels": reference, "return_to_start": False,
    }
    receipt = folder / "execution.json"
    if receipt.exists() and (folder / "experiment.json").exists():
        previous = json.loads(receipt.read_text())
        if (previous["return_code"] == 0 and json.loads((folder / "experiment.json").read_text()) == metadata
                and (folder / "navigation.jsonl.gz").is_file() and (folder / "track.gpx").is_file()):
            return run_id, 0, previous["wall_seconds"]
    write_json(folder / "experiment.json", metadata)
    (folder / "command.txt").write_text(shlex.join(command) + "\n")
    start = monotonic()
    with (folder / "run.log").open("w") as log:
        result = subprocess.run(command, cwd=WORKSPACE, env=env, stdout=log, stderr=subprocess.STDOUT)
    elapsed = monotonic() - start
    write_json(receipt, {**metadata, "return_code": result.returncode, "wall_seconds": elapsed})
    return run_id, result.returncode, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--acados-build-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--route", choices=ROUTES, action="append")
    parser.add_argument("--method", choices=METHODS, action="append")
    parser.add_argument("--base", type=int, choices=WINDS, action="append")
    parser.add_argument("--offset", type=int, choices=OFFSETS, action="append")
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--reference-kernels", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1 or args.smoke_steps is not None and args.smoke_steps < 1:
        parser.error("Worker count and smoke steps must be positive.")
    study = args.study.resolve()
    study.mkdir(parents=True, exist_ok=True)
    manifest = study / "input_sha256.json"
    if args.smoke_steps is None:
        hashes = fingerprint()
        if manifest.exists() and any(hashes.get(name) != digest for name, digest in json.loads(manifest.read_text()).items()):
            raise ValueError("An existing study input changed. Use a separate directory.")
        write_json(manifest, hashes)
    env = {**os.environ, "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "VECLIB_MAXIMUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
           "PYTHONPATH": os.pathsep.join([str(WORKSPACE), *([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])])}
    selection = args.method or list(METHODS)
    build_dir = args.acados_build_dir.resolve()
    prepare_solver(build_dir, {METHODS[method].controller for method in selection}, env)
    cases = [(route, method, base, offset)
             for route, base, offset, method in product(args.route or ROUTES, args.base or WINDS,
                                                        args.offset or OFFSETS, selection)]
    print(f"Running {len(cases)} route × method × wind cases on {args.jobs} workers.", flush=True)
    failed = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        jobs = [pool.submit(run_one, study, build_dir, *case, env, args.smoke_steps, args.reference_kernels)
                for case in cases]
        for index, future in enumerate(as_completed(jobs), 1):
            run_id, status, elapsed = future.result()
            print(f"{index}/{len(cases)} {run_id} exit={status} wall={elapsed:.1f}s", flush=True)
            if status:
                failed.append(run_id)
    if failed:
        raise SystemExit("Failed cases: " + ", ".join(failed))


if __name__ == "__main__":
    main()
