"""Rerun one of the four archived Otranto layline comparisons in a fresh workspace."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path("data/results")
ETA = RESULTS / "otranto_layline_eta"
CONTROL = RESULTS / "otranto_closest_layline_ablation"
NATIVE = RESULTS / "otranto_native_schedule_veto/reproduce"
ENTRY_POINT = NATIVE / "run_native_hybrid.py"
ARCHIVED_ENTRY_POINTS = (ENTRY_POINT, NATIVE / "run_archive.py")
INPUT_FLAGS = ("--traffic-fixture", "--underway-traffic-fixture", "--route-gpx", "--hybrid-model-dir")
THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def verified_sources(receipt, arm):
    snapshot = ROOT / ETA / "provenance" / arm
    sources = {}
    for name, checksum in receipt["source_sha256"].items():
        relative = Path(name)
        if relative.parts[0] != "world" and relative not in (Path("config.py"), *ARCHIVED_ENTRY_POINTS):
            continue
        copy = snapshot / relative
        source = copy if copy.is_file() else ROOT / relative
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Runtime source differs from the archived {arm} receipt: {relative}")
        if copy.is_file():
            sources[relative] = source
    if not sources:
        raise ValueError("The archived simulator overlay is missing.")
    return sources


def set_operand(parameters, flag, value):
    parameters[parameters.index(flag) + 1] = str(value)


def prepared_arguments(receipt, wind, arm, output):
    command = receipt["command"]
    if Path(command[2]).name != ENTRY_POINT.name:
        raise ValueError("Unexpected solver in the archived command.")
    parameters = command[3:]
    if (parameters[parameters.index("--wind-from") + 1] != str(wind)
            or ("--compare-waypoint-layline-eta" in parameters) != (arm == "eta")
            or parameters[parameters.index("--arrival-radius") + 1] != "50"
            or "--colreg" not in parameters or "--time-aware-ais" not in parameters):
        raise ValueError("The archived comparison does not match the requested wind, arm and protocol.")
    for flag in INPUT_FLAGS:
        relative = Path(parameters[parameters.index(flag) + 1])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Expected a checkout-relative input for {flag}: {relative}")
        source = ROOT / relative
        if not source.exists():
            raise FileNotFoundError(source)
        set_operand(parameters, flag, source)
    set_operand(parameters, "--output-dir", output)
    set_operand(parameters, "--output-gpx", output / "track.gpx")
    set_operand(parameters, "--record-replay", output / "navigation.jsonl.gz")
    set_operand(parameters, "--acados-build-dir", output / "acados_build")
    return parameters


def make_workspace(workspace, overrides):
    shutil.copytree(ROOT / "world", workspace / "world", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copyfile(ROOT / "config.py", workspace / "config.py")
    for relative, original in overrides.items():
        shutil.copyfile(original, workspace / relative)
    (workspace / "simulator_harmonic").symlink_to(ROOT / "simulator_harmonic", target_is_directory=True)
    (workspace / "data").mkdir()
    for name in ("results", "bathymetry", "weather", "waves", "currents"):
        (workspace / "data" / name).symlink_to(ROOT / "data" / name, target_is_directory=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wind", type=int, choices=(5, 355), required=True)
    parser.add_argument("--arm", choices=("off", "eta"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-command", action="store_true", help="Validate inputs and print the command.")
    args = parser.parse_args()
    output = args.output.resolve()
    treatment = ETA if args.arm == "eta" else CONTROL
    run_id = f"crossing_fan__otranto_{args.wind:03d}__layline_{args.arm}"
    path = ROOT / treatment / "runs" / run_id / "execution.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    parameters = prepared_arguments(receipt, args.wind, args.arm, output)
    sources = verified_sources(receipt, args.arm)
    command = [sys.executable, "-B", str(ROOT / ENTRY_POINT), *parameters]
    if args.check_command:
        print(json.dumps({"run": run_id, "checked_overlays": [str(path) for path in sources],
                          "command": command}, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    environment.update({name: "1" for name in THREAD_VARIABLES})
    environment.setdefault("MPLBACKEND", "Agg")
    with TemporaryDirectory(prefix="maritime_otranto_") as temporary:
        workspace = Path(temporary)
        make_workspace(workspace, sources)
        log = output / "simulator.log"
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=workspace, env=environment,
                                    stdout=stream, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise SystemExit(f"Simulator exited with status {result.returncode}. See {log}.")
    print(f"Saved replay to {output / 'navigation.jsonl.gz'}.")


if __name__ == "__main__":
    main()
