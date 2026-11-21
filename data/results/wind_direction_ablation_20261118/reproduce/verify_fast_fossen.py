"""Verify the experiment's grid-MPC predictor against native Fossen."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import astuple, fields
import gzip
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np

from run_wind_scenario import WORKSPACE
from fast_fossen import FastFossenPrediction, install
from world.boat import BoatAction, BoatState
from world.currents import Current
from world.fossen import Fossen3DOFSaildroneDynamics
from world.mpc import SaildroneMPC
from world.waves import Wave
from world.weather import Wind
from world.world import WorldObservation


SEED = 20261118
PILOT_ARGS = [
    "--wind-from", "90", "--wind-offset", "5",
    "--planner", "transformer_diffusion",
    "--transformer-diffusion-checkpoint",
    "simulator_harmonic/transformer_runs_diff_e30/run_diffusion_g41_n3578_e30/best.pt",
    "--transformer-diffusion-device", "cpu",
    "--controller", "grid-mpc", "--dynamics", "fossen", "--environment", "marine",
    "--route-gpx", "current_route/opencpn_001_004_test.gpx",
    "--global-map", "data/bathymetry/gebco_2026_n44.0_s37.0_w14.0_e22.0.nc",
    "--arrival-radius", "50", "--dt", "5", "--heading", "45", "--flap=-10",
    "--time", "2026-10-01T13:00:00Z", "--steps", "220", "--plot-every", "0",
]


def inputs():
    rng = np.random.default_rng(SEED)
    for index in range(320):
        boat = BoatState(
            rng.uniform(38, 42), rng.uniform(13, 21), rng.uniform(0, 360),
            rng.uniform(0.0, 1.5), 0.0, rng.uniform(-15, 15), 0.0,
            None if index % 4 == 0 else rng.uniform(-0.4, 1.5),
            rng.uniform(-0.35, 0.35), rng.uniform(-0.18, 0.18),
        )
        direction, speed = rng.uniform(0, 360), rng.uniform(0, 18)
        angle = np.radians(direction)
        cu, cv = rng.uniform(-0.45, 0.45, 2)
        wind = Wind(-speed * np.sin(angle), -speed * np.cos(angle), speed, direction)
        wave = Wave(rng.uniform(0, 2.5), rng.uniform(0, 360), rng.uniform(3, 8)) if index % 3 else None
        current = Current(cu, cv, np.hypot(cu, cv), np.degrees(np.arctan2(cu, cv)) % 360, 0.0) if index % 4 else None
        environment = WorldObservation(
            np.datetime64("2026-10-01T13:00:00", "ns"), boat, wind, wave, current, None, None, None, (),
        )
        yield boat, BoatAction(rng.uniform(-35, 35), rng.uniform(-20, 20)), environment, (0.05, 1.0, 5.0, 15.0)[index % 4]


def numeric_parity():
    native = Fossen3DOFSaildroneDynamics()
    fast = FastFossenPrediction(native)
    rows = list(inputs())
    for row in rows[:4]:
        fast.step(*row)
    start = monotonic()
    baseline = np.array([astuple(native.step(*row)) for row in rows])
    scalar_elapsed = monotonic() - start
    start = monotonic()
    predictions = np.array([astuple(fast.step(*row)) for row in rows])
    fast_elapsed = monotonic() - start
    np.testing.assert_allclose(baseline, predictions, atol=1e-10, rtol=1e-12)
    mpc = SaildroneMPC()

    def actions():
        return [mpc.choose_action(native, boat, environment, 26.5) for boat, _, environment, _ in rows[:50]]

    start = monotonic()
    expected = actions()
    native_mpc_s = monotonic() - start
    start = monotonic()
    with install():
        observed = actions()
    fast_mpc_s = monotonic() - start
    assert [result.action for result in expected] == [result.action for result in observed]
    return {
        "seed": SEED, "random_steps": len(rows), "tested_dt_seconds": [0.05, 1.0, 5.0, 15.0],
        "max_absolute_errors": dict(zip((field.name for field in fields(BoatState)),
                                        np.max(np.abs(baseline - predictions), axis=0).tolist(), strict=True)),
        "reference_prediction_wall_s": scalar_elapsed, "fast_prediction_wall_s": fast_elapsed,
        "step_speedup": scalar_elapsed / fast_elapsed,
        "mpc_choices_exact": len(observed),
        "mpc_max_absolute_cost_error": max(abs(x.cost - y.cost) for x, y in zip(expected, observed, strict=True)),
        "reference_mpc_wall_s": native_mpc_s, "fast_mpc_wall_s": fast_mpc_s,
        "mpc_speedup": native_mpc_s / fast_mpc_s,
    }


def run_case(target: Path, *, with_acceleration: bool):
    target.mkdir()
    runner_directory = str(Path(__file__).resolve().parent)
    code = (
        f"import sys; sys.path.insert(0, {runner_directory!r}); "
        "from run_wind_scenario import main; "
    )
    code += "\nfrom fast_fossen import install\nwith install(): main()" if with_acceleration else "\nmain()"
    replay_path = target / "navigation.jsonl.gz"
    start = monotonic()
    with (target / "output.log").open("w") as stream:
        result = subprocess.run(
            [sys.executable, "-B", "-c", code, *PILOT_ARGS,
             "--output-dir", str(target), "--record-replay", str(replay_path)],
            cwd=WORKSPACE, stdout=stream, stderr=subprocess.STDOUT, check=False,
        )
    elapsed = monotonic() - start
    if result.returncode:
        raise RuntimeError((target / "output.log").read_text())
    with gzip.open(replay_path, "rt") as stream:
        next(stream)
        frames = [json.loads(line) for line in stream]
    return elapsed, frames


def route_parity():
    with TemporaryDirectory(prefix="wind_ablation_fossen_pilot_") as directory:
        root = Path(directory)
        with ThreadPoolExecutor(max_workers=2) as executor:
            reference = executor.submit(run_case, root / "reference", with_acceleration=False)
            fast = executor.submit(run_case, root / "fast", with_acceleration=True)
            (reference_s, expected), (fast_s, observed) = reference.result(), fast.result()
    command_fields = ("rudder_command_deg", "flap_command_deg")
    assert all(a["state"] == b["state"] and a["observation"] == b["observation"]
               and all(a["navigation"].get(key) == b["navigation"].get(key) for key in command_fields)
               for a, b in zip(expected, observed, strict=True))
    return {
        "arguments": PILOT_ARGS, "identical_state_and_observation_frames": len(observed),
        "identical_applied_commands": len(observed) - 1, "reference_wall_s": reference_s,
        "fast_wall_s": fast_s, "speedup_without_field_reuse": reference_s / fast_s,
        "wall_time_method": "Concurrent isolated subprocesses. Both arms use native field calculation.",
    }


def main():
    study = Path(__file__).resolve().parents[1]
    result = {
        "accelerated_component": "Grid-MPC Fossen predictions. The applied-action plant is unchanged.",
        "predictor_sha256": sha256(Path(__file__).with_name("fast_fossen.py").read_bytes()).hexdigest(),
        "numeric": numeric_parity(), "closed_loop": route_parity(),
    }
    (study / "fast_fossen_parity.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
