"""Run an offline closed-loop Fossen waypoint exercise with a synthetic chart."""

import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
WAYPOINTS = (("001", 39.9), ("002", 39.9009), ("003", 39.9018))


def write_inputs(output):
    chart = output / "synthetic_chart.nc"
    axis = np.linspace(-0.02, 0.02, 81)
    coordinates = {"lat": 39.9 + axis, "lon": 18.4 + axis}
    depth = np.full((len(axis), len(axis)), -100, dtype=np.int16)
    xr.Dataset({"elevation": (("lat", "lon"), depth)}, coords=coordinates).to_netcdf(chart)

    route = output / "smoke_route.gpx"
    points = "\n".join(
        f'<rtept lat="{lat}" lon="18.4"><name>{name}</name></rtept>'
        for name, lat in WAYPOINTS
    )
    route.write_text(
        '<gpx version="1.1" creator="Maritime World" xmlns="http://www.topografix.com/GPX/1/1">\n'
        f"<rte><name>Smoke test</name>{points}</rte>\n</gpx>\n", encoding="utf-8"
    )
    return chart, route


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("repro-output/smoke"))
    parser.add_argument("--controller", choices=("grid-mpc", "acados-direct", "acados-los"),
                        default="grid-mpc")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    chart, route = write_inputs(output)
    replay = output / "navigation.jsonl.gz"
    command = [
        sys.executable, "-B", "-m", "world.run_legacy_fossen",
        "--planner", "gamma", "--controller", args.controller, "--dynamics", "fossen",
        "--environment", "constant", "--wind-speed", "8", "--wind-direction", "180",
        "--global-map", str(chart), "--route-gpx", str(route), "--arrival-radius", "50",
        "--heading", "0", "--dt", "5", "--steps", "160",
        "--local-span", "945", "--local-resolution", "45", "--obstacle-buffer", "0",
        "--plot-every", "0", "--output-dir", str(output),
        "--output-gpx", str(output / "track.gpx"), "--record-replay", str(replay),
    ]
    environment = dict(os.environ)
    environment.setdefault("MPLBACKEND", "Agg")
    log = output / "simulator.log"
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream,
                                   stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise SystemExit(f"Simulator exited with status {completed.returncode}. See {log}.")
    events = []
    with gzip.open(replay, "rt", encoding="utf-8") as stream:
        for line in stream:
            navigation = json.loads(line).get("navigation") or {}
            if navigation.get("waypoint_distance_m", float("inf")) <= 50:
                waypoint = navigation["waypoint"]
                if not events or waypoint != events[-1]:
                    events.append(waypoint)
    if events != ["002", "003"]:
        raise SystemExit(f"Unexpected waypoint order in the smoke replay: {events}. See {log}.")
    print("Verified ordered captures within 50 m: " + ", ".join(events))
    print(f"Replay: {replay}")


if __name__ == "__main__":
    main()
