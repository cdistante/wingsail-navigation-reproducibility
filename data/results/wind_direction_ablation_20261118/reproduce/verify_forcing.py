"""Compare replay samples with native xarray archive queries at their recorded positions."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
import gzip
import json
from math import atan2, degrees, hypot
from pathlib import Path

from run_study import OFFSETS, ROUTES, STUDY, WINDS, write_json
from summarize import load_mission
from world.currents import Currents
from world.waves import Waves
from world.weather import Weather


def angle_diff(value, expected):
    return abs((value - expected + 180.0) % 360.0 - 180.0)


def sampled_observations(run, steps):
    selected = {0, steps // 2, steps}
    with gzip.open(run / "navigation.jsonl.gz", "rt", encoding="utf-8") as replay:
        next(replay)
        for step, line in enumerate(replay):
            if step in selected:
                yield json.loads(line)["observation"]


def compare(observation, sources, direction):
    position = observation["boat"]["latitude"], observation["boat"]["longitude"]
    arguments = (*position, observation["time"])
    weather, currents, waves = sources
    wind = weather.sample(*arguments)
    current = currents.sample(*arguments)
    wave = waves.sample(*arguments)
    logged_wind, logged_current, logged_wave = (observation[key] for key in ("wind", "current", "wave"))
    wind_speed = hypot(logged_wind["u"], logged_wind["v"])
    wind_bearing = (degrees(atan2(logged_wind["u"], logged_wind["v"])) + 180.0) % 360.0
    return {
        "wind_speed_mps": abs(wind.speed - wind_speed),
        "wind_from_deg": angle_diff(wind_bearing, direction),
        "current_u_mps": abs(current.u - logged_current["u"]),
        "current_v_mps": abs(current.v - logged_current["v"]),
        "wave_height_m": abs(wave.height - logged_wave["height"]),
        "wave_direction_deg": angle_diff(wave.direction, logged_wave["direction"]),
        "wave_period_s": abs(wave.peak_period - logged_wave["peak_period"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=STUDY)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    site_bounds = {"latitude_bounds": (38.0, 42.5), "longitude_bounds": (12.5, 21.5)}
    errors = defaultdict(float)
    samples, cases = 0, 0
    with ExitStack() as stack:
        sources = tuple(layer(**site_bounds) for layer in (Weather, Currents, Waves))
        for source in sources:
            stack.callback(source.close)
            source.load()
        for route in ROUTES:
            for base in WINDS:
                for offset in OFFSETS:
                    run = args.study / "runs" / f"{route}__{base:03d}{offset:+03d}__forecast"
                    if args.allow_partial and not (run / "execution.json").is_file():
                        continue
                    mission = load_mission(run)
                    direction = (base + offset) % 360
                    for observation in sampled_observations(run, mission["steps"]):
                        for field, delta in compare(observation, sources, direction).items():
                            errors[field] = max(errors[field], delta)
                            if delta > 1e-8:
                                raise ValueError(f"{run.name}: {field} differs by {delta}")
                        samples += 1
                    cases += 1
    if not args.allow_partial and cases != len(ROUTES) * len(WINDS) * len(OFFSETS):
        raise ValueError("Missing forecast cases.")
    verification = {
        "cases": cases, "replay_samples": samples,
        "selection": "First, middle and final frames of the classical forecast run in each route and wind cell",
        "maximum_error_vs_native_xarray": dict(errors),
    }
    write_json(args.output or args.study / "forcing_verification.json", verification)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
