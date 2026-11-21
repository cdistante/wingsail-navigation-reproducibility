"""Run matched native Otranto with the frozen hybrid first-tack adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path.cwd()))

import numpy as np

from world import run_legacy_fossen as runner
from world.hybrid_tack_guidance import HybridForecastTackPlanner, HybridTackGuidance
from world.legacy_planner_adapter import LegacyPlannerAdapter


REPRODUCE = Path(__file__).resolve().parent
RESULTS = REPRODUCE.parents[1]


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    options = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    options.add_argument("--hybrid-forecast-tacks", action="store_true", required=True)
    options.add_argument("--hybrid-model-dir", type=Path, required=True)
    opt, native_args = options.parse_known_args()
    receipt = json.loads((opt.hybrid_model_dir / "training_receipt.json").read_text())
    selection_path = RESULTS / "learned_weather_ais_hybrid" / "validation_selection.json"
    selection = json.loads(selection_path.read_text())
    if (checksum(opt.hybrid_model_dir / "fresh_weather_encoder.pt") != receipt["weather_encoder_sha256"]
            or checksum(opt.hybrid_model_dir / "joint_critic.npz") != receipt["models_sha256"]["joint"]
            or selection["learner_checkpoint_sha256"] != receipt["models_sha256"]["joint"]
            or selection["learner_strength"] != receipt["strengths"]["joint"]["best_nonzero"]):
        raise ValueError("The native experiment requires the frozen encoder, critic and validation receipt.")

    original_parser = runner.parser

    def parser():
        result = original_parser()
        result.set_defaults(native_hybrid_enabled=True)
        return result

    controls, _ = parser().parse_known_args(native_args)
    if (not controls.forecast_tacks or not controls.ais_predictive_fallback or not controls.colreg
            or not controls.native_control_timing or controls.learned_tack_model is not None
            or controls.learned_arrival_model is not None or controls.tack_plan_replay is not None):
        raise ValueError("Hybrid guidance requires forecast tacks, strict AIS/COLREG and native timing.")
    hybrid = HybridTackGuidance(
        opt.hybrid_model_dir, gain_seconds=selection["minimum_forecast_gain_s"], strength=selection["learner_strength"],
        acceleration_bound=controls.ais_acceleration_bound, reserve=controls.ais_clearance_reserve,
        report_delay=controls.ais_report_delay, control_seconds=controls.dt,
    )

    class NativePlanner(HybridForecastTackPlanner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, hybrid=hybrid, **kwargs)

    class ObserveWeatherPlanner(LegacyPlannerAdapter):
        def plan(self, local_map, observation, goal, **kwargs):
            hybrid.observe(observation)
            grid = local_map.tensor
            blocked = local_map.static_blocked.copy()
            if kwargs.get("extra_blocked") is not None:
                blocked |= kwargs["extra_blocked"]
            start = self._boat_cell(grid.cells)
            blocked[start] = False

            def clear(velocity):
                speed = np.linalg.norm(velocity)
                if speed < .01:
                    return False
                end = start[0] - velocity[1]/speed, start[1] + velocity[0]/speed
                return self._segment_is_clear(blocked, start, end)

            hybrid.chart_allows = clear
            return super().plan(local_map, observation, goal, **kwargs)

    print("Native hybrid policy: " + json.dumps({
        "weather_encoder_sha256": receipt["weather_encoder_sha256"],
        "critic_sha256": receipt["models_sha256"]["joint"],
        "forecast_gain_seconds": selection["minimum_forecast_gain_s"],
        "physics_forecast_scope": "native archive polar windows; at most 48 minutes; no future AIS reports",
    }), flush=True)
    with (patch.object(runner, "ForecastTackPlanner", NativePlanner),
          patch.object(runner, "LegacyPlannerAdapter", ObserveWeatherPlanner),
          patch.object(runner, "parser", parser),
          patch.object(sys, "argv", [sys.argv[0], *native_args])):
        runpy.run_path(str(REPRODUCE / "run_archive.py"), run_name="__main__")


if __name__ == "__main__":
    main()
