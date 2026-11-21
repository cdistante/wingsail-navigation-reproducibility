"""Rebuild the 355-degree forecast at the previously failing tack decision."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from run_archive import directed_provider, regional_provider
from world.currents import Currents
from world.fossen import Fossen3DOFSaildroneDynamics
from world.hybrid_tack_guidance import HybridForecastTackPlanner, HybridTackGuidance
from world.navigation_replay import _decode_episode_frame
from world.scenario import Waypoint
from world.waves import Waves
from world.world import MaritimeWorld


STUDY = Path(__file__).resolve().parents[1]
RESULTS = STUDY.parent
SNAPSHOT_STEP = 10958
PRIOR = RESULTS / "otranto_native_hybrid" / "runs" / "crossing_fan__otranto_355__hybrid"
MODELS = RESULTS / "learned_weather_ais_policy"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    selection = json.loads((RESULTS / "learned_weather_ais_hybrid" / "validation_selection.json").read_text())
    policy = HybridTackGuidance(
        MODELS, gain_seconds=selection["minimum_forecast_gain_s"], strength=selection["learner_strength"],
        acceleration_bound=.026, reserve=3.0, report_delay=20.0, control_seconds=5.0,
    )

    # Match the recorded causal five-minute history and exact pre-tack state.
    path = PRIOR / "navigation.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        header = json.loads(next(stream))
        mark = header["route"][1]
        goal = Waypoint(mark["latitude"], mark["longitude"])
        for raw in stream:
            row = json.loads(raw)
            step = row["navigation"]["step"]
            if step < SNAPSHOT_STEP - 780:
                continue
            obs = _decode_episode_frame(row).observation
            policy.observe(obs)
            if step == SNAPSHOT_STEP:
                old = json.loads(next(stream))["navigation"]
                break
        else:
            raise ValueError("The matched replay is missing the baseline decision frame.")
    if old["hybrid_tack_decision"] != "accepted first trim":
        raise ValueError("The retained historical result is not the failed veto case.")

    weather = directed_provider(355.)()
    world = MaritimeWorld(weather=weather, waves=regional_provider(Waves)(),
                          currents=regional_provider(Currents)()).load()
    try:
        # Outbound uses the unrefined 120-second scheduler. At this exact
        # historical action the scheduler had no locked or scheduled switch.
        planner = HybridForecastTackPlanner(
            world, Fossen3DOFSaildroneDynamics(), 45., 50., hybrid=policy,
        )
        reference = planner.reference(obs, goal)
        decision = planner.maneuver_telemetry.copy()
    finally:
        world.close()
    assert reference is not None
    assert decision["hybrid_tack_decision"] == "veto forecast miss", decision
    assert decision["hybrid_tack_physics_schedule_class"] == 0
    assert decision["hybrid_tack_physics_eta_s"] > 120.0
    assert decision["hybrid_tack_learned_schedule_class"] in (1, 2)
    assert decision["hybrid_tack_learned_eta_s"] is None
    assert decision["hybrid_tack_proposed_bow_deg"] == 260.0
    assert reference.bow == decision["hybrid_tack_physics_bow_deg"] == 40.5
    assert decision["hybrid_tack_selected_first_mode"] == decision["hybrid_tack_physics_first_mode"]
    output = {
        "status": "PASS", "predecision_fix": SNAPSHOT_STEP, "observation_utc": str(obs.time),
        "previous_decision": {key: value for key, value in old.items() if key.startswith("hybrid_tack_")},
        "corrected_decision": decision, "emitted_bow_deg": reference.bow,
        "scope": "Forecast-only reconstruction at the saved predecision state; full closed-loop comparison is separate.",
        "previous_replay_sha256": sha256(path),
        "production_adapter_sha256": sha256(STUDY.parents[2] / "world/hybrid_tack_guidance.py"),
    }
    (STUDY / "veto_regression_probe.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
