"""Verify causal history, batch semantics and frozen-backbone parity."""
from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import torch

from train_weather_encoder import SOURCE, load_scenarios
from weather_embedding import STUDY, WeatherEmbedding, causal_history, forcing_vector
from world.boat import BoatState
from world.currents import Current
from world.mark_weather_embedding import FrozenMarkWeatherEncoder
from world.weather import Wind
from world.waves import Wave
from world.world import WorldObservation


LEGACY = STUDY.with_name("otranto_temporal_hold_20261118") / "models/temporal_hold.pt"


def observation(stage, elapsed_seconds):
    wind_e, wind_n, current_e, current_n, height = forcing_vector(stage)
    wind_from = stage.polar.wind_from
    wind = Wind(wind_e, wind_n, stage.wind_speed, wind_from)
    current_to = np.degrees(np.arctan2(current_e, current_n)) % 360
    current = Current(current_e, current_n, np.hypot(current_e, current_n), current_to, 0.0)
    when = np.datetime64("2000-01-01T00:00:00") + np.timedelta64(elapsed_seconds, "s")
    return WorldObservation(when, BoatState(0, 0, 0, 0), wind, Wave(height, wind_from, 5.0),
                            current, None, None, None, ())


def verify():
    torch.set_num_threads(2)
    scenario = next(case for case in load_scenarios(SOURCE)
                    if case.split == "train" and case.future is not None)
    case = replace(scenario, change_at=450.0)
    different_future = replace(case, future=replace(case.future, wave_height=case.future.wave_height + 0.3))
    new_weather = np.array(forcing_vector(case.future))
    old_weather = np.array(forcing_vector(case.first))
    start = np.broadcast_to(old_weather, (12, 5))
    np.testing.assert_array_equal(causal_history(case, 0), start)
    np.testing.assert_array_equal(causal_history(case, 599), start)
    np.testing.assert_array_equal(causal_history(case, 600)[:-1], start[:-1])
    np.testing.assert_array_equal(causal_history(case, 600)[-1], new_weather)
    np.testing.assert_array_equal(causal_history(different_future, 599), start)

    rng = np.random.default_rng(711)
    targets = rng.normal(size=(18, 2))
    fresh = WeatherEmbedding(batch_size=8)
    elapsed = 900
    batch = fresh.encode(case, elapsed, targets)
    individual = np.concatenate([fresh.encode(case, elapsed, target[None]) for target in targets])
    assert batch.dtype == np.float32 and batch.shape == (len(targets), 48)
    np.testing.assert_allclose(batch, individual, atol=2e-6, rtol=1e-6)
    np.testing.assert_allclose(batch, fresh.encode(case, elapsed, 3 * targets), atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(fresh.encode(case, 599, targets), fresh.encode(different_future, 599, targets))
    same_present = replace(case, first=case.future, future=None)
    np.testing.assert_array_equal(causal_history(case, elapsed)[-1], causal_history(same_present, elapsed)[-1])
    history_effect = float(np.max(np.abs(batch - fresh.encode(same_present, elapsed, targets))))
    assert history_effect > 1e-4
    assert fresh.encode(case, 0, np.empty((0, 2))).shape == (0, 48)

    mixed = fresh.encode_histories(np.stack((causal_history(case, 0), causal_history(case, elapsed))),
                                  targets[:2])
    np.testing.assert_allclose(mixed, np.concatenate((fresh.encode(case, 0, targets[0:1]),
                                                      fresh.encode(case, elapsed, targets[1:2]))),
                               atol=2e-6, rtol=1e-6)
    legacy = WeatherEmbedding(LEGACY, legacy=True, batch_size=8)
    original = FrozenMarkWeatherEncoder(LEGACY)
    legacy_errors = []
    for moment in (0, 300, 449, 599, 600, 900, 1200, 1500):
        actual = observation(case.first if moment < case.change_at else case.future, moment)
        original.observe(actual)
        reference = np.stack([original.encode(actual, target).values for target in targets])
        compared = legacy.encode(case, moment, targets)
        np.testing.assert_allclose(compared, reference, atol=2e-6, rtol=1e-6)
        legacy_errors.append(float(np.max(np.abs(compared - reference))))

    return {
        "synthetic_verification_id": case.identifier,
        "future_shift_inside_sample_window_s": case.change_at,
        "first_future_observation_s": 600,
        "initial_padding": "12 copies of first observation",
        "pre_observation_future_intervention_embedding_delta": 0.0,
        "equal_present_different_past_embedding_max_delta": history_effect,
        "fresh_batch_versus_individual_max_absolute_error": float(np.max(np.abs(batch - individual))),
        "fresh_checkpoint_sha256": fresh.checkpoint_sha256,
        "legacy_public_encoder_max_absolute_error": max(legacy_errors),
        "legacy_parity_examples": len(legacy_errors) * len(targets),
        "embedding_shape": [len(targets), 48],
        "embedding_dtype": str(batch.dtype),
    }


if __name__ == "__main__":
    result = json.dumps(verify(), indent=2) + "\n"
    (STUDY / "weather_embedding_verification.json").write_text(result)
    print(result, end="")
