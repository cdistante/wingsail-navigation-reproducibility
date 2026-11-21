"""Check reduced AIS sweeps against native reports and independently sampled truth."""

from dataclasses import replace
import unittest

import numpy as np

from data.results.learned_safe_mark_policy.reproduce.ais_scenarios import ReducedAISPreview, available_cases
from world._geo import local_east_north_metres
from world.learned_mark_policy import CandidateBatch, FEATURE_NAMES


def candidate(dt=60., turn=30.):
    features = np.zeros((1, 1, len(FEATURE_NAMES)), dtype=np.float32)
    features[..., FEATURE_NAMES.index("decision_seconds_300")] = dt / 300.
    features[..., FEATURE_NAMES.index("turn_seconds_180")] = turn / 180.
    drift = np.array([[[.2, -.1]]]) * min(turn, dt)
    displacement = drift + np.array([[[.8, .1]]]) * max(0., dt-turn)
    return CandidateBatch(features, displacement, -displacement, np.zeros((1, 1)),
                          np.full((1, 1), np.inf), np.zeros((1, 1)), drift)


def dense_margins(preview, elapsed, batch, dt, applied=None):
    horizon = dt if applied is None else applied
    times = np.linspace(0., horizon, int(horizon/.05)+1)
    turn = min(dt, float(batch.features[0, 0, FEATURE_NAMES.index("turn_seconds_180")] * 180.))
    drift = batch.turn_displacement[0, 0]
    sailing = batch.displacement[0, 0] - drift
    own = np.minimum(times/max(turn, 1e-12), 1.)[:, None] * drift
    own += np.maximum(0., times-turn)[:, None] / max(dt-turn, 1e-12) * sailing
    prediction = preview._snapshot(float(elapsed)).prediction
    report = np.linalg.norm(own[:, None] - prediction.centres_at(times), axis=-1) - prediction.radii_at(times)
    # Native provider's point sampler is independent of the adapter's vector
    # formula; apply the documented route rotation after geographic decoding.
    truth = np.array([
        [local_east_north_metres(t._at_epoch(*preview.origin, elapsed + seconds,
                                            preview.fixture._manoeuvres.get(t.mmsi)).latitude,
                                 t._at_epoch(*preview.origin, elapsed + seconds,
                                            preview.fixture._manoeuvres.get(t.mmsi)).longitude,
                                 *preview.origin) for t in preview.targets]
        for seconds in times
    ]) @ preview.rotation.T
    truth_margin = np.linalg.norm(own[:, None] - truth, axis=-1) - prediction.radii_at(times)
    return float(report.min()), float(truth_margin.min())


class ReducedAISScenariosTests(unittest.TestCase):
    def test_vector_truth_matches_native_analytical_motion_and_rotation(self):
        for name in available_cases():
            for bearing in (0., 37., 90., 270.):
                preview = ReducedAISPreview(name, bearing)
                for elapsed in (0., 85., 95., 105., 120., 130., 150., 180., 300.):
                    expected = []
                    for target in preview.targets:
                        t = target._at_epoch(*preview.origin, elapsed, preview.fixture._manoeuvres.get(target.mmsi))
                        expected.append(local_east_north_metres(t.latitude, t.longitude, *preview.origin))
                    np.testing.assert_allclose(preview.truth_positions(elapsed), np.asarray(expected) @ preview.rotation.T,
                                               atol=1e-8, rtol=0.)

    def test_dropouts_are_observed_on_the_sixty_second_decision_grid(self):
        for name in ("mixed_long_dropout", "mixed_sharp_dropout"):
            preview = ReducedAISPreview(name, 90.)
            for elapsed in (120., 180.):
                diagnostics = preview.report_diagnostics(elapsed)
                self.assertEqual(diagnostics["maximum_report_age_seconds"], 40.)
                self.assertEqual(diagnostics["dropped_reports_at_observation"], 6)
        self.assertEqual(ReducedAISPreview("mixed_dropout", 90.).report_diagnostics(120.)["dropped_reports_at_observation"], 0)

    def test_report_and_truth_bounds_are_conservative_against_native_samples(self):
        for name in ("crossing", "mixed", "mixed_long_dropout", "mixed_sharp_dropout"):
            preview = ReducedAISPreview(name, 90.)
            batch = candidate()
            result = preview.assess(90., [0., 0.], batch, 60.)
            report, truth = dense_margins(preview, 90., batch, 60.)
            self.assertLessEqual(result.report_margin_m[0, 0], report + 1e-8)
            self.assertLessEqual(result.truth_margin_m[0, 0], truth + 1e-8)
            self.assertEqual(result.allowed.shape, (1, 1))
            for hit in (10., 45.):
                bounded = preview.assess_applied(90., [0., 0.], batch, 60., 0, 0, hit)
                sampled = dense_margins(preview, 90., batch, 60., hit)
                self.assertLessEqual(bounded[0], sampled[0] + 1e-8)
                self.assertLessEqual(bounded[1], sampled[1] + 1e-8)

    def test_full_planned_interval_and_early_capture_are_distinct(self):
        preview = ReducedAISPreview("head_on", 90.)
        batch = candidate(300., 0.)
        full = preview.assess(0., [0., 0.], batch, 300.)
        early = preview.assess_applied(0., [0., 0.], batch, 300., 0, 0, 10.)
        self.assertFalse(full.allowed[0, 0])
        self.assertGreater(min(early), 0.)
        with self.assertRaisesRegex(ValueError, "actual decision duration"):
            preview.assess(0., [0., 0.], batch, 60.)

    def test_per_row_durations_and_far_contact_bound(self):
        first, second = candidate(60., 20.), candidate(120., 70.)
        batch = replace(first, **{name: np.concatenate((getattr(first, name), getattr(second, name)), axis=0)
                                 for name in first.__dataclass_fields__})
        preview = ReducedAISPreview("mixed_sharp_dropout", 45.)
        result = preview.assess(180., [[-10000., 10000.], [-12000., 14000.]], batch, [60., 120.])
        self.assertTrue(result.allowed.all())
        self.assertTrue(result.used_far_lower_bound)
        np.testing.assert_array_equal(result.screened_seconds, [[60.], [120.]])
        self.assertEqual(result.maximum_committed_report_age_seconds, 160.)

    def test_lane_offsets_move_truth_and_reports_before_rotation(self):
        offsets = {999000101: [140., 0.], 999000102: [0., 180.], 999000103: [0., -180.]}
        for name in ("crossing", "mixed_sharp_dropout"):
            for bearing in (0., 90., 217.):
                original = ReducedAISPreview(name, bearing)
                shifted = ReducedAISPreview(name, bearing, track_offsets_EN=offsets)
                expected = np.array([offsets[identity] for identity in original.identities]) @ original.rotation.T
                for elapsed in (0., 90., 120., 180.):
                    np.testing.assert_allclose(shifted.truth_positions(elapsed)-original.truth_positions(elapsed),
                                               expected, atol=1e-10, rtol=0.)
                    np.testing.assert_allclose(shifted._snapshot(elapsed).prediction.positions_metres
                                               -original._snapshot(elapsed).prediction.positions_metres,
                                               expected, atol=1e-8, rtol=0.)


if __name__ == "__main__":
    unittest.main()
