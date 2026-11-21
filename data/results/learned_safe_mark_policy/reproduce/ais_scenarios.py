"""Vectorized AIS screening for the reduced time-to-mark experiments.

These are the actual V2 synthetic fixtures and receiver timings. Their native
eastbound own-ship axis (90 degrees) is rotated onto the route-leg bearing.
The own trajectory is only ``candidate_batch``'s turn-drift and sail chords;
this module is not the native Fossen/AIS guardian and makes no COLREG claim.

All of the supplied action duration is screened. Reports are held at their
actual generation age at the decision, with the uncapped native uncertainty
growth and a 3 m reserve. The report mask never consults future truth. Truth
margins are separate conservative diagnostics of the analytical manoeuvres.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

from world._geo import local_east_north_metres, offset_latitude_longitude
from world.ais_avoidance import AISPrediction
from world.boat import BoatState
from world.learned_mark_policy import FEATURE_NAMES
from world.synthetic_ais import SyntheticAIS, SyntheticAISReporting
from world.world import WorldObservation


_CASES = (
    "crossing", "head_on", "overtaking", "mixed", "mixed_delayed",
    "mixed_dropout", "mixed_long_dropout", "mixed_sharp_dropout",
)
_DROP_WINDOWS = ((85.0, 105.0), (115.0, 135.0))
_LONG_DROP_WINDOWS = ((85.0, 125.0), (145.0, 195.0))
_TURN_FEATURE = FEATURE_NAMES.index("turn_seconds_180")


def available_cases() -> tuple[str, ...]:
    """Return the stable names accepted by ``ReducedAISPreview``."""
    return _CASES


@dataclass(frozen=True)
class AISPreviewAssessment:
    """Per-row/per-candidate full-duration margins; only reports set allowed."""

    allowed: np.ndarray
    report_margin_m: np.ndarray
    truth_margin_m: np.ndarray
    maximum_report_age_seconds: float
    maximum_committed_report_age_seconds: float
    dropped_reports_at_observation: int
    screened_seconds: np.ndarray
    used_far_lower_bound: bool = False


@dataclass(frozen=True)
class _Snapshot:
    prediction: AISPrediction
    maximum_age: float
    lost_reports: int


def _nearest_chord_distance(start, finish):
    delta = finish - start
    length = np.sum(delta * delta, axis=-1)
    fraction = np.divide(-np.sum(start * delta, axis=-1), length,
                         out=np.zeros_like(length), where=length > 1e-16)
    nearest = start + np.clip(fraction, 0., 1.)[..., None] * delta
    return np.linalg.norm(nearest, axis=-1)


class ReducedAISPreview:
    """Cached native reports and vectorized two-chord candidate screening.

    ``own_position_EN`` is relative to the start of this encounter/route leg,
    and has shape (2,) or (rows, 2). ``batch`` is the actual output of
    ``world.learned_mark_policy.candidate_batch``. ``duration_seconds`` must
    be the same scalar or per-row interval passed to ``candidate_batch``.
    ``assess`` returns arrays with shape ``batch.features.shape[:-1]``.

    A fresh instance starts a new encounter clock. Preserve a single instance
    and elapsed clock through an encounter; do not reset it after a detour.
    """

    def __init__(
        self, case_name: str, route_bearing_deg: float, *,
        origin_latitude: float = 40.225, origin_longitude: float = 18.75,
        clearance_metres: float = 25., reserve_metres: float = 3.,
        acceleration_bound_mps2: float = .026, chord_seconds: float = 5.,
        track_offsets_EN: dict[int, tuple[float, float] | list[float]] | None = None,
    ) -> None:
        if case_name not in _CASES:
            raise ValueError(f"Unknown AIS case {case_name!r}; choose from {_CASES}.")
        if not np.isfinite(route_bearing_deg):
            raise ValueError("Route bearing must be finite.")
        if not np.isfinite(chord_seconds) or chord_seconds <= 0:
            raise ValueError("AIS chord interval must be positive and finite.")
        if any(not np.isfinite(x) or x < 0 for x in (clearance_metres, reserve_metres, acceleration_bound_mps2)):
            raise ValueError("AIS clearance, reserve and acceleration bound must be nonnegative.")
        self.case_name = case_name
        self.route_bearing_deg = float(route_bearing_deg) % 360
        self.rotation_degrees = self.route_bearing_deg - 90.
        angle = np.radians(self.rotation_degrees)
        self.rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        self.origin = float(origin_latitude), float(origin_longitude)
        self.clearance_metres = float(clearance_metres)
        self.reserve_metres = float(reserve_metres)
        self.acceleration_bound_mps2 = float(acceleration_bound_mps2)
        self.chord_seconds = float(chord_seconds)
        scenario = case_name if case_name in SyntheticAIS.available_scenarios() else "mixed"
        reporting = None
        if case_name == "mixed_delayed":
            reporting = SyntheticAISReporting(5., 10.)
        elif case_name == "mixed_dropout":
            reporting = SyntheticAISReporting(5., 10., _DROP_WINDOWS)
        elif case_name in ("mixed_long_dropout", "mixed_sharp_dropout"):
            reporting = SyntheticAISReporting(5., 10., _LONG_DROP_WINDOWS)
        self.fixture = SyntheticAIS(scenario, *self.origin, reporting=reporting,
                                    sharp_manoeuvres=case_name == "mixed_sharp_dropout").load()
        self.targets = self.fixture._SCENARIOS[scenario]
        self.identities = tuple(target.mmsi for target in self.targets)
        offsets = {} if track_offsets_EN is None else track_offsets_EN
        known_identities = {target.mmsi for group in SyntheticAIS._SCENARIOS.values() for target in group}
        if any(int(identity) not in known_identities for identity in offsets):
            raise ValueError("AIS track offset identities must belong to a registered fixture.")
        self.track_offsets_EN = {int(identity): np.asarray(value, dtype=float).copy()
                                 for identity, value in offsets.items()}
        if any(value.shape != (2,) or not np.isfinite(value).all() for value in self.track_offsets_EN.values()):
            raise ValueError("AIS track offsets must be finite native-frame EN pairs.")
        self._initial = np.array([[t.initial_east_metres, t.initial_north_metres] for t in self.targets])
        self._initial += np.array([self.track_offsets_EN.get(t.mmsi, np.zeros(2)) for t in self.targets])
        angles = np.radians([t.course_degrees for t in self.targets])
        speed = np.array([t.speed_metres_per_second for t in self.targets])
        self._velocity = speed[:, None] * np.column_stack((np.sin(angles), np.cos(angles)))
        self._change = np.zeros_like(self._velocity)
        self._start = np.zeros(len(self.targets))
        self._duration = np.ones(len(self.targets))
        for i, target in enumerate(self.targets):
            manoeuvre = self.fixture._manoeuvres.get(target.mmsi)
            if manoeuvre is not None:
                a = np.radians(manoeuvre.course_degrees)
                finish_velocity = manoeuvre.speed_metres_per_second * np.array([np.sin(a), np.cos(a)])
                self._change[i] = finish_velocity - self._velocity[i]
                self._start[i] = manoeuvre.start_seconds
                self._duration[i] = manoeuvre.finish_seconds - manoeuvre.start_seconds
        self._acceleration = np.linalg.norm(self._change, axis=-1) / self._duration
        self._maximum_truth_speed = np.maximum(np.linalg.norm(self._velocity, axis=-1),
                                               np.linalg.norm(self._velocity + self._change, axis=-1))

    @lru_cache(maxsize=512)
    def _snapshot(self, elapsed_s: float) -> _Snapshot:
        reports = self.fixture.targets(elapsed_s)
        rotated = []
        for report in reports:
            position = np.array(local_east_north_metres(report.latitude, report.longitude, *self.origin))
            position += self.track_offsets_EN.get(report.mmsi, np.zeros(2))
            east, north = self.rotation @ position
            lat, lon = offset_latitude_longitude(*self.origin, east, north)
            rotated.append(replace(report, latitude=float(lat), longitude=float(lon),
                                   course=(report.course + self.rotation_degrees) % 360.,
                                   heading=(report.heading + self.rotation_degrees) % 360.))
        observation = WorldObservation(
            np.datetime64("2000-01-01", "ns") + np.timedelta64(int(round(elapsed_s * 1e9)), "ns"),
            BoatState(*self.origin, 90., 0.), None, None, None, None, None, None, tuple(rotated),
        )
        prediction = AISPrediction.from_observation(
            observation, safety_buffer_metres=self.clearance_metres,
            acceleration_bound_mps2=self.acceleration_bound_mps2,
            clearance_reserve_metres=self.reserve_metres,
        )
        lost = 0
        if self.fixture.reporting is not None:
            r = self.fixture.reporting
            epoch = r.latest_epoch(elapsed_s)
            expected = SyntheticAISReporting(r.interval_seconds, r.delay_seconds).latest_epoch(elapsed_s)
            lost = int(round((expected - epoch) / r.interval_seconds))
        return _Snapshot(prediction, float(prediction.report_ages_seconds.max()), lost)

    def truth_positions(self, elapsed_s) -> np.ndarray:
        """Analytical native target truth, shaped (..., contacts, EN)."""
        times = np.asarray(elapsed_s, dtype=float)
        if not np.isfinite(times).all() or np.any(times < 0):
            raise ValueError("AIS truth requires finite nonnegative elapsed times.")
        turning = np.clip(times[..., None] - self._start, 0., self._duration)
        after = np.maximum(0., times[..., None] - self._start - self._duration)
        integrated = .5 * turning**2 / self._duration + after
        native = self._initial + times[..., None, None] * self._velocity + integrated[..., None] * self._change
        return native @ self.rotation.T

    def report_diagnostics(self, elapsed_s: float) -> dict:
        """Expose receiver age/loss and exact profile without affecting decisions."""
        snapshot = self._snapshot(float(elapsed_s))
        r = self.fixture.reporting
        return {"case": self.case_name, "scenario": self.fixture.scenario,
                "route_bearing_deg": self.route_bearing_deg, "identities": list(self.identities),
                "report_ages_seconds": snapshot.prediction.report_ages_seconds.tolist(),
                "maximum_report_age_seconds": snapshot.maximum_age,
                "dropped_reports_at_observation": snapshot.lost_reports,
                "report_interval_seconds": None if r is None else r.interval_seconds,
                "report_delay_seconds": 0. if r is None else r.delay_seconds,
                "drop_windows": [] if r is None else list(r.drop_windows),
                "track_offsets_EN": {identity: value.tolist() for identity, value in self.track_offsets_EN.items()},
                "sharp_manoeuvres": self.case_name == "mixed_sharp_dropout"}

    def assess(
        self, elapsed_s: float, own_position_EN, batch, duration_seconds, *, applied_seconds=None,
    ) -> AISPreviewAssessment:
        """Screen the whole step, or explicitly truncate its actual applied path.

        ``applied_seconds`` optionally has shape (rows, candidates), bounded
        by the original decision duration. For terminal capture diagnostics,
        pass ``minimum(duration_seconds, batch.captured_after_s)``. This
        reconstructs the partial turn/sail displacement rather than treating
        a full endpoint as reached early. The usual planning mask omits this
        argument and checks the full proposed interval.

        Returned margins are conservative lower bounds. Far-away traffic can
        use a triangle-inequality bound without constructing all subchords;
        ``used_far_lower_bound`` identifies that fast path.
        """
        elapsed_s = float(elapsed_s)
        if not np.isfinite(elapsed_s) or elapsed_s < 0:
            raise ValueError("AIS decision clock must be finite and nonnegative.")
        shape = np.shape(batch.displacement)
        if len(shape) != 3 or shape[-1] != 2:
            raise ValueError("Expected candidate displacements shaped (rows, candidates, 2).")
        rows, candidates, _ = shape
        own = np.broadcast_to(np.asarray(own_position_EN, dtype=float), (rows, 2))
        dt = np.broadcast_to(np.asarray(duration_seconds, dtype=float), (rows,))
        if (not np.isfinite(own).all() or not np.isfinite(dt).all() or np.any(dt <= 0)
                or not np.isfinite(batch.displacement).all() or not np.isfinite(batch.turn_displacement).all()):
            raise ValueError("AIS trajectories and durations must be finite, with positive durations.")
        planned_dt = np.asarray(batch.features)[..., FEATURE_NAMES.index("decision_seconds_300")] * 300.
        # candidate_batch stores features as float32, including dt/300.
        if not np.allclose(planned_dt, dt[:, None], atol=5e-5, rtol=1e-6):
            raise ValueError("AIS duration must equal candidate_batch's actual decision duration.")
        turning = np.minimum(dt[:, None], np.asarray(batch.features)[..., _TURN_FEATURE] * 180.)
        horizon = np.broadcast_to(dt[:, None], (rows, candidates))
        if applied_seconds is not None:
            horizon = np.broadcast_to(np.asarray(applied_seconds, dtype=float), (rows, candidates))
            if not np.isfinite(horizon).all() or np.any(horizon < 0.) or np.any(horizon > dt[:, None]):
                raise ValueError("Applied AIS sweep duration must be between zero and the decision duration.")
        snapshot = self._snapshot(elapsed_s)
        prediction = snapshot.prediction
        partial_turn = np.divide(np.minimum(horizon, turning), turning,
                                 out=np.zeros_like(horizon), where=turning > 1e-12)
        partial_sail = np.divide(np.maximum(0., horizon - turning), dt[:, None] - turning,
                                 out=np.zeros_like(horizon), where=dt[:, None] - turning > 1e-12)
        own_length = (partial_turn * np.linalg.norm(batch.turn_displacement, axis=-1)
                      + partial_sail * np.linalg.norm(batch.displacement - batch.turn_displacement, axis=-1))
        radius_end = prediction.radii_at(horizon)
        report_distance = np.linalg.norm(own[:, None, :] - prediction.positions_metres[None, :, :], axis=-1)
        report_lower = (report_distance[:, None, :] - own_length[..., None]
                        - horizon[..., None] * np.linalg.norm(prediction.velocities_mps, axis=-1) - radius_end).min(axis=-1)
        truth_distance = np.linalg.norm(own[:, None, :] - self.truth_positions(elapsed_s)[None, :, :], axis=-1)
        truth_lower = (truth_distance[:, None, :] - own_length[..., None]
                       - horizon[..., None] * self._maximum_truth_speed - radius_end).min(axis=-1)
        if np.all(report_lower > 0.) and np.all(truth_lower > 0.):
            return AISPreviewAssessment(report_lower > 0., report_lower, truth_lower, snapshot.maximum_age,
                                        snapshot.maximum_age + float(horizon.max()), snapshot.lost_reports,
                                        horizon.copy(), True)
        # Uniform knots bound radius conservatism; the exact turn endpoint
        # splits own motion so every swept own subsegment is a true chord.
        segments = max(1, int(np.ceil(horizon.max() / self.chord_seconds)))
        regular = horizon[..., None] * np.linspace(0., 1., segments + 1)
        knots = np.sort(np.concatenate((regular, np.minimum(turning, horizon)[..., None]), axis=-1), axis=-1)
        turn_fraction = np.divide(np.minimum(knots, turning[..., None]), turning[..., None],
                                  out=np.zeros_like(knots), where=turning[..., None] > 1e-12)
        sail_fraction = np.divide(np.maximum(0., knots - turning[..., None]), (dt[:, None] - turning)[..., None],
                                  out=np.zeros_like(knots), where=(dt[:, None] - turning)[..., None] > 1e-12)
        own_track = (own[:, None, None, :] + turn_fraction[..., None] * batch.turn_displacement[..., None, :]
                     + sail_fraction[..., None] * (batch.displacement - batch.turn_displacement)[..., None, :])
        forecast = prediction.centres_at(knots)
        relative = own_track[..., None, :] - forecast
        # The native radius is monotone and convex. Its right-end maximum is
        # an upper bound everywhere on each subsegment, including touches.
        radius = prediction.radii_at(knots[..., 1:])
        report_margin = (_nearest_chord_distance(relative[..., :-1, :, :], relative[..., 1:, :, :]) - radius).min(axis=(-2, -1))
        truth = self.truth_positions(elapsed_s + knots)
        truth_relative = own_track[..., None, :] - truth
        begin, end = elapsed_s + knots[..., :-1], elapsed_s + knots[..., 1:]
        overlaps_turn = ((begin[..., None] < self._start + self._duration)
                         & (end[..., None] > self._start))
        # A bounded-acceleration curve stays within a*dt^2/8 of its chord.
        deviation = overlaps_turn * self._acceleration * (end - begin)[..., None]**2 / 8.
        truth_margin = (_nearest_chord_distance(truth_relative[..., :-1, :, :], truth_relative[..., 1:, :, :])
                        - radius - deviation).min(axis=(-2, -1))
        return AISPreviewAssessment(report_margin > 0., report_margin, truth_margin,
                                    snapshot.maximum_age, snapshot.maximum_age + float(horizon.max()), snapshot.lost_reports,
                                    horizon.copy())

    def assess_applied(self, elapsed_s, own_position_EN, batch, duration_seconds, row, candidate, hit):
        """Return (report, truth) margins through a selected action's first hit.

        This helper matches the external benchmark's selected row/candidate
        interface. ``hit`` is seconds after the current decision, not the
        absolute episode capture time; it must lie in the decision interval.
        """
        rows = batch.features.shape[0]
        own = np.broadcast_to(np.asarray(own_position_EN, dtype=float), (rows, 2))[row]
        dt = float(np.broadcast_to(np.asarray(duration_seconds, dtype=float), (rows,))[row])
        subset = replace(batch, **{
            name: np.asarray(getattr(batch, name))[row:row+1, candidate:candidate+1]
            for name in batch.__dataclass_fields__
        })
        result = self.assess(elapsed_s, own, subset, dt, applied_seconds=float(hit))
        return float(result.report_margin_m[0, 0]), float(result.truth_margin_m[0, 0])
