"""Time-indexed AIS clearance for tactical and control rollouts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import hypot, isfinite, radians

import numpy as np

from config import SAILDRONE_GEOMETRY_AND_HYDROSTATICS

from ._geo import local_east_north_metres
from .ais import AISTarget
from .boat import BoatState
from .world import WorldObservation


_OWN_HULL_RADIUS_METRES = 0.5 * hypot(
    SAILDRONE_GEOMETRY_AND_HYDROSTATICS["hull_length_overall_m"],
    SAILDRONE_GEOMETRY_AND_HYDROSTATICS["hull_beam_max_m"],
)
DEFAULT_AIS_SAFETY_BUFFER_METRES = 25.0


@dataclass(frozen=True)
class AISObstacle:
    """One obstacle at a controller stage in local east/north metres."""

    mmsi: int | None
    east_metres: float
    north_metres: float
    hull_radius_metres: float
    clearance_radius_metres: float
    extrapolation_limited: bool


@dataclass(frozen=True)
class AISTargetClearance:
    """Swept distances and first contacts with one target over a rollout."""

    target: AISTarget
    minimum_center_distance_metres: float
    minimum_hull_clearance_metres: float
    minimum_clearance_metres: float
    initial_clearance_metres: float
    final_clearance_metres: float
    closest_time_seconds: float
    minimum_clearance_time_seconds: float
    first_contact_seconds: float | None
    first_hull_contact_seconds: float | None


@dataclass(frozen=True)
class AISRolloutAssessment:
    """Distinguish contact-free rollouts from recovery within an existing buffer."""

    contacts: tuple[AISTargetClearance, ...]
    recovering: bool

    @property
    def safe(self) -> bool:
        return self.minimum_clearance_metres > 0.0

    @property
    def initial_overlap(self) -> bool:
        return any(contact.initial_clearance_metres <= 0.0 for contact in self.contacts)

    @property
    def minimum_clearance_metres(self) -> float:
        return min((contact.minimum_clearance_metres for contact in self.contacts), default=float("inf"))

    @property
    def minimum_hull_clearance_metres(self) -> float:
        return min((contact.minimum_hull_clearance_metres for contact in self.contacts), default=float("inf"))

    @property
    def minimum_center_distance_metres(self) -> float:
        return min((contact.minimum_center_distance_metres for contact in self.contacts), default=float("inf"))

    @property
    def first_contact_seconds(self) -> float | None:
        return min((c.first_contact_seconds for c in self.contacts if c.first_contact_seconds is not None), default=None)

    @property
    def first_hull_contact_seconds(self) -> float | None:
        return min((c.first_hull_contact_seconds for c in self.contacts if c.first_hull_contact_seconds is not None), default=None)


@dataclass(frozen=True)
class AISPrediction:
    """Constant-COG/SOG target centers and growing clearance radii.

    Local east/north is anchored to the boat at the observation time. Target
    arrays retain the order of ``WorldObservation.ais_targets``. Hull radii
    include both the target's circular hull approximation and the own hull.
    Clearance radii also include the requested buffer, position uncertainty,
    and optional steering reserve.
    """

    origin_latitude: float
    origin_longitude: float
    targets: tuple[AISTarget, ...]
    positions_metres: np.ndarray
    velocities_mps: np.ndarray
    hull_radii_metres: np.ndarray
    clearance_radii_metres: np.ndarray
    uncertainty_growth_mps: np.ndarray
    report_ages_seconds: np.ndarray
    extrapolation_limited: np.ndarray
    acceleration_bound_mps2: float = 0.0
    motion_growth_floor_mps: float = 0.0
    observed_at_ns: int = 0
    clearance_reserve_metres: float = 0.0
    preview_growth_seconds: np.ndarray | None = None

    @classmethod
    def from_observation(
        cls,
        observation: WorldObservation,
        *,
        own_hull_radius_metres: float = _OWN_HULL_RADIUS_METRES,
        safety_buffer_metres: float = DEFAULT_AIS_SAFETY_BUFFER_METRES,
        uncertainty_metres: float = 5.0,
        uncertainty_growth_mps: float = 0.05,
        report_age_seconds: Mapping[int, float] | None = None,
        max_extrapolation_seconds: float = 150.0,
        unknown_hull_radius_metres: float = 25.0,
        unknown_speed_uncertainty_mps: float = 1.0,
        acceleration_bound_mps2: float = 0.0,
        clearance_reserve_metres: float = 0.0,
    ) -> AISPrediction:
        """Project reports forward with a cap and account for age in the radii."""
        if any(not isfinite(value) or value < 0.0 for value in (
            own_hull_radius_metres, safety_buffer_metres, uncertainty_metres, uncertainty_growth_mps,
            max_extrapolation_seconds, unknown_hull_radius_metres, unknown_speed_uncertainty_mps,
            acceleration_bound_mps2, clearance_reserve_metres,
        )):
            raise ValueError("AIS geometry distances, growth, and time bounds must be finite and non-negative.")

        targets = tuple(observation.ais_targets)
        latitude, longitude = observation.boat.latitude, observation.boat.longitude
        latitudes = np.asarray([target.latitude for target in targets], dtype=float)
        longitudes = np.asarray([target.longitude for target in targets], dtype=float)
        if (not np.isfinite(latitudes).all() or np.any(np.abs(latitudes) > 90.0)
                or not np.isfinite(longitudes).all() or np.any(np.abs(longitudes) > 180.0)):
            raise ValueError("AIS target positions must contain valid latitudes and longitudes.")
        east, north = local_east_north_metres(latitudes, longitudes, latitude, longitude)
        positions = np.column_stack((east, north))

        velocities = np.empty_like(positions)
        hull_radii = np.empty(len(targets))
        growth = np.empty(len(targets))
        ages = np.empty(len(targets))
        for index, target in enumerate(targets):
            velocity, extra_growth = _target_velocity(target, unknown_speed_uncertainty_mps)
            velocities[index] = velocity
            growth[index] = uncertainty_growth_mps + extra_growth
            length, beam = target.length_metres, target.beam_metres
            if all(value is not None and isfinite(value) and value > 0.0 for value in (length, beam)):
                radius = 0.5 * hypot(length, beam)
            else:
                radius = unknown_hull_radius_metres
            hull_radii[index] = own_hull_radius_metres + radius

            age = getattr(target, "report_age_seconds", None)
            if report_age_seconds is not None and target.mmsi in report_age_seconds:
                age = report_age_seconds[target.mmsi]
            age = 0.0 if age is None else float(age)
            if not isfinite(age) or age < 0.0:
                raise ValueError("AIS report ages must be finite and non-negative.")
            ages[index] = age

        extrapolation = np.minimum(ages, max_extrapolation_seconds)
        positions += extrapolation[:, None] * velocities
        # Cover the remaining nominal extrapolation without trusting a stale center.
        stale_uncertainty = (ages - extrapolation) * np.linalg.norm(velocities, axis=1)
        clearance_radii = hull_radii + safety_buffer_metres + uncertainty_metres + ages * growth + stale_uncertainty
        clearance_radii += clearance_reserve_metres
        if acceleration_bound_mps2:
            clearance_radii += np.maximum(
                0.0, 0.5 * acceleration_bound_mps2 * ages**2 - uncertainty_growth_mps * ages,
            )
        observed_at_ns = int(np.datetime64(observation.time, "ns").astype(np.int64))
        return cls(latitude, longitude, targets, positions, velocities, hull_radii, clearance_radii,
                   growth, ages, ages > max_extrapolation_seconds,
                   acceleration_bound_mps2, uncertainty_growth_mps, observed_at_ns, clearance_reserve_metres)

    def centres_at(self, elapsed_seconds: float | np.ndarray) -> np.ndarray:
        """Project centers using seconds after the observation time."""
        elapsed = np.asarray(elapsed_seconds, dtype=float)
        return self.positions_metres + elapsed[..., None, None] * self.velocities_mps

    def for_preview(self, report_age_budget_seconds: float, control_interval_seconds: float) -> AISPrediction:
        """Forecast renewed reports beyond a declared report-age budget.

        Preserve accrued uncertainty and at least one full control interval of
        growth. Continue to use the original prediction for applied sweeps.
        """
        future_growth = np.maximum(control_interval_seconds, report_age_budget_seconds - self.report_ages_seconds)
        return replace(self, preview_growth_seconds=future_growth)

    def radii_at(self, elapsed_seconds: float | np.ndarray) -> np.ndarray:
        """Return clearance radii using seconds after the observation time."""
        elapsed = np.asarray(elapsed_seconds, dtype=float)[..., None]
        if self.preview_growth_seconds is not None:
            elapsed = np.minimum(elapsed, self.preview_growth_seconds)
        radii = self.clearance_radii_metres + elapsed * self.uncertainty_growth_mps
        if self.acceleration_bound_mps2:
            # A constant-velocity forecast may be off by 1/2 a t². Use
            # source-report time and keep the linear motion-error floor.
            # Convex radius growth is conservatively swept between knots.
            age = self.report_ages_seconds
            total = age + elapsed
            bound = 0.5 * self.acceleration_bound_mps2
            floor = self.motion_growth_floor_mps
            radii += (
                np.maximum(0.0, bound * total**2 - floor * total)
                - np.maximum(0.0, bound * age**2 - floor * age)
            )
        return radii

    def stage_obstacles(self, elapsed_seconds: float) -> tuple[AISObstacle, ...]:
        """Return the uncapped obstacle list at an optimizer stage."""
        return tuple(
            AISObstacle(target.mmsi, float(east), float(north), float(hull), float(radius), bool(stale))
            for target, (east, north), hull, radius, stale in zip(
                self.targets, self.centres_at(elapsed_seconds), self.hull_radii_metres,
                self.radii_at(elapsed_seconds), self.extrapolation_limited,
            )
        )

    def signed_margins(
        self, positions_metres: np.ndarray, times_seconds: float | np.ndarray,
    ) -> np.ndarray:
        """Return point-sample clearances with target on the last axis."""
        relative = np.asarray(positions_metres, dtype=float)[..., None, :] - self.centres_at(times_seconds)
        return np.linalg.norm(relative, axis=-1) - self.radii_at(times_seconds)

    def assess_rollout(
        self,
        states: Sequence[BoatState],
        times_seconds: Sequence[float],
        *,
        compute_contact_times: bool = True,
    ) -> AISRolloutAssessment:
        """Project geographic Fossen samples and check each synchronized segment."""
        east, north = local_east_north_metres(
            [state.latitude for state in states], [state.longitude for state in states],
            self.origin_latitude, self.origin_longitude,
        )
        return self.assess_local_rollout(np.column_stack((east, north)), times_seconds,
                                         compute_contact_times=compute_contact_times)

    def assess_local_rollout(
        self,
        positions_metres: np.ndarray,
        times_seconds: Sequence[float],
        *,
        compute_contact_times: bool = True,
    ) -> AISRolloutAssessment:
        """Check linear own-ship chords against moving targets at the same times.

        Rollout sampling sets the resolution of the own trajectory. A one-sample
        rollout checks current occupancy. Contact times can be omitted when
        screening candidates by clearance alone.
        """
        times = np.asarray(times_seconds, dtype=float)
        points = np.asarray(positions_metres, dtype=float)
        if (times.ndim != 1 or times.size == 0 or points.shape != (times.size, 2)
                or not np.isfinite(times).all() or not np.isfinite(points).all()
                or times[0] < 0.0 or np.any(np.diff(times) <= 0.0)):
            raise ValueError("AIS rollout needs finite east/north points at increasing non-negative times.")
        if not self.targets:
            return AISRolloutAssessment((), recovering=False)
        if times.size == 1:
            times, points = np.repeat(times, 2), np.repeat(points, 2, axis=0)

        relative = points[:, None, :] - self.centres_at(times)
        delta = np.diff(relative, axis=0)
        radii = self.radii_at(times)
        growth = np.diff(radii, axis=0)
        center_fraction = _closest_fraction(relative[:-1], delta, 0.0)
        buffer_fraction = _closest_fraction(relative[:-1], delta, growth)
        center_distance = np.linalg.norm(relative[:-1] + center_fraction[..., None] * delta, axis=-1)
        hull_margin = center_distance - self.hull_radii_metres
        buffer_margin = (np.linalg.norm(relative[:-1] + buffer_fraction[..., None] * delta, axis=-1)
                         - radii[:-1] - buffer_fraction * growth)
        initial = np.linalg.norm(relative[0], axis=-1) - radii[0]
        final = np.linalg.norm(relative[-1], axis=-1) - radii[-1]
        closest = np.argmin(center_distance, axis=0)
        least_clear = np.argmin(buffer_margin, axis=0)

        contacts = []
        for index, target in enumerate(self.targets):
            nearest, worst = closest[index], least_clear[index]
            buffer_contact = hull_contact = None
            if compute_contact_times:
                buffer_contact = _first_contact(times, relative[:, index], radii[:, index],
                                                buffer_margin[:, index], buffer_fraction[:, index])
                hull_contact = _first_contact(times, relative[:, index], self.hull_radii_metres[index],
                                              hull_margin[:, index], center_fraction[:, index])
            contacts.append(AISTargetClearance(
                target=target,
                minimum_center_distance_metres=float(center_distance[nearest, index]),
                minimum_hull_clearance_metres=float(hull_margin[nearest, index]),
                minimum_clearance_metres=float(buffer_margin[worst, index]),
                initial_clearance_metres=float(initial[index]),
                final_clearance_metres=float(final[index]),
                closest_time_seconds=_segment_time(times, nearest, center_fraction[nearest, index]),
                minimum_clearance_time_seconds=_segment_time(times, worst, buffer_fraction[worst, index]),
                first_contact_seconds=buffer_contact,
                first_hull_contact_seconds=hull_contact,
            ))
        overlapping = initial <= 0.0
        minima = np.min(buffer_margin, axis=0)
        recovering = bool(
            np.any(overlapping) and np.all(hull_margin > 0.0) and np.all(minima[~overlapping] > 0.0)
            and np.all(minima[overlapping] >= initial[overlapping] - 1e-9)
            and np.all(final[overlapping] > initial[overlapping] + 1e-9)
        )
        return AISRolloutAssessment(tuple(contacts), recovering)


def _target_velocity(target: AISTarget, unknown_speed_uncertainty_mps: float) -> tuple[np.ndarray, float]:
    speed = target.speed
    if speed is None or not isfinite(speed) or speed < 0.0:
        return np.zeros(2), unknown_speed_uncertainty_mps
    course = target.course
    heading_only = course is None or not isfinite(course) or not 0.0 <= course < 360.0
    if heading_only:
        course = target.heading
    if course is None or not isfinite(course) or not 0.0 <= course < 360.0:
        return np.zeros(2), speed
    angle = radians(course)
    return speed * np.array([np.sin(angle), np.cos(angle)]), speed * 0.25 if heading_only else 0.0


def _closest_fraction(relative: np.ndarray, delta: np.ndarray, growth: float | np.ndarray) -> np.ndarray:
    """Minimize the norm of a linear relative path minus a linear radius."""
    travel = np.linalg.norm(delta, axis=-1)
    direction = np.divide(delta, travel[..., None], out=np.zeros_like(delta), where=travel[..., None] > 0.0)
    along = np.sum(relative * direction, axis=-1)
    perpendicular = np.linalg.norm(relative - along[..., None] * direction, axis=-1)
    interior = travel > growth
    denominator = np.sqrt(np.maximum(0.0, (travel - growth) * (travel + growth)))
    shift = np.divide(growth * perpendicular, denominator, out=np.zeros_like(along), where=denominator > 0.0)
    fraction = np.divide(shift - along, travel, out=np.zeros_like(along), where=interior)
    return np.where(interior, np.clip(fraction, 0.0, 1.0), np.asarray(growth) > 0.0)


def _segment_time(times: np.ndarray, segment: int, fraction: float) -> float:
    return float(times[segment] + fraction * (times[segment + 1] - times[segment]))


def _first_contact(
    times: np.ndarray, relative: np.ndarray, radii: float | np.ndarray,
    minima: np.ndarray, fractions: np.ndarray,
) -> float | None:
    """Find the first entry into a radius, including a touch or an initial overlap."""
    hits = np.flatnonzero(minima <= 0.0)
    if hits.size == 0:
        return None
    segment = hits[0]
    start_radius, end_radius = (radii, radii) if np.ndim(radii) == 0 else radii[segment:segment + 2]
    start = relative[segment]
    if hypot(*start) <= start_radius:
        return float(times[segment])

    # Margin is convex. Bracket its first zero between this segment's start
    # and its minimum rather than selecting a possibly later quadratic root.
    direction = relative[segment + 1] - start
    lower, upper = 0.0, float(fractions[segment])
    for _ in range(38):
        fraction = (lower + upper) / 2.0
        radius = start_radius + fraction * (end_radius - start_radius)
        if hypot(*(start + fraction * direction)) > radius:
            lower = fraction
        else:
            upper = fraction
    return _segment_time(times, segment, upper)
