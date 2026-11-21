"""Time-available weather inputs and a separate native plant for the ablation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from native_branches import EpisodeResult, NativeSnapshot, _clock, _first_capture, _name
from native_efficient_guard import EfficientNativeEngine
from native_stale_guard import guarded_clearance
from seasonal_environment import SeasonalArchive
from world._geo import haversine_metres, local_east_north_metres, normalise_time


class _AsOfGrid:
    """Spatially interpolate the last available slice of one forcing product."""

    def __init__(self, source):
        self.source = source
        self._fallback_count = 0

    @lru_cache(maxsize=32)
    def _surface(self, index):
        data = self.source.fields[index]
        valid = np.isfinite(data).all(axis=-1)
        axes = self.source.latitudes, self.source.longitudes
        masked = RegularGridInterpolator(axes, np.where(valid[..., None], data, 0.))
        coverage = RegularGridInterpolator(axes, valid.astype(float))
        return masked, coverage, valid

    @lru_cache(maxsize=32)
    def _nearest_wet(self, index):
        _, _, valid = self._surface(index)
        rows, columns = np.nonzero(valid)
        if not len(rows):
            raise ValueError("The available forcing slice has no valid wet cells.")
        scale = np.cos(np.radians(np.mean(self.source.latitudes)))
        points = np.column_stack((self.source.latitudes[rows], self.source.longitudes[columns] * scale))
        return cKDTree(points), rows, columns, scale

    def values(self, latitude, longitude, when_ns):
        index = int(np.searchsorted(self.source.times, when_ns, side="right")) - 1
        if index < 0:
            raise ValueError("The query precedes the first available forcing slice.")
        masked, coverage, _ = self._surface(index)
        query = latitude, longitude
        weight = np.asarray(coverage(query)).item()
        if weight > 1e-9:
            return masked(query) / weight
        tree, rows, columns, scale = self._nearest_wet(index)
        distance, nearest = tree.query((latitude, longitude * scale))
        if distance > .15:
            raise ValueError(f"No available wet cell within 0.15° of ({latitude}, {longitude}).")
        self._fallback_count += 1
        return self.source.fields[index, rows[nearest], columns[nearest]]


class AsOfSeasonalArchive(SeasonalArchive):
    """Valid-hour availability proxy for archived products without issue times."""

    def __init__(self, interpolated_archive):
        self._wind = _AsOfGrid(interpolated_archive._wind)
        self._current = _AsOfGrid(interpolated_archive._current)
        self._wave = _AsOfGrid(interpolated_archive._wave)


class CausalScenarioEnvironment:
    """Share the fixture's geometry, causal AIS receiver, and chart across arms."""

    def __init__(self, scenario_environment, *, as_of_archive=None):
        self.scenario = scenario_environment
        self.as_of_archive = (
            as_of_archive if as_of_archive is not None else AsOfSeasonalArchive(scenario_environment.archive)
        )

    def __getattr__(self, name):
        return getattr(self.scenario, name)

    def observe(self, state, when):
        instant = normalise_time(when)
        return self.as_of_archive.observe(state, instant, reports=self.scenario.reports_at(instant))

    def weather_history(self, latitude, longitude, when):
        return self.as_of_archive.weather_history(latitude, longitude, when)

    def plant_observe(self, state, when):
        """Supply interpolation at an accepted helm update for plant integration."""
        return self.scenario.observe(state, when)


@dataclass
class CausalEpisodeResult(EpisodeResult):
    guarded_report_minimum_m: float = float("inf")
    applied_chart_violations: int = 0


class CausalEfficientNativeEngine(EfficientNativeEngine):
    """Apply the efficient screen with available weather and advance a separate plant."""

    def _advance_plant(self, state, when, action, dt):
        observation = self.environment.plant_observe(state, when)
        final, path, local = self.integrate(state, action, observation, dt)
        return observation, final, path, local


    def run(self, initial_state, goals, start_utc, *, strategy=None, forced_first=None,
            first_commit_seconds=None, cap_seconds=None, trace=True, on_decision=None,
            initial_elapsed_s=0., pending_switch_bow_deg=None):
        # Isolated from the byte-pinned seasonal runner. Its control loop is the
        # reference for scheduling, handoffs, captures, and the five-second budget.
        goals = tuple(goals)
        start = normalise_time(start_utc)
        cap = float(self.config.cap_seconds if cap_seconds is None else cap_seconds)
        if cap <= 0 or abs(cap / .05 - round(cap / .05)) > 1e-6:
            raise ValueError("Episode cap must be positive and a multiple of 0.05 s.")
        state, elapsed, goal_index = initial_state, 0., 0
        result = CausalEpisodeResult(False, "censored_cap", 0., state)
        chosen, next_decision, decision_count = None, 0., 0
        requested_commit = self.config.commit_seconds if first_commit_seconds is None else float(first_commit_seconds)
        result.first_commit_requested_s = requested_commit
        pending = pending_switch_bow_deg
        first_commit_interrupted = False
        while elapsed < cap - 1e-8 and goal_index < len(goals):
            control_started = perf_counter()
            callback_seconds = 0.
            goal = goals[goal_index]
            when = _clock(start, elapsed)
            if haversine_metres(state.latitude, state.longitude, goal.latitude, goal.longitude) <= self.config.capture_radius_m:
                result.captures.append({"name": _name(goal), "index": goal_index, "elapsed_s": elapsed,
                    "utc": str(when) + "Z", "distance_m": haversine_metres(state.latitude, state.longitude, goal.latitude, goal.longitude),
                    "latitude": state.latitude, "longitude": state.longitude, "scope": "great-circle native polyline"})
                goal_index += 1
                chosen, next_decision, pending = None, elapsed, None
                continue
            if chosen is None or elapsed >= next_decision - 1e-8:
                decision_started = perf_counter()
                decision = self.decision(state, goal, when, initial_elapsed_s + elapsed)
                if decision.physics_index < 0:
                    result.status = "no_safe_native_action"
                    result.control_attempts.append({"elapsed_s": elapsed, "utc": str(when) + "Z",
                        "goal": _name(goal), "applied": False, "duration_s": 0.,
                        "complete_control_seconds": perf_counter() - control_started,
                        "reason": result.status})
                    break
                if decision_count == 0 and forced_first is not None:
                    index = int(forced_first)
                elif pending is not None:
                    legal = np.flatnonzero(decision.allowed)
                    index = int(min(legal, key=lambda i: abs((decision.choices[i].bow_deg - pending + 180) % 360 - 180)))
                else:
                    index = decision.physics_index if strategy is None else int(strategy(decision))
                if not 0 <= index < len(decision.choices):
                    raise ValueError("Strategy returned an invalid candidate index.")
                requested = index
                snapshot_pending = pending
                if not decision.allowed[index]:
                    index = decision.physics_index
                    result.shield_overrides += 1
                    if decision_count == 0:
                        first_commit_interrupted = True
                chosen = decision.choices[index]
                commit = requested_commit if decision_count == 0 else self.config.commit_seconds
                commit = min(commit, chosen.first_duration_s)
                next_decision = elapsed + commit
                pending = chosen.following_bow_deg if chosen.switch_after_s <= commit and abs((chosen.following_bow_deg - chosen.bow_deg + 180) % 360 - 180) > 1e-6 else None
                result.decisions.append({"elapsed_s": elapsed, "utc": str(when) + "Z", "goal": _name(goal),
                    "requested_index": requested, "selected_index": index, "physics_index": decision.physics_index,
                    "choice_id": chosen.id, "bow_deg": chosen.bow_deg, "commit_s": commit,
                    "predicted_eta_s": chosen.predicted_eta_s, "schedule_class": chosen.schedule_class,
                    "decision_seconds": perf_counter() - decision_started})
                if decision_count == 0:
                    result.first_choice_requested = decision.choices[requested].id
                    result.first_choice_applied = chosen.id
                if on_decision is not None:
                    snapshot = NativeSnapshot(state, goals[goal_index:], when, initial_elapsed_s + elapsed,
                                              cap - elapsed, snapshot_pending)
                    callback_started = perf_counter()
                    on_decision(snapshot, decision)
                    callback_seconds = perf_counter() - callback_started
                decision_count += 1
            obs = self.environment.observe(state, when)
            dt = round(min(self.config.control_dt_s, cap - elapsed) / .05) * .05
            if dt < .05 - 1e-8:
                break
            margin, permitted, final, path, local, prediction, action = self._screen(state, chosen, obs, dt)
            if margin <= 0. or not permitted:
                # Fresh 5 s report/helm screening can interrupt a tack commitment.
                fresh = self.decision(state, goal, when, initial_elapsed_s + elapsed)
                if fresh.physics_index < 0:
                    result.status = "no_safe_native_action"
                    result.control_attempts.append({"elapsed_s": elapsed, "utc": str(when) + "Z",
                        "goal": _name(goal), "applied": False, "duration_s": 0.,
                        "complete_control_seconds": perf_counter() - control_started - callback_seconds,
                        "reason": result.status})
                    break
                chosen = fresh.choices[fresh.physics_index]
                result.shield_overrides += 1
                if elapsed < requested_commit:
                    first_commit_interrupted = True
                next_decision, pending = elapsed + self.config.commit_seconds, None
                margin, permitted, final, path, local, prediction, action = self._screen(state, chosen, obs, dt)
                if margin <= 0. or not permitted:
                    raise RuntimeError("A freshly admitted native candidate failed deterministic re-screening.")
            complete_control_seconds = perf_counter() - control_started - callback_seconds
            # The controller has already committed to the command using available inputs.
            # Advance the archived plant and audit only the actual swept trajectory.
            plant_obs, final, path, local = self._advance_plant(state, when, action, dt)
            hit, closest = _first_capture(path, goal, self.config.capture_radius_m)
            gate_entry = None
            applied_dt = dt
            if hit is not None:
                k = min(int(np.floor(hit)), len(path) - 2)
                f = float(hit - k)
                point = path[k] + f * (path[k + 1] - path[k])
                gate_entry = {"elapsed_s": elapsed + hit * .05,
                    "utc": str(_clock(when, hit * .05)) + "Z", "latitude": float(point[0]),
                    "longitude": float(point[1]), "distance_m": haversine_metres(*point, goal.latitude, goal.longitude)}
                # Continue/stop at an exact integrator knot. Gate-entry timing
                # is independently interpolated, never an interpolated plant reset.
                end = max(1, min(int(np.ceil(hit)), len(path) - 1))
                path = path[:end + 1]
                applied_dt = end * .05
                final = self._state_at(local[:, end], path[-1], action, plant_obs, previous=local[:, end - 1])
                _, closest = _first_capture(path, goal, self.config.capture_radius_m)
            result.closest_m = min(result.closest_m, closest)
            times = np.arange(len(path)) * .05
            east, north = local_east_north_metres(path[:, 0], path[:, 1], state.latitude, state.longitude)
            own_track = np.column_stack((east, north))
            applied_margin = prediction.assess_local_rollout(own_track, times,
                compute_contact_times=False).minimum_clearance_metres
            truth_margin = self._truth_margin(path, times, when, prediction)
            applied_guard_margin = guarded_clearance(prediction, own_track, times)
            applied_chart_clear = self.environment.chart_path_is_clear(path, obs)
            result.guarded_report_minimum_m = min(result.guarded_report_minimum_m, applied_guard_margin)
            result.applied_chart_violations += int(not applied_chart_clear)
            result.report_minimum_m = min(result.report_minimum_m, applied_margin)
            result.truth_minimum_m = min(result.truth_minimum_m, truth_margin)
            record = {"elapsed_s": elapsed, "utc": str(when) + "Z", "goal": _name(goal),
                "duration_s": applied_dt, "rudder_deg": action.rudder, "flap_target_deg": action.flap,
                "choice_id": chosen.id, "bow_deg": chosen.bow_deg, "report_minimum_m": applied_margin,
                "truth_minimum_m": truth_margin, "max_report_age_s": float(prediction.report_ages_seconds.max(initial=0.)),
                "guarded_report_minimum_m": applied_guard_margin, "screened_guard_margin_m": margin,
                "applied_chart_clear": applied_chart_clear,
                "native_substeps": len(path) - 1, "held_observation": True,
                "dropped_report_epochs": self._dropped_epochs(when),
                "complete_control_seconds": complete_control_seconds}
            result.controls.append(record)
            result.control_attempts.append({**record, "applied": True})
            if trace:
                result.trace.append({"elapsed_s": elapsed + applied_dt, "utc": str(_clock(when, applied_dt)) + "Z",
                                     **asdict(final)})
            if (not first_commit_interrupted and elapsed < requested_commit
                    and result.first_choice_applied == chosen.id):
                result.first_commit_applied_s += min(applied_dt, requested_commit - elapsed)
            state, elapsed = final, elapsed + applied_dt
            if hit is not None:
                result.captures.append({"name": _name(goal), "index": goal_index, **gate_entry,
                    "integrator_elapsed_s": elapsed, "scope": "great-circle native polyline"})
                goal_index += 1
                chosen, next_decision, pending = None, elapsed, None
        result.success = goal_index == len(goals)
        if result.success:
            result.status = "captured"
        result.censored = not result.success
        result.elapsed_s, result.final_state = elapsed, state
        result.first_commit_completed = result.first_commit_applied_s >= requested_commit - 1e-7
        return result
