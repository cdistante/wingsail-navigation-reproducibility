"""Causal two-trim escape previews for the Brindisi recovery comparison."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from causal_weather import CausalEfficientNativeEngine
from native_efficient_guard import FORECAST_REPORT_AGE_S, PREVIEW_HORIZON_S
from native_stale_guard import guarded_clearance
from world._geo import haversine_metres, local_east_north_metres, offset_latitude_longitude


SWITCH_TIMES_S = (30., 60.)
MAX_TRIMS = 6
MIN_IMPROVEMENT_M = 2.


def spread_indices(candidates, incumbent):
    """Sample across the legal wind-relative bows, retaining the reference trim."""
    legal = list(candidates)
    if len(legal) <= MAX_TRIMS:
        return legal
    sampled = [legal[i] for i in np.rint(np.linspace(0, len(legal) - 1, MAX_TRIMS)).astype(int)]
    if incumbent not in sampled:
        interior = min(range(1, len(sampled) - 1), key=lambda i: abs(sampled[i] - incumbent))
        sampled[interior] = incumbent
    return sorted(sampled)


class EscapeNativeEngine(CausalEfficientNativeEngine):
    """Try an early native trim change when every constant preview conflicts."""

    def __init__(self, environment, *, config, dynamics):
        super().__init__(environment, config=config, dynamics=dynamics)
        self.escape_events = []

    def _roll_trim(self, origin, choice, observation, duration_s):
        state = origin
        step = self.config.control_dt_s
        segments = []
        for _ in range(round(duration_s / step)):
            action = self.action(state, choice)
            state, path, _ = self.integrate(state, action, observation, step)
            segments.append(path[1:])
        return state, np.concatenate(segments)

    def _projected_colreg(self, observation, prediction, state, after_s):
        if not self.config.colreg:
            return False
        positions = offset_latitude_longitude(
            observation.boat.latitude, observation.boat.longitude,
            *prediction.centres_at(after_s).T,
        )
        targets = tuple(replace(target, latitude=float(latitude), longitude=float(longitude))
                        for target, latitude, longitude in zip(observation.ais_targets, *positions, strict=True))
        projected = replace(observation, boat=state, ais_targets=targets)
        return any(item.active for item in self.colreg.assess(state, projected))

    def _search(self, decision):
        observation = decision.observation
        prediction = self._prediction(observation).for_preview(
            FORECAST_REPORT_AGE_S, self.config.control_dt_s,
        )
        origin = decision.state
        choices = decision.choices
        legal = [i for i, choice in enumerate(choices) if choice.report_margin_m > 0. and choice.colreg_allowed]
        first_indices = spread_indices(legal, decision.physics_index)
        second_indices = spread_indices(range(len(choices)), decision.physics_index)
        times = np.arange(round(PREVIEW_HORIZON_S / self.config.integration_dt_s) + 1) * self.config.integration_dt_s
        start = np.array([[origin.latitude, origin.longitude]])
        winner = None
        checked = 0
        for first in first_indices:
            middle = origin
            prefix = start
            previous_switch = 0.
            for switch in SWITCH_TIMES_S:
                middle, advance = self._roll_trim(middle, choices[first], observation, switch - previous_switch)
                prefix = np.vstack((prefix, advance))
                previous_switch = switch
                if not self.environment.chart_path_is_clear(prefix, observation):
                    break
                east, north = local_east_north_metres(prefix[:, 0], prefix[:, 1], origin.latitude, origin.longitude)
                prefix_clearance = guarded_clearance(prediction, np.column_stack((east, north)), times[:len(prefix)])
                if prefix_clearance <= 0.:
                    break
                give_way = self._projected_colreg(observation, prediction, middle, switch)
                for second in second_indices:
                    if first == second:
                        continue
                    course = choices[second].course_deg
                    relative = (course - middle.heading + 180.) % 360. - 180.
                    if give_way and -self.colreg.starboard_turn_prohibited_sector_degrees <= relative <= -1.:
                        continue
                    final, suffix = self._roll_trim(middle, choices[second], observation, PREVIEW_HORIZON_S - switch)
                    path = np.vstack((prefix, suffix))
                    if not self.environment.chart_path_is_clear(path, observation):
                        continue
                    east, north = local_east_north_metres(path[:, 0], path[:, 1], origin.latitude, origin.longitude)
                    margin = guarded_clearance(prediction, np.column_stack((east, north)), times)
                    distance = haversine_metres(final.latitude, final.longitude, decision.goal.latitude,
                                                decision.goal.longitude)
                    score = margin, -distance
                    checked += 1
                    if winner is None or score > winner[0]:
                        winner = score, first, second, switch
        return winner, checked

    def decision(self, state, goal, when, elapsed_s=0.):
        preceding_events = len(self.anticipation_events)
        baseline = super().decision(state, goal, when, elapsed_s)
        if len(self.anticipation_events) == preceding_events:
            return baseline
        original = self.anticipation_events[-1]
        prior_margin = original["selected_preview_m"]
        if original["selection_rule"] != "largest_preview_margin" or prior_margin >= 0.:
            return baseline
        winner, checked = self._search(baseline)
        event = {
            "utc": original["utc"], "elapsed_s": original["elapsed_s"], "goal": goal.name,
            "constant_preview_margin_m": prior_margin, "evaluated_escapes": checked,
            "forecast_margin_m": None, "first_mode": None, "second_mode": None,
            "switch_seconds": None, "selected": False,
        }
        if winner is not None:
            (margin, _), first, second, switch = winner
            event.update(forecast_margin_m=margin, first_mode=baseline.choices[first].id,
                         second_mode=baseline.choices[second].id, switch_seconds=switch)
            if margin > prior_margin + MIN_IMPROVEMENT_M:
                # The extended sweep ranks actions. The base engine admits
                # each actual five-second trajectory with the uncapped guard.
                adjusted = list(baseline.choices)
                adjusted[first] = replace(adjusted[first], first_duration_s=self.config.control_dt_s,
                                          switch_after_s=PREVIEW_HORIZON_S)
                event["selected"] = True
                baseline = replace(baseline, choices=tuple(adjusted), physics_index=first,
                                   allowed=np.arange(len(adjusted)) == first)
        self.escape_events.append(event)
        return baseline
