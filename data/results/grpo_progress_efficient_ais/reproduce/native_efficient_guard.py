"""Reduce AIS detours by handing back to the sailing trim on a clear forecast."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from native_stale_guard import RobustNativeEngine, guarded_clearance
from world._geo import local_east_north_metres


PREVIEW_HORIZON_S = 240.0
FORECAST_REPORT_AGE_S = 65.0
RECHECK_MARGIN_M = 10.0


class EfficientNativeEngine(RobustNativeEngine):
    """Retain the nominal sailing trim once its forecast clears the encounter."""

    def __init__(self, environment, *, config=None, dynamics=None):
        super().__init__(environment, config=config, dynamics=dynamics)
        self.anticipation_events = []

    def preview(self, decision, choice, prediction):
        """Project a native trim using forcing observed at the decision time."""
        state = decision.state
        step = self.config.control_dt_s
        segments = [np.array([[state.latitude, state.longitude]])]
        for _ in range(round(PREVIEW_HORIZON_S / step)):
            action = self.action(state, choice)
            state, path, _ = self.integrate(state, action, decision.observation, step)
            segments.append(path[1:])
        path = np.concatenate(segments)
        origin = decision.state
        east, north = local_east_north_metres(path[:, 0], path[:, 1], origin.latitude, origin.longitude)
        times = np.arange(len(path)) * self.config.integration_dt_s
        margin = guarded_clearance(prediction, np.column_stack((east, north)), times)
        chart = getattr(self.environment, "chart_path_is_clear", None)
        return margin, chart is None or bool(chart(path, decision.observation))

    def decision(self, state, goal, when, elapsed_s=0.):
        decision = super().decision(state, goal, when, elapsed_s)
        nominal = decision.physics_index
        if nominal < 0 or not decision.observation.ais_targets:
            return decision

        observed = self._prediction(decision.observation)
        forecast = observed.for_preview(FORECAST_REPORT_AGE_S, self.config.control_dt_s)
        choices = decision.choices
        nominal_choice = choices[nominal]
        nominal_margin, nominal_chart = self.preview(decision, nominal_choice, forecast)
        if nominal_margin > RECHECK_MARGIN_M:
            return decision

        margin, selected, rule = nominal_margin, nominal, "nominal_recheck"
        candidates = [(nominal_margin, nominal)] if nominal_chart else []
        if nominal_margin < 0.0:
            for index in np.flatnonzero(decision.allowed):
                if index != nominal:
                    candidate_margin, chart_clear = self.preview(decision, choices[index], forecast)
                    if chart_clear:
                        candidates.append((candidate_margin, int(index)))
            clear = [item for item in candidates if item[0] > 0.0]
            if clear:
                # Once the forecast clears, use the existing time-to-mark ranking.
                margin, selected = min(clear, key=lambda item: (choices[item[1]].score, -item[0]))
                rule = "forecast_clear_route_rank"
            elif candidates:
                margin, selected = max(candidates, key=lambda item: (item[0], item[1] == nominal))
                rule = "largest_preview_margin"

        # A clear nominal forecast can resume immediately, but replan after
        # one update until it clears the preview margin used for commitment.
        option = replace(choices[selected], first_duration_s=self.config.control_dt_s)
        adjusted = list(choices)
        adjusted[selected] = option
        allowed = np.arange(len(choices)) == selected
        self.anticipation_events.append({
            "elapsed_s": float(elapsed_s),
            "utc": str(decision.when) + "Z",
            "goal": decision.goal.name,
            "report_age_s": float(observed.report_ages_seconds.max(initial=0.)),
            "forecast_horizon_s": PREVIEW_HORIZON_S,
            "forecast_report_age_s": FORECAST_REPORT_AGE_S,
            "nominal_choice": nominal_choice.id,
            "nominal_bow_deg": nominal_choice.bow_deg,
            "nominal_preview_m": nominal_margin,
            "selected_choice": option.id,
            "selected_bow_deg": option.bow_deg,
            "selected_preview_m": margin,
            "allowed_now": int(decision.allowed.sum()),
            "chart_admitted_previews": len(candidates),
            "reason": "ais_diversion" if selected != nominal else "nominal_recheck",
            "selection_rule": rule,
        })
        return replace(decision, choices=tuple(adjusted), allowed=allowed, physics_index=selected)
