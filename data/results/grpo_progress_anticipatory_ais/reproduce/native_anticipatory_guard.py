"""Schedule an early native AIS diversion while retaining the applied guard."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from native_stale_guard import RobustNativeEngine, guarded_clearance
from world._geo import local_east_north_metres


PREVIEW_HORIZON_S = 180.0
FORECAST_REPORT_AGE_S = 65.0
RELEASE_MARGIN_M = 8.0


class AnticipatoryNativeEngine(RobustNativeEngine):
    """Rank currently admissible tacks by a causal Fossen clearance forecast."""

    def __init__(self, environment, *, config=None, dynamics=None):
        super().__init__(environment, config=config, dynamics=dynamics)
        self.anticipation_events = []
        self.avoiding = False

    def preview(self, decision, choice, prediction):
        """Hold the offered sailing trim under the current observed forcing."""
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
        nominal_choice = decision.choices[nominal]
        nominal_margin, nominal_chart = self.preview(decision, nominal_choice, forecast)
        release = RELEASE_MARGIN_M if self.avoiding else 0.0
        if nominal_margin >= release:
            self.avoiding = False
            return decision

        choices = decision.choices
        candidates = [(nominal_margin, nominal)] if nominal_chart else []
        for index in np.flatnonzero(decision.allowed):
            if index != nominal:
                margin, chart_clear = self.preview(decision, choices[index], forecast)
                if chart_clear:
                    candidates.append((margin, int(index)))
        if candidates:
            margin, selected = max(candidates, key=lambda item: (item[0], item[1] == nominal))
        else:
            margin, selected = nominal_margin, nominal

        self.avoiding = True
        # Reconsider after every applied update in the encounter. The choice
        # still uses the existing full, uncapped robust control screen.
        option = replace(choices[selected], first_duration_s=self.config.control_dt_s)
        adjusted = list(choices)
        adjusted[selected] = option
        only_selected = np.arange(len(choices)) == selected
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
        })
        return replace(decision, choices=tuple(adjusted), allowed=only_selected, physics_index=selected)
