# Forecast tacks during the Otranto hold

This record describes the original experiment workspace. The [reviewer evidence index](../../../docs/RESULTS.md) lists the released files and verification commands.

## Experiment

The two arms start from the previous Transformer and two-window release-confirmation runs. Both use the saved Otranto 001–005 route, departure at 2026-09-30 23:50 UTC, five-second controls, 50 m waypoint capture and a 24-hour cap. Wind FROM is held at 355° or 5°. Wind speed, current, waves, chart and initial vessel state come from the same inputs within each pair. After release, the 355° pair uses committed laylines and the 5° pair uses reactive laylines. Both use Gamma with Direct acados for sailing and the same slew-limited Fossen trim controller during hold. The paired treatment is `--forecast-hold-tacks`.

The current hold chooses an observed trim to limit displacement from the point where holding began. The forecast hold searches tracks of attainable trims over a rolling three-hour window. It starts with the current observed polar and considers quarter-hour wind, currents and waves from the archive at the active waypoint. The search estimates drift during heading changes, applies an additional settling allowance when crossing the wind and ranks feasible endpoints by range to the active waypoint. When the search finds a forecast release, it scores the range at that point. During a prolonged hold it scores the range at the horizon. The planner chooses the immediate trim from observed modes and repeats the chart check before use. Missing forecasts or blocked proposals fall back to the local hold or the existing chart safeguard.

The release test runs on the simulated boat's observed forcing and current position. That position can move the first viable release even though both arms use the same rule. We compare range at each release and at the earlier shared release time. Retrospective archive look-ahead is available throughout the treatment. The forecast search uses a representative station at the active waypoint and steady velocities with turn drift. The actual run integrates the boat along its changing position in the environmental fields.

## Results

All four runs captured 002 → 003 → 004 → 005 within 50 m. Both current-hold reruns reproduced the previous trajectories and actuator commands. At release the forecast boat was 281 m closer to 002 at 355° and 80 m closer at 5°. At the earlier common release time the gains were 279 m and 80 m. The forecast supplied every applied hold trim in both treatment runs.

| Wind FROM | Current range at release | Forecast range at release | Current finish | Forecast finish | Time saved |
|---|---:|---:|---:|---:|---:|
| 355° | 2,321.6 m | 2,040.4 m | 15:09:25 | 14:38:05 | 31 min 20 s |
| 5° | 1,543.7 m | 1,463.4 m | 12:50:05 | 12:29:25 | 20 min 40 s |

At 355°, release moved from 10 h 05 min to 10 h elapsed. The forecast boat started sailing 265 m less far south and 315 m farther east. Sustained tack-side changes while holding dropped from 18 to six. At 5°, both boats released at 6 h elapsed. The forecast boat used one opening tack and started sailing 31 m less far south and 433 m farther east. The eastern positioning reduced range to 002 and increased peak distance from the launch point during holding by 26 m at 355° and 66 m at 5°.

The replay audit checked 39,688 frames and all 16 captures. It found zero blocked or out-of-coverage chart samples. All four labeled MP4s passed full decode, and all 170 simulator tests passed.

- `Paper comparison`, [CSV](paper_comparison.csv), [matched-time results](paired_runs.csv) and [waypoint audit](waypoint_metrics.csv)
- `Four labeled MP4s, full replays and GPX tracks`
- [Hold forecast decisions](forecast_hold_events.csv), [usage](forecast_hold_usage.csv), `independent replay audit`, `forcing verification` and `replay manifest`
- `Baseline holding analysis`, [optimistic drift bound](hold_drift_bound.csv) and [archived staging probe](staging_probe.csv)

## Reproduction

From V2, use the simulator's Python environment with acados, PyTorch and FFmpeg.

```sh
study="$PWD/data/results/otranto_forecast_hold_20261118"
output=$(mktemp -d)
solver=$(mktemp -d)
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python -B "$study/reproduce/run_forecast_hold.py" --study "$output" --acados-build-dir "$solver"
python -B "$study/reproduce/audit_forecast_hold.py" --study "$output"
python -B "$study/reproduce/render_forecast_hold.py" --study "$output"
```
