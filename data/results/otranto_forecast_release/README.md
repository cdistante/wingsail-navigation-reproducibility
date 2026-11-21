# Otranto forecast ETA hold-exit experiment

This record describes the original experiment workspace. The [reviewer evidence index](../../../docs/RESULTS.md) lists the released files and verification commands.

The two fresh runs are paired with the verified ETA-gated baselines. Each pair uses the same archived wind, current and waves, Gamma guidance, Direct acados NMPC, layline ETA rule, Transformer schedule veto, AIS traffic and dropouts, and COLREG checks. Times start at actual launch at 2026-09-30T23:54:50Z after 0:04:50 of protected staging.

During an admitted drift hold the new controller compares sailing a chart-clear trim now with retaining the hold through the remainder of the decision window, up to five minutes. Both candidates use the same 12-hour continuation search through archived Fossen polars sampled at 30-minute intervals and the same 50 m waypoint gate. A release requires two finite entry estimates, a gain above one 5 s control tick, and a smaller predicted range after the first window. The ETA-gated route planner continues after the classical hold ends. AIS/COLREG and chart checks still govern applied controls.

| Wind FROM | Guidance | Outbound 002 | Loop or stop | Ordered 50 m captures | Truth margin | Report margin | Slowest decision | 5 s checks |
|---:|:---|---:|---:|---:|---:|---:|---:|:---|
| 355° | ETA-gated baseline | 15:19:35 | 18:06:10 | 8/8 | +99.16 m | +99.16 m | 2.375 s | Pass |
| 355° | Forecast ETA hold exit | 15:19:35 | 18:06:10 | 8/8 | +99.16 m | +99.16 m | 2.451 s | Pass |
| 5° | ETA-gated baseline | 11:20:30 | 13:02:45 | 8/8 | +132.37 m | +132.37 m | 2.374 s | Pass |
| 5° | Forecast ETA hold exit | 11:19:10 | 13:10:10 | 8/8 | +131.39 m | +131.39 m | 2.419 s | Pass |

| Wind FROM | Change to outbound 002 | Change in round trip | Drift hold, baseline → release | Furthest south, baseline → release | Furthest from 002, baseline → release |
|---:|---:|---:|---:|---:|---:|
| 355° | 0:00:00 | 0:00:00 | 9:30:25 → 9:30:25 | 1,977.3 → 1,977.3 m | 2,699.9 → 2,699.9 m |
| 5° | −0:01:20 | +0:07:25 | 5:45:20 → 5:37:50 | 1,294.8 → 1,293.0 m | 2,023.1 → 2,008.1 m |

Opening displacement is measured south of the launch fix through the outbound 002 capture or the final fix of an unfinished opening leg. Hold time counts applied drift-hold controls. Negative changes are reductions.

- **355°**. The forecast controller retained zero applied hold exits. The paired states and actuator commands matched at every fix.
- **5°**. The first applied forecast release toward outbound 002 was at 2026-10-01T00:02:30 UTC. The estimated time from that decision to first entry was 10:46:10 if sailing began immediately and 10:46:15 if the hold continued for 0:02:30. Predicted ranges at the end of that window were 590.6 m and 591.5 m. The controller applied 90 release commands across 2 forecast windows, with 90 commands on outbound 001 → 002. The launch-relative southward excursion at the first release was 150.9 m. The commanded bow was 58.0° and the ground-course reference was 106.6°. The first paired state or actuator difference occurred at fix 95. The largest leg gain was 0:01:20 on outbound 001 → 002. The largest leg loss was 0:05:25 on return 004 → 003.

Waypoint order is outbound 002 → 003 → 004 → 005, then return 004 → 003 → 002 → 001. Swept full-clearance margins include hulls, the 25 m gap, uncertainty and 3 m reserve. Complete decision timing includes observation, chart, guidance, Direct NMPC and action acceptance.

[Run comparison](forecast_release_comparison.csv) · [Per-leg times, distances and captures](paired_captures.csv) · [Applied release fixes](applied_forecast_release_frames.csv) · `Paired replay data` · `Independent audit` · `Matched inputs` · `Release policy audit`

The independent audit fingerprinted the new forecast release module while the runs were underway. Its file modification time preceded launch, and the hash matched at run closure. `Source binding`

## Replays and labeled MP4s

| Wind FROM | Guidance | Replay | MP4 |
|---:|:---|:---|:---|
| 355° | ETA-gated baseline | [Replay](../otranto_layline_eta/runs/crossing_fan__otranto_355__layline_eta/navigation.jsonl.gz) | `Labeled MP4` |
| 355° | Forecast ETA hold exit | `Replay` | `Labeled MP4` |
| 5° | ETA-gated baseline | [Replay](../otranto_layline_eta/runs/crossing_fan__otranto_005__layline_eta/navigation.jsonl.gz) | `Labeled MP4` |
| 5° | Forecast ETA hold exit | `Replay` | `Labeled MP4` |

`MP4 verification`
