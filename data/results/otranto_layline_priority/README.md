# Otranto closest-layline priority experiment

This record describes the original experiment workspace. The [reviewer evidence index](../../../docs/RESULTS.md) lists the released files and verification commands.

The two priority runs are paired with the verified layline-off controls from the preceding Otranto study. Both arms use Gamma and Direct acados NMPC, the same Transformer schedule veto and archived wind speed, current and waves. Each pair shares a wind-from bearing, the crossing-fan AIS fixture, 20 s report delay, the reported dropouts and COLREG. The runs launched at 2026-09-30T23:54:50Z after 0:04:50 of protected staging. Times start at release.

With priority enabled, a chart-clear ground layline can supply the route reference as soon as AIS admits nominal guidance. The hold and forecast planners remain available when the layline search yields no candidate. Layline feasibility is tested with a 48-minute steady-polar lookahead into a 25 m planning circle. Waypoint capture uses the 50 m mission radius. Each action still passes the native AIS/COLREG checks. The full clearance margins include ship hulls, the 25 m nominal gap, model uncertainty and 3 m reserve.

| Wind FROM | Guidance | To outbound 002 | Round trip | Ordered captures at 50 m | Truth margin | Report margin | Worst full decision | 5 s checks | Applied layline references |
|---:|:---|---:|---:|---:|---:|---:|---:|:---|---:|
| 355° | Laylines off | 15:19:35 | 18:12:15 | 8/8 | +99.16 m | +99.16 m | 2.462 s | Pass | 0 |
| 355° | Closest layline priority | 15:25:30 | — | 6/8 | +99.16 m | +99.16 m | 2.402 s | Pass | 2411 |
| 5° | Laylines off | 11:20:35 | 13:10:15 | 8/8 | +132.37 m | +132.37 m | 2.443 s | Pass | 0 |
| 5° | Closest layline priority | 11:20:45 | 13:09:40 | 8/8 | +132.37 m | +132.37 m | 2.420 s | Pass | 1804 |

| Wind FROM | Change to 002 | Change in round-trip time | Drift hold before 002, off → priority | Greatest range from 002, off → priority | Greatest southward displacement to 002, off → priority |
|---:|---:|---:|---:|---:|---:|
| 355° | +0:05:55 | — | 9:30:25 → 9:30:25 | 2,700 → 2,700 m | 1,977 → 1,977 m |
| 5° | +0:00:10 | −0:00:35 | 5:45:20 → 5:45:20 | 2,023 → 2,023 m | 1,295 → 1,295 m |

Time changes are priority minus off. A positive change means the priority run took longer.

- At 355°, the first priority command was a closest reachable tack toward outbound 002 at 2026-10-01T14:33:35.000000000 UTC. The planner commanded bow 309.5° and ground course 278.8°. Wind was 8.08 m/s and current was 0.31 m/s toward 166.0°. Priority supplied 561 steering references before outbound 002. Nominal AIS guidance had been continuously admitted for 14:31:10 at first use. The first paired state or actuator difference appeared at fix 10,546. The observed run ended after 24:00:00 with status `step_limit`. The missing time is shown as —. The closest recorded range to return 002 was 120.34 m. That leg recorded 4:44:50 of drift hold and 0 priority references. The largest gain was 0:02:10 on outbound 003 → 004. The largest loss was 0:25:10 on return 005 → 004.
- At 5°, the first priority command was a direct gate approach toward outbound 002 at 2026-10-01T10:34:10.000000000 UTC. The planner commanded bow 318.4° and ground course 302.8°. Wind was 6.99 m/s and current was 0.34 m/s toward 139.9°. Priority supplied 497 steering references before outbound 002. Nominal AIS guidance had been continuously admitted for 10:31:40 at first use. The first paired state or actuator difference appeared at fix 7,673. The largest gain was 0:03:10 on outbound 003 → 004. The largest loss was 0:03:35 on return 005 → 004.

The prescribed capture order is outbound 002 → 003 → 004 → 005, then return 004 → 003 → 002 → 001. Full control timing includes observation, chart, guidance, Direct NMPC and action acceptance. Replay writing and integration run after the timed decision.

`Run comparison` · `Per-leg times, distances and captures` · `Applied priority references` · `Paired replay comparison` · [Independent audit](independent_final_summary.json) · `Matched inputs`

## Replays and labeled MP4s

| Wind FROM | Guidance | Replay | Labeled MP4 |
|---:|:---|:---|:---|
| 355° | Laylines off | [Replay](../otranto_closest_layline_ablation/runs/crossing_fan__otranto_355__layline_off/navigation.jsonl.gz) | `MP4` |
| 355° | Closest layline priority | [Replay](runs/crossing_fan__otranto_355__layline_priority/navigation.jsonl.gz) | `MP4` |
| 5° | Laylines off | [Replay](../otranto_closest_layline_ablation/runs/crossing_fan__otranto_005__layline_off/navigation.jsonl.gz) | `MP4` |
| 5° | Closest layline priority | [Replay](runs/crossing_fan__otranto_005__layline_priority/navigation.jsonl.gz) | `MP4` |

`Priority MP4 verification`
