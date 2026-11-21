# Otranto ETA-gated closest-layline experiment

This record describes the original experiment workspace. The [reviewer evidence index](../../../docs/RESULTS.md) lists the released files and verification commands.

The two ETA-gated runs are compared with the verified layline-off controls. Each pair uses the same archived wind, current and waves, Gamma guidance, Direct acados NMPC, Transformer schedule veto, AIS traffic and dropouts, and COLREG settings. Times start at release at 2026-09-30T23:54:50Z after 0:04:50 of protected staging.

After AIS admits route guidance, the closest feasible layline can replace the forecast tack if its predicted first entry to the next waypoint is earlier. The layline planner aims at the inner 25 m circle. Both candidate paths are reprojected from the decision state using the same archived forecast speed, polar and turn assumptions. The comparison measures first entry to the original 50 m gate and requires a finite capture prediction from both candidates. Holds and forecast turns already underway keep priority. Each applied command still passes chart and AIS/COLREG checks. The full clearance margin includes ship hulls, the 25 m gap, uncertainty and a 3 m reserve.

| Wind FROM | Guidance | To outbound 002 | Loop or stop | Ordered 50 m captures | Truth margin | Report margin | Slowest decision | 5 s checks | ETA references |
|---:|:---|---:|---:|---:|---:|---:|---:|:---|---:|
| 355° | Laylines off | 15:19:35 | 18:12:15 | 8/8 | +99.16 m | +99.16 m | 2.462 s | Pass | 0 |
| 355° | ETA-gated closest layline | 15:19:35 | 18:06:10 | 8/8 | +99.16 m | +99.16 m | 2.375 s | Pass | 420 |
| 5° | Laylines off | 11:20:35 | 13:10:15 | 8/8 | +132.37 m | +132.37 m | 2.443 s | Pass | 0 |
| 5° | ETA-gated closest layline | 11:20:30 | 13:02:45 | 8/8 | +132.37 m | +132.37 m | 2.374 s | Pass | 676 |

| Wind FROM | Change to 002 | Change in loop time | Drift hold before 002, off → ETA | Furthest from 002, off → ETA | Furthest south before 002, off → ETA |
|---:|---:|---:|---:|---:|---:|
| 355° | 0:00:00 | −0:06:05 | 9:30:25 → 9:30:25 | 2,700 → 2,700 m | 1,977 → 1,977 m |
| 5° | −0:00:05 | −0:07:30 | 5:45:20 → 5:45:20 | 2,023 → 2,023 m | 1,295 → 1,295 m |

Time changes are ETA-gated minus layline-off.

- **355°**. The first ETA-gated command was a direct gate approach toward outbound 004 at 2026-10-01T15:23:25 UTC, 15:21:00 after continuous AIS admission began. Predicted entry was 563.2 s for the layline and 701.2 s for forecast tacking, a gain of 138.0 s. The run retained 420 ETA-gated references, including 0 before outbound 002. The median predicted gain was 19.8 s. The first affected mark was captured 0:08:45 after that decision. The first paired state or actuator change occurred at fix 11,146. The largest leg gain was 0:06:05 on return 004 → 003. The largest leg loss was 0:00:40 on return 003 → 002.
- **5°**. The first ETA-gated command was a direct gate approach toward outbound 002 at 2026-10-01T10:34:10 UTC, 10:31:40 after continuous AIS admission began. Predicted entry was 2587.9 s for the layline and 2626.2 s for forecast tacking, a gain of 38.3 s. The run retained 676 ETA-gated references, including 381 before outbound 002. The median predicted gain was 25.8 s. The first affected mark was captured 0:41:10 after that decision. Layline-off took 0:41:15 from the same decision time. The first paired state or actuator change occurred at fix 7,673. The largest leg gain was 0:05:30 on return 004 → 003. The largest leg loss was 0:01:15 on outbound 004 → 005.

In the earlier priority-only trial, the closest feasible layline could take over without an ETA comparison.

- At 355°, the priority-only outcome was 6/8 captures at 24:00:00. The ETA-gated outcome was 18:06:10.
- At 5°, the priority-only outcome was 13:09:40. The ETA-gated outcome was 13:02:45. [Priority-only experiment](../otranto_layline_priority/README.md)

Capture order is outbound 002 → 003 → 004 → 005, then return 004 → 003 → 002 → 001. Decision timing includes observation, chart, guidance, Direct NMPC and action acceptance. Replay writing and integration follow the timed decision.

[Run comparison](eta_comparison.csv) · [Per-leg times and distances](paired_layline_captures.csv) · [Applied ETA comparisons](applied_eta_frames.csv) · [Paired replay data](eta_comparison.json) · [Independent audit](independent_final_summary.json) · [Matched inputs](independent_inputs_verification.json)

## Replays and labeled MP4s

| Wind FROM | Guidance | Replay | MP4 |
|---:|:---|:---|:---|
| 355° | Laylines off | [Replay](../otranto_closest_layline_ablation/runs/crossing_fan__otranto_355__layline_off/navigation.jsonl.gz) | `Labeled MP4` |
| 355° | ETA-gated closest layline | [Replay](runs/crossing_fan__otranto_355__layline_eta/navigation.jsonl.gz) | `Labeled MP4` |
| 5° | Laylines off | [Replay](../otranto_closest_layline_ablation/runs/crossing_fan__otranto_005__layline_off/navigation.jsonl.gz) | `Labeled MP4` |
| 5° | ETA-gated closest layline | [Replay](runs/crossing_fan__otranto_005__layline_eta/navigation.jsonl.gz) | `Labeled MP4` |

`ETA-gated MP4 verification`
