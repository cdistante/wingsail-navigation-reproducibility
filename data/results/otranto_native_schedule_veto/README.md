# Otranto native schedule veto rerun

This record describes the original experiment workspace. The [reviewer evidence index](../../../docs/RESULTS.md) lists the released files and verification commands.

The corrected hybrid matched physics at every capture and on the complete sailed tracks at both wind bearings. At 355° it recovered the prior hybrid’s **6 min 35 s** loss to outbound 002 and **17 min 10 s** loss over the loop.

| Wind FROM | Guidance | Time to outbound 002 | Full loop | Ordered 50 m captures | Minimum full AIS margin | Slowest AIS acceptance | Slowest full decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5° | Physics | 11:20:35 | 13:10:15 | 8/8 | +132.37 m | 2.397 s | 2.503 s |
| 5° | Native schedule veto | 11:20:35 | 13:10:15 | 8/8 | +132.37 m | 2.384 s | 2.513 s |
| 355° | Physics | 15:19:35 | 18:12:15 | 8/8 | +99.16 m | 2.373 s | 2.429 s |
| 355° | Native schedule veto | 15:19:35 | 18:12:15 | 8/8 | +99.16 m | 2.378 s | 2.444 s |

Each arm captured outbound 002 → 003 → 004 → 005 and return 004 → 003 → 002 → 001. The paired replays have identical physical states and actuator commands at all 9,484 shared fixes at 5° and 13,108 at 355°. The four runs used Gamma guidance, direct acados NMPC, a common gated launch at 23:54:50 UTC on September 30 and the same archived wind speeds, current and waves. Both methods used the same crossing-fan fleet, the same additional encounter underway, 20 s report delay, staged and underway emission gaps, COLREG and the candidate AIS handoff. Times are measured from actual launch after 4 min 50 s protected staging. `Waypoint distances and times` · `Pair comparison and regression fix`

## The corrected decision

V2 still ranks candidate first trims with the frozen Transformer weather and AIS scorer. The veto now reads the forecast planner’s committed schedule for each candidate. It uses capture class and first entry time, including the planned turn, switch timing, and the planner’s gate and forecast tracking settings. A native forecast miss has no arrival time. When both schedules capture, a learned alternative needs a gain above the frozen 180 s threshold. The native planner already minimizes its own schedule score over these candidates, so under this criterion the corrected veto retained all physics selections. The live comparisons still evaluated the learned proposals.

At the reproduced 355° decision at 15:08:00 UTC, the learned scorer again proposed the 260° first tack. The native physics schedule planned to enter in **428.5 s** and pass **41.9 m** from 002. The proposed schedule missed with **107.8 m** closest approach. The fixed veto retained the full physics schedule and its **40.5°** first bow reference. The resulting native capture occurred 385 s after the decision. These values were first reconstructed at the archived predecision fix and then confirmed in both fresh native trajectories. `Exact-state regression probe`

## Clearance and timing

The independent audit reconstructed the applied native Fossen steps at 0.05 s and checked the swept distances to received-report projections and analytical traffic truth. The clearance column gives the smaller remaining margin after the 25 m gap, full uncertainty allowance and an additional 3 m reserve. Every applied step cleared both margins and every AIS and full control decision met the 5 s budget. The full timer includes observation, chart and guidance planning, Direct NMPC, and command acceptance. Plant integration and recording run after this timer. The paired runs both exercised the two 65 s held-report peaks during the underway encounter. `Independent final audit` · `Clearance sweeps` · `Control timing`

## Recordings

| Wind FROM | Guidance | Navigation replay | Labeled MP4 |
| --- | --- | --- | --- |
| 5° | Physics | `Replay` | `MP4` |
| 5° | Native schedule veto | `Replay` | `MP4` |
| 355° | Physics | `Replay` | `MP4` |
| 355° | Native schedule veto | `Replay` | `MP4` |

The 355° videos retain the exact corrected tack decision and neighboring fixes. All four videos passed full decode. `Video verification` · `Reproduction and frozen inputs`
