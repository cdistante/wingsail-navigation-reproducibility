# Otranto closest-layline comparison

This record describes the original experiment workspace. The [reviewer evidence index](../../../docs/RESULTS.md) lists the released files and verification commands.

The four runs pair closest laylines off and on at wind bearings FROM 355° and 5°. They use wind speed, current and waves from the same 30 September 2026 archive, with wind direction fixed within each pair. Both modes run Gamma with Direct acados NMPC and the same Transformer weather and AIS schedule veto. The crossing-fan AIS encounter, 20 s receiver delay, staged and underway dropouts, COLREG and the 3 m reserve are shared. All four runs launched at 2026-09-30T23:54:50Z after 0:04:50 of protected staging.

Times start at actual release from protected staging. Total time measures return capture at 001. Clearance is the smallest measured excess over vessel hulls, the 25 m gap, full uncertainty and the 3 m reserve. Truth and report projection use separate independently swept checks. The worst full decision time includes observation, chart, guidance, Direct NMPC and action acceptance. Plant propagation and recording follow the timed decision.

| Wind FROM | Closest laylines | Outbound 002 | Round trip | Ordered 50 m captures | Truth margin | Report margin | Max full decision | 5 s budget | Applied layline frames |
|---:|:---|---:|---:|---:|---:|---:|---:|:---|---:|
| 355° | Off | 15:19:35 | 18:12:15 | 8/8 | +99.16 m | +99.16 m | 2.462 s | Pass | 0 |
| 355° | On | 15:19:35 | 18:12:15 | 8/8 | +99.16 m | +99.16 m | 2.448 s | Pass | 0 |
| 5° | Off | 11:20:35 | 13:10:15 | 8/8 | +132.37 m | +132.37 m | 2.443 s | Pass | 0 |
| 5° | On | 11:20:35 | 13:10:15 | 8/8 | +132.37 m | +132.37 m | 2.424 s | Pass | 0 |

| Wind FROM | Time to 002 on minus off | Total on minus off | Maximum track separation at a shared fix |
|---:|---:|---:|---:|
| 355° | 0:00:00 | 0:00:00 | 0.00 m |
| 5° | 0:00:00 | 0:00:00 | 0.00 m |

- The 355° run had zero controls with an applied closest-layline reference. All 13,108 corresponding states and commands matched.
- The 5° run had zero controls with an applied closest-layline reference. All 9,484 corresponding states and commands matched.

The enabled runs each recorded zero fallback controls before 002. Hold and forecast priorities applied through the opening leg.
The 355° run logged 9:30:25 of drift hold before 002, with a maximum recorded range to 002 of 2.70 km.
The 5° run logged 5:45:20 of drift hold before 002, with a maximum recorded range to 002 of 2.02 km.

A chart-clear layline can supply a steering reference after the hold and AIS gates admit nominal guidance. A forecast capture, scheduled tack or committed maneuver retains priority. An enabled flag therefore measures the effect of this fallback in the integrated controller.

The prescribed capture order is outbound 002 → 003 → 004 → 005, then return 004 → 003 → 002 → 001.

[Per-leg times, distances and captures](paired_layline_captures.csv) · [Applied layline references](applied_layline_frames.csv) · [Paired replay comparison](layline_comparison.json) · [Independent acceptance summary](independent_final_summary.json) · [Matched inputs](independent_inputs_verification.json)

## Replays and videos

| Wind FROM | Closest laylines | Raw replay | Labeled video |
|---:|:---|:---|:---|
| 355° | Off | [Replay](runs/crossing_fan__otranto_355__layline_off/navigation.jsonl.gz) | `MP4` |
| 355° | On | `Replay` | `MP4` |
| 5° | Off | [Replay](runs/crossing_fan__otranto_005__layline_off/navigation.jsonl.gz) | `MP4` |
| 5° | On | `Replay` | `MP4` |

`Video verification`
