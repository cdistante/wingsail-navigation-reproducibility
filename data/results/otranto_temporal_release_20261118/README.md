# Otranto temporal release confirmation

This record describes the original experiment workspace. The [reviewer evidence index](../../../docs/RESULTS.md) lists the released files and verification commands.

## Why the previous release was late

Ten hours after departure at 355°, the Fossen model estimated 0.04288 m/s of sustainable velocity toward 002. This exceeded the classical 0.04 m/s release threshold. The Transformer still assigned hold 99.29% probability and continued holding for 15 more five-minute windows. It finally proposed sail at 11 h 15 min, when the modeled sustainable progress had reached 0.09336 m/s. At 5°, the previous delay was three windows, and the gate VMG at release was 0.07831 m/s.

Replacing all 12 weather inputs with the current observation kept the hold probability at 99.37% at the first 355° release opportunity. Changing just the phase input from holding to underway reduced the probability to 24.23%. That phase change would also have proposed release before useful progress was available. All 50 nearest July training examples in the holding phase, ranked by normalized current waypoint-relative weather, labeled hold at the first 355° release opportunity. The model was overconfident in that weather regime. The timestamped perturbation audit includes the corresponding 5° results. `Release diagnosis` and [input sensitivity](release_lag_sensitivity.csv)

## Timing change

The model keeps proposing hold or sail from observed forcing. With `--confirm-temporal-release`, a pending learned hold can be released if the same waypoint meets the existing 0.04 m/s classical threshold on two consecutive five-minute observations. A missing or failed check breaks the confirmation sequence. After confirmation, the planner continues sailing while the current gate VMG meets the underway feasibility check (0.01 m/s). It resets at a new waypoint or when the feasibility check fails. The existing chart safeguards and controller still apply. Raw proposals and physical confirmation overrides are recorded separately. The one-window confirmation uses the existing weather cadence.

The paired runs use the 30 September 2026 23:50 UTC departure, the saved Otranto 001–005 GPX, five-second control updates and a 24-hour cap. The 355° and 5° controlled FROM directions keep archived wind speed, currents and waves. Each pair uses the same chart, Fossen plant, Direct acados controller and 50 m radius. The 355° pair uses committed laylines and the 5° pair uses reactive laylines, matching their respective incumbents. The revised learned runs use the previous frozen Transformer checkpoint. The prior runs provide the timing reference for this engineering retest.

## Why the first 5° screenshot shows southward motion

At step 3329 the wind is 2.9 m/s FROM 5°, and the current is 0.29 m/s TO 140°. The bow points 312° and the craft moves northwest through the water at 0.223 m/s. Adding the current gives ground motion at 172° and 0.083 m/s. The hold controller limits displacement from its holding point. The measured rate of losing range toward 002 is about 0.081 m/s, compared with 0.197 m/s under passive current drift.

For the same screenshot, the fastest forward trim at each beam-reach bow gives the following estimates. The ten-minute counterfactuals start from the exact recorded state, use the same Fossen dynamics and slew limits, and keep the screenshot forcing fixed at its observed values.

| Bow reference | Steady speed over ground | Steady velocity toward 002 | Range change in 10 min |
|---|---:|---:|---:|
| 312° recorded trim | 0.083 m/s | −0.080 m/s | +48 m |
| 95° east beam reach | 0.523 m/s | −0.194 m/s | +139 m |
| 275° west beam reach | 0.220 m/s | −0.210 m/s | +124 m |

At departure the east reach can briefly close range through its eastward velocity. In the ten-minute launch check, it moved 171 m south and ended 77 m farther from 002. The recorded trim ended 46 m farther away. By step 5673, wind had risen to 5.0 m/s and the northwest tack was making positive progress. The simulated gate VMG was about +0.06 m/s. The new videos distinguish the ground-course reference from the bow reference. [Beam-reach check with launch and both screenshots](beam_reach_check.csv)

## Results

All four runs captured 002 → 003 → 004 → 005 within 50 m. The classical trajectories reproduced their prior deterministic states and actions. Each learned run used the physical override on one decision at 002, with no fallback or return to holding after release.

| Wind FROM | Classical release | Prior learned release | Updated release | Classical finish | Prior learned finish | Updated finish |
|---|---:|---:|---:|---:|---:|---:|
| 355° | 10:00:00 | 11:15:00 | 10:05:00 | 15:08:15 | 15:41:00 | 15:09:25 |
| 5° | 05:55:00 | 06:10:00 | 06:00:00 | 12:51:35 | 12:50:05 | 12:50:05 |

At 355°, the release window moved 70 minutes earlier and the mission finished 31 min 35 s earlier. At 5°, release moved 10 minutes earlier and finish time was unchanged. The replay audit checked 40,316 frames and all 16 captures. There were zero chart-blocked or out-of-coverage samples. All 167 simulator tests passed and all four labeled MP4s passed full decode.

- `Paper comparison`, [CSV](paper_comparison.csv), `paper contrasts` and [paired results](paired_runs.csv)
- `Labeled MP4s, full replays and GPX tracks`
- [Release decisions](release_guard_events.csv), `independent verification` and [waypoint audit](waypoint_metrics.csv)

## Reproduction

Run from V2 with the simulator environment, acados, PyTorch and FFmpeg.

```sh
study="$PWD/data/results/otranto_temporal_release_20261118"
output=$(mktemp -d)
solver=$(mktemp -d)
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python -B "$study/reproduce/run_temporal_release.py" --study "$output" --acados-build-dir "$solver"
python -B "$study/reproduce/audit_temporal_release.py" --study "$output"
python -B "$study/reproduce/render_temporal_release.py" --study "$output"
python -B "$study/reproduce/check_beam_reach.py" --output "$output/beam_reach_check.csv"
```
