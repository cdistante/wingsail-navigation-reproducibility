# Stage 1 compact ridge in the closed-loop Brindisi benchmark

This replay tests the frozen 11-feature explicit ridge checkpoint from
Stage 1. It selected two safe gains above 25 s among three proposals on
the fixed-state October benchmark. The compact temporal ridge selected
the same actions there. The replay uses the explicit checkpoint and
recomputes eligible schedules and corrected ETAs at the boat's actual
closed-loop states. [Stage 1 fit and fixed-state results](../eta_stage1_expanded/README.md)

The 12 October departures, causal archive weather, simulated Fossen
plant, chart and AIS/COLREG guard match the archived physics comparator.
The controller uses 5 s updates and the plant integrates at 0.05 s.
Waypoint capture remains at 50 m with a six-hour mission cap. A proposed
alternative must be admissible and forecast to capture within 150 s of
the incumbent's ETA. The corrected gain must strictly exceed 25 s.
The replay verifies the model receipt and the original physics inputs,
episodes and trace checksums before launch.
[Original causal physics protocol](../temporal_embedding_native/README.md#protocol)

## Results

| Metric | Physics | Compact explicit ridge |
|---|---:|---:|
| Completed routes | 5/12 | 5/12 |
| Waypoints captured in order | 12/28 | 12/28 |
| Applied learned switches | 0 | 3 |
| Median finish time | 1:03:29.35 | 1:03:04.47 |
| Runs with motion and full AIS clearance | 10/10 | 10/10 |
| Worst full truth-centered AIS margin | +16.387 m | +16.387 m |
| Worst expanded report guard margin | +3.262 m | +3.262 m |
| Control decisions at or above 5 s | 0 | 0 |

The ridge saved a mean of 36.03 s over the five shared finishes. The
paired median change was 0.00 s, with two faster finishes, one slower
finish and two ties. Seven cases per arm remained incomplete, including
six no-safe-action stops and one six-hour cap. Two stopped before motion.

| October case that applied a switch | Elapsed at switch | Corrected gain predicted at switch | Ridge minus physics at 002 | Ridge minus physics at finish |
|---|---:|---:|---:|---:|
| Oct 13 18:00 Brindisi 003 crossing | 865 s | 73.46 s | −190.63 s | −161.90 s |
| Oct 14 12:00 Brindisi 003 overtake | 425 s | 56.41 s | −36.85 s | −24.88 s |
| Oct 14 12:00 Brindisi 003 crossing | 620 s | 59.75 s | +8.60 s | +6.63 s |

All three switches targeted waypoint 002. The Oct 14 crossing decision
at 620 s appears in the live run. The fixed-state Oct 14 overtake
alternative at 665 s was absent after the switch at 425 s changed the
boat's trajectory. Full AIS margins account for hull geometry, 25 m
separation, report uncertainty and the 3 m reserve. The separate replay
reconstructs swept clearance around the reports and synthetic contact
truth. [Paired results](native/paired_episodes.csv) ·
[Independent clearance audit](independent_clearance.json)

## Reproduction

Run from the V2 project in the native Fossen Python environment. Use a
fresh output location for an additional replay.

```sh
study=data/results/eta_ridge_closed_loop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -B "$study/reproduce/run_ridge_loop.py" --workers 8 --output "$study/reproduction/native"
python -B "$study/reproduce/audit_ridge_loop.py" --workers 6 --native "$study/reproduction/native"
```
