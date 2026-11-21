# Causal wind trend in the protected Brindisi tack planner

This comparison reuses the fixed twelve October Brindisi cases, the
protected-launch records, and the frozen 11-feature ridge selector. It
changes future wind in the native 48 min tack planner. Both arms use the
same 50 m gate, AIS request clock, six-hour request-to-finish cap,
interpolated Fossen plant, and uncapped AIS, COLREG and chart screens.
The logged persistence staging checks and release times are identical
across the arms. [Protected departure baseline](../brindisi_escape_departure/README.md)

The causal wind trend uses half the difference between the current and
previous valid-hour wind vectors. The forecast change is capped at
1.4203 m/s before the ramp. The first planning slot uses observed wind.
The future slots interpolate from the observed vector to the next-hour
prediction, then hold the prediction. Current and wave height persist
at their latest observed values across all planning slots. A following
tack uses the polar at its predicted switch slot. The applied controls
use the same measured inputs and screens as the persistence arm. The
trend is pinned to the April-selected setting from the offline study.
[Offline forecast selection](../stage3_forcing_offline/README.md)

## Results

| Metric | Protected departure with persistence | Protected departure with hybrid forcing |
|---|---:|---:|
| Completed routes | 8/12 | 8/12 |
| Waypoints captured in order | 20/28 | 20/28 |
| Median sailing time among the eight finishers | 1:04:53.91 | 1:04:54.30 |
| Median requested-departure-to-finish time | 1:06:31.41 | 1:06:57.38 |
| Full AIS clearance in cases with motion | 10/10 | 10/10 |
| Minimum full truth-centered margin | +14.746 m | +14.746 m |
| Minimum expanded report guard margin | +5.067 m | +5.067 m |
| Longest sailing control decision | 0.185 s | 0.476 s |
| Sailing control decisions at or above 5 s | 0 | 0 |

The median paired change was **+25.17 s** and the mean was **+43.65 s**.
The hybrid finished two cases faster and six slower. Identical waits
give the same paired changes in sailing and request-to-finish time.
The largest savings were 190.35 s on the Oct 13 Brindisi 003 overtake
and 70.27 s on the Oct 16 Brindisi 004 overtake. The Oct 16 and Oct 17
Brindisi 004 crossing runs took 254.29 s and 265.94 s longer,
respectively.

Both Brindisi Small cases still reached the six-hour cap, with zero
captures. At the recorded control endpoints, the closest approach to
002 remained **223.63 m** in the crossing and **202.81 m** in the
overtake. Both planners logged zero decisions with a finite ETA to
002 in those cases. The two October 18 cases retained their original
protected wait limit and stayed unlaunched.

[Paired cases and timeout diagnostics](native/paired_episodes.csv) ·
[Independent clearance and causality audit](independent_clearance.json)

## Reproduction

Run from the V2 root in the native Fossen Python environment. The
provider tests reproduce the offline trend at all 1,512 October hourly
endpoints and check that future archive values cannot affect the
computed forecast.

```sh
study=data/results/brindisi_hybrid_forcing
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -B "$study/reproduce/run_hybrid.py" --workers 8 --output "$study/reproduction/native"
python -B "$study/reproduce/audit_hybrid.py" --workers 6 --native "$study/reproduction/native"
```
