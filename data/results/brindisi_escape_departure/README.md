# Escape and protected departure in the Brindisi Fossen benchmark

The two experimental arms replay the same twelve October fixtures at
the 50 m capture radius and use the frozen compact explicit ridge.
They share the causal weather input, interpolated simulated plant,
original AIS request clock, COLREG and chart checks and uncapped
five-second admission guard. The prior independently audited ridge
run supplies the comparator. [Frozen ridge benchmark](../eta_ridge_closed_loop/README.md)

The escape arm activates when all chart-admitted constant-trim
forecasts conflict. It tests two-trim paths under the currently
delivered reports with switches after 30 or 60 s. The search screens
the path for chart clearance and models COLREG at the planned switch.
It uses the existing 240 s preview and takes the first step only when
the predicted expanded AIS margin improves by more than 2 m over the
existing fallback. Each applied control goes through the same live
uncapped screen.

The departure arm polls every 5 s at a modeled protected launch point,
for at most 900 s. It checks 60 s of uncapped expanded AIS clearance
and a 240 s conditional preview with chart and COLREG checks. The
initial boat state is secured during waiting. Sailing begins at the
first admitted launch, with fresh as-of weather and the original AIS
fixture clock. Wait is charged against the six-hour mission cap.
Swept clearance is measured along the simulated sailing paths.

## Results

| Arm | Completed routes | Ordered captures | Moving cases with full AIS clearance | Worst full truth margin | Largest control or launch check |
|---|---:|---:|---:|---:|---:|
| Existing ridge | 5/12 | 12/28 | 10/10 | +16.387 m | 0.199 s |
| Two-trim escape | 5/12 | 12/28 | 10/10 | +16.387 m | 0.558 s |
| Protected departure | 8/12 | 20/28 | 10/10 | +14.746 m | 0.216 s |

The escape arm selected eight escape previews across 149 searches.
It extended the October 13 overtaking run from 120 s to 165 s before
the expanded guard stopped it. It retained the same five successful
routes and their original finish times.

Protected departure removed all four stops that had occurred after
motion. Three overtaking cases finished with waits of 195–205 s.
The Brindisi Small overtaking case cleared traffic after a 205 s wait
and reached the six-hour mission cap without reaching 002. The
original Brindisi Small crossing also timed out. The two October 18
cases exhausted the 900 s wait limit before launch. Across the five
original common finishes, the October 14 overtake sailed 95.59 s
faster and finished 99.41 s later from the requested departure time.
The other four shared finishes tied.

Full AIS margin includes hulls, 25 m separation, report uncertainty
and 3 m reserve. All logged control attempts and staging checks were
below 5 s. The audit reconstructs the applied trajectories and
replays the staging checks using the original report availability
times. [Paired cases](native/paired_episodes.csv) ·
[Independent clearance audit](independent_clearance.json)

## Reproduction

Run from the V2 root in the native Fossen Python environment.

```sh
study=data/results/brindisi_escape_departure
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -B "$study/reproduce/run_recovery.py" --workers 6 --output "$study/reproduction/native"
python -B "$study/reproduce/recovery_audit.py" --workers 6 --native "$study/reproduction/native"
```
