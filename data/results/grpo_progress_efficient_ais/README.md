# Reducing sailing time with early AIS avoidance

The preceding early AIS retest recovered the Brindisi 003 overtaking
finish, with a 1:05:35.81 elapsed time and +65.894 m minimum full
truth-centered clearance. It also increased time on the four routes
that already completed with the robust-only guard. This retest changes
how early avoidance chooses between admissible tacks and returns to
the nominal sailing schedule.

## Navigation change

The native simulator forecasts a constant trim for 240 s under the
wind, waves, current and AIS observations available at each decision.
Forecast uncertainty growth uses the same 65 s report-age budget as
the preceding early AIS experiment, with at least one 5 s interval
of growth. The forecast screens the whole projected path against the
chart when comparing early alternatives.

When the nominal preview conflicts, alternative tacks must pass
the native immediate robust AIS, COLREG and chart screens. The
planner uses the existing route-time score to choose among candidate
previews with positive clearance. When no candidate preview clears,
it favors the largest forecast minimum clearance among the
available chart-clear previews. A nominal forecast with nonnegative
clearance can resume immediately. The route still replans after
five seconds while that nominal margin is at or below 10 m.

The applied guard retains the original uncapped robust requirement
\(B + 2U\). The replay audit uses the original \(B + U\) full envelope
around both the reported projection and the synthetic true contact.
Here \(B\) includes hulls, the 25 m gap and 3 m reserve, and \(U\) is the
position and motion uncertainty. The forecast ranks short-term control
choices. Only controls that pass the applied screen can move the boat.

[Planner implementation](reproduce/native_efficient_guard.py) ·
[Prior early AIS comparison](../grpo_progress_anticipatory_ais/README.md) ·
[Robust-only comparison](../grpo_progress_stale_ais/README.md)

## Matched rerun

Both policies use all 12 original cases. The three Brindisi routes,
six route and departure groups, two traffic profiles, AIS schedules,
weather archive and bathymetry are the same as before. The native
simulator retains 0.05 s integration, 5 s helm updates, 50 m waypoint
circles and six-hour caps. Both comparisons use the same physics
scheduler and frozen progress-trained GRPO checkpoint. Development
probes used the named overtaking case and selected crossing cases
from the matched roster.

## Results

| Metric | Previous early AIS, either policy | Revised physics | Revised GRPO |
| --- | ---: | ---: | ---: |
| Routes completed | 5/12 | 5/12 | 5/12 |
| Waypoints reached in order | 12/28 | 12/28 | 12/28 |
| Median sailing time among completed routes | 1:05:35.81 | 1:01:58.77 | 1:01:58.77 |
| Median paired sailing change vs previous early AIS | Reference | −62.8 s | −62.8 s |
| Full clearance on runs with applied motion | 10/10 | 10/10 | 10/10 |
| Worst full truth-centered margin | +16.324 m | +16.324 m | +16.324 m |
| Worst full report-centered margin | +16.324 m | +16.324 m | +16.324 m |
| Stops with no admissible control | 6 | 6 | 6 |
| Six-hour caps | 1 | 1 | 1 |

Both policies kept the Brindisi 003 overtaking finish in order
002–003. The revised elapsed time of 1:00:32.69 saves 303.12 s
against the previous early-avoidance result and 55.16 s against
the 1:01:27.85 finish before the robust guard. The minimum full
clearance on the revised run was +57.259 m about the true contact
and +62.696 m about the report projection. Waypoint 002 was reached
at 50:41.89.

The launch decision selected the same diversion as the previous
early avoidance. The first actuator difference occurs at 135 s,
when the route-ranked forecast picks mode 09 and the previous
planner picks mode 12. At 140 s the revised planner returns to the
nominal trim and continues to check it at 5 s intervals.

Among the four original robust-only finishes, the median added time
fell from 110.62 s to 46.37 s for physics and from 97.65 s to 25.08 s
for GRPO. Physics took 38.64 s longer than with the preceding early
AIS planner on the 16 October Brindisi 004 crossing. In the new
physics–GRPO pairing, GRPO was 42.58 s faster on that crossing and
8.76 s slower on the 17 October Brindisi 004 crossing. The three other
finishes tied, giving a median paired difference of 0 s. All complete
control decisions stayed below 5 s.

The original 13 October Brindisi 003 stale-report breach case again
stopped at 120 s with positive full clearance through the applied
prefix. Two cases per method stopped before motion and have no
swept-clearance measurement.

The independent audit replayed all 24 traces over 1,650,925 Fossen
substeps. It found zero full-clearance incursions and reproduced the
first forecast decision in each of the 20 affected traces with later
observations and synthetic truth withheld. The minimum margin under
the expanded applied guard was +3.199 m across the benchmark and
+4.104 m on the recovered overtaking finish.

[Paired results](native/paired_episodes.csv) ·
[Waypoint, timing and provenance audit](native/independent_comparison.json) ·
[Independent clearance and causal replay](independent_clearance.json) ·
[Physics telemetry](native/physics/telemetry) ·
[GRPO telemetry](native/grpo_progress/telemetry)

## Reproduce

Use the [native benchmark environment](../grpo_progress_native/README.md)
from the V2 root. The run receipts pin the locked scenarios, previous
comparisons, model and guard sources. Use a fresh output directory
when rerunning an arm.

~~~sh
study=data/results/grpo_progress_efficient_ais
export PYTHONPATH="$PWD" MPLCONFIGDIR=/tmp/maritime_grpo_mpl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -B "$study/reproduce/run_efficient_retest.py" physics --workers 6
python -B "$study/reproduce/run_efficient_retest.py" grpo_progress --workers 6
python -B "$study/reproduce/verify_efficient_retest.py"
python -B "$study/reproduce/audit_efficient_clearance.py" --workers 6
~~~
