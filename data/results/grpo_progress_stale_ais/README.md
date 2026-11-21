# Stale AIS clearance retest with physics and progress-shaped GRPO

The original 13 October 2026 18:00 UTC Brindisi 003 raking-overtake run
passed the report-centered clearance check during the interval from
165 to 170 s. The contact had begun turning and the last delivered
report was 45 s old at the start of the interval. Replaying the same
command gives a minimum margin of +0.166 m around the projected report
center and −1.096 m around the moving true center. Physics and GRPO
applied the same control in that interval.

## Shared guard change

Let \(B\) be the sum of the circular hull radii, the 25 m gap and the
3 m reserve. The existing full envelope has radius \(B + U\), where
\(U\) covers report position and age-dependent motion uncertainty.
The robust guard checks a radius of \(B + 2U\) around the report
projection. For any actual center within \(U\) of that projection,
the triangle inequality guarantees separation greater than \(B + U\)
around the actual center when the robust check passes. Each 0.05 s
chord is checked against the larger robust radius at its endpoints.
The check screens candidate and committed native commands using
only delivered AIS reports.

The replay audits keep the original \(B + U\) threshold around both
centers for prior-run comparisons. An independent replay of the
original failing command gives a robust guard margin of −37.355 m.
With the guard active, both policies stop at 150 s in that case
with a minimum measured full truth margin of +29.426 m.

[Native guard](reproduce/native_stale_guard.py) ·
[Original clearance audit](../grpo_progress_native/independent_clearance.json)

## Locked retest and results

Both arms reran all 12 original held-out cases. The three Brindisi
routes, six route and departure groups, autumn weather, the two AIS
profiles, 50 m capture circles, six-hour cap, 0.05 s Fossen integration,
5 s helm, and chart and COLREG screening match the first comparison.
The GRPO arm uses the same checkpoint, adapter and residual strength.
The simulator stops when the shared screen finds no admissible native
control. Margins apply to the motion executed before that stop.

| Metric | Previous, each policy | Physics with guard | GRPO with guard |
| --- | ---: | ---: | ---: |
| Routes completed in original waypoint order | 5/12 | 4/12 | 4/12 |
| Waypoints captured in order | 12/28 | 10/28 | 10/28 |
| Median time among completed routes | 1:01:28 | 1:03:50 | 1:03:50 |
| Full clearance on runs with applied motion | 9/10 | 10/10 | 10/10 |
| Worst full truth-centered margin | −1.096 m | +27.651 m | +27.651 m |
| Worst full report-centered margin | +0.166 m | +27.651 m | +27.651 m |
| Stops with no admissible native control | 6 | 8 | 8 |
| Capped at six hours | 1 | 0 | 0 |

The four jointly completed routes still have a median GRPO-minus-physics
time change of 0 s. Three tied. On the 17 October 12:00 UTC Brindisi 004
oblique crossing, GRPO retained four tack changes and a 179.72 s gain
(1:14:26 versus 1:17:26). All four retained completions reproduced their
prior finish times. The median over each arm's finishes increased
because the completed set changed.

The 14 October 12:00 UTC Brindisi 003 raking overtake previously finished
in 1:01:28. Under the robust guard it stopped before the first gate at
170 s in both arms. The 12 October Brindisi Small oblique crossing moved
from the six-hour cap to a stop at 175 s. In each arm, two cases stopped
before applying a control and have no swept-clearance measurement.

The independent auditor reintegrated 6,532 applied controls over
652,150 native substeps. All swept truth and report margins were
positive. The smallest margin under the expanded planning envelope
was +0.036 m, and the smallest remaining position allowance after
accounting for observed center error was +7.769 m. These are separate
from the original full-clearance metric in the table.

[Paired results with prior-run differences](native/paired_episodes.csv) ·
[Waypoint and timing audit](native/independent_comparison.json) ·
[Independent full-clearance replay](independent_clearance.json) ·
[Physics telemetry](native/physics/telemetry) ·
[GRPO telemetry](native/grpo_progress/telemetry)

## Reproduce

Run from the V2 root in the native Python environment used by the
[original benchmark](../grpo_progress_native/README.md). The runner
binds the inputs and unchanged checkpoint to the original receipts.
Use fresh output directories when regenerating episodes.

~~~sh
study=data/results/grpo_progress_stale_ais
export PYTHONPATH="$PWD" MPLCONFIGDIR=/tmp/maritime_grpo_mpl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -B "$study/reproduce/run_stale_retest.py" physics --workers 6
python -B "$study/reproduce/run_stale_retest.py" grpo_progress --workers 6
python -B "$study/reproduce/verify_stale_retest.py"
python -B "$study/reproduce/audit_stale_clearance.py" --workers 6
~~~
