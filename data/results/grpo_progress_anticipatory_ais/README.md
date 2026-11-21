# Earlier AIS avoidance in the native Fossen retest

The robust-only guard stopped both policies before waypoint 002 on
the 14 October 2026 12:00 UTC Brindisi 003 raking-overtake case. That
control loop checked the next five seconds, while a native tack
could remain committed for up to 120 s. By 170 s every available
native trim failed the immediate screen.

## Causal forecast and applied clearance

At a native scheduling decision, the early AIS layer projects the
nominal trim for 180 s. The preview uses the same Fossen dynamics and
trim feedback with the current wind, waves and current held constant.
Contacts follow the course and speed on the delivered AIS reports.
Forecast-only uncertainty growth uses a 65 s report-age budget, with
at least one 5 s interval of growth. The preview determines when to
divert. The applied-control check still uses the full robust swept
envelope without an age-growth cap.

If the nominal preview enters the expanded envelope, the layer
compares other trims that pass the immediate robust, chart and
COLREG screens. It prefers the largest preview minimum among
chart-clear forecasts and reconsiders after each 5 s control.
If every preview predicts a conflict, their margins can still
rank currently admissible trims. After diversion, regular sailing
resumes when the nominal forecast clears by at least 8 m.

The original full clearance metric includes hull radii, the 25 m
gap, report uncertainty and the 3 m reserve. The applied guard
uses the additional uncertainty allowance for the difference between
the projected and true contact positions. The navigator uses only
received AIS information. The audit independently sweeps the
original full margins around the two contact centers.

[Earlier AIS implementation](reproduce/native_anticipatory_guard.py) ·
[Previous robust guard](../grpo_progress_stale_ais/reproduce/native_stale_guard.py) ·
[Robust-only comparison](../grpo_progress_stale_ais/README.md)

## Matched test

The retest uses the original 12 cases across three Brindisi routes,
six route and departure groups, and the same paired oblique-crossing
and raking-overtake fixtures. The archive, actual traffic trajectories
and report delays and dropouts, waypoint order, 50 m radius, six-hour
cap and 0.05 s integration are locked to the previous comparison.
The frozen GRPO checkpoint and residual strength carry over.
The reported run set includes the case used to develop the earlier
avoidance maneuver.

For Brindisi 003, the forecast chose a new trim at launch, changing the
bow reference from 46.51° to 71.51° with the latest report 20 s old.
It reduced the projected deficit from −52.36 m to −38.38 m.
The robust applied screen accepted the first step. The slew-limited
helm first applied a different rudder command at 45 s. During this
encounter the planner logged 28 early decisions from 0 to 135 s.

## Results

| Metric | Robust-only, each policy | Earlier AIS with physics | Earlier AIS with GRPO |
| --- | ---: | ---: | ---: |
| Routes completed | 4/12 | 5/12 | 5/12 |
| Waypoints captured in order | 10/28 | 12/28 | 12/28 |
| Median time among completed routes | 1:03:50.44 | 1:05:35.81 | 1:05:35.81 |
| Runs with applied motion that maintained full clearance | 10/10 | 10/10 | 10/10 |
| Worst full truth-centered margin | +27.651 m | +16.324 m | +16.324 m |
| Worst full report-centered margin | +27.651 m | +16.324 m | +16.324 m |
| Stops with no admissible control | 8 | 6 | 6 |
| Six-hour caps | 0 | 1 | 1 |

Both policies regained the Brindisi 003 overtaking finish. Both captured
002 and 003 in order in 1:05:35.81, with minimum full margins of
+65.894 m about the true center and +67.027 m about the report center.
The first 50 m gate was entered at 54:59.92. The pre-robust version of
this case finished in 1:01:27.85, so the early maneuver added 247.96 s
to that finish. The robust-only run stopped at 170 s. A replay restricted
to observations available at launch reproduced the initial new tack
in both arms without accessing synthetic truth.

The four routes that finished under the robust-only guard stayed
complete. All four took longer under earlier avoidance, with median
additions of 110.62 s for physics and 97.65 s for GRPO. Among the five
new matched finishes, GRPO saved 252.85 s on the 17 October Brindisi
004 oblique crossing and tied on the other four. The median paired
GRPO-minus-physics difference was 0 s. Neither arm recorded a complete
control decision at or above 5 s.

There were no full clearance incursions on applied motion. Both arms
stopped before motion in two cases. The 12 October Brindisi Small oblique
crossing ran to the six-hour cap after the earlier maneuver. The
13 October Brindisi 003 historical breach case still stopped early,
with its full margin positive through the applied prefix.

The independent audit checked all 24 traces over 1,674,027 integration
substeps. The smallest margin against the expanded applied guard was
+3.199 m across the benchmark and +12.776 m on the recovered overtaking
run. These margins use the expanded planning envelope, separate from
the original full-clearance margins in the table.

[Paired results](native/paired_episodes.csv) ·
[Waypoint, timing and provenance audit](native/independent_comparison.json) ·
[Independent clearance and causal replay](independent_clearance.json) ·
[Physics telemetry](native/physics/telemetry) ·
[GRPO telemetry](native/grpo_progress/telemetry)

## Reproduce

Run from the V2 root in the [native benchmark
environment](../grpo_progress_native/README.md). Use a fresh output
directory when running the same arm again.

~~~sh
study=data/results/grpo_progress_anticipatory_ais
export PYTHONPATH="$PWD" MPLCONFIGDIR=/tmp/maritime_grpo_mpl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -B "$study/reproduce/run_anticipatory_retest.py" physics --workers 6
python -B "$study/reproduce/run_anticipatory_retest.py" grpo_progress --workers 6
python -B "$study/reproduce/verify_anticipatory_retest.py"
python -B "$study/reproduce/audit_anticipatory_clearance.py" --workers 6
~~~
