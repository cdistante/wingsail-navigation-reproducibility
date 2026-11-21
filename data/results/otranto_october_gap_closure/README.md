# October Otranto return-leg gap-closure experiments

This record describes the original experiment workspace. The [reviewer evidence index](../../../docs/RESULTS.md) lists the released files and verification commands.

This study targets the remaining 75 s gap to Grid MPC on return 003→002,
while preserving the successful outbound policy. It repeats departure at
2026-10-01 13:00 UTC with the same archived wind, waves and current, route,
chart and regional archive crops. True waypoint captures retain a 50 m
radius, ordered outbound 002→003→004→005 and return 004→003→002→001.
Navigation updates every 5 s with a 23,000 s cap, initial heading 45° and
initial flap −10°. No AIS or learned guidance is enabled.

## Selected configuration and current results

[selection.json](selection.json) identifies `gamma_upwind_commit` and
`diffusion_upwind_commit` as the selected pair. Both use a half-radius gate
aim, 60 s maneuver commitment and a 60° continuation-angle guard, active
only on the return phase. The original coarse outbound policy is retained.
The final paired audit passed. The selected Gamma and Diffusion runs have identical states, environmental observations and navigation records at all **1,817 frames**, excluding solver wall-clock measurements.

Both selected runs made all eight visits in order, with independently
verified capture distances between 46.43 and 49.77 m. They finished in
**2:31:20**. Outbound 004→005 remains **32:20**, preserving the **8:50**
gain over Grid MPC. Return 003→002 fell from the prior best **37:40**
to **35:25**, now **1:00 faster than Grid's 36:25**.

| Configuration | Return 004→003 | Return 003→002 | Combined 004→003→002 | Full loop |
|---|---:|---:|---:|---:|
| Grid MPC | 14:35 | 36:25 | 51:00 | 2:38:40 |
| Prior-best return refinement / fresh prior-policy control | 13:45 | 37:40 | 51:25 | 2:32:55 |
| Gate aim, no maneuver commitment | 15:45 | 37:10 | 52:55 | 2:34:40 |
| Gate aim + 60 s commitment, all continuation bearings | 14:10 | 35:55 | 50:05 | 2:31:45 |
| **Selected Gamma / Diffusion: upwind-only gate aim + 60 s commitment** | **14:25** | **35:25** | **49:50** | **2:31:20** |

The combined return interval improves by **1:35 versus the prior best**
and **1:10 versus Grid**, so the target-leg gain is not merely a consequence
of capturing 003 later. The complete loop improves by **1:35 versus the
prior best** and **7:20 versus Grid**. In contrast, gate aiming without
maneuver commitment delayed approach to 003 by 2:00; its 0:30 improvement
on the following leg still left the full loop 1:45 slower than the prior
best. Both positive and negative engineering variants remain recorded.
These are exploratory comparisons on the same archived departure, with
no claim of independent validation or a globally optimal tack schedule.

## Preserved controls

The `baseline manifest` fingerprints the prior
recordings without copying or editing their replays or media. Prior-best
Gamma and Diffusion completed in 2:32:55, with outbound 004→005 in 32:20 and
return 003→002 in 37:40. Grid MPC completed in 2:38:40, taking 41:10 and
36:25 on those two legs. The original coarse forecast completed in 2:33:45.
The launch-time prior-best forecast and runner sources and its comparison
are copied into `baseline_source` as read-only reference records.

## Launch an isolated candidate

Run from the V2 root using the existing simulator/acados environment:

```sh
# Set MARITIME_ACADOS_ENV to the local acados environment script.
source "$MARITIME_ACADOS_ENV"
export PYTHONPATH="$PWD:${PYTHONPATH:-}" MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
"$MARITIME_ACADOS_PYTHON" -B data/results/otranto_october_gap_closure/reproduce/launch.py \
  --arm gamma --name gamma \
  --extra-arg=--forecast-gate-aim-fraction --extra-arg=0.5 \
  --method-label 'Return gate-entry look-ahead (25 m); true captures 50 m'
```

Use `--arm diffusion` for the Diffusion/Direct NMPC counterpart and
`--arm grid_mpc` for a fresh Grid control. `--name` assigns an isolated
candidate folder. `--extra-arg` forwards runner options without changing
shared conditions, and `--method-label` preserves a concise description in
video labels. Existing runs are never overwritten. `--dry-run` prints the
command and manifest without creating a run directory or acados build.

Two further return-only candidate options are exposed in the runner:

- `--forecast-maneuver-commit-seconds 60` retains a newly initiated tack
  long enough to avoid an immediate side reversal during its turn.
- `--forecast-gate-aim-upwind-angle 60` limits nominal gate bias to
  continuations within 60° of wind FROM. The default 180° permits all
  continuation bearings. The true capture radius remains 50 m.

Use isolated names and descriptive labels, for example:

```sh
"$MARITIME_ACADOS_PYTHON" -B data/results/otranto_october_gap_closure/reproduce/launch.py \
  --arm gamma --name gamma_gate_commit \
  --extra-arg=--forecast-gate-aim-fraction --extra-arg=0.5 \
  --extra-arg=--forecast-maneuver-commit-seconds --extra-arg=60 \
  --method-label 'Return gate-entry look-ahead (25 m), 60 s turn commitment; true captures 50 m'
"$MARITIME_ACADOS_PYTHON" -B data/results/otranto_october_gap_closure/reproduce/launch.py \
  --arm gamma --name gamma_upwind_commit \
  --extra-arg=--forecast-gate-aim-fraction --extra-arg=0.5 \
  --extra-arg=--forecast-maneuver-commit-seconds --extra-arg=60 \
  --extra-arg=--forecast-gate-aim-upwind-angle --extra-arg=60 \
  --method-label 'Upwind-only return gate aim (25 m), 60 s turn commitment; true captures 50 m'
```

Each execution preserves its exact command, completed manifest, launch-time
forecast/runner/NMPC/adapter source snapshots, compressed replay, GPX and
full execution log. Gamma/Diffusion default to the prior selected return
configuration: original coarse 120 s outbound scheduling, then refined
angles, tracking feedback and 60 s scheduling at turnaround. Extra runner
options define each new candidate. An internal gate-aim offset, when used,
does not change the true 50 m capture criterion; the audit checks that
criterion independently from recorded positions.

## Verify and export

Run only once every candidate in the study has a completed manifest:

```sh
"$MARITIME_ACADOS_PYTHON" -B data/results/otranto_october_gap_closure/reproduce/audit.py
"$MARITIME_ACADOS_PYTHON" -B data/results/otranto_october_gap_closure/reproduce/render_round_trip.py \
  --include-baselines --jobs 2
```

The audit checks initial-state and forcing parity, every timestamp,
first sampled entry within 50 m in order, GPX/replay consistency and archive
resampling at departure and each capture. It fingerprints the complete
loaded regional archive fields and coordinates. For return-only candidates
it requires exact outbound state and actuator parity with the prior best.
`comparison.json` includes full states at each capture and comparisons
against Grid, prior best and original forecast. CSV exports preserve all
waypoint/leg metrics, capture states, run outcomes and every navigation
field in the five-second telemetry. Distances are sums of geodesic
consecutive-fix chords, bounded by each arm's own capture positions.

The renderer uses a shared north-up camera, sparse wind barbs, wind FROM,
current TO, separate bow/COG markers and full recorded tracks. It samples
frames for video but keeps every capture and every telemetry update in
the replay. Labeled MP4s have hash and full-decoding verification receipts.
`--include-baselines` writes reference clips into this study's
`baseline_media` folder and leaves original recordings untouched.
The readout includes committed tack side, remaining commitment time,
whether gate bias is active, and the continuation-angle guard reason
when recorded. A disabled gate bias leaves the true-center closest-range
label and the blue 50 m capture disks in place.

The final audit passed for **five fresh runs and four preserved controls**,
all with eight ordered captures within 50 m. It verifies all 16,572
recorded frames and 72 capture/segment rows, GPX consistency, and archive
resampling at departure and every capture. The largest archive discrepancy
was zero; the largest GPX/replay rounding difference was below 0.0001 m.
Both selected runs preserve all 1,069 outbound states, actuator values and
observations through capture of 005 relative to the prior best. The fresh
prior-policy control reproduces all 1,836 reference states. The selected
pair matches all 1,817 states and observations and all navigation values
except solver wall-clock timing.

See [comparison.json](comparison.json), [selection.json](selection.json),
[waypoint and capture-state metrics](waypoint_metrics.csv),
[leg metrics](segment_metrics.csv), [run metrics](run_metrics.csv),
[full telemetry](telemetry.csv) and `audit hashes`.
The selection record includes successful assertions for the target leg,
the combined return interval and complete-loop comparisons. No candidate
recording is overwritten.
