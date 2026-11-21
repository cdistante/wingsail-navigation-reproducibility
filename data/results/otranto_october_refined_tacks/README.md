# October Otranto forecast-tack refinement

This record describes the original experiment workspace. The [reviewer evidence index](../../../docs/RESULTS.md) lists the released files and verification commands.

This study repeats departure at 2026-10-01 13:00 UTC on the five-waypoint
Otranto route with the same archived marine providers and regional crops as
the [preceding comparison](../otranto_october_forecast_tacks/README.md).
All runs use a 50 m capture radius, 5 s updates, initial heading 45°, initial
flap −10°, and a 23,000 s cap, with explicit return to waypoint 001.
No AIS traffic or learned guidance is enabled. The intended capture order is
outbound 002→003→004→005, then return 004→003→002→001.

The selected configuration enables refinement on the return phase only, retaining
the original coarse scheduler without feedback and with 120 s intervals on
outbound. At turnaround it creates a new forecast scheduler with intermediate
close-hauled sailing modes and feedback from the observed ground velocity,
using 60 s tack intervals with the same 48-minute horizon. The
`gamma_all_60` and `diffusion_all_60` ablations apply these settings on both
phases. The `gamma_angles_only` ablation enables the extra angles
and disables this tracking feedback with 120 s tack intervals. The
`gamma_feedback_120` ablation adds feedback while retaining 120 s intervals.
Gamma and Diffusion arms both use Direct acados NMPC and the same forecast
settings; the Grid MPC reference uses Gamma with reactive guidance. The
original coarse scheduler remains selectable for a matched control. Both
controller and scheduler differ between the Grid and forecast arms.

## Results and choice of configuration

The selected Gamma and Diffusion runs reached all eight visits in order and
finished in **2:32:55**, compared with **2:33:45** for the original forecast
scheduler and **2:38:40** for the fresh Grid MPC run. Their outbound state
sequences are identical to their original forecast recordings through capture
of 005. Thus, the outbound 004→005 saving of **8:50** against Grid is
preserved exactly, including its **1,178.6 m** track length.

On return 003→002, the selected configuration took **37:40** over
**1,478.2 m**, improving the original forecast's **38:30 / 1,538.5 m** by
**50 s and 60.3 m**. It still took **1:15 longer** than Grid on this leg;
Grid covered **1,366.9 m in 36:25**. The selected forecast configuration
finished the full loop **5:45 ahead of Grid** and **50 s ahead of the
original forecast**. Its full ground track was **6,362.5 m**, versus
**6,415.1 m** for the original forecast and **6,685.3 m** for Grid.

| Completed arm | Refinement phase | Tack interval (s) | Outbound 004→005 | Return 003→002 | Full loop | Change from original forecast |
|---|---|---:|---:|---:|---:|---:|
| Fresh Gamma + Grid MPC | Reactive guidance | — | 41:10 | 36:25 | 2:38:40 | +4:55 |
| Original Gamma + Direct NMPC | Coarse modes, no feedback | 120 | 32:20 | 38:30 | 2:33:45 | Reference |
| Gamma, fine angles only | Both phases | 120 | 32:10 | 38:45 | 2:33:55 | +0:10 |
| Gamma, fine angles + feedback | Both phases | 120 | 32:15 | 38:30 | 2:34:00 | +0:15 |
| Gamma, fine angles + feedback | Both phases | 60 | 31:05 | 37:10 | 2:34:55 | +1:10 |
| Diffusion, fine angles + feedback | Both phases | 60 | 31:05 | 37:10 | 2:34:55 | +1:10 |
| **Selected Gamma / Diffusion, fine angles + feedback** | **Return only** | **120 outbound / 60 return** | **32:20** | **37:40** | **2:32:55** | **−0:50** |

All completed arms in this table made eight ordered 50 m captures. The
all-phase 60 s candidate improved both highlighted legs but delayed the
opening 001→002 leg from 38:10 to 43:25, leaving the complete loop 1:10
slower. Fine angles alone and the 120 s feedback candidate also failed to
reduce whole-route time. These unsuccessful engineering candidates remain
in the study as ablations, rather than being omitted from the comparison.
Restricting the change to the return phase preserves the useful original
outbound behavior while allowing a more responsive return schedule.

The selected Gamma return 003→002 run switched out of its initial
northeast tack after **14:35**, compared with **15:35** for the original
forecast. That first tack shortened by **35.3 m**, and the following
westward tack shortened by **25.0 m**. The westward tack took 10 s more,
so the 60 s earlier switch produced a net 50 s leg saving. These are
measured track and timing differences between complete runs; leg-entry
states and archive times may differ. The comparison does not isolate a
globally optimal schedule or establish performance on an independent
departure.

The feedback estimates settled velocity error separately on each tack,
using logged surge/sway, heading and current to reconstruct ground
velocity. It requires 60 s of settled motion, bounds each residual norm
at 0.15 m/s and smooths estimates with a 60 s time constant. Corrections
affect schedule prediction; the emitted nominal bow/course and Direct
NMPC actuator control are retained. The 60 s forecast interval retains
the same 48-minute horizon, and navigation still updates every 5 s.
The `independent tack/track diagnostics`
and `observer reconstruction` document the
recorded mechanism and numerical checks.

The final return-only Gamma and Diffusion runs have identical boat states
and navigation records at all **1,836 replay frames**, excluding solver
wall-clock timing. Both preserve all **1,069 outbound state frames** through
capture of 005 relative to their respective original forecast recordings.
The fresh Grid run reproduces its prior recording at all **1,905 frames**,
and the newly executed coarse Gamma control reproduces all **1,846 states**
of its original recording. Additional feedback metadata explains the
navigation-dictionary differences for comparisons against older recordings.

The final audit passed for **eight newly executed arms and three prior
recordings**, all with eight ordered captures inside 50 m. Full replay
timestamps and GPX consistency passed; the largest GPX/replay position
rounding difference was below **0.0001 m**. Archive resampling at departure
and every capture had zero discrepancy for the recorded wind, waves and
current. The tables contain **88 capture/segment rows**, and the combined
telemetry contains **20,437 recorded frames** across the eleven inputs.
See [comparison.json](comparison.json), [leg metrics](segment_metrics.csv),
[waypoint metrics](waypoint_metrics.csv), [run metrics](run_metrics.csv),
[full telemetry](telemetry.csv) and the `audit-output hashes`.

## Reproduction

Run from the V2 root after loading the existing acados Python environment:

```sh
# Set MARITIME_ACADOS_ENV to the local acados environment script.
source "$MARITIME_ACADOS_ENV"
export PYTHONPATH="$PWD:${PYTHONPATH:-}" MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
"$MARITIME_ACADOS_PYTHON" -B data/results/otranto_october_refined_tacks/reproduce/launch.py --arm grid_mpc
"$MARITIME_ACADOS_PYTHON" -B data/results/otranto_october_refined_tacks/reproduce/launch.py --arm gamma
"$MARITIME_ACADOS_PYTHON" -B data/results/otranto_october_refined_tacks/reproduce/launch.py --arm diffusion
```

Each launch creates a new isolated acados build directory, records its exact
command and source hashes, and saves the launch-time forecast/runner sources
under that run's `source_snapshot` directory. Existing run outputs are never
overwritten. Use a separate `--study` path for exploratory candidates.
The optional `gamma_original` and `diffusion_original` arms explicitly use
`--no-forecast-tack-refinement --no-forecast-tracking-feedback` for a newly
executed coarse control. Final candidate `gamma` and `diffusion` explicitly
enable both options, `--forecast-tack-step 60` and
`--forecast-refinement-phase return`. The all-phase 60 s controls use
`--forecast-refinement-phase all`. The
`gamma_feedback_120` arm uses both options and `--forecast-tack-step 120`.
`gamma_angles_only` explicitly disables feedback; the first
completed angles-only candidate predates the feedback option and therefore
has no feedback flag in its preserved historical command.

That first candidate was launched under `gamma` and is retained under
`gamma_angles_only`. Its command and execution manifest keep their original
output paths. After moving the completed folder, run
`reproduce/record_relocation.py` to record its canonical location and hashes
without rewriting its historical command or manifest.
The completed feedback candidate launched under `gamma` is likewise
retained under `gamma_feedback_120`, with an independent relocation record.
Its historical command predates the configurable interval and used the
120 s default.
The completed all-phase 60 s candidates are retained as `gamma_all_60` and
`diffusion_all_60`, with relocation records preserving their original
launch commands. Those commands predate the phase option and used its
all-phase behavior.

The selected final forecast arguments are explicit in both launch manifests
and per-run `command.txt` files:

```text
--forecast-tacks --forecast-tack-refinement --forecast-tracking-feedback
--forecast-tack-step 60 --forecast-refinement-phase return
```

The original control can be rerun with `launch.py --arm gamma_original`.
The ablations can be rerun with `--arm gamma_angles_only`,
`--arm gamma_feedback_120`, `--arm gamma_all_60` or
`--arm diffusion_all_60`, using a new `--study` destination. The saved
commands and source snapshots preserve the exact code used for the
completed candidates, including those predating later CLI options.

The original pre-refinement source and preceding comparison are preserved
under `baseline_source`. The previous replays, videos and exact commands
remain in their original study folder. The independent audit includes
those recorded controls alongside any fresh runs present in this folder:

```sh
"$MARITIME_ACADOS_PYTHON" -B data/results/otranto_october_refined_tacks/reproduce/audit.py
"$MARITIME_ACADOS_PYTHON" -B data/results/otranto_october_refined_tacks/reproduce/render_round_trip.py --jobs 3
```

The audit checks full timestamps, initial-state equality, recorded archive
samples, GPX/replay consistency, first sampled 50 m captures and ordered
waypoint completion. It exports `comparison.json`, `waypoint_metrics.csv`,
`segment_metrics.csv`, `run_metrics.csv`, `telemetry.csv` and a compact
`comparison.md`. All recorded navigation fields prefixed `forecast_` or
`tracking_` are retained in telemetry, including feedback diagnostics.
Distances are ground-track sums between each arm's own
capture positions. The original forecast runs and Grid reference are
identified separately from newly executed runs.
The audit also compares each final arm's complete outbound state sequence
with the corresponding original forecast recording to test whether the
outbound gain was preserved.

The renderer saves labeled `navigation.mp4` files and source-hashed decode
receipts, with a shared north-up camera, sparse wind barbs, current arrows,
bow and COG markers, full recorded tracks, and highlighted capture points.
Pass `--arms grid_mpc gamma` to render only a selected subset. Each replay
retains every 5 s update; video frames are sampled for efficient playback.

The final archive/capture audit is run only after every present arm has a
successful finished execution manifest. It includes both the fresh Grid
and coarse forecast controls and their corresponding prior recordings.
