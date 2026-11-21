# Predictive fallback corrective retest

This record describes the original experiment workspace. The [reviewer evidence index](../../../docs/RESULTS.md) lists the released files and verification commands.

This study tests a receding-horizon AIS fallback and strict full-clearance action acceptance against the preserved [new-traffic study](../otranto_fast_recovery_new_traffic/README.md). It is a corrective engineering retest on the same encounters, not a new unseen-geometry validation. The controller change is enabled by `--ais-predictive-fallback`; candidate 3 was frozen separately before the full launches.

The [protocol](protocol.json), `fixture eligibility record`, both parameter JSON files and both independent truth traces are byte-identical to their predecessor. The protocol retains its original definition identifier intentionally. `Retest configuration` supplies this study's identity and changed controller option. `Baseline provenance` binds the predecessor's 1,278-file numerical/source manifest and all eight delivered replay/track/video/execution bindings. The predecessor is read-only.

There are eight canonical arms: `crossing_fan` and `opposed_crossing`, wind FROM 355° and 5°, and classical or frozen Transformer-plus-confirmation weather guidance. Each full attempt retains departure 2026-09-30 23:50 UTC, the archived wind speed/current/waves, forecast hold/tack plans, true 50 m waypoint gates, native Fossen dynamics, five-second updates and a 24 h cap. The route order is 002 → 003 → 004 → 005 outbound, then 004 → 003 → 002 → 001 return.

Traffic reports retain the same 20 s delay and encounter-relative emission losses [85,125) s and [145,185) s, with a 65 s preview-age budget. Actual applied checks age reports without capping. The nominal 25 m gap, initial 5 m uncertainty, 0.05 m/s linear growth, 0.026 m/s² acceleration bound and additional 3 m reserve are unchanged. Runtime reads analytical fixture parameters through the unchanged study-local native AIS subclass; the independently generated truth CSV is used only for audit and display.

Every process captures the route, chart, configuration, checkpoint, protocol/retest inputs, traffic parameters/truth and all world Python sources. It gets an isolated acados build directory and one numerical thread. Full arms run in two explicit four-worker waves. Existing run folders are rejected. Every outcome is retained, including requirement failures and native early stops; there is no deadline-based timeout or outcome-based fixture adjustment.

Two timing records have separate scopes. `AIS decision timing` covers the complete AIS planner choice. `AIS applied decision timing` starts at the same instant and continues through the final full-clearance action acceptance or stop, including any action guard/fallback work. Both include unapplied terminal attempts. The latter is the complete decision against the 5 s requirement; constructor-inclusive cold probes are a distinct measurement. A native `no_full_clearance_action` outcome means no tested acceptable action was found for the next interval. It is a failed loop outcome and does not establish physical safe waiting or collision avoidance after simulation termination.

All eight authorized full attempts have now closed natively with unchanged 84-input source snapshots. Each ended with `no_full_clearance_action` before the first waypoint capture: all crossing-fan arms after 175 s, opposed-crossing 355° arms after 105 s, and opposed-crossing 5° arms after 110 s. The 24 h cap was available in every arm; native strict-clearance termination, rather than manual censoring, ended each run. There are 226 applied commands, eight additional terminal attempts and zero of 64 requested captures. The largest complete accept/stop time is 2.442796 s. Engineering pilots and source freezes remain distinguishable from these final full attempts. The final backup catalog discards non-full-clearance choices before ranking; this strict catalog filter was added after the C3 pilots, which already rejected such a selected action at their final guard. Pilot runtime/source records therefore remain separate from the final frozen actor evidence. The independent 600 s checkpoint audits live first-120-command prefixes, or the entire shorter closed record with insufficient exposure explicitly marked. Final audits check full report-centered and truth-centered uncertainty clearance, every timing attempt, physical waypoint captures and ordered completion separately. An early stop does not validate the missing portion of the delay/dropout protocol.

The independent native 0.05 s reconstruction verifies **positive full uncertainty plus 3 m reserve margins for all applied intervals in all eight arms**. Classical and Transformer tracks match exactly within each encounter/bearing. The Transformer makes no learned weather decisions while AIS has priority, so these outcomes show no added value from that model during these encounters. They do not evaluate its later weather guidance because no loop reaches that stage.

| Encounter | Wind FROM | Recorded duration, each guidance | Truth full + reserve minimum [m] | Report full + reserve minimum [m] | Max held age |
|---|---:|---:|---:|---:|---:|
| crossing_fan | 355° | 175 s | +10.126 | +1.634 | 55 s |
| crossing_fan | 5° | 175 s | +6.795 | +3.175 | 55 s |
| opposed_crossing | 355° | 105 s | +16.325 | +16.325 | 25 s |
| opposed_crossing | 5° | 110 s | +0.456 | +0.456 | 25 s |

These are clearance results for recorded prefixes, **not successful full-route tests**. All eight waypoint orders are empty prefixes and all eight loops fail completion. Crossing-fan records include 30 s of the first delayed-drop delivery effect and reach 55 s report age. Opposed-crossing records stop before any dropped-report delivery effect and reach 25 s age. No case reaches the full 65 s age peak, second dropout delivery effect or 600 s checkpoint. A strict stop leaves the last recorded fix outside the audited envelopes; it does not supply a physically executable safe waiting maneuver after termination or prove that no feasible navigation policy exists.

The final independent 24 fresh-process trials have a maximum constructor plus complete choose time of 2.328789042 s, with empty initial caches and stable source hashes. The actual full-attempt maximum, including the final acceptance/stop work, is 2.442796 s. All 234 complete attempts pass 5 s in this computational experiment. These measurements exclude real sensing/actuation latency and do not imply a universal real-time guarantee. The forecast tail used for ranking is not certified as a full-horizon clear trajectory; each applied command is checked anew against uncapped report aging.

[Independent sweeps](independent_traffic_sweep.json), [acceptance](independent_acceptance_check.json), [fault exposure](fault_exposure_check.json), [classical/Transformer pairing](independent_pairing_check.json), `complete choose timings` and `accept/stop timings` retain the evidence. The primary [comparison](comparison.json), [run table](run_metrics.csv), [waypoint table](waypoint_metrics.csv), [segment table](segment_metrics.csv) and `per-arm telemetry` provide machine-readable paper data. Empty waypoint tables are retained as failed mission outcomes.

Reproduction uses the simulator environment:

```sh
cd .
# Set MARITIME_ACADOS_ENV to the local acados environment script.
source "$MARITIME_ACADOS_ENV"
export PYTHONPATH="$PWD:$PYTHONPATH" MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
STUDY=data/results/otranto_predictive_fallback_retest
"$MARITIME_ACADOS_PYTHON" -B "$STUDY/reproduce/launch.py" --dry-run > "$STUDY/dry_run_commands.txt"
# Run only after the candidate freeze and task authorization:
"$MARITIME_ACADOS_PYTHON" -B "$STUDY/reproduce/launch.py" --jobs 4
```

Filters are `--encounter crossing_fan/opposed_crossing`, `--wind-from 355/5` and `--method classical/transformer`. A pilot uses a new `--study` output folder and an explicit `--steps` cap. The independent audit tools and primary route/forcing audits consume each run's own metadata and immutable inputs. Labeled renderer output distinguishes choose and accept/stop cost, applied and unapplied counts, true 50 m capture gates, actual fault exposure, full clearance and native termination. Short failures repeat only recorded states for readable playback; no motion is invented. Boat heading, signed surge and measured COG are displayed separately, with wind FROM and current TO. The `playback index` links all eight labeled MP4s, raw replays, GPX tracks and verification receipts. Every clip preserves all recorded states, repeats source indices to 168 frames (14 s at 12 fps), and includes a final two-second state hold. All eight share one fixed north-up camera for the full GPX route and opening encounter; exact bounds, boat scale, source indices and both timing scopes are in each video receipt. The `closed-launch check` and `complete launch receipts` retain all outcomes; `engineering pilots` retain the earlier corrective candidates separately.

The `archive array fingerprints`, `software runtime`, `baseline-preservation receipt` and `final numerical/source integrity manifest` complete reproducibility. The previous delivered paper and source package remain under `previous_paper/`.
