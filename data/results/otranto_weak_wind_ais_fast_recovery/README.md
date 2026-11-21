# Cold-start AIS recovery latency retest

This record describes the original experiment workspace. The [reviewer evidence index](../../../docs/RESULTS.md) lists the released files and verification commands.

This study seeks a complete first AIS recovery decision below 5 s while retaining the clearance and waypoint completion of the [safe slow recovery baseline](../otranto_weak_wind_ais_recovery/README.md). That baseline, its raw recordings and its labeled MP4s remain unchanged. Its first maneuver search took approximately 56–57 s in the full runs; it passed all 32 waypoint captures and the nominal/full uncertainty buffers, with minimum margins beyond the full uncertainty plus 3 m reserve of approximately +15.67 m at 355° and +11.36 m at 5°. The selected C03 implementation completed all four full round trips. Independent audits confirm 32 ordered captures within 50 m, zero buffer violations, all 44,300 complete AIS decisions below 5 s, and no loss of the declared truth-centered full-uncertainty-plus-reserve margin.

The final matrix preserves `otranto_355__classical`, `otranto_355__transformer`, `otranto_005__classical`, and `otranto_005__transformer`. All depart GPX waypoint 001 at 2026-09-30 23:50 UTC, capture 002 → 003 → 004 → 005, and return through 004 → 003 → 002 → 001. The true gate remains 50 m; control intervals remain 5 s and the cap remains 17,280 commands, or 24 h. Initial heading is 45° and flap is −10°.

Wind FROM remains controlled at 355° or 5°. Native archived speed, current and waves are unchanged, as are the regional archive crop (39.5–41.0° N, 18.0–19.5° E), GEBCO chart and 4,715/115/115 m local-map span/resolution/obstacle buffer. The study-local provider applies the same wind rotation to plant observations, future tacking/holding forecasts and replay barbs. Archive array fingerprints and matched command checks are recorded before experiments.

Both guidance arms retain Gamma + Direct acados NMPC, drift limiting, forecast hold tacks, original outbound forecasts, refined/feedback return forecasts with 60 s steps and a 48 min horizon, half-radius nominal gate aim in the 60° upwind continuation cone, and 60 s tack commitment. The Transformer arm alone adds the existing frozen `temporal_hold.pt` checkpoint and physical release confirmation. No model is retrained.

The same three-contact sharp mixed AIS fixture begins 40 s after departure. Reports have 10 s delay and are dropped during encounter-relative intervals 85–105 s and 115–135 s. The unchanged safety settings are the nominal 25 m buffer, full age/maneuver uncertainty, 0.026 m/s² acceleration bound, 3 m steering reserve and 35 s preview age budget, with time-aware AIS and advisory COLREG arbitration. Planning uses received reports and the vessel/environment model; hidden truth is used only by the independent auditor.

Each simulation runs in a fresh subprocess with an isolated acados build directory and one numerical thread. The launcher performs no solver warm-up. It records the exact command, source/input hashes, ROOT-relative source snapshot and matched safe-baseline replay/execution hashes. Existing outputs are never overwritten. A non-default cap is labeled an engineering pilot.

Cold-decision measurement must include lazy kernel/solver setup, candidate generation, optimization and final dense validation inside the complete AIS decision timer. Warmed results, scenario-specific cached controls, failed or timed-out calls cannot be silently excluded. Fresh-process latency repeats will be measured both in isolation and under the four-worker numerical load used for the final matrix. The actual first calls in fresh 300 s archived pilots remain the end-to-end acceptance test.

The final four full loops began only after explicit source freeze, measured cold decisions below 5 s in all required pilots/repeats, and independent clearance checks. Final acceptance additionally requires every measured AIS decision below 5 s, all eight waypoint captures in order within 50 m, positive nominal/report/truth margins, and a minimum truth-centered full-uncertainty-plus-reserve margin no more than 1 mm below the matched safe slow baseline. This numerical comparison tolerance is not a physical accuracy claim. Every interval of each complete loop has been independently audited. The final videos preserve solver provenance, measured computation time, model/classical/physical-source decisions, signed surge and STERNWAY labels, route gates and separate clearance outcomes.

To review the final protocol without launching:

```sh
# Set MARITIME_ACADOS_ENV to the local acados environment script.
source "$MARITIME_ACADOS_ENV"
"$MARITIME_ACADOS_PYTHON" -B data/results/otranto_weak_wind_ais_fast_recovery/reproduce/launch.py --study /tmp/otranto_fast_ais_dryrun --dry-run
```

An agreed candidate can be tested over the first 300 s in a separate fresh folder:

```sh
# Set MARITIME_ACADOS_ENV to the local acados environment script.
source "$MARITIME_ACADOS_ENV"
"$MARITIME_ACADOS_PYTHON" -B data/results/otranto_weak_wind_ais_fast_recovery/reproduce/launch.py --study /tmp/otranto_fast_ais_pilot_UNIQUE --steps 60 --jobs 4
```

Renderer labels identify native feedback plus 15-iteration Ipopt polish in the shared AIS recovery layer. The frozen plant permits modeled astern motion; glyph heading, measured COG and signed water-relative surge remain separate. A signed-surge label alone does not establish physical field validation or a real-time deadline guarantee.

The `C01 actual-pilot acceptance receipt` retains an unsuccessful development gate. All four fresh-process 300 s pilots kept complete AIS decisions below 5 s (3.999–4.048 s) and had no clearance violations. At 5° the independent full-plus-reserve margin improved to 11.528540 m. At 355° it was 15.660713 m, which is 6.222 mm below the matched safe baseline and fails the declared 1 mm no-loss tolerance. Full missions were held for C01; all failure evidence remains preserved.

Selected C03 uses batched native-model feedback candidates, a generic local search and bounded 15-iteration Ipopt best-iterate polish, followed by dense validation. It changes the shared AIS recovery search, while the Direct acados path tracker and frozen Transformer remain unchanged. Setup, search and validation remain inside the measured decision; there is no scenario-control or hidden-truth cache.

The `C03 actual four-arm pilot receipt` passes every declared gate: complete first/max decisions 4.291486–4.308455 s, zero uncertainty/reserve violations, and independent full-plus-reserve minima 15.689668898 m at 355° and 11.529076171 m at 5°. These improve on the safe slow margins by 22.734 mm and 172.165 mm. The 18 formal fresh-process cold calls also passed, including three four-worker waves; [selection provenance](selection.json) binds the chosen candidate and retained rejected trials. The complete-loop results below additionally check every subsequent interval.

## Complete-loop results

| Wind FROM | Guidance | Captures | Simulated trip | Maximum whole AIS choose | Minimum truth full uncertainty + 3 m reserve |
|---|---|---:|---:|---:|---:|
| 355° | Classical | 8/8 | 17:27:50 | 4.320212 s | +15.689668898 m |
| 355° | Transformer + confirmation | 8/8 | 17:26:35 | 4.297536 s | +15.689668898 m |
| 5° | Classical | 8/8 | 13:19:35 | 4.299709 s | +11.529076171 m |
| 5° | Transformer + confirmation | 8/8 | 13:17:40 | 4.296226 s | +11.529076171 m |

All 79 frozen source/input hashes match the final current files and ROOT-relative snapshots, and matched slow-baseline replay/execution hashes remain unchanged. The full audit covers 44,300 commands at 0.05 s reconstruction resolution. The initial maneuver remains the maximum measured decision in each run. The runtime gain is in the shared AIS recovery search; nominal Direct acados tracking and the frozen Transformer are unchanged. Changes in simulated trip time follow the changed control trajectories and subsequent tack plans.

The no-loss criterion refers specifically to the minimum truth-centered full uncertainty plus 3 m reserve. At 5°, the separate nominal-25 m minimum margin is 31.589583 m versus 31.668204 m in the slow baseline, a 0.078620 m reduction while remaining strongly positive. The table does not imply that every separation metric increased. The guarded-margin gains are +22.734 mm at 355° and +172.165 mm at 5°.

These are corrective retests of identified deterministic cases, with three-contact synthetic traffic on archived environmental conditions. They demonstrate the recorded latency and clearance behavior on the recorded 18-logical-CPU machine. They are not a universal worst-case execution-time or physical field-validation guarantee. Failed C01 evidence remains preserved alongside the selected C03 evidence.

[Independent full-route audit](independent_traffic_sweep.json), `per-interval histories`, `raw completion/source checks`, `executed commands`, `slow-baseline preservation`, and `actual software environment` are available. Per-arm directories preserve every five-second replay, GPX track, command, log and complete source snapshot.

The [final acceptance receipt](acceptance_verification.json) passes all four cases. Primary data include [the complete audit](comparison.json), [run metrics](run_metrics.csv), [waypoint times/ranges](waypoint_metrics.csv), [leg times and sailed distances](segment_metrics.csv), [paired waypoint differences](paired_waypoint_deltas.csv), `all 44,304 telemetry states`, `telemetry by arm`, [arbitration transitions](arbitration_transitions.csv), [fresh model decisions](model_decision_events.csv), and [comparison with the safe slow controller](baseline_comparison.csv). There are 211 fresh model proposal events at 355° and 161 at 5°; no learned control was applied with active COLREG decisions. The weather model and shared AIS recovery remain distinct components.

Reproducibility evidence includes [all 79 source/condition checks](independent_conditions_check.json), `C03 formal cold repetitions`, `retained C01 repetitions`, `C03 actual-pilot pairing`, [selection and rejected-trial provenance](selection.json), `native archive array fingerprints`, `final regression checks`, and `the numeric/protocol artifact hashes`. The preserved slow-baseline 57-entry manifest is copied byte-for-byte under `baseline_preservation/`; its source files and raw replay/execution/GPX/video bindings remain valid.

The `playback index` links all four labeled MP4s, full replays, GPX tracks and video verification receipts. Each MP4 contains 460 decoded frames sampled from the complete five-second recording, retaining the departure encounter, COLREG updates, source transitions and waypoint captures. Labels show the actual hybrid AIS backend, whole-decision first/maximum wall times, nominal acados tracking, classical or Transformer guidance, physical release guard and separately audited buffer margins. Bow heading, measured COG and signed water-relative surge remain distinct; STERNWAY receives an AIS-recovery suffix only when that controller is active. Video receipts bind the raw replay, frozen renderer, sampled source indices and MP4 hash.
