# Weak-wind Otranto AIS recovery retest

This record describes the original experiment workspace. The [reviewer evidence index](../../../docs/RESULTS.md) lists the released files and verification commands.

This study retests the shared AIS recovery policy after the preceding weak-wind comparison failed the nominal 25 m and full uncertainty buffers. The preserved reference is [the original weak-wind study](../otranto_weak_wind_tacks_ais_colreg/README.md). Its recordings, tables and videos remain unchanged. All four full reruns completed the eight requested waypoint captures in order within 50 m and passed the independent nominal-buffer and full-uncertainty-plus-reserve clearance checks. The complete audit covers 45,021 applied commands at 0.05 s reconstruction resolution.

The final matrix retains four arms: `otranto_355__classical`, `otranto_355__transformer`, `otranto_005__classical`, and `otranto_005__transformer`. Every mission departs Otranto GPX waypoint 001 at 2026-09-30 23:50 UTC, captures 002 → 003 → 004 → 005, and returns through 004 → 003 → 002 → 001. The true capture radius remains 50 m; control updates remain 5 s and the cap remains 17,280 commands, or 24 h. Initial heading is 45° and flap is −10°.

The controlled wind FROM bearings are 355° and 5°. Native archived wind speed, current and waves are retained, as are the regional archive crop (39.5–41.0° N, 18.0–19.5° E), GEBCO chart, and 4,715/115/115 m local-map span/resolution/obstacle buffer. `DirectedWeather.wind`, its inherited `sample`, and the batched replay barbs apply the same wind rotation to the plant, future sailing and hold forecasts, and temporal observations. No departure time or traffic geometry is changed to improve the outcome.

Both guidance arms retain Gamma + Direct acados NMPC, drift limiting, forecast hold tacks, the original outbound sailing forecast and improved return forecast: refined modes, tracking feedback, a 60 s forecast step, a 48 min horizon, half-radius nominal gate aim within a 60° upwind continuation cone, and 60 s minimum tack commitment. The non-forecast committed-recovery option remains disabled. The Transformer arm alone adds the existing frozen `temporal_hold.pt` checkpoint and physical release confirmation. No model is retrained in this retest.

The identical three-contact sharp mixed AIS fixture begins at 23:50:40 UTC, 40 s after departure. Reports have 10 s delay and are dropped in encounter-relative intervals 85–105 s and 115–135 s. Safety settings remain a nominal 25 m buffer, the full age/maneuver uncertainty allowance, 0.026 m/s² acceleration bound, a further 3 m steering reserve, and a 35 s preview age budget, with time-aware AIS and COLREG arbitration. Any revised recovery selects actions from received AIS information and the vessel/environment model; hidden traffic truth is reserved for independent auditing.

Each run preserves exact commands, source/input hashes, a complete ROOT-relative source snapshot, all logs, the replay and GPX track. It receives a separate acados build directory and one numerical thread; final runs can execute concurrently. The launch metadata also identifies and hashes the matched untouched reference recording. A non-default step cap is explicitly labeled an engineering pilot. Existing output directories are never overwritten.

For a dry run of the final matrix, execute from the V2 workspace:

```sh
# Set MARITIME_ACADOS_ENV to the local acados environment script.
source "$MARITIME_ACADOS_ENV"
"$MARITIME_ACADOS_PYTHON" -B data/results/otranto_weak_wind_ais_recovery/reproduce/launch.py --study /tmp/otranto_ais_recovery_dryrun --dry-run
```

An agreed candidate can be tested over the first 300 s in a fresh, separate engineering-pilot folder:

```sh
# Set MARITIME_ACADOS_ENV to the local acados environment script.
source "$MARITIME_ACADOS_ENV"
"$MARITIME_ACADOS_PYTHON" -B data/results/otranto_weak_wind_ais_recovery/reproduce/launch.py --study /tmp/otranto_ais_recovery_pilot_UNIQUE --method classical --wind-from 5 --steps 60 --jobs 1
```

The final four-arm launch began after the explicit core-source freeze and 118 targeted regression checks. The numerical audit, final recordings and decoded labeled MP4s are preserved below. Route completion and clearance remain separate outcomes; a positive result cannot be inferred from avoiding modeled hull intersection alone.

## Recovery method and engineering checks

The revised shared fallback retains rejected maneuver candidates as seeds for a 300 s actuator-control optimization. CasADi/Ipopt optimizes bounded, slew-limited rudder/flap commands using the Fossen model and received AIS reports with the unchanged full uncertainty and 3 m reserve. Promising controls are densely screened against the chart and traffic model before a sustained sequence is retained. The original acados controller continues nominal sailing; the separate AIS optimizer handles the emergency recovery. No hidden traffic truth enters that decision.

Full-clearance priority precedes the existing advisory starboard preference. The preference uses the first 30 s of verified predicted ground displacement, rather than treating an optimized seed heading as the executed departure course. A separate soft-override field records exceptions while COLREG decisions are active. These checks are not a legal COLREG compliance certificate.

The Fossen recovery can use modeled negative water-relative surge. Replay glyphs continue to show bow heading, while the measured COG arrow, signed surge and explicit STERNWAY label identify astern recovery. The method label names both nominal acados NMPC and emergency CasADi/Ipopt optimization. The C02 pilots spent about 55.5 s in the initial maneuver search; the five-second simulated control interval is not evidence of a five-second real-time optimization deadline.

| Engineering candidate | Wind FROM | Audited commands | Minimum truth full uncertainty + 3 m reserve margin |
|---|---:|---:|---:|
| candidate_01 | 355° | 60 | +15.666935 m |
| candidate_01 | 5° | 60 | +11.363322 m |
| candidate_02 | 355° | 60 | +15.666935 m |
| candidate_02 | 5° | 60 | +11.356911 m |

Both candidate pairs had zero full-buffer/reserve violations under independent 0.05 s native Fossen reconstruction, with no fallback use. C02 additionally fixes recovery after dense candidate rejection and ranks verified safe sequences using realized-course starboard preference. Its five-degree trajectory differs slightly from C01; all pilot inputs and logs remain preserved in separate folders. These short tests do not establish full waypoint completion.

## Complete round-trip results

| Wind FROM | Guidance | Captures | Round-trip time | Minimum nominal 25 m margin | Minimum full uncertainty + 3 m reserve margin |
|---|---|---:|---:|---:|---:|
| 355° | Classical | 8/8 | 18:01:50 | +37.529175 m | +15.666935 m |
| 355° | Transformer + confirmation | 8/8 | 17:33:15 | +37.529175 m | +15.666935 m |
| 5° | Classical | 8/8 | 13:28:55 | +31.668204 m | +11.356911 m |
| 5° | Transformer + confirmation | 8/8 | 13:27:45 | +31.668204 m | +11.356911 m |

All 32 first-entry captures passed the independent waypoint-order checks. The audit reconstructed every applied interval, with zero reported nominal/full/reserve violations and exact native endpoint, COG and COLREG ground-velocity agreement. Minimum clearances occur during the departure encounter and are identical within each guidance pair, because the Transformer defers to the shared AIS controller there. The original failed baselines remain visible in the separate comparison rather than being replaced.

The model-plus-confirmation treatment finished 28 min 35 s earlier than classical guidance at 355° and 1 min 10 s earlier at 5°. Both treatments still release five minutes after the classical rule at their first release opportunity. Later tacks and sampled environmental conditions depend on the resulting position and time, so departure delay is not the same quantity as final route delay. These two deterministic retests do not isolate the network from physical confirmation or establish a general timing advantage.

Compared with the matched failed-clearance baseline, the safe classical loop takes 17 min 50 s longer at 355° and 13 min 20 s longer at 5°. The safe Transformer loop is 5 min 15 s faster at 355° and 7 min 50 s slower at 5°. This is a same-case corrective retest of identified failures, rather than an unseen generalization test. The recovery was developed with these weak-wind cases available.

[Independent full-route results](independent_traffic_sweep.json), `per-interval traffic histories`, `launch/source/baseline verification`, `actual executed commands`, `native archive fingerprints`, and `first-300 s parity to audited C02` are preserved. Every arm retains its full five-second replay, GPX track, command, log and ROOT-relative source snapshot under `runs/`.

The primary audit independently resampled all 45,025 saved environmental/timestamp frames and checked the 32 captures. The Transformer has 213 fresh proposal events at 355° and 164 at 5°, distinct from repeated five-second applications of cached proposals. Each learned run records 60 physical-release override updates and no learned control applied with active COLREG decisions. This remains a weather hold/release model composed with the shared safety controller, rather than a learned AIS/COLREG policy.

Research data: [complete machine-readable audit](comparison.json), [run table](run_metrics.csv), [waypoint capture times/ranges](waypoint_metrics.csv), [consecutive-leg times and distances](segment_metrics.csv), [paired waypoint differences](paired_waypoint_deltas.csv), `all telemetry`, `telemetry by arm`, [arbitration transitions](arbitration_transitions.csv), [fresh model decisions](model_decision_events.csv), [before/after baseline comparison](baseline_comparison.csv), and `baseline-comparison provenance`.

Reproducibility checks: [unchanged protocol/plant/geometry and source conditions](independent_conditions_check.json), `software versions`, `archive coverage`, `regression checks`, and `audit artifact hashes`. Guidance/control parameters attached to a saved state describe the preceding applied interval; recovery horizon, turn and preview margin were evaluated at that interval's start. The independently checked step margin covers the full interval ending at the displayed state.

Development diagnostics are preserved separately: `355° feasibility probe`, `5° feasibility probe`, `355° proxy/native validation`, `5° proxy/native validation`, and `initial COLREG state probe`. Per-call maneuver-search wall times remain in the complete telemetry; the probe timings and software/version records are bound in the audit manifest.

The `playback index` links all four labeled MP4s, full replays, GPX tracks and video verification receipts. Every MP4 contains 460 decoded frames sampled from its complete five-second replay. Labels identify the nominal acados controller, emergency CasADi/Ipopt recovery, classical or Transformer guidance, physical release guard, AIS/report margins and final outcome. Bow heading, measured COG and signed water-relative surge are separate; a STERNWAY label adds an AIS-recovery suffix only when that recovery is active. Each receipt binds the raw replay, frozen rendering script, selected source indices and final MP4 hash.
