# Weak-wind Otranto round trips with AIS and COLREG

This record describes the original experiment workspace. The [reviewer evidence index](../../../docs/RESULTS.md) lists the released files and verification commands.

This study compares classical hold/release guidance with the frozen temporal Transformer and physical release confirmation at controlled wind FROM bearings of 355° and 5°. Each mission starts at Otranto waypoint 001 on 30 September 2026 at 23:50 UTC, follows 002 → 003 → 004 → 005, and returns through 004 → 003 → 002 → 001. Every capture uses the original GPX center and a 50 m radius. The cap is 24 simulated hours: 17,280 five-second commands.

The four final run directories are `runs/otranto_355__classical`, `runs/otranto_355__transformer`, `runs/otranto_005__classical`, and `runs/otranto_005__transformer`. All four completed the eight requested waypoint captures in order within 50 m. All four failed the nominal 25 m AIS buffer and full uncertainty clearance during the departure encounter. Route completion therefore does not establish safe traffic avoidance in these weak-wind cases.

Both methods use Gamma guidance with Direct acados NMPC and the Fossen plant. Both enable archived forecast tacks, drift limiting, and forecast tacks during holding. Sailing uses the original outbound forecast and the improved return forecast: refined modes, tracking feedback, a 60 s forecast step, a 48 min horizon, a nominal aim displaced by half of the capture radius when the continuation bearing lies within 60° of wind FROM, and a 60 s minimum tack commitment. The true waypoint gate remains 50 m. The non-forecast `--commit-recovery-tacks` option is omitted because forecast sailing is active.

The sole within-bearing guidance treatment adds the existing `temporal_hold.pt` checkpoint and `--confirm-temporal-release`; no model is trained in this study. Both methods use the same shared physical recovery and safety checks. The wind-bearing treatment rotates interpolated archived wind velocity while preserving its exact native speed at every queried position and time. Currents and waves are unmodified archive samples. The study-local `DirectedWeather.wind` override also covers inherited `sample` calls, so the plant, future sailing/hold forecasts, temporal observations, and replay wind barbs receive the same controlled bearing.

The native marine archive crop is latitude 39.5–41.0° N and longitude 18.0–19.5° E, with the same GEBCO chart and local-map settings as the October comparison: 4,715 m span, 115 m resolution, and 115 m obstacle buffer. Initial bow heading is 45° and flap is −10°. `protocol_preflight.json` records the four parsed commands and controlled future samples. `archive_coverage_check.json` independently verifies coverage through the full 24 h cap plus the 3 h hold-forecast horizon; no archive extrapolation is required.

All runs use the same three-contact sharp-maneuver mixed AIS fixture as the preceding October study, translated in time to an encounter at 23:50:40 UTC, 40 s after departure. Reports arrive with 10 s delay and are dropped during encounter-relative intervals 85–105 s and 115–135 s. Avoidance uses time-aware AIS, COLREG arbitration, a 25 m nominal clearance buffer, the full age/maneuver uncertainty allowance with acceleration bound 0.026 m/s², a further 3 m steering reserve, and a 35 s report-age budget in preview. These are synthetic AIS contacts on real archived environmental conditions. The fixture injects no new contacts later in the loop; clearance auditing continues for the full recording.

Each final run receives an isolated acados build directory and a separate subprocess. The four subprocesses run concurrently with numerical thread counts fixed at one per run. `experiment.json` records the launch command, source hashes, frozen checkpoint hash, and build directory. `source_snapshot/` preserves all Python files under `world/`, the configuration, launch scripts, checkpoint, route, chart, and traffic fixture. `execution.json` records the exit code, elapsed wall time, and any input changes detected after the process finishes. Existing recordings are never overwritten.

To reproduce the protocol without launching simulations:

```sh
# Set MARITIME_ACADOS_ENV to the local acados environment script.
source "$MARITIME_ACADOS_ENV"
"$MARITIME_ACADOS_PYTHON" -B data/results/otranto_weak_wind_tacks_ais_colreg/reproduce/launch.py --study /tmp/otranto_weak_wind_ais_dryrun --dry-run
```

To run all four cases into a fresh study folder:

```sh
# Set MARITIME_ACADOS_ENV to the local acados environment script.
source "$MARITIME_ACADOS_ENV"
"$MARITIME_ACADOS_PYTHON" -B data/results/otranto_weak_wind_tacks_ais_colreg/reproduce/launch.py --study /tmp/otranto_weak_wind_ais_reproduction --jobs 4
```

The replay preserves every five-second state, observation, actuator command and guidance/safety parameter in `navigation.jsonl.gz`; `track.gpx` and `run.log` are saved alongside it. Rendering follows completion and clearance audits. The paper comparison must distinguish the Transformer’s raw proposals, physically confirmed overrides, and actually applied control, including AIS/COLREG deferrals. A successful run and a missed waypoint must both remain represented in the results.

## Results and limits

The independent 0.05 s swept-interval audit covers every applied command through the final capture. It reconstructs the Fossen substeps and evaluates closed-form closest approach on each short chord; this is numerical trajectory verification, rather than a continuous-ODE proof. All 32 captures passed the original 50 m gate and waypoint-order checks. All four processes exited successfully, and all 78 source/input hashes match each other, the live files, and the preserved per-run snapshots.

| Wind FROM | Guidance | Captures | Round-trip time | Minimum nominal 25 m margin | Minimum full uncertainty + 3 m reserve margin |
|---|---|---:|---:|---:|---:|
| 355° | Classical | 8/8 | 17:44:00 | -14.805 m | -35.271 m |
| 355° | Transformer + confirmation | 8/8 | 17:38:30 | -14.805 m | -35.271 m |
| 5° | Classical | 8/8 | 13:15:35 | -23.256 m | -42.243 m |
| 5° | Transformer + confirmation | 8/8 | 13:19:55 | -23.256 m | -42.243 m |

Negative margins mean buffer intrusion. Minimum gaps to the truth-centered hull circles remained positive: 355°: 10.195 m, 5°: 1.744 m. The same minima occur in both guidance arms within each wind case, because their departure-encounter trajectories and shared AIS arbitration are identical. Neither uncertainty-envelope validity nor waypoint completion excuses the clearance failures.

At 355°, the Transformer-plus-confirmation treatment finished 5 min 30 s earlier; at 5°, it finished 4 min 20 s later. The first transition from hold to recovery occurs 5 min later with the treatment at both bearings. Small changes in hold duration, position and subsequent tacks produce different downstream waypoint times, so the two cases do not establish a consistent timing advantage. The neural model and physical confirmation form one treatment; these results do not isolate the network alone.

The avoidance system entered minimum-risk fallback during the weak-wind encounter and failed to maintain the configured clearance. The independent truth stayed inside the assumed report envelopes, and reconstructed safety margins match the native calculations. These checks distinguish an executed avoidance failure from missing AIS data or an incorrectly drawn buffer. Global unavoidability is unproven; the study makes no claim that every physically feasible control sequence would fail. No safety parameters, traffic trajectories, departure times, or environmental fields were retuned after observing the failure.

The ships are time-shifted synthetic maneuvers on the archived forcing. This is a four-run, two-bearing engineering comparison, with correlated five-second samples. It is not a statistical safety validation or evidence that the weather Transformer has learned AIS prediction or COLREG rules.

The primary audit resampled all 44,620 saved environmental/timestamp frames from the native archives and checked all 32 captures. The model produced 214 fresh proposal events at 355° and 162 at 5°; repeated five-second uses of cached telemetry are counted separately. Each Transformer run applied 60 physical-release override updates and 60 hold updates that differed from the classical rule evaluated at the same state. Neither run applied a learned decision while COLREG was active. These counts describe weather guidance composed with a shared safety layer, rather than a learned AIS/COLREG controller.

Comparison data: [machine-readable audit](comparison.json), [run table](run_metrics.csv), [waypoint capture times/ranges](waypoint_metrics.csv), [consecutive-leg times and distances](segment_metrics.csv), [paired waypoint differences](paired_waypoint_deltas.csv), `all per-update telemetry`, `telemetry by arm`, [arbitration transitions](arbitration_transitions.csv), and [fresh model decisions](model_decision_events.csv). Each arm retains its complete five-second replay, GPX track, log and actual launch command under `runs/`.

Verification: [independent full-route clearance audit](independent_traffic_sweep.json), `independent interval histories`, `archive coverage`, `archive array fingerprints`, `protocol preflight`, `barb interpolation equivalence`, `dry-run commands`, `executed command manifests`, `launch/source verification`, `regression checks`, and `audited data hashes`.

The `playback index` links all four labeled MP4s, complete replays and GPX tracks. Every clip contains 460 decoded frames sampled from its complete replay; waypoint captures, the departure encounter, and guidance transitions are retained. The labels distinguish route completion from the failed nominal/full-uncertainty clearance checks. The per-run video verification receipts bind the source replay, frozen rendering script, sampled indices, decoded frame count and final MP4 hash.
