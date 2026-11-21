# Fast AIS recovery with new encounters and longer reporting faults

This record describes the original experiment workspace. The [reviewer evidence index](../../../docs/RESULTS.md) lists the released files and verification commands.

This validation extends the [preserved C03 experiment](../otranto_weak_wind_ais_fast_recovery/README.md) to two traffic geometries fixed before their first navigation outcomes. It retains both controlled wind bearings (355° and 5°), classical and frozen Transformer-plus-confirmation weather guidance, the original 50 m GPX circles, and the same archived marine conditions. All eight geometry × bearing × guidance arms are retained, including deadline failures, partial waypoint prefixes and documented safety stops.

The [protocol](protocol.json) and [fixture eligibility check](fixture_eligibility.json) bind the definitions of `crossing_fan` and `opposed_crossing`. Each has three ships with new starting geometry, courses/speeds, dimensions and velocity ramps. Eligibility requires initially clear full uncertainty envelopes, bounded target acceleration, and a received-report constant-velocity threat to a passive-current vessel. Eligibility is not evidence that a feasible control sequence exists. No fixture is retimed, repositioned or excluded after observing the controller.

All departures remain 2026-09-30 23:50 UTC, with the encounter reference 40 s later. The reporting interval is 5 s, delivery delay is **20 s**, and encounter-relative emission drops span **[85,125) s** and **[145,185) s**. The last pre-drop emissions are held until delivery resumes, producing a continuous report-age supremum of **65 s**. The planning age budget is therefore 65 s, while the applied-step checks retain uncapped actual report aging. The nominal 25 m gap, initial 5 m allowance, 0.05 m/s linear growth, 0.026 m/s² acceleration bound and extra 3 m reserve are unchanged.

The simulator uses only each fixture's analytical parameter JSON through a study-local subclass of the native synthetic AIS provider. This preserves native motion integration and report generation/delivery. The navigation controller receives delayed observations; it does not load the independently exported truth CSV. The JSON, truth trace, study wrapper/helper/protocol, checkpoint, chart, route, configuration and all world Python sources are hashed and copied into every run's ROOT-relative source snapshot. The `traffic-helper equivalence check` verifies the injection mechanism against the original native provider without running a navigation controller.

The route, archive crop, native Fossen plant, actuator limits and forecast tack/hold logic are unchanged. Initial bow is 45°, speed is zero and flap is −10°. Gamma + Direct acados NMPC remains the nominal tracker. The Transformer arm alone loads the existing frozen hold checkpoint and physical release confirmation; no training occurs. Every numerical process has its own acados build directory and one numerical thread.

The 5 s timing check counts **every complete AIS decision attempt** from the try/finally timing JSON in `run.log`. A rejected final command or safety stop can have a slow search even though it produces no replay frame; that cost is retained and included in the maximum. Applied-frame timings are cross-checked against the attempt log. Constructor-inclusive fresh-process timings, when collected, are reported separately. No 5 s timeout or post-hoc omission is used to make a runtime pass.

## Initial baseline screen

The first authorized screen used the unchanged C03 control code with attempt-level instrumentation, four fresh classical processes, and a one-command cap. All four outcomes and frozen 81-input snapshots are retained under [pilot_studies/c03_initial_step](pilot_studies/c03_initial_step/). This is a startup diagnostic, not a full encounter, clearance or waypoint-completion result.

| New encounter | Wind FROM | First complete AIS attempt | 5 s budget | First applied mode |
|---|---:|---:|---|---|
| crossing_fan | 355° | 70.065 s | FAIL | ordinary refined policy; negative full preview |
| crossing_fan | 5° | 70.175 s | FAIL | ordinary refined policy; negative full preview |
| opposed_crossing | 355° | 62.020 s | FAIL | minimum-risk fallback |
| opposed_crossing | 5° | 61.951 s | FAIL | minimum-risk fallback |

The fast seed search had negative predicted full-uncertainty margins in every case, and all four calls entered the retained slow recovery branch. Crossing returned an ordinary refined policy with a negative full-uncertainty preview, not a certified full-clearance recovery; opposed crossing used minimum-risk fallback. The one applied command in each startup screen does not establish a full-route outcome. The [first-decision summary](pilot_studies/c03_initial_step/first_decision_summary.json) preserves exact timings, search profiles, received-state inputs and raw replay/execution hashes. Subsequent tests require a separately recorded source freeze and authorization; these first-step outcomes are never overwritten.

## Computation correction and full-loop attempts

The post-diagnosis `candidate02_batched` revision is a corrective engineering retest, not a new unseen-geometry validation. It batches the existing coarse policy rollouts and limits nonlinear refinements to 15 iterations with native best-iterate checks. If no full-clearance recovery sequence is verified, it returns no certificate and proceeds to the separately screened one-step last resort; the former three-solve long recovery is not invoked. A positive screened sequence no longer needs an additional 6 m threshold beyond the configured full uncertainty and reserve. Physics, uncertainty, ship geometry, faults, waypoints and checkpoint remain unchanged.

The [selected source and timing receipt](selection.json) binds the corrected code and all retained initial failures. Its independent 24 fresh-process cold trials passed the computation check, with a maximum constructor plus complete choose time of 4.224015417 s. Every initial call returned no verified 300 s recovery sequence. This computation result does not establish future clearance or mission completion.

All eight full-loop attempts were then launched as two explicit four-worker waves. Each has the original 24 h cap, and failure of a buffer, timing or waypoint check does not censor the other virtual outcomes. The independent live 600 s checkpoint uses the first 120 applied commands when available; a run closing earlier is audited in full and marked with insufficient exposure. Final audits cover every applied command and every complete decision attempt, including unapplied terminal attempts.

## Full-attempt outcomes

All eight planned full-loop attempts have closed with unchanged 81-input snapshots. The two `opposed_crossing` 355° runs completed eight ordered captures. The other six simulations terminated natively before the next predicted hull-contact step, with no waypoint captures; this is a failed loop outcome, not evidence of an actual recorded hull collision or a physically stationary vessel. No case was manually ended or omitted for failing a requirement.

| Encounter | Wind FROM | Guidance | Captures | Native outcome | Simulated time | Maximum complete AIS attempt | Max held age |
|---|---:|---|---:|---|---:|---:|---|
| crossing_fan | 5° | Classical | 0/8 | Terminated: predicted next-step hull contact | 0:02:15 | 4.206485s | 25 s (peak untested) |
| crossing_fan | 5° | Transformer + confirmation | 0/8 | Terminated: predicted next-step hull contact | 0:02:15 | 4.202146s | 25 s (peak untested) |
| crossing_fan | 355° | Classical | 0/8 | Terminated: predicted next-step hull contact | 0:02:15 | 4.209884s | 25 s (peak untested) |
| crossing_fan | 355° | Transformer + confirmation | 0/8 | Terminated: predicted next-step hull contact | 0:02:15 | 4.208577s | 25 s (peak untested) |
| opposed_crossing | 5° | Classical | 0/8 | Terminated: predicted next-step hull contact | 0:02:00 | 4.253604s | 25 s (peak untested) |
| opposed_crossing | 5° | Transformer + confirmation | 0/8 | Terminated: predicted next-step hull contact | 0:02:00 | 4.244857s | 25 s (peak untested) |
| opposed_crossing | 355° | Classical | 8/8 | Completed | 17:33:20 | 4.266312s | 65 s |
| opposed_crossing | 355° | Transformer + confirmation | 8/8 | Completed | 17:24:10 | 4.305253s | 65 s |

All 25,332 complete AIS attempts stayed below 5 s, including the six final rejected attempts absent from applied replay frames. The matrix contains 25,326 applied commands and 16 of 64 requested captures. Independent audits distinguish computation, clearance and route outcomes. **All eight full uncertainty-plus-reserve outcomes fail**: the six short records already contain buffer incursions, and the surviving 355° pair's 600 s prefixes also contain an uncertainty incursion. Later route completion does not repair those earlier violations.

The six short cases reached only 25 s held-report age and stopped before the first dropped report's delayed delivery effect. They test the new geometry and 20 s delay, with a conservative 65 s planning budget, but do not validate actual 65 s dropout exposure. The two long 355° records did experience the 65 s age peak. The 600 s exposure requirement is therefore unmet in six cases rather than silently treated as passed.

The `raw launch/timing/source check` and `complete launch records` preserve every process. Per-case native 0.05 s safety reconstructions and primary route/forcing audits provide the final numerical margins separately. The [original C03 results](../otranto_weak_wind_ais_fast_recovery/README.md) remain the earlier successful fixture evidence; the new tests do not establish physical infeasibility of these encounters or the full-route behavior of an untested original slow-fallback run.

## Independently reconstructed clearance

All eight final native 0.05 s sweeps are complete. Clearance minima are identical between guidance treatments within each geometry/bearing, because the traffic encounter precedes their weather-guidance divergence. Margins are metres outside each indicated envelope; negative values are incursions. Hull margin omits the nominal 25 m gap and uncertainty. Full columns subtract hull radii, the 25 m gap, full report-age uncertainty and the additional 3 m reserve. Minima across different columns need not coincide.

| Encounter | Wind FROM | Truth hull gap | Truth nominal gap margin | Truth full + reserve | Report full + reserve | Max held age |
|---|---:|---:|---:|---:|---:|---:|
| crossing_fan | 355° | +10.988 | -14.012 | -30.137 | -31.086 | 25s |
| crossing_fan | 5° | +6.079 | -18.921 | -35.046 | -35.918 | 25s |
| opposed_crossing | 355° | +6.270 | -18.730 | -63.089 | -63.089 | 65s |
| opposed_crossing | 5° | +1.612 | -23.388 | -39.513 | -39.513 | 25s |

No applied trajectory intersects a modeled hull, but all eight enter both the nominal and full uncertainty envelopes. The smallest actual hull gap is only 1.612 m. Native termination for predicted next-step hull contact preserves the recorded stopping point; it is not a demonstrated physical braking or safe waiting policy. Completing the two 355° loops after an earlier buffer incursion does not make them successful safety tests.

[Final independent sweeps](independent_traffic_sweep.json), [fault exposure](fault_exposure_check.json), [paired traffic-prefix/source checks](independent_pairing_check.json), [the deliberately failing combined acceptance receipt](independent_acceptance_check.json), `all decision-attempt timings`, `per-interval clearance histories`, and [the frozen computation-only cold gate](cold_benchmarks/candidate02_batched/benchmark.json) retain the evidence. The primary [comparison](comparison.json), [run metrics](run_metrics.csv), [waypoint times and center ranges](waypoint_metrics.csv), [segment metrics](segment_metrics.csv) and `complete telemetry` provide the paper/table data.

## Reproduction and final artifacts

`Dry-run commands` and `native archive array fingerprints` record the matched environmental inputs. The study-local launcher accepts `--encounter crossing_fan` or `--encounter opposed_crossing`, `--wind-from 355/5`, and `--method classical/transformer`; absent filters select all eight arms. `--steps` sets a clearly labeled engineering-pilot cap. A full loop uses the unchanged 17,280-command/24 h cap. Existing output directories are rejected.

The `playback index` links all eight labeled MP4s, full raw replays and GPX tracks. Six short failures share a tighter fixed camera showing the full GPX route and nearby opening traffic; the long pair shares its full-track camera. Exact camera bounds and enlarged boat scales are recorded per video. Each short clip repeats recorded indices to 144 base frames plus 24 final-state-hold frames (14 s at 12 fps), preserving every actual state without inventing motion. Labels distinguish predicted next-step termination from actual contact, actual held-age exposure from the configured 65 s budget, and timing from safety/waypoint outcomes. Every 5 s state and all original decision parameters remain in the raw replay. Safety, deadline and waypoint completion are separate outcomes. Simulation feasibility does not validate reverse-flow hydrodynamics, physical sensing/actuation latency or universal real-time/COLREG behavior.

The `final software environment`, `C03 baseline-preservation check`, and `numeric/protocol integrity manifest` complete the reproducibility records. The previous delivered paper PDF and source ZIP remain unchanged under `previous_paper/`.
