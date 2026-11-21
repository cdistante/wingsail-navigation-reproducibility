# Received-AIS safe-departure retest

This record describes the original experiment workspace. The [reviewer evidence index](../../../docs/RESULTS.md) lists the released files and verification commands.

This study adds a received-report departure screen to the [strict predictive-fallback retest](../otranto_predictive_fallback_retest/README.md). The eight arms retain the same two traffic geometries, wind FROM 355°/5°, classical or frozen Transformer-plus-confirmation guidance, Gamma + Direct acados NMPC, forecast hold/tack plans, 50 m waypoint circles, native dynamics and postlaunch full-clearance action checks. Every requested outcome is retained, including a failure to launch or a native stop after launch.

The [original protocol](protocol.json), `traffic eligibility`, parameter JSON and independent truth traces are byte-identical. The protocol's original study identifier is intentionally retained as the frozen encounter definition. `Departure configuration` records the new treatment and its source freeze; `baseline provenance` binds the strict retest's 1,048-file integrity manifest and eight delivered raw/video bindings. Prior studies remain read-only.

Monitoring starts at **2026-09-30 23:50:00 UTC**. The traffic epoch remains **23:50:40 UTC**, and the original dropped-emission windows remain **23:52:05–23:52:45 UTC** and **23:53:05–23:53:45 UTC**. A delayed departure does not restart or shift target motion, AIS report history, manoeuvres or missing-report schedules. The same archived environmental dataset is sampled at the actual later launch UTC; the resulting forcing sequence is an intentional consequence of delaying departure, rather than an identical time-shifted weather replay.

Prelaunch monitoring assumes a **shielded external berth outside the modeled open-water domain**. It does not simulate the weak-wind plant holding station at waypoint 001, and it makes no open-water clearance claim for that waiting period. The recorded navigation replay starts at the actual launch. The gate log independently retains every received report and decision before launch; it never reads the traffic truth CSV.

A release needs three consecutive renewed, roster-consistent observations with report ages no greater than 20 s, a positive robust receding-speed lower bound, and positive received-report tube margins. The startup tube uses uncapped report aging for 120 s; the 300 s forecast tube uses the existing 65 s age budget. Both include a 30 m startup-room allowance and a 2 m/s expanding ownship reachability bound. The monitoring limit is 1,800 s. Failure to find a release records `actual_departure_utc=null` and an explicit failed execution receipt. These report-based conditions are a conservative release screen, not proof of a certified full-loop trajectory or a universal AIS/COLREG guarantee.

After launch, the nominal 25 m gap, 5 m initial uncertainty, 0.05 m/s linear growth, 0.026 m/s² acceleration bound and additional 3 m reserve remain unchanged. Full uncertainty uses uncapped actual report age on every applied interval. Gamma + Direct NMPC, receding-horizon escape and strict action acceptance run with the original five-second step and 24 h postlaunch cap. The target runtime clock adds the staged waiting time while retaining the original 40 s launch-lead parameter.

All eight frozen full attempts are launched in **one eight-worker wave**, with one numerical thread and a separate acados build per process. The host reports 18 logical CPUs and 128 GiB physical memory; memory pressure reported 95% free before launch. Prior successful four-worker loop batches took approximately 11–16.4 min per wave, so the eight-worker batch avoids duplicating two long waves. Every measured timing outcome remains visible under the actual eight-worker loading. Live 600 s checkpoints audit completed prefixes without restarting the missions.

Each run captures all world Python sources, wrapper/helper, departure configuration, protocol, fixture/truth inputs, GPX, chart and checkpoint. `AIS decision timing` measures complete planner choice; `AIS applied decision timing` includes subsequent controller selection, guard, fallback and terminal rejection. Gate decision cost is separate. Final audits distinguish sheltered report exposure from sailed report exposure, predicted received-report margins from independently reconstructed truth margins, command timing from physical safety, and ordered captures from complete-route success.

All **eight full loops completed all eight ordered captures**: outbound 002 → 003 → 004 → 005, then return 004 → 003 → 002 → 001. The independent native 0.05 s reconstruction verifies strictly positive report-centered and truth-centered full uncertainty plus 3 m reserve margins throughout every sailed interval. All 85 captured inputs remain unchanged, all processes exited normally, and no outcome was manually stopped or replaced.

| Encounter | Wind FROM | Guidance | Protected wait | Sailed time | Captures | Truth full + reserve min [m] | Report full + reserve min [m] |
|---|---:|---|---:|---:|---:|---:|---:|
| crossing_fan | 5° | Classical | 0:04:50 | 12:55:20 | 8/8 | +189.272 | +189.272 |
| crossing_fan | 5° | Transformer + confirmation | 0:04:50 | 12:57:15 | 8/8 | +189.272 | +189.272 |
| crossing_fan | 355° | Classical | 0:04:50 | 17:30:20 | 8/8 | +189.272 | +189.272 |
| crossing_fan | 355° | Transformer + confirmation | 0:04:50 | 17:31:25 | 8/8 | +189.272 | +189.272 |
| opposed_crossing | 5° | Classical | 0:05:15 | 13:02:40 | 8/8 | +232.137 | +232.137 |
| opposed_crossing | 5° | Transformer + confirmation | 0:05:15 | 13:00:45 | 8/8 | +232.137 | +232.137 |
| opposed_crossing | 355° | Classical | 0:05:15 | 17:19:30 | 8/8 | +232.137 | +232.137 |
| opposed_crossing | 355° | Transformer + confirmation | 0:05:15 | 17:02:45 | 8/8 | +232.137 | +232.137 |

Crossing-fan launches were **23:54:50 UTC**, after 290 s sheltered monitoring; opposed-crossing launches were **23:55:15 UTC**, after 315 s. All 64 captures lie within the unchanged 50 m circles (largest center range 49.9683 m). The batch contains 87,360 applied commands. The largest complete choose-through-accept decision is **0.067981583 s**; the largest gate decision is **0.000442209 s**. The eight-worker numerical batch finished in **1,024.8 s (17.1 min)**. These are measured algorithm costs under this hardware/loading and exclude physical sensing/actuation latency.

Both original dropped-emission delivery effects occurred during sheltered monitoring. Instantaneous monitored snapshots reached 60 s age; aging the held report until the next five-second poll reached 65 s. At release reports were 20 s old, and the largest age during any sailed interval was 25 s. Consequently this treatment demonstrates safe launch and loop completion after waiting through the outages; it **does not validate underway tolerance of a 65 s AIS outage**. The independent gate audit confirms the original absolute traffic/fault clocks, the first qualifying three renewed report observations, launch-state/report equality and the stated tube calculations. The actual first 300 s displacements remain inside the launch tube assumptions.

`Full launch checks`, [independent gate audit](independent_departure_gate_check.json), [native clearance sweeps](independent_traffic_sweep.json), [combined acceptance](independent_acceptance_check.json), [separate exposure record](fault_exposure_check.json), and [paired-input check](independent_pairing_check.json) retain the evidence. The [primary comparison](comparison.json), [run table](run_metrics.csv), [waypoint table](waypoint_metrics.csv), [segment table](segment_metrics.csv) and `complete telemetry` provide paper-ready numerical data. All expected failures and partial records from the predecessor remain unchanged; these successful launch-gated outcomes do not retroactively repair those earlier runs.

Reproduction uses:

```sh
cd .
# Set MARITIME_ACADOS_ENV to the local acados environment script.
source "$MARITIME_ACADOS_ENV"
export PYTHONPATH="$PWD:$PYTHONPATH" MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
STUDY=data/results/otranto_ais_departure_retest
"$MARITIME_ACADOS_PYTHON" -B "$STUDY/reproduce/launch.py" --dry-run > "$STUDY/dry_run_commands.txt"
# All full attempts require the recorded source freeze and task authorization.
"$MARITIME_ACADOS_PYTHON" -B "$STUDY/reproduce/launch.py" --jobs 8
```

The launcher rejects existing run folders. Filters select an encounter, controlled bearing or guidance; an explicit smaller `--steps` and new output `--study` directory denote an engineering pilot. Native no-launch or postlaunch failures are preserved rather than replaced or censored.

Videos are produced after the final numerical audits. Their opening four seconds summarize the recorded release certificate over the repeated **actual launch fix**: protected-staging duration, fixed traffic/dropout UTC anchors, snapshot report ages, robust receding lower bound and both projected tube margins. No ownship trajectory is invented for the sheltered waiting period. The navigation phase labels actual launch UTC, sailed time and total time since the original request separately. Wind FROM, current TO, bow heading, signed surge and actual COG remain visible. Full raw replays and gate telemetry retain the data between the sampled video frames.

The `playback index` links all eight labeled MP4s, raw replays, GPX tracks, gate summaries and gate telemetry. The `120 preview/layout checks` cover launch cards, navigation, all capture fixes and the longest recorded readouts. Every capture and its neighboring fixes remain represented in the video; full raw replays retain every five-second update.

The `native archive fingerprints`, `actual software environment`, `unchanged-baseline check` and `final numerical/source integrity manifest` complete reproducibility. Previous delivered paper/source artifacts remain under `previous_paper/`.
