# Underway AIS encounter and race-guidance retest

This record describes the original experiment workspace. The [reviewer evidence index](../../../docs/RESULTS.md) lists the released files and verification commands.

This fresh study extends the [completed protected-departure experiment](../otranto_ais_departure_retest/README.md) with one newly acquired vessel and additional AIS emission losses during navigation. The eight arms retain the two original fleet geometries, wind FROM 355°/5°, classical or frozen Transformer-plus-confirmation weather guidance, archived marine data, 50 m waypoint gates and native plant. The shared regatta time-to-mark option was frozen before the authorized full launch.

The original [protocol](protocol.json) and first-fleet parameter/truth files remain byte-identical to the departure study. `Underway configuration` records the added fixture hashes, source-visibility schedule, additional report faults and guidance treatment separately. `Baseline provenance` verifies the predecessor's 1,020-file integrity manifest and all eight raw/video/gate bindings. No previous study is changed.

The original traffic epoch stays fixed at **2026-09-30 23:50:40 UTC**. The later contact has a fixed reference at original-epoch **t=560 s**, which is **2026-10-01 00:00:00 UTC**. It first emits AIS at **t=400 s** and can first be received at **t=420 s** after the unchanged 20 s delivery delay. New global emission drops occur at **[645,685) s** and **[705,745) s**, corresponding to **00:01:25–00:02:05 UTC** and **00:02:25–00:03:05 UTC**. The original two emission-loss windows also remain active. Report/target clocks are never restarted at actual departure.

The departure gate observes the original received fleet before the later contact can be acquired. It retains the shielded external-berth assumption: no in-water waiting trajectory or clearance is asserted during prelaunch monitoring. After release, the vessel can encounter the additional contact and its delayed/dropped reports while underway. Whether a controller completes the route, preserves the full uncertainty buffer, experiences both later 65 s held-age peaks, or stops early is an experimental result, not a fixture-tuning criterion.

The later fixture is fixed before controller outcomes and has a separate parameter JSON and independent analytical trace for each fleet. Runtime reads synthetic motion parameters solely to generate delayed reports; guidance and the release screen never read the independent truth trace. The auditor combines physical truth for all four vessels. Before first acquisition, any physical safety diagnostic for the unseen vessel uses a separately labeled 65 s equivalent allowance; it is not a received-AIS certificate or controller input.

The regatta flag is `--regatta-time-to-mark`, applied equally to classical and Transformer arms. Existing AIS/chart hard checks, strict native action acceptance and the original uncertainty parameters remain in force. The frozen Transformer continues to supply weather hold/release decisions with physical confirmation; it is not retrained for AIS or COLREG in this study. Full attempts retain five-second integration updates, a 24 h postlaunch cap, one numerical thread and a separate acados build per process. The launcher supports a single eight-worker wave and rejects existing run folders. Every native failure is retained.

Eight full dry-run commands and a distinct 180-command engineering-prefix plan are available. The latter covers the later encounter through original-epoch t=900 s when the release times remain as in the predecessor. The CLI and two gate-only smoke tests passed before the separately authorized full eight-worker launch. Smoke records are retained under `gate_smoke/`; the full batch is a distinct experiment, with full outcomes established by its final audits.

All **8 loops completed all 64 ordered captures within the unchanged 50 m gates**. Every sailed interval retained positive independent truth- and report-centered full uncertainty plus the additional 3 m reserve; the smallest audited margin was **+92.86 m outside that full envelope**. All 8 experienced both actual 65 s held-report-age peaks while underway, with 39–56 recorded AIS/COLREG interventions per arm. The worst complete AIS selection-through-acceptance time was **2.848 s** across 89,367 applied commands; gate and upstream regatta-search costs remain separate. All 88 captured inputs remained unchanged. The one eight-worker numerical batch took 1,347.93 s (22.47 min).

`Playback` links the labeled MP4s, full raw replays, GPX tracks and gate telemetry. `Native mission/source checks`, [independent full sweeps](independent_traffic_sweep.json), [acceptance](independent_acceptance_check.json), [actual fault exposure](fault_exposure_check.json), [run comparison](comparison.json), [waypoint data](waypoint_metrics.csv) and `per-arm telemetry` preserve the numerical evidence. The joint change in later traffic and regatta guidance means a time difference from the predecessor is descriptive rather than an isolated regatta ablation.

```sh
cd .
# Set MARITIME_ACADOS_ENV to the local acados environment script.
source "$MARITIME_ACADOS_ENV"
export PYTHONPATH="$PWD:$PYTHONPATH" MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
STUDY=data/results/otranto_underway_ais_retest
"$MARITIME_ACADOS_PYTHON" -B "$STUDY/reproduce/launch.py" --dry-run
# Only after the relevant source freeze and task authorization:
"$MARITIME_ACADOS_PYTHON" -B "$STUDY/reproduce/launch.py" --steps 180 --study "$STUDY/engineering_prefix" --jobs 8
"$MARITIME_ACADOS_PYTHON" -B "$STUDY/reproduce/launch.py" --jobs 8
```

Media retains every recorded update from original traffic epoch **t=420–900 s**, in addition to waypoint captures/neighbors, COLREG changes and the broader route. An early stop clips this window honestly; no new state extends missing exposure. The overlay aligns contacts by MMSI as the visible roster grows from three to four. Red uncertainty domains exist only for received reports; a gray diamond for preacquisition physical truth is explicitly an offline audit/display overlay. The original protected-staging launch card remains separate from the later underway encounter and report-delivery gaps. Wind FROM, current TO, bow heading, signed surge, measured COG and both planner/accepted-action computation scopes remain identifiable.

Regatta guidance ranks light-wind trims by projected first entry into the active 50 m mark circle over a receding 10 h forecast, with tack-transition allowances, then uses refined one-tack search, observed-velocity feedback and side commitment on both legs. Recorded race-search cost is the last upstream search duration and may repeat when a forecast is cached; it is not a whole controller-decision timer. The complete AIS choice and acceptance/stop timers retain their separate scopes.

Final `media layout checks` bind every capture/neighborhood and the recorded t=420–900 s encounter window, including the tactical inset. `Archive fingerprints`, `software versions`, `baseline preservation` and `numerical/source integrity` complete provenance. Independent decoded-video checks are recorded separately.

The [playback finalizer](reproduce/finalize_playback_index.py) updates study-specific index prose after rendering without altering the videos or their frozen renderer receipts.
