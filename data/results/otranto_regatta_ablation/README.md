# Otranto paired regatta comparison

This record describes the original experiment workspace. The [reviewer evidence index](../../../docs/RESULTS.md) lists the released files and verification commands.

This study reruns all eight [original underway-AIS cases](../otranto_underway_ais_retest/README.md) with regatta mode on and off. The 16 fresh attempts cross two encounter fixtures, winds from 355° and 5°, and classical or Transformer-plus-confirmation guidance. A pair uses identical commands after removing the regatta flag and normalizing only fixture-copy, wrapper, output and solver-build paths.

Regatta on enables the time-to-mark holding objective and activates the configured tack refinement on the outbound leg. Both settings keep the same forecast hold/tack options, existing return refinement, Gamma/Direct acados NMPC, 50 m capture radius, five-second update and strict AIS/COLREG controls. The comparison measures the full regatta option. The study preserves any stop and any decision that exceeds its 5 s budget.

The two AIS fixtures and two second-encounter additions were copied byte-for-byte. The four 40 s emission gaps retain their fixed original encounter-clock positions, two during staging and two underway. The reporting delay is 20 s. Forecast arrays and coordinates, simulator source, route, bathymetry and the frozen Transformer checkpoint match the preceding study by content hash. `Preflight` records these checks and the normalized command hash for each pair. The previous study files and their playback remain available separately.

All 16 processes launch together on the same 18-logical-core host with one numerical thread per process and separate output and acados-build directories. Each process retains complete choose-through-acceptance/stop AIS timings in its own log. These wall timings describe the measured host load. Virtual boat and receiver clocks advance in fixed five-second steps. Time to 002 and total time are measured after the recorded launch from staged protection.

## Replay the protocol

From the V2 project root in the pinned runtime environment, the following prints the 16 fixed commands with placeholder solver-build paths.

```sh
# Set MARITIME_ACADOS_ENV to the local acados environment script.
source "$MARITIME_ACADOS_ENV"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$MARITIME_ACADOS_PYTHON" -B data/results/otranto_regatta_ablation/reproduce/launch.py --dry-run
```

For a new recording, use a separate study folder at the same depth containing this `reproduce` directory, the byte-identical AIS fixture files, `protocol.json`, `underway_config.json` and `baseline_provenance.json`. Run its `reproduce/launch.py` without `--dry-run`. The launcher refuses to overwrite any existing run. The original study is a read-only provenance reference. Each new directory under `runs/` ends in `__regatta_on` or `__regatta_off` and contains the executed command, source snapshot, gate data, unfiltered native log and closed execution receipt.

The executed flag and `regatta_enabled` field are the source for playback labels. The original simulator writes an outbound scheduler note from `--forecast-refinement-phase return` alone. In the regatta-on runs the flag overrides that phase setting outbound, including when the native replay header reports return-only refinement. The corrected playback copies and labeled videos use the actual setting. The playback index also links to the raw recordings used for the audit.

`reproduce/verify_launch.py --gate-only` checks the received-report sequence and staged release against the previous run of each arm. The default mode binds all closed execution receipts and compares the exact states, archived observations, received reports and applied controls against predecessor regatta-on runs. It records the first actual divergence within each on/off pair.

## Results and playback

All 16 attempts completed the eight expected captures in order. Regatta on reached outbound 002 sooner in three pairs and completed the full loop sooner in seven. The seven full-loop gains ranged from 1 min 55 s to 10 min 40 s. The crossing-fan 355° Transformer case took 1 h 4 min longer with regatta on, including 30 min 10 s of extra time to 002. The full AIS selection records contain 178,340 decisions, with a maximum of 3.5776 s on the measured host. All eight regatta-on reruns reproduced the prior physical traces at every recorded fix.

- [Paired times, waypoint verification, clearance and AIS latency](independent_comparison.csv)
- [Full independent audit receipts](independent_comparison.json)
- [All waypoint time differences](independent_waypoint_deltas.csv)
- `Native clearance and timing summary`
- `Sixteen replays and labeled MP4s`
- `Input and physical reproducibility verification`
- `Updated manuscript`
