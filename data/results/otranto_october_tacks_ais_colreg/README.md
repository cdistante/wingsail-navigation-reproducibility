# October Otranto loop with improved tacks, AIS and COLREG

This record describes the original experiment workspace. The [reviewer evidence index](../../../docs/RESULTS.md) lists the released files and verification commands.

This paired study repeats the Otranto 001–005 loop and return, departing
at 2026-10-01 13:00 UTC. Both arms use original archived wind direction,
wind speed, current and waves, the same GPX/chart, 50 m true capture
radius, five-second navigation updates, a 23,000 s cap, initial heading
45°, initial flap −10°, and local map span/resolution/buffer
4715/115/115 m as in the traffic-free October comparisons.

Both use Gamma, Direct acados NMPC, classical forecast tacks and the
selected return-only refinement: fine sailing angles, observed tracking
feedback, one-minute planning over 48 minutes, a nominal aim offset up
to 25 m for close-upwind continuations (60° threshold), and a 60 s
minimum maneuver commitment. True waypoint capture remains immediate
first entry into the original 50 m GPX circles, in order
outbound 002→003→004→005, then return 004→003→002→001.

The classical arm uses the physical hold/release supervisor. The
Transformer arm adds the existing frozen temporal weather model and
two-window physical release confirmation; it is not retrained. AIS
and COLREG supervision remain independent and take priority.

## Traffic and reporting

The mixed-traffic fixture supplies a ferry, cargo ship and tug with the
same sharp course/speed ramps used in the earlier traffic study.
The encounter reference is 13:00:40 UTC, 40 s after launch. Reports
are emitted every 5 s, delivered 10 s late, and emissions during
encounter seconds [85,105) and [115,135) are dropped. The safety
model uses a 25 m nominal buffer, 5 m initial position uncertainty,
0.05 m/s linear uncertainty growth, 0.026 m/s² acceleration bound,
3 m steering reserve and 35 s preview report-age budget. No wind
rotation or traffic rotation is applied. The recorded traffic-free
counterfactual demonstrates that this geometry produces relevant
encounters: see `traffic_relevance_diagnostic.json`.

## Reproduction

From the V2 root, after loading the simulator/acados environment:

```sh
# Set MARITIME_ACADOS_ENV to the local acados environment script.
source "$MARITIME_ACADOS_ENV"
export PYTHONPATH="$PWD:${PYTHONPATH:-}" MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
"$MARITIME_ACADOS_PYTHON" -B data/results/otranto_october_tacks_ais_colreg/reproduce/launch.py --method classical
"$MARITIME_ACADOS_PYTHON" -B data/results/otranto_october_tacks_ais_colreg/reproduce/launch.py --method transformer
```

Each launch creates an isolated acados build directory and writes its
exact command, source snapshots, frozen checkpoint hash, execution
manifest, navigation replay, GPX and full log under `runs/METHOD`.
Existing recordings are never overwritten. `--name NAME`, `--study PATH`
and `--steps N` permit isolated engineering pilots. Dry-run prints the
command without creating a solver or output folder.

After both final runs finish:

```sh
"$MARITIME_ACADOS_PYTHON" -B data/results/otranto_october_tacks_ais_colreg/reproduce/audit_independent.py --jobs 2
"$MARITIME_ACADOS_PYTHON" -B data/results/otranto_october_tacks_ais_colreg/reproduce/audit.py
"$MARITIME_ACADOS_PYTHON" -B data/results/otranto_october_tacks_ais_colreg/reproduce/render_round_trip.py --jobs 2
```

The independent traffic audit reconstructs applied Fossen actions at
0.05 s substeps and computes exact minima on the resulting position/radius
chords against truth and report-projected targets. This is a dense
simulation audit, not an infinite-resolution ODE clearance guarantee.
The route audit checks every timestamp, GPX consistency, first sampled
50 m captures and order, archive samples at departure/captures, paired
runtime arguments, and when Transformer proposals are actually used.

The renderer preserves all first-15-minute updates and every COLREG-active
update, the first learned action and arbitration-transition neighbors,
adds capture holds, and samples the remainder of the full loop. Both
methods share physical frame indices for the arbitration anchors.
Raw replays retain every five-second update. Videos distinguish true
blue 50 m gates from orange nominal planning disks, age-projected
AIS reports from purple truth centers, and nominal versus full
uncertainty-plus-reserve traffic domains. The boat's bow, actual COG,
wind FROM barbs, current TO arrow, actuator states, AIS/COLREG mode,
weather-policy action and independent clearance are displayed.
Classical weather-action labels are derived for display from recorded
AIS priority, hold mode and active forecast references; Transformer
labels retain the recorded policy action/source. The receipt documents
this display-only derivation and raw replays are unchanged.

## Verified results

Both runs completed the eight ordered captures in **3:09:40**
(2,276 applied commands, 2,277 saved frames per run), with identical
boat states, environmental observations and actuator commands. The total
ground-track length was **7,792.4 m** in each arm. The frozen Transformer
changed the recorded policy provenance but did not change this trajectory.

| Guidance | Captures | Loop time | Truth margin beyond full uncertainty | Truth margin after additional 3 m reserve |
|---|---:|---:|---:|---:|
| Classical | 8/8 | 3:09:40 | +23.49 m | +20.49 m |
| Frozen temporal Transformer | 8/8 | 3:09:40 | +23.49 m | +20.49 m |

The independent report-projected margins were +23.494918 m beyond the
full uncertainty buffer and +20.494918 m after reserve. Truth-centered
values were +23.494893 m and +20.494893 m, respectively. No applied
interval entered the nominal 25 m buffer, full uncertainty allowance,
or additional steering reserve. The minimum occurred against tug
999000103 at 115 s after departure. Received report ages ranged from
10 to 30 s. Independent endpoint reconstruction had zero position
residual, and the largest difference from the native swept margin was
1.09e-11 m. See [independent swept audit](independent_traffic_sweep.json)
and `per-interval/per-contact history`.

Both arms had **20 COLREG-active updates**, **67 AIS buffer-maneuver
updates**, no AIS fallback updates, and **1,940 applied forecast
references**. The Transformer supplied sail at 1,940 updates across
33 weather windows, with no hold decision and no decision differing
from its classical comparator. These are applied/cached policy updates,
not 1,940 independent neural-network evaluations. The learned weather
policy deferred for 269 AIS-encounter updates and 67 AIS-steering
overrides; its first accepted action began at 1,680 s. No learned
decision was applied during an active COLREG update. The independent
AIS layer and nominal NMPC remained separate from the weather policy.
Zeros in learned-policy/temporal-deferral columns for the classical arm
mean that no learned supervisor was enabled; shared AIS rows describe
the actual controls in both arms.

| Phase | Waypoint | Elapsed | Distance to original center |
|---|---|---:|---:|
| Outbound | 002 | 1:14:10 | 49.28 m |
| Outbound | 003 | 1:24:45 | 47.26 m |
| Outbound | 004 | 1:33:15 | 47.17 m |
| Outbound | 005 | 2:03:40 | 49.26 m |
| Return | 004 | 2:09:30 | 46.67 m |
| Return | 003 | 2:24:30 | 48.08 m |
| Return | 002 | 3:02:15 | 49.15 m |
| Return | 001 | 3:09:40 | 48.85 m |

The source launch hashes, archived departure/capture samples, every
timestamp and GPX consistency passed the paired audit. The tables above
give shared outcomes; exact distances and all other metrics are in
[waypoint metrics](waypoint_metrics.csv), [leg metrics](segment_metrics.csv),
[run comparison](run_metrics.csv) and [complete comparison audit](comparison.json).
[Telemetry](telemetry.csv) preserves all 4,554 saved frames, including
each initial state; [arbitration transitions](arbitration_transitions.csv)
record handoffs. `Labeled MP4s, full replays and tracks`
are produced by the renderer after audit.
`Complete regional marine-array fingerprints`
include every loaded variable and coordinate and can be regenerated with
`reproduce/fingerprint_archive.py`.

The traffic-free improved October run finished in 2:31:20. This
traffic-enabled loop was 38:20 longer, with much of the difference on
the initial approach to 002. Different realized routes and times sample
different locations/times in the same archive. This comparison is one
synthetic three-contact departure encounter, with no new contacts injected
later in the route. It demonstrates the shared safety arbitration and
waypoint completion in this case, not learned collision avoidance or
general performance under arbitrary AIS traffic.
