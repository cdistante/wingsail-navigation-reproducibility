# Autonomous wingsail navigation reproducibility

Research materials for *Forecast Guidance and Learned Tack Selection for an Autonomous Wingsail Vessel under Delayed AIS*. The study evaluates closed-loop waypoint execution under archived wind, current and waves, with synthetic traffic and controlled AIS report delays and dropouts.

## Start here

- [Verify the reported results](docs/RESULTS.md) using the recorded outcomes and paired comparisons.
- [Inspect the fresh Otranto and Brindisi reruns](data/reviewer_reruns/README.md) with per-case audits, replays, and telemetry.
- [Run the simulator](docs/RUN_SIMULATION.md) for setup, a small test, and the commands and prerequisites for the native Brindisi and Otranto workflows.
- [Check data provenance and access](docs/DATA.md) for the archived forcing, bathymetry, and AIS fixtures.

The `paper-v1.1` tag pins the tested simulator setup. The `paper-v1.2` tag includes the fresh rerun evidence. From a fresh checkout, verify the released files and recompute the headline comparisons:

```sh
shasum -a 256 -c SHA256SUMS
python3 scripts/verify_reported_results.py
```

## What's in the repository

| Path | Contents |
|---|---|
| `world/` | Vessel dynamics, guidance, tack scheduling, NMPC integration and AIS screening |
| `current_route/` | GPX waypoints for the study routes |
| `data/results/` | Selected experiment records, inputs, reference outputs and reproduction scripts organized by study |
| `tests/` | Simulator regression tests |

The headline comparisons cover the compact-route sweep, the two matched Otranto ETA-gated loops, and the held-out Brindisi tests of tack selection, protected departure and weather forcing. The individual study records define the capture radius, AIS timing and clearance measure for each comparison. Clearance is assessed over applied simulated motion. Runtime measurements reflect the original hardware, solver and workload.

Follow the attribution and source terms in [the data guide](docs/DATA.md) when reusing the environmental inputs.
