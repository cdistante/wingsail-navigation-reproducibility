# October Otranto round trip with forecast tacks

This record describes the original experiment workspace. The [reviewer evidence index](../../../docs/RESULTS.md) lists the released files and verification commands.

The three fresh runs used the same V2 runner, archived wind, waves and
current, GEBCO bathymetry, departure at 13:00 UTC on 1 October 2026, initial
heading of 45°, and initial flap angle of −10°. The marine grids covered
39.5–41.0° N and 18.0–19.5° E. The simulated route started at waypoint 001,
turned at 005 and returned along the reverse waypoint order. All arms used
a 50 m waypoint capture radius, 5 s steps and a 23,000 s time cap. Wind
direction was sampled from the archive along each track.

- `Gamma and Grid MPC` is the fresh reference run.
- `Gamma and Direct NMPC` uses classical forecast tacking.
- `Diffusion and Direct NMPC` uses the same classical forecast tacking.

The forecast planner consults the archived fields for the next 48 minutes
and updates the tack plan during the run. The comparison changes the planner
and controller stack together. The older `Grid MPC run`
provides a separate historical comparison. Its capture times differ from
the fresh Grid run under the current code.

All three fresh runs completed eight ordered captures. Grid MPC finished
in 2:38:40. Both forecast arms finished in 2:33:45 and produced identical
boat trajectories and navigation references at all 1,846 replay frames.
The fresh Grid finished 12:25 earlier than the older Grid recording.

| Phase | Waypoint | Grid time from departure | Grid range (m) | Forecast time from departure, both arms | Forecast range (m) |
| --- | --- | ---: | ---: | ---: | ---: |
| Outbound | 002 | 0:36:10 | 49.41 | 0:38:10 | 48.06 |
| Outbound | 003 | 0:44:35 | 47.09 | 0:48:00 | 46.43 |
| Outbound | 004 | 0:53:15 | 46.71 | 0:56:40 | 47.83 |
| Outbound | 005 | 1:34:25 | 49.61 | 1:29:00 | 49.44 |
| Return | 004 | 1:40:35 | 46.03 | 1:34:40 | 49.77 |
| Return | 003 | 1:55:10 | 49.98 | 1:48:25 | 49.56 |
| Return | 002 | 2:31:35 | 48.82 | 2:26:55 | 49.51 |
| Return | 001 | 2:38:40 | 49.94 | 2:33:45 | 47.35 |

Grid reached the first three outbound waypoints earlier. The forecast arms
gained 8:50 on outbound 004→005, covering 1,179 m compared with 1,571 m for
Grid. Both forecast arms selected a forecast reference at each of the
1,845 control updates. The paired comparison uses one archived departure.

`The audit` verifies arrival order, first sampled capture
within 50 m, full replay timestamps, GPX positions and archive values
resampled at departure and capture. It writes [comparison.json](comparison.json)
with per-waypoint elapsed time, segment duration and capture distance, as
well as a replay comparison between the two Grid runs and between the two
forecast arms.

Each run directory contains the captured launch command, execution log,
compressed navigation replay and GPX track. The copied
`archive launcher` preserves the regional marine
provider settings used by the runs. Its optional batching affects displayed
wind barbs only. From the V2 root, use the simulator's Python environment
to execute the saved commands, substituting available interpreter paths
and separate acados build directories where needed. To rerun the audit,
use the same Python environment and execute:

```sh
python -B data/results/otranto_october_forecast_tacks/reproduce/audit.py
```

The current scheduler also supports later refinements. To repeat these
original navigation runs with the current source, append
`--no-forecast-tack-refinement --no-forecast-tracking-feedback` to each
forecast arm's captured command. The subsequent refinement study and source
snapshots are in [otranto_october_refined_tacks](../otranto_october_refined_tacks/README.md).

## Why the upwind legs differ

The recorded forcing and tack geometry explain the opposite leg effects.
On outbound 004→005, the methods faced wind near 8.1 m/s from 339–344°
and an adverse mean northward current component near −0.28 m/s. The
forecast arms switched from the first tack after 15:20 rather than
21:00, limiting the westward excursion to approximately 99 m instead
of 299 m relative to waypoint 001. The 392.6 m shorter track saved 8:50,
despite lower mean ground speed.

On return 003→002, both first sailed northeast to build northing
against a southward current near 0.256 m/s. The forecast arms held
this tack 90 s longer and then sailed a more southerly westward ground
course (median 259.2° versus 262.6°). Their full track was 171.6 m longer
and took 2:05 more, despite higher mean ground speed. Mean recorded
currents were nearly equal. They still captured 002 at 15:26:55 UTC,
4:40 before Grid MPC, having entered the leg 6:45 earlier.

The `outbound diagnostics` and
`return diagnostics` give measured
wind/current components, track geometry and commanded tack intervals.
The scripts below reproduce these records from the saved replays and
capture audit, without rerunning navigation. A tack switch is timestamped
at the start of the interval applying the new bow reference. Mean ground
speed is geodesic track length divided by interval duration. Gamma and
Diffusion have identical tracks in this comparison. The controller and
scheduler change together, so these diagnostics describe the observed
mechanism without isolating the causal contribution of forecast planning.

```sh
python -B data/results/otranto_october_forecast_tacks/reproduce/analyze_outbound_forcing.py
python -B data/results/otranto_october_forecast_tacks/reproduce/analyze_return_forcing.py
```
