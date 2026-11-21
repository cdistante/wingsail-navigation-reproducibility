# Otranto temporal recovery with AIS and COLREG

This record describes the original experiment workspace. The [reviewer evidence index](../../../docs/RESULTS.md) lists the released files and verification commands.

The paired experiment follows a single eastbound gate off Otranto. Both trials start at 2026-10-01 13:49:20 UTC at 40.225° N, 18.746666667° E. The goal is at 40.225° N, 18.758425190° E. Both use a 50 m capture radius, 5 s control updates, a 50-minute cap, a 20 m local grid, Fossen dynamics, Gamma guidance, Direct acados, time-aware AIS, COLREG course preferences and current-time hold trim. The Transformer arm adds the frozen temporal hold model and two-window release confirmation.

Wind speed, current and waves come from the marine archive. Wind direction is set to 180° FROM. The saved sharp mixed-traffic fixture supplies the ferry, cargo ship and tug. The encounter reference is 13:50:00 UTC, 40 s after launch. AIS reports arrive 10 s late and emissions are dropped at encounter seconds 85–105 and 115–135. The uncertainty model uses a declared target acceleration bound of 0.026 m/s² and a 3 m steering reserve.

| Hold guidance | Gate capture | Final range | Full uncertainty margin after reserve | COLREG-active updates |
|---|---:|---:|---:|---:|
| Classical | 41 min 15 s | 48.68 m | +39.18 m | 27 |
| Temporal Transformer with release confirmation | 41 min 10 s | 49.86 m | +39.18 m | 27 |

The independent audit replays each applied actuator demand at 0.05 s using the recorded weather and reconstructed target motion. Both boats stayed outside the full uncertainty envelope, with a smallest truth-centered margin of +42.18 m before the reserve. The independently computed report-centered margin after reserve was +36.99 m in both trials. [Independent swept audit](independent_traffic_sweep.json) · `Per-step truth and report history`

The Transformer deferred to the planner for 152 updates during the AIS encounter. Its first accepted evaluation occurred 760 s after launch. It selected sail for 342 updates across seven weather windows. Every accepted choice matched classical guidance. No learned hold trim was applied, and no learned decision was applied during a COLREG-active update. The trajectories first differed at the opening five-second AIS buffer maneuver, before learned guidance resumed. The shared-clock separation was at most 0.584 m. [Arbitration transitions](arbitration_transitions.csv) · `Matched replay verification`

The experiment measures the traffic handoff, clearance and single-gate capture on one synthetic encounter geometry, archived forcing window and controlled wind direction. The paired runs used explicit start and goal coordinates and were recorded before the later GPX departure anchoring correction. The 420-step pilot reached its cap while both boats were still closing on the gate. `Pilot records` · `Pilot clearance audit`

## Data and playback

- [Paper comparison CSV](paper_comparison.csv) and `LaTeX table`
- `Full navigation replays, GPX tracks and labeled MP4s`
- `Run inputs and archived source hashes`, `archive sampling verification` and `replay manifest`
- [Production runner](../../../world/run_legacy_fossen.py), `matching synthetic reference benchmark` and `frozen Transformer checkpoint`

## Reproduction

Use the V2 project interpreter with native acados and FFmpeg. Match the recorded source hashes when reproducing the saved experiment. Use a fresh output directory and point `solver` at a compatible acados build directory.

```sh
study="$PWD/data/results/otranto_temporal_ais_20261118"
output=$(mktemp -d)
solver=/path/to/acados-build-directory
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -B "$study/reproduce/run_temporal_ais.py" --study "$output" --acados-build-dir "$solver"
python -B "$study/reproduce/audit_marine_truth.py" --study "$output"
python -B "$study/reproduce/audit_temporal_ais.py" --study "$output"
python -B "$study/reproduce/render_temporal_ais.py" --study "$output"
```
