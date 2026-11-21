# Gamma and Diffusion on the sharper mixed-traffic benchmark

This record describes the original experiment workspace. The [reviewer evidence index](../../../../docs/RESULTS.md) lists the released files and verification commands.

The 25 matched pairs use the same 6° ship turns, 0.35 m/s speed changes, chart grids, constant wind, Fossen dynamics, acados Direct controller, and time-aware AIS layer. Each 12-run matrix covers 20 m and 115 m grids, COLREG preference off and on, a 10 s delay, two 20 s dropouts, and combined faults. The remaining pair is the 20 m COLREG combined probe with the updated controller at the original launch. The command audit verifies matched options and common source hashes within each pair. `Command and source parity`

| Launch and safety settings | Gamma waypoint completion | Diffusion waypoint completion | Gamma worst full-buffer margin | Diffusion worst full-buffer margin |
|---|---:|---:|---:|---:|
| 03:00:00 UTC, zero extra reserve | 12/12 | 12/12 | −16.20 m | −16.20 m |
| 02:59:20 UTC, 3 m reserve and 35 s preview budget | 12/12 | 12/12 | +11.56 m | +11.73 m |
| 03:00:00 UTC probe, 3 m reserve and 35 s preview budget | 1/1 | 1/1 | −18.99 m | −18.99 m |

The margins use ship centers from independently integrated traffic truth and the full report-age uncertainty over each 0.05 s Fossen integration substep. The extra steering reserve is separate. Both earlier-launch matrices cleared that reserve and used no emergency fallback. The original-launch zero-reserve pairs had identical positions and actuator commands at every replay sample. [Matched results](matched_comparison.csv)

Diffusion finished four earlier-launch cases faster, six slower and two at the same simulated time. The differences ranged from 130 s faster to 85 s slower. In the ten early pairs with different commands, the first difference appeared in the shared time-limited AIS maneuver refinement. The initial nominal heading, observed ship states and local mask matched. The trajectory and clearance differences are whole-pipeline results and do not isolate a tactical planner effect. [Per-run results](run_metrics.csv)

Gamma and Diffusion produced the same tactical heading, grid path and navigation target at all 37 sampled identical inputs. Those samples span both grids over the combined earlier-launch runs. The potential fields differed. The local approaches in these samples were unobstructed and used the same straight path or open-water reference rules. Median tactical calculation time in this paired wall-clock sample was 10.94 ms for Gamma and 104.77 ms for Diffusion. The benchmark tests moving traffic through the shared AIS layer. [Identical-input comparisons](same_input_references.csv)

## Data for later comparisons

- [Per-run CSV](run_metrics.csv) contains 50 runs with route settings, waypoint outcome, timing, distance, track length, cross-track error, full and nominal clearance, fallback counts, AIS age and links to the raw replays, logs, GPXs and commands.
- [Matched CSV](matched_comparison.csv) contains 25 pairs with clearance and timing differences, first differing controls and synchronized track separation.
- `Navigation samples` and `raw AIS contact samples` cover both planners. The gzip-compressed CSVs contain 15,528 navigation rows and 46,584 contact rows. Join on `run_id` and `step`. Raw AIS rows carry reports as received, including held reports during outages. Initial-frame controls are blank because no action has yet been applied.
- `Diffusion substep audit` and `Gamma report` link independent swept clearance and per-contact uncertainty histories. The shared analytic traffic trace spans both launch times. `Traffic truth trace`
- The playback indexes link replays and labeled MP4s for both launch times and the separate original-launch probes. [Diffusion playback index](independent_video_artifacts.json) · `Gamma playback index`
- `Reproduction scripts` run the Diffusion matrix from the saved Gamma commands, export telemetry and recompute the comparisons. For the original-launch matrix, pass the workspace matching the archived `zero_reserve_source.sha256` as `--original-source`.
