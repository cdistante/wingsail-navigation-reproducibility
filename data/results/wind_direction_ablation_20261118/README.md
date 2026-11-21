# Saved-route wind-direction and planner ablation study

This record describes the original experiment workspace. The [reviewer evidence index](../../../docs/RESULTS.md) lists the released files and verification commands.

The study uses the nine GPX files in `current_route`. Each route is evaluated with ten navigation configurations and eight matched wind cases. A separate arrival-model ablation removes the planned continuation inputs and repeats the same 72 route and wind combinations.

## Experimental protocol

The wind cases use fixed meteorological bearings of 355°, 5°, 85°, 95°, 175°, 185°, 265° and 275° FROM. Each run retains the archived wind-speed magnitude, wave conditions and current at the sampled positions and times. Thus methods share the forcing fields and starting conditions, and their realized environmental samples can differ as their paths diverge. Forecast queries use archive valid times. This experiment controls wind bearing and preserves the archived speed variation in space and time.

Every run starts at GPX point 001 with a 45° bow and −10° flap. V2 applies a five-second control update in the native Fossen plant. All arms share the GEBCO chart, 115 m local cells, the 3 m minimum depth, the 115 m chart buffer, and tactical chart guidance. Grid MPC penalizes blocked rollout states. acados assesses the chart clearance of the first action. Each mission follows the file's listed waypoints in order. It adds zero return legs. The study evaluates chart-constrained route following in the absence of traffic contacts.

| Saved GPX | Departure UTC | Targets | Cap | Radius |
|---|---|---:|---:|---:|
| `small route.gpx` | 1 Oct 2026 13:00 | 2 | 6,000 s | 50 m |
| `opencpn_001_003_test.gpx` | 1 Oct 2026 13:00 | 2 | 6,000 s | 50 m |
| `opencpn_001_004_test.gpx` | 1 Oct 2026 13:00 | 3 | 6,000 s | 50 m |
| `costiera.gpx` | 30 Sep 2026 03:00 | 2 | 6,000 s | 50 m |
| `otranto_001_002_test.gpx` | 30 Sep 2026 03:00 | 1 | 6,000 s | 50 m |
| `otranto_short_001_003_loop.gpx` | 30 Sep 2026 02:40 | 2 | 6,000 s | 50 m |
| `otranto_short_test_loop.gpx` | 30 Sep 2026 23:50 | 4 | 6,000 s | 50 m |
| `naples_8_8nm_round_trip.gpx` | 30 Sep 2026 13:00 | 4 | 10,800 s | 100 m |
| `rotta_corfu_nord.gpx` | 30 Sep 2026 00:00 | 3 | 10,800 s | 100 m |

The main completion denominator is 56 compact-route runs per method. The 16 Naples and Corfu runs per method retain the full GPX but use a three-hour progress window. The first Corfu leg exceeds 99 km. For these trials, report the number of ordered captures and the best reduction in range to the first target. Completion times are conditional on capture. A cap indicates an uncompleted trial within the window.

Costiera waypoint 003 lies in a GEBCO cell with +48 m modeled elevation. The closest cell boundary is more than 143 m from the waypoint, so its entire 50 m capture circle lies inside the blocked cell on this chart. The sensitivity table removes the same eight Costiera cases from every method, leaving 48 cases per method. Corfu waypoint 002 samples 4 m against the chart's 3 m depth threshold. These are chart-resolution checks. [Chart preflight](route_chart_preflight.csv)

## Paired comparisons

| Compared configurations | Component varied |
|---|---|
| Standard vs Gamma vs anisotropic Diffusion | Harmonic planner with grid MPC held fixed |
| Guarded direction Transformer vs plain Diffusion | Learned direction policy with Diffusion fallback and grid MPC held fixed |
| Gamma with grid MPC vs Direct or LOS acados | Controller with Gamma held fixed |
| Gamma with Direct vs LOS acados | acados path reference and objective |
| Gamma with Direct acados, with vs without classical forecast tacks | Forecast tack scheduler |
| Classical forecast vs either learned reranker | Choice of first-leg schedule within the physical guardrails |
| Learned 120 s progress vs learned arrival time | Deployed selector configurations, including their distinct datasets and objectives |
| Guarded DAgger actuator Transformer vs Gamma plus grid MPC | Learned rudder and flap proposals with an explicit grid-MPC fallback |
| Full arrival model vs retrained arrival model with eight continuation inputs masked | Planned switch time and continuation bearings, with the same labels, split, architecture, loss and seed |

The tactical planners share open-water guidance and chart checks. An open-water leg that uses that guidance may bypass the direction Transformer. The two learned rerankers may retain the classical schedule when their gates lack evidence for an alternative. Both arrival-model checkpoints cover a 50 m radius, so at the 100 m radius in Naples and Corfu they retain the classical schedule when asked to rerank. The raw replays and run logs retain these decisions and the learning arms' acceptance and fallback counters. The actuator checkpoint was trained with archives from this period, so its results include potentially familiar weather.

Compare methods within the same route and wind cell. Compare ±5° outcomes within each route, method and cardinal family. The sweep is deterministic. The two perturbations within a family and the overlapping Otranto route itineraries are related cases, so the tables report counts and paired effects as descriptive evidence.

## Compact-route results

- Adding classical forecast tacking to Gamma with Direct acados increased completion from 29/56 to 37/56 and ordered captures from 74/128 to 86/128. All 29 cases completed in the Direct arm also completed with forecast tacking. The median paired completion change among these was −15 s. When Costiera is excluded, the completion counts are 29/48 and 37/48 respectively.
- Both learned rerankers completed 37/56. The arrival-time reranker made three overrides, reached the same 86 waypoints as the classical forecast, and matched its state and control traces in 54/56 runs. The two changed trajectories, on the Brindisi OpenCPN routes at 175° FROM, each finished 60 s later. The 120 s progress reranker made 90 overrides and reached 85 waypoints. Among mutual completions, it was slower than classical in 11 cases, with a maximum increase of 480 s.
- Plain Diffusion and the guarded direction Transformer each completed 30/56 and reached 72/128 waypoints. Their state and control traces matched in 54/56 trials.
- Moving from −5° to +5° changed completion in 6/28 pairs for each of Standard, Gamma, Diffusion and the direction Transformer with grid MPC. It changed completion in 1/28 pairs for each of the three forecast-tacking configurations.
- The Otranto 001–005 itinerary remained incomplete in all 80 trials at the 6,000 s cap. Direct acados reached five of the pooled 32 targets across its eight trials, and classical forecast tacking reached two. [Per-route results](route_method_summary.csv)

## Long routes and continuation inputs

In Naples, Direct and LOS acados each captured waypoint 002 in 3/8 trials. Each other configuration reached it in 2/8. Across all 80 Naples trials, the boat reached 002 a total of 22 times and reached zero subsequent waypoints. In Corfu, none of the 80 three-hour trials reached waypoint 002. Median best reduction in range to that target varied between 2.185 km and 3.599 km across methods. `Long-route table`

Removing the eight continuation inputs from the arrival model increased held-out arrival MAE from 325.03 s to 401.92 s and increased capture ROC AUC from 0.791 to 0.827. In the compact route runs, the two models completed the same 37/56 trials and reached the same waypoint counts. Five trajectories differed. Among the 37 mutual completions, removing the inputs changed three times by +15 s, −40 s and −55 s. The 16 long-route trajectories matched between the two models. [Matched feature ablation](transition_ablation/README.md)

## Verification and playback

The runner records each control update as a replay frame. The audit checks the exact GPX itinerary, capture order and radius, replay clock and commanded wind direction, and compares each replay position against its GPX track. A separate diagnostic checks each recorded position against the raw GEBCO depth cell. The diagnostic uses five-second samples and does not establish swept clearance between them. The legacy runner signals a no-feasible-heading stop with a nonzero exit. The audit classifies that exact outcome as an incomplete navigation trial after checking the preserved track and frame count. Other execution errors still fail verification. The manifest keeps input and output hashes. Grid MPC uses a checked equivalent rollout kernel. Applied Fossen dynamics remain native. Gamma and Diffusion field memoization reuses exact fields for repeated solver inputs. Logged wall-clock times are affected by parallel execution.

The main audit verified all 720 runs and 798,014 frames. It found 319 completions, 391 step limits and 10 stops with no feasible heading. The 15 runs with recorded positions in raw blocked cells were all on Costiera. All 319 completed runs had zero recorded blocked or out-of-coverage positions. A direct archive check on 216 frames spanning all 72 route and wind combinations reproduced wind speed, current and waves to numerical precision.

Every run retains full navigation telemetry, a replay, a GPX track, and a run log. The 18 main MP4s and two feature-ablation MP4s label the planner, controller, wind direction, and outcome. The videos sample long sequences for playback while retaining the first frame, final frame and waypoint capture events. The 874,598 original frames remain in the corresponding replays.

## Study files

- `Main paper table` and [CSV](paper_comparison.csv)
- `Four wind families`, `±5° sensitivity`, and `paired ablations`
- `Range progress in Naples and Corfu` and [chart-feasible sensitivity](chart_feasible_sensitivity.csv)
- `Recorded-position chart sampling` and [CSV](chart_sampling_summary.csv)
- `Completion figure` and `long-route progress figure`
- [Per-run metrics](run_metrics.csv), [ordered captures and distances](waypoint_metrics.csv), [paired results](ablation_pairs.csv), and [complete telemetry manifest](telemetry_manifest.json)
- `Labeled MP4s with replays and GPX tracks`
- [Continuation-feature ablation](transition_ablation/README.md)
- [Independent archived forcing check](forcing_verification.json), [prediction kernel parity](fast_fossen_parity.json), and [Costiera stop parity](costiera_stop_parity.json)

## Reproducing the runs

Run from the V2 workspace with the simulator's Python dependencies, native acados, and FFmpeg available. Set the thread limits as needed for the host. The commands use a clean output directory.

```sh
study="$PWD/data/results/wind_direction_ablation_20261118"
output=$(mktemp -d)
solver=$(mktemp -d)
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -B "$study/reproduce/run_study.py" --study "$output" --acados-build-dir "$solver" --jobs 12
python -B "$study/reproduce/summarize.py" --study "$output"
python -B "$study/reproduce/verify_forcing.py" --study "$output"
python -B "$study/reproduce/plot_summary.py" --study "$output"
python -B "$study/reproduce/render.py" --study "$output"
python -B "$study/reproduce/run_transition_ablation.py" --main-study "$output" \
  --output "$output/transition_ablation" --acados-build-dir "$solver" --jobs 3
```

Use `--route`, `--method`, `--base`, or `--offset` on the launcher for named subsets. Continue to the summary if the launcher reports a nonzero exit for a no-feasible-heading stop. The strict summary verifies known stops and the complete main matrix. Use `--run-id` with the renderer for any main-study replay. The portable replay player is available with `python -m world.navigation_replay path/to/navigation.jsonl.gz`.
