# Verify the manuscript results

Run the following from the repository root with Python 3.9 or later.

~~~sh
python3 scripts/verify_reported_results.py
~~~

The check uses the Python standard library. It recomputes the headline counts, paired differences and wind vector errors from the bundled CSV records. It also reads the Otranto audit receipt and verifies the recorded checksums for the pinned native Brindisi and GRPO run records. These checks verify the archived experiments. The simulation and independent swept-geometry audit scripts are separate entry points.

## Evidence by experiment

| Experiment | Start with | What the records show |
| --- | --- | --- |
| Wind-direction sweep | [Run metrics](../data/results/wind_direction_ablation_20261118/run_metrics.csv), [paired ablations](../data/results/wind_direction_ablation_20261118/ablation_pairs.csv) | 720 runs. Compact-route completions increase from 29/56 to 37/56 with forecast tacking. The two longer routes contribute bounded progress records. |
| Matched Otranto ETA gating | [Paired outcomes](../data/results/otranto_layline_eta/eta_comparison.csv), [capture sequence and distances](../data/results/otranto_layline_eta/paired_layline_captures.csv), [independent audit](../data/results/otranto_layline_eta/independent_final_summary.json) | Eight ordered 50 m gates per loop at 355° and 5°. The loops save 365 s and 450 s relative to the matched laylines-off controls. Truth- and report-centered margins include the 3 m reserve. |
| Brindisi compact ridge | [Physics and ridge pairs](../data/results/eta_ridge_closed_loop/native/paired_episodes.csv), [independent clearance](../data/results/eta_ridge_closed_loop/independent_clearance.json) | The ridge saves a mean 36.03 s on five shared finishes. |
| Brindisi protected departure | [Ridge, escape and departure pairs](../data/results/brindisi_escape_departure/native/paired_episodes.csv), [independent clearance](../data/results/brindisi_escape_departure/independent_clearance.json) | Protected departure increases completion from 5/12 to 8/12 and ordered captures from 12/28 to 20/28. |
| Offline wind forecasts | [Forecast and target vectors](../data/results/stage3_forcing_offline/prediction_records.csv.gz), [metric definitions and results](../data/results/stage3_forcing_offline/metrics.json) | October wind vector RMSE falls from 0.672 m/s to 0.545 m/s with the causal trend. The verifier uses all 1,512 held-out October rows. |
| Brindisi trend feedback | [Matched persistence and trend pairs](../data/results/brindisi_hybrid_forcing/native/paired_episodes.csv), [independent clearance](../data/results/brindisi_hybrid_forcing/independent_clearance.json) | With the same staging waits, trend adds 25.17 s to the median paired sailing time. Each arm clears the full AIS envelope and reserve in its ten cases with motion, with a minimum truth-centered margin of +14.746 m. |
| GRPO and GRPODropout | [Six rollouts](../data/results/grpo_dropout_native/native/paired_episodes.csv), [twelve rollouts](../data/results/grpo_group12_native/native/paired_episodes.csv), [ETA gain diagnosis](../data/results/grpo_group12_native/tack_diagnosis.json) | Each arm completes 5/12 routes and 12/28 ordered captures, with a 3,718.77 s median sailing time over the five finishes and a minimum full truth-centered margin of +16.324 m. Accepted tack changes and predicted gains are recorded for both group sizes. |

The Otranto [ETA-gated replays](../data/results/otranto_layline_eta/runs/) and [laylines-off replays](../data/results/otranto_closest_layline_ablation/runs/) contain compressed navigation JSON Lines. Selected priority replays and mission outcomes support the matched-track figure. Brindisi traces under each native arm's <code>telemetry/</code> contain five-second control commands, capture events and planner decisions. Those arm directories retain their original <code>run.json</code> receipts. The compressed traces can be inspected without building a controller.

## Record scope

- The holdout comparisons distinguish simulated contact truth from the last delivered synthetic AIS report. The archived margin fields distinguish full uncertainty, full uncertainty plus the steering reserve, and the expanded report-centered screen used for admission in later native trials.
- The protected Brindisi comparison counts clearance only where a command led to applied motion. Two cases per arm never launched. The Otranto and Brindisi wall times describe the original run hosts and measured timing windows.
- Selected text records use repository-relative paths and portable placeholders for original temporary build directories. The wind-sweep data fields and Otranto run parameters retain their scientific values. Checksums of original environment and source inputs are preserved as provenance. Native Brindisi and GRPO runs and their pinned input records retain their original bytes.
- The release includes primary results, selected replay trajectories and training evidence. Additional historical runs are represented by their aggregate or paired records where available. The archived study READMEs also describe pilot runs, videos and intermediate outputs from the original research workspace.
