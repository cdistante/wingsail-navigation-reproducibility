# Causal weather history ablation in the native Fossen simulator

The four arms share one native physics scheduler, trim-feedback helm and
time-aware AIS/COLREG and chart guard. The physics arm uses its schedule ETA
directly. The other arms fit the same state-weighted ridge loss to relative
arrival-time errors in seconds. Their inputs are the 34 native explicit
features, those features plus the frozen Transformer with observed history,
or those features plus the same Transformer with the latest token repeated.
The two embedding arms both use the same three candidate-bow interactions.

## Protocol

Controller weather takes the last archived valid time available at each
query, with spatial interpolation on that slice. History uses twelve
five-minute tokens of wind and current vectors and significant wave height,
queried at the boat's decision position. The accepted command advances the
native Fossen plant using interpolated archived weather. Synthetic contact
truth enters only the independent clearance audit. Valid time serves as a
simulation proxy for availability because the forcing archives lack
operational issuance timestamps. [Causality checks](reproduce/test_causal_weather.py)

Frozen route files, departures, AIS emissions, reception delays, dropouts,
course and speed changes, and chart data come from the expanded seasonal
manifest. The training partition uses January and July, with April reserved
for validation. The fixed-state October evaluation and 12 closed-loop tests
reuse the six registered Brindisi route and departure groups, each crossed
with two traffic profiles. [Locked roster](../seasonal_native_training/manifest_expanded.json)

The branch collector records up to eight eligible physics starts per
scenario, at least 180 s apart on each leg. Each saved start branches the
incumbent and all legal schedule candidates within 150 s of its ETA. Branches
end at the next geometric 50 m gate, at a safe-action stop, or after up to
80 min. Branches share the native physics continuation after executing their
first schedule. Ridge fitting uses training pairs where both requested
initial actions were applied and both outcomes were captures with positive
full AIS clearance. Censored and failed branches remain in the evaluation.

The corrected ETA must improve by strictly more than 25 s, with the
uncorrected candidate ETA still inside the shared 150 s window. Each
candidate's correction is bounded to ±900 s and used at strength one. The
ridge parameter is fixed at two. The 240 s AIS preview and uncapped robust
guard match the latest 12-case comparison. The 5 s control update, 0.05 s
integration, six-hour mission cap, and 50 m capture radius are shared.
The report and truth replays evaluate the full uncertainty envelope,
including hulls, 25 m separation, uncertainty and a 3 m reserve.
[AIS configuration](../grpo_progress_efficient_ais/README.md#navigation-change)

## October results

All arms completed the same five routes and captured the same 12 waypoints
in order. The seven nonfinishes per arm consisted of six no-safe-action
stops and one six-hour cap. Two stopped before motion. The ten runs with
motion per arm all remained outside the full AIS uncertainty envelope.

| Selector | Scored decisions | Applied overrides | Routes completed | Ordered captures | Median finish time | Median paired change vs physics | Worst full truth margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| Physics | — | 0 | 5/12 | 12/28 | 1:03:29.35 | — | +16.387 m |
| Explicit residual | 147 | 33 | 5/12 | 12/28 | 1:08:47.10 | +60.35 s | +16.387 m |
| Temporal residual | 148 | 42 | 5/12 | 12/28 | 1:09:31.04 | +291.52 s | +16.387 m |
| Repeated-snapshot residual | 149 | 39 | 5/12 | 12/28 | 1:09:39.04 | +237.08 s | +16.387 m |

Paired changes use the five common finishes. The temporal residual was
72.11 s slower than the explicit residual by the paired median, and 8.00 s
faster than the repeated-snapshot residual. Physics finished fastest in
three of the five cases and explicit residual finished fastest in two.
The smallest report-centered full margin was +16.387 m and the smallest
expanded-guard margin was +3.262 m. All measured controller decisions
finished within 5 s. [Matched missions](native/paired_episodes.csv) ·
[Independent clearance](independent_clearance.json)

## Fixed-state calibration

The fit used 141 safe capture pairs from 53 of 61 January and July starts.
The other eight starts had timed-out incumbent branches. April supplied
27 sampled starts and October supplied 40. The same model fit, gain
threshold and candidate rules applied to every partition.

| Selector | April gain MAE | October gain MAE | October predicted overrides | October realized safe savings above 25 s |
|---|---:|---:|---:|---:|
| Physics | 108.47 s | 64.51 s | 0 | 0 |
| Explicit residual | 138.87 s | 57.80 s | 13 | 3 |
| Temporal residual | 210.08 s | 305.52 s | 17 | 3 |
| Repeated-snapshot residual | 197.44 s | 240.60 s | 14 | 3 |

Gain MAE weights sampled starts equally. October supplied 97 eligible
alternatives at 40 matched starts. Four starts had an eligible alternative
with a safe saving above 25 s. Both embedding arms reduced training
prediction error and increased held-out error. The temporal arm crossed
the gate most frequently and found the same three beneficial switches on
the fixed October states as the other learned arms. The richer feature set
therefore showed limited transfer under this weather stream and split.
The 55 min history span contained at most two distinct archived weather
vectors because the source products refresh hourly.
[Fit receipt](models/fit_receipt.json) · [April branches](fixed_states_validation.json) ·
[October branches](fixed_states_test.json)

## Reproduction

The source hash was recorded before the worker module gained its `asdict`
import for serializing states. The audit preserves both versions and checks
that this import is the only source difference. A training case regenerated
with both versions matched the archived branch rows exactly.
[Branch serialization parity](provenance/branch_serialization_parity.json)

Run from the V2 root with NumPy, SciPy, xarray, netCDF4, CasADi, PyYAML and
PyTorch in the native simulator environment. Use a fresh output folder when
regenerating results.

```sh
study=data/results/temporal_embedding_native
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -B "$study/reproduce/run_temporal_ablation.py" all --workers 8 --output "$study/reproduction"
python -B "$study/reproduce/ablation_report.py" --workers 6 --native "$study/reproduction/native"
```
