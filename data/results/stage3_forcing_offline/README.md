# Offline next-hour forcing forecasts

The causal trend produced the lowest October wind error. Persistence produced the lowest current error overall. The Transformer improved on persistence for wind, and its current estimates had higher error than both baselines. All three forecasts used the same archived issues, locations and target hours. [Prediction records](prediction_records.csv.gz) · [Full metrics](metrics.json)

## Matched archive and models

The experiment used pinned hourly ERA5 wind, SMOC total surface current and ECMWF WAM significant wave height. It evaluated 168 UTC issue hours in each of four seasonal weeks at nine distinct directed legs in the registered Otranto and Brindisi GPXs. Weather was sampled at each leg origin. Where two GPXs contained the same directed leg, it was counted once. January and July provided 3,024 leg-hours for fitting. April provided 1,512 for model selection. October supplied 1,512 held-out leg-hours over seven days. The same locations appear in each season. [Registered routes and seasonal split](../seasonal_native_training/manifest_expanded.json) · [Archive coverage](../seasonal_native_training/forcing/README.md)

Forecasts predict wind and current east/north components and significant wave height at the next common hourly valid time. Each issue has twelve five-minute tokens of both vector components and wave height. Each token holds the latest common hourly knot available through its timestamp. At an hourly issue this yields eleven copies of the preceding hourly vector and one copy of the issue-hour vector. Valid time acts as the availability proxy for these offline archive fields. The separate target vector is the next hourly knot at the same point.

- **Persistence** holds the issue-hour vector.
- **Causal trend** adds half the observed one-hour change. April selected that multiplier from 0, 0.25, 0.5, 0.75 and 1.
- **Transformer forcing** uses the existing frozen 48-component encoder and a linear ridge head trained to estimate the next-hour change relative to persistence. The encoder receives history in the leg frame. The head estimates wind and current changes in that frame, and the forecast rotates them back to east/north. April selected ridge penalty 100 from 0.1, 1, 10, 100 and 1000.

Both change-based forecasts bound their vector increments at the January/July 98th percentile of the corresponding observed hourly change magnitudes. Wave changes use the same percentile rule on absolute scalar increments. Model selection averages the wind and current vector mean squared errors scaled by their corresponding training persistence errors. The trend, ridge head and training scales were saved before October was loaded. [Selection receipt](selection_receipt.json) · [Ridge head](forcing_ridge_head.npz)

## Held-out October

| Method | Wind vector RMSE (m/s) | Wind along-leg MAE (m/s) | Current vector RMSE (m/s) | Current along-leg MAE (m/s) | Wave height MAE (m) |
|---|---:|---:|---:|---:|---:|
| Persistence | 0.67227 | 0.35213 | **0.02041** | **0.00860** | 0.06015 |
| Causal trend | **0.54533** | **0.27943** | 0.02051 | 0.00980 | **0.04480** |
| Transformer forcing | 0.61620 | 0.32979 | 0.02179 | 0.01084 | 0.05988 |

The trend reduced wind vector RMSE by 18.9% against persistence and by 11.5% against the Transformer. It beat both on each of the seven October issue days for wind vector RMSE. Persistence beat the Transformer on all seven days for both current vector RMSE and current along-leg MAE. Multiple leg projections can share nearby archive cells and consecutive issues share weather, so day-level comparisons are descriptive. [Daily results](daily_metrics.csv)

| Verification regime | Leg-hours | Days | Persistence wind / current RMSE (m/s) | Trend wind / current RMSE (m/s) | Transformer wind / current RMSE (m/s) |
|---|---:|---:|---:|---:|---:|
| Headwind | 799 | 7 | 0.67207 / 0.02066 | **0.54171** / 0.02092 | 0.63253 / 0.02212 |
| Opposing current | 356 | 7 | 0.69354 / 0.02366 | **0.55634 / 0.02312** | 0.67488 / 0.02559 |
| Weak wind and opposing current | 63 | 4 | 0.59725 / **0.01748** | 0.49965 / 0.01950 | **0.48586** / 0.01980 |

Regimes use the verifying weather at the target hour. An opposing current has an along-leg component of at most −0.15 m/s. Weak wind is below 4 m/s. Headwind means the wind TO vector falls within 60° of the direction opposite the leg. The Transformer had lower wind vector RMSE in the 63 weak-wind and opposing-current leg-hours than the trend. The trend still had slightly lower along-leg wind MAE there, 0.24009 versus 0.24556 m/s. The experiment measures forcing error at one hour offline. It leaves the navigation selector, AIS guard and closed-loop controller untouched. [Regime and family results](metrics.json)

## Reproduction

Use the simulator Python environment with NumPy, SciPy, xarray, a NetCDF4 backend and PyTorch. Run from the V2 root.

```sh
study=data/results/stage3_forcing_offline
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python -B "$study/reproduce/run_forcing.py"
python -B "$study/reproduce/verify_forcing.py"
```

Verification checks the one-hour issue–target spacing, adjacent archive-vector alignment, the independently calculated trend and vector RMSE, and reproducibility of the five numerical artifacts. [Verification receipt](verification.json)
