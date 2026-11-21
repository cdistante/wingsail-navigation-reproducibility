# Scheduled-continuation feature ablation

This record describes the original experiment workspace. The [reviewer evidence index](../../../../docs/RESULTS.md) lists the released files and verification commands.

The supplemental arm removes the planned switch time, the continuation course and bow features, and the indicator for a planned switch. It uses the same 1,488 labeled candidates, Fossen arrival times, split, architecture, seed, loss, and validation selection rule as the full arrival-time model. The published checkpoint retains the original input schema with zero first-layer weights on these eight inputs. Both selectors use the same runtime ranking and override thresholds.

| July holdout | Full model | Continuation inputs removed |
|---|---:|---:|
| Capture ROC AUC | 0.7911 | 0.8273 |
| Capture Brier score | 0.07049 | 0.06898 |
| Arrival MAE on 47 captures | 325.03 s | 401.92 s |
| Guarded overrides in 92 decisions | 0 | 0 |

The route comparison uses the same 72 route and wind combinations from the primary experiment. The 56 short-route cases retain the full saved waypoint sequence and have a 6,000 s cap. The 16 Naples and Corfu cases measure progress over 10,800 s. Both use the same observed marine conditions and Direct acados configuration per pair. Only the arrival checkpoint changes.

Both models completed 37 of 56 short-route cases and reached the same number of waypoints in every pair. Masking the continuation features increased accepted overrides from 3 to 9 and changed five short-route traces. Among jointly completed cases, the masked model finished 15 s later on Brindisi short at 95° and 40 s and 55 s sooner on the two OpenCPN routes at 175°. The other 34 completion times matched. All 16 bounded long-route cases had identical states and controls.

- `Supplementary table in LaTeX` and [CSV](paper_transition_ablation.csv)
- [Paired cases and trace parity](comparison.csv), [run metrics](run_metrics.csv), and [waypoint metrics](waypoint_metrics.csv)
- `Replay pairs` and `matched Brindisi 95° MP4s`
- `Feature-ablated checkpoint` and `offline metrics`

From the V2 workspace, prepare the same Python and acados environment as the parent study. These commands regenerate the model in a scratch folder and repeat the route comparison using the published checkpoint.

```sh
study="$PWD/data/results/wind_direction_ablation_20261118"
scratch=$(mktemp -d)
python -B "$study/reproduce/train_arrival_without_transition.py" --output "$scratch"
python -B "$study/reproduce/run_transition_ablation.py" --output "$scratch/route_runs" \
  --acados-build-dir "$parent_acados_build" --jobs 3
```
