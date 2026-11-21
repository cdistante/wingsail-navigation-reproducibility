# Offline ETA acceptance ablation

The offline experiment reuses the locked causal-weather native Fossen
branches from the four-arm study. The fit uses January and July, the
lower bound uses April, and evaluation uses the existing October
fixed-state benchmark. There are 40 October starts and 97 incumbent
comparisons across four sampled route and departure groups. Four starts
had an admissible alternative with a realized safe saving above 25 s.
[Source branch protocol](../temporal_embedding_native/README.md)

## Methods

The existing 34-feature ridge model provides the symmetric reference.
Two Huber comparators use identical five-seed bootstrap draws of whole
route and departure groups. One fits symmetric Huber loss and the other
weights overestimation of savings three times more heavily than
underestimation. Both use a 60 s Huber transition, equal weight per
decision start within a bootstrap draw, and the existing ±900 s
per-candidate correction cap. Fitting uses exact safe capture pairs.

The lower bound subtracts the April calibration multiplier times the
ensemble standard deviation, with a floor of 25 s on the standard
deviation. Calibration takes the maximum standardized overprediction
over all candidate pairs within each departure, then uses the grouped
80% split-conformal order statistic. A failed alternative is assigned
the branch horizon for this finite-horizon calibration utility. A
sensitivity analysis uses exact safe pairs only. Both produce the same
multiplier of 27.52, driven by an exact pair in the April 15 group.
With six calibration groups this statistic is the largest observed
group score. Evaluation measures transfer between seasons and routes.

Each selector ranks only frozen admissible alternatives within 150 s
of the incumbent schedule ETA. A selected score must exceed 25 s.
A true positive requires both branches to capture with full clearance
and the alternative to arrive more than 25 s sooner.

## October results

| Method | Switches | Safe gains >25 s | False positives | Precision |
|---|---:|---:|---:|---:|
| Archived symmetric ridge | 13 | 3 | 10 | 23.1% |
| Symmetric bootstrap Huber | 7 | 0 | 7 | 0% |
| Asymmetric bootstrap Huber | 5 | 0 | 5 | 0% |
| Asymmetric with April lower bound | 0 | 0 | 0 | Undefined |

The asymmetric fit reduced state-weighted point gain error from 57.80 s
to 56.33 s and failed to identify any of the four profitable starts.
The calibrated rule abstained at every start with either treatment of
April's three timed-out alternatives. These results describe fixed-state
branch outcomes. [Per-method metrics, bootstrap draws and selected actions](results.json)

## Reproduction

Run from the V2 project with NumPy and SciPy installed.

```sh
python -B data/results/eta_uncertainty_offline/reproduce/evaluate_eta_gate.py
```
