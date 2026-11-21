# Hard-start Fossen branch expansion and ETA gate retest

This record describes the original experiment workspace. The [reviewer evidence index](../../../docs/RESULTS.md) lists the released files and verification commands.

The training roster uses the January and July Otranto archives. April
supplies separate calibration departures. The fixed October Brindisi
branch benchmark is checksum-pinned to the preceding study. It contains
40 exact states with 97 incumbent–alternative comparisons.

## Branch design

The prospective sampler takes the two Otranto source legs at archived
hourly valid times. Synthetic variations rotate the leg bearing, scale
the leg length, and change initial bow, surge and AIS reports. The
published branch plans record the original and generated geometry,
departure, initial conditions, traffic and candidate identities. Train
and calibration states come from separate weather blocks. Variants
sharing a six-hour weather window share a group for resampling.

A prospective start qualifies when the mark lies within 45° of the
observed wind-from bearing or the current component toward the mark is
below −0.15 m/s. It also needs an admissible incumbent and at least two
legal, forecast-capturing alternatives within 150 s of its ETA. Plan
selection balances the available weather, current and wind regimes and
groups before any branch outcome is integrated. Up to six branches per
start are retained, including the incumbent. When the candidate pool
is larger, selection prioritizes the lowest forecast ETAs and retains
an alternative on the other side of the wind when available.

Every saved state supplies the same initial Fossen state, time and
received AIS history to its incumbent and alternatives. Each branch
executes its first candidate under the existing robust report-based AIS
and COLREG and chart checks, then follows the common native physics
rule to the next 50 m gate, a shield stop or the 80 min cap. The helm
updates every 5 s and the plant integrates every 0.05 s. The controller
uses the latest available validity slice and the simulated plant uses
the interpolated archive. Outcome processing records swept margins
around AIS report projections and synthetic truth.
`Causal simulation and shield`

## Models and acceptance

The explicit model uses 11 schedule, maneuver, instantaneous weather
and received-AIS features. The temporal model adds four candidate
features decoded from the same frozen 48-value weather Transformer.
They represent the observed history's wind rotation, along-goal current
change, cross-goal current change and wave-height change, with a
candidate-specific geometric interaction for each. The weather decoder
fits on training groups using weather tokens available at the decision.

The paired targets use seconds gained against the physics incumbent.
Fitting uses matched first actions and safe captures for both branches.
The predeclared comparison includes ridge and symmetric and
asymmetric Huber models under the same 150 s eligibility window. The
asymmetric loss weights overpredicting savings three times more heavily.
Five bootstrap members sample entire training weather groups. Its
conservative variant subtracts an April-calibrated one-sided uncertainty
margin before applying the strict 25 s threshold. Full outcomes and
calibration sensitivity results remain in the fit receipt.

## Collection and October results

| Partition | Starts | Branches | Weather groups | Usable safe pairs | Starts with a safe gain above 25 s |
|---|---:|---:|---:|---:|---:|
| January and July training | 604 | 3,004 | 32 | 2,360 | 85 |
| April calibration | 111 | 482 | 26 | 371 | 34 |
| Frozen October benchmark | 40 | 137 | 4 | 97 | 4 |

Training included 564 upwind-only starts, 29 opposing-current-only
starts and 11 starts meeting both criteria. Each planned initial action
was applied. The training batch had 42 timeouts. All applied branches
in the training and calibration batches maintained positive full AIS
clearance and passed the chart check. Independent spot replays of one
upwind and one opposing-current start reproduced all nine saved branches.
[Training receipt](branches/train_receipt.json) ·
[Calibration receipt](branches/calibration_receipt.json)

| Model | October switches | Realized safe gains above 25 s | Precision |
|---|---:|---:|---:|
| Explicit ridge | 3 | 2 | 66.7% |
| Temporal ridge | 3 | 2 | 66.7% |
| Explicit symmetric Huber | 3 | 2 | 66.7% |
| Temporal symmetric Huber | 3 | 2 | 66.7% |
| Explicit asymmetric Huber | 2 | 1 | 50.0% |
| Temporal asymmetric Huber | 5 | 2 | 40.0% |
| Explicit asymmetric with April lower bound | 0 | 0 | Undefined |
| Temporal asymmetric with April lower bound | 0 | 0 | Undefined |

The two ridge models selected the same three actions. Their realized
gains were +190.63 s, +36.85 s and +12.07 s. The best point results
therefore found two of the four profitable October starts. The
asymmetric temporal model recovered one additional profitable start
relative to the asymmetric explicit arm and added two false positives.
Both calibrated variants abstained on all 40 October starts. The
April group multipliers were 16.96 for the explicit model and 8.79 for
the temporal model. These are fixed-state results from an historical
benchmark with four route and departure groups.
[Per-start selections and realized time gains](fit/results.json) ·
[Frozen fit and calibration](fit/fit_receipt.json)

## Reproduction

Run from the V2 root in the existing native Fossen Python environment.
The first command generates the training and calibration branch records.
The fitting command requires their recorded hashes and verifies the
prior October hash.

```sh
stage=data/results/eta_stage1_expanded
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -B "$stage/reproduce/run_stage1.py" --workers 14
train_sha=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["labels_sha256"])' "$stage/branches/train_receipt.json")
calibration_sha=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["labels_sha256"])' "$stage/branches/calibration_receipt.json")
python -B "$stage/reproduce/fit_stage1.py" \
  --train "$stage/branches/train.jsonl.gz" --train-sha256 "$train_sha" \
  --calibration "$stage/branches/calibration.jsonl.gz" --calibration-sha256 "$calibration_sha" \
  --test-sha256 832e18f50ba417425a0c2565b0136fc04e540c6ffc32f8fc64501dc1bf66a9de \
  --output "$stage/fit"
```
