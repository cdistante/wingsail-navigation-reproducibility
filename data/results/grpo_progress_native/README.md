# Progress-shaped GRPO in the native Fossen simulator

This experiment asks whether a progress reward supplies a denser GRPO training
signal and whether the selected policy improves a full native route under
held-out forcing and AIS. It reruns the registered autumn seasonal roster
with physics and with the trained residual.

## Training and selection

Three seeds each ran 64 GRPO updates, with three groups of six rollouts per
update. A group's rollouts start at the same reduced boat state, mark and
forcing. The time objective rewards early gate entry and penalizes timeout
and blocked episodes. The added shaping term tracks signed changes in the
fraction of the initial gap to the capture circle closed. The potential is
clipped to \([-1,1]\), and the change is discounted by 0.985 per 300 s and
scaled by 3,600 s. Moving away can reduce the reward. A timeout has a 43,200 s
cost, and a blocked episode carries an additional 7,800 s penalty.

The original synthetic split supplied 96 training scenarios and 32 validation
scenarios. Validation runs each scenario in open water and with the chart
obstacle. It ranks checkpoints by captures and then mean capped time. The
active checkpoint was chosen from post-update checkpoints using only this
validation split. The receipt records the physics validation result, exact
split IDs, checkpoint hash and per-seed curves.

| Training seed | Groups with different terminal-time outcomes | Groups with different shaped returns |
| --- | ---: | ---: |
| 7309 | 35/192 | 192/192 |
| 8779 | 33/192 | 191/192 |
| 10921 | 37/192 | 192/192 |

One seed improved on its physics initialization in validation. The selected
checkpoint, seed 10921 at update 48, reached 31/64 validation marks versus
30/64 for physics. Its mean capped time was 23,438.20 s versus 24,051.70 s.

[Training implementation](reproduce/train_progress_grpo.py) ·
[Training receipt](training/training_results.json) ·
[Frozen checkpoint](training/grpo_progress_selected.npz)

## Native held-out protocol

The native test consists of all 12 cases in the prior locked seasonal
roster. Three Brindisi GPXs supply six distinct route and departure groups
from the 12–18 October 2026 UTC weather block. Each has an oblique crossing
and a raking overtake, with delayed AIS reports, emission gaps and finite
course and speed changes. The native roster was excluded from training and
checkpoint selection. Both arms reuse the same archived wind, current and
waves, original waypoint order, 50 m capture circles and six-hour cap.

Both arms integrate the same 3-DOF Fossen plant at 0.05 s and steer with the
same native trim-feedback controller every 5 s. The physics comparator
uses a 48-minute persistence forecast. The GRPO adapter adds the trained
residual to live native schedule ETAs. It considers admissible capture
schedules within 150 s of the physics ETA and switches only when its
corrected ETA improves by more than 25 s. These thresholds follow the
existing native benchmark. AIS and COLREG, chart screening, the native
helm and commitment handling are shared. The guard uses reports available
at the decision time. Full clearance is checked around both the projected
report center and the synthetic truth, including hulls, a 25 m gap,
report-age uncertainty and a 3 m reserve.

[Locked roster and archive index](../seasonal_native_training/manifest_expanded.json) ·
[Native GRPO adapter](reproduce/native_grpo.py) ·
[Physics inputs](native/physics/inputs.json) ·
[GRPO inputs](native/grpo_progress/inputs.json)

## Held-out results

| Metric | Physics | Progress-shaped GRPO |
| --- | ---: | ---: |
| Routes completed in GPX order | 5/12 | 5/12 |
| Waypoints captured in order | 12/28 | 12/28 |
| Median sailing time among completed routes | 1:01:28 | 1:01:28 |
| Completed routes with positive full clearance | 5 | 5 |
| Capped at six hours | 1 | 1 |
| Stopped with no admissible native action | 6 | 6 |
| Full clearance on episodes with applied motion | 9/10 | 9/10 |
| Minimum full truth-centered margin | −1.096 m | −1.096 m |
| Minimum full report-centered margin | +0.166 m | +0.166 m |
| Maximum complete control decision | 0.087 s | 0.122 s |
| Control decisions at or above 5 s | 0 | 0 |

Both policies finished the same five routes. Four shared finishes tied
and one was faster with GRPO, for a median paired change of 0 s and a mean
paired change of −35.94 s. The GRPO adapter changed four applied tack
choices, all during the 17 October 12:00 UTC Brindisi 004 oblique crossing.
It finished in 1:14:26.11 versus 1:17:25.83, saving 179.72 s. It reached
002 20.60 s later, then sailed the 002–003 leg 198.41 s faster. That run
retained a positive minimum full truth-centered margin of 61.41 m.

In the 13 October 18:00 UTC Brindisi 003 raking overtake, both policies
followed the same controls. The worst full truth margin was −1.096 m
at 169.95 s, during an interval that began with a 45 s old report.
The full report-centered margin was +0.166 m. Each run stopped after
that interval. Two additional cases stopped before applying a control
in each arm, leaving no travelled-segment margin to measure.

The trace audit checked waypoint order and first gate entry for all 24
episodes. The offline clearance audit reintegrated 16,530 applied controls
over 1,651,664 native substeps and recomputed the swept margins around
both sets of contact positions. The 12 physics cases reproduced their
prior seasonal outcomes. The benchmark shows one clear speed gain within
six route and departure groups in one autumn week. The completion and
clearance outcomes were the same in all paired cases.

[Paired episode results](native/paired_episodes.csv) ·
[Waypoint and comparison audit](native/independent_comparison.json) ·
[Independent clearance replay](independent_clearance.json) ·
[Physics telemetry](native/physics/telemetry) ·
[GRPO telemetry](native/grpo_progress/telemetry)

## Reproduction

From the V2 root, use Python 3.12 with NumPy, SciPy, xarray, netCDF4,
CasADi, PyYAML, Matplotlib and PyTorch. The two native input receipts and
training receipt record the versions used in this study. A fresh study
can use the default output directories below. The training and native
runners also accept explicit output directories for follow-up runs.

~~~sh
study=data/results/grpo_progress_native
export PYTHONPATH="$PWD" MPLCONFIGDIR=/tmp/maritime_grpo_mpl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -B "$study/reproduce/train_progress_grpo.py"
python -B "$study/reproduce/run_native_comparison.py" physics --workers 6
python -B "$study/reproduce/run_native_comparison.py" grpo_progress --workers 6 \
  --checkpoint "$study/training/grpo_progress_selected.npz"
python -B "$study/reproduce/verify_native_comparison.py"
python -B "$study/reproduce/recheck_clearance.py" --workers 6
~~~
