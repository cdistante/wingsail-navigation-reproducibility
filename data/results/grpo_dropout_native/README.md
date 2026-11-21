# Matched GRPODropout on the Fossen tack selector

This comparison changes the policy-training update and tests both frozen
checkpoints with the same native sailing and AIS configuration.

## Training controls

Both arms use the same physics-initialized 26-feature tack actor, reduced
Fossen simulator, signed progress reward and scenario bank. Seeds 7309,
8779 and 10921 each run 64 updates of three groups with six same-start
trajectories per group, for 3,456 sampled trajectories per method. The
original GRPO arm uses its archived matched training checkpoint. The
GRPODropout arm trains from the original physics initialization.

The reward includes capture time, the same terminal penalties and the same
discounted signed progress toward the waypoint circle. Training draws from
96 synthetic cases and checkpoint selection uses 32 separate synthetic
cases, each evaluated with and without the chart exclusion disk. Both
methods select the best post-update checkpoint by validation capture count
and then mean capped elapsed time. These criteria and the native test roster
are separate.

GRPODropout estimates the entropy contribution of each group member from the
standardized reward advantage and old-policy surprisal. It uses the mean
old-policy log probability over sampled decisions with competing actions.
The adaptive filter selects removals from the group, restores nonpositive
advantages, and recenters the retained advantages with weights based on the
old policy. The clipped per-decision likelihood ratios and the KL penalty
use the existing GRPO objective. [GRPODropout, Sections 3.2–3.4](https://arxiv.org/html/2610.11854v1) ·
[Original GRPO training receipt](../grpo_progress_native/training/training_results.json)

### Training results

| Metric | Progress GRPO | Progress GRPODropout |
| --- | ---: | ---: |
| Sampled trajectories over three seeds | 3,456 | 3,456 |
| Trajectories excluded from updates | 0 | 799 |
| Best checkpoint seed | 10921 | 10921 |
| Best checkpoint update | 48 | 16 |
| Captures on synthetic validation | 31/64 | 30/64 |
| Mean capped validation time | 23,438.20 s | 24,050.00 s |

The adaptive filter excluded 23.1% of the sampled trajectories from
updates and retained a nonzero reward signal in 574 of the 575
progress-informative groups. The training receipt records each seed
and all validation checkpoints. [Training receipt](training/training_results.json) ·
[Frozen GRPODropout checkpoint](training/grpo_dropout_selected.npz)

## Matched native test

The two learned policies each rerun the same 12 Brindisi cases from the
original seasonal roster. Every pair shares its GPX route, departure,
autumn wind, waves and current, traffic pattern, AIS report delays and
dropouts, chart, 50 m waypoint radii and six-hour cap. The native dynamics
integrate at 0.05 s and normally apply a control every 5 s. Each policy
uses the same existing native score adapter and trust rule.

The latest early-avoidance configuration applies to both arms. It uses
a 240 s tactical forecast and the uncapped robust AIS guard on every
applied interval. COLREG and chart screens remain active. The independent
replay checks full clearance about the projected report center and the
synthetic truth, including hulls, the 25 m gap, report uncertainty and
a 3 m reserve. Sailing-time differences are paired on routes both policies
complete. [Shared AIS configuration](../grpo_progress_efficient_ais/README.md) ·
[Locked native roster](../seasonal_native_training/manifest_expanded.json)

## Native results

| Metric | Progress GRPO | Progress GRPODropout |
| --- | ---: | ---: |
| Routes completed | 5/12 | 5/12 |
| Waypoints reached in order | 12/28 | 12/28 |
| Median sailing time among completed routes | 1:01:58.77 | 1:01:58.77 |
| Runs with applied motion that maintained full AIS clearance | 10/10 | 10/10 |
| Worst full margin about a true contact | +16.324 m | +16.324 m |
| Worst full margin about a projected report center | +16.324 m | +16.324 m |
| Worst margin under the expanded applied guard | +3.199 m | +3.199 m |
| Applied learned tack changes | 2 | 0 |
| Stops with no admissible control | 6 | 6 |
| Six-hour caps | 1 | 1 |

The fresh GRPO rerun reproduced the previous efficient-AIS GRPO outcomes
in all 12 cases. Each policy completed the same five routes. Relative to
GRPO, GRPODropout was 42.58 s slower on the 16 October Brindisi 004 crossing
and 8.75 s faster on the 17 October Brindisi 004 crossing. The other three
finishes tied. The median paired GRPODropout-minus-GRPO time difference was
0 s and the mean was +6.76 s. GRPODropout retained the incumbent at all
native strategy decisions and matched the previously measured physics
outcomes across the 12 cases.

Both policies retained the Brindisi 003 overtaking finish, reaching 002
and 003 in 1:00:32.69 with +57.259 m minimum full truth-centered margin.
The historical stale-report breach case still stopped at 120 s, with
positive clearance through the applied prefix. Two cases per method stopped
before motion and have no swept-clearance measurement.

The independent audit replayed all 24 trajectories and 1,650,925 native
Fossen substeps. It found no full-clearance incursions and reproduced the
first forecast decision in all 20 affected traces with later observations
and synthetic truth withheld. All complete control timings stayed below
5 s.

[Paired results](native/paired_episodes.csv) ·
[Waypoint, timing and protocol verification](native/independent_comparison.json) ·
[Independent clearance and causal replay](independent_clearance.json) ·
[GRPO telemetry](native/grpo_progress/telemetry) ·
[GRPODropout telemetry](native/grpo_dropout/telemetry)

## Reproduce

Run from the V2 root. Train using the same NumPy 2.5.5 and PyTorch 2.15.0
environment as the [original GRPO training](../grpo_progress_native/training/README.md).
The native reruns use the separate [Fossen benchmark environment](../grpo_progress_native/README.md)
and match the package versions pinned by the [efficient AIS study](../grpo_progress_efficient_ais/README.md).
Use fresh output directories when regenerating artifacts.

~~~sh
study=data/results/grpo_dropout_native
python -B "$study/reproduce/train_grpo_dropout.py"

# In the native simulator environment
export PYTHONPATH="$PWD" MPLCONFIGDIR=/tmp/maritime_grpo_mpl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -B "$study/reproduce/run_dropout_native.py" grpo_progress --workers 6
python -B "$study/reproduce/run_dropout_native.py" grpo_dropout --workers 6
python -B "$study/reproduce/verify_dropout_native.py"
python -B "$study/reproduce/audit_dropout_clearance.py" --workers 6
~~~
