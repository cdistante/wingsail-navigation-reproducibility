# Twelve-rollout groups for the native Fossen tack selector

This study compares progress GRPO and progress GRPODropout at 12 rollouts
per same-start group. It also diagnoses the absence of learned tack changes
in the six-rollout GRPODropout result.

## Six-rollout tack diagnosis

The native adapter considers candidates that pass the shared chart, AIS,
COLREG and forecast checks. A candidate's forecast arrival time must be
within 150 s of the incumbent. Its ETA after the learned correction must
improve on the incumbent by more than 25 s.

The six-rollout GRPODropout policy received 603 native strategy calls. In
380 calls the incumbent lacked a forecast capture, 94 lacked a competing
capture within the trust rule, and 48 were outside the actor's supported
inputs. It scored competing alternatives at the remaining 81 decisions.
Six of those decisions had a positive predicted gain. The largest gain
was just 3.21 s, so none met the 25 s gate.

A same-state replay used the boat position, UTC, forecast options and
reports from the two actual six-rollout GRPO changes. Both actors saw the
same seven or four eligible candidates. Replayed GRPO gains were 39.37 s
and 27.48 s. GRPODropout's best gains at those same starts were 0 s and
2.14 s.

[Decision diagnosis and same-state replay](tack_diagnosis.json) ·
[Six-rollout reference study](../grpo_dropout_native/README.md)

## Training protocol

Both arms use the same 26-feature physics-initialized actor, signed
progress reward, dataset, validation criterion and optimizer settings.
Seeds 7309, 8779 and 10921 each sample 1,152 trajectories per method.
The 12-rollout setup has 96 groups per seed, compared with 192 in
the six-rollout setup. With three groups per update, the 12-rollout
setup has 32 updates per seed. Its validation checkpoints occur every
eight updates and match the six-rollout checkpoints by cumulative
trajectory count. Total sampled trajectories per method remain 3,456.

The GRPO arm retains all sampled rollouts in its update. The
GRPODropout arm uses the same adaptive filter, nonpositive-advantage
restoration and probability-weighted recentering as the six-rollout
comparison. Each method selects its best post-update checkpoint by
captures and then capped time on the same 64 validation cases.

This comparison holds rollout count fixed. The change in group size
also changes the number of groups and optimizer updates. Native testing
uses the validation-selected checkpoints for both methods.

### Training results

| Metric | Progress GRPO | Progress GRPODropout |
| --- | ---: | ---: |
| Sampled groups across three seeds | 288 | 288 |
| Sampled trajectories across three seeds | 3,456 | 3,456 |
| Groups with different progress returns | 287/288 | 287/288 |
| Trajectories excluded from the update | 0 | 606 |
| Selected seed | 10921 | 10921 |
| Selected update | 16 | 8 |
| Validation captures | 30/64 | 30/64 |
| Mean capped validation time | 24,050.00 s | 24,050.00 s |

The 12-rollout GRPODropout filter retained a nonzero reward signal
in all 287 informative groups. It excluded 17.53% of trajectories
from optimizer updates, versus 23.12% in the six-rollout training.
The six-rollout GRPO checkpoint reached 31/64 validation marks and
the six-rollout GRPODropout checkpoint reached 30/64.

[GRPO training receipt](training/grpo_progress/training_results.json) ·
[GRPO checkpoint](training/grpo_progress/grpo_progress_selected.npz) ·
[GRPODropout training receipt](training/grpo_dropout/training_results.json) ·
[GRPODropout checkpoint](training/grpo_dropout/grpo_dropout_selected.npz)

## Matched native protocol

Both methods use the same 12 Fossen cases, consisting of six route
and departure groups crossed with two AIS fixtures. Every pair
shares its GPX route, departure, wind, waves, current, traffic,
report delays and dropouts, chart, 50 m waypoint radii and six-hour
cap. The native integration step is 0.05 s and the normal helm
update is 5 s.

Both arms use the same score adapter, 240 s early-avoidance forecast,
chart and COLREG screens, and uncapped robust AIS check on each applied
control. Full AIS margins are independently replayed about both the
projected report centers and synthetic truth. Sailing differences are
paired on shared finishes. [Shared AIS configuration](../grpo_progress_efficient_ais/README.md) ·
[Locked seasonal roster](../seasonal_native_training/manifest_expanded.json)

## Native results

| Metric | GRPO, 12 rollouts | GRPODropout, 12 rollouts |
| --- | ---: | ---: |
| Routes completed | 5/12 | 5/12 |
| Waypoints reached in order | 12/28 | 12/28 |
| Median sailing time among finishers | 1:01:58.77 | 1:01:58.77 |
| Full AIS clearance on runs with motion | 10/10 | 10/10 |
| Worst full margin around a true contact | +16.324 m | +16.324 m |
| Worst full margin around a report projection | +16.324 m | +16.324 m |
| Worst margin under the expanded applied guard | +3.199 m | +3.199 m |
| Largest learned ETA gain | 8.07 s | 3.56 s |
| Learned tack changes applied | 0 | 0 |

Both actors scored alternatives at 81 decisions. Seven decisions for
GRPO and six for GRPODropout showed a positive predicted gain. The
largest predicted gains stayed below the 25 s gate for a tack change.
Both physical tracks and applied controls matched the six-rollout
GRPODropout runs in all 12 cases. The five shared finishes all tied,
including the Brindisi 003 overtake in 1:00:32.69 with +57.259 m minimum
full truth-centered clearance.

Compared with six-rollout GRPO, 12-rollout GRPO took 42.58 s longer on
the 16 October Brindisi 004 crossing and 8.75 s less on the 17 October
crossing. Its other three finishes tied. Every six-rollout GRPODropout
finish tied its 12-rollout counterpart. Two cases per policy stopped
before applying a control and have no swept-clearance measurement.

The independent replay checked all 24 native traces over 1,651,602
Fossen substeps, including 20 causal first AIS forecasts. Every applied
control maintained full clearance. The original stale-report breach
regression was rejected by the shared guard.

[Paired results, including the six-rollout baselines](native/paired_episodes.csv) ·
[Waypoint, timing and protocol verification](native/independent_comparison.json) ·
[Independent clearance and causal replay](independent_clearance.json) ·
[GRPO telemetry](native/grpo_progress/telemetry) ·
[GRPODropout telemetry](native/grpo_dropout/telemetry)

## Reproduce

Use the [same training environment](../grpo_dropout_native/README.md)
and the existing [native Fossen environment](../grpo_progress_native/README.md).
Run from the V2 root and use fresh output directories when regenerating
training or native artifacts.

~~~sh
study=data/results/grpo_group12_native
python -B "$study/reproduce/train_group12_grpo.py" grpo_progress
python -B "$study/reproduce/train_group12_grpo.py" grpo_dropout

# In the native simulator environment
export PYTHONPATH="$PWD" MPLCONFIGDIR=/tmp/maritime_grpo_mpl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -B "$study/reproduce/run_group12_native.py" grpo_progress --workers 6
python -B "$study/reproduce/run_group12_native.py" grpo_dropout --workers 6
python -B "$study/reproduce/verify_group12_native.py"
python -B "$study/reproduce/audit_group12_clearance.py" --workers 6
python -B "$study/reproduce/diagnose_native_tacks.py"
~~~
