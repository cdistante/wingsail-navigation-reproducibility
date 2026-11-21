# Joint weather Transformer and AIS policy

This record describes the original experiment workspace. The [reviewer evidence index](../../../docs/RESULTS.md) lists the released files and verification commands.

The joint policy captured **110 of 216** held-out waypoint cases. The physics scorer captured **114 of 216**. Among the 110 shared captures, the learned policy was slower by **239.5 s at the median**. Both stayed beyond the full AIS uncertainty buffer and 3 m reserve on every applied sweep in this experiment. The minimum remaining margin was **0.144 m** for the joint policy and **0.369 m** for physics. Physics had the strongest completion, paired timing and clearance result.

`Comparison figure` · [Episode results](test_episodes.csv) · `Applied trajectories` · [Independent checks](independent_metrics.json)

## Split and protocol

| Partition | Cases | AIS profiles |
| --- | ---: | --- |
| Training | 96 synthetic and 63 July Brindisi leg and weather cases | Crossing, head-on, short mixed dropout, long mixed dropout |
| Policy validation | 32 separate synthetic cases | Same four training profiles |
| Test | 108 September/October archived leg and weather cases from five GPX files | Single-ship overtaking and sharp mixed dropout |

The test contains the three Otranto GPXs, Costiera and Corfu. The July Brindisi training group contains the two OpenCPN GPXs and Small Route. The train/test sets share neither GPX files nor exact leg endpoints, and the split groups nearby variants. Some Otranto GPXs repeat exact legs within the test. Wind-direction variants and traffic treatments are therefore paired sensitivity cases. The two test traffic profiles were withheld as complete profiles. They share vessel templates with the mixed training fixtures, and the sharp case shares the long-dropout timing but changes vessel course and speed during it. [Split ledger](splits.json)

The runs are independent legs in the reduced Fossen steady-motion simulator. Every archived case uses observations at the leg origin from its recorded departure and three hours later. The model uses these two snapshots over its six-hour cap. Wind speed, current and waves are archival. Each leg includes the native wind direction and eight prescribed wind directions. The physics continuation and all learned arms receive the same two-snapshot forecast. The embedding uses causal observed five-minute windows only. An exact segment/circle intersection records the first entry into each GPX's arrival radius.

Eighteen test cases ended at chart preflight because their entire target disk was within a blocked grid cell. The scorers used the physics fallback on 54 Corfu cases with 100 m capture radii outside the learned 50 m domain. The remaining 144 cases exercised the learned methods. Full native route chaining, chart traversal and COLREG behavior are outside the reduced experiment.

## Learned candidate score

The 48-value Transformer follows the existing weather encoder's two-layer architecture. This experiment pretrains a fresh checkpoint on synthetic weather denoising. It fits scaling and weights on 84 synthetic training cases and chooses the encoder epoch using another 12 from the policy training partition. Its 12 observed windows contain wind and current velocities projected along and across the current mark bearing, and wave height. Launch padding repeats the first available observation. A future-weather perturbation produced exactly zero representation change before the observation entered the history. `Encoder protocol`

The decision head receives 26 explicit physics/candidate features, six AIS inputs, and the 48 weather values. The AIS values are projected candidate margin, maximum report age, lost-report count, fraction of admissible trims, additional required guard margin and a traffic indicator. For training, the simulator applied up to six legal first-trim alternatives from the same state and then used physics continuation. This produced 27,682 terminal time-to-mark labels across 4,614 states from 636 paired training cells. The head predicts a bounded correction to the physics score using differences from the reference trim. Failed episodes have a fixed 43,200 s training and evaluation cost. All four learned variants have the same parameter count, training schedule and action masks. The explicit, weather and AIS ablations zero the unused feature groups. Validation selected the score-mixing strengths before the held-out test. [Training receipt](training_receipt.json)

AIS selection uses report-derived geometry only. The common guard checks each full committed step against projected contacts with a 25 m gap, uncertainty allowance and a 3 m reserve. For reports older than 10 s, it executes 5 s steps with an additional bounded-displacement allowance. A separate retrospective sweep computes conservative lower bounds around analytical contact truth, truncating the final trim at the first capture. Both remaining truth-centered and report-centered margins must be positive to confirm clearance. Recorded report ages reached 60 s. Every applied path had positive audited margins. `Guard checks`

## Held-out results

| Scorer | Captures / 216 | Timeouts | Median paired delay vs physics | Minimum truth margin beyond full buffer + reserve |
| --- | ---: | ---: | ---: | ---: |
| Physics | 114 | 84 | — | +0.369 m |
| Learned explicit features | 108 | 90 | +170.4 s | +0.369 m |
| Weather embedding | 111 | 87 | +296.9 s | +0.369 m |
| AIS inputs | 105 | 93 | +279.9 s | +0.369 m |
| Weather embedding + AIS | 110 | 88 | +239.5 s | +0.144 m |

The 18 chart preflight outcomes are included in each denominator. Median delays use each arm's shared captures with physics, so those subsets can differ. Of the joint scorer's 110 shared captures, 19 were faster, 74 slower and 17 tied. Its mean time with the fixed failure penalty was 22,844 s against physics at 21,899 s. On the 144 learned-eligible cases, physics captured 106 and joint captured 102, with a median paired delay of 301.1 s. Physics had 1.549 m minimum truth margin in that subset. The tighter joint margin came from an Otranto 002–003 capture at wind from 275°, during sharp mixed traffic. [Full results and strata](test_summary.json)

| Held-out traffic | Physics captures / 108 | Joint captures / 108 | Median paired delay | Physics minimum margin | Joint minimum margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| Overtaking | 57 | 55 | +168.9 s | +0.369 m | +0.369 m |
| Sharp mixed dropout | 57 | 55 | +284.2 s | +1.315 m | +0.144 m |

Validation favored the joint correction on synthetic conditions, with 62 captures versus 59 for physics. That gain did not transfer. Adding AIS inputs to the weather-embedding scorer yielded one fewer capture and a higher mean penalized time on this split. All arms use the same AIS guard, so the ablation measures whether supplying AIS to the learned scorer adds value.

### Where the joint scorer lost captures

All four misses came on the longer Otranto route's 002–003 leg, with wind from 5° or 355°, under both traffic profiles. The physics scorer selected its initial trim toward positive easterly bearings. The joint scorer began turning the other way even though both trims passed the clearance mask. For 5°, replaying only that learned first trim and reverting immediately to physics reproduced both timeouts. In the sharp-dropout case, physics reached the 50 m circle in 19 min 36 s and the joint scorer's nearest logged fix was 134 m from 003. For 355°, reverting to physics after the first learned trim recovered both captures in about 20 min, showing that later learned choices also contributed. [First-trim counterfactuals](first_trim_diagnostic.json)

## Reproduce

From the V2 root in the simulator environment with Torch, NumPy and Matplotlib available:

```sh
source_study=data/results/learned_safe_mark_policy/reproduce
joint_study=data/results/learned_weather_ais_policy/reproduce
export PYTHONPATH="$PWD:$PWD/$source_study:$PWD/$joint_study:$PYTHONPATH"
python -B "$joint_study/make_splits.py"
python -B "$joint_study/train_weather_encoder.py"
python -B "$joint_study/train_joint.py" train
python -B "$joint_study/train_joint.py" test
python -B "$joint_study/diagnose_and_plot.py"
python -B "$joint_study/verify_joint.py"
```

The independent verifier reconciles all five arms against the split, all 52,697 applied trace segments and capture geometry to within 2 µm. It derives the episode timing and clearance summaries from the raw path records. The saved weights, counterfactual labels, validation runs and held-out trajectories are available in this study folder.
