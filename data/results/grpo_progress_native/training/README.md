# Progress-reward GRPO training

This record describes the original experiment workspace. The [reviewer evidence index](../../../../docs/RESULTS.md) lists the released files and verification commands.

This arm starts from the previous physics-initialized residual actor and
uses the frozen reduced Fossen scenario bank. Training uses 96 synthetic
cases. Checkpoint selection uses 32 separate synthetic cases, each with
and without the chart exclusion disk. The native routes, weather and AIS
cases do not enter this process.

At each applied interval, let \(d\) be the remaining distance to the
waypoint circle, with \(d=0\) at first-circle capture, and \(d_0\) the
initial gap. The potential is \(\phi=\mathrm{clip}(1-d/d_0,-1,1)\).
The episode reward includes 3,600 seconds times the signed changes in
\(\phi\), discounted by \(0.985\) per 300 elapsed seconds. Moving away
reduces the progress reward. The terminal reward is negative capture
time for a finish and \(-43{,}200\) seconds at timeout. Absence of a
chart-admissible action adds a 7,800-second penalty. The chart mask and
within-interval first-entry check are shared with the prior study.

Every seed trains for 64 updates with three same-start groups per update
and six trajectories per group. The optimizer uses group-standardized
advantages, clips likelihood ratios and penalizes KL divergence from the
physics prior. Checkpoints are scored every 16 updates by capture count
and then mean capped elapsed time on validation. The best
post-update result was locked before the native comparison.

| Seed | Groups with differing terminal returns | Groups with differing shaped returns | Selected update | Validation captures |
| --- | ---: | ---: | ---: | ---: |
| 7309 | 35/192 | 192/192 | 16 | 30/64 |
| 8779 | 33/192 | 191/192 | 32 | 30/64 |
| 10921 | 37/192 | 192/192 | 48 | 31/64 |

Seed 10921, update 48, is the native comparison checkpoint. It captured
31/64 validation marks with mean capped time 23,438.20 s. Physics captured
30/64 with mean capped time 24,051.70 s. Both had 33 timeouts. The chart
blocked one physics case and zero selected-checkpoint cases. The other
two training seeds did not improve on their physics initialization.
The reward variation counts above compare two return definitions on
the same sampled training trajectories. Replaying validation with the
exported checkpoint reproduced the reported metrics using both PyTorch
and NumPy inference.

- [Training script](../reproduce/train_progress_grpo.py)
- [Selected checkpoint](grpo_progress_selected.npz)
- [Training receipt and split identifiers](training_results.json)
- `Training log`

Run from the project root with the simulator's Python dependencies and
PyTorch. Use a fresh output directory to repeat the archived run.

~~~sh
python -B data/results/grpo_progress_native/reproduce/train_progress_grpo.py \
  --output /tmp/maritime-grpo-progress-repeat
~~~
