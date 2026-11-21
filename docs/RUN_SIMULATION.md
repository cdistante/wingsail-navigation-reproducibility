# Run the simulator

## Environment

Run commands from the checkout root. Python 3.12 was used for the archived Brindisi runs. The pinned packages reproduce the package versions checked by the stored run receipts. The reduced GRPO training environment uses a separate NumPy version.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

The Brindisi studies use the Fossen model and its CasADi discrete dynamics. The archived Otranto layline comparisons additionally use native Direct acados NMPC. To run them, build the acados shared libraries and install `acados_template` into the same Python environment. Set `ACADOS_SOURCE_DIR` and the shared-library path as described in the [acados Python installation instructions](https://docs.acados.org/python_interface/index.html#installation). Verify that the acados minimal OCP example builds and runs first. The archived Otranto receipts identify run settings and source hashes. Their wall-clock timings depend on the solver build, hardware, and concurrent workload.

## Offline smoke test

```sh
python -B scripts/smoke_simulation.py --output repro-output/smoke
```

The script generates a small chart and three waypoints, then runs Gamma guidance and Grid MPC against the native Fossen dynamics in constant wind. It verifies that the replay records waypoint 002 and waypoint 003 entering their 50 m capture disks in order. The output directory receives a GPX track, a compressed replay and the simulator log. Choose a fresh output directory for subsequent runs.

To exercise a second controller, repeat with `--controller acados-direct` or `--controller acados-los` after installing acados. To test the AIS and layline logic in isolation, run the included regression tests.

```sh
python -B -m unittest discover -s tests -p 'test_synthetic_ais.py'
python -B -m unittest discover -s tests -p 'test_ais_avoidance.py'
python -B -m unittest discover -s tests -p 'test_waypoint_laylines.py'
python -B -m unittest discover -s tests -p 'test_waypoint_handoff.py'
```

Run the full regression suite using `python -B -m unittest discover -s tests` after also installing `pyais` and `copernicusmarine`. Socket tests require a local network loopback. The optional GeoTIFF acquisition audit needs `rasterio`.

## October Brindisi

The archived manifests lock the 12 October missions, 50 m gates, synthetic traffic, report timing and separate departure dates. The run scripts check source, forcing and checkpoint hashes against the frozen comparison. Keep downloaded archive files in the layout in [Data provenance](DATA.md).

### Offline forcing and protected departure

The offline forcing verifier independently recomputes the vector RMSE and checks the five reproduced artifacts byte for byte. It refreshes a verification receipt in the study directory.

```sh
python -B data/results/stage3_forcing_offline/reproduce/verify_forcing.py
```

To produce fresh output for the protected-departure and causal-wind replays, use the native environment above. The hybrid runner takes its protected launch times from the staged persistence comparator.

```sh
python -B data/results/brindisi_escape_departure/reproduce/run_recovery.py \
  --method both --workers 4 --output repro-output/brindisi-departure
python -B data/results/brindisi_hybrid_forcing/reproduce/run_hybrid.py \
  --workers 4 --output repro-output/brindisi-hybrid
python -B data/results/brindisi_hybrid_forcing/reproduce/audit_hybrid.py \
  --workers 4 --native repro-output/brindisi-hybrid
```

These produce ordered episode records, per-case integration traces and run receipts. The hybrid audit reconstructs the full AIS clearance from the applied dynamics and the synthetic truth tracks, then pairs the outcomes against the frozen protected-departure run. Timing and receipt digests that include timing or platform fields can differ between fresh runs.

### Compact counterfactual Ridge selector

Run the fixed-state evaluation using the released training and calibration branch labels, then evaluate the included 11-feature checkpoint in native execution.

```sh
stage=data/results/eta_stage1_expanded
train_sha=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["labels_sha256"])' "$stage/branches/train_receipt.json")
calibration_sha=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["labels_sha256"])' "$stage/branches/calibration_receipt.json")
python -B "$stage/reproduce/fit_stage1.py" \
  --train "$stage/branches/train.jsonl.gz" --train-sha256 "$train_sha" \
  --calibration "$stage/branches/calibration.jsonl.gz" --calibration-sha256 "$calibration_sha" \
  --test-sha256 832e18f50ba417425a0c2565b0136fc04e540c6ffc32f8fc64501dc1bf66a9de \
  --output repro-output/stage1-fit
python -B data/results/eta_ridge_closed_loop/reproduce/run_ridge_loop.py \
  --workers 4 --output repro-output/brindisi-ridge
```

Use `run_stage1.py --output repro-output/stage1-branches --workers 4` to start generating matched Fossen branches from the seasonal archive. The shipped fixed-state comparison uses the included fixed October branch labels and the 25 s corrected ETA threshold.

### GRPO and GRPODropout

The six- and twelve-rollout experiments use identical held-out cases and the same native simulator. To rerun both policies at both group sizes using their released checkpoints, run the following in the native environment.

```sh
for method in grpo_progress grpo_dropout; do
  python -B data/results/grpo_dropout_native/reproduce/run_dropout_native.py \
    "$method" --workers 4 --output "repro-output/grpo6/$method"
  python -B data/results/grpo_group12_native/reproduce/run_group12_native.py \
    "$method" --workers 4 --output "repro-output/grpo12/$method"
done
python -B data/results/grpo_group12_native/reproduce/audit_group12_clearance.py \
  --native repro-output/grpo12 --output repro-output/grpo12-clearance.json
```

Use `--scenario autumn__20261012T0600Z__brindisi_small__test_oblique_fan` and a fresh output path to run a single case. To repeat training, make a Python 3.12 environment with `requirements-training.txt`, and set `PYTHONPATH` to the checkout root. The training entry points check the archived synthetic roster and package versions.

```sh
python -B data/results/grpo_progress_native/reproduce/train_progress_grpo.py \
  --output repro-output/train6-grpo
python -B data/results/grpo_dropout_native/reproduce/train_grpo_dropout.py \
  --output repro-output/train6-dropout
for method in grpo_progress grpo_dropout; do
  python -B data/results/grpo_group12_native/reproduce/train_group12_grpo.py \
    "$method" --output "repro-output/train12-$method"
done
```

## Otranto ETA-gated laylines

These four runs use identical launch and underway traffic, 20 s AIS report delay with staged and underway dropouts, 50 m gates, COLREG, archived marine fields and native Direct acados NMPC within each wind-bearing pair. The launcher validates the recorded invocation and simulator source hashes. It builds a temporary workspace that restores the exact archived versions of the three simulator modules that changed between studies.

```sh
python -B scripts/rerun_otranto_laylines.py --wind 5 --arm eta \
  --output repro-output/otranto-eta-5 --check-command
```

Omit `--check-command` to execute the run. Repeat for `--wind 355` and for `--arm off`. Use a fresh directory for each run. The logged outputs contain the accepted navigation track, capture events and AIS timing. Compare with the [saved results](RESULTS.md). Verify the supplied regional fields with the procedure in [Data provenance](DATA.md).

Inspect a saved or recreated navigation replay on a machine with a graphical display.

```sh
python -B -m world.navigation_replay repro-output/smoke/navigation.jsonl.gz
```

To export an MP4, install FFmpeg and add `--export-video repro-output/smoke.mp4`. A graphical window is optional when exporting.

The included study scripts document earlier experimental workflows and link to their original protocol files. The selected commands above use the staged fixtures and receipts. The archive includes precomputed comparisons for additional experimental arms listed in [Results](RESULTS.md).
