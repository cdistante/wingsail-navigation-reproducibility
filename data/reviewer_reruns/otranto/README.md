# Fresh Otranto reruns

These four replays are fresh native Fossen simulations from the published `paper-v1` input revision. They compare forecast-ETA layline selection with the same controller and protected-departure AIS/COLREG settings with that selection disabled. The capture radius was 50 m, the control interval was 5 s, and the clearance metric used the full AIS uncertainty envelope plus the 3 m reserve.

| Wind from | ETA layline selection | 002 after launch (s) | Finish after launch (s) | Captures in order | Min. truth-centered full clearance + reserve (m) | Min. report-centered full clearance + reserve (m) | Max. full control decision (s) |
|---:|:---:|---:|---:|:---:|---:|---:|---:|
| 355° | Off | 55,175 | 65,535 | 8/8 | 99.162130 | 99.162350 | 3.991668 |
| 355° | On | 55,175 | 65,170 | 8/8 | 99.162130 | 99.162350 | 3.996688 |
| 5° | Off | 40,835 | 47,415 | 8/8 | 132.374204 | 132.374429 | 4.004014 |
| 5° | On | 40,830 | 46,965 | 8/8 | 132.374204 | 132.374429 | 4.004097 |

`environment.json` and `installed-packages.txt` record the runtime. Each case directory contains the simulator log, replay, GPX and departure records, plus the independent gate, clearance and timing JSON and CSV outputs. All four audits confirmed waypoint order, positive swept clearance, the staged and underway AIS report-age checks, and the 5 s timing budget. The capture times, capture distances and clearance margins matched the archived comparison within the registered tolerances. The [frame comparison](frame_comparison.json) records frame counts and exact recorded ownship state and actuator values, aligned on simulation time, step, phase and waypoint. Wall-clock telemetry varies with the runtime.

## Repeat the independent audit

Activate the Otranto audit environment in [Run the simulation](../../../docs/RUN_SIMULATION.md) and run from the repository root. Use a fresh directory for each regenerated audit.

```sh
python -B scripts/audit_otranto_rerun.py \
  data/reviewer_reruns/otranto/otranto-355-eta \
  --wind 355 --arm eta --audit-dir repro-output/reaudit-355-eta \
  --require-archived-match
```

Repeat for the 5° runs and for the `off` arm. The script resolves the relative route and fixture paths against the checkout, binds the fixtures by hash, replays applied controls from each saved scenario, checks swept AIS clearance and compares the outcomes with the frozen archive. Pass `--process-exit-code 0` to include the original successful process exit in the regenerated audit receipt.

The replays, GPX tracks, departure records and independent CSVs preserve the original bytes. The simulator and acados smoke logs use portable paths. The receipts use checkout-relative evidence paths and record the SHA-256 of the distributed simulator log. [Original and distributed log hashes](normalized_log_hashes.json)
