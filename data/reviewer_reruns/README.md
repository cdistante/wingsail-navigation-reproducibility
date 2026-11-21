# Reviewer reproduction of the main experiments

The simulations used a fresh public checkout of `paper-v1` at `ea0fe5f7cbaf45aa80df1b46419afa167f2564f7`, cloned without signing in. All 1,612 release checksums passed. The headline verifier recovered the published comparisons from the archived records, including the 720-run sweep. Fresh runs covered four Otranto cases and 48 Brindisi missions.

Start with the linked per-case audits below. The [archived-result check](archived-result-check.txt) and [fresh Brindisi summary](fresh_brindisi_comparison.json) give the corresponding machine-readable counts and paired time differences. The [simulator guide](../../docs/RUN_SIMULATION.md) covers fresh execution.

## Otranto

The four loops captured outbound 002 → 003 → 004 → 005 and return 004 → 003 → 002 → 001 at 50 m. The fresh outputs match the archive in all 45,021 vessel state and rudder/flap command frames. The independent audit matched the capture times and distances and the swept minimum margins. Sailing times begin at launch after 290 s of protected staging.

| Wind from | Layline arm | Time to first 002 | Sailing time | Minimum truth margin | Minimum report margin | Max control decision |
|---|---|---:|---:|---:|---:|---:|
| 355° | [Off](otranto/otranto-355-off/audit.json) | 15:19:35 | 18:12:15 | +99.162 m | +99.162 m | 3.992 s |
| 355° | [ETA](otranto/otranto-355-eta/audit.json) | 15:19:35 | 18:06:10 | +99.162 m | +99.162 m | 3.997 s |
| 5° | [Off](otranto/otranto-5-off/audit.json) | 11:20:35 | 13:10:15 | +132.374 m | +132.374 m | 4.004 s |
| 5° | [ETA](otranto/otranto-5-eta/audit.json) | 11:20:30 | 13:02:45 | +132.374 m | +132.374 m | 4.004 s |

The margins include the full AIS uncertainty envelope and 3 m steering reserve. ETA gating saved 6 min 05 s at 355° and 7 min 30 s at 5°. Both 65 s underway AIS report-age peaks were verified for every arm. All measured departure and control decisions satisfied the 5 s budget on this host. The four Otranto reruns ran concurrently with one software thread per numerical library.

## Brindisi and offline forcing

| Arm | Missions completed | Ordered captures | Minimum full truth margin |
|---|---:|---:|---:|
| [Compact ridge](brindisi/brindisi-ridge/independent_comparison.json) | 5/12 | 12/28 | +16.387 m |
| [Escape](brindisi/brindisi-departure/independent_comparison.json) | 5/12 | 12/28 | +16.387 m |
| [Protected departure](brindisi/brindisi-departure/independent_comparison.json) | 8/12 | 20/28 | +14.746 m |
| [Causal hybrid forcing](brindisi/brindisi-hybrid/independent_comparison.json) | 8/12 | 20/28 | +14.746 m |

Each Brindisi arm had 10 cases with applied motion. Independent trace audits found zero clearance violations. Each measured decision satisfied the 5 s budget. Outcome rows matched the archive apart from wall-clock time. The ridge saved 36.03 s on average against the frozen physics comparator over the five shared completions. The causal hybrid added a 25.17 s paired median penalty against the freshly rerun protected-departure arm.

The offline forcing rerun regenerated all five output files byte for byte. October wind vector RMSE was 0.672 m/s for persistence and 0.545 m/s for causal trend. The [forcing receipt](forcing/verification.json) lists the artifact checksums.

## Reviewer guide correction

The original shared setup installed CasADi 3.8.2, which changed the early AIS escape maneuvers in Otranto. The corrected Otranto environment uses CasADi 3.7.2 and pandas 2.3.3 with acados v0.6.0. Brindisi and the offline forcing check use their separate CasADi 3.8.2 environment. The replay launcher checks the Otranto pins before starting. The guide includes the acados build, the independent fresh-run audit, and a read-only check of all 22 loaded regional weather, current and wave variables and coordinates.

The updated guide uses the public [paper-v1.1 tag](https://github.com/cdistante/wingsail-navigation-reproducibility/tree/paper-v1.1). A fresh signed-out checkout passed all 1,614 file checksums and the headline verifier. The benchmark and setup tag commits are recorded in [the checkout receipt](public-revision-refs.txt).

## Recheck the stored trajectories

Clone the `paper-v1.2` evidence tag and run commands from its root. The repository-wide `SHA256SUMS` includes the evidence files.

```sh
git clone --branch paper-v1.2 \
  https://github.com/cdistante/wingsail-navigation-reproducibility.git
cd wingsail-navigation-reproducibility
shasum -a 256 -c SHA256SUMS
python3 scripts/verify_reported_results.py
```

The Otranto case folders include the simulator logs, navigation replays, GPX tracks, and departure receipts. With the Otranto Python environment from the [simulator guide](../../docs/RUN_SIMULATION.md), the following command reconstructs the applied-motion audit from one published replay. Repeat with the other bearings and arms. Use a fresh output directory for each check.

```sh
python -B scripts/audit_otranto_rerun.py \
  data/reviewer_reruns/otranto/otranto-5-eta --wind 5 --arm eta \
  --audit-dir repro-output/published-otranto-5-eta --require-archived-match
```

The [trajectory comparison](otranto/frame_comparison.json) records the per-case counts of exactly matching state and command frames. The [Otranto evidence index](otranto/README.md) describes the per-case files and portable log paths. The [Brindisi evidence guide](brindisi/README.md) gives commands to re-audit the three study folders using scratch copies of the published traces.
