# Brindisi reviewer reruns

The fresh `paper-v1` checkout produced 48 missions across four arms. Each arm uses the same 12 October cases and 50 m waypoint gates. The independent swept-clearance audits report zero violations among the 10 cases with applied motion per arm. Two cases per arm had no applied motion.

| Arm | Completed | Ordered captures | Minimum full AIS truth margin |
|---|---:|---:|---:|
| Compact ridge | 5/12 | 12/28 | +16.387 m |
| Escape | 5/12 | 12/28 | +16.387 m |
| Protected departure | 8/12 | 20/28 | +14.746 m |
| Causal hybrid forcing | 8/12 | 20/28 | +14.746 m |

The ridge saved a mean 36.03 s across five shared finishes against the [archived physics comparator](../../results/temporal_embedding_native/native/physics/). Hybrid forcing took a median 25.17 s longer than protected departure across eight shared finishes. The hybrid audit uses the [archived protected-departure comparator](../../results/brindisi_escape_departure/native/departure/). The fresh departure replay reproduced its sailing outcomes.

The three study folders contain per-arm episode tables, input and run receipts, 48 compressed simulator traces, and paired comparisons. Independent clearance records are in `brindisi-ridge-audit/`, `brindisi-departure-audit/`, and the top-level `independent_clearance.json` for hybrid forcing. The run outputs preserve their original bytes. Each run receipt gives the SHA-256 hashes of its inputs, episode table and traces. The environment and resolved package versions are recorded alongside the outputs. Wall-clock decision times reflect the local host.

## Re-audit the traces

Follow the [simulator environment setup](../../../docs/RUN_SIMULATION.md#environment). From the checkout root, copy the recorded runs into three scratch subdirectories. The audit commands recompute swept clearance and write fresh receipts beside each scratch copy. They use the archived manifests and comparators in `data/results/`.

```sh
evidence=data/reviewer_reruns/brindisi
scratch=$(mktemp -d)
mkdir "$scratch/ridge" "$scratch/departure" "$scratch/hybrid"
cp -R "$evidence/brindisi-ridge" "$scratch/ridge/native"
cp -R "$evidence/brindisi-departure" "$scratch/departure/native"
cp -R "$evidence/brindisi-hybrid" "$scratch/hybrid/native"
python -B data/results/eta_ridge_closed_loop/reproduce/audit_ridge_loop.py \
  --workers 4 --native "$scratch/ridge/native"
python -B data/results/brindisi_escape_departure/reproduce/recovery_audit.py \
  --workers 4 --native "$scratch/departure/native"
python -B data/results/brindisi_hybrid_forcing/reproduce/audit_hybrid.py \
  --workers 4 --native "$scratch/hybrid/native"
printf 'Audit outputs: %s\n' "$scratch"
```

Each audit should report `verified` with an empty `unsafe_applied_cases` list.
