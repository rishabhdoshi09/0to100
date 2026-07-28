# EXP-006 — experiment run records (append-only)

Immutable, auditable record of every EXP-006 evidence-run attempt. **Nothing here is
overwritten.** Each run has a distinct id under `runs/<run-id>/` and keeps a
`run_manifest.json` (run id, code commit, config hash, dataset snapshot id, start/
completion, data-quality status, verdict, artifact SHA-256 hashes) plus its artifacts.

Raw market data is never stored here (git-ignored, non-redistributable). Only derived
provenance + verdicts are committed.

## Runs

| Run id | Status | Verdict | Snapshot | Code commit | Notes |
|---|---|---|---|---|---|
| `0001-blocked` | BLOCKED before candidate generation | INCONCLUSIVE — DATA_UNAVAILABLE | `ad652107580ddae1` | `a634be3` | No NSE data / no network; economic hypothesis UNEVALUATED. Relocated from `docs/overhaul/exp006_run/` (committed `6a865c8`); content unchanged. |

## How a future real run is recorded

1. Satisfy the EXP-006 readiness gate (see `../../data_acquisition/README.md`).
2. Run the unchanged frozen runner:
   `python -m research.momentum_breakout.runner --out logs/experiments/EXP-006`.
3. Copy the produced artifact set into a NEW `runs/<next-id>/`, write its
   `run_manifest.json`, and add a row above. Never edit an existing run.
