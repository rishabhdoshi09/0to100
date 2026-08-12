# A2 — Generic Multi-Horizon Target Framework

**Milestone:** Phase A / A2  
**Status:** Research label/split contract only — `ml/multi_horizon.py` live path unchanged  
**Authority:** `QUANTTERM_INSTITUTIONAL_AI_AUDIT.md`

## Goal

Stop treating one forecasting horizon as universally correct. Provide a
**leakage-safe** way to express and evaluate targets across many horizons.

## Reuse

| Need | Canonical source |
|------|------------------|
| Purged / embargoed CV | `research.harness.purged_kfold_indices` |
| Legacy 1d/5d/10d thresholds | `ml.multi_horizon._HORIZONS` (reproduced, not rewritten) |
| PIT bars | A1 `PitContract` / `Snapshot` (consumers supply close series) |
| Research harness | Unchanged — horizons are inputs, not a fork |

## What was added

- `research/horizons/` — `HorizonSpec`, `TargetSpec`, label builders, split helpers, capability catalog
- Docs + focused leakage tests
- Compatibility targets that reproduce `ml/multi_horizon` 1/5/10d label thresholds

## What was deliberately not done

- No mandatory simultaneous models for 5…252d
- No deep learning / ensemble
- No automatic promotion into scanner or live ML
- No rewrite of `MultiHorizonSignalGenerator` training loop (behaviour preserved)

## Leakage rules

1. Feature timestamp `t` → target realised at `t + horizon` (exit after entry).
2. Labels whose realisation window overlaps a test fold are **purged** from train.
3. Embargo after test folds defaults to the horizon length.
4. Overlap policy is explicit on `TargetSpec` (`purge` default).
