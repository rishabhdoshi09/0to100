# A3 — Challenger Model Wiring

**Milestone:** Phase A / A3  
**Status:** Research bake-off infrastructure — **no live behaviour change by default**  
**Authority:** `QUANTTERM_INSTITUTIONAL_AI_AUDIT.md`

## Goal

Wire INCUMBENT vs CHALLENGER under identical research conditions through the
existing evidence spine:

experiment registration → provenance → train/val/OOS → costs → harness /
committee → verdict

## Reuse

| Need | Canonical source |
|------|------------------|
| Pre-registration / champion helpers | `research.registry` (`register_hypothesis`, `should_promote`) |
| Adversarial committee | `research.autonomy.challenge.promotion_committee` |
| Stats gate | `research.harness.evaluate` |
| Labels / splits | `research.horizons` |
| Costs | `core.costs.round_trip_cost_pct` |
| Failed evidence | `research.scientific_memory.record_negative` / `record_belief` |

## Deliberate non-actions

- Does **not** call `evaluate_challenger` for production roles by default (that
  helper mutates the champions table). Bake-offs use `should_promote` for the
  comparison and keep `persist_champion=False` unless an explicit research
  role is requested.
- Does **not** alter `ml/*` live inference or scanner scoring.
- Does **not** add deep learning, RL, or ensembles.
- Baseline models: naive majority + logistic regression (sklearn already present).

## Verdicts

`PROMOTE` | `KEEP_INCUMBENT` | `FAIL` | `INCONCLUSIVE`

`PROMOTE` means *research nomination only* — never automatic live cutover.
