# SEPA-003 decay analysis

Winning era: first eligible date → 2023-12-31.  
Weak era: 2024-01-01 → 2026-08-21 (validation + **consumed** confirmation).

These eras are for **diagnosis**. They are not a new holdout.

## Verdict

**UNSTABLE_EDGE**

The strategy did not fail for one clean reason.

- The **market mix** changed: STRONG_BULL setups 318 → 41.
- The **calendar already flipped inside development**: 2022 E[R]=−0.027
  (reconstructed) / −0.096 (R2.1 official).
- Feature distributions (dry-up, final contraction, pivot distance, stop
  width, gap, RS) do not show a single large, consistent population
  shift that explains the collapse (Cliff’s δ generally small).
- Outcomes in the same RS / VCP bins still deteriorate later.

So: not purely “the market changed”, not purely “later VCPs were worse
objects”, and not a stable edge that simply needed a regime filter.

## Reconstruct n

| Era | n | Mapped sector % |
|---|---|---|
| Winning | 2,020 | 26.6% |
| Weak | 1,412 | 24.7% |

## What later setups were *not*

Later F fills were not systematically deeper, drier, or more extended
in a way that survives as a single smoking gun. Continuous VCP / entry
fields are UNSTABLE across years (see component survival).

Allowed labels used: UNSTABLE_EDGE. INCONCLUSIVE was the runner-up;
year-level sign flips were enough to prefer UNSTABLE_EDGE.
