# FEATURE-001 — Results (explanatory)

**n events** = 120270 · **family rows** = 231145
**Primary hypotheses predeclared:** 5 · **Tests recorded:** 6 · FDR q = 0.1.

No result in this file is VALIDATED_EDGE.

## Hypotheses

| ID | stat | n | p | q | FDR reject |
|---|---:|---:|---:|---:|---|
| H1 | 0.166 | 18281 | 0.000 | 0.000 | True |
| H2 | 0.306 | 32816 | 0.000 | 0.000 | True |
| H3 | -0.068 | 231145 | 0.000 | 0.000 | True |
| H4_trend | 0.052 | 120270 | 0.000 | 0.000 | True |
| H4_rs | 0.071 | 120270 | 0.000 | 0.000 | True |
| H5 | 0.101 | 15 | — | 1.000 | False |

## Final feature status

- Trend (`trend_features_v1`): **FORWARD-VALIDATE AS RANK FEATURE**
- RS (`rs_features_v1` / `rs_cs_v1`): **FORWARD-VALIDATE AS RANK FEATURE**

See companion study notes for family cells and year splits.

Dataset first/last: `2020-09-28` → `2026-07-23`; Nifty bench available = True.

The 200MB+ event / family-row jsonl panels are **not in git**. Regenerate with `python -m research.feature001` (see `FEATURE_001_DATASET.md`). This window is already consumed — a regenerate is not new confirmation.
