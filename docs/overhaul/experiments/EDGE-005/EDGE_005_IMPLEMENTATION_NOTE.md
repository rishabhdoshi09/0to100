# EDGE-005 — Implementation Note

Written before results. FEATURE-002 frozen. EDGE-001–004 not retuned.

## Why this hypothesis

George–Hwang (2004): proximity to the 52-week high predicts returns, distinct from raw past return. QuantTerm uses distance-from-high as a **demote** on scanner fires, not as a standalone CS book. EDGE-001 ranked 12-1 *return*; a name can have mediocre 12-1 and still sit on a 52-week high.

## Reuse

Same universe, same-session print, next-open, CNC costs, month-end calendar, EW bench.

## Signal (locked)

P1 = close[T] / max(last 252 closes ≤ T). Score = P1. Top 20.

## PIT

Max window uses bars ≤ T only. Append-future at the same j must not change P1.
