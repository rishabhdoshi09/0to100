# EDGE-002 — Implementation Note

**Written before** strategy results. FEATURE-002 remains frozen.

## Reuse (do not duplicate)

| Piece | Use |
|---|---|
| `FastInvestable.snapshot` + EDGE-001 `live_on_session` | Universe on T; drop stale last prints |
| `research/edge001/calendar.py` | Month-ends, next-open, costs-as-percent |
| `core.costs.round_trip_cost_pct("CNC")` | Sole cost model (0.32% RT incl. slippage) |
| `OpenCache` pattern | Next-open fills |
| `classify_regime_level` | H4 descriptive only |
| Nifty-50 EW proxy | Full-sample bench; official Nifty from 2024-04-08 overlay |
| Harness PSR/DSR/block-bootstrap | Monthly excess inference; DSR N_trials = 3 lookbacks × 4 sizes × 4 cadences = 48 |

## Not reused as the primary

- EXP-NEXT-02 (`research/phase_next/exp_lowvol.py`): 29-name panel, 20d vol, long-short quintile, 21d hold. Different object.
- EDGE-001 M1 ranks: not an input.
- FEATURE-002 shadow ranks: untouched.

## Execution (locked)

Rank at month-end close T using vol from bars ≤ T. Trade next session open. Exit at next rebalance’s next open. No stop. Equal weight. Long only.

## PIT limitations

Same as EDGE-001: listing PIT_DEGRADED, sector PIT_DEGRADED, CA adjustment-on-read, no exhaustive unresolved-gap re-audit. Same-session print required.

2024–2026 was used by EXP-NEXT-02 on 29 names. Confirmation is held-out **for this protocol**, not lifetime-pristine OOS.
