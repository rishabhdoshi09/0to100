# Common-man readability (non-negotiable)

QuantTerm may be institutionally rigorous internally, but **user-facing** copy
must stay understandable to a normal retail trader.

This document is the product contract. Implementation lives in
`product/plain_language.py` (presentation only). **Do not rename** internal
canonical fields in schemas, DBs, or research manifests merely to sound friendly.

```text
TECHNICAL TRUTH          USER PRESENTATION
(network_concentration) → "Portfolio overlap risk"
(evidence_level)        → "Research confidence"
(pit_state)             → "Historical data quality"
(trust_class)           → "Data quality"
```

## Every technical output needs four parts

1. Plain-language **label**
2. One-line **explanation**
3. Practical **implication**
4. Optional advanced detail behind **Why? / See details / Technical details**

## Two layers

| Layer | Shows |
|-------|--------|
| **1 — Normal user** | What is happening? Good/bad? How confident? What to watch? Why? |
| **2 — Advanced** | Correlations, centrality, Sharpe, FDR/DSR/PSR, PIT state, experiment ids, snapshot hashes |

Never remove technical information — progressively disclose it.

## Traffic lights (always with text)

Prefer states such as: `GOOD` / `CAUTION` / `RISKY` / `NOT_ENOUGH_DATA`,
or `STRONG` / `MODERATE` / `WEAK`, or `PROVEN` / `PROMISING` / `UNPROVEN` / `FAILED`.

Colour alone is not enough.

## No quant-dump dashboards

Order of presentation:

**CONCLUSION → REASON → WHAT IT MEANS → WHAT TO DO / WATCH → TECHNICAL EVIDENCE**

## Data quality examples (enums unchanged)

| Internal | Plain meaning |
|----------|----------------|
| `DISPLAY_ONLY` | Good enough for charts and exploration, not for proving a strategy. |
| `NOT_PIT_SAFE` | History may include facts not known at the time — not for serious backtests. |
| `RESEARCH_GRADE` | Passed checks required for scientific historical testing. |

## Navigation

User nav describes **jobs** (Home, Find Stocks, Portfolio, Market, Research,
Reports, Alerts, Assistant, Settings, Help) — not engineering concepts
(Feature Store, Evidence Graph, PIT, Gauntlet, Centrality, …).

## Beginner test

> Could a reasonably intelligent person with basic stock-market knowledge
> understand this in 10 seconds?

If no → simplify the **presentation**, not the **calculation**.

## Writing style

Short, clear, specific, non-academic, non-hype. Prefer “QuantTerm found…”,
“Historical evidence shows…”, “This has not been proven yet…”. Avoid
“AI predicts”, “guaranteed edge”, “institutional-grade algorithm says”.

## Related code

- `product/plain_language.py` — cards, trust/verdict/decision/metric helpers
- `product/projection.py` — home/readiness projection + `TERMINOLOGY`
- `frontend/src/productLanguage.ts` — web glossary / page guides
- Tests: `tests/test_plain_language.py`, `tests/test_product_ux.py`
