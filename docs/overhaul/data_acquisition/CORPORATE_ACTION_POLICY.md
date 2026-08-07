# Corporate-action & price-integrity policy

**Do not issue PASS or FAIL from materially distorted price histories.**

## What is detected / handled (data-quality gate + detector)

| Condition | Handling |
|---|---|
| Splits / bonuses | Adjusted ONLY if `logs/ca_events.json` present (`data/corporate_actions.adjust_frame`); else prices are RAW and flagged |
| Extreme discontinuities | `data_quality_report` counts unexplained >40% one-day moves (`ca_gap_anomalies`) — possible unadjusted CA |
| Duplicate sessions | Duplicate calendar dates → **fatal** (fail closed) |
| Missing prices | Counted; missing stays MISSING (never 0); a base spanning a gap is rejected (no fabricated candidate) |
| Inconsistent OHLC bars | `high<low` / `high<max(open,close)` / `low>min(open,close)` → **fatal** (fail closed) |
| Non-positive prices | **fatal** (fail closed) |
| Symbol transitions / merger-replacement | current-symbol keyed; unmapped transitions are a known limitation (needs symbol-change master) |

## Adjustment status → trust

- **RESEARCH_GRADE prices** require a real corporate-action ledger (`ca_events.json`)
  sourced from NSE CA filings. Without it, prices are **RAW** and cannot back a PASS.
- The gate **fails closed** on proven corruption (duplicates, non-positive, HLOC
  inconsistency) rather than producing a distorted verdict.

## Limitation direction classification (drives the verdict gate)

Each active limitation is classified by the direction it biases the primary hypothesis:

| Limitation | Direction | Why |
|---|---|---|
| Survivorship incomplete | **FAVOURABLE** | Today's survivors inflate returns → the edge looks *better* than reality |
| Corporate actions RAW (unadjusted) | **EITHER** | Phantom split/bonus gaps fabricate *both* fake breakouts (favourable) and fake stop-hits/breakdowns (unfavourable) |
| Valuation unavailable | **NEUTRAL** | Context only; never a primary input |

## Verdict rule under incomplete data (enforced in `runner._decide`)

- A would-be **PASS** on non-research-grade data (survivorship-incomplete OR CA-raw) is
  **downgraded to INCONCLUSIVE** — a biased PASS is not defensible.
- An economic **FAIL** is retained **only** when every active limitation is
  one-directional **FAVOURABLE** (or NEUTRAL/NONE) — i.e. the data was biased *toward*
  the hypothesis and it still failed. If any limitation is **EITHER / UNFAVOURABLE /
  UNKNOWN** (e.g. CA-raw), a FAIL could be a data artefact → **INCONCLUSIVE**.

This is a verdict-interpretation safety rule; it changes no EXP-006 threshold, feature,
pivot/base definition or config hash.
