# PIT fundamentals coverage

**As of:** 2026-08-22 (Phase II)  
**Ledger:** `logs/pit_fundamentals.json`  
**Status:** dataset **RESEARCH_READY_WITH_LIMITATIONS**

Yahoo / Screener `.info` remain **UNUSABLE** as historical truth.

---

## Symbols / time

| Item | Value |
|---|---|
| Symbols with research-grade PIT rows | **2057** (65% of 3156 bhav names; 90% of 2293 official EQ) |
| First available date | 2019-04-10 |
| Last available date | 2025-06-17 |
| Rows | 19151 |
| Median quarters / company | 9 |
| ≥4 / ≥8 / ≥12 quarters | 1571 / 1261 / 32 |
| Restatement pairs | preserved by `available_at` + `row_id` (not overwritten) |

Ingest: NSE corporates-financial-results broadcast time → `available_at`; NSE XBRL → statement fields. Parser prefers context `OneD`.

---

## Field confidence (this ledger)

| Field | PIT class | Notes |
|---|---|---|
| revenue, PBT, PAT, basic/diluted EPS | FUNDAMENTAL_PIT_STRONG | Official XBRL + broadcast date + raw hash |
| other income, face value, paid-up equity | FUNDAMENTAL_PIT_STRONG | Present on nearly all rows |
| debt/equity (XBRL ratio) | FUNDAMENTAL_PIT_DEGRADED | 5432 / 19151; definition varies |
| operating profit | FUNDAMENTAL_MISSING | 0 rows; tags not in these instances |
| EBITDA / EBIT / CFO / capex / FCF | UNUSABLE | not mapped |
| revenue / PAT growth, margins | PIT_STRONG when both periods known by T and alignment is YoY or QoQ | Nine-month ≠ quarter |
| ROE (paid-up equity) | PIT_DEGRADED | not book equity |
| ROCE, CFO/PAT | UNUSABLE | until cash-flow + capital-employed ledger |

---

## Restatements

Schema keeps original and later restated rows. `get_period_as_of(symbol, period_end, as_of)` returns the version **known at T**.

---

## Source distribution (on disk)

| Source | Rows | Role |
|---|---|---|
| nse_xbrl / nse_xbrl_financial_results | 19151 | research source |
| Yahoo / Screener | not ingested | forbidden as historical PIT |
