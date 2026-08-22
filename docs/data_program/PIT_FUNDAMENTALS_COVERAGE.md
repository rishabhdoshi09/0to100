# PIT fundamentals coverage

**As of:** 2026-08-22  
**Ledger:** `logs/pit_fundamentals.json` — **absent**  
**Status:** schema `RESEARCH_READY_WITH_LIMITATIONS`; dataset **DESCRIPTIVE_ONLY** until official ingest.

Yahoo / Screener `.info` and the live `fundamentals/` cache are **UNUSABLE** as historical truth. They answer “what does the website say now?”, not “what was public as of T?”.

---

## Symbols / time

| Item | Value |
|---|---|
| Symbols with research-grade PIT rows | **0** (0% of ~3156 bhav names; 0% of NSE EQ) |
| First available date | none |
| Years covered | 0 |
| Quarters covered | 0 |
| Restatement pairs on file | 0 |

The ingest path (`data.nse_results_ingest`) already maps NSE financial-results broadcast time → `available_at` and XBRL tags → statement fields. It was **not** executed under this mandate (network ingest is a separate stage; historical loops must not fetch).

---

## Field confidence (architecture, not this empty ledger)

| Field | PIT class if sourced from NSE XBRL + broadcast time | If sourced from Yahoo/Screener live cache |
|---|---|---|
| revenue, PBT, PAT, basic/diluted EPS | PIT_STRONG | UNUSABLE |
| other income, face value, paid-up equity | PIT_STRONG | UNUSABLE |
| debt/equity (XBRL ratio) | PIT_DEGRADED (definition varies) | UNUSABLE |
| EBITDA / EBIT / CFO / capex / FCF | UNUSABLE (not in current XBRL map) | UNUSABLE |
| revenue growth, EPS growth, PAT margin | PIT_STRONG when both periods known by T | UNUSABLE |
| ROE (paid-up equity denominator) | PIT_DEGRADED | UNUSABLE |
| ROCE, CFO/PAT, FCF margin | UNUSABLE until cash-flow ledger | UNUSABLE |

Do not label the dataset PIT_STRONG. Empty ledger ⇒ nothing is research-grade yet.

---

## Restatements

Schema keeps original and later restated rows (distinct `available_at` / `seq_id` / `source_hash`).  
`get_period_as_of(symbol, period_end, as_of)` returns the version **known at T**. Tests prove a 2025-02-10 restatement cannot change a 2024-12-01 read.

---

## Source distribution (on disk)

| Source | Rows | Role |
|---|---|---|
| nse_xbrl / nse_financial_results | 0 | intended research source |
| operator ingest | 0 | allowed if `available_at` is honest |
| Yahoo / Screener | not ingested here | forbidden as historical PIT |

---

## Next action (not started)

Offline (or operator) ingest of NSE corporates-financial-results + XBRL into `logs/pit_fundamentals.json`. Refresh must create a new file/version, not rewrite a snapshot already cited by an experiment manifest.
