# PIT fundamentals validation

**Date:** 2026-08-22  
**Ledger:** `logs/pit_fundamentals.json` (19,151 rows / 2,057 symbols)  
**Parser:** `qt_xbrl_core.v1` with **OneD-preferred** context selection  
**Sample:** deterministic stride of 80 rows (`sample_hash=7c559a317ddc6282`)

## What was compared

Fields: revenue_from_operations, profit_before_tax, profit_after_tax, basic_eps, operating_profit.

Primary check: re-parse the **same official XBRL instance** from the immutable cache and compare to the ledger.

Secondary pass: last-wins tag extraction on that same instance (not a vendor restatement). The ledger is **not** auto-repaired from last-wins.

No Screener/Yahoo values were written into history.

## Rates

| Measure | Value |
|---|---|
| Sample size | 80 |
| Compared with raw XBRL | 80 |
| Missing raw | 0 |
| Exact match (ledger vs OneD/first-policy reparse) | **100%** |
| Tolerance match | **100%** |
| Disagreement (ledger vs reparse) | **0%** |
| Unresolved ledger discrepancies | 0 |
| last-wins vs first/OneD tag differences | 281 field occurrences |

## Common causes

1. **NSE `FourD` YTD / cumulative** reuses the quarter’s start/end dates. Example: RELIANCE Q3 FY25 `OneD` revenue 2,438,650,000,000 vs `FourD` 7,155,630,000,000. Last-wins would treat nine-month revenue as the quarter. Parser prefers `OneD`.
2. **Multiple PAT tags** (period vs continuing operations) on the same context. Map keeps the first mapped local-name that wins under OneD rank.
3. **operating_profit** is unmapped in these instances (0/19151). Compared as both-missing.
4. **xbrl_download_failed** (2,870 anomaly-queue rows) — candidate listed, file not retrieved. Quarantined, not guessed.

## Unresolved

- No independent BSE/issuer-PDF sample was completed in this pass. Same-exchange instance reproducibility is 100%; cross-exchange numerical audit remains open.
- Context policy is now explicit; older document-order parses that already preferred OneD (document order usually lists OneD first) did not change row counts on rematerialize (19,151).
- Names absent from the NSE results dump (TATAMOTORS in this host’s raw JSON) cannot be validated because they were never ingested.

## Disposition

Do not silently replace OneD with FourD. Future researchers requiring `minimum_quality = FUNDAMENTAL_PIT_STRONG` get structured NSE XBRL + broadcast `available_at` + raw hash. They should still exclude YTD-labelled rows via `quarterly_usable`.
