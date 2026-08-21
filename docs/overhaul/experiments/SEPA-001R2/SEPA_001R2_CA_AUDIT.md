# SEPA-001R2 corporate-action audit

Policy remains `ca_sharecount_v1`: split / bonus / consolidation only.
**No factor is inferred from a price gap.**

## Ledger

| Field | Value |
|---|---|
| Source | NSE corporates API (`nse_corporates_api`), years 2019–2026 merged |
| Adjusting events | **607** |
| Symbols with share-count events | **462** |
| ABFRL events in ledger | **none** |
| Dividends | provenance only, not applied |
| Threshold | unresolved consecutive symbol rate ≤ 0.002 — **not lowered** |

## ABFRL (the SEPA-001R fail)

On the **adjusted** official series:

- Date: **2025-05-22**
- Close 268.95 → 89.85 (**−66.6%**)
- Consecutive session
- No parseable split/bonus/consolidation subject in the NSE CA feed for this symbol
- Treatment: **quarantine** (Option B). Not a fabricated 3-for-1 / demerger factor.

This is consistent with a demerger / restructuring discontinuity (Aditya Birla Fashion lifestyle separation around that date). Continuity cannot be reconstructed from a share-count ledger, so ABFRL is excluded from trend, VCP, and return research.

## Verify

`verify_ca_adjustment` is still **FAIL** on a mixed sample because other unresolved consecutive events remain. R2 does **not** claim `ca_complete=true`. Overall PIT class remains **`PIT_DEGRADED`**.

Quarantine uses `discontinuity_audit` classifications: `UNRESOLVED` consecutive events only. Genuine ≥35% market days and suspension spans are not treated as missing CA.

Unresolved events in the research payload list symbol, date, discontinuity %, classification, source, resolved=false, treatment=quarantine.
