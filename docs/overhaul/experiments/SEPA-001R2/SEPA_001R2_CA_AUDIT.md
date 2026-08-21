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

This is consistent with a demerger / restructuring discontinuity (Aditya Birla Fashion lifestyle separation around that date). Continuity cannot be reconstructed from a share-count ledger.

## R2.1 causal segments (replaces static symbol quarantine)

R2 initially converted unresolved events into a **static symbol set** and
supplied that set to every historical as-of. That leaked future information:
ABFRL’s 2025-05-22 gap would have removed ABFRL from 2021–2024 universes.

R2.1 treatment is date/segment-aware (`research.sepa.ca_audit.CATimeline`):

| Window | Treatment |
|---|---|
| Strictly before D | Observations remain valid if lookback and forward outcome do not cross D |
| Forward path that includes D | `CA_CENSORED_OUTCOME` — excluded from expectancy, counted in the funnel. No fabricated through-gap return |
| At/after D | Indicators may only use bars **after** D. Re-entry requires 252+ clean post-event sessions (Stage-2 / RS) plus VCP lookback |

No adjustment factor is inferred. Pre-event history is not deleted.

`ca_complete` remains **false** while `verify_ca_adjustment` fails. A separate
`ca_research_acceptable` flag may be true when every unresolved event is
enumerated, contaminated paths are censored, and the audit is persisted.
That flag does **not** rewrite the global verifier.

Unresolved events in the research payload list symbol, date, discontinuity %,
classification, source, resolved=false, treatment, and clean segment
start/end.

## Verify

`verify_ca_adjustment` is still **FAIL** on a mixed sample because other
unresolved consecutive events remain. R2.1 does **not** claim
`ca_complete=true`. Overall PIT class remains **`PIT_DEGRADED`**.

The global verifier sample is a documented secondary diagnostic. It does
**not** certify the study. Canonical CA unresolved-event audit is exhaustive
over the research-relevant store.

Quarantine uses `discontinuity_audit` classifications: `UNRESOLVED`
consecutive events only. Genuine ≥35% market days and suspension spans are
not treated as missing CA.
