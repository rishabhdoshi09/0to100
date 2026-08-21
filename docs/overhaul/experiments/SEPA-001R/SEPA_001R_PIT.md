# SEPA-001R PIT / data integrity

**Overall classification: `PIT_DEGRADED`**

This run must not be labelled PIT-safe.

## Price integrity

- Source: official NSE bhavcopy, `get_ohlcv` on-read, sliced `index <= as_of`.
- Store: **2751** symbols, **424** sessions, latest **2026-08-21**.
- Research book after screen: **100** names (from 2660 store symbols with enough bars to rank).
- Unresolved consecutive-session gaps **in the passed book**: 0 (gap names excluded).
- This does **not** prove CA completeness for the whole store.

## Corporate-action integrity

| Field | Value |
|---|---|
| Source | NSE corporates API (`nse_corporates_api`), share-count events only |
| Policy | `ca_sharecount_v1` (split / bonus / consolidation). Dividends provenance-only, not applied |
| Coverage | 2024-01-02 → 2026-08-21 |
| Symbols with events | 254 |
| Adjusting events | 290 |
| Ledger hash | `e260673881d9e5c3` |
| `verify_ca_adjustment` | **FAIL** — `gap_rate=0.0125` (threshold 0.002), flagged `ABFRL` |
| `ca_complete` | **false** |
| Never invents | true — no gap-inferred factors |

38 symbols were excluded from the research book for unresolved CA gaps.

## Universe integrity

| Field | Value |
|---|---|
| Membership source | `bhav_inferred` (local first/last session) |
| Official listing archive | **absent** |
| `research_grade` | false |
| Class | `PIT_DEGRADED` |
| `survivorship_complete` in the ledger API | true — **do not treat as a research claim** |

Starting store **2660** → exclusions: short history 501, min price 194, min turnover 338, unresolved CA gap 38 → **eligible 100** (top turnover among remaining). Turnover distribution of screened names: p25 ₹12.2m, p50 ₹88.8m, p75 ₹406m, p90 ₹1.49bn (20-session close×volume).

## RS integrity

- Formula `rs_cs_v1`: `0.40*r63 + 0.20*r126 + 0.20*r189 + 0.20*r252`
- Cross-sectional percentile vs the as-of research book
- Fail-closed if any horizon is missing
- Appending future bars / extra names not in the as-of universe does not change historical ranks (`tests/test_sepa_001r.py`)

## Timestamp integrity

- Eligibility uses `frame.index <= as_of` only
- Swings are causal zigzag: extreme index vs `confirmed_index`; no fractal right-window on the money path
- Fill = next session open vs the versioned buy-zone; gap-through / missed / extended / invalidated are **not** market chases
- Tests: future bars cannot alter past JSON; confirmation cannot back-date a pivot

## Remaining limitations

- CA verification did not PASS — prices may still contain residual discontinuities outside the excluded set
- Universe is inferred from local bhav, not an NSE listing/delisting archive (delisted names that never hit this cache are missing)
- History depth is **424** sessions (~1.7y), 2025–2026 heavy
- Sector map is today's NIFTY500 comment groups — **not PIT**. Large `UNKNOWN` sector bucket
- Nifty regime series did not bucket this run (`unknown` for all fills) — measure, do not invent a regime gate
- Research book is the **100 most liquid remaining names**, not the full 2660

**Overall: `PIT_DEGRADED`**
