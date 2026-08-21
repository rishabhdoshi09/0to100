# SEPA-001R2 universe audit

## Was SEPA-001R look-ahead biased?

**YES.**

`research/sepa/universe_screen.py` `load_research_frames()` screened price, 20-day turnover, and top-N using each frame’s **last bar** (2026), then `ablation_r.run_ablation_r` ranked RS against that frozen 100-name book on every historical date.

Tests in `tests/test_sepa_001r2.py`:

- future prices cannot change investable-at-T
- future turnover cannot pass a historical screen
- top-N liquidity is as-of
- future listings cannot change historical RS ranks

## R2 construction

At every `as_of_date`:

1. Candidates = names with at least one official bar ≤ as_of (inferred membership).
   **Not** every symbol that exists somewhere in the 2019–2026 store. A name
   first listed in 2025 is not a 2021 candidate, is not hashed into
   `membership_hash`, and is not counted as `no_bars`.
2. Slice OHLCV through as_of
3. Min 260 sessions as-of, counted on the **clean CA segment** (pre-event
   history is kept; post-event names need 260 sessions *after* the
   unresolved discontinuity)
4. Min close ₹20 as-of
5. Trailing 20-session turnover ≥ ₹5m as-of
6. Unresolved CA is **date/segment-aware**. A 2025 demerger does not delete
   the name from 2021–2024 universes. Static `quarantine_symbols` is an
   event catalogue, not an as-of filter.
7. Canonical RS denominator = **full investable set** (`top_n=None`), using
   only clean-segment closes. A future unresolved CA cannot change
   historical RS ranks.

`FastInvestable.snapshot` and `screen_investable_as_of` must agree on
investable names, candidate set, candidate hash, and exclusion counts when
given the same as-of membership (tested in `tests/test_sepa_001r21.py`).

R2.1 also fixes a FastInvestable date-unit bug: on pandas builds where
`DatetimeIndex` is `datetime64[us]`, comparing `idx.asi8` to
`Timestamp.value` (always ns) made every as-of resolve to the **last** bar.
Dates are now stored as `datetime64[ns]` epoch integers.

Membership source is still `bhav_inferred`.

**Class: `PIT_DEGRADED`.**

Delisted names that never entered this bhav cache are missing (survivorship bias remains, favourable to survivors).

First eval date investable count is produced by the R2.1 daily run and
reported in `SEPA_001R2_RESULTS.md`.
