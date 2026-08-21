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

1. Candidates = names with at least one official bar ≤ as_of (inferred membership)
2. Slice OHLCV through as_of
3. Min 260 sessions as-of
4. Min close ₹20 as-of
5. Trailing 20-session turnover ≥ ₹5m as-of
6. Drop CA-quarantined names
7. Canonical RS denominator = **full investable set** (`top_n=None`)

Membership source is still `bhav_inferred` (`logs/universe_history.json`). There is **no official listing/delisting archive** in the repository. `point_in_time_universe` `survivorship_complete=true` on an inferred ledger is **not** a research claim.

**Class: `PIT_DEGRADED`.**

Delisted names that never entered this bhav cache are missing (survivorship bias remains, favourable to survivors).

First eval date investable count (post 252-session warm-up, 2020-09-16): **1714** names vs SEPA-001R’s frozen 100.
