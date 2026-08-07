# Momentum Breakout — Feature Definitions (EXP-006)

Reference for every raw feature, component score, and data-quality flag the
framework produces. Code is the source of truth (`research/momentum_breakout/`);
this is the human index. All features are **point-in-time**: computed only from bars
available at the observation (breakout-close) bar `i`, via `pit.py` primitives.

## Group 1 — prior upmove & leadership (`features.prior_upmove_features`)
`ret_3m_pct`, `ret_6m_pct`, `ret_12m_pct`, `ret_12m1m_pct` (12-1),
`rel_to_bench_pct` (stock − Nifty return over the RS lookback), `dist_from_52w_high_pct`,
`dist_from_ath_pct` (only when ≥5y history), `dma50_slope_pct`, `dma200_slope_pct`,
`above_20ema` / `above_50dma` / `above_200dma`, `prior_upmove_max_dd_pct`.
Primary setup requires genuine leadership (positive rel-strength, above a rising
200-DMA); a stock emerging from a long downtrend does not qualify on one breakout bar.
→ **`leadership` score.**

## Group 2 — long-base structure (`features.base_features`)
`base_high`, `base_low`, `base_depth_pct`, `pivot`, `pivot_tests`,
`time_near_pivot_pct`, `range_contraction_ratio`, `atr_contraction_ratio`,
`realised_vol_pct`, `volume_dryup_ratio`, `tight_close_freq_pct`,
`upper_half_close_freq_pct`, `higher_lows`, `failed_breakout_attempts`,
`base_above_rising_200dma`, `base_near_52w_high`. The base detector is deterministic
and versioned; it never uses future bars to validate an earlier base. → **`base_quality` score.**

## Group 3 — breakout quality (`features.breakout_features`)
`close_above_pivot`, `breakout_dist_above_pivot_pct`, `breakout_clv`,
`breakout_range_atr`, `breakout_rvol`, `breakout_volume_z`, `gap_pct`,
`dist_from_20ema_pct`, `dist_from_50dma_pct`, `is_52w_high_breakout`,
`confirmed_close`, `extension_atr`, `overextended`. Signal fires only after the
breakout bar CLOSES above a pre-existing pivot. → **`breakout_quality` score.**

## Group 4 — structural stop & initial risk (`features.stop_candidates` / `select_structural_stop`)
Stop candidates: `swing_low`, `breakout_bar_low`, `tight_range_low`,
`short_lookback_low`, `pivot_minus_atr`, `base_support` (+ `selected_rule`). The
primary stop is the highest candidate below entry (tightest structurally justified
risk); `initial_risk_pct` and `initial_risk_atr` follow. Setups over
`max_initial_risk_pct` are rejected. → **`risk_efficiency` score.**

## Group 5 — sector strength (assembled point-in-time by the caller)
`sector_rs_pct`, `breadth_pct_above_50dma`, plus optional `n_leaders`, `turnover_cr`,
`membership_pit`. Primary hypothesis requires positive sector RS or breadth.
Membership is not historically dated → `SECTOR_MEMBERSHIP_NOT_PIT` flag.
→ **`sector_strength` score.**

## Group 6 — participation (`features.participation_features`)
`breakout_rvol`, `breakout_volume_z`, `base_volume_dryup_ratio`, `delivery_pct`,
`delivery_available`. Missing delivery is MISSING (flag `DELIVERY_DATA_UNAVAILABLE`),
never silently 0. → **`participation` score.**

## Trend extension (`features.trend_extension_features`)
`extension_atr`, `dist_from_50dma_pct`, `overextended`. → **`extension_risk` score**
(higher = more chase risk; subtracted in the combined ranking score).

## Group 7 — valuation CONTEXT (`features.valuation_features`) — never a primary reject
`pe`, `price_to_sales`, `ev_to_sales`, `market_cap_cr`, `sales_growth_pct`,
`earnings_growth_pct`, `pe_percentile_own`, `available`. Flags: `EXTREME_PE`,
`EXTREME_PRICE_TO_SALES`, `HIGH_EXPECTATION_RISK`, `VALUATION_DATA_STALE`,
`VALUATION_DATA_UNAVAILABLE`. Fails closed to UNAVAILABLE unless a real
`available_ts` proves the data pre-dates the observation; never forward-filled.

## Weakening / exit-state (`features.weakening_state`) — simulator only, pre-registered
`close_below_pivot`, `close_below_20ema`, `close_below_50dma`, `high_volume_reversal`.
Exit variants are pre-registered before the run; the simulator never peeks at the
whole trade to pick the best exit.

## Component scores & combined ranking
Transparent 0–100 per component (`leadership`, `base_quality`, `breakout_quality`,
`sector_strength`, `participation`, `risk_efficiency`, `extension_risk`). The
`combined_score` is for RANKING only, uses versioned weights, keeps raw components
available, and never hides a rejection reason.

## Data-quality flags (observation-level)
`SURVIVORSHIP_INCOMPLETE`, `SECTOR_MEMBERSHIP_NOT_PIT`, `VALUATION_DATA_UNAVAILABLE`,
`VALUATION_DATA_STALE`, `EXTREME_PE`, `EXTREME_PRICE_TO_SALES`,
`HIGH_EXPECTATION_RISK`, `DELIVERY_DATA_UNAVAILABLE`, `INSUFFICIENT_HISTORY`.

## Rejection reason codes (eligibility, features always kept)
`INSUFFICIENT_HISTORY`, `NO_BASE`, `WEAK_PRIOR_RS`, `NOT_ABOVE_RISING_200DMA`,
`NO_BASE_CONTRACTION`, `UNCONFIRMED_BREAKOUT`, `OVEREXTENDED_CHASE`,
`LOW_BREAKOUT_RVOL`, `STRUCTURAL_RISK_TOO_HIGH`, `NO_STRUCTURAL_STOP`, `WEAK_SECTOR`,
`ILLIQUID`, `PIT_VIOLATION:<detail>`.
