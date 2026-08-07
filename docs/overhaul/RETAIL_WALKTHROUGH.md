# QuantTerm — Retail Experience Walkthrough

Companion to `UI_AUDIT.md`. Shows the default retail experience for four scenarios, with values
produced by the canonical product layer (`product/`, `data/fno_universe.py`). A rendered version is
`docs/overhaul/retail_ux_demo.html`.

The retail layer is a **read-only projection**: `product/projection.build_product_state` turns
backend facts into a plain-language home (`headline`, `readiness`, `activity`, `primary_action`,
`useful_actions`, `setup_steps`, `attention`); `data/fno_universe` derives the F&O universe from the
instrument master and runs the underlying-first funnel. Neither recomputes momentum, F&O eligibility,
backtests, risk or execution — those stay in the engine.

## Scenario A — new user, no data
`build_product_state(ProductInputs(kite_connected=True))` → primary action **Update Market Data**;
setup steps: *Connect Zerodha* (Ready) → *Download market history* (Not ready) → verify → backtest →
enable. The blank state is replaced by this guided journey.

## Scenario B — market closed, data ready
Headline **“Market closed — Research mode active”**; primary **Run Backtest**; useful actions:
Run Backtest · Find Momentum Stocks · See F&O Momentum · Prepare Tomorrow's Watchlist · Review Paper
Trades · See What QuantTerm Learned. The backtest reads as a plain verdict (₹1 lakh grew/fell, trades,
win/loss, largest fall, costs, vs Nifty, trustworthiness) with scientific stats behind a details flag.

## Scenario C — market open
Readiness *Ready for automatic paper trading*; the Automatic Paper Trading page states **“No
per-trade approval is required”** and offers pause/close controls. A data or safety problem flips the
mode to paused automatically — the projection never presents RUNNING through a failed gate.

## Scenario D — F&O Momentum funnel (from `data/fno_universe`)
For an instrument master with 6 stock-future underlyings (one unmappable) + an index future:

```
Current listed F&O underlyings: 6      (unique_stock_underlyings)
Successfully mapped to NSE cash: 5      (MISSING → canonical_mapping exclusion)
Data ready: 4
Momentum evaluated: 4
Momentum qualified: 3
Displayed: 3
```

Recorded exclusions (nothing silent): `MISSING` — canonical_mapping (“could not be mapped to an NSE
cash equity”); `WEAKO` — history (“Insufficient history: 20 sessions available, 60 required”);
`HDFCBANK` — momentum (“Momentum conditions not met”). Qualified: RELIANCE, INFY, TCS (expiry
2026-08-27, lot 500). Momentum is computed on the **underlying equity**; futures are a confirmation
layer, and user filters change only the displayed set, not the evaluated universe.

## Acceptance
`tests/test_product_ux.py` and `tests/test_fno_universe.py` cover the market-closed home, guided
setup, visible backtest, jargon map, dynamic F&O universe, funnel counts, recorded exclusions, and
filters-after-evaluation. Full suite: **894 passed** (RSS `feedparser` import made optional so the
news module degrades gracefully in test/CI instead of breaking collection).
