# Conviction Buying and Long-Term Picks

## Purpose

QuantTerm now exposes two separate retail decision pipelines. They share data
infrastructure but must not be interpreted as the same strategy.

### Conviction Buying

Source of truth: the canonical whole-market scan in
`logs/product/latest_momentum_scan.json`.

The retail projection combines:

- scanner score and verdict,
- breakout/entry status,
- volume confirmation,
- RSI room and chase risk,
- current market health,
- leading/lagging sector context.

Output states are `HIGH_CONVICTION`, `AWAIT_CONFIRMATION`,
`WAIT_FOR_PULLBACK`, and `WATCH`. The page is read-only/control-only. “Run
fresh scan” enqueues the existing supervisor-owned market scan. No label is an
order and LIVE execution remains locked.

### Long-Term Picks

The long-term technical pre-screen reuses `scan/long_term.py` and official
bhavcopy history. The final retail classification is produced by
`scan/long_term_service.py` using current cached/refreshed fundamentals:

- ROCE and ROE,
- three-year sales and profit growth,
- debt/equity and interest coverage for non-financial companies,
- operating-cash-flow conversion for non-financial companies,
- promoter holding and pledge,
- current P/E valuation context.

Missing fields reduce explicit coverage. A row with insufficient current
fundamental coverage becomes `NEEDS_FUNDAMENTALS`; it cannot become a quality
pick. Severe debt, pledge, profitability, or coverage failures become
`AVOID_REVIEW`.

The current classifications are:

- `QUALITY_COMPOUNDER`
- `GARP_CANDIDATE`
- `QUALITY_BUT_EXPENSIVE`
- `LONG_TERM_WATCH`
- `NEEDS_FUNDAMENTALS`
- `AVOID_REVIEW`

Current fundamentals are not publication-dated. They are never passed into
historical backtests or represented as point-in-time evidence.

## Ownership

- Streamlit reads saved results and writes durable owner controls.
- `quantterm-autonomy` performs scans, optional fundamental refreshes,
  Telegram notification, and tracking mutations.
- Automatic long-term review is scheduled in the Friday EOD window.
- Manual controls are idempotent jobs in the existing autonomy ledger.

## Controls

- `RUN_SCAN_NOW`
- `RUN_LONG_TERM_SCAN_NOW`
- `REFRESH_LONG_TERM_NOW`
- `TRACK_LONG_TERM_IDEA`

## Persistence

- Market scan: `logs/product/latest_momentum_scan.json`
- Long-term scan: `logs/product/latest_long_term_scan.json`
- Tracked long-term ideas: `logs/long_term.db`
- Telegram deduplication: `logs/autonomy/telegram_notifications.json`
