# Live vs research parity

Long-term rule: any feature that influences a production trade must have a historically reproducible version **or** be marked non-backtestable.

This mandate does **not** modify production.

| Feature | Production use | Classification | Historical stand-in | Notes |
|---|---|---|---|---|
| Official OHLCV (bhav + CA adjust-on-read) | Scanner bars | same-source parity | `bhavcopy_store.get_ohlcv` / EvidenceSnapshot.prices | Raw store + CA ledger. |
| Intraday live overlay (Kite/NSE/Google) | Today’s price on cards | live-only | none | Non-backtestable. |
| Unified scanner signals (16 families) | BUY/WATCH | approximate parity | Replay scanner on as-of bhav | Intraday/news extras absent historically. |
| Close location / RSI / 52w-high demotes | Grade demote | approximate parity | Same formulas on as-of bars | 52w window is store-relative. |
| Conviction (news + Yahoo growth) | Top-40 enrichment | live-only | none | Yahoo growth ≠ PIT earnings. Non-backtestable. |
| Sector heat packs | Score nudge | approximate parity | STATIC_BACKFILL map | Not PIT sector history. |
| Institutional FII/DII / bulk deals | Tag only | live-only | none | Context; not a gate. |
| Breadth adv/decl, % above DMA | Brain demote | approximate parity | Full as-of cache | NARROW veto is live policy. |
| Live edge / EV engine | Rank / demote | live-only | signal_log outcomes | Path-dependent; not a PIT feature. |
| Regime engine | Posture | approximate parity | Official index CSVs from 2024-04-08 | Pre-2024 official Nifty short. |
| FEATURE-001 Trend / RS | Not in production rank | research-only | Frozen feature code | Forward-validated only via FEATURE-002. |
| FEATURE-002 R3 shadow rank | Isolated hook | research-only | Shadow ledger | Must not change production rank. |
| Screener/Yahoo deep fundamentals | UI | live-only / UNUSABLE historically | PIT fundamentals ledger (empty) | Do not backtest. |
| Autopilot / GTT / ticket | Execution | live-only | none | Telegram taps paper-only. |
| Sector index RS / rank | not in production | research-only | `data.benchmarks.sector_index_context` | Prepared; unused in trading. |
| Earnings surprise | not computed | impossible historically | — | No consensus series. |

Production trading remains unchanged.
