# EDGE-001 — Data Integrity

EDGE-001 reuses official bhav OHLCV via `load_store_frames` → `get_ohlcv` (corporate-action adjustment **on read**). The store is not rewritten.

- Store sessions: `1788` (2019-08-23 → 2026-08-21)
- Frames loaded: `3126`
- Month-ends ranked: `72` (2020-09-30 → 2026-08-21)
- Primary complete holding periods: `70`
- Mean candidates / investable / ranked: 2351.7 / 1413.1 / 1246.8
- Listing identity: **PIT_DEGRADED_bhav_inferred** — membership is bars present ≤ T, not an official historical listing file.
- Sector map: **PIT_DEGRADED_contemporaneous_map** — today’s NIFTY 500 comment map applied historically. Concentration tables are descriptive only.
- CA policy: adjustment_on_read_plus_min_sessions; no exhaustive unresolved quarantine in EDGE-001. Resolved official events are already in the adjusted read path. EDGE-001 does **not** re-run the SEPA exhaustive unresolved gap audit; residual phantom gaps remain a PIT limitation.
- Nifty source: `NIFTY50_EQUALWEIGHT_PROXY_BHAV` (Nifty 500 official local series available: `False`).
- Fill: `next_open` (no same-close). Stop: `none_scheduled_rebalance_only`.
- Cost model: `core.costs.round_trip_cost_pct('CNC')` = 0.32 percent points per round-trip, applied to one-way turnover (`cost = one_way × rt_pct / 100`).
- Protocol SHA: `4a9ac87fc31bbd59` activated 2026-08-22T00:00:00+05:30.
- Official Nifty 50 local series: `586` sessions from `2024-04-08`. Too short for a full-sample official benchmark, so the primary Nifty comparison is the Nifty-50 equal-weight bhav proxy.

## Honest limitations

1. Survivorship: names that never appear in the 2019–2026 store cannot enter. `FastInvestable.snapshot` reuses a delisted/renamed name’s last print forever. EDGE-001 drops those stale last prints (`live_on_session`): a name must have an official bar **on T**. Mean stale drops are in `universe_snapshots.json` (`stale_dropped`). Ranked < investable: 1246.8 vs 1413.1.
2. FEATURE-001 / SEPA already mined this window. Confirmation 2025–2026 is held-out **for this protocol**, not philosophically pristine lifetime OOS.
3. Index period returns use the local Nifty close series (entry date ≤ t ≤ exit date), not a traded futures roll. Broad alternative is equal-weight investable.
4. Missing next-open drops the name from that month’s equal-weight (no invented fill).
