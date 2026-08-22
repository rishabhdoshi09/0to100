# FEATURE-002 — Implementation Note (audit before code)

Written from repository inspection **before** the shadow hook was added.

---

## 1. Production decision path (where rank actually happens)

`scan/auto_scan.py` `_scan_once_locked` is the live cycle:

1. `UnifiedScanner.scan` → `list[StockSignal]` sorted by `score`
2. `_log_buys_for_tracking` / `_log_non_events_for_learning` (fail-open)
3. `_serialize` → dict cards (`signals` become **labels**, not keys)
4. `apply_sector_heat` (may change score, re-sort)
5. `build_conviction` (news/earnings on top-N — **not** used for FEATURE-002 features)
6. live quote overlay (may demote BUY→WATCH if price slipped)
7. `combo_edge` + proven-loser veto + re-sort by `(verdict_rank, score + edge_r*40)`
8. `tag_conviction` / `tag_ev` / decision journal
9. `tag_prime` / bulk-deal tag
10. Telegram `_push_new_setups` / `_push_breakout_confirmed`
11. `execution.autopilot.on_setups(serialized[:20])`
12. store `_results = serialized` and persist `logs/scan_store.json`

**Production rank R0** is the list order after step 7–9 (final stored order).  
Ready (`product/trade_desk.py`) reads the last scan + Ideas Trend Quality overlay. It does not see research shadow ranks today.

`UnifiedScanner._analyze` itself must not be modified. FEATURE-002 must not change `score`, `verdict`, `entry`, `stop`, `target`, `chase_risk`, or sort order.

---

## 2. Existing journals (do not overload)

| Store | Role | FEATURE-002 use |
|---|---|---|
| `core/signal_outcome_tracker.py` `signal_log` | BUY outcomes for live_edge | Read later for “did production trade it?”; do not write shadow ranks here |
| `core/decision_journal.py` | TAKEN/REJECTED + 5d | Context only |
| `research/feature_store.py` | Generic frozen vectors | Pattern to copy (write-once features); dedicated FEATURE-002 DB so other research cannot mix versions |
| `research/non_event.py` | Rejection controls | Unchanged |
| `research/intelligence/runtime/cycle_context.py` | Cycle overlays | Do not attach ranks here — Brain/execution must not see them |
| Lab / `signal_backtest.trading_playbook` | Keep/skip demote | Unchanged |

---

## 3. Feature calculators (reuse, do not retune)

- `research/feature001/trend_features.py` `trend_features_v1`
- `research/feature001/rs_features.py` + `research/sepa/rs.py` `rs_cs_v1` via `FastRS`
- History: official bhav via `scan.bulk_fetcher.get_cached` / `bhavcopy_store` already warm after a scan
- RS table: PIT investable-or-scan universe as-of last session date. No `scan/relative_strength.py` (yfinance)

---

## 4. Safest attach point

**After** `_results` is assigned and `_save_state()` has run, **after** autopilot has already received its slice.

```
observe_production_scan(serialized)   # deepcopy immediately; fail-open; optional thread
```

Why here:

- Production rank, verdict, ticket fields, Telegram, and autopilot queue already finalized
- A logging exception cannot prevent store update
- A slow RS table cannot delay autopilot
- The hook receives a snapshot; it must never write back

Not attached inside `UnifiedScanner.scan` or `_analyze` (would sit on the money-path compute and invite accidental mutation).

---

## 5. Influence invariant

The only allowed differences with shadow ON vs OFF:

- `logs/feature002/*` research files
- debug log lines
- optional research-only UI later, labelled shadow (not in this milestone)

Identical must remain:

- BUY/WATCH and side
- entry / stop / target
- production list order
- Ready admission inputs
- autopilot `on_setups` argument
- GTT / sizer / broker (never called from this hook)

---

## 6. Idempotency and contamination

- `event_id` = hash(`feature-002.v1`, `session_date`, `symbol`) — one primary row per name per session
- First write freezes features; later scans the same session **do not** overwrite the feature snapshot
- `scan_cycle_id` is stored; duplicate cycles do not double-count primary stats
- Pre-`2026-07-24` rows refused for primary insert
- Replay/synthetic `source` excluded from primary evaluate()

---

## 7. Out of scope

- FEATURE-003 paper ranking
- Production rank change
- RS≥70 or Stage-2 hard gates
- Regime/sector as rank multipliers
- Re-opening SEPA Core F
