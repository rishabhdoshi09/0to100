# FEATURE-002 live_scan path audit

**Date:** 2026-08-22  
**Scope:** production scan → shadow ledger → outcome resolver.  
**Not in scope:** FEATURE-002 rank formulas, graduation gates, forward-start date, production ranking.

This audit answers why the primary `live_scan` ledger is empty and whether that is a logging bug.

---

## Path (as implemented)

1. `scan.auto_scan._scan_once_locked` runs `UnifiedScanner.scan`.
2. Cards are serialized, enriched (sector heat, conviction, live overlay, edge, EV, prime).
3. Telegram / autopilot run on the **same** serialized list.
4. `_save_state()` writes `logs/scan_store.json` (results + unix `ts`).
5. Fail-open hook: `observe_production_scan(serialized, source="live_scan")`.
6. Observe deep-copies cards, then (default) starts a **daemon** thread.
7. Thread builds an immutable candidate set + per-symbol rows and writes `logs/feature002/shadow.db`.
8. `event_id = sha256(feature-002.v1 | session_date | SYMBOL)` — first write wins.
9. `eligible_primary` requires: version match, `source=live_scan`, `session_date >= 2026-07-24`, `recorded_at >= 2026-08-22T00:00:00+05:30`.
10. Resolver (`research.feature002.resolve`) attaches outcomes later; never rewrites `feature_snapshot`.

Production BUY/rank/autopilot/broker code does not import FEATURE-002.

---

## Answers (this environment, 2026-08-22 Saturday)

| # | Question | Finding |
|---|---|---|
| 1 | Why are primary `live_scan` observations zero? | Protocol IST start is **2026-08-22T00:00:00+05:30**. Ledger tables exist but contain **0** rows. `logs/scan_store.json` is **absent**. No post-activation production scan has persisted results on this host. |
| 2 | Expected because no post-activation scan ran? | **Yes.** NSE is closed (Saturday). Worker only scans in market hours (09:15–15:30 IST, weekdays). Friday 21 Aug scans, if any, would be before protocol IST and stored as `implementation_test` or refused — not primary. |
| 3 | Is the logging path not receiving production scans? | **Not proven as a wiring miss.** The hook sits after `_save_state` and autopilot (tests assert source order). This host simply has no `scan_store` and an empty ledger. |
| 4 | Rows written but rejected by protocol filters? | **No rows at all** (primary or test) in `shadow.db`. Filters are not hiding a silent write. |
| 5 | Timestamps / session dates wrong? | **Partial quality bug (fixed in this mandate, spec unchanged).** `build_shadow_records` previously overwrote `session_date` with max history bar date. A weekend/Monday scan could collide with Friday's `event_id` and first-write-wins would drop a genuine later session. Session date is now the IST scan calendar date; hist as-of lives only on the feature snapshot. |
| 6 | Does restart lose observations? | **Scan cards:** `_save_state` / `_load_state` restore results. **Shadow rows:** SQLite persists; restart does not drop committed rows. **In-flight daemon thread:** if the process exits before the thread commits, that cycle is lost (fail-open by design). Hook receipts now record the invocation even if persist fails. |
| 7 | Duplicate suppression of genuine scans? | Same symbol + same IST session → one row (intentional). Different sessions must not share an `event_id`. The hist-as-of overwrite was the accidental suppressor; that is fixed. Same-day rescan still first-write-wins. |

**Verdict:** empty primary ledger is **expected** on activation Saturday with no market-hours scan. It is **not** a FEATURE-002 specification bug. If a weekday market-hours scan later exists in `scan_store` with `ts` after protocol IST and the ledger stays empty, the watchdog treats that as a **logging bug**.

---

## Failure modes (unchanged production behaviour)

- Hook exception → `feature002_shadow_skip` debug log; scan still `ready`.
- Observe exception inside thread → `feature002_shadow_failed`; no raise into the worker.
- Empty serialized list → observe returns `None` (now logged as `hook_skipped`).
- Disabled observe (`set_enabled(False)`) → no writes.
- Autopilot / Telegram / GTT are independent; shadow cannot place orders.

---

## Health / watchdog

Machine-readable: `docs/data_program/research_logging_health.json` (also `logs/feature002/research_logging_health.json`).  
User-facing: `docs/overhaul/experiments/FEATURE-002/FEATURE_002_STATUS.md`.  
Watchdog: `research.feature002.watchdog` — log/alert only.
