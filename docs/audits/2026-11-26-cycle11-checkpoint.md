# Cycle 11 checkpoint: host recovery, live-host verification, truthfulness fixes (26 Nov 2026)

Branch `claude/cycle9-20261126`, from `9af48283` (the cycle-10 candidate). Development only. Production
code stays `125880a6`, and live trading stays HARD-LOCKED.

## Host operations (owner-authorized, 18:20–19:40 IST)

- **One controlled detach and remount, on the owner's words.**
  - **Reverified first:** the image path equals the configured QT_STORAGE_BUNDLE; the device was /dev/disk3; nothing held it (lsof); AC at 100%; the Seagate was on the bus; Expansion was mounted.
  - **Detach:** `hdiutil detach /dev/disk3` with no force, rc 0.
  - **Recovery:** the supervisor re-attached the image in 52 s. APFS UUID `23E55E74-…`; the preflight passed; the API was healthy at 18:22.
- **The catch-up ran by itself:**
  - the 26 Nov bhavcopy (3,481 symbols);
  - outcome resolution (1,693 official outcomes, 0 book closes);
  - 24 Nov was an NSE holiday.
- **After the owner's Kite login (19:30):**
  - broker `SESSION_VALID`; the data refresh reported "genuine snapshot active";
  - the EOD scan succeeded at 19:39 (481 setups, session 2026-11-26); health READY;
  - Home: "No BUY today. 29 setup(s) technically ready, none with a proven edge (closest SAFARI +0.10R/trade)". 78 cards, all WATCH;
  - Telegram scan alerts were delivered (no new send errors);
  - paper book unchanged (RPEL 2 @ 1935.58, capital 100,000, realized 0); F&O book 0/0, integrity ok;
  - live LOCKED, verified.
- **Observations:**
  - `/api/radar-home` took 24.4 s on the first build after the scan;
  - the USB port draws 896 of 900 mA (a 4 mA margin);
  - a 46 s battery blip at 16:04;
  - two 23 Nov paper cycles were retired as orphaned (by design); the completed cycles that day found NO_ELIGIBLE_TRADE.

## Software defects (each a failing-first test; isolated worktree)

| Defect | Evidence | Fix | Tests |
|---|---|---|---|
| `/api/autonomy-desk` served the product loop's 2026-09-18 summary as current: readiness SESSION_VALID with EXECUTION_READY true, beside a live LOGIN_REQUIRED. Each request also scanned the dormant 13.8 MB candidates.db (no session index): 4.9 s warm, 11.8 s cold | live payload; `sample` + `lsof` on the running API | `loop_summary` (age, stale); broker capabilities from one live observation; a stale summary's candidate sections are withheld. No schema change | 6 (`test_cycle11_autonomy_desk_staleness.py`) |
| The 23 Nov "Nifty disagreement" (+1.122% vs −1.406%): +1.122% was the 20 Nov close shown undated as "Latest session"; the F&O value was 23 Nov live (that session closed −1.683%). On 26 Nov the tile showed 23 Nov as latest, two sessions behind | host index store (read-only) | Tiles carry as_of, source, basis and stale (against the latest session whose 15:30 close passed); the Market Reports route keeps them and re-judges when served; the text names the session | 8 (`test_cycle11_index_tile_freshness.py`) |
| A login after the close never released AUTH_READY: the EOD Kite refresh and index warm-up stayed BLOCKED, and the index store stayed at 23 Nov | the host's job store (read-only) | The closed-market tick schedules the day's idempotent auth probe when a recent job waits on AUTH_READY | 4 (`test_cycle11_closed_market_auth.py`) |
| GPT review of cycles 10–11: C10-02 concurrent short writes, C10-04 the final status write, C11-01..04 | each reproduced | see commit c9a8634d | 7 more across the files |
| `/api/product-readiness` latency | 0.71 s warm vs 3.1 s cold; the samples are waits, so cold HDD reads | none (hardware) | — |

**Not fixed, with reasons:**
- **C10-01 / A1:** a fill durable before its post-save bookkeeping. Needs a durable entry-origin record committed with the fill.
- **C10-03:** a late heartbeat says "running" while the process is shutting down. Disagreed: it is running.
- The production instance of the AUTH_READY defect clears at the next open with the deployed code.
