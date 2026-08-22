# Data coverage report (Phase II)

**As of:** 2026-08-22  
**Host ledgers:** `logs/` (gitignored). Machine copy: `docs/data_program/DATA_COVERAGE_REPORT.json`  
**No strategy metrics.** Validation and coverage only.

## Fundamentals (NSE XBRL + results broadcast)

| Metric | Value |
|---|---|
| Research / bhav symbols | 3156 |
| Official current EQ (v2) | 2293 |
| Symbols with ≥1 historical filing | **2057** (65% of bhav; 90% of official EQ) |
| Symbols with ≥4 quarters | 1571 |
| Symbols with ≥8 quarters | 1261 |
| Symbols with ≥12 quarters | 32 |
| Median quarters per company | 9 |
| Rows | 19151 |
| Annual / quarterly rows | 3827 / 15324 |
| Earliest filing | 2019-04-10 |
| Latest filing | 2025-06-17 |
| PIT_STRONG / DEGRADED / DESCRIPTIVE / MISSING | 19151 / 0 / 0 / (1100 bhav names with no row) |

Coverage by announcement year (available_at):

| Year | Rows |
|---|---|
| 2019 | 2164 |
| 2020 | 411 |
| 2021 | 402 |
| 2022 | 2759 |
| 2023 | 6500 |
| 2024 | 6915 |

**Class:** `RESEARCH_READY_WITH_LIMITATIONS` — not `RESEARCH_READY`.

Why not full RESEARCH_READY:

- Bounded XBRL ingest (max 12 consolidated filings / symbol from 2019). 34,362 linked instances were not downloaded.
- 2020–2021 row counts are thin versus 2023–2024.
- Only 32 names have ≥12 quarterly-usable rows.
- Operating profit and CFO were not extractable from the mapped tags (0 operating_profit rows).
- Parser must prefer NSE `OneD` (current period) over `FourD` (often YTD with reused dates). First-vs-last tag disagreement is expected and is **not** a silent repair signal.
- Some economically important names (example: TATAMOTORS) have **zero** rows in the official results API dump used here — identity / filing-symbol hole.

Canonical research preference: **CONSOLIDATED**. Standalone rows are labelled. Nine-month / YTD spans are not quarterly.

## Earnings / result events

| Metric | Value |
|---|---|
| Event count | 137212 |
| Symbols | 2489 |
| Range | 2016-01-08 → 2026-08-06 |
| EVENT_TIMESTAMP_STRONG | 137212 (100%) |
| EVENT_DATE_ONLY | 0 in this dump |
| Consensus / surprise | **unavailable** — not computed |

**Class:** `RESEARCH_READY_WITH_LIMITATIONS`

**Event effective-date policy:** if only a calendar date is known (`EVENT_DATE_ONLY`), causal backtests use **NEXT_SESSION** availability. Same-day close must not react to an announcement whose time is unknown. This dump carries IST wall-clock `broadCastDate` / `exchdisstime`, so session class (before / during / after market) is inferable. Do not invent missing times.

## Universe / identity

| Layer | Rows | Class |
|---|---|---|
| Default `universe_history.json` | 3156 bhav-inferred | `PIT_DEGRADED` / `DESCRIPTIVE_ONLY` |
| Official `universe_history_v2.json` | 2293 (4 delist membership rows) | `RESEARCH_READY_WITH_LIMITATIONS` |
| Official delists omitted (no listing date) | 324 | remaining survivorship hole |
| Identity graph | 2618 securities; 1047 symbol changes | official EQUITY_L + symbolchange + delisted |

v2 was **created** because official listing dates for current EQ are material. It was **not** promoted to the silent default: 324 undated official delists mean survivorship is still incomplete. Bhav-inferred sidecar: `logs/universe_history_bhav_inferred.json`.

Stale-bar rule is now generic (`data/universe_freshness.py` + `listing_archive.is_investable`): a last print is not a living listing. Tests cover delist, suspension, one-session miss, hard stale, relist. Not MAGMA-specific.

## Sector

| Item | Value |
|---|---|
| Mapped names | 845 (408 SEPA overlay + 752 official Nifty Total Market industry, union) |
| Official industry rows | 752 |
| PIT class | **STATIC_BACKFILL** (unchanged) |
| vs all bhav (3156) | 27% mapped — UNKNOWN still dominant |
| vs official EQ (2293) | 37% |
| vs Nifty Total Market | industry column present for the 752-name list |

Coverage rose. PIT quality did not. No dated reclassification archive.

## Corporate actions

607 share-count events / 462 symbols. `CA_RESEARCH_ACCEPTABLE`. **Not** `CA_COMPLETE`.

Unresolved classes (no inferred factors):

| Class | Disposition |
|---|---|
| rights | quarantine / segment |
| demerger | quarantine / segment |
| merger | quarantine / segment |
| special_distribution | quarantine / segment |
| symbol_restructuring | research segmentation |
| unknown_discontinuity | quarantine / segment |

## Benchmarks (official `ind_close_all`, price-return)

| Index | Sessions | First | Last | Kind |
|---|---|---|---|---|
| Nifty 50 | 2660 | 2015-11-09 | 2026-08-21 | price_return |
| Nifty 500 | 2660 | 2015-11-09 | 2026-08-21 | price_return |
| Nifty Total Market | 1199 | 2016-07-07 | 2026-08-21 | price_return |
| India VIX | 2864 | 2015-01-01 | 2026-08-21 | price_return |
| Sector PR (Bank/IT/…) | 2660 | 2015-11-09 | 2026-08-21 | price_return |

2,864 official daily CSVs on disk. Research pickle: `logs/indices/research_index_store.pkl`. **No cash TRI.** Never overwrite PR with TRI. Internally constructed EW-investable remains a separate series if used.

**Class:** `RESEARCH_READY_WITH_LIMITATIONS` (official PR depth is now 2015-11+, still no TRI).

## FEATURE-002 (operational only)

| Item | Value |
|---|---|
| Operational state | `NO_POST_ACTIVATION_SCAN` |
| Research maturity | `FORWARD VALIDATION ACTIVE — INSUFFICIENT NEW DATA` |
| Primary rows | 0 (expected; Saturday; no production scan) |
| First-real-scan acceptance | **not accepted** — evaluator does not fabricate rows |
| Spec | frozen (`feature-002.v1`, activation `2026-08-22T00:00:00+05:30`) |

`HEALTHY_COLLECTING + INSUFFICIENT_NEW_DATA` remains a valid future pair. Not observed today.
