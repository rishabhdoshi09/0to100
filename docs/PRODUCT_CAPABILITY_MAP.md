# QuantTerm live-desk capability map

Judged from the Vite/React terminal on `cursor/live-terminal-contract-858e`
(`scripts/run_quantterm_complete.sh` → :5173 / :8765 / :8766). Files and tests
alone do not make a surface current.

## Classification key

| Tag | Meaning |
|---|---|
| WORKING_AND_VISIBLE | Click → durable work or honest empty → UI shows the artifact |
| WORKING_BUT_HIDDEN | Backend contract exists; React nav does not expose it |
| PARTIALLY_WORKING | Executes, but progress/result/inspect is incomplete |
| OUT_OF_SYNC | Live path and another engine/UI disagree |
| LEGACY | Streamlit / unmounted React / historical branch language |
| DUPLICATED | Two registrars or two engines for one job |
| DEAD | Visible or named, no useful execution path |
| PLACEHOLDER | UI exists, not the capability the label implies |
| BROKEN | Control names a removed path or fails dishonestly |

## Markets

| Surface | UI | Backend | Artifact | Consumed? | Visible? | Class |
|---|---|---|---|---|---|---|
| Market / Home | `RadarHomeView` | `GET /api/radar-home`, `POST /api/product-bootstrap`, `RUN_SCAN_NOW` | `latest_momentum_scan.json` | Yes — scanner, recos, long-term overlay | Yes | WORKING_AND_VISIBLE |
| Market Scanner | `MarketScannerView` | `GET /api/scanner-workspace/{mode}`, `RUN_SCAN_NOW` | same scan + coverage | Yes | Yes | WORKING_AND_VISIBLE |
| Recommendations | `RecommendationsView` | `GET /api/recommendations-workspace` | `latest_recommendations.json`, reco ledger | Yes | Yes | WORKING_AND_VISIBLE |
| Market Reports | `MarketReportsView` | `REFRESH_MARKET_REPORT_NOW` | `logs/product/market_reports/` | Yes | Yes | WORKING_AND_VISIBLE |
| Long-Term Picks | `EnhancedLongTermView` + scanner tab | `REFRESH_LONG_TERM_NOW` | `latest_long_term_scan.json` | Yes | Yes (also a scanner tab) | DUPLICATED |
| Watchlist | `WatchlistView` | `/api/watchlist` | watchlist store | Local | Yes | WORKING_AND_VISIBLE |
| Compare | `CompareView` | `GET /api/compare` | read projection | Local | Yes | WORKING_AND_VISIBLE |
| ☰ Filters | Reco + Reports buttons | none | none | No | Decorative | DEAD |

Scan coverage ledger (`scan/scan_coverage.py`, `GET /api/scan-audit`) is
WORKING_BUT_HIDDEN from nav. Evidence scorecards are computed then `methods`
are stripped by `slim_workspace_for_desk`.

## Intelligence

| Surface | UI | Backend | Class |
|---|---|---|---|
| Stock Intelligence | `ProductStockIntelligenceView` | `/api/stock-intelligence/{symbol}`, `/api/due-diligence/{symbol}`, acquire op | WORKING_AND_VISIBLE — sector frameworks already run |
| Stock Investigator | `StockInvestigatorView` | same due-diligence engine + suggest | DUPLICATED |
| Thesis breakers / valuation / capital allocation | Investigate tabs | `build_due_diligence` has concerns, flags, valuation snapshot | PARTIALLY_WORKING — no mandatory “why not buy” section |
| Named scores (Piotroski etc.) | Investigate | `generic_scores.py` | WORKING_AND_VISIBLE when coverage exists; Unmeasured otherwise |

## Research / learning / backtest

| Surface | UI | Backend | Class |
|---|---|---|---|
| Backtest page | `RecoBacktestView` | `dashboard.paper.learning.self_feed` | PLACEHOLDER — paper diary, not production-strategy backtests |
| Signal calibration | Streamlit / `scan/signal_backtest.py` | `logs/signal_backtest.json` | WORKING_BUT_HIDDEN — affects scan scores, not keyed on reco cards |
| Strategy registry | none in React | `research/intelligence/registry.py` | WORKING_BUT_HIDDEN — paper/autonomy only; **not** retail recos |
| Decision journal API | none in React | `GET /api/decision-journal` | WORKING_BUT_HIDDEN |
| Evidence authority API | note on reco page | `GET /api/evidence-authority` | WORKING_BUT_HIDDEN |
| Paper learning | Backtest + Health panels | `product/paper_learning.py` | PARTIALLY_WORKING |
| Retail/AlgoLab backtesters | Streamlit | `product/retail_backtest.py`, `backtest/backtester.py` | LEGACY |

Production recommendations have **no** `strategy_id` / `rules_hash`. Attaching
Brain-1 strategy backtests to today’s BUY list would be a silent mismatch.

## System

| Surface | UI | Backend | Class |
|---|---|---|---|
| System Health | `AutomationView` | dashboard.autonomy/operations/data | PARTIALLY_WORKING — one worker pill, not separate contracts |
| Research Data | `ResearchDataView` | :8766 evidence + controls | PARTIALLY_WORKING — no LiveScanBanner |
| News / Education / F&O / Overview | own pages | news/FNO/market payloads | WORKING_AND_VISIBLE to PARTIALLY_WORKING (controls lack progress) |
| Operator health / product-contract | API only | `/api/operator-health`, `/api/product-contract` | WORKING_BUT_HIDDEN |
| Streamlit `ui/*` | not the product | `app.py` redirects to React | LEGACY |

## Honest invariants already in force

- Empty high-conviction is a valid day.
- Missing fundamentals stay Unmeasured (not zero).
- GET routes do not scrape.
- User work goes through durable operations (`MARKET_SCAN`, `MARKET_REPORT`, `DUE_DILIGENCE_ACQUIRE`, …).

## Recovery order used after this map

1. Collapse nav to Markets / Intelligence / Research / System.
2. Remove dead Filters; keep methods on reco cards; expose journal, coverage, health lanes.
3. Canonical **production method** registry with `BACKTEST PARITY: UNVERIFIED` unless the same rules_hash is proven.
4. Research / Learning view from existing paper + journal + registry (research-only).
5. Mandatory thesis-breakers on Company Intelligence (existing flags/concerns).
6. Fundamentals remain one reco family — never an independent BUY.
