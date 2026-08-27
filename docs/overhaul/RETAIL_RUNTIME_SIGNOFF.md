# QuantTerm Retail Runtime — Production-Readiness Sign-off

Scope: **product/runtime integrity** of the retail application — not proof of trading edge. Answers:
*can a retail user open QuantTerm, understand the market state, see what was evaluated, tell
"no opportunity" from "system failure," and trust that every displayed conclusion comes from real,
traceable data?*

## Verdict

**CONDITIONALLY READY** for paper/observation use.

The runtime boots, every retail page renders (with and without data), the four journeys are correct,
the F&O funnel reconciles exactly, exclusions are fully traceable, ranking is deterministic, failures
are honest, and no retail action can reach a broker order. The two conditions before unattended
paper/observation use are operational, not architectural:
1. A one-time **live-data smoke** on the user's connected environment — this sandbox has no Zerodha
   token, so Scenario C was proven with injected fixtures + the live server's empty-data states, not
   on a genuine activated snapshot.
2. The **Market** page depends on the regime engine's index feed; now bounded (no longer an infinite
   hang) but slow (~30 s) on a cold, fully-offline first load. Fast once a cache or network exists.

## Launch command

```bash
bash scripts/run_desk.sh        # API :8765 + Vite desk :5173 + autonomy
# then open http://127.0.0.1:5173
```
Verified: desk is the Vite/React app. Streamlit is not started.

## Canonical architecture (unchanged)

```
app.py                     st.navigation, retail default, Home = default page
ui/retail_pages_v2.py      page export hub → retail_home_momentum / retail_trade_market /
                           retail_backtest_data / retail_pages + ui/fno_momentum_page
product/                   READ-ONLY projection (view-model): projection, gather, market_view,
                           no_trade, scan_store, retail_backtest, runtime
data/fno_universe.py       dynamic underlying-first F&O universe + evaluation funnel
```
Boundary: **gather** (read backend) → **domain engines** (UnifiedScanner, Backtester, regime_engine,
auto_research brain — unchanged) → **product projection** (plain language, counts, next action) →
**Streamlit render**. The projection owns no trading/risk/evidence/execution logic and holds no second
source of truth.

## Pages exercised (Streamlit real script runtime, AppTest)

All 14 render without exception on the sandbox's no-data state: Home, Momentum Stocks, Market News,
Automatic Paper Trading, Portfolio, Market, Backtest, What We've Learned, Reports, Data and Zerodha,
Alerts, Settings, Help, Research Laboratory. Live-server screenshots captured for Home, Momentum,
Automatic Paper Trading (`docs/overhaul/` referenced from the session).

## Four mandatory journeys (values from the canonical modules)

| Scenario | Result |
|----------|--------|
| **A — system ready, no valid trade** | `no_trade`: "The latest scan found no entry-ready setup." Funnel stages shown; unknown counts render **"Not exposed"** (never a fake 0). Not labelled an error. |
| **B — data unavailable/incomplete** | `build_product_state` → headline "Get market data ready", action "Update Market Data". Distinct from A. Momentum page: "No saved scan exists yet · Run fresh scan" — never "no opportunity". |
| **C — qualified opportunities** | Funnel returns ranked qualified rows with per-row reason/score/expiry/lot; "qualified" is presented as an evaluation pass, not a deploy signal. (Proven with fixtures; live-data smoke is condition #1.) |
| **D — mixed F&O funnel** | See reconciliation below — every underlying enters, each exclusion recorded, totals reconcile exactly. |

No-data vs no-trade are unambiguous: A and B produce different headlines/actions, and neither uses the
word "error"; genuine failures surface separately as `attention`/`last_error`.

## F&O reconciliation (worked example)

Mixed instrument master (AAA×2 futures, BBB, CCC, MISSING, NIFTY index):

```
Current listed F&O underlyings (unique_stock_underlyings): 4     (NIFTY counted as index future, excluded)
Mapped to NSE cash (mapped_underlyings):                   3
  Data ready:                                              2
  Evaluated:                                               2
  Momentum qualified:                                      1     (AAA)
Mapping exclusions:  MISSING → canonical_mapping
Funnel rows:  AAA qualified · BBB rejected(momentum) · CCC rejected(history: 20/60 sessions)

Reconciliation: qualified(1) + funnel-excluded(2) + mapping-exclusions(1) = 4 = unique(4) ✓
```
Every exclusion carries symbol/underlying, stage, plain reason (with observed value + required
threshold, e.g. "20 sessions available, 60 required"); the universe is built against an explicit
`as_of`. Filters change only what is displayed, never the evaluated counts. Nothing disappears
silently.

## Data sources & freshness

- Instrument master / F&O universe: data-only Kite client → `logs/instruments_cache.csv` fallback →
  `source="unavailable"` (never a hard-coded shortlist).
- History: NSE bhavcopy snapshot + `scan.bulk_fetcher`; regime index feed: NSE index store → Kite →
  yfinance (now `timeout=8`).
- Freshness: `data_ready` = active snapshot verified **and** has benchmark; saved scans carry
  `scanned_at` and surface age in hours; unknown age is `None`, not a fake 0.

## Failure behaviour

- Missing Zerodha token → Home "Waiting for Zerodha login" + Connect action; no crash, no fabricated
  data.
- Provider error inside F&O evaluation → recorded as a per-symbol exclusion (stage `history`/
  `analysis`), never a silent empty "no opportunities".
- Unavailable market feed → Market page shows an understandable "temporarily unavailable" state
  (bounded, was previously an unbounded hang — fixed at source; see defects).
- Corrupt/empty/malformed instrument input → safe empty report, no crash.
- No retail action reaches `place_order`/`place_gtt`/`modify_order`/`cancel_order` (asserted by test).

## Defects found & fixed (at source)

1. **Market page could hang indefinitely.** `data/index_store._build_index_store_locked` waited on
   `as_completed` with no timeout; with no cache + no network it fired hundreds of per-day requests.
   Fixed: whole-batch budget (`_BUILD_BUDGET_S`), `shutdown(cancel_futures=True)`, and a short
   re-attempt cooldown (`_BUILD_COOLDOWN_S`) so concurrent regime tickers don't each re-pay it.
   `core/regime_engine._fetch_ohlcv` yfinance call now passes `timeout=8`. `ui/retail_trade_market
   .render_market` wraps the view in an error boundary → understandable state, never a stack trace.
2. **Non-deterministic ranking on ties.** `ui/fno_momentum_page` qualified sort and
   `product/scan_store` record/watchlist sorts used score only. Fixed: symbol as the stable
   secondary key everywhere.
3. **News optional-dependency boot** (prior milestone, retained): `news/fetcher` degrades when
   `feedparser` is absent instead of breaking import/collection.

## Point-in-time & data-integrity findings

- No `date.today()` inside the F&O **evaluation** path; `build_fno_universe` takes an injectable
  `as_of` (default today only for current-contract selection, which is a *current* view). Asserted.
- `evaluate_all_underlyings` touches only its injected `history_getter`/`analyzer` — no live-provider
  call, no `prefetch`/`requests`/`yf`/`download` in the evaluation function. Asserted.
- Backtest uses the real event-driven `backtest/backtester.py` on `.tail(days)` history (ends at the
  latest available bar — no look-ahead); empty data → an honest "not usable" summary, never fake PnL.
- Determinism: repeated evaluation yields identical rows/counts; `build_product_state` is a frozen
  projection (equal on repeat) — reruns don't mutate canonical results. Asserted.
- Duplicate futures contracts collapse to one underlying (nearest contract kept). Asserted.

## Tests run & exact results

```
$ python -m pytest tests/test_retail_signoff.py -q            → 18 passed
$ python -m pytest tests/test_product_ux.py tests/test_fno_universe.py \
      tests/test_retail_signoff.py -q                          → 30 passed
$ python -m pytest tests/ -q                                   → 912 passed in ~95s
$ python -m py_compile app.py ui/retail_*.py product/*.py \
      data/fno_universe.py data/index_store.py core/regime_engine.py  → OK
```
New coverage: `tests/test_retail_signoff.py` (the 18 acceptance guarantees). All page renders
verified via Streamlit `AppTest`; live server verified via `/_stcore/health` + Playwright screenshots.

## Known limitations

- No live Zerodha token in this environment → Scenario C on **genuine** qualified data was proven with
  fixtures + the live server's empty-data states, not on a real activated snapshot (condition #1).
- Market/regime page is bounded but slow (~30 s) on a cold, fully-offline first load; fast with cache
  or network.
- Screenshots limited to Home/Momentum/Automatic-Paper-Trading (Playwright CLI, URL-addressed pages);
  remaining pages verified via AppTest, not pixels.

## What should happen next

1. Run one live-data smoke on the connected environment: after the daily Zerodha login, confirm
   Home → data ready, run a whole-market Momentum scan, open the F&O funnel (real underlying count),
   and confirm PAPER_AUTO opens/among/manages a paper trade or reports a genuine no-trade.
2. Optionally warm `logs/index_store.pkl` so the Market page is instant offline.
3. Keep this sign-off's verdict at CONDITIONALLY READY until the live smoke passes; then promote to
   READY FOR PAPER/OBSERVATION USE. **Do not** infer live-capital readiness from this document.
