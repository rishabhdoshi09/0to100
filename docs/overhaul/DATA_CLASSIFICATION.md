# Data Classification — trust classes for every source

Every dataset and data source in QuantTerm declares exactly one **trust class**.
The class governs which subsystems may consume it. This is the enforcement mechanism
for Decision E ("no implicit research-data fallback").

## The four classes

| Class | Meaning | May feed |
|---|---|---|
| `RESEARCH_GRADE` | Point-in-time correct, corporate-action adjusted, survivorship-complete, provenance-hashed, coverage-checked | Research simulator, gauntlet, evidence registry, strategy verdicts |
| `OPERATIONAL_ONLY` | Good enough to *trade today* (live quotes, today's snapshot) but NOT reconstructible/point-in-time | Live/paper execution, live risk, current-setup display |
| `DISPLAY_ONLY` | Convenience/approximate; may be stale, adjusted-differently, or scraped | UI decoration, charts, news, non-critical summaries |
| `PROHIBITED` | Must never be used for any decision (synthetic, demo, hard-coded, unadjusted-in-research) | Nothing |

**Rule:** a `RESEARCH_GRADE` consumer refuses any input of a lower class and fails
closed. A consumer never silently "downgrades" its own inputs.

## Current sources → assigned class (Phase-0 baseline)

| Source / module | Today's reality | Assigned class | Notes / required work |
|---|---|---|---|
| NSE bhavcopy (`data/bhavcopy_store.py`) | official EOD, ~7y, **raw/unadjusted**, current-symbol keyed | `OPERATIONAL_ONLY` → `RESEARCH_GRADE` **after** CA ledger + security master + snapshot manifest | Becomes research-grade only once CA-adjusted, survivorship-mapped, hashed |
| NSE index store (`data/index_store.py`) | official index OHLC, ~9y | `OPERATIONAL_ONLY` → `RESEARCH_GRADE` after snapshot manifest + coverage checks | Benchmark + regime source |
| NSE live snapshot (`data/nse_live.py`) | intraday, partial today bar | `OPERATIONAL_ONLY` | Trading only; never research |
| Kite quotes/historical (`data/kite_client.py`) | broker feed | `OPERATIONAL_ONLY` | Trading/live; historical needs manifesting before research use |
| **yfinance** (`data/us_data.py`, `scan/bulk_fetcher.py`, `gauntlet/momentum.py --source yf`) | survivorship-biased, adjusted-differently, flaky | `DISPLAY_ONLY` | **Must be unreachable from research grade** (root cause of EXP-005). `--source yf` stays but is labelled DISPLAY/exploratory-only and can never yield a PASS beyond EXPLORATORY |
| Google Finance (`data/google_finance.py`) | scrape | `DISPLAY_ONLY` | Last-resort quote only |
| `ca_events.json` (absent) | — | `RESEARCH_GRADE` once supplied from real NSE CA filings | Blocking dependency for research-grade prices |
| `universe_history.json` (absent) | — | `RESEARCH_GRADE` once supplied (listing/delisting) | Blocking dependency for point-in-time universe |
| Hard-coded index lists (`data/nse_universe.py` NIFTY50/100/500 constants) | current constituents | `DISPLAY_ONLY` | Never a research universe; survivorship trap |
| Synthetic/demo/fixtures | test only | `PROHIBITED` in prod; fixtures allowed only in tests | Public demo uses clearly-labelled synthetic fixtures |

## Enforcement plan

1. Introduce a `TrustClass` enum and a `@requires_trust(RESEARCH_GRADE)` boundary at
   the entry of the research simulator, gauntlet, and evidence registry.
2. Each dataset manifest (`data_platform/manifests/`) carries `trust_class`.
3. A research run records the trust class of every input in its experiment stamp; any
   input below `RESEARCH_GRADE` aborts the run (fail closed).
4. `gauntlet/momentum.py --source yf` remains for exploration but is hard-capped at
   verdict tier `EXPLORATORY` and stamped `DISPLAY_ONLY` — it can never emit a
   `HISTORICAL_SURVIVOR` or higher (EXP-005 lesson, encoded).

## Blocking external dependencies (honest statement)

`RESEARCH_GRADE` for Indian equities is **not achievable from free sources today**:
CA events and survivorship history must be supplied (NSE archives assembly or a paid
vendor). Until then the platform operates the full pipeline on `OPERATIONAL_ONLY`
data, labels every result accordingly, and refuses to mint a research-grade verdict.
This is the correct fail-closed behaviour, not a limitation to paper over.
