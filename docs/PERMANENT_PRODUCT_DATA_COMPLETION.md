# Permanent Product and Data Completion

Record of the permanent QuantTerm retail terminal and canonical data platform work on `overhaul/evidence-lab`.

## Implementation baseline

- Branch: `overhaul/evidence-lab`
- Checkpoint before this mission: `349c4994070e7ca230a8a11b773209d07e39bd46`
- Acceptance commit: see final remote SHA at delivery (not embedded here to avoid stale docs)

## Architecture

### One truth

| Domain | Canonical owner |
|--------|-----------------|
| Symbol identity | `data_platform/security_master.py` + `data/nse_universe.py` |
| Price history | `data/bhavcopy_store.py` |
| Latest quotes | `data/live_quotes.py` (Kite → NSE → Google) |
| Fundamentals cache | `fundamentals/cache.py` |
| Ratios | `data_platform/ratios.py` (from cache, never React) |
| Scan classifications | `product/radar_workspace.py` + `scan/*` |
| Company workspace | `product/stock_workspace.py` |
| Watchlist | `product/watchlist_store.py` |
| Compare | `product/compare_workspace.py` |
| Trade Plan | `product/trade_plan.py` |

### Data platform package (`data_platform/`)

- `contracts.py` — observation metadata, `QualityStatus`, capabilities
- `security_master.py` — `CompanyProfile` projection
- `provider_registry.py` — priority-ordered provider capabilities and health
- `coverage.py` — per-symbol audit + remediation queue
- `ratios.py` — central ratio formulas with missing-input reasons
- `jobs.py` — observable refresh job registry
- `import_pipeline.py` — validated JSON/CSV user import for fundamentals

### API routes (`product/data_api.py`)

- `GET /api/data/providers`
- `GET /api/data/coverage` (universe or `?symbol=`)
- `GET /api/data/jobs`
- `GET /api/data/security-master`
- `GET /api/data/ratios/{symbol}`

## Provider priority (live data)

1. Verified local datasets (`logs/bhav`, fundamentals cache)
2. Official NSE downloads (bhavcopy)
3. Kite broker API (token required)
4. NSE snapshot API
5. Screener fundamentals (cached scrape)
6. Google Finance — fallback only, never primary

## Scan engines

Breakout and momentum sub-states extended in `product/radar_workspace.py`:

- Breakout: confirmed, near, under observation, without volume, failed, extended, insufficient data
- Momentum: actionable, extended, steady leadership, improving, weakening, high-vol, insufficient history

React tables display server classifications only (`frontend/src/marketRadarViews.tsx`).

## UI structure

Primary nav: Home, Market Scanner, Stock Intelligence, Long-Term Picks, Compare, Watchlist.

Design tokens: `frontend/src/design-tokens.css`

Reusable components: `frontend/src/designSystem.tsx` (SectionTabs, StatusBadge, EmptyState)

Stock Intelligence tabs: Overview, Chart, Financials, Ratios, Ownership, Events, Peers, Evidence

HMR fix: `DisplayDepthToggle` moved to `frontend/src/displayDepth.tsx`

## Tests

Hermetic isolation: `tests/conftest.py` autouse fixture resets bhavcopy in-memory store and market-memory corpus cache before/after every test.

`data/bhavcopy_store.reset_in_memory_store()` and `research/market_memory.reset_analog_corpus_cache()` support isolation.

### Collection reconciliation

| Run | Collected | Notes |
|-----|-----------|-------|
| Default `python -m pytest` | **1193** | `tests/integration/` excluded via `collect_ignore` |
| `QT_INTEGRATION=1 python -m pytest` | **1240** (+47 integration) | Full operational scan chain |

Previous ~740 counts likely reflected partial runs, stale processes, or integration exclusion — not deleted tests.

### Result (acceptance)

- **1193 collected, 1193 passed, 0 failed, 0 skipped, 0 xfailed** (default suite, twice, with warm `logs/bhav/`)
- Frontend vitest: **5 passed**
- `npm run build`: pass

## Browser acceptance

Run `terminal_product_api:app` on `:8765` with Vite on `:5173`.

Screenshots: `docs/product/acceptance/2026-08-02-market-radar/` (webp; PNG gitignored)

## Remaining external constraints

- Full bitemporal fundamentals ledger (publication-dated filings) — import path + cache exist; licensed filing feeds require user credentials or file import
- Ownership detail beyond Screener cache — user import or licensed provider
- Economic edge unproven — LIVE and LIMITED_LIVE remain blocked
- No real broker order path in React terminal

## User import

`data_platform/import_pipeline.py` — JSON fundamentals import with provenance and idempotency (no overwrite without `overwrite=True`).

## Performance

- Server-side scan/ratio/coverage computation
- Vite `/api` proxy to `:8765`
- Design tokens reduce ad-hoc gradient overrides

## Safety

- No LIVE Buy button
- `broker_mutations_enabled: false` on observer API
- Missing fundamentals/ratios remain empty in UI
