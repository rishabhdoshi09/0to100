# Market Radar Retail Terminal

Completion record for the three-engine retail terminal (Breakouts · Momentum · Long-Term Picks).

## Product center

```text
Breakouts
Momentum
Long-Term Picks
```

Supporting surfaces: Stock Intelligence, Compare, Watchlist.  
Secondary only: Paper Portfolio, System Health, Market Overview.

## Navigation

| Route | Label |
|-------|--------|
| Home | Daily command centre |
| Market Scanner | Dense professional tables |
| Stock Intelligence | Company workspace |
| Long-Term Picks | Business research lane |
| Compare | 2–5 symbol comparison |
| Watchlist | Personal symbol list |

## API additions

| Endpoint | Purpose |
|----------|---------|
| `GET /api/radar-home` | Three lane payloads + market strip |
| `GET /api/scanner-workspace/{mode}` | Enriched rows (sector, breakout/momentum state) |
| `GET /api/compare?symbols=A,B` | Side-by-side comparison |
| `GET/POST /api/watchlist` | User watchlist CRUD |
| `DELETE /api/watchlist/{id}` | Remove watchlist row |

Canonical owners: `product/radar_workspace.py`, `product/compare_workspace.py`, `product/watchlist_store.py`.

## Frontend

- `frontend/src/marketRadarViews.tsx` — Home, Scanner, Compare, Watchlist
- `frontend/src/MarketSidebar.tsx` — Discovery-first navigation
- `frontend/src/radar.css` — Professional dense styling
- Live scan UX preserved via `scanRunner.ts`

## Validation

Starting SHA: `c6bf31a` · Final SHA: `2580379` (pushed to `overhaul/evidence-lab`).

```bash
python -m pytest tests/test_radar_workspace.py tests/test_compare_workspace.py tests/test_watchlist_api.py
python -m pytest
cd frontend && npm test && npm run build
```

## Honest limitations

- Compare loads full stock workspaces per symbol (acceptable for ≤5 symbols).
- Fundamental metrics depend on existing coverage; missing stays missing.
- Zerodha live execution not certified in browser pass.
- Economic edge unproven; LIVE and LIMITED_LIVE remain blocked.

## Screenshots

`docs/product/acceptance/2026-08-02-market-radar/`
