# QuantTerm Retail Product Completion

Authoritative completion record for the React retail terminal on branch `overhaul/evidence-lab`.

## Navigation (final)

**Primary (RESEARCH sidebar)**

| Route key | Label |
|-----------|--------|
| Command Center | Home |
| Scanner | Discover |
| Stock Intelligence | Stock Intelligence |
| Long-Term | Long-Term Research |
| Research Data | Research Data |

**Secondary (OPERATIONS & EVIDENCE)**

| Route key | Label |
|-----------|--------|
| Market Internals | Market & Breadth |
| News & Events | News & Events |
| F&O Desk | F&O Coverage |
| Automation | System Health |
| Portfolio | Paper Portfolio |

Implementation: `frontend/src/MarketSidebar.tsx`.

## User journey

```text
market context (Home strip)
  → opportunity board / Discover modes
  → Stock Intelligence workspace
  → Trade Plan (read-only risk lens)
  → optional paper rehearsal (Paper Portfolio — secondary)
```

## Page purposes

| Page | Purpose |
|------|---------|
| Home | Idea-first daily board: market strip, opportunity lanes, central chart, decision lens |
| Discover | Unified scanner modes from persisted backend rankings |
| Stock Intelligence | Full stock workspace + trade plan + evidence |
| Long-Term Research | Business quality workspace (not a scanner clone) |
| Research Data | Upload / validate / activate verified snapshots |
| Market & Breadth | Regime, breadth, sectors with source time |
| News & Events | Dated context; never backtest news |
| F&O Coverage | Contract coverage metadata |
| Paper Portfolio | Recorded paper positions and closed trades |
| System Health | Operations, autonomy, infrastructure detail |

## Canonical owners (do not duplicate)

| Domain | Owner |
|--------|--------|
| Market scan execution | `operations/market_ops.py` + `operations/store.py` |
| Scan results | `scan/market_scan_service.py` → `logs/product/` |
| Workspace projection | `product/workspace.py`, `product/observer_api.py` |
| Trade Plan | `product/trade_plan.py` |
| Stock Intelligence | `product/stock_workspace.py` |
| Terminal API | `terminal_api.py` / `terminal_product_api.py` |
| React UI | projection + controls only — `frontend/src/` |

## Live scan semantics

1. User clicks **Scan now** → `POST /api/controls/RUN_SCAN_NOW`.
2. API enqueues `MARKET_SCAN` in `OperationStore` (deduplicates active PENDING/RUNNING).
3. Frontend `useScanRunner` (`frontend/src/scanRunner.ts`) attaches to `operation_id`, polls `GET /api/operations/{id}` every **1s** while active.
4. Friendly phases map real backend `stage` values (e.g. `PREPARING_HISTORY`, `SCANNING`).
5. Progress bar only when `progress_total > 0`; no invented percentages.
6. On `SUCCEEDED`, dashboard refresh runs once (completion guard prevents storms).
7. Reload recovery: active operation seeded from `dashboard.operations.active`.

Long-term scan uses `LONG_TERM_SCAN` / `RUN_LONG_TERM_SCAN_NOW` with the same hook.

## Simple vs Professional

Same backend state. Simple hides raw operation diagnostics; Professional shows stage, status and provenance in `LiveScanBanner` and Trade Plan footers.

## Institutional surfaces

OMS, Risk Governor, reconciliation, protection, TCA remain backend-only projections on System Health / institutional API routes. No LIVE order button in React.

## Source and freshness

- Prices: official bhavcopy sessions (`dashboard.data.bhavcopy`).
- Scan rows: persisted scan JSON with `scanned_at`.
- Fundamentals: current snapshot; historical point-in-time disclosed in Stock Intelligence when missing.
- Missing data stays missing — no synthetic fill.

## Validation (current head)

```bash
# Python (network-free)
python -m pytest
python -m compileall -q data scan risk execution ui reports ai core alerts news research product reporting operations fundamentals
python -m py_compile terminal_api.py terminal_product_api.py report_api.py

# Frontend
cd frontend && npm install && npm test && npm run build
```

## Browser acceptance

Verified against `uvicorn terminal_product_api:app` on port 8765 + Vite on 5173:

- Home idea-first layout with Scan now immediate state
- Discover live scan banner and mode switching
- Stock Intelligence, Long-Term, Research Data, System Health navigable
- Paper Portfolio secondary in sidebar
- API degraded banner when backend down

## Remaining external blockers

| Blocker | Impact |
|---------|--------|
| Zerodha daily login | Live quotes, broker observation — UI shows disconnected state without credentials |
| Market-ops worker | Scans queue but need worker process for completion (API auto-spawns via `_ensure_ops_worker`) |
| Economic edge | Engineering does not prove profitability |

## Honest status matrix

| Area | Status |
|------|--------|
| Retail product | **Complete** for discovery → decision journey |
| React build | **Passing** |
| Home | **Idea-first** with opportunity board |
| Discover | **Unified modes** + live scan |
| Stock Intelligence | **Workspace + trade plan** (existing) |
| Trade Plan | **Read-only** API |
| Long-Term Research | **Dedicated lane** |
| Research Data | **Workflow UI** (existing) |
| Production PAPER | **Backend** — not primary UI |
| Broker observation | **Read-only** — needs credentials for live |
| Operational certification | **Partial** — see institutional readiness |
| Economic edge | **Not claimed** |
| New OMS broker submission | **Disconnected** |
| LIMITED_LIVE | **Blocked** |
| LIVE | **Blocked** |
