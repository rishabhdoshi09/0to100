# Final Retail Product Browser Acceptance

> **Historical record** (branch `overhaul/evidence-lab`, commit listed below).
> Not the current product path. Canonical launcher:
> `bash scripts/run_quantterm_complete.sh` (Vite/React desk). Streamlit is not started.

Authoritative acceptance record for the QuantTerm React retail terminal.

## Environment

| Field | Value |
|-------|--------|
| **Repository** | `rishabhdoshi09/0to100` |
| **Branch** | `overhaul/evidence-lab` |
| **Commit under test** | `1781b027e561f0847a7f04e64d337b1bf1661734` (live-scan UX + initial doc/tests) |
| **Acceptance date** | 2026-08-02, ~04:07 UTC |
| **Frontend URL** | `http://127.0.0.1:5173` (Vite dev) |
| **API URL** | `http://127.0.0.1:8765` (`terminal_product_api:app`) |
| **Data context** | Persisted bhavcopy (~764 sessions), saved market scan (~2,407 universe, 834+ scan records), paper book at ₹1,00,000 with 0 open positions. No Zerodha live session for this pass. |

## Home

| Criterion | Result |
|-----------|--------|
| Opportunity board | **Pass** — Momentum, Breakouts, Pre-breakout, Conviction, Long-term, Avoid/extended lanes with sourced counts |
| Selected idea | **Pass** — row selection drives chart and decision lens |
| Scan button | **Pass** — immediate `Scanning…` state |
| Friendly live phases | **Pass** — e.g. “Starting the scan…”, not raw `PENDING` |
| Completion refresh | **Pass** — dashboard reload after `SUCCEEDED` |
| No raw operation IDs in hero | **Pass** — operation id not shown in top bar or banner headline |
| Real counts / timestamps | **Pass** — universe 2,407, scan date from persisted payload |

**Screenshots:** `home-opportunity-board.webp`, `home-live-scan.webp`, `home-scan-complete.webp`

## Discover

| Criterion | Result |
|-----------|--------|
| Modes | **Pass** — Momentum, Breakouts, Long-Term, Avoid (Simple); + Conviction, Pre-Breakout, F&O Coverage (Professional) |
| Filters / search | **Pass** — sector, minimum score, chase exclusion |
| Live scan | **Pass** — shared `useScanRunner` banner |
| Result / universe counts | **Pass** — match count + universe in header |
| Stock Intelligence navigation | **Pass** — CTA from selected row |

**Screenshot:** `discover.webp`

## Stock Intelligence

| Criterion | Result |
|-----------|--------|
| Symbol tested | **ADVANCE** (ADVANCE AGROLIFE) |
| Chart | **Pass** — official daily history with trend interpretation |
| Classification / confidence | **Pass** — unclassified, 45% technical confidence shown |
| Risk Lens | **Pass** — read-only shares, ₹ risk, reward:risk, book heat |
| Trade Plan | **Pass** — via `/api/trade-plan/{symbol}` (no order button) |
| Fundamentals vs evidence | **Pass** — current fundamentals separated; missing historical disclosed when absent |
| Action buttons | **Pass** — refresh fundamentals, news, research data (owner controls, not broker) |
| Missing-state honesty | **Pass** — gaps listed, no fabricated fair value |

**Screenshot:** `stock-intelligence.webp`

## System Health

| Criterion | Result |
|-----------|--------|
| Market ops worker | **Pass** — ONLINE (PID reported) |
| Paper supervisor | **Pass** — state STARTING / heartbeat visible |
| Data sessions | **Pass** — 764 official sessions |
| Observer / failures | **Pass** — 0 active failures shown |
| Blockers | **Pass** — capability notes visible |

**Screenshot:** `system-health.webp`

## Paper Portfolio

| Criterion | Result |
|-----------|--------|
| Sidebar position | **Pass** — under OPERATIONS & EVIDENCE, after System Health |
| Capital / equity | **Pass** — ₹1,00,000 / ₹1,00,000, 0 positions (persisted) |
| No fabricated performance | **Pass** — no win rate / Sharpe without mature closed trades |

**Screenshot:** `paper-portfolio.webp`

## Console

| Item | Result |
|------|--------|
| Runtime errors | **None** observed during acceptance |
| Quirks Mode | **Fixed** — `frontend/index.html` restored with `<!doctype html>` (was fragment-only before handoff completion) |
| Image `id`/`name` warning | **Documented** — no `<img>` in React source; likely browser audit heuristic on decorative/CSS assets or tooling |

## Screenshot index

All committed under `docs/product/acceptance/2026-08-02/`:

| File | Page |
|------|------|
| `home-opportunity-board.webp` | Home — opportunity board |
| `home-live-scan.webp` | Home — active scan banner |
| `home-scan-complete.webp` | Home — scan completion |
| `discover.webp` | Discover |
| `stock-intelligence.webp` | Stock Intelligence (ADVANCE) |
| `system-health.webp` | System Health |
| `paper-portfolio.webp` | Paper Portfolio |

Console-only screenshots (`143c2.webp`, `be1fd.webp`) were **not** committed — no unresolved product defect documented.

## Validation commands (final head)

```bash
python -m py_compile terminal_api.py terminal_product_api.py report_api.py
python -m compileall -q .
python -m pytest

cd frontend && npm install && npm test && npm run build
```

Targeted suites exercised: `test_terminal_operations_api.py`, `test_terminal_api.py`, `test_market_operations.py`, full money-path pytest.

## Honest limitations

- **Browser acceptance passed** for retail discovery → decision journey on persisted data.
- **Connected Zerodha execution was not certified** by this browser pass (no live Kite session).
- **Economic edge remains unproven** — engineering completeness does not imply profitability.
- **Real OMS broker submission remains disconnected** — no LIVE buy path in React.
- **`LIMITED_LIVE` remains blocked.**
- **`LIVE` remains blocked.**

QuantTerm is **not** live-ready for real-money trading and **not** claimed profitable.

## Related documentation

- `docs/RETAIL_PRODUCT_COMPLETION.md` — navigation, architecture owners, live scan semantics
- `frontend/src/scanRunner.ts` — canonical live scan hook
