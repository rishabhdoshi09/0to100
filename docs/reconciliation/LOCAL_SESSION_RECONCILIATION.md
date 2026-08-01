# Local-session reconciliation, audit & validation

Reconciling this Claude Code workspace against the integrated GitHub branch, then auditing and
validating the integrated system. Real-broker submission stays disconnected; `LIMITED_LIVE`/`LIVE`
stay blocked.

## 1. Local-work preservation report
- **Original branch:** `overhaul/evidence-lab`
- **Original HEAD:** `cae4811bc616841ae5f67661b4e5d46782e7d4dd`
- **Working tree at start:** clean — 0 modified, 0 staged, 0 untracked (`--exclude-standard`).
- **Local-only commits vs integrated branch:** **0** (`git log origin/overhaul/evidence-lab..HEAD`
  empty; `git merge-base --is-ancestor cae4811 origin/overhaul/evidence-lab` → **YES**). My entire
  milestone lineage is already merged into the integration.
- **Rescue branch:** `rescue/quantterm-local-20260801-1658` @ `cae4811` — **pushed** to origin as
  redundant insurance (my lineage is preserved both inside the integration and on this branch).
- **Reconciliation branch:** `agent/reconcile-local-session` @ integrated `7ba6b9b`.

**Conclusion:** there is no unique uncommitted/unpushed local work. Nothing to cherry-pick. The
integration is strictly ahead of my HEAD (265 commits / 199 files: React `frontend/`, `execution/oms`,
`risk/governor`, `operations/`, `product/` workspaces) and *extended* my autonomy foundation (added
`controls.py`, `live_feed.py`, and the data jobs I had left designed-only).

## 2. Reconciliation matrix
Every prior local commit is an ancestor of `7ba6b9b`, so each classifies as **ALREADY_MERGED**.
One new change was authored during this audit:

| file | classification | reason | action | canonical owner | test coverage |
|------|----------------|--------|--------|-----------------|---------------|
| all prior local commits (Kite data, retail UX, autonomy) | ALREADY_MERGED | ancestors of `7ba6b9b` | none | integrated branch | full suite |
| `research/intelligence/evidence_brain.py` | UNIQUE_AND_REQUIRED (audit fix) | latent circular import: module unimportable first → OMS/target-portfolio/terminal API tests failed to collect in isolation | made `growth` import lazy in `build_card()` | integrated branch | 1160 passed; the 9 API test files now collect standalone |

No file was carried over merely for being newer; no other-session code was discarded.

## 3. Architecture audit (sources of truth — no parallel ownership)
1. **React product entry point** — `frontend/src/main.tsx` → `App.tsx` (`experience.tsx`,
   `productViews.tsx`).
2. **API each workspace consumes** — `frontend/src/api.ts` (`/api/dashboard`, `/api/chart`,
   `/api/controls`) + `frontend/src/productApi.ts` (`/api/product-*`, `/api/oms`,
   `/api/risk-governor`, `/api/target-portfolio`, `/api/reconciliation`, `/api/protection`,
   `/api/tca`, `/api/stock-intelligence`). Served by the umbrella app `terminal_product_api:app`
   (imports `terminal_api as core`).
3. **Scanner classifications** — `product/scan_store.py` + `scan/unified_scanner.py` (server-projected;
   React does not rank/classify).
4. **Target Portfolio state** — `research/intelligence/runtime/target_portfolio.py`.
5. **Trade Intents** — `execution/oms/models.py::TradeIntent`, owned by `execution/oms/store.py::OmsStore`.
6. **OMS state store** — `execution/oms/store.py::OmsStore` (durable state machine; validates intents,
   idempotency, fills).
7. **Risk Governor decisions** — `risk/governor.py::evaluate` (authority) + `risk/governor_store.py`
   (`RiskDecisionStore`); `risk/oms_gate.py` bridges OMS→governor.
8. **Simulated PAPER positions** — the intelligence `PaperBook` (`research/auto_research/paper_book.py`),
   persisted at `logs/intelligence/intel_book.json`.
9. **PAPER entry routing** — `execution/paper_pipeline.py`: `OmsStore.ingest_intent` (durable) →
   `oms_gate/governor` approval → simulated ack → `PaperBook.open_position` (the **only** call site) →
   `ProtectionStore` → `TcaStore`.
10. **PAPER exit** — `execution/paper_exit.py`: closes the PaperBook position and synchronizes OMS +
    protection (cancellation) state.
11. **Broker-observed reality** — `execution/reconciliation/` snapshot/internal-state stores.
12. **Reconciliation** — `execution/reconciliation/engine.py` + `service.py`.
13. **Zerodha observation scheduler** — `operations/zerodha_observer.py` +
    `execution/reconciliation/zerodha_cycle.py` (read-only), under the autonomy supervisor.
14. **Direct `TradeIntent → PaperBook` shortcut?** — **No.** `PaperBook.open_position` is reached only
    from `paper_pipeline.py:138`, after risk approval + simulated ack.
15. **Legacy LIVE reachable without unsafe override?** — **No.** `execution/trade_executor.py` defaults
    to PAPER; LIVE requires `legacy_live_enabled()` (explicit unsafe override) *and* the governor, else
    `BLOCKED_LEGACY_LIVE_LOCK`. The new OMS has no broker-submission call.
16. **Duplicate stores/workers/sources of truth?** — none found; the autonomy supervisor is the single
    scheduler/mutation owner (single-instance lock). My earlier `ui/product/` duplicate was already
    removed in a prior milestone.
17. **Local changes still useful after audit** — only the circular-import fix (above).

## 4. Implementation summary (unique local work retained)
- `research/intelligence/evidence_brain.py` — lazy `growth` import to break a load-order circular
  import so every module (incl. the OMS/terminal API test modules) collects in any order.

## 5. Removed / rejected work
- None discarded. All prior local commits were already merged; nothing obsolete or unsafe was carried.

## 6. Validation report (integrated head + fix)
- **Commit under test:** `08fa840` (on `agent/reconcile-local-session`, = `7ba6b9b` + the fix).
- `python -m py_compile terminal_api.py terminal_product_api.py report_api.py` → **OK**
- `python -m compileall -q .` → **exit 0**
- `python -m pytest` → **1160 passed** in ~99s (after installing `fastapi`, `httpx`, `reportlab`,
  which the API/PDF tests require and which were absent from this sandbox).
- The 9 FastAPI/OMS/terminal test modules that previously failed to **collect in isolation** now pass
  standalone after the fix.
- `cd frontend && npm install && npm run build` → **`tsc --noEmit` clean + `vite build` OK** (53
  modules, 431 KB bundle).

## 7. Browser-acceptance report (API/projection level)
Full pixel-level browser acceptance against **real persisted state** is **not achievable in this
sandbox** — there is no Zerodha session/data, and background server processes do not persist across
tool calls in this harness. The read-only projections were exercised directly by launching the
umbrella API (`terminal_product_api:app`) and querying endpoints. Findings (all honest, no fabrication):
- `/api/health` → `autonomy_running:false`, `autonomy_state:"UNKNOWN"` (supervisor not running).
- `/api/data-readiness` → `ready:false`, `snapshot_id:""`, `"No active verified snapshot"`, bhavcopy
  0 sessions/0 symbols — **missing data stays missing**.
- `/api/institutional-readiness` → `system_state:"PAPER_ONLY"`, Economic edge `BLOCKED`, "no aggregate
  score can override a hard safety blocker".
- `/api/oms` → `broker_connected:false`, `submission_enabled:false`, "broker-neutral shadow".
- `/api/risk-governor` → `mode:"SHADOW"`, `authoritative_state_connected:false`.
- `/api/reconciliation` → `certified_for_live:false`, `broker_snapshot_connected:false`,
  `entry_freeze_required:true` — **incomplete state freezes new risk**.
- `/api/protection` → `exchange_adapter_connected:false`, `certified_for_live:false`.
- `/api/target-portfolio` → empty, reported as "not persisted yet" (no invented curve/metrics).

**Remaining product issue:** none observed at the projection layer. A genuine end-to-end browser pass
against a live Zerodha snapshot is the outstanding step (blocked by data/token, not by code).

## 8. Honest completion status (independent)
| Domain | Status |
|--------|--------|
| Engineering foundation | COMPLETE |
| React product | COMPLETE (builds: tsc + vite) |
| Research / data readiness | NOT READY in this environment (no active verified snapshot; missing data reported honestly) |
| Production PAPER | COMPLETE, subject to data & evidence gates (pipeline wired OMS→risk→fill→protection→TCA; tests green) |
| Broker observation | COMPLETE as scheduled read-only; not exercised against a live session here |
| Operational certification | NOT YET PROVEN (no genuine connected smoke; reconciliation not certified_for_live) |
| Economic edge | NOT ESTABLISHED |
| New OMS real broker submission | NOT CONNECTED (by design) |
| LIMITED_LIVE | BLOCKED |
| LIVE | BLOCKED |

Engineering completion does not imply economic profitability. Live remains locked.
