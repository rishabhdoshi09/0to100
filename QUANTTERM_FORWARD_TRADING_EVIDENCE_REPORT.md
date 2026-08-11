# QUANTTERM FORWARD TRADING EVIDENCE REPORT

> PAPER forward-evidence architecture. **LIVE trading is NOT enabled.**
> EXP-FUND-03 remains INCONCLUSIVE_FOLLOWUP / not paper-enabled from status alone.

## Plain English (read this first)

- **Paper trading is ON** for allowlisted strategies.
- QuantTerm can take **simulated** trades automatically (no per-trade click).
- Those strategies are being **observed, not trusted**.
- **No real money** is being used.
- Earnings-growth (EXP-FUND-03) is **not** auto paper-enabled just because discovery once said CONFIRMED.

---

## 1. Existing infrastructure reused

| Need | Reused (authoritative) |
|---|---|
| Orchestration | `research/intelligence/runtime/autonomous_loop.py` (`run_intelligence_cycle`) |
| Mode ladder | `research/intelligence/runtime/modes.py` (`PAPER_AUTO` + alias `PAPER_FORWARD_EVIDENCE`) |
| Trade Intent / Target Portfolio | `schemas.TradeIntent`, `runtime/target_portfolio.py` |
| OMS + risk + protection + TCA | `execution/oms`, `risk/governor`, `execution/protection`, `execution/tca` |
| Paper fills | `execution/paper_pipeline.py` (`BROKER_MUTATIONS_ENABLED=False`) + `PaperBook` |
| Outcomes (intel) | `outcome_decoder` → `OutcomeObservation(split=forward)` |
| Scientific memory | `research/scientific_memory.py` |
| Strategy / experiment registries | existing — not duplicated |
| Zerodha read-only | existing observer/recon — ingestion-ready, submission blocked |
| Retail autopilot / Telegram | left as secondary UX; evidence authority = institutional path |
| Evidence levels | `core/evidence_levels.py` (no auto-promote from trade count alone) |

**Not created:** second OMS, second portfolio store, second risk engine, second scientific memory.

---

## 2. PAPER auto-trading architecture

```
Market data / snapshot
  → frozen strategy runtime signals
  → Evidence Card + Allocation Decision
  → Target Portfolio → TradeIntent
  → PAPER allowlist gate (paper_enabled, live_enabled=False)
  → PaperBook / PaperExecutionPipeline (OMS→risk→fill→protection→TCA)
  → decision snapshot freeze
  → exits → OutcomeObservation + ForwardOutcome(evidence_source=PAPER_FORWARD)
  → scientific memory (WATCH, source-labelled)
```

Conceptual mode label: **PAPER_FORWARD_EVIDENCE** (same execution rights as `PAPER_AUTO`; live modes still raise `LiveModeDisabled`).

---

## 3. Paper policy authorization model

Durable allowlist: `logs/forward_evidence/paper_policy_allowlist.json`  
Code: `research/forward_evidence/policy_allowlist.py`

Fields: `research_policy_id`, `version`, `paper_enabled`, `live_enabled`, `frozen_config_hash`, `approved_at`, `approval_reason`, `evidence_source`, `scientific_status`, `paper_observation_status`.

**Critical:** `paper_enabled != live_enabled` is normal. This module **never** sets `live_enabled=True`.

Scientific PASS / CONFIRMED / INCONCLUSIVE / ROBUST_CONFIRMED do **not** auto-grant paper observation.

---

## 4. Live authorization separation

| Gate | Status |
|---|---|
| `modes.assert_no_live` | LIVE modes raise |
| `PaperExecutionPipeline.BROKER_MUTATIONS_ENABLED` | `False` |
| Allowlist `may_live_trade()` | always `False` |
| Zerodha observer | read-only |
| Legacy live unlock envs | untouched |

Code may **record** future LIVE outcomes with `evidence_source=LIVE` without enabling submission.

---

## 5. Decision snapshot schema

`logs/forward_evidence/decision_snapshots.jsonl`  
`research/forward_evidence/decision_snapshot.py`

Frozen at open: `decision_id`, policy id/version, config hash, timestamp, symbol, market price, features, fundamentals/events, AVAILABLE_AT state, market/portfolio/risk state, score, entry/stop/target, qty, evidence state, snapshot IDs, git provenance, cycle/intent ids.

Idempotent on restart (same `decision_id` not duplicated). Later outcomes never recompute the decision with future data.

---

## 6. Forward outcome schema

`logs/forward_evidence/forward_outcomes.jsonl`  
`research/forward_evidence/outcome_ledger.py`

Every row carries **`evidence_source`** ∈  
`HISTORICAL_BACKTEST` | `PAPER_FORWARD` | `LIMITED_LIVE` | `LIVE`.

Plus decision/intent/order/fill ids, prices, qty, gross/net pnl, R, fees, slippage, MAE/MFE, hold, exit reason, regime/portfolio context, outcome status (`STOPPED`/`TARGET`/`TIME_EXIT`/…).

Paper and live rows are **separate observations** (different `outcome_id`).

---

## 7. Paper fill methodology

Uses existing production-parity path:

- `PaperExecutionPipeline` / `PaperBook` with India cash costs + slippage bps when wired
- OMS durable order lifecycle; paper IDs prefixed `paper-`
- Protection plans + TCA assessments
- Gap-aware exits in `PaperBook.mark`
- Insufficient market data → no fabricated fills (`DATA_UNKNOWN` vocabulary available)

---

## 8. Live broker-truth design (future-ready)

When LIVE is separately owner-authorized later:

- Ingest submitted price, accept time, exchange order id, fill qty/price, partials, rejects, latency, charges, GTT/protection, recon state via existing OMS + Zerodha observation cycle
- Link to originating `decision_id` / `intent_id`
- Store as `evidence_source=LIVE` **without overwriting** paper rows

---

## 9. Paper-vs-live comparison

`research/forward_evidence/paper_vs_live.py`

Compares n, fill rate, slippage, expectancy, pnl; emits plain language.  
If no live rows → **`NO LIVE EVIDENCE YET`** (never shows a fake healthy live column).

---

## 10. Scientific-memory integration

`research/forward_evidence/memory_bridge.py`

On closed paper trades, records a **WATCH** belief labelled `[source=PAPER_FORWARD; live_authorized=False]`.  
Does **not** mutate frozen strategy rules. Historical vs paper stats stay separated in `reporting.policy_report()`.

---

## 11. UI changes

`ui/retail_trade_market.py` (Automatic Paper Trading page):

- Clear banner: paper armed / live not authorized / outcomes count
- “What are we learning?” plain lines
- Per-policy table: scientific status · paper observation · live status · trades · expectancy · reliability
- Paper vs real money panel with **NO LIVE EVIDENCE YET**
- Technical expander for paths / allowlist / hashes

---

## 12. Restart / idempotency safety

- Cycle completion: `runtime_state.is_cycle_done(cycle_id)`
- Decision snapshots & outcomes: content-hashed ids; append skipped if present
- OMS intent ingest already crash-resumable
- Paper book persisted under `logs/intelligence/intel_book.json`
- Allowlist + system status under `logs/forward_evidence/`

---

## 13. Tests

`tests/test_forward_trading_evidence.py` — allowlist, paper≠live, FUND-03 deny, snapshot freeze idempotency, outcome source separation, paper-vs-live empty live, mode live block, hooks, ensure_armed, reporting separation.

Also green: `test_paper_auto`, `test_intelligence_runtime`, `test_paper_execution_pipeline`, `test_oms_store`.

---

## 14. Policies currently paper-enabled (observation)

| Policy | Scientific status | Paper observation | Live |
|---|---|---|---|
| cross_sectional_momentum | UNPROVEN | ACTIVE | NOT AUTHORIZED |
| breakout | UNPROVEN | ACTIVE | NOT AUTHORIZED |
| pullback | UNPROVEN | ACTIVE | NOT AUTHORIZED |
| trend_following | UNPROVEN | ACTIVE | NOT AUTHORIZED |
| relative_strength | UNPROVEN | ACTIVE | NOT AUTHORIZED |
| sector_rotation | UNPROVEN | ACTIVE | NOT AUTHORIZED |

---

## 15. Policies explicitly not paper-enabled

| Policy | Scientific status | Paper observation | Reason |
|---|---|---|---|
| EXP-FUND-03 | INCONCLUSIVE_FOLLOWUP | DENIED | Status ≠ paper authority |
| EXP-FUND-03-FOLLOWUP | INCONCLUSIVE_FOLLOWUP | DENIED | Not a trade policy |
| EXP-FUND-01/02/04 | INCONCLUSIVE | DENIED | HOLD_NO_TUNING |
| earnings_growth | INCONCLUSIVE_FOLLOWUP | DENIED | Alias deny |
| low_vol | FAIL | DENIED | Closed branch |

---

## 16. Confirmation — no live broker authority changed

- No new broker submission path
- `LIMITED_LIVE` / `LIVE` still disabled
- `BROKER_MUTATIONS_ENABLED` remains `False`
- Allowlist forces `live_enabled=False`
- System status reports `live_trading_enabled: false`

---

## 17. Plain-English operating guide

1. Keep QuantTerm + autonomy supervisor running in market hours.
2. Paper mode stays armed for allowlisted families (persisted).
3. Eligible decisions become paper trades automatically via the existing loop.
4. Outcomes land in `logs/forward_evidence/` labelled `PAPER_FORWARD`.
5. Review the Automatic Paper Trading page for “what we are learning.”
6. Do **not** treat paper results as live permission.
7. Earnings-growth needs an explicit future paper-observation approval if ever observed — not this cycle.

---

## Status card

| Field | Value |
|---|---|
| PAPER AUTO-TRADING READY? | **YES** |
| PAPER MODE ARMED? | **YES** |
| CURRENT PAPER POLICIES | momentum, breakout, pullback, trend_following, relative_strength, sector_rotation |
| LIVE TRADING ENABLED? | **NO** |
| LIVE DATA INGESTION READY? | **YES** (observer/recon; submission blocked) |
| FORWARD EVIDENCE LOCATION | `logs/forward_evidence/` |
| NEXT OPERATIONAL ACTION | Keep autonomy supervisor running; collect paper outcomes; do not unlock live |

Code entrypoints: `research/forward_evidence/` · arm via `ensure_armed()`.
