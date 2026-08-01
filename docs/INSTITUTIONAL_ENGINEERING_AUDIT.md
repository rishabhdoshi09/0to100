# QuantTerm Institutional Engineering Audit

Status: repository-traced initial audit  
Integration base: `agent/quantterm-terminal-ui` at `93bb9350e2367aa38dd5554c5810cc6289d6a992`

## Executive finding

QuantTerm has a strong broker-free research and PAPER autonomy core, immutable evidence records,
verified snapshot infrastructure, restart-aware internal paper state, explicit operating controls,
and a dedicated read-only product terminal.

It is not yet a production trading operation. The missing bridge is:

```text
Evidence -> canonical target portfolio -> independent risk -> durable OMS/EMS
-> broker events -> protection -> reconciliation -> TCA -> live evidence
```

Research or UI readiness must never be interpreted as LIVE readiness.

## Repository-traced capability map

| Domain | State | Repository evidence | Required amendment |
|---|---|---|---|
| Product data readiness | PARTIAL | `product/product_readiness.py::build_product_readiness` projects market, history, scan, long-term, news, F&O and operations freshness through a weighted score. | Preserve it as a research/product convenience score only. Never use it to unlock execution. |
| Institutional readiness | ADDED IN PR | `product/institutional_readiness.py::build_institutional_readiness` separates economic, data, research, portfolio, execution, risk, reconciliation, protection and operations gates. | Wire certified capabilities only after their implementations and tests exist. |
| Terminal architecture | COMPLETE FOR READ-ONLY PRODUCT | `terminal_api.py` reads authoritative Python stores and dispatches market operations; `terminal_product_api.py` extends that projection. | Keep the terminal read-only for canonical trading state. Do not add broker-order endpoints to the product layer. |
| Verified market snapshots | COMPLETE/PARTIAL | `terminal_api.py::_snapshot_payload`, `research.intelligence.data.snapshot_store.SnapshotStore`, and autonomy data activation expose an active verified snapshot. | Add full PIT corporate-action, universe, symbol-lineage, calendar and fundamental-availability certification. |
| Strategy/evidence records | STRONG | `research/intelligence/schemas.py` provides frozen deterministic records with provenance and content-derived IDs. | Preserve these schemas as the evidence boundary. Add new schema versions rather than mutating historical meaning. |
| Brain 1 / Brain 2 separation | COMPLETE FOR PAPER | `research/intelligence/runtime/autonomous_loop.py` builds Evidence Cards, then allocation decisions. | Keep strategy evidence separate from portfolio, risk and execution authority. |
| Trade Intent | COMPLETE FOR PAPER CONTRACT | `schemas.py::TradeIntent` is broker-independent and carries strategy, snapshot, entry, stop, target and allocation provenance. | Add desired quantity/target-position linkage only through a versioned portfolio contract. |
| Canonical Target Portfolio | MISSING | `_open_new_positions` iterates strategy decisions and directly creates one intent per top signal. | Aggregate all desired exposures before execution; compare target positions against reconciled actual positions and pending orders. |
| Paper portfolio gate | PARTIAL | `runtime/portfolio_gate.py::check` enforces data, reconciliation, regime, duplicate-symbol, family, cluster and position-count checks. | Include existing book exposure, open-order exposure, cash, sector exposure, turnover and target-portfolio feasibility. |
| Allocation-to-sizing parity | UNSAFE/CORRECTNESS DEFECT | `AllocationConfig` emits 0.25%, 0.5% and 1.0% risk targets; `TradeIntent.intended_risk_pct` records them. `_open_new_positions` then calls `PaperBook.open_position` without the intended risk. `PaperBook` sizes from its fixed `risk_per_trade_pct`. | Make actual paper sizing consume the approved intent risk, with a hard book-level maximum and explicit units. Add regression tests. |
| Paper execution | COMPLETE FOR SIMULATION | `PaperBook` is broker-free, deterministic, cost-aware and enforces per-name, total-risk and position caps. | Keep it as the simulation adapter. Production PAPER should later exercise the same OMS state machine used by LIVE. |
| Internal restart recovery | PARTIAL | `AutoResearchBrain` restores `intel_book`, then calls `runtime_state.reconcile(book)`. | Replace silent restore failures with explicit corruption/quarantine state and operator-visible recovery. |
| Broker OMS/EMS | MISSING | The authoritative loop explicitly refuses LIVE modes and calls only the paper book. | Build a separate broker-neutral execution kernel after Trade Intent; do not add broker calls to the research loop. |
| Independent live Risk Governor | MISSING | Existing gates are inside the PAPER orchestration and use PAPER state. | Build an independent service from reconciled positions, orders, cash, margin, pending exposure, P&L and connectivity. |
| Broker reconciliation | MISSING | Current reconciliation is internal runtime state versus `PaperBook`; no broker order/trade/position/cash authority is present in the traced path. | Add startup, continuous and EOD reconciliation with deterministic repair and ambiguity quarantine. |
| Protection Manager | MISSING FOR PRODUCTION | PAPER exits are simulated inside `PaperBook.mark`. | Add exchange-side entry protection, partial-fill protection, verification, restart recovery and orphan detection. |
| Transaction-cost analysis | PARTIAL | PAPER records slippage and cost through `ExecutionAssessment` and the book cost model. | Store signal, decision, risk approval, submission, acknowledgement and fill timestamps; attribute implementation shortfall. |
| Operations | PARTIAL/STRONG FOR RESEARCH | `operations/market_ops.py`, job store, runtime heartbeat and autonomy supervisor provide separated worker lanes. | Add production execution health, duplicate execution-worker prevention, incident severity, recovery runbooks and failure injection. |
| AI boundary | ACCEPTABLE | The traced authoritative PAPER loop is deterministic and broker-free; product APIs project stored state. | Keep JARVIS explanatory. Never grant it order, risk, capital-limit or certification authority. |

## Canonical state ownership

| State | Canonical owner |
|---|---|
| Historical and active market state | snapshot and official-history stores |
| Strategy definition and evidence | frozen strategy registry and immutable evidence records |
| Strategy allocation proposal | Brain 2 allocation decision |
| Desired portfolio | **new canonical Target Portfolio service** |
| Risk authorisation | **new independent Risk Governor** |
| Order lifecycle | **new durable OMS** |
| Broker translation and events | **new broker adapter / event ingestion layer** |
| Actual executable position | reconciled broker position service |
| Simulated position | `PaperBook` |
| Product display | terminal/product API read-only projections |

No UI, LLM, scanner or strategy module may become a second owner of these states.

## Immediate implementation sequence

### Milestone 0 — fail-closed readiness contract

Implemented in the current branch:

- independent readiness domains
- no aggregate institutional score
- explicit capability certifications
- `LIMITED_LIVE` and `LIVE` blocked by default
- owner approval cannot be inferred
- `/api/institutional-readiness`

### Milestone 1 — allocation-to-sizing parity

Repair the current PAPER correctness defect:

1. Define risk units explicitly: `intended_risk_pct=1.0` means one percent of capital.
2. Extend `PaperBook.open_position` with an approved risk argument while preserving old callers.
3. Cap requested risk at the book's configured maximum.
4. Pass `TradeIntent.intended_risk_pct` into the paper execution adapter.
5. Record approved risk, actual risk amount and resulting quantity.
6. Add tests for 0.25%, 0.5%, 1.0%, invalid values and cap enforcement.

### Milestone 2 — canonical Target Portfolio

Introduce a pure, broker-independent contract containing:

- symbol and direction
- current reconciled quantity
- desired quantity and target weight
- required change
- strategy contribution
- expected net return and holding period
- risk contribution
- sector and correlation contribution
- liquidity/turnover constraints
- inclusion or exclusion reason
- snapshot and strategy provenance

The current loop must construct this object before creating Trade Intents.

### Milestone 3 — production execution kernel

Build outside the research loop:

- durable order state machine
- idempotent submission
- broker acknowledgement and event ingestion
- partial fills
- cancel/replace
- ambiguous-submission quarantine
- restart recovery
- append-only transition journal

### Milestone 4 — independent risk, reconciliation and protection

Implement in this order:

1. reconciled broker state
2. independent Risk Governor
3. Protection Manager
4. continuous reconciliation
5. transaction-cost attribution
6. SHADOW certification
7. production PAPER parity
8. tightly capped `LIMITED_LIVE`

## Non-goals

Do not add:

- new signal families merely to increase strategy count
- direct broker calls in `autonomous_loop.py`
- order endpoints in the terminal UI
- LLM-controlled sizing or risk
- microsecond or order-book infrastructure unsuitable for retail horizons
- a second research, evidence, portfolio or position store

## Definition of the next safe checkpoint

The next checkpoint is reached when:

- PAPER quantity is demonstrably derived from approved allocation risk
- every requested and actual risk amount is auditable
- regression tests prove smaller evidence buckets receive smaller real simulated exposure
- no broker path has been introduced
- institutional readiness continues to report portfolio, execution, risk, reconciliation and protection as blocked until separately certified
