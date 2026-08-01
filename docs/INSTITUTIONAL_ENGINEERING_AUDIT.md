# QuantTerm Institutional Engineering Audit

Status: repository-traced initial audit and first correctness milestone  
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
| Institutional readiness | IMPLEMENTED | `product/institutional_readiness.py::build_institutional_readiness` separates economic, data, research, parity, portfolio, execution, risk, reconciliation, protection and operations gates. | Wire certifications only after repository-backed implementations and tests exist. |
| Terminal architecture | COMPLETE FOR READ-ONLY PRODUCT | `terminal_api.py` reads authoritative Python stores and dispatches market operations; `terminal_product_api.py` extends that projection. | Keep the terminal read-only for canonical trading state. Do not add broker-order endpoints to the product layer. |
| Verified market snapshots | COMPLETE/PARTIAL | `terminal_api.py::_snapshot_payload`, `research.intelligence.data.snapshot_store.SnapshotStore`, and autonomy data activation expose an active verified snapshot. | Certify PIT corporate actions, universe, symbol lineage, calendar and fundamental availability independently. |
| Strategy/evidence records | STRONG | `research/intelligence/schemas.py` provides frozen deterministic records with provenance and content-derived IDs. | Preserve these schemas as the evidence boundary. Add new schema versions rather than mutating historical meaning. |
| Brain 1 / Brain 2 separation | COMPLETE FOR PAPER | `research/intelligence/runtime/autonomous_loop.py` builds Evidence Cards, then allocation decisions. | Keep strategy evidence separate from portfolio, risk and execution authority. |
| Trade Intent | COMPLETE FOR PAPER CONTRACT | `schemas.py::TradeIntent` is broker-independent and carries strategy, snapshot, entry, stop, target and allocation provenance. | Add desired quantity/target-position linkage only through a versioned portfolio contract. |
| Canonical Target Portfolio | MISSING | `_open_new_positions` still iterates strategy decisions and creates one intent per top signal. | Aggregate desired exposures before execution; compare targets against reconciled positions and pending orders. |
| Paper portfolio gate | PARTIAL | `runtime/portfolio_gate.py::check` enforces data, reconciliation, regime, duplicate-symbol, family, cluster and position-count checks. | Include existing book exposure, open-order exposure, cash, sector exposure, turnover and target-portfolio feasibility. |
| Allocation-to-sizing parity | CORRECTED | `PaperBook.open_intent` now consumes `TradeIntent.intended_risk_pct`; requests are capped by the book-level maximum and requested/approved risk is auditable. Runtime events preserve requested risk, actual risk and quantity. | Keep this contract covered by unit and runtime-path regression tests. Do not silently change its percentage-point units. |
| Paper execution | COMPLETE FOR SIMULATION | `PaperBook` is broker-free, deterministic, cost-aware and enforces per-name, total-risk and position caps. | Keep it as the simulation adapter. Production PAPER should later exercise the same OMS state machine used by LIVE. |
| Internal restart recovery | PARTIAL | `AutoResearchBrain` restores `intel_book`, then calls `runtime_state.reconcile(book)`. | Replace silent restore failures with explicit corruption/quarantine state and operator-visible recovery. |
| Broker OMS/EMS | MISSING | The authoritative loop explicitly refuses LIVE modes and calls only the paper book. | Build a separate broker-neutral execution kernel after Trade Intent; do not add broker calls to the research loop. |
| Independent live Risk Governor | MISSING | Existing gates are inside the PAPER orchestration and use PAPER state. | Build an independent service from reconciled positions, orders, cash, margin, pending exposure, P&L and connectivity. |
| Broker reconciliation | MISSING | Current reconciliation is internal runtime state versus `PaperBook`; no broker order/trade/position/cash authority is present in the traced path. | Add startup, continuous and EOD reconciliation with deterministic repair and ambiguity quarantine. |
| Protection Manager | MISSING FOR PRODUCTION | PAPER exits are simulated inside `PaperBook.mark`. | Add exchange-side entry protection, partial-fill protection, verification, restart recovery and orphan detection. |
| Transaction-cost analysis | PARTIAL | PAPER records slippage and cost through `ExecutionAssessment` and the book cost model. | Store signal, decision, risk approval, submission, acknowledgement and fill timestamps; attribute implementation shortfall. |
| Operations | PARTIAL/STRONG FOR RESEARCH | `operations/market_ops.py`, job store, runtime heartbeat and autonomy supervisor provide separated worker lanes. | Add production execution health, duplicate execution-worker prevention, incident severity, recovery runbooks and failure injection. |
| AI boundary | ACCEPTABLE | The authoritative PAPER loop is deterministic and broker-free; product APIs project stored state. | Keep JARVIS explanatory. Never grant it order, risk, capital-limit or certification authority. |

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

## Implementation sequence

### Milestone 0 — fail-closed readiness contract — COMPLETE

Implemented in the current branch:

- independent readiness domains
- no aggregate institutional score
- explicit capability certifications with timestamps and evidence
- PIT-data and research-to-production parity certifications
- `LIMITED_LIVE` and `LIVE` blocked by default
- owner approval cannot be inferred
- `/api/institutional-readiness`

### Milestone 1 — allocation-to-sizing parity — COMPLETE

Implemented in the current branch:

1. `intended_risk_pct=1.0` explicitly means one percent of capital.
2. `PaperBook.open_position` accepts an approved risk budget while preserving old callers.
3. Requested risk is capped at the configured book maximum.
4. `PaperBook.open_intent` consumes `TradeIntent.intended_risk_pct`.
5. The paper runtime records requested risk, approved risk, actual risk amount and quantity.
6. Regression tests cover exploratory versus established sizing, caps, invalid risk, legacy snapshots and the full intent-to-book runtime path.

### Milestone 2 — canonical Target Portfolio — NEXT

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

- a canonical Target Portfolio exists as durable, versioned, broker-independent state
- desired quantities are derived from strategy allocations and portfolio-wide constraints
- current positions and pending exposure are included before calculating required changes
- Trade Intents are generated only from target-versus-actual deltas
- exclusions and clipped targets carry explicit reason codes
- no broker path has been introduced
- institutional readiness continues to report execution, risk, reconciliation and protection as blocked until separately certified
