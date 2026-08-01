# QuantTerm Institutional Engineering Audit

Status: repository-traced audit with readiness, sizing and Target Portfolio milestones implemented  
Integration base: `agent/quantterm-terminal-ui` at `93bb9350e2367aa38dd5554c5810cc6289d6a992`

## Executive finding

QuantTerm has a strong broker-free research and PAPER autonomy core, immutable evidence records,
verified snapshot infrastructure, restart-aware internal paper state, explicit operating controls,
and a dedicated read-only product terminal.

The authoritative PAPER chain is now:

```text
Evidence
-> Brain 2 allocation proposal
-> canonical Target Portfolio
-> exact target-versus-current quantity delta
-> Trade Intent
-> PaperBook revalidation and execution
-> durable evidence
```

It is still not a production trading operation. The remaining production bridge is:

```text
Target Portfolio -> independent risk -> durable OMS/EMS -> broker events
-> protection -> reconciliation -> TCA -> limited-live evidence
```

Research or UI readiness must never be interpreted as LIVE readiness.

## Repository-traced capability map

| Domain | State | Repository evidence | Required amendment |
|---|---|---|---|
| Product data readiness | PARTIAL | `product/product_readiness.py::build_product_readiness` projects market, history, scan, long-term, news, F&O and operations freshness through a weighted score. | Preserve it as a research/product convenience score only. Never use it to unlock execution. |
| Institutional readiness | IMPLEMENTED | `product/institutional_readiness.py::build_institutional_readiness` separates economic, data, research, parity, portfolio, execution, risk, reconciliation, protection and operations gates. | Wire certifications only after repository-backed implementations and tests exist. |
| Terminal architecture | COMPLETE FOR READ-ONLY PRODUCT | `terminal_api.py` reads authoritative Python stores; `terminal_product_api.py` projects readiness and the latest persisted Target Portfolio. | Keep the terminal read-only for canonical trading state. Do not add broker-order endpoints to the product layer. |
| Verified market snapshots | COMPLETE/PARTIAL | Snapshot activation and official history exist. | Certify PIT corporate actions, universe, symbol lineage, calendar and fundamental availability independently. |
| Strategy/evidence records | STRONG | `research/intelligence/schemas.py` provides frozen deterministic records with provenance and content-derived IDs. | Preserve schemas as the evidence boundary; version material semantic changes. |
| Brain 1 / Brain 2 separation | COMPLETE FOR PAPER | `autonomous_loop.py` builds Evidence Cards and then allocation decisions. | Keep strategy evidence separate from portfolio, risk and execution authority. |
| Canonical Target Portfolio | IMPLEMENTED FOR PAPER | `runtime/target_portfolio.py` aggregates ranked proposals into immutable `TargetPosition` and `TargetPortfolio` records before any Trade Intent exists. | Add sector, liquidity and turnover constraints; later feed pending exposure from the durable OMS instead of empty PAPER maps. |
| Target-versus-actual transition | IMPLEMENTED FOR PAPER | Current and pending quantities are subtracted from desired quantity. Duplicate economic exposure, cash, total risk, family, cluster and position-count blocks carry explicit reason codes. | Replace PAPER state with broker-reconciled state for production certification. |
| Trade Intent | VERSIONED CONTRACT | `TradeIntent` schema v2 references exact Target Portfolio and Target Position IDs and carries current, pending, desired and required quantities. | Preserve broker independence. A broker adapter may consume the intent but never rewrite its meaning. |
| Allocation-to-sizing parity | CORRECTED | Shared `position_sizing.py` is used by portfolio planning and PaperBook. PaperBook revalidates the exact planned quantity. | Keep percentage-point units and exact-quantity regression tests frozen. |
| Paper execution | COMPLETE FOR SIMULATION | `PaperBook` is broker-free, deterministic, cost-aware and enforces house limits. | Production PAPER should later exercise the same durable OMS state machine as LIVE. |
| Internal restart recovery | PARTIAL | The paper book and runtime state restore and reconcile after restart. | Replace silent corruption handling with explicit quarantine and operator-visible recovery. |
| Legacy live executor | LOCKED / UNSAFE OVERRIDE ONLY | `execution/trade_executor.py` cannot use connected-broker LIVE by default. It requires `QT_ENABLE_UNSAFE_LEGACY_LIVE=1`; governance uncertainty blocks and ambiguous submission becomes `RECOVERY_REQUIRED`. | Retire this route after the institutional OMS is operational. Never certify it for LIVE. |
| Broker OMS/EMS | MISSING | No durable broker-neutral order lifecycle owns live order state. | Build a separate execution kernel after Trade Intent; do not add broker calls to the research loop. |
| Independent live Risk Governor | MISSING | Current constraints operate within PAPER construction and simulation state. | Build an independent service from reconciled positions, orders, cash, margin, pending exposure, P&L and connectivity. |
| Broker reconciliation | MISSING | Internal paper reconciliation exists; broker order/trade/position/cash reconciliation does not. | Add startup, continuous and EOD reconciliation with deterministic repair and ambiguity quarantine. |
| Protection Manager | MISSING FOR PRODUCTION | PAPER exits are simulated; the legacy GTT path is not a durable protection service. | Add partial-fill protection, verification, restart recovery and orphan detection. |
| Transaction-cost analysis | PARTIAL | PAPER records slippage and explicit costs. | Store signal, decision, approval, submission, acknowledgement and fill timestamps; attribute implementation shortfall. |
| Operations | PARTIAL/STRONG FOR RESEARCH | Separated worker lanes, job stores, heartbeats and autonomy supervision exist. | Add production execution health, duplicate execution-worker prevention, incident severity, recovery runbooks and failure injection. |
| AI boundary | ACCEPTABLE | The authoritative PAPER loop is deterministic and broker-free; product APIs only project stored state. | Keep JARVIS explanatory. Never grant it order, risk, capital-limit or certification authority. |

## Canonical state ownership

| State | Canonical owner |
|---|---|
| Historical and active market state | snapshot and official-history stores |
| Strategy definition and evidence | frozen strategy registry and immutable evidence records |
| Strategy allocation proposal | Brain 2 allocation decision |
| Desired portfolio | `TargetPortfolio` and `TargetPosition` records in the intelligence event store |
| Simulated executable position | `PaperBook` |
| Risk authorisation | **new independent Risk Governor** |
| Order lifecycle | **new durable OMS** |
| Broker translation and events | **new broker adapter / event-ingestion layer** |
| Actual executable position | **new reconciled broker-position service** |
| Product display | terminal/product API read-only projections |

No UI, LLM, scanner or strategy module may become a second owner of these states.

## Completed milestones

### Milestone 0 — fail-closed readiness contract

- Independent readiness domains
- No aggregate institutional score
- Evidence-backed capability certifications
- PIT-data and research-to-production parity gates
- `LIMITED_LIVE` and `LIVE` blocked by default
- `/api/institutional-readiness`

### Milestone 1 — allocation-to-sizing parity

- `1.0` explicitly means one percent of capital
- Brain 2 risk reaches PaperBook
- Shared pure sizing logic
- Exact quantity revalidation
- Requested and actual risk are auditable
- Legacy snapshots remain readable

### Milestone 2 — canonical Target Portfolio

- Immutable `TargetPosition` and `TargetPortfolio`
- Current, pending, desired and required quantities remain separate
- Cash and worst-case pending exposure are explicit inputs
- Portfolio-wide total-risk, family, cluster and position-count limits
- Duplicate-symbol ownership and duplicate proposal detection
- Trade Intents originate only from executable target deltas
- Exact Target Portfolio and Target Position provenance in every intent
- Latest persisted portfolio projected through `/api/target-portfolio`
- Product dashboard composition remains read-only

### P0 containment — legacy live executor

- Connected-broker legacy LIVE locked by default
- Unsafe compatibility override is environment-only and not exposed to UI
- Governance exceptions fail closed
- DE_RISK quantity reduction is enforced
- Ambiguous broker response enters `RECOVERY_REQUIRED`
- No new autonomous or terminal broker route was created

## Next milestone — durable broker-neutral OMS

Build outside the research loop and without connecting it to Kite initially:

1. Durable SQLite order and transition ledger
2. Explicit lifecycle:
   `PROPOSED -> RISK_APPROVED -> SUBMISSION_PENDING -> BROKER_ACKNOWLEDGED -> PARTIALLY_FILLED -> FILLED -> PROTECTION_PENDING -> PROTECTED -> EXIT_PENDING -> CLOSED`
3. Exception states:
   `REJECTED`, `CANCELLED`, `EXPIRED`, `UNKNOWN`, `QUARANTINED`, `RECOVERY_REQUIRED`
4. Validated state-transition matrix
5. Idempotency key from immutable Trade Intent
6. Duplicate-submission prevention
7. Partial-fill aggregation
8. Ambiguous-submission quarantine
9. Restart reconstruction and deterministic replay
10. Read-only OMS projection for the board

Broker submission remains out of scope until this state machine, independent risk and reconciliation are tested.

## Later sequence

1. Reconciled broker-state model
2. Independent Risk Governor
3. Broker adapter and event ingestion
4. Protection Manager
5. Continuous reconciliation
6. Transaction-cost attribution
7. SHADOW certification
8. Production PAPER parity
9. Tightly capped `LIMITED_LIVE`

## Non-goals

Do not add:

- New signal families merely to increase strategy count
- Direct broker calls in `autonomous_loop.py`
- Order endpoints in the terminal UI
- LLM-controlled sizing or risk
- Microsecond or order-book infrastructure unsuitable for retail horizons
- A second research, evidence, target-portfolio or position store
- Any certification merely because a class or dashboard panel exists

## Definition of the next safe checkpoint

The next checkpoint is reached when:

- every Trade Intent can create exactly one durable OMS order intent
- repeated ingestion cannot duplicate an order
- illegal lifecycle transitions fail closed
- an uncertain broker submission cannot be automatically retried
- partial fills reconstruct correctly after restart
- OMS state can be projected read-only to the terminal
- no broker API call is introduced
- institutional readiness continues to report execution, risk, reconciliation and protection as blocked
