# QuantTerm Architecture

QuantTerm is an evidence-driven market-intelligence and paper-trading system. Its job is to find opportunities, decide conservatively, manage risk, record outcomes, and learn without confusing historical evidence with real forward evidence.

This document is the canonical architecture map. If older design notes disagree with it, this document and the executable contracts/tests win.

## The system in one line

```text
Data → Market Intelligence → Decision → Portfolio/Risk → Paper Execution → Outcomes → Evidence/Learning
                                  ↑                                           │
                                  └──────────── future decisions ─────────────┘
```

The user interface and scheduler are adapters around that flow. They do not get independent trading authority.

## Canonical product

- UI: Vite/React desk in `frontend/`
- API: FastAPI product API
- Start command: `bash scripts/run_quantterm_complete.sh`
- Persistent host install: `bash scripts/install_quantterm_host.sh`
- Durable runtime state: outside the Git checkout under the QuantTerm runtime root
- Streamlit and historical compatibility surfaces are not the canonical product
- Live-money execution remains locked unless separately certified and explicitly authorized

## Nine ownership domains

| Domain | Owns | Must not own |
|---|---|---|
| Platform | config, clock, runtime paths, lifecycle | market decisions |
| Data | acquisition, history, quotes, universe, corporate actions, readiness | setup ranking |
| Market Intelligence | scanner, setup detection, regime/context | portfolio mutation |
| Decision | thesis, eligibility, ranking, abstention | order submission |
| Portfolio/Risk | allocation, sizing, exposure, hard risk gates | strategy learning |
| Execution | paper OMS, fills, protection, reconciliation | thesis changes |
| Evidence/Learning | outcomes, calibration, evidence classes | bypassing risk gates |
| Operations | scheduling, jobs, recovery, health | deciding what is a good trade |
| API/Experience | commands and read projections | reimplementing business rules |

## Single-authority rules

The same business fact must not be independently re-derived in several layers.

- Market-data readiness has one canonical authority. Consumers ask whether a snapshot is usable for their purpose; they do not compare dates independently.
- A thesis/version has one canonical identity.
- A decision has one canonical representation and lineage.
- PAPER_FORWARD and HISTORICAL_REPLAY are separate evidence classes.
- The scheduler decides **when** a job runs, never **what trade** should be taken.
- The UI displays and requests actions; it does not decide eligibility.
- Historical replay invokes the same production decision seam using point-in-time inputs. It is not a second strategy.
- Learning can reorder or demote only within explicit policy. It cannot bypass data, risk, evidence, or live-money gates.

## Product workspaces

The operator sees five workspaces:

1. **Today** — what matters now.
2. **Opportunities** — current candidates and company drill-down.
3. **Research** — evidence, experiments, learning, backtests and data quality.
4. **Portfolio** — paper positions, outcomes and risk.
5. **System** — runtime health, jobs, data readiness and manual recovery controls.

Focused diagnostics remain available as tools. They are not separate products.

## Dependency direction

Prefer dependencies that point downward toward stable domain contracts:

```text
frontend
   ↓
api / projections
   ↓
application/domain services
   ↓
stores + external adapters
```

Avoid reverse dependencies such as a data module importing UI code, a scheduler importing React-facing projection logic, or an API route becoming the place where trading rules are calculated.

## Code-placement rule

Before creating a file, class, endpoint, store, worker or dependency:

1. Identify the canonical domain owner.
2. Extend that owner if the requirement fits cleanly.
3. Create a new abstraction only when it removes duplication or establishes a real boundary.
4. Do not create speculative “future” infrastructure.
5. Do not create parallel implementations while the canonical path remains active.
6. Temporary compatibility code must have an explicit reason and deletion path.

A new engineer should be able to infer the system from the directory tree. Generic dumping grounds such as `utils`, `core`, or `product` must not absorb unrelated business responsibilities.

## Safety invariants

- No fake market data.
- Stale or unusable data cannot create new risk.
- Missing evidence stays missing.
- Historical replay never masquerades as real P&L or forward evidence.
- Learning never grants itself live-money permission.
- Telegram actions remain paper-only.
- Every live-capable path must fail closed behind the live-money interlock.
- Risk limits remain independent of model confidence.
- Every surfaced/taken/rejected decision keeps outcome lineage.

## Handoff test

A competent engineer unfamiliar with QuantTerm should be able to answer, from the root docs and directory structure:

- How do I start the system?
- Where does market data enter?
- Which component decides eligibility?
- Which component owns paper execution?
- Where are outcomes settled?
- How does historical evidence differ from forward paper evidence?
- Where is runtime state stored?
- How do I run tests?
- Is live money enabled?

If answering one of these requires reconstructing months of repository history, the architecture is not simple enough.
