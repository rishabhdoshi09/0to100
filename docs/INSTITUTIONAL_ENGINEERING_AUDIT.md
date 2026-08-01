# QuantTerm Institutional Engineering Audit

Status: engineering foundation complete; operational and economic evidence remain gated  
Integration base: latest `agent/quantterm-terminal-ui`, merged into this branch

## Executive finding

QuantTerm now has a complete broker-neutral institutional PAPER and read-only broker-observation
chain. The system preserves immutable research evidence, constructs a canonical Target Portfolio,
persists every Trade Intent in a durable OMS, applies independent risk, simulates exact approved
fills, verifies quantity-aware protection, reconciles read-only Zerodha state, attributes
transaction costs, and exposes read-only board projections.

The production chain implemented in software is:

```text
Verified research evidence
-> canonical Target Portfolio
-> exact target-versus-current delta
-> durable Trade Intent
-> independent Risk Governor
-> broker-neutral OMS
-> production-parity PAPER fill
-> quantity-aware Protection Manager
-> PAPER exit and protection cancellation
-> TCA
-> scheduled read-only Zerodha observation
-> reconciliation and quarantine
-> read-only board projections
```

This completes the engineering foundation. It does not manufacture operational history or a
profitable strategy.

## Completed engineering domains

| Domain | State | Canonical implementation |
|---|---|---|
| Institutional readiness | COMPLETE | `product/institutional_readiness.py` keeps economic, data, parity, portfolio, execution, risk, reconciliation, protection and operations independent and fail-closed. |
| Terminal product integration | COMPLETE | Latest terminal workspace/navigation is merged; workspace and institutional routes coexist without duplicate registration. |
| Allocation-size parity | COMPLETE | Brain 2 risk reaches the shared sizing contract and exact PaperBook quantity. |
| Canonical Target Portfolio | COMPLETE | Strategy proposals are aggregated and constrained before Trade Intent creation. |
| Durable OMS | COMPLETE FOR PAPER/SHADOW | Legal transitions, idempotency, ambiguous-submission recovery, fills, restart reconstruction and quarantine are persisted. |
| Independent Risk Governor | COMPLETE FOR PAPER/SHADOW | Approve, reduce, reject and freeze decisions are deterministic and persisted. |
| Production PAPER entry | COMPLETE | `execution/paper_pipeline.py` routes intents through OMS, Risk, simulated acknowledgement/fill, Protection and TCA. |
| Production PAPER exit | COMPLETE | `execution/paper_exit.py` synchronises PaperBook exits to OMS `CLOSED` and cancels protection. |
| Reconciliation | COMPLETE FOR READ-ONLY OBSERVATION | Complete broker lanes reconcile; incomplete lanes remain unknown and freeze entries. |
| Protection Manager | COMPLETE FOR PAPER/SHADOW | Partial-fill-aware plans require exact quantity and stop/target identity. |
| Transaction-cost analysis | COMPLETE | Decision, submission and fill shortfall plus fees and opportunity cost are persisted without invented values. |
| Zerodha observation | COMPLETE AND READ-ONLY | Orders, trades, positions, margin and GTT state are observed without mutation. |
| Observation scheduling | COMPLETE | Locked startup, premarket, 15-minute intraday and EOD slots run under a separate read-only process. |
| Failure injection | COMPLETE FOR IMPLEMENTED FOUNDATION | Timeout-after-accept, restart-before-protection, stale data and duplicate-worker ownership are tested. |
| Legacy LIVE containment | COMPLETE | Connected legacy execution is locked by default; governance uncertainty fails closed. |
| Recovery documentation | COMPLETE | Completion contract, observation guide and failure/recovery rules are retained in-repository. |

## Safety defects corrected

1. Brain 2 allocation risk was recorded but not consumed by actual PAPER sizing.
2. Strategy proposals could move too directly toward a PaperBook position without one canonical
   target portfolio.
3. Existing exposure could be silently treated as a satisfied target instead of a duplicate
   economic-exposure refusal.
4. Governance exceptions in the legacy connected-broker route could fail open.
5. Ambiguous submission could be treated like ordinary failure rather than uncertain external
   state.
6. Failed broker endpoints could be mistaken for an empty account or protection book.
7. GTT stop-limit prices could be confused with stop trigger values.
8. A failed duplicate observer could unlink the active observer's lock file.
9. Paper positions could close while durable OMS/protection state remained open.
10. Workspace and observer routes could diverge across stacked terminal branches.

## Canonical state ownership

| State | Canonical owner |
|---|---|
| Historical and active market state | verified snapshot and official-history stores |
| Strategy definition and evidence | frozen strategy registry and immutable evidence records |
| Desired exposure | Target Portfolio records |
| Risk authorisation | independent Risk Governor decision store |
| Order lifecycle | durable OMS |
| Simulated position and outcomes | PaperBook, wrapped by the institutional PAPER adapter |
| Broker reality | append-only read-only Zerodha snapshot ledger |
| Position/order agreement | reconciliation reports and controlled internal repairs |
| Exit protection | Protection Manager |
| Execution economics | TCA store |
| Product display | read-only terminal projections |

No UI, scanner, strategy or LLM owns broker, position, risk or order truth.

## What remains intentionally uncompleted

### Operational certification

This requires elapsed real sessions and cannot be generated by a code commit:

- repeated successful startup/premarket/intraday/EOD observations;
- stable complete Zerodha snapshots;
- zero unexplained broker/internal mismatches;
- restart and database recovery evidence;
- no duplicate workers or orders;
- every simulated fill protected and closed correctly over multiple sessions;
- no unresolved critical incident.

### Economic certification

The historical evidence previously showed no validated production edge. Engineering cannot turn
failed or inconclusive strategies into profitable ones. Economic readiness requires a frozen
strategy to pass realistic net historical, forward, production PAPER, capacity, turnover, TCA,
benchmark and drift gates.

### LIMITED_LIVE and LIVE

Both remain blocked. Real broker submission through the new OMS is intentionally not connected.
LIMITED_LIVE requires completed operational/economic certifications and explicit owner approval.
LIVE additionally requires limited-live evidence and a second approval.

## Honest final classification

- Engineering foundation: **COMPLETE**
- Research and production-parity PAPER: **AVAILABLE SUBJECT TO DATA/EVIDENCE GATES**
- Scheduled read-only Zerodha observation: **COMPLETE**
- Operational evidence: **ACCUMULATING / NOT YET CERTIFIED**
- Economic edge: **NOT ESTABLISHED**
- New OMS real broker submission: **NOT CONNECTED**
- LIMITED_LIVE: **BLOCKED**
- LIVE: **BLOCKED**
