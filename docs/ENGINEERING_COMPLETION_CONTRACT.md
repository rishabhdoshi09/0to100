# QuantTerm Engineering Completion Contract

## Purpose

QuantTerm must never use the word **complete** without naming the completion domain.
Engineering capability, operational evidence, economic edge and live-authorisation are
independent gates.

## 1. Engineering foundation — code-complete when CI is green

The institutional engineering foundation is code-complete only when the repository and its
canonical test suite prove all of the following:

- immutable point-in-time research inputs and strategy provenance
- canonical Target Portfolio before Trade Intent creation
- exact target-versus-current quantity deltas
- durable broker-neutral OMS with idempotency and recovery states
- independent Risk Governor decisions
- production PAPER entries through OMS, Risk, simulated fill, Protection and TCA
- production PAPER exits through OMS close and protection cancellation
- read-only Zerodha orders, trades, positions, margin and GTT observation
- incomplete broker lanes represented as unknown, never empty
- startup, 15-minute intraday and EOD read-only observation slots
- duplicate observer prevention with an ownership-safe process lock
- deterministic reconciliation and quarantine of ambiguity
- partial-fill-aware protection state
- transaction-cost and execution-latency attribution
- failure-injection coverage for timeout, restart-before-protection, stale data and duplicate workers
- read-only board projections with no trading-state mutation endpoints
- legacy connected-broker execution locked by default
- full pytest, integration checks and compile-all passing

A green engineering foundation means the software contracts exist and agree. It does not prove
that the strategy makes money or that live operation is safe enough.

## 2. Operational evidence — time-dependent

Operational readiness requires evidence produced by the running system, not source-code claims:

- multiple successful startup reconciliations
- complete premarket, intraday and EOD broker snapshots
- zero unexplained broker/internal position mismatches
- successful restart recovery after simulated failures
- no duplicate orders or workers
- every simulated fill protected for the exact cumulative quantity
- stable database backup and restore checks
- measured alert and recovery response
- no unresolved critical incident

These observations must be retained append-only. Missing sessions cannot be backfilled with
synthetic success records.

## 3. Economic readiness — market-evidence dependent

A strategy is economically ready only after it independently passes:

- registered and frozen hypothesis
- realistic net historical evidence
- multiple-testing controls
- point-in-time universe and corporate-action integrity
- forward observation
- production PAPER evidence through the institutional execution chain
- capacity and turnover assessment
- implementation-shortfall limits
- benchmark-relative performance
- drift and regime checks

Engineering quality cannot promote a losing or inconclusive strategy.

## 4. LIMITED_LIVE readiness — explicit approval required

`LIMITED_LIVE` remains blocked until all required engineering, operational, data, parity,
economic, risk, reconciliation and protection certifications contain:

- `certified: true`
- a certification timestamp
- concrete evidence references
- explicit owner approval

LIMITED_LIVE must use a small capital envelope, hard expiry, independent kill switch, verified
exchange-side protection and continuous reconciliation.

## 5. LIVE readiness — not automatic

LIVE cannot be unlocked merely because LIMITED_LIVE ran without a software crash. It additionally
requires sufficient limited-live economic and operational evidence and a second explicit owner
approval. There is no UI-only shortcut and no LLM authority.

## Current honest classification

When the branch CI is green:

- **Engineering foundation:** complete for broker-neutral PAPER and read-only broker observation
- **Research/PAPER operation:** available subject to data and evidence gates
- **Operational certification:** accumulating, not yet proven over multiple sessions
- **Economic edge:** not established by engineering
- **LIMITED_LIVE:** blocked
- **LIVE:** blocked
- **real broker submission through the new OMS:** intentionally not connected

This classification is a safety property, not unfinished wording.
