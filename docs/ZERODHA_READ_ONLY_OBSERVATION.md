# Zerodha Read-Only Observation Workflow

## Purpose

This workflow supplies authoritative broker facts to QuantTerm's reconciliation and protection
systems without enabling broker mutation.

It reads only:

- orders
- trades
- positions
- cash and margin
- GTT/protection state

It cannot place, modify or cancel an order or GTT.

## Capture one broker snapshot

```bash
python scripts/capture_zerodha_snapshot.py --require-complete
```

Default ledger:

```text
logs/reconciliation/broker_snapshots.db
```

A failed endpoint is persisted as incomplete evidence. It is never converted into an empty
order, trade, position or protection book. With `--require-complete`, the command exits non-zero
when any account or protection lane is incomplete.

## Run one observation cycle

```bash
python scripts/run_zerodha_observation.py \
  --internal-cash <INTERNAL_CASH> \
  --internal-margin <INTERNAL_AVAILABLE_MARGIN> \
  --require-entry-ready
```

Default state stores:

```text
logs/oms/orders.db
logs/protection/plans.db
logs/reconciliation/broker_snapshots.db
logs/reconciliation/reports.db
```

The cycle performs, in order:

1. read-only Zerodha capture
2. append-only snapshot persistence
3. internal position reconstruction from durable OMS fills
4. broker order/trade/position/cash/margin reconciliation
5. deterministic internal ACK/fill/reject/cancel catch-up when safe
6. conflict quarantine when state is ambiguous
7. partial-fill-aware protection verification
8. one explicit `entries_allowed` decision with reason codes

## Comparison-only mode

```bash
python scripts/run_zerodha_observation.py --no-repairs
```

This persists the broker snapshot and reconciliation report but does not apply deterministic
catch-up repairs to the OMS.

## Fail-closed rules

New entries remain blocked when any of the following is true:

- orders, trades, positions, margins or GTT lane is incomplete
- a broker order has no internal OMS owner
- an internal order expected at the broker is missing
- fills disagree between OMS, broker order book and broker trade book
- position quantity differs
- cash or margin differs beyond tolerance
- a position is not fully protected
- an OMS order is unknown, quarantined or recovery-required
- protection is missing, under-sized, stale or orphaned
- an internal repair fails

## What this does not do

This workflow does not:

- submit an order
- cancel an order
- place or alter a GTT
- unlock `LIMITED_LIVE` or `LIVE`
- certify the Risk Governor, OMS, reconciliation or protection service for live use
- infer ownership by symbol when a broker reference is absent

Broker submission remains disconnected until multi-session shadow evidence, failure-injection
results and explicit institutional certifications are complete.
