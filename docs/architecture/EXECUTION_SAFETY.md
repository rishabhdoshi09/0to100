# Execution Safety & Live-Trading Graduation

**Status during the overhaul: LIVE execution is DISABLED.** Paper mode remains
available only through the (to-be-)isolated execution service. This document defines
the exact gate that must be cleared before live trading is restored.

## Current enforcement (implemented)

- `execution/autopilot.py::_live_enabled()` — LIVE arming fails closed unless the
  environment flag `QT_LIVE_ENABLED` is explicitly truthy (`1/true/yes/on`). Default =
  disabled. Paper arming is unaffected.
- Invariant retained: LIVE arming additionally requires the exact `ARM LIVE` phrase, a
  working Kite session, and allocation ≤ broker margin.
- Telegram taps remain paper-only (invariant #4); no live-order path from Telegram.

Setting `QT_LIVE_ENABLED` is **necessary but not sufficient** — it is the operator's
explicit acknowledgement that every criterion below has been met and documented.

## Graduation criteria (all mandatory before live capital)

A strategy and the execution stack may go live only when ALL hold:

1. **Forward-paper evidence.** The strategy has ≥300 settled forward paper outcomes
   with a positive, correlation-adjusted expectancy CI lower bound (Evidence Level E4
   per `core/evidence_levels.py`).
2. **Portfolio-ledger validated.** Its historical result comes from the chronological
   portfolio simulator's daily NAV ledger (not per-trade R), on `RESEARCH_GRADE` data.
3. **Broker-state reconciliation.** `execution` reconciles positions/orders/funds
   against Kite on startup and on a schedule; a mismatch halts (fail closed).
4. **Idempotent order flow.** Deterministic client order IDs; an order-state machine;
   duplicate-order prevention; stale-order detection; restart recovery.
5. **Atomic entry+exit.** Every entry ships an exchange-side stop (GTT OCO); GTT
   reconciliation confirms the exit actually exists.
6. **Kill switch + audit log.** A working global kill switch and an append-only order
   audit log.
7. **Isolated service.** Live execution runs in the dedicated execution worker, not in
   Streamlit; explicit arming originates from the trusted Research/Trading UI only.
8. **Slippage OBSERVED, not modelled.** Realised slippage reconciled to ≤1.5× the
   modelled assumption over the paper period.
9. **Governance NORMAL.** `core/governance.py` sentinel is not in DE_RISK/HALT.

Until every item is `implemented` + `tested` + `forward-paper observed`, live stays
disabled. This document is updated as each criterion graduates; `QT_LIVE_ENABLED` is
only ever set by a human who has verified the full list.
