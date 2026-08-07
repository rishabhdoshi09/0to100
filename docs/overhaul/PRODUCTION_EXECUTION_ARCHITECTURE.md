# Production Execution Architecture — Audit & Plan

Audit-first (Phase 0) for the broker-neutral Execution Management System (EMS), independent
Risk Governor, reconciliation and recovery. Simulator-certified this milestone; **no real
broker is wired**. Reuses concepts, isolates the legacy live path.

## 1. Existing broker/execution modules
- `execution/trade_executor.py` — Kite entry order + **GTT OCO** exit + journal (`place_trade`,
  `paper=True` forces paper). The real live path today.
- `execution/zerodha_broker.py` (legacy engine), `execution/autopilot.py`
  (`_live_enabled()` = `QT_LIVE_ENABLED` temporary interlock), `data/kite_client.py` (KiteConnect).
- `risk/`: `portfolio_risk.py`, `position_sizer.py`, `correlation.py`, `position_manager.py`,
  `iron_lock.py`, etc.

## 2. Safe to reuse
- Risk math (`position_sizer`, `portfolio_risk`, `correlation`) as *inputs* to the Risk Governor.
- Cost model (`execution/cost_model.py`) for expected costs on plans.
- The intelligence `TradeIntent` schema (broker-independent) — the EMS consumes it.

## 3. Components that bypass boundaries (isolate, don't extend)
- `trade_executor.place_trade` places a broker order directly from app/autopilot code — that is
  exactly the coupling the EMS removes. The EMS becomes the ONLY order-lifecycle owner; a real
  Kite adapter (future) wraps `kite_client`, gated behind the Operating Envelope + preflight.

## 4. Existing order lifecycle
Ad-hoc: place entry → wait for fill → place GTT → journal. No explicit state machine, no
idempotency key, no journaled-before-submit, no reconciliation, no recovery.

## 5–7. Reconciliation / restart / protection today
`ensure_pending_gtts()` re-attempts missing GTTs; otherwise no broker-authority reconciliation,
no restart recovery of in-flight orders, protection = "GTT placed" (not broker-verified).

## 8. Existing live-risk controls
`_live_enabled` interlock + paper-only Telegram + portfolio meter. No independent governor, no
daily-loss/drawdown state machine, no owner capital envelope object.

## 9. Critical weaknesses (drive this milestone)
No idempotent submission; timeouts assumed failed; no broker-verified protection; no explicit
order states; no persistent execution ledger surviving crash; no independent risk authority.

## 10. Migration path
New broker-neutral package `ems/` owns: schemas · order state machine · execution ledger ·
idempotency · Risk Governor · Operating Envelope · protection verification · reconciliation ·
recovery · preflight. A deterministic **SimBroker** certifies the loop. The real Kite adapter
is a later, isolated `BrokerAdapter` implementation — NOT wired here.

## 11. Compatibility
Legacy `execution/*` untouched and still paper-locked. Intelligence package keeps zero broker
imports (the EMS lives outside it and only *reads* `TradeIntent`). `USER_APPROVED` unchanged.

## 12. First implementation sequence (this milestone ★)
1. ★ ems/schemas.py (envelope, RiskDecision, ExecutionPlan, Order/Fill/Position records,
   Protection, Reconciliation, Incident) + modes + readiness states.
2. ★ ems/state_machine.py (explicit order lifecycle + legal transitions).
3. ★ ems/broker.py (BrokerAdapter contract + normalized types) + ems/simulator.py (SimBroker).
4. ★ ems/ledger.py (persistent journaled execution ledger) + ems/idempotency.
5. ★ ems/risk_governor.py (independent) + ems/envelope.py (owner-approved) + daily-loss/drawdown.
6. ★ ems/ems.py (manager: submit→risk→plan→journaled submit→fills→protection→position; reconcile;
   recover) + ems/preflight.py (readiness).
7. ★ tests/test_ems.py (boundaries, EMS, risk, protection, reconciliation, recovery, modes) +
   end-to-end certification test.
8. Deferred (documented, cannot enable unsafe live): real Kite adapter; full health/alerts/UI;
   shadow certification; scaling ladder; secrets vault; calendar-freshness fix; the full 75/35
   test matrices (a strong subset is delivered).

## Readiness after this milestone
`LIMITED_LIVE_ARCHITECTURE_READY` → target `LIMITED_LIVE_SIMULATOR_CERTIFIED`. NOT
broker-connected, NOT user-activated. Real live remains impossible without a real adapter,
an approved Operating Envelope, and full preflight.
