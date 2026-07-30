"""
🎛️ Execution Management System (Phase 2) — the sole owner of the order lifecycle.

Broker-neutral, strategy-neutral, brain-neutral, persistent, idempotent, restartable. It turns
an approved TradeIntent into a live order ONLY after the independent Risk Governor approves and
the Operating Envelope permits, journals the intended submission BEFORE calling the broker,
verifies protection against broker state (not a local flag), handles partial fills on actual
quantity, reconciles against the broker as authority, and recovers on restart without
duplicating orders.

No strategy or brain can submit an order — only the EMS, and only in a live mode with a
user-approved envelope. SHADOW plans but never submits; PAPER_AUTO/HALTED never submit live.
"""
from __future__ import annotations

import time

from ems import schemas as SC
from ems import state_machine as SM
from ems.broker import BrokerTimeout, BrokerError
from ems.risk_governor import RiskGovernor


class SubmissionRefused(Exception):
    pass


class EMS:
    def __init__(self, broker, ledger, governor: RiskGovernor, *, adapter_id="sim"):
        self.broker = broker
        self.ledger = ledger
        self.governor = governor
        self.adapter_id = adapter_id
        self.new_entries_blocked = False              # set by reconciliation / recovery / risk

    # ── the one entry point: intent → live order ─────────────────────────────────
    def submit_intent(self, intent, *, envelope, mode: str, portfolio: dict, account: dict,
                      limit_price: float, stop_price: float, target_price: float,
                      expected_risk_pct: float, family: str = "", sector: str = "",
                      cluster: str = "") -> SC.OrderStateRecord:
        idem = f"idem-{getattr(intent, 'record_id', '')}"
        # idempotency: an order for this intent already exists ⇒ return it (no duplicate)
        existing = self.ledger.get_order(idem)
        if existing is not None:
            return existing

        if not SC.is_live_mode(mode):
            raise SubmissionRefused(f"mode {mode} may not submit a live order")
        if mode in (SC.HALTED, SC.LIQUIDATE_ONLY, SC.NO_NEW_ENTRIES):
            raise SubmissionRefused(f"mode {mode} blocks new entries")
        if self.new_entries_blocked:
            raise SubmissionRefused("new entries blocked (reconciliation/recovery pending)")

        trace = {"idempotency_key": idem, "cycle_id": getattr(intent, "cycle_id", ""),
                 "snapshot_id": getattr(intent, "data_snapshot_id", ""),
                 "strategy_id": getattr(intent, "strategy_id", ""),
                 "strategy_version": getattr(intent, "strategy_version", 0),
                 "rules_hash": getattr(intent, "rules_hash", ""),
                 "card_id": getattr(intent, "card_id", ""),
                 "allocation_id": getattr(intent, "allocation_id", ""),
                 "intent_id": getattr(intent, "record_id", ""),
                 "broker_account": envelope.broker_account, "broker_adapter": self.adapter_id,
                 "mode": mode, "envelope_checksum": envelope.checksum}
        # risk context is passed to the governor only (not a record field)
        rtrace = {**trace, "family": family, "sector": sector, "cluster": cluster}

        order = SC.OrderStateRecord(
            symbol=intent.symbol, side="BUY",
            requested_qty=int(_qty_from(intent, limit_price, stop_price, envelope)),
            state=SM.INTENT_RECEIVED, **trace)
        self.ledger.record_order(order)

        # envelope permission
        ok, why = envelope.allows(family=family, symbol=intent.symbol,
                                  capital=order.requested_qty * max(limit_price, 1e-9))
        if not ok:
            return self._fail(order, SM.RISK_REJECTED, "ENVELOPE", why, from_state=SM.INTENT_RECEIVED,
                              via=SM.RISK_PENDING)

        # independent Risk Governor
        SM.apply_transition(order, SM.RISK_PENDING)
        plan_stub = _PlanStub(intent.symbol, order.requested_qty, limit_price, stop_price,
                              target_price, expected_risk_pct)
        rd = self.governor.evaluate(plan=plan_stub, envelope=envelope, portfolio=portfolio,
                                    account=account, trace=rtrace)
        if rd.decision in (SC.REJECT, SC.BLOCK_NEW_ENTRIES):
            return self._fail(order, SM.RISK_REJECTED, rd.limit_code, rd.reason)
        qty = rd.approved_qty if rd.decision == SC.APPROVE_REDUCED else order.requested_qty
        order.requested_qty = qty
        SM.apply_transition(order, SM.RISK_APPROVED, f"{rd.decision} qty={qty}")

        # frozen execution plan
        plan = SC.ExecutionPlan(symbol=intent.symbol, qty=qty, order_type="LIMIT",
                                limit_price=limit_price, stop_price=stop_price,
                                target_price=target_price, expected_risk=expected_risk_pct, **trace)
        plan.plan_id = plan.frozen_id()
        order.plan_id = plan.plan_id
        SM.apply_transition(order, SM.PLAN_CREATED)

        # journaled submit: persist INTENT to submit BEFORE the broker call
        SM.apply_transition(order, SM.SUBMISSION_PENDING)
        self.ledger.record_order(order)
        self._submit_to_broker(order, plan)
        # fills + protection
        self._ingest_fills(order)
        if order.state in (SM.FILLED, SM.PARTIALLY_FILLED):
            self._protect(order, plan)
        self.ledger.record_order(order)
        return order

    # ── broker submission (idempotent + timeout-safe) ───────────────────────────
    def _submit_to_broker(self, order, plan) -> None:
        SM.apply_transition(order, SM.SUBMITTING)
        try:
            bo = self.broker.place_order(plan, order.idempotency_key)
        except BrokerTimeout:
            # NEVER assume failure — reconcile: search the broker by idempotency key
            SM.apply_transition(order, SM.RECONCILIATION_REQUIRED, "submit timeout")
            found = self.broker.find_by_idempotency(order.idempotency_key)
            if found is None:
                self._incident("HIGH", "SUBMIT_TIMEOUT_UNRESOLVED", order.symbol)
                SM.apply_transition(order, SM.FAILED_SAFE)
                return
            bo = found                                    # the exchange DID accept it — adopt, no re-submit
        except BrokerError as e:
            self._incident("HIGH", "SUBMIT_ERROR", str(e))
            SM.apply_transition(order, SM.RECONCILIATION_REQUIRED, "submit error")
            SM.apply_transition(order, SM.FAILED_SAFE)
            return
        order.broker_order_id = bo.broker_order_id
        if bo.status == "REJECTED":
            SM.apply_transition(order, SM.REJECTED, bo.reject_reason)
            return
        SM.apply_transition(order, SM.SUBMITTED)
        SM.apply_transition(order, SM.ACKNOWLEDGED)

    def _ingest_fills(self, order) -> None:
        if not order.broker_order_id or SM.is_terminal(order.state):
            return
        fills = self.broker.fills(order.broker_order_id)
        total = 0; cost = 0.0; fees = 0.0
        for f in fills:
            fr = SC.FillRecord(broker_order_id=order.broker_order_id, fill_id=f.fill_id,
                               qty=f.qty, price=f.price, fees=f.fees,
                               idempotency_key=order.idempotency_key,
                               strategy_id=order.strategy_id, intent_id=order.intent_id)
            self.ledger.record_fill(fr)
            total += f.qty; cost += f.qty * f.price; fees += f.fees
        if total <= 0:
            SM.apply_transition(order, SM.OPEN); return
        order.filled_qty = total
        order.avg_fill_price = round(cost / total, 4)
        order.fees = round(fees, 2)
        if total >= order.requested_qty:
            if order.state == SM.ACKNOWLEDGED:
                SM.apply_transition(order, SM.FILLED)
            elif order.state != SM.FILLED:
                SM.apply_transition(order, SM.FILLED)
        else:
            SM.apply_transition(order, SM.PARTIALLY_FILLED)

    # ── protection is VERIFIED against the broker, never a local flag ────────────
    def _protect(self, order, plan) -> None:
        SM.apply_transition(order, SM.PROTECTION_PENDING)
        pplan = SC.ProtectionPlan(symbol=order.symbol, qty=order.filled_qty,
                                  stop_price=plan.stop_price, target_price=plan.target_price,
                                  idempotency_key=order.idempotency_key,
                                  strategy_id=order.strategy_id)
        try:
            self.broker.place_protection(pplan)
        except BrokerError as e:
            self._incident("CRITICAL", "PROTECTION_REJECTED", f"{order.symbol}: {e}")
            self.new_entries_blocked = True               # protection infra unhealthy → no new risk
            SM.apply_transition(order, SM.EXIT_PENDING, "protection failed → exit")
            self._exit_position(order, "PROTECTION_FAILED")
            return
        # VERIFY broker-side (a local record is not proof)
        protected = any(p.symbol == order.symbol and p.qty >= order.filled_qty
                        for p in self.broker.list_protections())
        if not protected:
            self._incident("CRITICAL", "PROTECTION_UNVERIFIED", order.symbol)
            self.new_entries_blocked = True
            return
        SM.apply_transition(order, SM.PROTECTED)
        pos = SC.PositionRecord(strategy_id=order.strategy_id, symbol=order.symbol,
                                qty=order.filled_qty, avg_price=order.avg_fill_price,
                                stop_price=plan.stop_price, target_price=plan.target_price,
                                protected=True, protection_order_id="verified",
                                opened_ts=str(time.time()))
        self.ledger.record_position(pos)

    def _exit_position(self, order, reason: str) -> None:
        SM.apply_transition(order, SM.CLOSED, f"exit:{reason}")
        self.ledger.remove_position(order.strategy_id, order.symbol)

    # ── reconciliation (broker is authority) ─────────────────────────────────────
    def reconcile(self) -> SC.ReconciliationReport:
        rep = SC.ReconciliationReport()
        broker_pos = {p.symbol: p.qty for p in self.broker.list_positions()}
        local_pos = {p.symbol: p.qty for p in self.ledger.open_positions()}
        owned = {p.symbol for p in self.ledger.open_positions()}
        for sym, bq in broker_pos.items():
            if sym not in local_pos:
                rep.add("UNKNOWN_MANUAL_POSITION", f"{sym} x{bq} at broker, unknown locally")
            elif bq != local_pos[sym]:
                rep.add("QUANTITY_MISMATCH", f"{sym}: broker {bq} vs local {local_pos[sym]}")
            else:
                rep.add("MATCHED", sym)
        for sym in local_pos:
            if sym not in broker_pos:
                rep.add("BROKER_MISSING", f"{sym} local but absent at broker")
        if rep.critical:
            self.new_entries_blocked = True               # critical conflict blocks new risk
            self._incident("CRITICAL", "RECON_CONFLICT", str(rep.findings))
        return rep

    # ── restart recovery (no duplicate submission) ───────────────────────────────
    def recover(self) -> dict:
        """Load unresolved orders, reconcile against the broker, and resolve ambiguous
        submissions by broker query (never blind re-submit). Blocks new entries until done."""
        self.new_entries_blocked = True
        resolved = []
        for order in self.ledger.unresolved_orders():
            bo = None
            if order.broker_order_id:
                bo = self.broker.get_order(order.broker_order_id)
            if bo is None or bo.status == "UNKNOWN":
                bo = self.broker.find_by_idempotency(order.idempotency_key)
            if bo is not None and bo.status != "UNKNOWN":
                # the order exists at the broker — adopt its state, DO NOT resubmit
                if bo.status == "COMPLETE" and order.state != SM.FILLED:
                    order.filled_qty = bo.filled_qty
                    if SM.can_transition(order.state, SM.RECONCILIATION_REQUIRED):
                        SM.apply_transition(order, SM.RECONCILIATION_REQUIRED, "recovery")
                    SM.apply_transition(order, SM.FILLED, "recovered fill")
                resolved.append(order.idempotency_key)
            else:
                if SM.can_transition(order.state, SM.FAILED_SAFE):
                    SM.apply_transition(order, SM.FAILED_SAFE, "recovery: not found at broker")
            self.ledger.record_order(order)
        rep = self.reconcile()
        if not rep.critical:
            self.new_entries_blocked = False              # recovery complete → resume
        return {"resolved": resolved, "reconciliation": rep.as_dict(),
                "new_entries_blocked": self.new_entries_blocked}

    # ── helpers ────────────────────────────────────────────────────────────────────
    def _fail(self, order, dst, code, reason, *, from_state=None, via=None):
        if via and SM.can_transition(order.state, via):
            SM.apply_transition(order, via)
        if SM.can_transition(order.state, dst):
            SM.apply_transition(order, dst, f"{code}: {reason}")
        self.ledger.record_order(order)
        return order

    def _incident(self, severity, code, detail):
        self.ledger.record_incident(SC.ExecutionIncident(severity=severity, code=code,
                                                          detail=detail, ts=str(time.time())))


class _PlanStub:
    def __init__(self, symbol, qty, limit_price, stop, target, risk):
        self.symbol = symbol; self.qty = qty; self.limit_price = limit_price
        self.stop_price = stop; self.target_price = target; self.expected_risk = risk


def _qty_from(intent, limit_price, stop_price, envelope) -> int:
    """Size from the envelope's per-trade risk when the intent didn't carry an absolute qty."""
    risk_amt = envelope.max_live_capital * envelope.max_risk_per_trade_pct
    unit = max(limit_price - stop_price, 1e-9)
    import math
    return max(1, int(math.floor(risk_amt / unit)))
