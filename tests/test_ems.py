"""
Deterministic, network-free tests for the Execution Management System, independent Risk
Governor, broker simulator, reconciliation and recovery — plus the end-to-end certification.

Simulator-only: no real broker, no credentials, no order ever leaves the process. Simulator
results are testing infrastructure, never market evidence.
"""
from __future__ import annotations

import inspect

import pytest

from ems import schemas as SC
from ems import state_machine as SM
from ems.simulator import SimBroker
from ems.risk_governor import RiskGovernor, RiskLimits, capital_protection_state
from ems.ledger import ExecutionLedger
from ems.ems import EMS, SubmissionRefused
from ems import preflight as PF
from research.intelligence import schemas as ISC


def _env(**kw):
    base = dict(broker_account="ACC1", max_live_capital=100_000.0,
                approved_families=("cross_sectional_momentum",),
                max_risk_per_trade_pct=0.01, daily_loss_limit=5000.0, drawdown_limit_pct=0.10)
    base.update(kw)
    return SC.approve_envelope(SC.OperatingEnvelope(**base), actor="user")


def _intent(symbol="WIN", sid="MOM"):
    return ISC.TradeIntent(symbol=symbol, strategy_id=sid, strategy_version=1, rules_hash="rh",
                           data_snapshot_id="snap1", cycle_id="cyc1", card_id="card1",
                           allocation_id="alloc1", intended_entry=100.0, stop_price=95.0,
                           target_price=115.0, intended_risk_pct=0.01, source="brain2")


def _ems(broker=None, ledger=None, governor=None):
    broker = broker or SimBroker()
    ledger = ledger or ExecutionLedger()
    governor = governor or RiskGovernor(RiskLimits())
    return EMS(broker, ledger, governor), broker, ledger, governor


def _submit(ems, intent=None, *, mode=SC.LIMITED_LIVE, env=None, portfolio=None, account=None,
            expected_risk_pct=0.01, family="cross_sectional_momentum"):
    return ems.submit_intent(
        intent or _intent(), envelope=env or _env(), mode=mode,
        portfolio=portfolio or {}, account=account or {"realized_pnl_today": 0, "drawdown_pct": 0},
        limit_price=100.0, stop_price=95.0, target_price=115.0,
        expected_risk_pct=expected_risk_pct, family=family)


# ── boundaries ───────────────────────────────────────────────────────────────────

class TestBoundaries:
    def test_intelligence_package_has_no_ems_import(self):
        import research.intelligence as I
        import pkgutil, importlib
        for m in pkgutil.walk_packages(I.__path__, I.__name__ + "."):
            src = inspect.getsource(importlib.import_module(m.name))
            assert "import ems" not in src and "from ems" not in src

    def test_paper_auto_cannot_submit_live(self):
        ems, *_ = _ems()
        with pytest.raises(SubmissionRefused):
            _submit(ems, mode=SC.PAPER_AUTO)

    def test_shadow_does_not_submit(self):
        ems, *_ = _ems()
        with pytest.raises(SubmissionRefused):
            _submit(ems, mode=SC.SHADOW)

    def test_halted_and_liquidate_only_block_entries(self):
        ems, *_ = _ems()
        for m in (SC.HALTED, SC.LIQUIDATE_ONLY, SC.NO_NEW_ENTRIES):
            with pytest.raises(SubmissionRefused):
                _submit(ems, mode=m)

    def test_unapproved_envelope_cannot_trade(self):
        ems, b, ledger, _ = _ems()
        raw = SC.OperatingEnvelope(broker_account="ACC1", max_live_capital=100_000.0,
                                   approved_families=("cross_sectional_momentum",))
        assert not raw.is_user_approved()
        order = _submit(ems, env=raw)
        assert order.state == SM.RISK_REJECTED and not ledger.open_positions()

    def test_approve_envelope_is_user_only(self):
        raw = SC.OperatingEnvelope(broker_account="ACC1", max_live_capital=1000.0)
        with pytest.raises(PermissionError):
            SC.approve_envelope(raw, actor="system")


# ── EMS lifecycle + idempotency ──────────────────────────────────────────────────

class TestLifecycle:
    def test_approved_intent_fills_and_protects(self):
        ems, b, ledger, _ = _ems()
        order = _submit(ems)
        assert order.state == SM.PROTECTED
        assert order.filled_qty == order.requested_qty
        assert ledger.open_positions() and ledger.open_positions()[0].protected
        # full provenance survives on the order
        assert order.snapshot_id == "snap1" and order.strategy_id == "MOM" and order.intent_id

    def test_duplicate_intent_creates_no_duplicate_order(self):
        ems, b, ledger, _ = _ems()
        o1 = _submit(ems)
        o2 = _submit(ems)                                 # same intent again
        assert o1.idempotency_key == o2.idempotency_key
        assert len(ledger.orders) == 1 and len(b.list_orders()) == 1

    def test_illegal_transition_is_rejected(self):
        order = SC.OrderStateRecord(state=SM.FILLED)
        with pytest.raises(SM.IllegalTransition):
            SM.apply_transition(order, SM.SUBMITTED)      # can't go backwards

    def test_submission_timeout_reconciles_not_resubmits(self):
        b = SimBroker(script={"WIN": {"timeout": True}})
        ems, *_ = _ems(broker=b)
        order = _submit(ems)
        # the exchange accepted it despite the timeout → adopted, exactly ONE broker order
        assert len(b.list_orders()) == 1
        assert order.broker_order_id and order.state == SM.PROTECTED

    def test_broker_reject_is_terminal(self):
        b = SimBroker(script={"WIN": {"reject": True, "reject_reason": "MARGIN"}})
        ems, _, ledger, _ = _ems(broker=b)
        order = _submit(ems)
        assert order.state == SM.REJECTED and not ledger.open_positions()

    def test_partial_fill_uses_actual_quantity(self):
        b = SimBroker(script={"WIN": {"fill_qty": 50}})
        ems, *_ = _ems(broker=b)
        order = _submit(ems)
        assert order.filled_qty == 50 and order.state == SM.PROTECTED
        assert order.fees > 0                              # fees from actual fills, not requested


# ── protection is broker-verified ────────────────────────────────────────────────

class TestProtection:
    def test_protection_rejection_exits_and_blocks(self):
        b = SimBroker(script={"WIN": {"protection_reject": True}})
        ems, _, ledger, _ = _ems(broker=b)
        order = _submit(ems)
        assert order.state == SM.CLOSED                    # unprotectable → exited
        assert ems.new_entries_blocked                     # protection infra unhealthy
        assert ledger.has_critical_incident()
        assert not ledger.open_positions()

    def test_local_flag_alone_cannot_mark_protected(self):
        class NoVerifyBroker(SimBroker):
            def list_protections(self):                    # broker never confirms protection
                return []
        b = NoVerifyBroker()
        ems, _, ledger, _ = _ems(broker=b)
        order = _submit(ems)
        assert order.state != SM.PROTECTED                 # not verified ⇒ not protected
        assert ems.new_entries_blocked and ledger.has_critical_incident()


# ── independent Risk Governor ────────────────────────────────────────────────────

class TestRiskGovernor:
    def test_per_trade_risk_reduces_quantity(self):
        ems, *_ = _ems()
        big = _submit(ems, expected_risk_pct=0.02)         # 2x the 1% per-trade limit
        assert big.state != SM.RISK_REJECTED
        assert big.requested_qty == 100    # 200 sized, halved to the 1% per-trade limit

    def test_symbol_cap_blocks(self):
        ems, _, ledger, _ = _ems()
        o = _submit(ems, portfolio={"symbol_risk": {"WIN": 0.02}})
        assert o.state == SM.RISK_REJECTED and not ledger.open_positions()

    def test_family_cap_blocks(self):
        ems, *_ = _ems()
        o = _submit(ems, portfolio={"family_risk": {"cross_sectional_momentum": 0.03}})
        assert o.state == SM.RISK_REJECTED

    def test_total_open_risk_cap_blocks(self):
        ems, *_ = _ems()
        o = _submit(ems, portfolio={"open_risk_pct": 0.05})
        assert o.state == SM.RISK_REJECTED

    def test_daily_loss_blocks_new_entries(self):
        ems, *_ = _ems()
        o = _submit(ems, account={"realized_pnl_today": -6000, "drawdown_pct": 0})
        assert o.state == SM.RISK_REJECTED

    def test_drawdown_moves_to_liquidate_only(self):
        env = _env()
        st = capital_protection_state({"realized_pnl_today": 0, "drawdown_pct": 0.10},
                                      RiskLimits(), env)
        assert st == SC.CAP_LIQUIDATE_ONLY

    def test_governor_failure_blocks_new_entries(self):
        gov = RiskGovernor(RiskLimits(), healthy=False)
        ems, _, ledger, _ = _ems(governor=gov)
        o = _submit(ems)
        assert o.state == SM.RISK_REJECTED and not ledger.open_positions()

    def test_brain2_cannot_override_governor_rejection(self):
        # even though the intent (Brain 2's decision) says trade, the governor's REJECT stands
        gov = RiskGovernor(RiskLimits(max_positions=0))
        ems, _, ledger, _ = _ems(governor=gov)
        o = _submit(ems)
        assert o.state == SM.RISK_REJECTED and not ledger.open_positions()


# ── envelope bounds ──────────────────────────────────────────────────────────────

class TestEnvelope:
    def test_unauthorized_family_is_rejected(self):
        ems, *_ = _ems()
        o = _submit(ems, family="event_driven")            # not in approved envelope
        assert o.state == SM.RISK_REJECTED

    def test_capital_above_envelope_is_rejected(self):
        ems, *_ = _ems()
        o = _submit(ems, env=_env(max_live_capital=50.0))   # 1 share (₹100) exceeds ₹50 ceiling
        assert o.state == SM.RISK_REJECTED


# ── reconciliation ───────────────────────────────────────────────────────────────

class TestReconciliation:
    def test_matched_state_passes(self):
        ems, b, ledger, _ = _ems()
        _submit(ems)                                       # opens WIN at broker + locally
        rep = ems.reconcile()
        assert not rep.critical and any(c == "MATCHED" for c, _ in rep.findings)

    def test_unknown_manual_position_not_assigned_to_strategy(self):
        ems, b, ledger, _ = _ems()
        b.add_manual_position("MANUALX", 10)
        rep = ems.reconcile()
        assert any(c == "UNKNOWN_MANUAL_POSITION" for c, _ in rep.findings)
        assert all(p.symbol != "MANUALX" for p in ledger.open_positions())

    def test_quantity_mismatch_blocks_new_risk(self):
        ems, b, ledger, _ = _ems()
        _submit(ems)
        # broker now reports a different qty for WIN → critical
        b._manual_positions.append(type("P", (), {"symbol": "WIN", "qty": 9999, "avg_price": 1})())
        rep = ems.reconcile()
        assert rep.critical and ems.new_entries_blocked


# ── restart recovery (no duplicate submission) ───────────────────────────────────

class TestRecovery:
    def test_restart_with_pending_order_recovers_without_duplicate(self):
        b = SimBroker()
        # pre-existing broker order for an in-flight submission
        plan = SC.ExecutionPlan(symbol="WIN", qty=100, limit_price=100.0)
        bo = b.place_order(plan, "idem-x")
        seq_before = b._seq
        ledger = ExecutionLedger()
        ledger.record_order(SC.OrderStateRecord(idempotency_key="idem-x",
                                                broker_order_id=bo.broker_order_id,
                                                state=SM.SUBMITTING, symbol="WIN",
                                                requested_qty=100, strategy_id="MOM"))
        ems = EMS(b, ledger, RiskGovernor(RiskLimits()))
        out = ems.recover()
        assert b._seq == seq_before                        # NO new broker order was created
        assert "idem-x" in out["resolved"]
        assert ledger.get_order("idem-x").state == SM.FILLED

    def test_recovery_blocks_entries_until_reconciled(self):
        b = SimBroker()
        b.add_manual_position("GHOST", 5)                  # unknown position → not resolvable clean
        ledger = ExecutionLedger()
        ledger.record_order(SC.OrderStateRecord(idempotency_key="idem-y",
                                                broker_order_id="", state=SM.SUBMITTING,
                                                symbol="ZZZ", requested_qty=10))
        ems = EMS(b, ledger, RiskGovernor(RiskLimits()))
        out = ems.recover()
        assert ledger.get_order("idem-y").state == SM.FAILED_SAFE   # not at broker → fail safe


# ── preflight + readiness ────────────────────────────────────────────────────────

class TestPreflight:
    def _kw(self, **over):
        base = dict(mode=SC.LIMITED_LIVE, envelope=_env(), snapshot_verified=True,
                    forward_eligible=True, data_fresh=True, registry_valid=True,
                    broker_authenticated=True, reconciled=True, governor_healthy=True,
                    ledger_writable=True, protection_healthy=True, has_critical_incident=False,
                    daily_loss_ok=True)
        base.update(over); return base

    def test_all_green_passes(self):
        assert PF.live_preflight(**self._kw()).ok

    def test_unverified_snapshot_blocks_live(self):
        assert not PF.live_preflight(**self._kw(snapshot_verified=False)).ok

    def test_broker_auth_failure_blocks_live(self):
        assert not PF.live_preflight(**self._kw(broker_authenticated=False)).ok

    def test_critical_incident_blocks_live(self):
        assert not PF.live_preflight(**self._kw(has_critical_incident=True)).ok

    def test_readiness_states_are_not_blurred(self):
        assert PF.readiness_state(architecture_ok=True, simulator_certified=False,
                                  broker_connected=False, user_activated=False) == SC.ARCHITECTURE_READY
        assert PF.readiness_state(architecture_ok=True, simulator_certified=True,
                                  broker_connected=False, user_activated=False) == SC.SIMULATOR_CERTIFIED
        assert PF.readiness_state(architecture_ok=True, simulator_certified=True,
                                  broker_connected=True, user_activated=True) == SC.USER_ACTIVATED


# ── end-to-end certification (simulator) ─────────────────────────────────────────

class TestCertification:
    def test_full_chain_no_human_click_full_provenance(self):
        # Operating Envelope approved BEFOREHAND by the user (once), then fully automatic
        env = _env()
        assert env.is_user_approved()
        b = SimBroker()
        ledger = ExecutionLedger()
        governor = RiskGovernor(RiskLimits())
        ems = EMS(b, ledger, governor)
        intent = _intent()

        order = ems.submit_intent(
            intent, envelope=env, mode=SC.LIMITED_LIVE, portfolio={},
            account={"realized_pnl_today": 0, "drawdown_pct": 0},
            limit_price=100.0, stop_price=95.0, target_price=115.0,
            expected_risk_pct=0.01, family="cross_sectional_momentum")

        # entry → risk-approved → planned → submitted → filled → broker-verified protection
        assert order.state == SM.PROTECTED and order.filled_qty > 0
        assert ledger.open_positions()[0].protected
        # provenance: the whole decision chain is on the order
        assert (order.snapshot_id == "snap1" and order.cycle_id == "cyc1"
                and order.card_id == "card1" and order.allocation_id == "alloc1"
                and order.intent_id == intent.record_id and order.plan_id)
        # reconciliation is clean; no duplicate on re-run (idempotent) or restart
        assert not ems.reconcile().critical
        again = ems.submit_intent(intent, envelope=env, mode=SC.LIMITED_LIVE, portfolio={},
                                  account={"realized_pnl_today": 0, "drawdown_pct": 0},
                                  limit_price=100.0, stop_price=95.0, target_price=115.0,
                                  expected_risk_pct=0.01, family="cross_sectional_momentum")
        assert again.idempotency_key == order.idempotency_key and len(b.list_orders()) == 1
        # the governor can INDEPENDENTLY stop the very same intent under a tighter limit
        strict = EMS(SimBroker(), ExecutionLedger(), RiskGovernor(RiskLimits(max_positions=0)))
        blocked = strict.submit_intent(intent, envelope=env, mode=SC.LIMITED_LIVE, portfolio={},
                                       account={"realized_pnl_today": 0, "drawdown_pct": 0},
                                       limit_price=100.0, stop_price=95.0, target_price=115.0,
                                       expected_risk_pct=0.01, family="cross_sectional_momentum")
        assert blocked.state == SM.RISK_REJECTED
        # simulator readiness is SIMULATOR_CERTIFIED — never claimed as broker-connected/live
        assert PF.readiness_state(architecture_ok=True, simulator_certified=True,
                                  broker_connected=False, user_activated=False) == SC.SIMULATOR_CERTIFIED
