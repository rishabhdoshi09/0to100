"""India-only mode: no new US paper risk, existing US positions stay true.

QuantTerm is India-first (NSE equities + NSE F&O). By default the US paper lane
opens nothing: no universe scan, no entries, no counterfactual decisions. Its
existing positions are still exited at their stops/targets and settled exactly
once, so nothing is lost or frozen. Disposable files only; no network/broker.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest


@pytest.fixture(autouse=True)
def default_scope(monkeypatch):
    monkeypatch.delenv("QT_MARKET_SCOPE", raising=False)


@pytest.mark.parametrize("raw, scope, allowed, invalid", [
    (None, "INDIA_ONLY", False, False),
    ("INDIA_ONLY", "INDIA_ONLY", False, False),
    ("india_and_us", "INDIA_AND_US", True, False),
    ("US", "INDIA_ONLY", False, True),            # unclear => no new risk
    ("yes please", "INDIA_ONLY", False, True),
])
def test_scope_defaults_to_india_only_and_fails_closed(monkeypatch, raw, scope, allowed, invalid):
    from core import market_scope as MS
    if raw is not None:
        monkeypatch.setenv("QT_MARKET_SCOPE", raw)
    st = MS.scope_status()
    assert st["market_scope"] == scope
    assert st["us_new_entries_allowed"] is allowed
    assert MS.us_new_entries_allowed() is allowed
    assert st["configured_value_invalid"] is invalid
    assert st["us_existing_positions_supervised"] is True


def _us(tmp_path, monkeypatch):
    import execution.trade_executor as te
    import execution.us_autopilot as us
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    (tmp_path / "logs").mkdir(exist_ok=True)
    monkeypatch.setattr(te, "_DB", tmp_path / "logs" / "trades.db")
    monkeypatch.setattr(us, "_STATE_FILE", tmp_path / "us_ap.json")
    monkeypatch.setattr(us, "_state", {})
    st = us._load()
    st.update({"allocation": 100_000.0, "realized_pnl": 0.0, "armed": True,
               "accounting_ids_migrated": True})
    assert us._save() is True
    monkeypatch.setattr(us, "_in_window", lambda now_et=None: True)
    monkeypatch.setattr(us, "_anchor_live", lambda sym, e, s, c: (e, ""))
    sent: list[str] = []
    monkeypatch.setattr(us, "_notify", sent.append)
    decisions: list[dict] = []
    monkeypatch.setattr(us, "_record_learning_decision", lambda **kw: decisions.append(kw))
    return us, te, sent, decisions


def _open_us(te, symbol, qty, entry, stop):
    conn = te.connect()
    te.ensure_schema(conn)
    conn.execute(
        "INSERT INTO trades(placed_at, mode, symbol, qty, entry_type, entry_price, "
        "stop_price, target_price, status, note) VALUES "
        "('2026-11-05T10:00', 'PAPER', ?, ?, 'MARKET', ?, ?, ?, 'PAPER_OPEN', "
        "'US_AUTOPILOT:us_scanner|session=2026-11-05')",
        (symbol, qty, entry, stop, entry * 1.04),
    )
    conn.commit()
    conn.close()


def test_india_only_opens_no_us_position_and_says_why(tmp_path, monkeypatch):
    us, te, sent, decisions = _us(tmp_path, monkeypatch)
    assert us.consider("AAPL", 100.0, 95.0, 80.0, 70.0) is False
    assert us._open_trades() == []
    assert us.reject_funnel()["rejects"] == {"India-only mode (new US entries paused)": 1}
    ok, msg = us.arm()
    assert ok is False and "India-only" in msg
    status = us.get_status()
    assert status["new_entries_paused"] is True
    assert status["entries_paused_reason"] == "INDIA_ONLY_MODE"
    assert status["market_scope"] == "INDIA_ONLY"
    # The owner's arm consent is preserved, not rewritten by the pause.
    assert us._load()["armed"] is True


def test_paused_lane_evaluates_nothing_and_records_no_decisions(tmp_path, monkeypatch):
    us, te, sent, decisions = _us(tmp_path, monkeypatch)
    called = []
    monkeypatch.setattr(us, "consider", lambda **kw: called.append(kw))
    us.on_setups([{"symbol": "NVDA", "verdict": "STRONG BUY", "entry": 100, "stop": 95}])
    assert called == [] and decisions == [] and sent == []


def test_existing_us_position_is_still_exited_and_settled_once(tmp_path, monkeypatch):
    us, te, sent, decisions = _us(tmp_path, monkeypatch)
    _open_us(te, "HELD", 100, 100.0, 95.0)
    monkeypatch.setattr("data.us_data.us_live_prices",
                        lambda syms: {s: {"price": 94.0} for s in syms})
    monkeypatch.setattr("product.us_learning.settle_pending", lambda: None)
    us.review_cycle()
    closed = us._trades(us._LOSS)
    assert [t["symbol"] for t in closed] == ["HELD"]
    pnl = round(us._net_pnl(closed[0]), 2)
    assert us._load()["realized_pnl"] == pnl
    us.review_cycle()                                    # idempotent settlement
    assert us._load()["realized_pnl"] == pnl
    assert len(sent) == 1


def test_us_and_india_scope_still_trades_when_explicitly_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_MARKET_SCOPE", "INDIA_AND_US")
    us, te, sent, decisions = _us(tmp_path, monkeypatch)
    assert us.consider("AAPL", 100.0, 95.0, 80.0, 70.0) is True
    assert us.get_status()["new_entries_paused"] is False


def _ops(monkeypatch):
    import operations.market_ops as market_ops
    ops = market_ops.MarketOperationsWorker.__new__(market_ops.MarketOperationsWorker)
    monkeypatch.setattr(ops, "_progress", lambda *a, **k: None, raising=False)
    return market_ops, ops


@pytest.mark.parametrize("payload_mode", ["SUPERVISE_ONLY", "SCAN_AND_TRADE", ""])
def test_market_ops_supervises_without_scanning_in_india_only(monkeypatch, payload_mode):
    import execution.us_autopilot as autopilot
    import scan.us_scanner as scanner
    market_ops, ops = _ops(monkeypatch)
    calls = []
    monkeypatch.setattr(scanner, "scan_us", lambda **kw: calls.append("scan") or [])
    monkeypatch.setattr(autopilot, "review_cycle", lambda: calls.append("review"))
    monkeypatch.setattr(autopilot, "get_status",
                        lambda: {"armed": True, "open_trades": [{"id": 1}], "realized_pnl": 12.5})
    result = ops._run_us_market_scan(
        {"operation_id": "op", "payload": {"scope": "S&P 500", "mode": payload_mode}})
    assert calls == ["review"]                 # scope decided at execution time
    assert result["mode"] == "SUPERVISE_ONLY"
    assert result["scan"] == "SKIPPED_INDIA_ONLY"
    assert result["new_entries_allowed"] is False
    assert result["paper"]["open_trades"] == 1
    assert result["live_locked"] is True


def test_failed_us_supervision_is_blocked_not_succeeded(monkeypatch):
    import execution.us_autopilot as autopilot
    market_ops, ops = _ops(monkeypatch)

    def broken():
        raise RuntimeError("US_PAPER_STATE_RECOVERY_REQUIRED")

    monkeypatch.setattr(autopilot, "review_cycle", broken)
    with pytest.raises(market_ops.OperationBlocked) as info:
        ops._run_us_market_scan({"operation_id": "op", "payload": {"mode": "SUPERVISE_ONLY"}})
    assert info.value.code == "US_PAPER_SUPERVISION_FAILED"
    assert "RECOVERY_REQUIRED" in info.value.result["error"]


def _schedule(monkeypatch, latest):
    import data.us_data as us_data
    import operations.store as op_store
    from research.autonomy.supervisor import Supervisor
    captured = []

    class FakeStore:
        def __init__(self, *_a, **_k):
            pass

        def latest(self, _kind):
            return dict(latest)

        def enqueue(self, kind, **kwargs):
            captured.append(kwargs)
            return ({"status": "PENDING"}, True)

    monkeypatch.setattr(op_store, "OperationStore", FakeStore)
    monkeypatch.setattr(us_data, "us_market_open", lambda _now=None: True)
    dummy = SimpleNamespace(_incident=lambda *_a, **_k: None)
    Supervisor._ensure_us_market_pipeline(
        dummy, datetime(2026, 9, 25, 20, 0, tzinfo=ZoneInfo("Asia/Kolkata")))
    return captured


def test_supervisor_schedules_supervise_only_in_india_only(monkeypatch):
    captured = _schedule(monkeypatch, {})
    assert captured and captured[0]["payload"]["mode"] == "SUPERVISE_ONLY"
    monkeypatch.setenv("QT_MARKET_SCOPE", "INDIA_AND_US")
    captured = _schedule(monkeypatch, {})
    assert captured and captured[0]["payload"]["mode"] == "SCAN_AND_TRADE"


def test_failed_supervision_is_not_retried_every_tick_in_the_same_slot(monkeypatch):
    slot = "2026-09-25T10:30"
    captured = _schedule(monkeypatch, {
        "status": "BLOCKED", "error_code": "US_PAPER_SUPERVISION_FAILED",
        "payload": {"slot": slot},
    })
    assert captured == []


def test_home_projection_reports_the_pause_truthfully(monkeypatch):
    import execution.us_autopilot as autopilot
    from product import us_market_status
    monkeypatch.setattr(autopilot, "get_status", lambda: {
        "armed": True, "state_recovery_required": False, "open_trades": [],
    })
    out = us_market_status.status()
    assert out["paper"]["new_entries_paused"] is True
    assert out["paper"]["entries_paused_reason"] == "INDIA_ONLY_MODE"
    assert out["parity"]["state"] == "PAUSED_INDIA_ONLY"
    assert out["parity"]["paper_auto_execution"] is False
    assert out["parity"]["autonomous_scan"] is False
    assert out["parity"]["existing_positions_supervised"] is True
    assert out["live_locked"] is True


def _projection(monkeypatch):
    import execution.us_autopilot as autopilot
    from product import us_market_status
    monkeypatch.setattr(autopilot, "get_status", lambda: {
        "armed": True, "state_recovery_required": False, "open_trades": [],
    })
    return us_market_status.status()


@pytest.mark.parametrize("scope", [None, "INDIA_AND_US"])
def test_product_acceptance_grades_the_real_projection_for_each_scope(monkeypatch, scope):
    from scripts._product_acceptance_core import grade_us_market_truth
    if scope:
        monkeypatch.setenv("QT_MARKET_SCOPE", scope)
    out = _projection(monkeypatch)
    assert grade_us_market_truth(out) == {"status": "PASS", "blocker_reason": ""}


def test_product_acceptance_rejects_a_stale_capability_claim(monkeypatch):
    from scripts._product_acceptance_core import grade_us_market_truth
    out = _projection(monkeypatch)                       # INDIA_ONLY
    claimed = dict(out, parity=dict(out["parity"], paper_auto_execution=True))
    assert grade_us_market_truth(claimed)["status"] == "FAIL"
    no_scope = {k: v for k, v in out.items() if k != "market_scope"}
    assert grade_us_market_truth(no_scope)["blocker_reason"] == "US market scope is not reported"
    unlocked = dict(out, live_locked=False)
    assert grade_us_market_truth(unlocked)["status"] == "FAIL"


def test_paused_lane_does_not_claim_counterfactuals(monkeypatch):
    out = _projection(monkeypatch)
    assert out["parity"]["forward_counterfactuals"] is False
    assert out["parity"]["forward_outcome_settlement"] is True
