"""Independent broker / account-risk audit (GPT, 21 Nov 2026, d0b668fe): one base-failing test per fixed finding.

Every broker is a fake; no test can reach a network or a real order.
"""
from __future__ import annotations

import pytest


# ── QT-B05: the INSTALLED paper composer is as fail-closed as the base one ──────

@pytest.mark.parametrize("status, auth_error", [
    ("BLOCKED_RECOVERY", False), ("BLOCKED_DURABILITY", False), ("SKIPPED_LOCKED", False),
    ("NO_ACTION", True),
])
def test_b05_installed_cycle_blocks_unknown_paper_authority(monkeypatch, status, auth_error):
    from product import paper_autopilot as pa
    from research.autonomy import jobs, paper_cycle_truth
    from tests.test_paper_cycle_upstream_failclosed import _wired_deps

    monkeypatch.setattr(jobs.Deps, "run_paper_cycle", jobs.Deps.run_paper_cycle)
    monkeypatch.setattr(paper_cycle_truth, "_INSTALLED", False)
    paper_cycle_truth.install_paper_cycle_truth()
    assert getattr(jobs.Deps.run_paper_cycle, "_quantterm_paper_cycle_truth", False)
    deps, notes = _wired_deps(monkeypatch, core_status=status, auth_error=auth_error)
    brain = __import__("research.auto_research.scheduler", fromlist=["x"]).get_brain()
    brain.manage_intelligence_positions_day = lambda **kw: {"status": status, "as_of_date": "2026-11-06"}
    calls = []
    monkeypatch.setattr(pa, "run_reco_paper_cycle", lambda **kw: calls.append(kw) or {})
    result = deps.run_paper_cycle(True)
    assert calls == [] and result["eligibility"] == "BLOCKED_SAFETY" and result["new_entries_allowed"] is False
    assert notes


def test_b05_installed_cycle_still_trades_when_upstream_is_healthy(monkeypatch):
    from product import paper_autopilot as pa
    from research.autonomy import jobs, paper_cycle_truth
    from tests.test_paper_cycle_upstream_failclosed import _wired_deps

    monkeypatch.setattr(jobs.Deps, "run_paper_cycle", jobs.Deps.run_paper_cycle)
    monkeypatch.setattr(paper_cycle_truth, "_INSTALLED", False)
    paper_cycle_truth.install_paper_cycle_truth()
    deps, _ = _wired_deps(monkeypatch)
    brain = __import__("research.auto_research.scheduler", fromlist=["x"]).get_brain()
    brain.manage_intelligence_positions_day = lambda **kw: {"status": "NO_ACTION", "as_of_date": "2026-11-06"}
    from types import SimpleNamespace
    brain.state = SimpleNamespace(last_intel_cycle=None)
    calls = []
    monkeypatch.setattr(pa, "run_reco_paper_cycle", lambda **kw: calls.append(kw) or {})
    deps.run_paper_cycle(True)
    assert len(calls) == 1 and calls[0]["paper_enabled"] is True and calls[0]["entries_allowed"] is True


# ── QT-B02: per-trade 1% / 10% are hard at the one shared gate ──────────────────

@pytest.fixture
def _account(tmp_path, monkeypatch):
    from execution import trade_executor as te
    from risk import portfolio_risk as pr

    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setattr(te, "_DB", tmp_path / "trades.db")
    monkeypatch.setattr(pr, "_capital", lambda: 100_000.0)
    return te, pr


def test_b02_disconnected_ticket_cannot_commit_over_limit(_account, monkeypatch):
    te, _pr = _account
    monkeypatch.setattr(te, "kite_ready", lambda: False)
    result = te.place_trade("INFY", 600, "LIMIT", 100, 90, 120)            # 60% notional, 6% risk
    assert not result["ok"] and result.get("reason_code") == "TRADE_RISK_CAP"
    assert te.recent_trades() == []


@pytest.mark.parametrize("qty, entry, stop, code", [
    (150, 100, 90, "TRADE_RISK_CAP"),          # the legacy 1.5x conviction size: 1.5% risk
    (200, 100, 99.5, "POSITION_SIZE_CAP"),     # 20% of capital in one name at a tiny risk
    (10, 100, 100, "INVALID_TRADE_PLAN"),      # a zero-distance stop
    (10, 100, -5, "INVALID_TRADE_PLAN"),
])
def test_b02_the_gate_enforces_the_house_limits(_account, qty, entry, stop, code):
    _te, pr = _account
    gate = pr.account_exposure_gate("INFY", qty=qty, entry=entry, stop=stop)
    assert not gate["ok"] and gate["reason_code"] == code


def test_b02_a_house_sized_paper_trade_still_records(_account, monkeypatch):
    te, _pr = _account
    out = te.place_trade("INFY", 100, "LIMIT", 100, 90, 120, paper=True)    # 1% risk, 10% notional
    assert out["ok"] and [r["symbol"] for r in te.recent_trades()] == ["INFY"]


def test_b02_a_caller_holding_the_lock_can_still_record(_account):
    te, pr = _account
    with pr.account_mutation_lock():                                       # re-entrant, no deadlock
        out = te.place_trade("INFY", 50, "LIMIT", 100, 90, 120, paper=True)
    assert out["ok"]


# ── QT-B03: F&O and equity share the five slots and one position per name ───────

def _fill_equity(te, symbols):
    for s in symbols:
        assert te.place_trade(s, 10, "LIMIT", 100, 95, 120, paper=True)["ok"]


def test_b03_equity_gate_counts_fno_positions(_account, monkeypatch):
    te, pr = _account
    _fill_equity(te, ["INFY", "TCS", "HDFCBANK", "SBIN"])
    monkeypatch.setattr(pr, "_fno_open_rows", lambda: [{"symbol": "NIFTY", "qty": 50, "entry": 10.0,
                                                        "stop": 5.0, "book": pr.FNO_BOOK,
                                                        "premium_loss_exposure": 500.0}])
    gate = pr.account_exposure_gate("ITC", qty=10, entry=100, stop=95)
    assert not gate["ok"] and gate["reason_code"] == "MAX_POSITIONS"


def test_b03_fno_book_respects_equity_slots_and_underlyings():
    from product.fo_paper import FoPaperBook

    def open_one(book, underlying):
        return book.open_position(underlying=underlying, option_symbol=f"{underlying}26SEP100CE",
                                  option_type="CE", entry=10.0, stop=6.0, target=20.0, lot_size=25,
                                  opened_at="2026-09-01T10:15:00+05:30", max_holding_sessions=5)

    book = FoPaperBook(capital=100_000)
    book.external_positions, book.external_underlyings = 5, {"INFY", "TCS", "HDFCBANK", "SBIN", "ITC"}
    assert open_one(book, "RELIANCE") is None and book.refusals[-1][1] == "MAX_POSITIONS"
    book = FoPaperBook(capital=100_000)
    book.external_positions, book.external_underlyings = 1, {"RELIANCE"}
    assert open_one(book, "RELIANCE") is None and book.refusals[-1][1] == "UNDERLYING_OPEN_IN_EQUITY"
    assert open_one(book, "INFY") is not None


# ── QT-B04: a stale F&O cycle can never erase a position another lane committed ──

def test_b04_fno_management_cannot_erase_concurrent_open(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime
    from threading import Event
    from execution import trade_executor as te
    from product.fo_paper_runtime import run_fo_paper_cycle
    from product.fo_paper_store import FoPaperStore
    from tests.test_fo_paper_runtime import IST, _QuoteClient, _directional

    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setattr(te, "_DB", tmp_path / "trades.db")
    loaded, committed = Event(), Event()
    path = tmp_path / "fno.sqlite3"
    now = datetime(2026, 9, 1, 10, 15, tzinfo=IST)

    def cycle(enter):
        with FoPaperStore(path) as store:
            if not enter:
                read = store.load_positions

                def pause_after_read():
                    snapshot = read()
                    loaded.set()
                    committed.wait(3)
                    return snapshot

                store.load_positions = pause_after_read
            try:
                return run_fo_paper_cycle(_directional() if enter else {"available": False}, store=store,
                                          client=_QuoteClient(), now_ist=now, allow_new_entries=enter)
            finally:
                if enter:
                    committed.set()

    with ThreadPoolExecutor(2) as workers:
        management = workers.submit(cycle, False)
        assert loaded.wait(5)
        entry = workers.submit(cycle, True)
        assert entry.result(timeout=30)["opened_count"] == 1
        management.result(timeout=30)
    with FoPaperStore(path) as store:
        assert len(store.load_positions()) == 1


# ── QT-B01: the proxy hands out no SDK handle; typed code never runs beside it ───

@pytest.mark.parametrize("name, attribute", [("quote", "__self__"), ("place_order", "__wrapped__"),
                                             ("quote", "__wrapped__"), ("place_order", "__self__")])
def test_b01_proxy_does_not_expose_sdk_capability(name, attribute):
    from data.kite_client import _GuardedKiteProxy
    from product.live_execution_interlock import get_live_execution_state

    class FakeKite:
        def quote(self, keys):
            return {"k": keys}

        def place_order(self, **order):
            pytest.fail("broker mutation reached")

    assert get_live_execution_state().locked
    proxy = _GuardedKiteProxy(FakeKite())
    assert getattr(getattr(proxy, name), attribute, None) is None
    assert proxy.quote(["NSE:TCS"]) == {"k": ["NSE:TCS"]}                 # reads still work
    with pytest.raises(Exception):
        proxy.place_order(symbol="TCS")                                   # the lock still refuses


def test_b01_the_legacy_live_runner_never_executes_typed_strategy_code():
    import pandas as pd
    from ui import live_runner

    ran = []
    code = "ran = __import__('builtins'); generate_signals = lambda df: df['close']"
    with pytest.raises(RuntimeError, match="disabled"):
        live_runner._evaluate_strategy(code, pd.DataFrame({"close": [1.0, 2.0]}))
    assert ran == []


# ── QT-B10: the kill switch takes effect at once, fails closed and is confirmed ──

def test_b10_governance_halt_overrides_cached_permission(tmp_path, monkeypatch):
    from core import governance as gov

    monkeypatch.setattr(gov, "_STATE_FILE", tmp_path / "governance.json")
    monkeypatch.setattr(gov, "_cache", {"ts": 0.0, "data": None})
    monkeypatch.setattr(gov, "_collect_signals", gov._read_state)
    assert gov.can_place_order()[0]
    gov.set_manual_halt(True)
    assert not gov.can_place_order()[0]


def test_b10_a_halt_written_by_another_process_is_seen_despite_the_cache(tmp_path, monkeypatch):
    import json
    from core import governance as gov

    monkeypatch.setattr(gov, "_STATE_FILE", tmp_path / "governance.json")
    monkeypatch.setattr(gov, "_cache", {"ts": 0.0, "data": None})
    monkeypatch.setattr(gov, "_collect_signals", gov._read_state)
    assert gov.can_place_order()[0]                                   # caches NORMAL
    (tmp_path / "governance.json").write_text(json.dumps({"manual_halt": True}))
    assert not gov.can_place_order()[0]


def test_b10_an_unreadable_state_file_is_a_halt_and_a_failed_write_raises(tmp_path, monkeypatch):
    from core import governance as gov

    state = tmp_path / "governance.json"
    monkeypatch.setattr(gov, "_STATE_FILE", state)
    monkeypatch.setattr(gov, "_cache", {"ts": 0.0, "data": None})
    monkeypatch.setattr(gov, "_collect_signals", gov._read_state)
    state.write_text("{not json")
    allowed, _mult, why = gov.can_place_order()
    assert not allowed and "unreadable" in why
    monkeypatch.setattr(gov, "_STATE_FILE", tmp_path / "missing_dir" / "x" / "governance.json")
    monkeypatch.setattr(gov, "_write_state", lambda d: False)
    with pytest.raises(RuntimeError):
        gov.set_manual_halt(True)


# ── QT-B07: a refused or failed GTT keeps the protection work pending ────────────

def _pending_gtt_world(tmp_path, monkeypatch, history, place_gtt):
    from data import kite_client as kc
    from execution import trade_executor as te

    class FakeSDK:
        GTT_TYPE_OCO = "oco"

        def order_history(self, order_id):
            return history

    FakeSDK.place_gtt = place_gtt
    monkeypatch.setattr(te, "_DB", tmp_path / "trades.db")
    monkeypatch.setattr(te, "kite_ready", lambda: True)
    monkeypatch.setattr(kc, "KiteConnect", lambda **kw: FakeSDK())
    client = kc.KiteClient(api_key="fake", access_token="", api_secret="")
    monkeypatch.setattr(kc, "KiteClient", lambda: client)
    assert te._journal(dict(mode="LIVE", symbol="INFY", qty=10, entry_type="LIMIT", entry_price=100,
                            stop_price=90, target_price=120, entry_order_id="FAKE-BUY", status="PENDING_GTT"))
    return te


def test_b07_locked_gtt_recovery_keeps_work_pending(tmp_path, monkeypatch):
    def place_gtt(self, **kwargs):
        pytest.fail("locked GTT reached SDK")

    te = _pending_gtt_world(tmp_path, monkeypatch, [{"status": "COMPLETE", "filled_quantity": 10}], place_gtt)
    assert te.ensure_pending_gtts() == 0
    assert te.recent_trades(1)[0]["status"] == "PENDING_GTT"


def test_b07_a_partly_filled_cancelled_entry_is_not_a_failed_entry(tmp_path, monkeypatch):
    def place_gtt(self, **kwargs):
        pytest.fail("locked GTT reached SDK")

    te = _pending_gtt_world(tmp_path, monkeypatch, [{"status": "CANCELLED", "filled_quantity": 4}], place_gtt)
    te.ensure_pending_gtts()
    assert te.recent_trades(1)[0]["status"] == "PENDING_GTT"            # 4 shares still need protection


# ── QT-B06: a quote through the stop never removes LIVE exposure ─────────────────

def test_b06_quote_through_stop_does_not_remove_live_exposure(tmp_path, monkeypatch):
    from data import live_quotes
    from execution import autopilot as ap, trade_executor as te
    from risk.position_manager import _open_trades

    monkeypatch.setattr(te, "_DB", tmp_path / "trades.db")
    assert te._journal(dict(mode="LIVE", symbol="INFY", qty=10, entry_type="LIMIT", entry_price=100,
                            stop_price=90, target_price=120, status="PLACED_NO_GTT",
                            entry_order_id="FAKE-BUY", note="AUTOPILOT:scanner"))
    monkeypatch.setattr(live_quotes, "get_live_quotes", lambda _: {"INFY": {"price": 85}})
    monkeypatch.setattr(ap, "_log_activity", lambda msg: None)
    ap._close_crossed_live_trades()
    assert {row["symbol"] for row in _open_trades(strict=True)} == {"INFY"}


def test_b06_an_older_sell_never_closes_a_newer_trade():
    from execution import autopilot as ap

    trade = {"placed_at": "2026-09-01T10:00:00+05:30"}
    assert not ap._exit_after_entry({"order_timestamp": "2026-09-01 09:20:00"}, trade)
    assert ap._exit_after_entry({"order_timestamp": "2026-09-01 14:00:00"}, trade)
    assert not ap._exit_after_entry({}, trade)                              # unknown time: never matched


# ── QT-B09: the EMS never reports CLOSED while the broker still holds the fill ───

def test_b09_ems_reports_closed_only_after_broker_flat():
    from ems import simulator, state_machine as sm
    from tests.test_ems import _ems, _submit

    broker = simulator.SimBroker(script={"WIN": {"protection_reject": True}})
    ems, *_ = _ems(broker=broker)
    order = _submit(ems)
    assert ems.new_entries_blocked
    assert order.state != sm.CLOSED or not broker.list_positions()


# ── Cycle-2 checkpoint review (QT-C2-01..05, 10) ─────────────────────────────────

def test_c2_03_equity_cannot_open_a_sixth_account_slot(_account, monkeypatch):
    from config import settings

    te, pr = _account
    monkeypatch.setattr(settings, "max_open_positions", 10)
    monkeypatch.setattr(pr, "_fno_open_rows", lambda: [dict(symbol="NIFTY", qty=25, entry=10., stop=8.,
                                                           book=pr.FNO_BOOK)])
    for symbol in ("INFY", "TCS", "SBIN", "ITC"):
        assert te.place_trade(symbol, 10, "LIMIT", 100, 95, 120, paper=True)["ok"]
    assert not te.place_trade("RELIANCE", 10, "LIMIT", 100, 95, 120, paper=True)["ok"]


def test_c2_03_the_account_cap_uses_unrounded_risk(_account, monkeypatch):
    te, pr = _account
    for symbol in ("INFY", "TCS", "SBIN", "ITC"):
        assert te.place_trade(symbol, 100, "LIMIT", 100, 90, 120, paper=True)["ok"]
    monkeypatch.setattr(pr, "_trade_cap_capital", lambda: 101_000.)
    out = te.place_trade("RELIANCE", 100, "LIMIT", 100, 89.97, 120, paper=True)   # Rs 5,003 total
    assert not out["ok"] and out.get("reason_code") == "MAX_PORTFOLIO_RISK"


def test_c2_04_cancelled_partial_survives_legacy_reconciliation(tmp_path, monkeypatch):
    import sqlite3
    from types import SimpleNamespace
    from data import kite_client
    from execution import autopilot
    from risk.position_manager import _open_trades

    order = dict(order_id="FAKE-BUY", status="CANCELLED", filled_quantity=4)
    te = _pending_gtt_world(tmp_path, monkeypatch, [order], lambda *a, **k: pytest.fail("locked GTT reached broker"))
    conn = sqlite3.connect(te._DB)
    conn.execute("UPDATE trades SET note='AUTOPILOT:scanner'")
    conn.commit()
    conn.close()
    assert te.ensure_pending_gtts() == 0
    monkeypatch.setattr(kite_client, "KiteClient", lambda: SimpleNamespace(is_connected=lambda: True,
                                                                         get_orders=lambda: [order]))
    monkeypatch.setattr(autopilot, "_log_activity", lambda *_: None)
    autopilot._reconcile_live_fills()
    assert _open_trades(strict=True)[0]["status"] == "PENDING_GTT"


def test_c2_05_proxy_state_does_not_export_sdk():
    import copy
    import pickle
    from data.kite_client import _GuardedKiteProxy

    sdk = object()
    proxy = _GuardedKiteProxy(sdk)
    for attempt in (lambda: proxy.__getstate__(), lambda: pickle.dumps(proxy), lambda: copy.copy(proxy),
                    lambda: copy.deepcopy(proxy), lambda: proxy.__reduce_ex__(2)):
        with pytest.raises(TypeError):
            attempt()


def test_c2_01_a_rolling_grid_run_cannot_attest_as_production(monkeypatch, tmp_path):
    from research import alpha_ledger
    from tests.test_alpha_ledger import _stub_walk_forward

    sb = _stub_walk_forward(monkeypatch, tmp_path)
    monkeypatch.setattr(sb, "MIN_SYMBOLS_TO_SAVE", 1)
    monkeypatch.setattr(sb, "_code_sha", lambda: "test-build")
    report = sb.run_backtest(decision_grid=sb.GRID_ROLLING)
    att = report["evidence_provenance"]["alpha_accounting"]
    assert not att.get("look_id") and att.get("unreserved") == "REPORT_NOT_SAVED"


def test_c2_10_modern_equity_sizes_down_to_the_configured_account_limit(monkeypatch):
    from config import settings
    from research.auto_research.paper_book import PaperBook
    from tests.test_paper_autopilot_money_path import _cycle, _eligible_card

    monkeypatch.setattr(settings, "risk_per_trade_pct", 0.005)
    book = PaperBook(capital=100_000)
    result = _cycle(book, [_eligible_card()])
    assert result["taken"]
    pos = next(iter(book.open.values()))
    assert pos.qty * (pos.entry_price - pos.stop_price) <= 500 + 1e-6
