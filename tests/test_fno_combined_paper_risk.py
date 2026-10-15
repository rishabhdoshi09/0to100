"""Combined equity-plus-F&O PAPER risk must share a single 5% account ceiling."""
import pytest

from product.fo_paper import FoPaperBook
from product.fo_paper_store import FoPaperStore


def _open(book, option_symbol="TEST26OCTCE"):
    return book.open_position(
        underlying="TEST", option_symbol=option_symbol, option_type="CE",
        entry=100.0, stop=90.0, target=120.0,
        lot_size=50, ask=100.0, opened_at="2026-10-14T10:45:00+05:30",
        max_holding_sessions=1,
    )


def test_equity_risk_consumes_fno_aggregate_five_percent_cap():
    blocked = FoPaperBook(capital=100_000, external_open_risk=4_900)
    assert _open(blocked) is None
    assert blocked.refusals[-1][1] == "RISK_OR_PREMIUM_BUDGET_TOO_SMALL_FOR_ONE_LOT"

    permitted = FoPaperBook(capital=100_000, external_open_risk=4_500)
    pos = _open(permitted)
    assert pos is not None
    assert pos.risk_amount == 500
    assert 4_500 + pos.risk_amount == 5_000


def test_account_equity_gate_reads_durable_fno_paper_rows(tmp_path, monkeypatch):
    from core import runtime_paths
    from product import fo_paper_store, paper_status
    from risk import portfolio_risk, position_manager

    filename = tmp_path / "fo_paper.sqlite3"
    with FoPaperStore(filename) as store:
        store.replace_positions([{
            "trade_id": "one", "underlying": "TEST", "option_symbol": "TEST26OCTCE",
            "quantity": 50, "entry_price": 100.0, "stop_price": 90.0,
            "opened_at": "2026-10-14T10:45:00+05:30",
        }])

    monkeypatch.setattr(runtime_paths, "logs_path", lambda *args: filename)
    monkeypatch.setattr(fo_paper_store, "logs_path", lambda *args: filename)
    monkeypatch.setattr(position_manager, "_open_trades", lambda: [])
    monkeypatch.setattr(paper_status, "modern_engine_open_positions", lambda: [])

    rows = portfolio_risk._combined_open_rows()
    assert rows == [{"symbol": "TEST", "qty": 50, "entry": 100.0, "stop": 90.0}]
    assert portfolio_risk._combined_open_rows(include_fno=False) == []
    assert portfolio_risk.account_exposure_gate("TEST")["reason_code"] == "DUPLICATE_POSITION"


def test_corrupt_existing_fo_book_cannot_be_treated_as_zero_risk(tmp_path, monkeypatch):
    from core import runtime_paths
    from product import fo_paper_store, paper_status
    from risk import portfolio_risk, position_manager

    filename = tmp_path / "fo_paper.sqlite3"
    filename.write_text("NOT A SQLITE DB", encoding="utf-8")
    monkeypatch.setattr(runtime_paths, "logs_path", lambda *args: filename)
    monkeypatch.setattr(fo_paper_store, "logs_path", lambda *args: filename)
    monkeypatch.setattr(position_manager, "_open_trades", lambda: [])
    monkeypatch.setattr(paper_status, "modern_engine_open_positions", lambda: [])
    with pytest.raises(Exception):
        portfolio_risk._combined_open_rows()



def test_actual_fno_paper_cycle_obeys_smaller_equity_account_capital(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from datetime import datetime, timedelta, timezone
    from product.fo_paper_runtime import run_fo_paper_cycle
    from product import fno_exposure_tracking
    from risk import portfolio_risk

    monkeypatch.setattr(portfolio_risk, "_combined_open_rows", lambda **kwargs: [])
    monkeypatch.setattr(portfolio_risk, "account_mutation_lock", nullcontext)
    monkeypatch.setattr(portfolio_risk, "_capital", lambda: 2_000.0)
    monkeypatch.setattr(fno_exposure_tracking, "record_scan_exposure", lambda *a, **k: None)
    directional = {
        "available": True,
        "candidates": [{
            "symbol": "TEST", "decision": "PAPER_OPTION_CANDIDATE",
            "setup": {"direction": "LONG", "score": 78,
                      "expected_move": {"holding_days": 1}},
            "selected_contract": {
                "symbol": "TEST26OCTCE", "option_type": "CE", "eligible": True,
                "lot_size": 50, "ask": 100, "score": 85,
                "trade_plan": {"entry": 100, "stop": 90, "target": 120},
            },
        }],
    }
    now = datetime(2026, 10, 14, 10, 45, tzinfo=timezone(timedelta(hours=5, minutes=30)))
    with FoPaperStore(tmp_path / "cross-book-cap.sqlite3") as ledger:
        blocked = run_fo_paper_cycle(
            directional, client=object(), now_ist=now,
            allow_new_entries=True, store=ledger, capital=100_000,
        )
        assert blocked["opened_count"] == 0
        assert blocked["skipped"][-1]["reason"] == "RISK_OR_PREMIUM_BUDGET_TOO_SMALL_FOR_ONE_LOT"
        # Same market signal is allowed under a genuinely larger account risk
        # limit, showing the rejection came from the shared capital ceiling.
        monkeypatch.setattr(portfolio_risk, "_capital", lambda: 100_000.0)
        accepted = run_fo_paper_cycle(
            directional, client=object(), now_ist=now,
            allow_new_entries=True, store=ledger, capital=100_000,
        )
        assert accepted["opened_count"] == 1
        assert accepted["opened"][0]["risk_amount"] <= 1000
