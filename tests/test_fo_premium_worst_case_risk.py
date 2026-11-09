"""Long-option risk is capped by full premium, fees, and shared account heat."""
from __future__ import annotations

from product.fo_paper import FoPaperBook, long_option_max_loss_exposure
from risk.portfolio_risk import FNO_BOOK, _row_risk


def _enter(book, symbol="RELIANCECE", *, lot_size=10, entry=50, stop=40, ask=None):
    return book.open_position(
        underlying="RELIANCE" if symbol == "RELIANCECE" else "TCS",
        option_symbol=symbol, option_type="CE", entry=entry, stop=stop,
        target=entry * 1.5, lot_size=lot_size, ask=ask,
        opened_at="2026-11-09T10:15:00+05:30", max_holding_sessions=2,
    )


def test_reject_one_lot_when_premium_exceeds_entire_1pct_budget():
    book = FoPaperBook(capital=100_000, slippage_bps=0)
    assert _enter(book, lot_size=25) is None  # Rs 1250 premium > Rs 1000
    assert book.refusals[-1][1] == "MAX_LOSS_PREMIUM_BUDGET_TOO_SMALL_FOR_ONE_LOT"


def test_stop_is_not_the_max_loss_budget_and_costs_are_included():
    book = FoPaperBook(
        capital=100_000, slippage_bps=0,
        cost_model=lambda _entry, _exit, qty: 100.0,
        cost_model_name="TEST_FEES",
    )
    pos = _enter(book, lot_size=10)
    assert pos is not None
    # One lot 500 + 100 fees, two lots 1000 + 100 fees exceeds 1% cap.
    assert pos.lots == 1
    assert pos.risk_amount == 100.0  # old stop-based metric remains comparable
    assert pos.premium_loss_exposure == 600.0
    assert long_option_max_loss_exposure(pos) == 600.0
    assert _row_risk({"book": FNO_BOOK, "qty": 10, "entry": 50, "stop": 40,
                      "premium_loss_exposure": 600}) == 600.0


def test_executable_ask_and_full_fee_estimate_control_sizing():
    book = FoPaperBook(
        capital=100_000, slippage_bps=0,
        cost_model=lambda _entry, _exit, _qty: 100.0,
        cost_model_name="TEST_FEES",
    )
    pos = _enter(book, lot_size=10, entry=50, stop=40, ask=95)
    assert pos is None  # premium 950 + costs 100 > 1000


def test_old_open_positions_are_still_counted_at_full_premium():
    book = FoPaperBook(capital=100_000, slippage_bps=0)
    old = _enter(FoPaperBook(capital=200_000, slippage_bps=0), lot_size=10)
    assert old is not None
    old.premium_loss_exposure = 0.0
    old.risk_amount = 100.0
    assert long_option_max_loss_exposure(old) == old.entry_price * old.quantity
    book.open[old.option_symbol] = old
    # Rs 500 open premium + Rs 4900 external account risk blocks one more lot.
    book.external_open_risk = 4_900.0
    assert _enter(book, "TCSCE") is None


def test_shared_equity_gate_uses_full_premium_for_legacy_fno_rows():
    old = {"book": FNO_BOOK, "qty": 25, "entry": 50.0, "stop": 49.0}
    assert _row_risk(old) == 1250.0  # not 25 of stop-distance risk


def test_invalid_cost_model_fails_closed_with_no_position_mutation():
    book = FoPaperBook(
        capital=100_000, slippage_bps=0,
        cost_model=lambda *_args: float("nan"),
        cost_model_name="BROKEN_COSTS",
    )
    assert _enter(book) is None
    assert book.refusals[-1][1] == "OPTION_COST_ESTIMATE_UNAVAILABLE"
    assert book.open == {}
