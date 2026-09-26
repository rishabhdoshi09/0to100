from product.fo_costs import (
    NseLongOptionCostSchedule,
    zerodha_nse_option_cost_model_from_env,
)
from product.fo_paper import FoPaperBook


def test_standard_zerodha_nse_option_round_trip_cost_matches_published_schedule():
    schedule = NseLongOptionCostSchedule()
    # Buy ₹100 x 50, sell ₹110 x 50.
    # Includes ₹20/order brokerage, NSE option transaction charges,
    # sell-side STT, buy-side stamp, SEBI turnover fee and GST.
    assert schedule.round_trip_cost(100.0, 110.0, 50) == 59.76
    assert schedule.effective_from == "2026-04-01"
    assert "ZERODHA_NSE_LONG_OPTIONS_2026_04" in schedule.name


def test_stt_uses_nearest_rupee_half_up_rounding():
    schedule = NseLongOptionCostSchedule(
        brokerage_per_order=0.0,
        transaction_charge_pct=0.0,
        stt_sell_pct=0.15,
        stamp_buy_pct=0.0,
        sebi_charge_per_crore=0.0,
        gst_pct=0.0,
    )
    # ₹1,000 sell premium turnover × 0.15% = ₹1.50 -> ₹2 STT.
    assert schedule.round_trip_cost(10.0, 10.0, 100) == 2.0


def test_cost_schedule_supports_explicit_rate_overrides(monkeypatch):
    monkeypatch.setenv("QT_ZERODHA_OPTION_BROKERAGE_PER_ORDER", "40")
    monkeypatch.setenv("QT_NSE_OPTION_TRANSACTION_CHARGE_PCT", "0.04")
    model, name = zerodha_nse_option_cost_model_from_env()

    assert ":B40:" in name
    assert ":TXN0.04%:" in name
    assert model(100.0, 110.0, 50) > NseLongOptionCostSchedule().round_trip_cost(100.0, 110.0, 50)


def test_configured_cost_model_makes_closed_paper_trade_net_and_evidence_eligible():
    schedule = NseLongOptionCostSchedule()
    book = FoPaperBook(
        capital=100_000.0,
        cost_model=schedule.round_trip_cost,
        cost_model_name=schedule.name,
        slippage_bps=0.0,
    )
    pos = book.open_position(
        underlying="TEST",
        option_symbol="TESTCE",
        option_type="CE",
        entry=100.0,
        stop=90.0,
        target=120.0,
        lot_size=10,
        opened_at="2026-09-26T10:00:00+05:30",
        max_holding_sessions=1,
        context_key="CTX",
        setup_score=75.0,
        option_score=80.0,
        ask=100.0,
        requested_lots=1,
    )
    assert pos is not None

    settled = book.mark(
        {"TESTCE": {"open": 100.0, "high": 121.0, "low": 100.0, "close": 120.0}},
        session="2026-09-27",
    )
    assert len(settled) == 1
    trade = settled[0]
    assert trade.costs > 0
    assert trade.net_pnl < trade.gross_pnl
    assert trade.cost_model_status.startswith("CONFIGURED:ZERODHA_NSE_LONG_OPTIONS_2026_04")

    evidence = book.evidence_rows()
    assert evidence[0]["production_evidence_eligible"] is True
