from product.fo_paper import FoPaperBook


def test_option_paper_book_enforces_lot_risk_and_premium_caps():
    book = FoPaperBook(
        capital=100_000,
        risk_per_trade_pct=0.01,
        max_premium_pct=0.10,
        slippage_bps=0,
    )
    pos = book.open_position(
        underlying="RELIANCE",
        option_symbol="RELIANCECE",
        option_type="CE",
        entry=50,
        stop=40,
        target=70,
        lot_size=25,
        opened_at="2026-09-26",
        max_holding_sessions=2,
    )
    assert pos is not None
    assert pos.quantity % 25 == 0
    assert pos.risk_amount <= 1000.0 + 1e-6
    assert pos.entry_price * pos.quantity <= 10_000.0 + 1e-6


def test_option_book_never_supports_naked_short_or_invalid_stop():
    book = FoPaperBook()
    assert book.open_position(
        underlying="RELIANCE", option_symbol="BAD", option_type="SHORT_CE",
        entry=50, stop=40, target=70, lot_size=25,
        opened_at="2026-09-26", max_holding_sessions=2,
    ) is None
    assert book.open_position(
        underlying="RELIANCE", option_symbol="BAD2", option_type="CE",
        entry=50, stop=55, target=70, lot_size=25,
        opened_at="2026-09-26", max_holding_sessions=2,
    ) is None


def test_ambiguous_same_bar_uses_stop_first_conservatively():
    book = FoPaperBook(capital=200_000, slippage_bps=0)
    pos = book.open_position(
        underlying="RELIANCE", option_symbol="RELIANCECE", option_type="CE",
        entry=50, stop=40, target=70, lot_size=25,
        opened_at="2026-09-26", max_holding_sessions=4,
    )
    assert pos is not None
    closed = book.mark({
        "RELIANCECE": {"open": 50, "high": 75, "low": 35, "close": 60, "bid": 40}
    }, session="2026-09-27")
    assert len(closed) == 1
    assert closed[0].exit_reason == "AMBIGUOUS_BAR_STOP_FIRST"
    assert closed[0].exit_price == 40


def test_unconfigured_statutory_costs_are_not_promoted_as_net_evidence():
    book = FoPaperBook(capital=200_000, slippage_bps=0)
    pos = book.open_position(
        underlying="RELIANCE", option_symbol="RELIANCECE", option_type="CE",
        entry=50, stop=40, target=70, lot_size=25,
        opened_at="2026-09-26", max_holding_sessions=1,
    )
    assert pos is not None
    book.mark({
        "RELIANCECE": {"open": 50, "high": 55, "low": 45, "close": 52, "bid": 52}
    }, session="2026-09-27")
    row = book.evidence_rows()[0]
    assert row["cost_model_status"] == "UNCONFIGURED_GROSS_ONLY"
    assert row["production_evidence_eligible"] is False


def test_configured_cost_model_flows_into_net_option_return():
    def costs(entry, exit, qty):
        return 100.0

    book = FoPaperBook(
        capital=200_000, slippage_bps=0,
        cost_model=costs, cost_model_name="TEST_VERSIONED_COSTS",
    )
    pos = book.open_position(
        underlying="RELIANCE", option_symbol="RELIANCECE", option_type="CE",
        entry=50, stop=40, target=70, lot_size=25,
        opened_at="2026-09-26", max_holding_sessions=1,
    )
    assert pos is not None
    closed = book.mark({
        "RELIANCECE": {"open": 50, "high": 75, "low": 49, "close": 72, "bid": 70}
    }, session="2026-09-27")
    assert closed[0].costs == 100.0
    assert closed[0].net_pnl == closed[0].gross_pnl - 100.0
    assert book.evidence_rows()[0]["production_evidence_eligible"] is True
