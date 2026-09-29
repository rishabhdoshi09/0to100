from product.fo_paper import ENTRY_MINUTE_CLEAR, FoPaperBook


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
    pos.entry_minute_status = ENTRY_MINUTE_CLEAR
    closed = book.mark({
        "RELIANCECE": {"open": 50, "high": 75, "low": 49, "close": 72, "bid": 70}
    }, session="2026-09-27")
    assert closed[0].costs == 100.0
    assert closed[0].net_pnl == closed[0].gross_pnl - 100.0
    assert book.evidence_rows()[0]["production_evidence_eligible"] is True


def test_intraday_repeated_marks_do_not_consume_holding_sessions():
    book = FoPaperBook(capital=200_000, slippage_bps=0)
    pos = book.open_position(
        underlying="RELIANCE", option_symbol="RELIANCECE", option_type="CE",
        entry=50, stop=40, target=80, lot_size=25,
        opened_at="2026-09-26T10:00:00+05:30", max_holding_sessions=2,
    )
    assert pos is not None
    quote = {"open": 50, "high": 55, "low": 45, "close": 52, "bid": 52}
    assert book.mark({"RELIANCECE": quote}, session="2026-09-26") == []
    assert pos.bars_held == 0
    assert book.mark({"RELIANCECE": quote}, session="2026-09-26") == []
    assert pos.bars_held == 0
    assert book.mark({"RELIANCECE": quote}, session="2026-09-27") == []
    assert pos.bars_held == 1


def test_only_one_option_position_per_underlying_is_allowed():
    book = FoPaperBook(capital=500_000, slippage_bps=0)
    first = book.open_position(
        underlying="RELIANCE", option_symbol="RELIANCECE1", option_type="CE",
        entry=50, stop=40, target=80, lot_size=25,
        opened_at="2026-09-26", max_holding_sessions=2,
    )
    assert first is not None
    second = book.open_position(
        underlying="RELIANCE", option_symbol="RELIANCECE2", option_type="CE",
        entry=45, stop=35, target=70, lot_size=25,
        opened_at="2026-09-26", max_holding_sessions=2,
    )
    assert second is None
    assert book.refusals[-1][1] == "UNDERLYING_ALREADY_OPEN"


def test_total_open_risk_cap_is_enforced_across_positions():
    book = FoPaperBook(
        capital=100_000,
        risk_per_trade_pct=0.03,
        max_total_risk_pct=0.05,
        max_premium_pct=1.0,
        slippage_bps=0,
    )
    first = book.open_position(
        underlying="AAA", option_symbol="AAACE", option_type="CE",
        entry=50, stop=40, target=80, lot_size=100,
        opened_at="2026-09-26", max_holding_sessions=2,
    )
    assert first is not None
    second = book.open_position(
        underlying="BBB", option_symbol="BBBCE", option_type="CE",
        entry=50, stop=40, target=80, lot_size=100,
        opened_at="2026-09-26", max_holding_sessions=2,
    )
    assert second is not None
    third = book.open_position(
        underlying="CCC", option_symbol="CCCCE", option_type="CE",
        entry=50, stop=40, target=80, lot_size=100,
        opened_at="2026-09-26", max_holding_sessions=2,
    )
    assert third is None
    assert book.refusals[-1][1] == "RISK_OR_PREMIUM_BUDGET_TOO_SMALL_FOR_ONE_LOT"


def test_executable_ask_cannot_cross_above_target():
    book = FoPaperBook(capital=200_000, slippage_bps=0)
    pos = book.open_position(
        underlying="RELIANCE",
        option_symbol="RELIANCECE",
        option_type="CE",
        entry=50,
        stop=40,
        target=55,
        lot_size=25,
        opened_at="2026-09-26",
        max_holding_sessions=2,
        ask=56,
    )
    assert pos is None
    assert book.refusals[-1][1] == "TARGET_NOT_ABOVE_EXECUTABLE_ENTRY"



def test_configured_costs_still_hold_probability_when_entry_path_is_ambiguous():
    def costs(entry, exit, qty):
        return 10.0

    book = FoPaperBook(
        capital=200_000,
        slippage_bps=0,
        cost_model=costs,
        cost_model_name="TEST_COSTS",
    )
    pos = book.open_position(
        underlying="RELIANCE",
        option_symbol="RELIANCECE",
        option_type="CE",
        entry=50,
        stop=40,
        target=70,
        lot_size=25,
        opened_at="2026-09-26T10:15:12+05:30",
        max_holding_sessions=1,
    )
    assert pos is not None
    pos.entry_minute_status = "AMBIGUOUS_BOUNDARY_TOUCH"
    book.mark({
        "RELIANCECE": {"open": 50, "high": 55, "low": 45, "close": 52, "bid": 52}
    }, session="2026-09-27")

    row = book.evidence_rows()[0]
    assert row["cost_model_status"].startswith("CONFIGURED:")
    assert row["path_observation_complete"] is False
    assert row["production_evidence_eligible"] is False
    assert row["evidence_exclusion_reason"] == "ENTRY_MINUTE_AMBIGUOUS_BOUNDARY_TOUCH"


def test_daily_premium_cap_accumulates_across_multiple_positions():
    book = FoPaperBook(
        capital=100_000,
        risk_per_trade_pct=0.10,
        max_premium_pct=0.10,
        max_daily_premium_pct=0.10,
        max_total_risk_pct=0.50,
        slippage_bps=0,
    )
    first = book.open_position(
        underlying="AAA",
        option_symbol="AAACE",
        option_type="CE",
        entry=50,
        stop=40,
        target=80,
        lot_size=50,
        opened_at="2026-09-29T10:00:00+05:30",
        max_holding_sessions=2,
        requested_lots=2,
    )
    second = book.open_position(
        underlying="BBB",
        option_symbol="BBBCE",
        option_type="CE",
        entry=50,
        stop=40,
        target=80,
        lot_size=50,
        opened_at="2026-09-29T10:05:00+05:30",
        max_holding_sessions=2,
        requested_lots=2,
    )
    third = book.open_position(
        underlying="CCC",
        option_symbol="CCCCE",
        option_type="CE",
        entry=50,
        stop=40,
        target=80,
        lot_size=50,
        opened_at="2026-09-29T10:10:00+05:30",
        max_holding_sessions=2,
        requested_lots=1,
    )

    assert first is not None
    assert second is not None
    assert book.premium_deployed_today == 10_000.0
    assert third is None
    assert book.refusals[-1][1] == "DAILY_PREMIUM_BUDGET_EXHAUSTED"


def test_seeded_daily_premium_prevents_restart_budget_reset():
    book = FoPaperBook(
        capital=100_000,
        risk_per_trade_pct=0.10,
        max_premium_pct=0.10,
        max_daily_premium_pct=0.10,
        premium_deployed_today=9_000.0,
        max_total_risk_pct=0.50,
        slippage_bps=0,
    )
    pos = book.open_position(
        underlying="AAA",
        option_symbol="AAACE",
        option_type="CE",
        entry=50,
        stop=40,
        target=80,
        lot_size=50,
        opened_at="2026-09-29T11:00:00+05:30",
        max_holding_sessions=2,
        requested_lots=1,
    )

    assert pos is None
    assert book.premium_deployed_today == 9_000.0
    assert book.refusals[-1][1] == "DAILY_PREMIUM_BUDGET_EXHAUSTED"


def test_trailing_stop_uses_prior_observed_path_not_same_bar_high():
    book = FoPaperBook(
        capital=200_000,
        slippage_bps=0,
        trail_activation_r=1.0,
        trail_distance_r=1.0,
    )
    pos = book.open_position(
        underlying="RELIANCE",
        option_symbol="TRAILCE",
        option_type="CE",
        entry=50,
        stop=40,
        target=80,
        lot_size=25,
        opened_at="2026-09-29T10:00:00+05:30",
        max_holding_sessions=4,
    )
    assert pos is not None

    # This bar first earns the trail. Its own low may not be tested against a
    # stop tightened by its later/unknown-order high.
    assert book.mark({
        "TRAILCE": {"open": 50, "high": 62, "low": 45, "close": 60, "bid": 59}
    }, session="2026-09-29T10:30:00+05:30", advance_session=False) == []
    assert pos.trailing_stop_price == 52.0

    closed = book.mark({
        "TRAILCE": {"open": 60, "high": 61, "low": 51, "close": 52, "bid": 51.5}
    }, session="2026-09-29T10:31:00+05:30", advance_session=False)
    assert len(closed) == 1
    assert closed[0].exit_reason == "TRAIL_STOP"
    assert closed[0].exit_price == 52.0


def test_intraday_eod_policy_exits_at_current_bid_only_when_forced():
    book = FoPaperBook(capital=200_000, slippage_bps=0)
    pos = book.open_position(
        underlying="RELIANCE",
        option_symbol="EODCE",
        option_type="CE",
        entry=50,
        stop=40,
        target=80,
        lot_size=25,
        opened_at="2026-09-29T11:00:00+05:30",
        max_holding_sessions=1,
        horizon="INTRADAY",
        exit_policy="EOD",
    )
    assert pos is not None
    quote = {"open": 55, "high": 55, "low": 55, "close": 55, "bid": 54.5}
    assert book.mark(
        {"EODCE": quote},
        session="2026-09-29T15:34:00+05:30",
        advance_session=False,
        force_eod=False,
    ) == []
    closed = book.mark(
        {"EODCE": quote},
        session="2026-09-29T15:35:00+05:30",
        advance_session=False,
        observation_complete=False,
        force_eod=True,
    )
    assert len(closed) == 1
    assert closed[0].exit_reason == "EOD"
    assert closed[0].exit_price == 54.5


def test_iv_crush_exit_uses_current_bid_and_does_not_affect_other_positions():
    book = FoPaperBook(capital=500_000, slippage_bps=0)
    crushed = book.open_position(
        underlying="AAA",
        option_symbol="AAACE",
        option_type="CE",
        entry=50,
        stop=35,
        target=80,
        lot_size=25,
        opened_at="2026-09-29T10:00:00+05:30",
        max_holding_sessions=2,
    )
    clear = book.open_position(
        underlying="BBB",
        option_symbol="BBBCE",
        option_type="CE",
        entry=50,
        stop=35,
        target=80,
        lot_size=25,
        opened_at="2026-09-29T10:00:00+05:30",
        max_holding_sessions=2,
    )
    assert crushed is not None and clear is not None
    quotes = {
        "AAACE": {"open": 45, "high": 45, "low": 45, "close": 45, "bid": 44.5},
        "BBBCE": {"open": 52, "high": 52, "low": 52, "close": 52, "bid": 51.5},
    }
    closed = book.mark(
        quotes,
        session="2026-09-29T12:00:00+05:30",
        advance_session=False,
        observation_complete=False,
        iv_crush_symbols={"AAACE"},
    )
    assert len(closed) == 1
    assert closed[0].option_symbol == "AAACE"
    assert closed[0].exit_reason == "IV_CRUSH"
    assert closed[0].exit_price == 44.5
    assert "BBBCE" in book.open


def test_exit_bar_does_not_credit_post_exit_mfe():
    book = FoPaperBook(capital=200_000, slippage_bps=0)
    pos = book.open_position(
        underlying="RELIANCE",
        option_symbol="MFECE",
        option_type="CE",
        entry=50,
        stop=40,
        target=80,
        lot_size=25,
        opened_at="2026-09-29T10:00:00+05:30",
        max_holding_sessions=2,
    )
    assert pos is not None
    closed = book.mark({
        # The bar high is irrelevant after the conservative stop exit.
        "MFECE": {"open": 50, "high": 79, "low": 39, "close": 70, "bid": 69}
    }, session="2026-09-29T10:30:00+05:30", advance_session=False)
    assert len(closed) == 1
    assert closed[0].exit_reason == "STOP"
    assert closed[0].mfe_pct == 0.0


def test_target_exit_caps_mfe_at_target_not_later_bar_extension():
    book = FoPaperBook(capital=200_000, slippage_bps=0)
    pos = book.open_position(
        underlying="RELIANCE",
        option_symbol="TARGETMFECE",
        option_type="CE",
        entry=50,
        stop=40,
        target=70,
        lot_size=25,
        opened_at="2026-09-29T10:00:00+05:30",
        max_holding_sessions=2,
    )
    assert pos is not None
    closed = book.mark({
        "TARGETMFECE": {"open": 50, "high": 90, "low": 45, "close": 85, "bid": 84}
    }, session="2026-09-29T10:30:00+05:30", advance_session=False)
    assert len(closed) == 1
    assert closed[0].exit_reason == "TARGET"
    assert closed[0].mfe_pct == 40.0
