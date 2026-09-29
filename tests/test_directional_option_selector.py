from datetime import date

import options.directional_selector as selector
from options.directional_selector import (
    black_scholes,
    score_option_contract,
    select_option_contracts,
)


def _contract(symbol: str, strike: float, delta=None, *, option_type="CE"):
    fair = black_scholes(
        spot=3050.0, strike=strike, dte=9, iv=0.24, option_type=option_type,
    )["price"]
    row = {
        "symbol": symbol,
        "option_type": option_type,
        "strike": strike,
        "expiry": "2026-10-01",
        "dte": 9,
        "as_of_date": "2026-09-22",
        "quote_timestamp": "2026-09-22T10:00:00",
        "trading_sessions_to_expiry": 7,
        "session_dates_to_expiry": [
            "2026-09-23",
            "2026-09-24",
            "2026-09-25",
            "2026-09-28",
            "2026-09-29",
            "2026-09-30",
            "2026-10-01",
        ],
        "holiday_calendar_loaded": 1,
        "expiry_session_model": "NSE_SESSIONS_WITH_RUNTIME_HOLIDAYS",
        "ltp": fair,
        "bid": max(0.05, fair - 0.25),
        "ask": fair + 0.25,
        "volume": 5500,
        "oi": 32000,
        "iv": 24.0,
    }
    if delta is not None:
        row["delta"] = delta
    return row


def test_dte_fallback_uses_exchange_local_today(monkeypatch):
    monkeypatch.setattr(selector, "_today_ist", lambda: date(2026, 9, 29))
    assert selector._dte({"expiry": "2026-10-01"}) == 2


def test_black_scholes_has_sane_call_and_put_greeks():
    call = black_scholes(spot=1000, strike=1000, dte=10, iv=0.25, option_type="CE")
    put = black_scholes(spot=1000, strike=1000, dte=10, iv=0.25, option_type="PE")
    assert call["price"] > 0
    assert put["price"] > 0
    assert 0 < call["delta"] < 1
    assert -1 < put["delta"] < 0
    assert call["gamma"] > 0
    assert call["theta_per_day"] < 0


def test_directional_contract_is_scored_and_scenario_repriced():
    result = score_option_contract(
        _contract("RELIANCE27SEP3050CE", 3050.0, 0.62),
        direction="LONG",
        spot=3050.0,
        expected_move_pct=2.0,
        horizon="1_TO_2D",
        holding_days=2,
        underlying_stop_price=2995.0,
        iv_percentile=45.0,
    )
    assert result["eligible"] is True
    assert result["score"] >= 60
    assert result["score_is_probability"] is False
    assert result["scenario_model"] == "BLACK_SCHOLES_CONSTANT_IV_ESTIMATE"
    assert result["trade_plan"]["entry"] > result["trade_plan"]["stop"]
    assert result["trade_plan"]["target"] > result["trade_plan"]["entry"]
    assert result["moneyness"] == "ATM"
    assert result["gamma_delta_change_for_1pct_move"] > 0
    assert result["vega_pct_of_premium_per_vol_point"] > 0
    moves = {row["underlying_move_pct"]: row for row in result["scenarios"]}
    assert moves[2.0]["projected_option_return_pct"] > moves[1.0]["projected_option_return_pct"]


def test_missing_bid_ask_fails_closed_even_with_attractive_delta():
    contract = _contract("RELIANCE27SEP3050CE", 3050.0, 0.62)
    contract["bid"] = 0
    contract["ask"] = 0
    result = score_option_contract(
        contract,
        direction="LONG",
        spot=3050.0,
        expected_move_pct=2.0,
        horizon="1_TO_2D",
        holding_days=2,
        underlying_stop_price=2995.0,
    )
    assert result["eligible"] is False
    assert "BID_ASK_UNAVAILABLE" in result["blockers"]


def test_wrong_side_and_far_otm_delta_are_rejected():
    wrong = score_option_contract(
        _contract("RELIANCE27SEP3000PE", 3000.0, -0.6, option_type="PE"),
        direction="LONG",
        spot=3050.0,
        expected_move_pct=2.0,
        horizon="1_TO_2D",
        holding_days=2,
        underlying_stop_price=2995.0,
    )
    assert wrong["eligible"] is False
    assert "WRONG_OPTION_SIDE" in wrong["blockers"]

    far = score_option_contract(
        _contract("RELIANCE27SEP3300CE", 3300.0, 0.18),
        direction="LONG",
        spot=3050.0,
        expected_move_pct=2.0,
        horizon="1_TO_2D",
        holding_days=2,
        underlying_stop_price=2995.0,
    )
    assert far["eligible"] is False
    assert "DELTA_OUTSIDE_DIRECTIONAL_BAND" in far["blockers"]


def test_selector_ranks_eligible_contracts_only():
    rows = [
        _contract("GOOD", 3050.0, 0.62),
        _contract("DEEP", 2900.0, 0.82),
        _contract("WRONG", 3050.0, -0.60, option_type="PE"),
    ]
    result = select_option_contracts(
        rows,
        direction="LONG",
        spot=3050.0,
        expected_move_pct=2.0,
        horizon="1_TO_2D",
        holding_days=2,
        underlying_stop_price=2995.0,
        iv_percentile=40.0,
    )
    assert result["eligible_count"] >= 1
    assert all(row["eligible"] for row in result["best_contracts"])
    assert all(row["option_type"] == "CE" for row in result["best_contracts"])


def test_missing_iv_percentile_adds_no_unearned_score():
    contract = _contract("RELIANCE27SEP3050CE", 3050.0, 0.62)
    missing = score_option_contract(
        contract,
        direction="LONG",
        spot=3050.0,
        expected_move_pct=2.0,
        horizon="1_TO_2D",
        holding_days=2,
        underlying_stop_price=2995.0,
        iv_percentile=None,
    )
    known = score_option_contract(
        contract,
        direction="LONG",
        spot=3050.0,
        expected_move_pct=2.0,
        horizon="1_TO_2D",
        holding_days=2,
        underlying_stop_price=2995.0,
        iv_percentile=45.0,
    )

    assert missing["iv_percentile"] is None
    assert missing["iv_percentile_available"] is False
    assert missing["components"]["iv"] == 0.0
    assert known["iv_percentile_available"] is True
    assert known["components"]["iv"] > 0.0
    assert known["score"] > missing["score"]


def test_contract_must_outlive_holding_horizon_in_trading_sessions():
    contract = _contract("RELIANCE27SEP3050CE", 3050.0, 0.62)
    # Calendar DTE alone looks sufficient, but only four future NSE sessions
    # remain. A four-session hold needs an additional buffer session.
    contract["dte"] = 6
    contract["trading_sessions_to_expiry"] = 4

    result = score_option_contract(
        contract,
        direction="LONG",
        spot=3050.0,
        expected_move_pct=2.0,
        horizon="2_TO_4D",
        holding_days=4,
        underlying_stop_price=2995.0,
        iv_percentile=45.0,
    )

    assert result["eligible"] is False
    assert "TRADING_SESSIONS_SHORTER_THAN_HOLDING_HORIZON" in result["blockers"]


def test_missing_trading_session_expiry_provenance_fails_closed():
    contract = _contract("RELIANCE27SEP3050CE", 3050.0, 0.62)
    contract.pop("trading_sessions_to_expiry")

    result = score_option_contract(
        contract,
        direction="LONG",
        spot=3050.0,
        expected_move_pct=2.0,
        horizon="1_TO_2D",
        holding_days=2,
        underlying_stop_price=2995.0,
        iv_percentile=45.0,
    )

    assert result["eligible"] is False
    assert "TRADING_SESSION_EXPIRY_UNAVAILABLE" in result["blockers"]


def test_missing_holiday_calendar_blocks_trading_session_eligibility():
    contract = _contract("RELIANCE27SEP3050CE", 3050.0, 0.62)
    contract["holiday_calendar_loaded"] = 0
    contract["expiry_session_model"] = "WEEKDAYS_ONLY_NO_HOLIDAY_TABLE"

    result = score_option_contract(
        contract,
        direction="LONG",
        spot=3050.0,
        expected_move_pct=2.0,
        horizon="1_TO_2D",
        holding_days=2,
        underlying_stop_price=2995.0,
        iv_percentile=45.0,
    )

    assert result["eligible"] is False
    assert "TRADING_CALENDAR_HOLIDAYS_UNAVAILABLE" in result["blockers"]


def test_expiry_fit_ranks_by_trading_sessions_not_calendar_dte():
    preferred = _contract("PREFERRED", 3050.0, 0.62)
    farther = _contract("FARTHER", 3050.0, 0.62)
    # Hold calendar DTE constant so only exchange-session distance can affect
    # expiry-fit ranking. Six future sessions is near/next-week fit for 1_TO_2D;
    # thirteen is outside the preferred band.
    preferred["dte"] = 30
    farther["dte"] = 30
    preferred["trading_sessions_to_expiry"] = 6
    farther["trading_sessions_to_expiry"] = 13

    preferred_score = score_option_contract(
        preferred,
        direction="LONG",
        spot=3050.0,
        expected_move_pct=2.0,
        horizon="1_TO_2D",
        holding_days=2,
        underlying_stop_price=2995.0,
        iv_percentile=45.0,
    )
    farther_score = score_option_contract(
        farther,
        direction="LONG",
        spot=3050.0,
        expected_move_pct=2.0,
        horizon="1_TO_2D",
        holding_days=2,
        underlying_stop_price=2995.0,
        iv_percentile=45.0,
    )

    assert preferred_score["expiry_fit_basis"] == "TRADING_SESSIONS_TO_EXPIRY"
    assert preferred_score["components"]["expiry_fit"] == 10.0
    assert farther_score["components"]["expiry_fit"] == 6.0
    assert preferred_score["components"]["expiry_fit"] > farther_score["components"]["expiry_fit"]


def test_two_session_hold_across_weekend_uses_four_calendar_days_of_decay():
    contract = _contract("WEEKEND", 3050.0, 0.62)
    contract.update({
        "expiry": "2026-10-01",
        "dte": 6,
        "as_of_date": "2026-09-25",
        "trading_sessions_to_expiry": 4,
        "session_dates_to_expiry": [
            "2026-09-28",
            "2026-09-29",
            "2026-09-30",
            "2026-10-01",
        ],
    })

    result = score_option_contract(
        contract,
        direction="LONG",
        spot=3050.0,
        expected_move_pct=2.0,
        horizon="1_TO_2D",
        holding_days=2,
        underlying_stop_price=2995.0,
        iv_percentile=45.0,
    )

    expected_target = black_scholes(
        spot=3050.0 * 1.02,
        strike=3050.0,
        dte=2,  # six calendar DTE less Fri->Tue's four elapsed calendar days
        iv=0.24,
        option_type="CE",
    )["price"]
    assert result["calendar_days_to_first_session"] == 3
    assert result["calendar_days_to_holding_horizon"] == 4
    assert result["scenario_decay_basis"] == "CALENDAR_DAYS_TO_NSE_SESSION_HORIZON"
    assert result["trade_plan"]["target"] == round(expected_target, 2)


def test_missing_session_date_path_fails_closed_for_decay_model():
    contract = _contract("NO_SESSION_PATH", 3050.0, 0.62)
    contract.pop("session_dates_to_expiry")

    result = score_option_contract(
        contract,
        direction="LONG",
        spot=3050.0,
        expected_move_pct=2.0,
        horizon="1_TO_2D",
        holding_days=2,
        underlying_stop_price=2995.0,
        iv_percentile=45.0,
    )

    assert result["eligible"] is False
    assert "HOLDING_CALENDAR_DECAY_UNAVAILABLE" in result["blockers"]
    assert result["scenarios"] == []
    assert result["trade_plan"]["target"] is None
    assert result["trade_plan"]["stop"] is None


def test_known_extreme_iv_percentile_blocks_long_premium_entry():
    result = score_option_contract(
        _contract("EXTREME_IV", 3050.0, 0.62),
        direction="LONG",
        spot=3050.0,
        expected_move_pct=2.0,
        horizon="1_TO_2D",
        holding_days=2,
        underlying_stop_price=2995.0,
        iv_percentile=90.0,
    )
    assert result["eligible"] is False
    assert "IV_CRUSH_RISK_EXTREME_PERCENTILE" in result["blockers"]


def test_intraday_horizon_uses_fractional_decay_to_eod():
    contract = _contract("INTRADAY", 3050.0, 0.62)
    contract["quote_timestamp"] = "2026-09-22T10:35:00"
    result = score_option_contract(
        contract,
        direction="LONG",
        spot=3050.0,
        expected_move_pct=1.0,
        horizon="INTRADAY",
        holding_days=0,
        underlying_stop_price=2995.0,
        iv_percentile=45.0,
    )
    assert result["eligible"] is True
    assert round(result["calendar_days_to_holding_horizon"], 6) == round(5.0 / 24.0, 6)
    assert result["scenario_decay_basis"] == "FRACTIONAL_CALENDAR_DAYS_TO_1535_IST_EOD"


def test_intraday_candidate_after_eod_cutoff_fails_closed():
    contract = _contract("LATE_INTRADAY", 3050.0, 0.62)
    contract["quote_timestamp"] = "2026-09-22T15:36:00"
    result = score_option_contract(
        contract,
        direction="LONG",
        spot=3050.0,
        expected_move_pct=1.0,
        horizon="INTRADAY",
        holding_days=0,
        underlying_stop_price=2995.0,
        iv_percentile=45.0,
    )
    assert result["eligible"] is False
    assert "HOLDING_CALENDAR_DECAY_UNAVAILABLE" in result["blockers"]
