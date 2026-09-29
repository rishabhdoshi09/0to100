from options.directional_selector import black_scholes
from product.fo_options_pipeline import evaluate_fo_opportunity


def _setup():
    return {
        "symbol": "RELIANCE",
        "is_fo": True,
        "price": 3050.0,
        "breakout_level": 3025.0,
        "vwap": 3010.0,
        "ema20": 2990.0,
        "ema50": 2920.0,
        "rsi": 64.0,
        "adx": 31.0,
        "rvol": 2.3,
        "relative_strength_pct": 1.8,
        "sector_strength_pct": 1.4,
        "nifty_change_pct": 0.8,
        "futures_price_change_pct": 1.2,
        "futures_oi_change_pct": 7.5,
        "atr_pct": 1.7,
        "avg_turnover_crore": 250.0,
        "underlying_spread_bps": 3.0,
        "chase_distance_atr": 0.6,
        "false_breakout": False,
    }


def _options():
    fair = black_scholes(
        spot=3050.0, strike=3050.0, dte=9, iv=0.24, option_type="CE",
    )["price"]
    return [{
        "symbol": "RELIANCE27SEP3050CE",
        "option_type": "CE",
        "strike": 3050.0,
        "expiry": "2026-10-01",
        "dte": 9,
        "as_of_date": "2026-09-22",
        "quote_timestamp": "2026-09-22T10:30:00",
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
        "volume": 5000,
        "oi": 30000,
        "iv": 24.0,
        "delta": 0.62,
    }]


def test_pipeline_only_selects_option_after_underlying_gate():
    result = evaluate_fo_opportunity(
        underlying_features=_setup(),
        direction="LONG",
        option_contracts=_options(),
        iv_percentile=40.0,
    )
    assert result["decision"] == "PAPER_OPTION_CANDIDATE"
    assert result["selected_contract"]["option_type"] == "CE"
    plan = result["selected_contract"]["trade_plan"]
    assert plan["entry"] > plan["stop"]
    assert plan["target"] > plan["entry"]
    assert result["paper_only"] is True
    assert result["live_execution_allowed"] is False


def test_pipeline_does_not_score_options_when_underlying_is_invalid():
    setup = _setup()
    setup["false_breakout"] = True
    result = evaluate_fo_opportunity(
        underlying_features=setup,
        direction="LONG",
        option_contracts=_options(),
    )
    assert result["decision"] == "NO_TRADE"
    assert result["options"]["skipped"] is True


def test_valid_stock_with_bad_option_is_not_forced_into_trade():
    bad = _options()
    bad[0]["bid"] = 0
    bad[0]["ask"] = 0
    result = evaluate_fo_opportunity(
        underlying_features=_setup(),
        direction="LONG",
        option_contracts=bad,
    )
    assert result["decision"] == "NO_OPTION_TRADE"
    assert "selected_contract" not in result


def test_pipeline_carries_explicit_intraday_eod_horizon_into_option_decay():
    setup = _setup()
    setup.update({
        "rvol": 3.5,
        "adx": 38.0,
        "relative_strength_pct": 3.0,
        "sector_strength_pct": 2.5,
        "nifty_change_pct": 1.2,
        "futures_price_change_pct": 2.0,
        "futures_oi_change_pct": 10.0,
        "price": 3075.0,
        "breakout_level": 3025.0,
    })
    options = _options()
    # Reprice the fixture consistently with the stronger current spot.
    fair = black_scholes(
        spot=3075.0, strike=3050.0, dte=9, iv=0.24, option_type="CE",
    )["price"]
    options[0].update({
        "ltp": fair,
        "bid": max(0.05, fair - 0.25),
        "ask": fair + 0.25,
        "delta": 0.62,
    })

    result = evaluate_fo_opportunity(
        underlying_features=setup,
        direction="LONG",
        option_contracts=options,
        iv_percentile=45.0,
    )
    assert result["decision"] == "PAPER_OPTION_CANDIDATE"
    assert result["setup"]["expected_move"]["horizon"] == "INTRADAY"
    assert result["setup"]["expected_move"]["holding_days"] == 0
    assert result["setup"]["expected_move"]["exit_policy"] == "EOD"
    selected = result["selected_contract"]
    assert selected["scenario_decay_basis"] == "FRACTIONAL_CALENDAR_DAYS_TO_1530_IST_EOD"
    assert 0 < selected["calendar_days_to_holding_horizon"] < 1
