from product.fo_setup import (
    LONG_BUILDUP,
    LONG_UNWINDING,
    OI_NEUTRAL,
    SHORT_BUILDUP,
    SHORT_COVERING,
    classify_futures_oi,
    score_fo_setup,
)


def _features():
    return {
        "symbol": "RELIANCE",
        "is_fo": True,
        "price": 3050.0,
        "breakout_level": 3025.0,
        "breakdown_level": 2940.0,
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


def test_classifies_all_price_oi_quadrants_and_noise():
    assert classify_futures_oi(1.0, 5.0) == LONG_BUILDUP
    assert classify_futures_oi(-1.0, 5.0) == SHORT_BUILDUP
    assert classify_futures_oi(1.0, -5.0) == SHORT_COVERING
    assert classify_futures_oi(-1.0, -5.0) == LONG_UNWINDING
    assert classify_futures_oi(0.05, 5.0) == OI_NEUTRAL
    assert classify_futures_oi(1.0, 0.2) == OI_NEUTRAL


def test_strong_long_setup_passes_without_claiming_probability():
    result = score_fo_setup(_features(), "LONG")
    assert result["tradable"] is True
    assert result["score"] >= 65
    assert result["score_is_probability"] is False
    assert result["probability"] is None
    assert result["futures_oi_state"] == LONG_BUILDUP
    assert result["expected_move"]["holding_days"] in {1, 2, 4}
    assert result["paper_only"] is True
    assert result["live_execution_allowed"] is False


def test_false_breakout_and_extension_fail_closed():
    features = _features()
    features["false_breakout"] = True
    features["chase_distance_atr"] = 2.2
    result = score_fo_setup(features, "LONG")
    assert result["tradable"] is False
    assert "FALSE_BREAKOUT_FILTER" in result["blockers"]
    assert "EXTENDED_CHASE" in result["blockers"]


def test_short_requires_confirmed_breakdown_and_directional_alignment():
    features = _features()
    features.update({
        "price": 2910.0,
        "vwap": 2950.0,
        "ema20": 2960.0,
        "ema50": 3000.0,
        "rsi": 36.0,
        "relative_strength_pct": -1.7,
        "sector_strength_pct": -1.2,
        "nifty_change_pct": -0.7,
        "futures_price_change_pct": -1.1,
        "futures_oi_change_pct": 8.0,
    })
    result = score_fo_setup(features, "SHORT")
    assert result["tradable"] is True
    assert result["futures_oi_state"] == SHORT_BUILDUP


def test_non_fo_symbol_is_rejected_even_if_technical_score_is_high():
    features = _features()
    features["is_fo"] = False
    result = score_fo_setup(features, "LONG")
    assert result["tradable"] is False
    assert "NOT_FO_UNIVERSE" in result["blockers"]
