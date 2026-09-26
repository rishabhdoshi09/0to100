"""Pure NSE F&O directional setup scoring.

This module deliberately does not fetch market data and does not place orders.
It converts point-in-time underlying/futures features into a transparent
0-100 *quality score*. The score is NOT a probability; probability calibration
belongs to historical/forward evidence.
"""
from __future__ import annotations

from typing import Any, Mapping


LONG = "LONG"
SHORT = "SHORT"

LONG_BUILDUP = "LONG_BUILDUP"
SHORT_BUILDUP = "SHORT_BUILDUP"
SHORT_COVERING = "SHORT_COVERING"
LONG_UNWINDING = "LONG_UNWINDING"
OI_NEUTRAL = "NEUTRAL"


def _f(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:
        return default
    return number


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def classify_futures_oi(
    price_change_pct: float,
    oi_change_pct: float,
    *,
    min_price_change_pct: float = 0.15,
    min_oi_change_pct: float = 1.0,
) -> str:
    """Classify the canonical price/OI quadrant, ignoring tiny noisy changes."""
    price = _f(price_change_pct)
    oi = _f(oi_change_pct)
    if abs(price) < min_price_change_pct or abs(oi) < min_oi_change_pct:
        return OI_NEUTRAL
    if price > 0 and oi > 0:
        return LONG_BUILDUP
    if price < 0 and oi > 0:
        return SHORT_BUILDUP
    if price > 0 and oi < 0:
        return SHORT_COVERING
    if price < 0 and oi < 0:
        return LONG_UNWINDING
    return OI_NEUTRAL


def _directional(value: float, direction: str) -> float:
    return value if direction == LONG else -value


def _trend_score(features: Mapping[str, Any], direction: str) -> tuple[float, list[str]]:
    price = _f(features.get("price"))
    vwap = _f(features.get("vwap"))
    ema20 = _f(features.get("ema20"))
    ema50 = _f(features.get("ema50"))
    score = 0.0
    reasons: list[str] = []
    if price <= 0:
        return score, reasons
    if vwap > 0 and _directional(price - vwap, direction) > 0:
        score += 5.0
        reasons.append("VWAP aligned")
    if ema20 > 0 and _directional(price - ema20, direction) > 0:
        score += 5.0
        reasons.append("EMA20 aligned")
    if ema50 > 0 and _directional(price - ema50, direction) > 0:
        score += 4.0
        reasons.append("EMA50 aligned")
    if ema20 > 0 and ema50 > 0 and _directional(ema20 - ema50, direction) > 0:
        score += 4.0
        reasons.append("EMA20/50 trend aligned")
    return score, reasons


def _breakout_score(features: Mapping[str, Any], direction: str) -> tuple[float, float]:
    price = _f(features.get("price"))
    level_key = "breakout_level" if direction == LONG else "breakdown_level"
    level = _f(features.get(level_key))
    if price <= 0 or level <= 0:
        return 0.0, 0.0
    signed = (price - level) / level * 100.0
    move = signed if direction == LONG else -signed
    if move <= 0:
        return 0.0, move
    # Confirmation improves rapidly through the first ~1%, then saturates.
    return 8.0 + 12.0 * _clamp(move / 1.25, 0.0, 1.0), move


def _rsi_score(rsi: float, direction: str) -> float:
    directional_rsi = rsi if direction == LONG else 100.0 - rsi
    if 55.0 <= directional_rsi <= 70.0:
        return 10.0
    if 50.0 <= directional_rsi < 55.0 or 70.0 < directional_rsi <= 75.0:
        return 7.0
    if 45.0 <= directional_rsi < 50.0 or 75.0 < directional_rsi <= 80.0:
        return 3.0
    return 0.0


def _adx_score(adx: float) -> float:
    if adx >= 30.0:
        return 10.0
    if adx >= 25.0:
        return 8.0
    if adx >= 20.0:
        return 5.0
    if adx >= 15.0:
        return 2.0
    return 0.0


def _rvol_score(rvol: float) -> float:
    if rvol >= 2.5:
        return 12.0
    if rvol >= 2.0:
        return 10.0
    if rvol >= 1.5:
        return 8.0
    if rvol >= 1.2:
        return 5.0
    if rvol >= 1.0:
        return 2.0
    return 0.0


def _signed_strength_score(value: float, direction: str, maximum: float) -> float:
    aligned = _directional(value, direction)
    if aligned <= 0:
        return 0.0
    return maximum * _clamp(aligned / 2.0, 0.0, 1.0)


def _oi_score(state: str, direction: str) -> float:
    if direction == LONG:
        return {
            LONG_BUILDUP: 10.0,
            SHORT_COVERING: 6.0,
            OI_NEUTRAL: 2.0,
            LONG_UNWINDING: 0.0,
            SHORT_BUILDUP: 0.0,
        }.get(state, 0.0)
    return {
        SHORT_BUILDUP: 10.0,
        LONG_UNWINDING: 6.0,
        OI_NEUTRAL: 2.0,
        SHORT_COVERING: 0.0,
        LONG_BUILDUP: 0.0,
    }.get(state, 0.0)


def _expected_move(features: Mapping[str, Any], score: float) -> dict[str, Any]:
    atr_pct = max(0.0, _f(features.get("atr_pct")))
    rvol = max(0.0, _f(features.get("rvol")))
    adx = max(0.0, _f(features.get("adx")))
    base = atr_pct if atr_pct > 0 else 1.0
    expansion = 1.0 + 0.12 * _clamp(rvol - 1.0, 0.0, 2.5) + 0.006 * _clamp(adx - 20.0, 0.0, 30.0)
    lower = _clamp(base * 0.75, 0.35, 4.0)
    upper = _clamp(base * expansion * (1.25 if score >= 80 else 1.10), lower, 6.0)

    if score >= 80 and rvol >= 2.0 and adx >= 25.0:
        horizon = "INTRADAY_TO_1D"
        holding_days = 1
    elif score >= 70:
        horizon = "1_TO_2D"
        holding_days = 2
    else:
        horizon = "2_TO_4D"
        holding_days = 4
    return {
        "lower_pct": round(lower, 2),
        "upper_pct": round(upper, 2),
        "mid_pct": round((lower + upper) / 2.0, 2),
        "horizon": horizon,
        "holding_days": holding_days,
        "model": "ATR_RVOL_ADX_HEURISTIC_UNCALIBRATED",
    }


def score_fo_setup(
    features: Mapping[str, Any],
    direction: str,
    *,
    minimum_score: float = 65.0,
    max_chase_atr: float = 1.5,
) -> dict[str, Any]:
    """Return a fail-closed directional F&O setup assessment."""
    direction = str(direction or "").upper()
    if direction not in {LONG, SHORT}:
        raise ValueError("direction must be LONG or SHORT")

    blockers: list[str] = []
    reasons: list[str] = []
    price = _f(features.get("price"))
    if not bool(features.get("is_fo", False)):
        blockers.append("NOT_FO_UNIVERSE")
    if price <= 0:
        blockers.append("INVALID_PRICE")

    # This lane is intentionally stricter than the ordinary equity scanner:
    # a missing confirmation must never be silently compensated by points from
    # other features. Zero is a valid market change for strength/OI fields, so
    # availability is checked separately from magnitude.
    required_present = (
        "vwap", "ema20", "ema50", "rsi", "adx", "rvol",
        "relative_strength_pct", "sector_strength_pct", "nifty_change_pct",
        "futures_price_change_pct", "futures_oi_change_pct",
    )
    for key in required_present:
        if key not in features or features.get(key) is None:
            blockers.append(f"MISSING_{key.upper()}")
    if _f(features.get("vwap")) <= 0:
        blockers.append("INVALID_VWAP")
    if _f(features.get("ema20")) <= 0 or _f(features.get("ema50")) <= 0:
        blockers.append("INVALID_EMA")
    if not (0 < _f(features.get("rsi")) < 100):
        blockers.append("INVALID_RSI")
    if _f(features.get("adx")) <= 0:
        blockers.append("INVALID_ADX")
    if _f(features.get("rvol")) <= 0:
        blockers.append("INVALID_RVOL")

    if bool(features.get("false_breakout", False)):
        blockers.append("FALSE_BREAKOUT_FILTER")
    chase_atr = _f(features.get("chase_distance_atr"))
    if chase_atr > max_chase_atr:
        blockers.append("EXTENDED_CHASE")
    spread_bps = _f(features.get("underlying_spread_bps"))
    if spread_bps > 30.0:
        blockers.append("UNDERLYING_SPREAD_TOO_WIDE")
    turnover = _f(features.get("avg_turnover_crore"))
    if 0 < turnover < 5.0:
        blockers.append("UNDERLYING_LIQUIDITY_TOO_LOW")

    breakout_score, breakout_pct = _breakout_score(features, direction)
    if breakout_score <= 0:
        blockers.append("BREAKOUT_NOT_CONFIRMED" if direction == LONG else "BREAKDOWN_NOT_CONFIRMED")
    else:
        reasons.append(f"{direction.lower()} level cleared by {breakout_pct:.2f}%")

    trend_score, trend_reasons = _trend_score(features, direction)
    reasons.extend(trend_reasons)

    rvol = _f(features.get("rvol"))
    rsi = _f(features.get("rsi"))
    adx = _f(features.get("adx"))
    rel = _f(features.get("relative_strength_pct"))
    sector = _f(features.get("sector_strength_pct"))
    nifty = _f(features.get("nifty_change_pct"))

    if direction == LONG and rsi >= 82.0:
        blockers.append("RSI_HARD_OVERBOUGHT")
    if direction == SHORT and rsi > 0 and rsi <= 18.0:
        blockers.append("RSI_HARD_OVERSOLD")

    oi_state = classify_futures_oi(
        _f(features.get("futures_price_change_pct")),
        _f(features.get("futures_oi_change_pct")),
    )

    components = {
        "breakout": breakout_score,                   # 20
        "trend": trend_score,                         # 18
        "rvol": _rvol_score(rvol),                    # 12
        "rsi": _rsi_score(rsi, direction),            # 10
        "adx": _adx_score(adx),                       # 10
        "relative_strength": _signed_strength_score(rel, direction, 8.0),
        "sector_strength": _signed_strength_score(sector, direction, 7.0),
        "nifty_alignment": _signed_strength_score(nifty, direction, 5.0),
        "futures_oi": _oi_score(oi_state, direction), # 10
    }
    score = round(_clamp(sum(components.values()), 0.0, 100.0), 1)

    if rvol >= 1.5:
        reasons.append(f"RVOL {rvol:.2f}x")
    if adx >= 20:
        reasons.append(f"ADX {adx:.1f}")
    reasons.append(f"futures OI: {oi_state}")

    if score < minimum_score:
        blockers.append("SETUP_SCORE_BELOW_THRESHOLD")

    tradable = not blockers
    expected = _expected_move(features, score)
    return {
        "direction": direction,
        "score": score,
        "score_is_probability": False,
        "probability": None,
        "probability_status": "UNCALIBRATED_REQUIRES_POINT_IN_TIME_EVIDENCE",
        "components": {key: round(value, 2) for key, value in components.items()},
        "futures_oi_state": oi_state,
        "breakout_distance_pct": round(breakout_pct, 3),
        "expected_move": expected,
        "tradable": tradable,
        "blockers": blockers,
        "reasons": reasons,
        "paper_only": True,
        "live_execution_allowed": False,
    }
