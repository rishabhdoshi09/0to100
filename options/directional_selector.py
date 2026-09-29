"""Directional option-contract selection for the NSE F&O paper lane.

The engine is pure and fail-closed. It never places broker orders. Scenario
returns are Black-Scholes constant-IV estimates, not promises or calibrated
probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import erf, exp, log, pi, sqrt
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo


CE = "CE"
PE = "PE"
IST = ZoneInfo("Asia/Kolkata")


def _today_ist() -> date:
    """Current exchange-local date for expiry fallbacks."""
    return datetime.now(IST).date()


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


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _iv_decimal(value: Any) -> float:
    iv = _f(value)
    if iv > 3.0:
        iv /= 100.0
    return iv


def _dte(contract: Mapping[str, Any], *, as_of: date | None = None) -> int:
    direct = contract.get("dte")
    if direct is not None:
        return max(0, int(_f(direct)))
    raw = str(contract.get("expiry") or "").strip()
    if not raw:
        return 0
    try:
        expiry = datetime.fromisoformat(raw).date()
    except ValueError:
        for fmt in ("%d-%b-%Y", "%d-%b-%y", "%Y-%m-%d"):
            try:
                expiry = datetime.strptime(raw, fmt).date()
                break
            except ValueError:
                expiry = None  # type: ignore[assignment]
        if expiry is None:
            return 0
    return max(0, (expiry - (as_of or _today_ist())).days)


def black_scholes(
    *,
    spot: float,
    strike: float,
    dte: float,
    iv: float,
    option_type: str,
    rate: float = 0.065,
) -> dict[str, float]:
    """Black-Scholes price and standard Greeks for one European-style option."""
    spot = _f(spot)
    strike = _f(strike)
    years = max(_f(dte), 0.0) / 365.0
    sigma = _iv_decimal(iv)
    kind = str(option_type or "").upper()
    if spot <= 0 or strike <= 0 or years <= 0 or sigma <= 0 or kind not in {CE, PE}:
        return {"price": 0.0, "delta": 0.0, "gamma": 0.0, "theta_per_day": 0.0, "vega_per_vol_point": 0.0}

    root_t = sqrt(years)
    d1 = (log(spot / strike) + (rate + 0.5 * sigma * sigma) * years) / (sigma * root_t)
    d2 = d1 - sigma * root_t
    discount = exp(-rate * years)

    if kind == CE:
        price = spot * _norm_cdf(d1) - strike * discount * _norm_cdf(d2)
        delta = _norm_cdf(d1)
        theta_year = (
            -(spot * _norm_pdf(d1) * sigma) / (2.0 * root_t)
            - rate * strike * discount * _norm_cdf(d2)
        )
    else:
        price = strike * discount * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
        delta = _norm_cdf(d1) - 1.0
        theta_year = (
            -(spot * _norm_pdf(d1) * sigma) / (2.0 * root_t)
            + rate * strike * discount * _norm_cdf(-d2)
        )
    gamma = _norm_pdf(d1) / (spot * sigma * root_t)
    vega_per_decimal = spot * _norm_pdf(d1) * root_t
    return {
        "price": round(max(price, 0.0), 4),
        "delta": round(delta, 6),
        "gamma": round(gamma, 8),
        "theta_per_day": round(theta_year / 365.0, 6),
        "vega_per_vol_point": round(vega_per_decimal / 100.0, 6),
    }


def implied_volatility(
    *,
    market_price: float,
    spot: float,
    strike: float,
    dte: float,
    option_type: str,
    rate: float = 0.065,
    lower: float = 0.01,
    upper: float = 5.0,
    iterations: int = 80,
) -> float:
    """Infer annualized IV with bounded bisection.

    Returns 0 when the quote violates intrinsic bounds or cannot be bracketed.
    The solver is deterministic and dependency-free, suitable for live quote
    normalization and historical replay.
    """
    market = _f(market_price)
    spot = _f(spot)
    strike = _f(strike)
    kind = str(option_type or "").upper()
    if market <= 0 or spot <= 0 or strike <= 0 or dte <= 0 or kind not in {CE, PE}:
        return 0.0
    intrinsic = max(spot - strike, 0.0) if kind == CE else max(strike - spot, 0.0)
    if market + 1e-9 < intrinsic:
        return 0.0

    lo = max(1e-6, float(lower))
    # black_scholes accepts either decimal IV (0.24) or percent-style IV (24).
    # Keep the solver's internal bracket unambiguously decimal. Values above
    # 3.0 would be reinterpreted by _iv_decimal as percent-style input, so cap
    # the fail-closed inversion range at 300% annualized volatility.
    hi = min(3.0, max(lo * 2.0, float(upper)))
    low_price = black_scholes(
        spot=spot, strike=strike, dte=dte, iv=lo,
        option_type=kind, rate=rate,
    )["price"]
    high_price = black_scholes(
        spot=spot, strike=strike, dte=dte, iv=hi,
        option_type=kind, rate=rate,
    )["price"]
    if market < low_price - 1e-6 or market > high_price + 1e-6:
        return 0.0

    for _ in range(max(1, int(iterations))):
        mid = (lo + hi) / 2.0
        value = black_scholes(
            spot=spot, strike=strike, dte=dte, iv=mid,
            option_type=kind, rate=rate,
        )["price"]
        if value < market:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2.0, 6)


@dataclass(frozen=True)
class OptionSelectionPolicy:
    min_volume: int = 100
    min_oi: int = 500
    max_spread_pct: float = 4.0
    min_premium: float = 2.0
    minimum_score: float = 60.0
    min_delta_abs: float = 0.35
    max_delta_abs: float = 0.85
    max_iv_percentile: float = 85.0


def _spread_pct(contract: Mapping[str, Any]) -> float | None:
    bid = _f(contract.get("bid"))
    ask = _f(contract.get("ask"))
    if bid <= 0 or ask <= 0 or ask < bid:
        return None
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid * 100.0 if mid > 0 else None


def _calendar_days_for_session_horizon(
    contract: Mapping[str, Any],
    sessions: int,
) -> int | None:
    """Exact elapsed calendar days from as-of to the Nth future NSE session."""
    needed = max(0, int(sessions))
    if needed == 0:
        return 0
    raw_as_of = str(contract.get("as_of_date") or "").strip()
    raw_sessions = contract.get("session_dates_to_expiry")
    if not raw_as_of or not isinstance(raw_sessions, Sequence) or isinstance(raw_sessions, (str, bytes)):
        return None
    try:
        as_of = date.fromisoformat(raw_as_of)
    except ValueError:
        return None

    parsed: list[date] = []
    previous = as_of
    for raw in raw_sessions:
        try:
            current = date.fromisoformat(str(raw))
        except ValueError:
            return None
        if current <= previous:
            return None
        parsed.append(current)
        previous = current

    if len(parsed) < needed:
        return None
    elapsed = (parsed[needed - 1] - as_of).days
    return elapsed if elapsed > 0 else None


def _intraday_calendar_days_to_eod(contract: Mapping[str, Any]) -> float | None:
    """Fractional calendar days from observed quote time to the 15:30 IST EOD exit."""
    raw = str(contract.get("quote_timestamp") or "").strip()
    if not raw:
        return None
    try:
        stamp = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone(IST).replace(tzinfo=None)
    eod = stamp.replace(hour=15, minute=30, second=0, microsecond=0)
    if eod <= stamp:
        return None
    return max((eod - stamp).total_seconds() / 86400.0, 1.0 / 1440.0)


def _holding_decay_days(
    contract: Mapping[str, Any],
    *,
    holding_days: int,
    horizon: str,
) -> float | None:
    if str(horizon or "").upper() == "INTRADAY":
        return _intraday_calendar_days_to_eod(contract)
    sessions = _calendar_days_for_session_horizon(contract, holding_days)
    return float(sessions) if sessions is not None else None


def _moneyness(
    contract: Mapping[str, Any],
    *,
    spot: float,
    atm_strike: float | None = None,
) -> tuple[str, float | None]:
    strike = _f(contract.get("strike"))
    if strike <= 0 or spot <= 0:
        return "UNKNOWN", None
    kind = str(contract.get("option_type") or "").upper()
    signed = (spot - strike) / spot * 100.0
    if kind == PE:
        signed = -signed
    if atm_strike is not None and abs(strike - float(atm_strike)) < 1e-9:
        label = "ATM"
    elif atm_strike is None and abs(signed) < 1e-9:
        label = "ATM"
    elif signed > 0:
        label = "ITM"
    else:
        label = "OTM"
    return label, signed


def _reference_premium(contract: Mapping[str, Any]) -> float:
    bid = _f(contract.get("bid"))
    ask = _f(contract.get("ask"))
    if bid > 0 and ask >= bid:
        return (bid + ask) / 2.0
    return _f(contract.get("ltp") or contract.get("last_price"))


def _entry_premium(contract: Mapping[str, Any]) -> float:
    """Executable long-premium reference: ask when available, else last price."""
    ask = _f(contract.get("ask"))
    if ask > 0:
        return ask
    return _f(contract.get("ltp") or contract.get("last_price"))


def _delta_score(delta_abs: float) -> float:
    if 0.55 <= delta_abs <= 0.70:
        return 25.0
    if 0.45 <= delta_abs < 0.55:
        return 15.0 + (delta_abs - 0.45) / 0.10 * 10.0
    if 0.70 < delta_abs <= 0.80:
        return 25.0 - (delta_abs - 0.70) / 0.10 * 10.0
    if 0.35 <= delta_abs < 0.45:
        return 5.0 + (delta_abs - 0.35) / 0.10 * 10.0
    if 0.80 < delta_abs <= 0.85:
        return 15.0 - (delta_abs - 0.80) / 0.05 * 10.0
    return 0.0


def _expiry_score(trading_sessions_to_expiry: int, horizon: str) -> float:
    """Prefer near/next-week expiries using market sessions, never calendar DTE."""
    preferred = {
        "INTRADAY": (1, 5),
        "INTRADAY_TO_1D": (2, 6),
        "1D": (2, 6),
        "1_TO_2D": (3, 8),
        "2_TO_4D": (5, 12),
    }.get(horizon, (3, 12))
    lo, hi = preferred
    sessions = int(trading_sessions_to_expiry)
    if lo <= sessions <= hi:
        return 10.0
    if max(1, lo - 1) <= sessions < lo or hi < sessions <= hi + 5:
        return 6.0
    if sessions >= 1:
        return 3.0
    return 0.0


def _liquidity_score(volume: float, oi: float, spread_pct: float | None, policy: OptionSelectionPolicy) -> float:
    if spread_pct is None:
        return 0.0
    volume_score = 8.0 * _clamp(volume / max(policy.min_volume * 5.0, 1.0), 0.0, 1.0)
    oi_score = 8.0 * _clamp(oi / max(policy.min_oi * 5.0, 1.0), 0.0, 1.0)
    spread_score = 9.0 * _clamp((policy.max_spread_pct - spread_pct) / max(policy.max_spread_pct, 0.01), 0.0, 1.0)
    return volume_score + oi_score + spread_score


def _theta_score(theta_per_day: float, premium: float) -> float:
    if premium <= 0:
        return 0.0
    decay_pct = abs(theta_per_day) / premium * 100.0
    if decay_pct <= 1.0:
        return 12.0
    if decay_pct >= 5.0:
        return 0.0
    return 12.0 * (5.0 - decay_pct) / 4.0


def _iv_score(iv_percentile: float | None) -> float:
    # Missing point-in-time IV percentile is unknown evidence, not a neutral
    # positive. Keep the contract eligible on its observable market qualities,
    # but award no IV-regime points until a trustworthy history exists.
    if iv_percentile is None:
        return 0.0
    pct = _clamp(iv_percentile, 0.0, 100.0)
    if pct <= 50:
        return 12.0
    if pct <= 70:
        return 12.0 - (pct - 50.0) / 20.0 * 4.0
    if pct <= 85:
        return 8.0 - (pct - 70.0) / 15.0 * 5.0
    return 1.0


def scenario_reprice(
    contract: Mapping[str, Any],
    *,
    spot: float,
    holding_days: int,
    calendar_holding_days: float | None = None,
    horizon: str = "",
    moves_pct: Sequence[float] = (-2.0, -1.0, 1.0, 2.0, 3.0, 4.0),
    rate: float = 0.065,
) -> list[dict[str, float]]:
    premium = _entry_premium(contract)
    strike = _f(contract.get("strike"))
    iv = _iv_decimal(contract.get("iv"))
    dte = _dte(contract)
    kind = str(contract.get("option_type") or "").upper()
    resolved_calendar_days = (
        max(0.0, float(calendar_holding_days))
        if calendar_holding_days is not None
        else _holding_decay_days(contract, holding_days=holding_days, horizon=horizon)
    )
    rows: list[dict[str, float]] = []
    if (
        resolved_calendar_days is None
        or premium <= 0
        or strike <= 0
        or spot <= 0
        or iv <= 0
        or kind not in {CE, PE}
    ):
        return rows
    remaining = max(0.25, float(dte - resolved_calendar_days))
    for move in moves_pct:
        new_spot = spot * (1.0 + float(move) / 100.0)
        model = black_scholes(
            spot=new_spot,
            strike=strike,
            dte=remaining,
            iv=iv,
            option_type=kind,
            rate=rate,
        )
        projected = model["price"]
        ret = (projected - premium) / premium * 100.0
        rows.append({
            "underlying_move_pct": round(float(move), 2),
            "projected_option_price": round(projected, 2),
            "projected_option_return_pct": round(ret, 2),
        })
    return rows


def score_option_contract(
    contract: Mapping[str, Any],
    *,
    direction: str,
    spot: float,
    expected_move_pct: float,
    horizon: str,
    holding_days: int,
    underlying_stop_price: float | None = None,
    iv_percentile: float | None = None,
    atm_strike: float | None = None,
    policy: OptionSelectionPolicy | None = None,
    rate: float = 0.065,
) -> dict[str, Any]:
    policy = policy or OptionSelectionPolicy()
    kind = str(contract.get("option_type") or "").upper()
    desired = CE if str(direction).upper() == "LONG" else PE
    blockers: list[str] = []
    if kind != desired:
        blockers.append("WRONG_OPTION_SIDE")

    premium = _entry_premium(contract)
    strike = _f(contract.get("strike"))
    iv = _iv_decimal(contract.get("iv"))
    volume = _f(contract.get("volume"))
    oi = _f(contract.get("oi"))
    dte = _dte(contract)
    raw_trading_sessions = contract.get("trading_sessions_to_expiry")
    trading_sessions_to_expiry = (
        int(_f(raw_trading_sessions, -1.0))
        if raw_trading_sessions is not None else -1
    )
    spread_pct = _spread_pct(contract)

    if premium < policy.min_premium:
        blockers.append("PREMIUM_TOO_LOW")
    if strike <= 0 or spot <= 0:
        blockers.append("INVALID_STRIKE_OR_SPOT")
    if iv <= 0:
        blockers.append("IV_UNAVAILABLE")
    if iv_percentile is not None and float(iv_percentile) > policy.max_iv_percentile:
        blockers.append("IV_CRUSH_RISK_EXTREME_PERCENTILE")
    if volume < policy.min_volume:
        blockers.append("OPTION_VOLUME_TOO_LOW")
    if oi < policy.min_oi:
        blockers.append("OPTION_OI_TOO_LOW")
    if spread_pct is None:
        blockers.append("BID_ASK_UNAVAILABLE")
    elif spread_pct > policy.max_spread_pct:
        blockers.append("BID_ASK_TOO_WIDE")
    if dte < 2:
        blockers.append("EXPIRY_TOO_CLOSE")
    # Holding horizons are trading-session oriented; calendar DTE is retained
    # for Black-Scholes/time decay only and must not stand in for market sessions.
    holiday_calendar_loaded = int(_f(contract.get("holiday_calendar_loaded")))
    if trading_sessions_to_expiry < 0:
        blockers.append("TRADING_SESSION_EXPIRY_UNAVAILABLE")
    elif holiday_calendar_loaded <= 0:
        blockers.append("TRADING_CALENDAR_HOLIDAYS_UNAVAILABLE")
    elif trading_sessions_to_expiry < max(1, int(holding_days) + 1):
        blockers.append("TRADING_SESSIONS_SHORTER_THAN_HOLDING_HORIZON")

    calendar_holding_days = _holding_decay_days(
        contract,
        holding_days=holding_days,
        horizon=horizon,
    )
    first_session_calendar_days = (
        calendar_holding_days
        if str(horizon or "").upper() == "INTRADAY"
        else _holding_decay_days(contract, holding_days=1, horizon="1D")
    )
    if calendar_holding_days is None or first_session_calendar_days is None:
        blockers.append("HOLDING_CALENDAR_DECAY_UNAVAILABLE")

    greeks = black_scholes(
        spot=spot,
        strike=strike,
        dte=max(dte, 0.25),
        iv=iv,
        option_type=kind,
        rate=rate,
    )
    # Prefer observed/externally computed Greeks when supplied, otherwise use BS.
    delta = _f(contract.get("delta"), greeks["delta"])
    gamma = _f(contract.get("gamma"), greeks["gamma"])
    theta = _f(contract.get("theta_per_day"), greeks["theta_per_day"])
    vega = _f(contract.get("vega_per_vol_point"), greeks["vega_per_vol_point"])
    delta_abs = abs(delta)
    if not (policy.min_delta_abs <= delta_abs <= policy.max_delta_abs):
        blockers.append("DELTA_OUTSIDE_DIRECTIONAL_BAND")
    theta_decay_pct = (
        abs(theta) / premium * 100.0
        if premium > 0 else float("inf")
    )
    scenarios = scenario_reprice(
        contract,
        spot=spot,
        holding_days=holding_days,
        calendar_holding_days=calendar_holding_days,
        horizon=horizon,
        rate=rate,
    )
    aligned_move = abs(expected_move_pct) if desired == CE else -abs(expected_move_pct)
    target_model_price = 0.0
    if calendar_holding_days is not None:
        target_model = black_scholes(
            spot=spot * (1.0 + aligned_move / 100.0),
            strike=strike,
            dte=max(
                0.25,
                float(dte - max(0.0, float(calendar_holding_days))),
            ),
            iv=iv,
            option_type=kind,
            rate=rate,
        )
        target_model_price = float(target_model["price"])
    expected_return = (
        (target_model_price - premium) / premium * 100.0
        if premium > 0 and target_model_price > 0
        else 0.0
    )

    stop_model_price = 0.0
    stop_underlying = _f(underlying_stop_price)
    if (
        first_session_calendar_days is not None
        and stop_underlying > 0
        and strike > 0
        and iv > 0
        and dte > 0
    ):
        stop_model = black_scholes(
            spot=stop_underlying,
            strike=strike,
            dte=max(
                0.25,
                float(dte - max(0.0, float(first_session_calendar_days))),
            ),
            iv=iv,
            option_type=kind,
            rate=rate,
        )
        stop_model_price = float(stop_model["price"])

    option_entry = premium
    option_stop = min(option_entry * 0.98, stop_model_price) if stop_model_price > 0 else 0.0
    option_target = target_model_price
    if option_stop <= 0 or option_stop >= option_entry:
        blockers.append("OPTION_STOP_MODEL_UNAVAILABLE")
    if option_target <= option_entry:
        blockers.append("UNFAVORABLE_OPTION_TARGET")
    option_risk = option_entry - option_stop if option_stop > 0 else 0.0
    option_reward = option_target - option_entry
    risk_reward = option_reward / option_risk if option_risk > 0 and option_reward > 0 else 0.0
    trade_plan = {
        "entry": round(option_entry, 2),
        "stop": round(option_stop, 2) if option_stop > 0 else None,
        "target": round(option_target, 2) if option_target > 0 else None,
        "risk_reward": round(risk_reward, 3),
        "underlying_invalidation": round(stop_underlying, 2) if stop_underlying > 0 else None,
        "model": "CONSTANT_IV_UNDERLYING_INVALIDATION_AND_EXPECTED_MOVE",
    }

    components = {
        "delta_fit": _delta_score(delta_abs),                     # 25
        "liquidity": _liquidity_score(volume, oi, spread_pct, policy),  # 25
        "theta": _theta_score(theta, premium),                    # 12
        "iv": _iv_score(iv_percentile),                           # 12
        "expiry_fit": _expiry_score(trading_sessions_to_expiry, horizon),  # 10
        "expected_payoff": 16.0 * _clamp(expected_return / 25.0, 0.0, 1.0),  # 16
    }
    score = round(sum(components.values()), 1)
    if score < policy.minimum_score:
        blockers.append("OPTION_SCORE_BELOW_THRESHOLD")

    moneyness, moneyness_pct = _moneyness(
        contract,
        spot=spot,
        atm_strike=atm_strike,
    )
    gamma_delta_change_for_1pct_move = abs(gamma) * spot * 0.01
    vega_pct_of_premium_per_vol_point = (
        abs(vega) / premium * 100.0 if premium > 0 else 0.0
    )

    return {
        "symbol": str(contract.get("symbol") or contract.get("tradingsymbol") or ""),
        "instrument_token": contract.get("instrument_token"),
        "option_type": kind,
        "strike": strike,
        "lot_size": int(_f(contract.get("lot_size"))),
        "tick_size": _f(contract.get("tick_size")),
        "source": str(contract.get("source") or ""),
        "expiry": contract.get("expiry"),
        "dte": dte,
        "trading_sessions_to_expiry": (
            trading_sessions_to_expiry if trading_sessions_to_expiry >= 0 else None
        ),
        "holiday_calendar_loaded": holiday_calendar_loaded,
        "expiry_session_model": str(contract.get("expiry_session_model") or ""),
        "expiry_fit_basis": "TRADING_SESSIONS_TO_EXPIRY",
        "holding_sessions": max(0, int(holding_days)),
        "calendar_days_to_holding_horizon": calendar_holding_days,
        "calendar_days_to_first_session": first_session_calendar_days,
        "quote_timestamp": str(contract.get("quote_timestamp") or ""),
        "moneyness": moneyness,
        "moneyness_pct": round(moneyness_pct, 4) if moneyness_pct is not None else None,
        "atm_reference_strike": round(float(atm_strike), 4) if atm_strike is not None else None,
        "premium": round(premium, 2),
        "bid": _f(contract.get("bid")),
        "ask": _f(contract.get("ask")),
        "spread_pct": round(spread_pct, 3) if spread_pct is not None else None,
        "volume": int(volume),
        "oi": int(oi),
        "iv": round(iv * 100.0, 2),
        "iv_percentile": round(float(iv_percentile), 2) if iv_percentile is not None else None,
        "iv_percentile_available": iv_percentile is not None,
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta_per_day": round(theta, 4),
        "theta_decay_pct_of_premium_per_day": round(theta_decay_pct, 4),
        "vega_per_vol_point": round(vega, 4),
        "vega_pct_of_premium_per_vol_point": round(vega_pct_of_premium_per_vol_point, 4),
        "gamma_delta_change_for_1pct_move": round(gamma_delta_change_for_1pct_move, 6),
        "score": score,
        "score_is_probability": False,
        "components": {key: round(value, 2) for key, value in components.items()},
        "expected_move_pct": round(abs(expected_move_pct), 2),
        "projected_return_at_expected_move_pct": round(expected_return, 2),
        "trade_plan": trade_plan,
        "scenarios": scenarios,
        "scenario_model": "BLACK_SCHOLES_CONSTANT_IV_ESTIMATE",
        "scenario_decay_basis": (
            "FRACTIONAL_CALENDAR_DAYS_TO_1530_IST_EOD"
            if str(horizon or "").upper() == "INTRADAY"
            else "CALENDAR_DAYS_TO_NSE_SESSION_HORIZON"
        ),
        "eligible": not blockers,
        "blockers": blockers,
        "paper_only": True,
        "live_execution_allowed": False,
    }


def select_option_contracts(
    contracts: Sequence[Mapping[str, Any]],
    *,
    direction: str,
    spot: float,
    expected_move_pct: float,
    horizon: str,
    holding_days: int,
    underlying_stop_price: float | None = None,
    iv_percentile: float | None = None,
    limit: int = 5,
    policy: OptionSelectionPolicy | None = None,
) -> dict[str, Any]:
    strikes_by_expiry: dict[str, set[float]] = {}
    for contract in contracts:
        strike = _f(contract.get("strike"))
        if strike <= 0:
            continue
        expiry_key = str(contract.get("expiry") or "")
        strikes_by_expiry.setdefault(expiry_key, set()).add(strike)
    atm_by_expiry = {
        expiry_key: min(strikes, key=lambda strike: (abs(strike - spot), strike))
        for expiry_key, strikes in strikes_by_expiry.items()
        if strikes and spot > 0
    }
    rows = [
        score_option_contract(
            contract,
            direction=direction,
            spot=spot,
            expected_move_pct=expected_move_pct,
            horizon=horizon,
            holding_days=holding_days,
            underlying_stop_price=underlying_stop_price,
            iv_percentile=iv_percentile,
            atm_strike=atm_by_expiry.get(str(contract.get("expiry") or "")),
            policy=policy,
        )
        for contract in contracts
    ]
    eligible = sorted(
        (row for row in rows if row["eligible"]),
        key=lambda row: (
            float(row["score"]),
            float(row["projected_return_at_expected_move_pct"]),
            float(row.get("gamma_delta_change_for_1pct_move") or 0.0),
            -float(row.get("vega_pct_of_premium_per_vol_point") or 999.0),
            -float(row.get("spread_pct") or 999.0),
        ),
        reverse=True,
    )
    return {
        "direction": str(direction).upper(),
        "spot": round(_f(spot), 2),
        "eligible_count": len(eligible),
        "best_contracts": eligible[: max(1, int(limit))],
        "all_candidates": rows,
        "paper_only": True,
        "live_execution_allowed": False,
        "probability_claim": None,
    }
