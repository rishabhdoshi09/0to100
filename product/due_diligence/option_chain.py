"""Nearest-expiry option-chain snapshot from an NSE JSON payload.

Does not import options.analytics (that module pulls Streamlit). Empty stays
empty. PCR / max pain / ATM IV are descriptive, not a trade signal.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def _f(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def _oi(block: Mapping[str, Any] | None) -> float:
    return _f((block or {}).get("openInterest")) or 0.0


def _iv(block: Mapping[str, Any] | None) -> float | None:
    number = _f((block or {}).get("impliedVolatility"))
    if number is None or number <= 0:
        return None
    return number


def compute_max_pain(rows: Sequence[Mapping[str, Any]]) -> float | None:
    strikes: list[float] = []
    call_oi: dict[float, float] = {}
    put_oi: dict[float, float] = {}
    for row in rows:
        strike = _f(row.get("strikePrice") or row.get("strike"))
        if strike is None:
            continue
        strikes.append(strike)
        call_oi[strike] = call_oi.get(strike, 0.0) + _oi(row.get("CE") if isinstance(row.get("CE"), Mapping) else None)
        put_oi[strike] = put_oi.get(strike, 0.0) + _oi(row.get("PE") if isinstance(row.get("PE"), Mapping) else None)
        if "ce_oi" in row:
            call_oi[strike] = call_oi.get(strike, 0.0) + (_f(row.get("ce_oi")) or 0.0)
        if "pe_oi" in row:
            put_oi[strike] = put_oi.get(strike, 0.0) + (_f(row.get("pe_oi")) or 0.0)
    unique = sorted(set(strikes))
    if not unique:
        return None
    best: float | None = None
    best_pain: float | None = None
    for settlement in unique:
        pain = 0.0
        for strike in unique:
            pain += call_oi.get(strike, 0.0) * max(settlement - strike, 0.0)
            pain += put_oi.get(strike, 0.0) * max(strike - settlement, 0.0)
        if best_pain is None or pain < best_pain:
            best_pain = pain
            best = settlement
    return best


def _top_strikes(rows: Sequence[Mapping[str, Any]], side: str, limit: int = 5) -> list[dict[str, Any]]:
    key = "CE" if side == "call" else "PE"
    ranked: list[tuple[float, float]] = []
    for row in rows:
        strike = _f(row.get("strikePrice"))
        if strike is None:
            continue
        oi = _oi(row.get(key) if isinstance(row.get(key), Mapping) else None)
        if oi <= 0:
            continue
        ranked.append((oi, strike))
    ranked.sort(reverse=True)
    return [{"strike": strike, "oi": oi} for oi, strike in ranked[:limit]]


def summarize_option_chain(payload: Mapping[str, Any] | None, *, source_url: str = "") -> dict[str, Any]:
    """Compact nearest-expiry snapshot. available=False when the JSON has no chain."""
    empty = {
        "available": False,
        "source": "NSE option-chain-equities",
        "source_url": source_url,
        "not_a_signal": True,
        "places_orders": False,
    }
    payload = dict(payload or {})
    records = payload.get("records") if isinstance(payload.get("records"), Mapping) else payload
    if not isinstance(records, Mapping):
        return empty
    expiries = [str(item) for item in list(records.get("expiryDates") or []) if item]
    nearest = expiries[0] if expiries else ""
    rows = [
        row for row in list(records.get("data") or [])
        if isinstance(row, Mapping) and (not nearest or str(row.get("expiryDate") or "") == nearest)
    ]
    if not rows:
        return {**empty, "reason": "NSE returned no option-chain rows for this symbol."}
    call_oi = 0.0
    put_oi = 0.0
    for row in rows:
        call_oi += _oi(row.get("CE") if isinstance(row.get("CE"), Mapping) else None)
        put_oi += _oi(row.get("PE") if isinstance(row.get("PE"), Mapping) else None)
    spot = _f(records.get("underlyingValue"))
    pcr = round(put_oi / call_oi, 3) if call_oi > 0 else None
    atm_strike = None
    atm_iv = None
    if spot is not None:
        closest = min(rows, key=lambda row: abs((_f(row.get("strikePrice")) or 0.0) - spot))
        atm_strike = _f(closest.get("strikePrice"))
        ivs = [
            iv for iv in (
                _iv(closest.get("CE") if isinstance(closest.get("CE"), Mapping) else None),
                _iv(closest.get("PE") if isinstance(closest.get("PE"), Mapping) else None),
            )
            if iv is not None
        ]
        if ivs:
            atm_iv = round(sum(ivs) / len(ivs), 2)
    return {
        "available": True,
        "expiry": nearest or None,
        "spot": spot,
        "call_oi": int(call_oi),
        "put_oi": int(put_oi),
        "pcr": pcr,
        "max_pain": compute_max_pain(rows),
        "atm_strike": atm_strike,
        "atm_iv": atm_iv,
        "top_call_oi": _top_strikes(rows, "call"),
        "top_put_oi": _top_strikes(rows, "put"),
        "n_strikes": len(rows),
        "source": "NSE option-chain-equities",
        "source_url": source_url,
        "not_a_signal": True,
        "places_orders": False,
        "note": "Nearest-expiry snapshot from the last acquire. Not live depth, not Greeks, not a buy/sell.",
    }


def summarize_kite_option_chain(
    contracts: Sequence[Mapping[str, Any]],
    *,
    spot: float | None,
    source: str = "ZERODHA_KITE_NFO_READ_ONLY",
) -> dict[str, Any]:
    """Summarize nearest-expiry Kite contracts without replacing broker fields.

    Contracts are normalized by data.nfo_market.quote_to_option_contract;
    therefore OI/volume/depth originate from Kite and IV is explicitly derived
    from observed Kite option prices.
    """
    rows = [dict(row) for row in contracts if isinstance(row, Mapping)]
    empty = {
        "available": False,
        "source": source,
        "source_url": "",
        "not_a_signal": True,
        "places_orders": False,
    }
    if not rows:
        return {**empty, "reason": "Kite returned no eligible option contracts."}
    expiries = sorted({str(row.get("expiry") or "") for row in rows if row.get("expiry")})
    nearest = expiries[0] if expiries else ""
    rows = [row for row in rows if not nearest or str(row.get("expiry") or "") == nearest]
    if not rows:
        return {**empty, "reason": "Kite returned no nearest-expiry option contracts."}

    call_oi = sum(
        int(_f(row.get("oi")) or 0)
        for row in rows
        if str(row.get("option_type") or "").upper() == "CE"
    )
    put_oi = sum(
        int(_f(row.get("oi")) or 0)
        for row in rows
        if str(row.get("option_type") or "").upper() == "PE"
    )
    pcr = round(put_oi / call_oi, 3) if call_oi > 0 else None

    strike_rows: dict[float, dict[str, Any]] = {}
    for row in rows:
        strike = _f(row.get("strike"))
        side = str(row.get("option_type") or "").upper()
        if strike is None or side not in {"CE", "PE"}:
            continue
        bucket = strike_rows.setdefault(
            strike,
            {"strike": strike, "ce_oi": 0.0, "pe_oi": 0.0},
        )
        if side == "CE":
            bucket["ce_oi"] += _f(row.get("oi")) or 0.0
        else:
            bucket["pe_oi"] += _f(row.get("oi")) or 0.0
    pain_rows = list(strike_rows.values())

    spot_value = _f(spot)
    atm_strike = None
    atm_iv = None
    if spot_value is not None and pain_rows:
        atm_strike = min(strike_rows, key=lambda strike: abs(strike - spot_value))
        ivs = [
            _f(row.get("iv"))
            for row in rows
            if _f(row.get("strike")) == atm_strike and (_f(row.get("iv")) or 0) > 0
        ]
        ivs = [iv for iv in ivs if iv is not None]
        if ivs:
            atm_iv = round(sum(ivs) / len(ivs), 2)

    call_rank = sorted(
        (
            (int(_f(row.get("oi")) or 0), _f(row.get("strike")))
            for row in rows
            if str(row.get("option_type") or "").upper() == "CE"
            and (_f(row.get("oi")) or 0) > 0
        ),
        reverse=True,
    )
    put_rank = sorted(
        (
            (int(_f(row.get("oi")) or 0), _f(row.get("strike")))
            for row in rows
            if str(row.get("option_type") or "").upper() == "PE"
            and (_f(row.get("oi")) or 0) > 0
        ),
        reverse=True,
    )

    return {
        "available": True,
        "expiry": nearest or None,
        "spot": spot_value,
        "call_oi": int(call_oi),
        "put_oi": int(put_oi),
        "pcr": pcr,
        "max_pain": compute_max_pain(pain_rows),
        "atm_strike": atm_strike,
        "atm_iv": atm_iv,
        "top_call_oi": [{"strike": strike, "oi": oi} for oi, strike in call_rank[:5]],
        "top_put_oi": [{"strike": strike, "oi": oi} for oi, strike in put_rank[:5]],
        "n_strikes": len(strike_rows),
        "n_contracts": len(rows),
        "source": source,
        "source_url": "",
        "iv_source": "IMPLIED_FROM_KITE_MARKET_QUOTES",
        "not_a_signal": True,
        "places_orders": False,
        "note": (
            "Nearest-expiry snapshot from Zerodha Kite instrument metadata + full option quotes. "
            "OI/volume/depth are Kite fields; IV is derived from observed Kite prices."
        ),
    }
