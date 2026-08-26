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
