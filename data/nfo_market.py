"""Read-only NSE NFO instrument/quote normalization.

All broker access here is market-data only. The module never places, modifies,
or cancels orders. Missing quote depth/IV/OI remains missing so downstream
selection can fail closed instead of fabricating tradability.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Iterable, Mapping, Sequence

from data.fno_universe import INDEX_UNDERLYINGS
from options.directional_selector import implied_volatility


def _f(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:
        return default
    return number


def _expiry_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%b-%y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _underlying(row: Mapping[str, Any]) -> str:
    return str(row.get("name") or "").strip().upper()


def _kind(row: Mapping[str, Any]) -> str:
    return str(row.get("instrument_type") or "").strip().upper()


def _segment(row: Mapping[str, Any]) -> str:
    return str(row.get("segment") or row.get("exchange") or "").strip().upper()


def equity_fo_universe(instruments: Iterable[Mapping[str, Any]]) -> list[str]:
    """Return stock underlyings that have both a future and at least one option."""
    futures: set[str] = set()
    options: set[str] = set()
    for row in instruments:
        name = _underlying(row)
        if not name or name in INDEX_UNDERLYINGS:
            continue
        segment = _segment(row)
        if not segment.startswith("NFO"):
            continue
        kind = _kind(row)
        if kind == "FUT":
            futures.add(name)
        elif kind in {"CE", "PE"}:
            options.add(name)
    return sorted(futures & options)


def nearest_future(
    instruments: Iterable[Mapping[str, Any]],
    underlying: str,
    *,
    as_of: date | None = None,
) -> dict[str, Any] | None:
    today = as_of or date.today()
    wanted = str(underlying or "").upper()
    rows: list[tuple[date, Mapping[str, Any]]] = []
    for row in instruments:
        if _underlying(row) != wanted or _kind(row) != "FUT":
            continue
        expiry = _expiry_date(row.get("expiry"))
        if expiry is None or expiry < today:
            continue
        rows.append((expiry, row))
    if not rows:
        return None
    rows.sort(key=lambda item: item[0])
    expiry, row = rows[0]
    return {**dict(row), "expiry": expiry.isoformat()}


def option_instruments(
    instruments: Iterable[Mapping[str, Any]],
    underlying: str,
    *,
    as_of: date | None = None,
    max_expiries: int = 2,
) -> list[dict[str, Any]]:
    """Nearest stock-option expiries only; expired/index contracts excluded."""
    today = as_of or date.today()
    wanted = str(underlying or "").upper()
    rows: list[tuple[date, Mapping[str, Any]]] = []
    for row in instruments:
        if _underlying(row) != wanted or _kind(row) not in {"CE", "PE"}:
            continue
        expiry = _expiry_date(row.get("expiry"))
        if expiry is None or expiry < today:
            continue
        rows.append((expiry, row))
    expiries = sorted({expiry for expiry, _ in rows})[: max(1, int(max_expiries))]
    keep = set(expiries)
    out = [{**dict(row), "expiry": expiry.isoformat()} for expiry, row in rows if expiry in keep]
    out.sort(key=lambda row: (str(row.get("expiry")), _f(row.get("strike")), _kind(row)))
    return out


def _top_depth_price(quote: Mapping[str, Any], side: str) -> float:
    depth = quote.get("depth")
    if not isinstance(depth, Mapping):
        return 0.0
    rows = depth.get(side)
    if not isinstance(rows, Sequence) or not rows:
        return 0.0
    first = rows[0]
    return _f(first.get("price")) if isinstance(first, Mapping) else 0.0


def quote_to_option_contract(
    instrument: Mapping[str, Any],
    quote: Mapping[str, Any],
    *,
    spot: float,
    as_of: date | None = None,
    rate: float = 0.065,
) -> dict[str, Any]:
    """Normalize Kite NFO metadata + quote into the selector contract schema."""
    expiry = _expiry_date(instrument.get("expiry"))
    today = as_of or date.today()
    dte = max(0, (expiry - today).days) if expiry is not None else 0
    last = _f(quote.get("last_price"))
    bid = _top_depth_price(quote, "buy")
    ask = _top_depth_price(quote, "sell")
    market_for_iv = (bid + ask) / 2.0 if bid > 0 and ask >= bid else last
    strike = _f(instrument.get("strike"))
    kind = _kind(instrument)
    iv = implied_volatility(
        market_price=market_for_iv,
        spot=_f(spot),
        strike=strike,
        dte=dte,
        option_type=kind,
        rate=rate,
    )
    return {
        "symbol": str(instrument.get("tradingsymbol") or ""),
        "instrument_token": instrument.get("instrument_token"),
        "option_type": kind,
        "strike": strike,
        "expiry": expiry.isoformat() if expiry is not None else "",
        "dte": dte,
        "lot_size": int(_f(instrument.get("lot_size"), 0.0)),
        "tick_size": _f(instrument.get("tick_size")),
        "ltp": last,
        "bid": bid,
        "ask": ask,
        "volume": int(_f(quote.get("volume"))),
        "oi": int(_f(quote.get("oi"))),
        "iv": round(iv * 100.0, 4) if iv > 0 else 0.0,
        "quote_depth_available": bid > 0 and ask > 0,
        "iv_source": "IMPLIED_FROM_MARKET_QUOTE" if iv > 0 else "UNAVAILABLE",
        "source": "ZERODHA_KITE_NFO_READ_ONLY",
    }


def futures_oi_features(
    *,
    current_price: float,
    previous_price: float,
    current_oi: float,
    previous_oi: float,
) -> dict[str, float | None]:
    """Point-in-time futures price/OI deltas. Missing/zero baselines remain unknown."""
    current_price = _f(current_price)
    previous_price = _f(previous_price)
    current_oi = _f(current_oi)
    previous_oi = _f(previous_oi)
    price_change = (
        (current_price - previous_price) / previous_price * 100.0
        if current_price > 0 and previous_price > 0
        else None
    )
    oi_change = (
        (current_oi - previous_oi) / previous_oi * 100.0
        if current_oi >= 0 and previous_oi > 0
        else None
    )
    return {
        "futures_price_change_pct": round(price_change, 4) if price_change is not None else None,
        "futures_oi_change_pct": round(oi_change, 4) if oi_change is not None else None,
    }


def read_nfo_instruments(client=None) -> list[dict[str, Any]]:
    """Read the broker instrument master. No trading capability is used."""
    if client is None:
        from data.kite_client import KiteClient
        client = KiteClient()
    rows = client.raw.instruments("NFO")
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def read_nfo_quotes(tradingsymbols: Sequence[str], client=None) -> dict[str, dict[str, Any]]:
    """Read full NFO quotes in bounded batches, retaining OI and market depth."""
    if client is None:
        from data.kite_client import KiteClient
        client = KiteClient()
    wanted = [str(symbol).strip() for symbol in tradingsymbols if str(symbol).strip()]
    out: dict[str, dict[str, Any]] = {}
    for start in range(0, len(wanted), 500):
        chunk = wanted[start:start + 500]
        keys = [f"NFO:{symbol}" for symbol in chunk]
        raw = client.raw.quote(keys)
        if not isinstance(raw, Mapping):
            continue
        for key, value in raw.items():
            if isinstance(value, Mapping):
                out[str(key).split(":", 1)[-1]] = dict(value)
    return out
