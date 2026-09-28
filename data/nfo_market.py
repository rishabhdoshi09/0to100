"""Read-only NSE NFO instrument/quote normalization.

All broker access here is market-data only. The module never places, modifies,
or cancels orders. Missing quote depth/IV/OI remains missing so downstream
selection can fail closed instead of fabricating tradability.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from data.fno_universe import INDEX_UNDERLYINGS
from options.directional_selector import implied_volatility
from research.intelligence.data.nse_calendar import is_session, load_holidays


IST = ZoneInfo("Asia/Kolkata")
LIVE_QUOTE_MAX_AGE_SECONDS = 180
LIVE_QUOTE_MAX_SKEW_SECONDS = 120


class NfoMarketDataClient:
    """Read-only broker facade for derivatives market data.

    The underlying authenticated SDK remains private. No order/GTT mutation
    method is exposed on this object, so the F&O paper lane cannot call one.
    """
    is_data_only = True

    def __init__(self, session) -> None:
        self._session = session

    @classmethod
    def from_config(cls) -> "NfoMarketDataClient":
        from data.kite_client import KiteClient

        client = KiteClient()
        if not client.is_connected():
            raise RuntimeError("no connected Zerodha session for NFO market data")
        return cls(client.raw)

    def instruments(self, exchange: str) -> list:
        return list(self._session.instruments(exchange))

    def quote(self, keys: Sequence[str]) -> Mapping[str, Any]:
        return self._session.quote(list(keys))

    def historical_with_oi(
        self,
        instrument_token: int,
        from_date: str,
        to_date: str,
        interval: str = "day",
    ) -> list:
        return list(
            self._session.historical_data(
                instrument_token,
                from_date,
                to_date,
                interval,
                oi=True,
            )
        )

    def historical(
        self,
        instrument_token: int,
        from_date: str,
        to_date: str,
        interval: str = "minute",
    ) -> list:
        """Read price bars only; this facade still exposes no order surface."""
        return list(
            self._session.historical_data(
                instrument_token,
                from_date,
                to_date,
                interval,
                oi=False,
            )
        )


def _f(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:
        return default
    return number


def quote_exchange_datetime(quote: Mapping[str, Any] | None) -> datetime | None:
    """Exchange quote-packet timestamp normalized to an IST wall clock."""
    if not isinstance(quote, Mapping):
        return None
    raw = quote.get("timestamp")
    if raw is None:
        raw = quote.get("exchange_timestamp")
    if isinstance(raw, datetime):
        stamp = raw
    else:
        text = str(raw or "").strip()
        if not text:
            return None
        try:
            stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone(IST).replace(tzinfo=None)
    return stamp


def quote_provenance(
    quote: Mapping[str, Any] | None,
    *,
    as_of: date,
    now: datetime | None = None,
    max_age_seconds: int = LIVE_QUOTE_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    """Prove session identity and, for a live-session scan, quote freshness."""
    stamp = quote_exchange_datetime(quote)
    if stamp is None:
        return {
            "ok": False,
            "reason": "QUOTE_TIMESTAMP_UNAVAILABLE",
            "timestamp": None,
            "age_seconds": None,
            "freshness_checked": False,
        }
    if stamp.date() != as_of:
        return {
            "ok": False,
            "reason": "QUOTE_SESSION_MISMATCH",
            "timestamp": stamp.isoformat(),
            "age_seconds": None,
            "freshness_checked": False,
        }

    reference = now
    if reference is not None and reference.tzinfo is not None:
        reference = reference.astimezone(IST).replace(tzinfo=None)
    freshness_checked = bool(reference is not None and reference.date() == as_of)
    age_seconds = None
    if freshness_checked and reference is not None:
        age_seconds = (reference - stamp).total_seconds()
        if age_seconds < -30.0:
            return {
                "ok": False,
                "reason": "QUOTE_TIMESTAMP_IN_FUTURE",
                "timestamp": stamp.isoformat(),
                "age_seconds": round(age_seconds, 3),
                "freshness_checked": True,
            }
        if age_seconds > max(1, int(max_age_seconds)):
            return {
                "ok": False,
                "reason": "QUOTE_STALE",
                "timestamp": stamp.isoformat(),
                "age_seconds": round(age_seconds, 3),
                "freshness_checked": True,
            }

    return {
        "ok": True,
        "reason": "CURRENT_SESSION_QUOTE",
        "timestamp": stamp.isoformat(),
        "age_seconds": round(age_seconds, 3) if age_seconds is not None else None,
        "freshness_checked": freshness_checked,
    }


def quote_timestamp_skew_seconds(*quotes: Mapping[str, Any] | None) -> float | None:
    stamps = [quote_exchange_datetime(quote) for quote in quotes]
    if not stamps or any(stamp is None for stamp in stamps):
        return None
    valid = [stamp for stamp in stamps if stamp is not None]
    return (max(valid) - min(valid)).total_seconds()


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


def candidate_option_instruments(
    instruments: Iterable[Mapping[str, Any]],
    underlying: str,
    *,
    spot: float,
    as_of: date | None = None,
    max_expiries: int = 2,
    max_moneyness_pct: float = 8.0,
) -> list[dict[str, Any]]:
    """Bound quote traffic to plausible near-money contracts."""
    spot = _f(spot)
    if spot <= 0:
        return []
    width = max(0.01, float(max_moneyness_pct)) / 100.0
    lo, hi = spot * (1.0 - width), spot * (1.0 + width)
    return [
        row for row in option_instruments(
            instruments, underlying, as_of=as_of, max_expiries=max_expiries,
        )
        if lo <= _f(row.get("strike")) <= hi
    ]


def _top_depth_price(quote: Mapping[str, Any], side: str) -> float:
    depth = quote.get("depth")
    if not isinstance(depth, Mapping):
        return 0.0
    rows = depth.get(side)
    if not isinstance(rows, Sequence) or not rows:
        return 0.0
    first = rows[0]
    return _f(first.get("price")) if isinstance(first, Mapping) else 0.0


def _trading_sessions_to_expiry(*, as_of: date, expiry: date | None) -> tuple[int | None, int]:
    """Future NSE sessions in (as_of, expiry], plus loaded-holiday provenance."""
    if expiry is None or expiry < as_of:
        return None, 0
    holidays = load_holidays()
    cursor = as_of + timedelta(days=1)
    sessions = 0
    while cursor <= expiry:
        if is_session(cursor, holidays):
            sessions += 1
        cursor += timedelta(days=1)
    return sessions, len(holidays)


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
    trading_sessions_to_expiry, holiday_calendar_loaded = _trading_sessions_to_expiry(
        as_of=today,
        expiry=expiry,
    )
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
        "trading_sessions_to_expiry": trading_sessions_to_expiry,
        "holiday_calendar_loaded": holiday_calendar_loaded,
        "expiry_session_model": (
            "NSE_SESSIONS_WITH_RUNTIME_HOLIDAYS"
            if holiday_calendar_loaded > 0 else "WEEKDAYS_ONLY_NO_HOLIDAY_TABLE"
        ),
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
        if current_oi > 0 and previous_oi > 0
        else None
    )
    return {
        "futures_price_change_pct": round(price_change, 4) if price_change is not None else None,
        "futures_oi_change_pct": round(oi_change, 4) if oi_change is not None else None,
    }


def read_nfo_instruments(client=None) -> list[dict[str, Any]]:
    """Read the broker instrument master. No trading capability is used."""
    if client is None:
        client = NfoMarketDataClient.from_config()
    if hasattr(client, "instruments"):
        rows = client.instruments("NFO")
    else:
        rows = client.raw.instruments("NFO")
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def read_nfo_quotes(tradingsymbols: Sequence[str], client=None) -> dict[str, dict[str, Any]]:
    """Read full NFO quotes in bounded batches, retaining OI and market depth."""
    if client is None:
        client = NfoMarketDataClient.from_config()
    wanted = [str(symbol).strip() for symbol in tradingsymbols if str(symbol).strip()]
    out: dict[str, dict[str, Any]] = {}
    for start in range(0, len(wanted), 500):
        chunk = wanted[start:start + 500]
        keys = [f"NFO:{symbol}" for symbol in chunk]
        raw = client.quote(keys) if hasattr(client, "quote") else client.raw.quote(keys)
        if not isinstance(raw, Mapping):
            continue
        for key, value in raw.items():
            if isinstance(value, Mapping):
                out[str(key).split(":", 1)[-1]] = dict(value)
    return out


def read_market_quotes(keys: Sequence[str], client=None) -> dict[str, dict[str, Any]]:
    """Read full quotes for already exchange-qualified market-data keys."""
    if client is None:
        client = NfoMarketDataClient.from_config()
    wanted = [str(key).strip() for key in keys if str(key).strip()]
    out: dict[str, dict[str, Any]] = {}
    for start in range(0, len(wanted), 500):
        chunk = wanted[start:start + 500]
        raw = client.quote(chunk) if hasattr(client, "quote") else client.raw.quote(chunk)
        if not isinstance(raw, Mapping):
            continue
        for key, value in raw.items():
            if isinstance(value, Mapping):
                out[str(key)] = dict(value)
    return out


def previous_future_close_oi(
    instrument_token: int,
    *,
    as_of: date,
    client=None,
    lookback_days: int = 12,
) -> dict[str, Any] | None:
    """Latest completed daily futures close/OI strictly before the as-of date."""
    if not instrument_token:
        return None
    if client is None:
        client = NfoMarketDataClient.from_config()
    end = as_of - timedelta(days=1)
    start = end - timedelta(days=max(3, int(lookback_days)))
    if hasattr(client, "historical_with_oi"):
        rows = client.historical_with_oi(
            int(instrument_token), start.isoformat(), end.isoformat(), "day",
        )
    else:
        rows = client.raw.historical_data(
            int(instrument_token), start.isoformat(), end.isoformat(), "day", oi=True,
        )
    valid = []
    for row in rows or []:
        if not isinstance(row, Mapping):
            continue
        close = _f(row.get("close"))
        oi = _f(row.get("oi"))
        if close > 0 and oi > 0:
            valid.append(dict(row))
    if not valid:
        return None
    row = valid[-1]
    return {
        "date": str(row.get("date") or "")[:10],
        "close": _f(row.get("close")),
        "oi": _f(row.get("oi")),
        "source": "ZERODHA_KITE_NFO_HISTORICAL_OI",
    }


def read_option_intraday_bars(
    instrument_token: int,
    *,
    from_dt: datetime,
    to_dt: datetime,
    client=None,
    interval: str = "minute",
) -> list[dict[str, Any]]:
    """Return ordered read-only option bars inside an explicit post-entry window."""
    token = int(instrument_token or 0)
    if token <= 0 or to_dt <= from_dt:
        return []
    if client is None:
        client = NfoMarketDataClient.from_config()

    frm = from_dt.strftime("%Y-%m-%d %H:%M:%S")
    to = to_dt.strftime("%Y-%m-%d %H:%M:%S")
    if hasattr(client, "historical"):
        rows = client.historical(token, frm, to, interval)
    else:
        raw = getattr(client, "raw", client)
        rows = raw.historical_data(token, frm, to, interval, oi=False)

    valid: list[dict[str, Any]] = []
    for sequence, raw_row in enumerate(rows or []):
        if not isinstance(raw_row, Mapping):
            continue
        close = _f(raw_row.get("close"))
        if close <= 0:
            continue
        open_px = _f(raw_row.get("open"), close)
        high = _f(raw_row.get("high"), close)
        low = _f(raw_row.get("low"), close)
        raw_time = raw_row.get("date") or raw_row.get("timestamp") or ""
        timestamp = raw_time.isoformat() if isinstance(raw_time, datetime) else str(raw_time or "")
        valid.append({
            "timestamp": timestamp,
            "sequence": sequence,
            "open": open_px if open_px > 0 else close,
            "high": high if high > 0 else close,
            "low": low if low > 0 else close,
            "close": close,
            "last_price": close,
        })

    # Kite normally returns chronological bars. Preserve provider order when a
    # timestamp is absent; otherwise make chronology explicit and deterministic.
    valid.sort(key=lambda row: (
        0 if str(row.get("timestamp") or "") else 1,
        str(row.get("timestamp") or ""),
        int(row.get("sequence") or 0),
    ))
    return valid


def read_option_intraday_mark(
    instrument_token: int,
    *,
    from_dt: datetime,
    to_dt: datetime,
    client=None,
    interval: str = "minute",
) -> dict[str, float] | None:
    """Aggregate post-entry bars for callers that only need a range summary."""
    valid = read_option_intraday_bars(
        instrument_token,
        from_dt=from_dt,
        to_dt=to_dt,
        client=client,
        interval=interval,
    )
    if not valid:
        return None
    return {
        "open": float(valid[0]["open"]),
        "high": max(float(row["high"]) for row in valid),
        "low": min(float(row["low"]) for row in valid),
        "close": float(valid[-1]["close"]),
        "last_price": float(valid[-1]["close"]),
    }


def quote_to_option_paper_mark(quote: Mapping[str, Any]) -> dict[str, float]:
    """Normalize a full option quote into a conservative paper mark."""
    ohlc = quote.get("ohlc")
    ohlc = ohlc if isinstance(ohlc, Mapping) else {}
    last = _f(quote.get("last_price"))
    bid = _top_depth_price(quote, "buy")
    return {
        "open": _f(ohlc.get("open"), last),
        "high": _f(ohlc.get("high"), last),
        "low": _f(ohlc.get("low"), last),
        "close": last,
        "last_price": last,
        "bid": bid,
    }
