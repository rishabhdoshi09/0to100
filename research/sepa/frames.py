"""PIT frame helpers — official bhavcopy only, never future bars."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd


def as_of_stamp(value) -> pd.Timestamp:
    if isinstance(value, pd.Timestamp):
        ts = value
    elif isinstance(value, datetime):
        ts = pd.Timestamp(value)
    elif isinstance(value, date):
        ts = pd.Timestamp(value)
    else:
        ts = pd.Timestamp(str(value))
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def iso_date(value) -> str:
    ts = as_of_stamp(value)
    return str(ts.date())


def slice_as_of(frame: Any, as_of) -> pd.DataFrame | None:
    """Bars with index date <= as_of. Empty/None stays None. Never fabricates."""
    if frame is None:
        return None
    try:
        if len(frame) == 0:
            return None
        data = frame.sort_index()
        cutoff = as_of_stamp(as_of)
        idx = pd.DatetimeIndex(data.index).tz_localize(None) if getattr(data.index, "tz", None) else pd.DatetimeIndex(data.index)
        # compare on dates so a session timestamp still includes that day
        mask = idx.normalize() <= cutoff
        out = data.loc[mask]
        return out if len(out) else None
    except Exception:
        return None


def last_session_iso(frame: Any) -> str:
    if frame is None or len(frame) == 0:
        return ""
    try:
        ts = pd.Timestamp(frame.index[-1])
        return str(ts.date())
    except Exception:
        return ""


def close_series(frame: Any) -> pd.Series | None:
    if frame is None or len(frame) == 0:
        return None
    for col in ("close", "Close"):
        if col in frame.columns:
            s = pd.to_numeric(frame[col], errors="coerce").dropna()
            return s if len(s) else None
    return None


def sma(close: pd.Series | None, window: int) -> float | None:
    if close is None or len(close) < window:
        return None
    val = float(close.iloc[-window:].mean())
    if val != val:
        return None
    return val


def atr(frame: Any, period: int = 14) -> float | None:
    if frame is None or len(frame) < period + 1:
        return None
    try:
        high = pd.to_numeric(frame["high"], errors="coerce")
        low = pd.to_numeric(frame["low"], errors="coerce")
        close = pd.to_numeric(frame["close"], errors="coerce")
    except Exception:
        return None
    trs = []
    for i in range(-period, 0):
        prev = float(close.iloc[i - 1])
        h = float(high.iloc[i])
        l = float(low.iloc[i])
        trs.append(max(h - l, abs(h - prev), abs(l - prev)))
    if not trs:
        return None
    return float(sum(trs) / len(trs))


def load_symbol_frame(symbol: str, as_of):
    """Official OHLCV through as_of. CA applied on read inside get_ohlcv."""
    from data.bhavcopy_runtime import get_ohlcv

    raw = get_ohlcv(str(symbol).upper())
    return slice_as_of(raw, as_of)


def pit_universe(as_of) -> dict[str, Any]:
    from data.nse_universe import point_in_time_universe

    info = point_in_time_universe(as_of_stamp(as_of))
    complete = bool(info.get("survivorship_complete")) and bool(info.get("research_grade", True))
    # research_grade False still has membership rows — treat complete membership
    # as universe_complete; research_grade is a separate honesty flag.
    universe_complete = bool(info.get("survivorship_complete"))
    return {
        "symbols": [str(s).upper() for s in (info.get("symbols") or [])],
        "universe_complete": universe_complete,
        "research_grade": bool(info.get("research_grade")),
        "source": str(info.get("source") or ""),
        "note": str(info.get("note") or ""),
        "as_of": str(info.get("as_of") or iso_date(as_of)),
        "complete_alias": complete,
    }


def ca_status() -> dict[str, Any]:
    """Honest CA flag: events on disk are not enough. Verify must PASS."""
    try:
        from data.corporate_actions import events_path, load_events, ledger_status
    except Exception:
        return {
            "ca_complete": False, "n_symbols": 0, "n_events": 0,
            "verified": False, "source": "", "path": "",
            "note": "corporate_actions unread",
        }
    try:
        events = load_events() or {}
    except Exception:
        events = {}
    n_symbols = len(events)
    n_events = sum(len(v) for v in events.values())
    try:
        status = ledger_status() or {}
    except Exception:
        status = {}
    verified = bool(status.get("adjustment_verified"))
    # ca_complete is earned only when share-count events exist AND verify PASS.
    ca_complete = bool(n_events) and verified
    path = ""
    try:
        path = str(events_path())
    except Exception:
        path = str(status.get("path") or "")
    note = str(status.get("verify_note") or status.get("note") or "")
    if not n_events:
        note = note or "logs/ca_events.json absent — prices are raw/unadjusted"
    elif not verified:
        note = note or "CA ledger present but adjustment_verified is false"
    return {
        "ca_complete": ca_complete,
        "n_symbols": n_symbols,
        "n_events": n_events,
        "verified": verified,
        "source": str(status.get("source") or ""),
        "path": path,
        "research_grade": bool(status.get("research_grade")),
        "gap_rate": status.get("gap_rate"),
        "note": note,
    }
