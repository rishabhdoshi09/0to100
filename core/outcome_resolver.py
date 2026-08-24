"""Canonical outcome resolver — trading sessions on official bhavcopy.

Every YES / NO / WAIT experiment, every Case, and every non-event settles
through this module. Never live quotes. Never “5 calendar days then today’s
price.”

Two measurements, one clock:

  session_close_return  — close-to-close at the horizon-th session
                          (decision journal, non-events / control group)
  first_touch_path      — stop before target, same spirit as the backtest
                          (Cases and signal_log when geometry exists)

A row is due only when the bar series actually contains the horizon-th
session after the decision date. Missing history stays pending. Places no
orders.
"""
from __future__ import annotations

import os
from typing import Any, Mapping

DECISION_HORIZON_SESSIONS = int(os.getenv("QT_DECISION_HORIZON", "5") or 5)
PATH_HORIZON_SESSIONS = int(os.getenv("QT_OUTCOME_HORIZON", "15") or 15)


def _ohlcv(symbol: str):
    try:
        from data.bhavcopy_store import get_ohlcv
        return get_ohlcv(str(symbol or "").upper())
    except Exception:
        return None


def _after(df, from_date: str):
    if df is None or getattr(df, "empty", True):
        return None
    try:
        import pandas as pd
        since = df[df.index >= pd.Timestamp(str(from_date or "")[:10])]
    except Exception:
        return None
    if since is None or getattr(since, "empty", True):
        return None
    return since


def session_close_return(
    symbol: str,
    from_date: str,
    *,
    horizon: int = DECISION_HORIZON_SESSIONS,
) -> tuple[float, float] | None:
    """% change from the close on/after `from_date` to the horizon-th
    later trading session. None if official history is short — never simulated."""
    df = _ohlcv(symbol)
    if df is None or getattr(df, "empty", True) or "close" not in getattr(df, "columns", []):
        return None
    after = _after(df, from_date)
    if after is None or len(after) < int(horizon) + 1:
        return None
    try:
        entry = float(after["close"].iloc[0])
        exit_px = float(after["close"].iloc[int(horizon)])
    except Exception:
        return None
    if entry <= 0:
        return None
    return exit_px, (exit_px - entry) / entry * 100.0


def first_touch_path(
    symbol: str,
    opened_at: str,
    entry: float,
    stop: float,
    target: float,
    *,
    horizon: int = PATH_HORIZON_SESSIONS,
) -> tuple[float, float, int] | None:
    """First-touch target vs stop on official bars.

    Returns (exit_price, outcome_pct, worked) where worked is 1/0/-1 (no fill).
    None = still open or no usable path. Stop is checked before target.
    """
    try:
        entry_f, stop_f, target_f = float(entry), float(stop), float(target)
    except (TypeError, ValueError):
        return None
    if entry_f <= 0 or stop_f <= 0 or target_f <= entry_f or stop_f >= entry_f:
        return None
    df = _ohlcv(symbol)
    if df is None or getattr(df, "empty", True):
        return None
    if not {"high", "low", "close"} <= set(df.columns):
        return None
    since = _after(df, opened_at)
    if since is None:
        return None
    try:
        highs = since["high"].to_numpy(dtype=float)
        lows = since["low"].to_numpy(dtype=float)
        closes = since["close"].to_numpy(dtype=float)
    except Exception:
        return None
    n = len(highs)
    cap = min(n, int(horizon))
    filled = False
    for i in range(cap):
        if not filled:
            if highs[i] >= entry_f:
                filled = True
            else:
                continue
        if lows[i] <= stop_f:
            return (stop_f, (stop_f - entry_f) / entry_f * 100.0, 0)
        if highs[i] >= target_f:
            return (target_f, (target_f - entry_f) / entry_f * 100.0, 1)
    if not filled:
        return (0.0, 0.0, -1) if n >= int(horizon) else None
    if n >= int(horizon):
        last = float(closes[cap - 1])
        return (last, (last - entry_f) / entry_f * 100.0, 1 if last >= entry_f else 0)
    return None


def row_session_return(row: Mapping[str, Any], *, horizon: int = DECISION_HORIZON_SESSIONS):
    """Convenience for journal / non-event rows."""
    return session_close_return(
        str(row.get("symbol") or ""),
        str(row.get("decided_at") or row.get("opened_at") or row.get("ts") or "")[:10],
        horizon=horizon,
    )
