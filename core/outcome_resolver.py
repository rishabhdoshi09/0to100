"""Canonical outcome resolver — trading sessions on official bhavcopy.

Every YES / NO / WAIT experiment, every Case, and every non-event settles
through this module. Never live quotes. Never “5 calendar days then today’s
price.”

Two measurements, one clock:

  session_close_return  — close-to-close at the horizon-th session
                          (decision journal, non-events / control group)
  first_touch_path      — stop before target, same spirit as the backtest
                          (Cases and signal_log when geometry exists)

WAIT is a different experiment from YES/NO close-to-close:

  wait_patience — after we refused to buy at P, did the name later
                  offer a cheaper entry, run away without us, or fail?

A row is due only when the bar series actually contains the horizon-th
session after the decision date. Missing history stays pending. Places no
orders.
"""
from __future__ import annotations

import os
from typing import Any, Mapping

# One rulebook (core.evaluation_spec): decisions, the forward tracker and the
# backtest judge a trade over the same horizon (were 5, 15 and 10 sessions).
from core.evaluation_spec import (  # noqa: E402
    DECISION_HORIZON_SESSIONS,
    OUTCOME_HORIZON_SESSIONS as PATH_HORIZON_SESSIONS,
)
WAIT_PULLBACK_PCT = float(os.getenv("QT_WAIT_PULLBACK", "3") or 3)
WAIT_RUN_PCT = float(os.getenv("QT_WAIT_RUN", "2") or 2)

WAIT_OFFERED = "OFFERED"
WAIT_RAN_AWAY = "RAN_AWAY"
WAIT_FAILED = "FAILED"


def _ohlcv(symbol: str):
    try:
        from data.bhavcopy_store import get_ohlcv
        return get_ohlcv(str(symbol or "").upper())
    except Exception:
        return None


def _store_loaded() -> bool:
    """True when the official bhavcopy store is loaded in this process (production)."""
    try:
        from data.bhavcopy_store import is_ready
        return bool(is_ready())
    except Exception:
        return False


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


def resolve_trade(
    symbol: str,
    opened_at: str,
    entry: float,
    stop: float,
    target: float,
    *,
    horizon: int | None = None,
    frame: Any = None,
    sessions: list[str] | None = None,
):
    """Resolve a logged signal exactly as the walk-forward backtest does.

    Uses core.evaluation_spec.simulate_trade on the official bars after the
    signal day (buy-stop fill, gap-aware exits, breakeven trail, mark to the
    close at the horizon). Returns the TradeResult, or None while the trade is
    still open or the bars are not there yet: a trade that has not exited is
    final only once the horizon-th session exists.

    Only OFFICIAL session bars count:

    * a live intraday overlay (``attrs["quantterm_live_overlay_date"]``, set by
      data.nse_live) is not a published bar and is dropped;
    * the bars must be exactly the next official sessions after the signal day
      (core.session_window). A missing session (suspension, missing archive)
      or an unofficial bar ends the verified window there, so a target first
      touched in the 11th official session is never a 10-session WIN, and a
      trade that has not exited inside its verified bars stays unresolved.

    ``sessions`` is the sorted official-session list (a test seam); by default
    it comes from the bhavcopy archive. Bars read from the store with no
    archive listing are unresolved (fail closed); an explicit ``frame`` with no
    listing (tests, an offline copy) is taken as given.
    """
    from core.evaluation_spec import simulate_trade

    horizon = int(horizon or PATH_HORIZON_SESSIONS)
    from_store = frame is None
    df = frame if frame is not None else _ohlcv(symbol)
    if df is None or getattr(df, "empty", True) or not {"high", "low", "close"} <= set(df.columns):
        return None
    try:
        import pandas as pd

        overlay = str((getattr(df, "attrs", None) or {}).get("quantterm_live_overlay_date") or "")[:10]
        if overlay:
            df = df[df.index < pd.Timestamp(overlay)]
        since = df[df.index > pd.Timestamp(str(opened_at or "")[:10])]
    except Exception:
        return None
    window = since.iloc[:horizon]
    if window.empty:
        return None
    if sessions is None:
        from core.session_window import official_dates

        sessions = official_dates()
    if not sessions and from_store and _store_loaded():
        # Store bars (possibly restored from the pickle cache) without the
        # official session list cannot be verified: never settle positionally
        # (cross-review verification QT-06). An explicit frame is the
        # caller's (tests, an offline copy) and is taken as given.
        return None
    if sessions:
        from core.session_window import verified_prefix

        days = [str(getattr(ix, "date", lambda: ix)())[:10] for ix in window.index]
        window = window.iloc[:verified_prefix(days, str(opened_at or "")[:10], list(sessions))]
        if window.empty:
            return None
    opens = window["open"].to_numpy(dtype=float) if "open" in window.columns else None
    result = simulate_trade(
        entry, stop, target,
        window["high"].to_numpy(dtype=float),
        window["low"].to_numpy(dtype=float),
        window["close"].to_numpy(dtype=float),
        opens=opens,
    )
    if result.outcome in ("WIN", "LOSS", "SCRATCH"):
        return result
    return result if len(window) >= horizon else None


def counterfactual_outcome(
    symbol: str,
    as_of: str,
    entry: Any,
    stop: Any,
    target: Any,
    *,
    horizon: int | None = None,
    frame: Any = None,
    sessions: list[str] | None = None,
) -> dict[str, Any] | None:
    """A rejected or waited candidate, judged exactly as a taken trade would be.

    The same rulebook as signal_log and the paper book (resolve_trade ->
    core.evaluation_spec.simulate_trade: buy-stop fill, gap-aware stop and
    target, breakeven trail, marked at the horizon close) and the same costs
    as the walk-forward (core.costs.cost_in_r). Returns None while the trade
    is still open or the bars are not published, so nothing is settled early.
    A candidate without valid levels cannot be judged and comes back as
    NO_LEVELS with no R.
    """
    from core.evaluation_spec import spec_tag

    h = int(horizon or PATH_HORIZON_SESSIONS)
    base = {"eval_spec": spec_tag(h), "horizon_sessions": h, "not_pnl": True}
    try:
        e, s, t = float(entry), float(stop), float(target)
    except (TypeError, ValueError):
        return {**base, "outcome": "NO_LEVELS", "gross_r": None, "net_r": None, "return_pct": None}
    if not (e > s > 0 and t > e):
        return {**base, "outcome": "NO_LEVELS", "gross_r": None, "net_r": None, "return_pct": None}
    res = resolve_trade(symbol, as_of, e, s, t, horizon=h, frame=frame, sessions=sessions)
    if res is None:
        return None
    if not res.filled:
        return {**base, "outcome": "NO_FILL", "gross_r": 0.0, "net_r": 0.0, "return_pct": 0.0}
    try:
        from core.costs import cost_in_r
        cost = cost_in_r((res.fill_price - s) / res.fill_price, "CNC")
    except Exception:
        cost = 0.0
    return {**base, "outcome": res.outcome, "gross_r": round(float(res.r), 6),
            "net_r": round(float(res.r) - cost, 6), "cost_r": round(cost, 6),
            "fill_price": float(res.fill_price), "exit_price": float(res.exit_price),
            "return_pct": round((res.exit_price / res.fill_price - 1.0) * 100.0, 4)}


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


def first_touch_detail(
    symbol: str,
    opened_at: str,
    entry: float,
    stop: float,
    target: float,
    *,
    horizon: int = PATH_HORIZON_SESSIONS,
) -> dict | None:
    """first_touch_path plus the fill and exit session dates.

    Returns {"exit_price", "pct", "worked", "fill_date", "exit_date"} (dates
    are ISO strings; both None for a no-fill, worked -1). None = still open or
    no usable path. The dates let a benchmark be measured over exactly the
    sessions the trade was held.
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
        import pandas as pd
        # The decision-day bar predates (EOD) or straddles (intraday) the
        # signal, so like the walk-forward backtest the path starts next session.
        since = since[since.index > pd.Timestamp(str(opened_at or "")[:10])]
        highs = since["high"].to_numpy(dtype=float)
        lows = since["low"].to_numpy(dtype=float)
        closes = since["close"].to_numpy(dtype=float)
        days = [pd.Timestamp(x).date().isoformat() for x in since.index]
    except Exception:
        return None

    def _out(price, worked, fill_i, exit_i):
        return {
            "exit_price": float(price),
            "pct": (float(price) - entry_f) / entry_f * 100.0 if worked != -1 else 0.0,
            "worked": worked,
            "fill_date": days[fill_i] if fill_i is not None else None,
            "exit_date": days[exit_i] if exit_i is not None else None,
        }

    n = len(highs)
    cap = min(n, int(horizon))
    fill_i = None
    for i in range(cap):
        if fill_i is None:
            if highs[i] >= entry_f:
                fill_i = i
            else:
                continue
        if lows[i] <= stop_f:
            return _out(stop_f, 0, fill_i, i)
        if highs[i] >= target_f:
            return _out(target_f, 1, fill_i, i)
    if fill_i is None:
        return _out(0.0, -1, None, None) if n >= int(horizon) else None
    if n >= int(horizon):
        last = float(closes[cap - 1])
        return _out(last, 1 if last >= entry_f else 0, fill_i, cap - 1)
    return None


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
    got = first_touch_detail(symbol, opened_at, entry, stop, target, horizon=horizon)
    if got is None:
        return None
    if got["worked"] == -1:
        return (0.0, 0.0, -1)
    return (got["exit_price"], got["pct"], got["worked"])


def wait_patience(
    symbol: str,
    from_date: str,
    entry: float,
    *,
    horizon: int = PATH_HORIZON_SESSIONS,
    pullback_pct: float = WAIT_PULLBACK_PCT,
    run_pct: float = WAIT_RUN_PCT,
) -> tuple[float, float, str] | None:
    """Timing-patience experiment for WAIT — not a second NO.

    We refused to buy at `entry`. Over the next `horizon` sessions:

      OFFERED   — price later tagged a valid cheaper entry (default 3% pullback)
      RAN_AWAY  — never offered that entry, and price ran without us
      FAILED    — never offered, never ran: the setup died or went nowhere

    Uses the session after the decision day, not the decision bar itself.
    None if official history is short. Never live quotes.
    """
    try:
        entry_f = float(entry)
        pull = float(pullback_pct) / 100.0
        run = float(run_pct) / 100.0
    except (TypeError, ValueError):
        return None
    if entry_f <= 0 or pull <= 0 or run <= 0:
        return None
    df = _ohlcv(symbol)
    if df is None or getattr(df, "empty", True):
        return None
    if not {"high", "low", "close"} <= set(df.columns):
        return None
    after = _after(df, from_date)
    if after is None or len(after) < int(horizon) + 1:
        return None
    try:
        window = after.iloc[1: int(horizon) + 1]
        lows = window["low"].to_numpy(dtype=float)
        highs = window["high"].to_numpy(dtype=float)
        last = float(window["close"].iloc[-1])
    except Exception:
        return None
    if len(lows) < int(horizon):
        return None
    offered_px = entry_f * (1.0 - pull)
    ran_px = entry_f * (1.0 + run)
    offered = bool((lows <= offered_px).any())
    ran = bool((highs >= ran_px).any())
    if offered:
        result = WAIT_OFFERED
    elif ran:
        result = WAIT_RAN_AWAY
    else:
        result = WAIT_FAILED
    pct = (last - entry_f) / entry_f * 100.0
    return last, pct, result


def row_session_return(row: Mapping[str, Any], *, horizon: int = DECISION_HORIZON_SESSIONS):
    """Convenience for journal / non-event rows."""
    return session_close_return(
        str(row.get("symbol") or ""),
        str(row.get("decided_at") or row.get("opened_at") or row.get("ts") or "")[:10],
        horizon=horizon,
    )
