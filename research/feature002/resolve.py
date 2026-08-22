"""Attach later official-bar outcomes. Never rewrite frozen feature snapshots."""
from __future__ import annotations

from typing import Any

from research.feature002.ledger import attach_outcome, list_primary_observations, get_observation


def _frame(symbol: str):
    try:
        from data.bhavcopy_store import get_ohlcv
        return get_ohlcv(symbol)
    except Exception:
        return None


def _session_loc(df, session_date: str) -> int | None:
    import pandas as pd
    idx = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    target = pd.Timestamp(session_date)
    hits = [i for i, t in enumerate(idx) if t.date() == target.date()]
    if hits:
        return hits[-1]
    # last bar on or before session (should not invent a later as-of)
    before = [i for i, t in enumerate(idx) if t.date() <= target.date()]
    return before[-1] if before else None


def compute_outcome(symbol: str, session_date: str, *,
                    entry: float | None = None, stop: float | None = None) -> dict[str, Any]:
    df = _frame(symbol)
    if df is None or len(df) < 2:
        return {"unresolved_reason": "no_history", "resolved_at": None}
    loc = _session_loc(df, session_date)
    if loc is None:
        return {"unresolved_reason": "session_not_in_store", "resolved_at": None}
    close = df["close"].astype(float)
    high = df["high"].astype(float) if "high" in df.columns else close
    low = df["low"].astype(float) if "low" in df.columns else close
    opn = df["open"].astype(float) if "open" in df.columns else close
    c0 = float(close.iloc[loc])
    if c0 <= 0:
        return {"unresolved_reason": "bad_close", "resolved_at": None}

    def ret(n: int) -> float | None:
        j = loc + n
        if j >= len(close):
            return None
        cj = float(close.iloc[j])
        if cj != cj or cj <= 0:
            return None
        return cj / c0 - 1.0

    next_open = None
    if loc + 1 < len(opn):
        next_open = float(opn.iloc[loc + 1])

    horizon = min(20, len(close) - loc - 1)
    mae = mfe = None
    if horizon >= 1:
        ref = float(entry) if entry and entry > 0 else c0
        window_h = high.iloc[loc + 1: loc + 1 + horizon]
        window_l = low.iloc[loc + 1: loc + 1 + horizon]
        if len(window_h):
            mfe = float(window_h.max() / ref - 1.0)
            mae = float(1.0 - window_l.min() / ref)

    hit_1r = hit_2r = None
    if entry and stop and float(entry) > float(stop):
        risk = float(entry) - float(stop)
        if horizon >= 1 and risk > 0:
            mx = float(high.iloc[loc + 1: loc + 1 + horizon].max())
            hit_1r = int(mx >= float(entry) + risk)
            hit_2r = int(mx >= float(entry) + 2 * risk)

    r5 = ret(5)
    unresolved = None if r5 is not None else "horizon_incomplete"
    resolved_at = None
    if r5 is not None:
        try:
            from core.market_clock import now_ist
            resolved_at = now_ist().isoformat()
        except Exception:
            from datetime import datetime, timezone
            resolved_at = datetime.now(timezone.utc).isoformat()

    return {
        "resolved_at": resolved_at,
        "next_open": next_open,
        "ret_1d": ret(1),
        "ret_5d": r5,
        "ret_10d": ret(10),
        "ret_20d": ret(20),
        "mae": mae,
        "mfe": mfe,
        "hit_1r": hit_1r,
        "hit_2r": hit_2r,
        "production_traded": None,
        "production_outcome": None,
        "unresolved_reason": unresolved,
    }


def resolve_event(event_id_value: str, *, path=None) -> dict[str, Any]:
    obs = get_observation(event_id_value, path=path)
    if not obs:
        return {"status": "missing"}
    snap_before = obs["feature_snapshot"]
    out = compute_outcome(
        obs["symbol"], obs["session_date"],
        entry=obs.get("entry"), stop=obs.get("stop"),
    )
    attach_outcome(event_id_value, out, path=path)
    snap_after = get_observation(event_id_value, path=path)["feature_snapshot"]
    if snap_before != snap_after:
        raise RuntimeError("feature snapshot mutated during resolve")
    return {"status": "resolved" if out.get("ret_5d") is not None else "unresolved",
            "event_id": event_id_value, "outcome": out}


def resolve_due(*, path=None, limit: int = 500) -> dict[str, int]:
    rows = list_primary_observations(path=path)
    n_ok = n_open = 0
    for row in rows[:limit]:
        if row.get("ret_5d") is not None:
            continue
        res = resolve_event(row["event_id"], path=path)
        if res.get("status") == "resolved":
            n_ok += 1
        else:
            n_open += 1
    return {"resolved": n_ok, "still_open": n_open}
