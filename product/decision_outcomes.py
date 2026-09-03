"""Settle mature outcomes from frozen entry/stop. No retrospective better stops.

Uses official bars after T. Path metrics (MFE/MAE, time to favorable/adverse)
are measured against the frozen invalidation, never a rebuilt stop.
"""
from __future__ import annotations

from typing import Any, Mapping

from product.risk_audit import r_multiple


def _f(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _high(row: Mapping[str, Any]) -> float | None:
    return _f(row.get("high") or row.get("High") or row.get("h"))


def _low(row: Mapping[str, Any]) -> float | None:
    return _f(row.get("low") or row.get("Low") or row.get("l"))


def _close(row: Mapping[str, Any]) -> float | None:
    return _f(row.get("close") or row.get("Close") or row.get("c") or row.get("ltp"))


def path_metrics(
    *,
    entry: float | None,
    stop: float | None,
    target: float | None,
    bars: list[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """bars are later sessions, chronological. Frozen entry/stop only."""
    e = _f(entry)
    s = _f(stop)
    t = _f(target)
    if e is None or not bars:
        return {
            "return_pct": None,
            "r_multiple": None,
            "mfe_pct": None,
            "mae_pct": None,
            "target_status": "UNKNOWN",
            "stop_status": "UNKNOWN",
            "time_to_favorable": None,
            "time_to_adverse": None,
            "time_to_invalidation": None,
            "path_note": "No later official bars or missing frozen entry.",
        }
    mfe = None
    mae = None
    t_fav = None
    t_adv = None
    t_inv = None
    hit_target = False
    hit_stop = False
    last_close = None
    for i, bar in enumerate(bars, start=1):
        hi, lo, cl = _high(bar), _low(bar), _close(bar)
        if hi is not None:
            fav = (hi - e) / e * 100.0
            mfe = fav if mfe is None else max(mfe, fav)
            if t_fav is None and fav >= 1.0:
                t_fav = i
        if lo is not None:
            adv = (lo - e) / e * 100.0
            mae = adv if mae is None else min(mae, adv)
            if t_adv is None and adv <= -1.0:
                t_adv = i
            if s is not None and lo <= s and t_inv is None:
                t_inv = i
                hit_stop = True
        if t is not None and hi is not None and hi >= t:
            hit_target = True
        if cl is not None:
            last_close = cl
    ret = None if last_close is None else round((last_close - e) / e * 100.0, 3)
    r_mult = r_multiple(entry=e, stop=s, exit_price=last_close)
    return {
        "return_pct": ret,
        "r_multiple": r_mult,
        "mfe_pct": None if mfe is None else round(mfe, 3),
        "mae_pct": None if mae is None else round(mae, 3),
        "target_status": "HIT" if hit_target else ("ARTIFICIAL_UNHIT" if t is not None else "NO_STRUCTURAL_TARGET"),
        "stop_status": "HIT" if hit_stop else "INTACT",
        "time_to_favorable": t_fav,
        "time_to_adverse": t_adv,
        "time_to_invalidation": t_inv,
        "bars_used": len(bars),
        "stop_was_frozen": s is not None,
        "retrospective_stop": False,
        "path_note": "Measured on official later bars against the frozen entry/stop.",
    }


def later_bars(symbol: str, as_of: str, *, horizon: int = 10) -> list[dict[str, Any]]:
    """Official bars strictly after T. Empty if history is missing."""
    try:
        from product.historical_replay import ohlcv_as_of
        from datetime import date, timedelta

        end = (date.fromisoformat(str(as_of)[:10]) + timedelta(days=horizon * 3)).isoformat()
        frame = ohlcv_as_of(symbol, end)
    except Exception:
        return []
    if frame is None or getattr(frame, "empty", True):
        return []
    out = []
    cutoff = str(as_of)[:10]
    for idx, row in frame.iterrows():
        day = str(getattr(idx, "date", lambda: idx)())[:10]
        if day <= cutoff:
            continue
        rec = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        rec["date"] = day
        out.append(rec)
        if len(out) >= horizon:
            break
    return out


def settle_frozen(row: Mapping[str, Any], *, horizon: int = 10) -> dict[str, Any]:
    """Attach path metrics. Does not mutate the freeze payload."""
    symbol = str(row.get("symbol") or "")
    as_of = str(row.get("as_of") or "")[:10]
    bars = later_bars(symbol, as_of, horizon=horizon)
    metrics = path_metrics(
        entry=_f(row.get("entry") or row.get("hypothetical_entry")),
        stop=_f(row.get("stop") or row.get("hypothetical_stop")),
        target=_f(row.get("target") or row.get("hypothetical_target")),
        bars=bars,
    )
    out = dict(row)
    out["path"] = metrics
    out["forward_return_pct"] = metrics.get("return_pct")
    out["r_multiple"] = metrics.get("r_multiple")
    out["mfe_pct"] = metrics.get("mfe_pct")
    out["mae_pct"] = metrics.get("mae_pct")
    out["outcome_rewrote_freeze"] = False
    return out
