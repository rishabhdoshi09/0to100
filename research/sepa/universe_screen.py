"""Historically defensible NSE investability screen for SEPA-001R."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

import numpy as np
import pandas as pd

from research.sepa.frames import iso_date, pit_universe, slice_as_of
from research.sepa.integrity import unresolved_gap_symbols


def _turnover(df: pd.DataFrame, window: int = 20) -> float:
    if df is None or len(df) == 0:
        return 0.0
    close = pd.to_numeric(df["close"], errors="coerce")
    vol = pd.to_numeric(df["volume"], errors="coerce") if "volume" in df.columns else pd.Series(0.0, index=df.index)
    t = (close * vol).tail(window)
    val = float(t.mean()) if len(t) else 0.0
    return val if val == val else 0.0


def screen_investable(
    frames: Mapping[str, pd.DataFrame],
    *,
    as_of=None,
    min_price: float = 20.0,
    min_turnover: float = 5_000_000.0,
    min_sessions: int = 260,
    exclude_symbols: set[str] | None = None,
) -> dict[str, Any]:
    """Point-in-time screen. `as_of=None` uses each frame's last bar (research book build)."""
    exclude = {s.upper() for s in (exclude_symbols or set())}
    kept: dict[str, pd.DataFrame] = {}
    reasons: dict[str, int] = defaultdict(int)
    excluded_rows: list[dict[str, Any]] = []
    turnovers: list[float] = []
    starting = len(frames)

    for raw_sym, df in frames.items():
        sym = str(raw_sym).upper()
        sliced = slice_as_of(df, as_of) if as_of is not None else df
        if sliced is None or len(sliced) == 0:
            reasons["no_bars"] += 1
            excluded_rows.append({"symbol": sym, "reason": "no_bars"})
            continue
        if len(sliced) < min_sessions:
            reasons["short_history"] += 1
            excluded_rows.append({"symbol": sym, "reason": "short_history", "sessions": int(len(sliced))})
            continue
        need = {"open", "high", "low", "close"}
        if not need.issubset(set(c.lower() for c in sliced.columns) | set(sliced.columns)):
            # columns are expected lowercase from bhav
            missing = [c for c in ("open", "high", "low", "close") if c not in sliced.columns]
            if missing:
                reasons["incomplete_ohlcv"] += 1
                excluded_rows.append({"symbol": sym, "reason": "incomplete_ohlcv", "missing": missing})
                continue
        try:
            px = float(sliced["close"].iloc[-1])
        except Exception:
            reasons["bad_close"] += 1
            excluded_rows.append({"symbol": sym, "reason": "bad_close"})
            continue
        if px < min_price:
            reasons["min_price"] += 1
            excluded_rows.append({"symbol": sym, "reason": "min_price", "close": px})
            continue
        to = _turnover(sliced)
        turnovers.append(to)
        if to < min_turnover:
            reasons["min_turnover"] += 1
            excluded_rows.append({"symbol": sym, "reason": "min_turnover", "turnover": to})
            continue
        if sym in exclude:
            reasons["unresolved_ca_gap"] += 1
            excluded_rows.append({"symbol": sym, "reason": "unresolved_ca_gap"})
            continue
        kept[sym] = df

    arr = np.array(turnovers, dtype=float) if turnovers else np.array([0.0])
    q = lambda p: float(np.nanpercentile(arr, p)) if arr.size else 0.0
    return {
        "as_of": iso_date(as_of) if as_of is not None else "",
        "starting_universe": starting,
        "eligible_universe": len(kept),
        "exclusions": dict(reasons),
        "excluded": excluded_rows[:500],
        "n_excluded_listed": min(500, len(excluded_rows)),
        "n_excluded": len(excluded_rows),
        "turnover": {
            "min_required": min_turnover,
            "p25": round(q(25), 2),
            "p50": round(q(50), 2),
            "p75": round(q(75), 2),
            "p90": round(q(90), 2),
        },
        "min_price": min_price,
        "min_sessions": min_sessions,
        "symbols": sorted(kept),
        "frames": kept,
    }


def yearly_eligible_counts(
    frames: Mapping[str, pd.DataFrame],
    years: list[int],
    **screen_kwargs,
) -> dict[str, int]:
    out: dict[str, int] = {}
    for y in years:
        as_of = f"{int(y)}-12-31"
        packed = screen_investable(frames, as_of=as_of, **screen_kwargs)
        out[str(y)] = int(packed["eligible_universe"])
    return out


def load_research_frames(
    *,
    max_symbols: int | None = None,
    min_price: float = 20.0,
    min_turnover: float = 5_000_000.0,
    min_sessions: int = 260,
    drop_unresolved_gaps: bool = True,
) -> dict[str, Any]:
    """Build the 001R book from the official bhav store. No invented bars."""
    from data.bhavcopy_runtime import ensure_loaded
    from data.bhavcopy_store import get_ohlcv, store_symbols

    ensure_loaded(rebuild_from_local=False)
    raw: dict[str, pd.DataFrame] = {}
    for sym in store_symbols() or []:
        df = get_ohlcv(sym)
        if df is None or len(df) < 80:
            continue
        raw[str(sym).upper()] = df
    gaps = unresolved_gap_symbols(raw) if drop_unresolved_gaps else []
    exclude = {g["symbol"] for g in gaps}
    screened = screen_investable(
        raw, min_price=min_price, min_turnover=min_turnover,
        min_sessions=min_sessions, exclude_symbols=exclude,
    )
    frames = screened["frames"]
    if max_symbols and len(frames) > int(max_symbols):
        ranked = []
        for sym, df in frames.items():
            ranked.append((_turnover(df), sym))
        ranked.sort(reverse=True)
        keep = {sym for _, sym in ranked[: int(max_symbols)]}
        frames = {s: frames[s] for s in frames if s in keep}
        screened["eligible_universe"] = len(frames)
        screened["capped_at"] = int(max_symbols)
    screened["frames"] = frames
    screened["unresolved_gaps"] = gaps
    screened["starting_store"] = len(raw)
    try:
        last = max(df.index[-1] for df in frames.values()) if frames else None
        if last is not None:
            u = pit_universe(last)
            screened["pit_universe"] = {
                "n": len(u.get("symbols") or []),
                "complete": u.get("universe_complete"),
                "source": u.get("source"),
                "note": u.get("note"),
                "research_grade": u.get("research_grade"),
            }
    except Exception as exc:
        screened["pit_universe"] = {"error": str(exc)}
    return screened
