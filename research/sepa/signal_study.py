"""Variant G — pure Stage-2 + RS signal study. Not SEPA R expectancy."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from research.sepa.frames import iso_date


def forward_path_study(fwd: pd.DataFrame | None, *, entry_open: float | None = None) -> dict[str, Any] | None:
    """Next-open entry with genuine OHLC MAE/MFE %. No placeholder R fields."""
    if fwd is None or len(fwd) < 1:
        return None
    o = float(fwd["open"].iloc[0] if entry_open is None else entry_open)
    if o <= 0:
        return None
    high = pd.to_numeric(fwd["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(fwd["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(fwd["close"], errors="coerce").to_numpy(dtype=float)

    def _pct_at(n: int) -> float | None:
        if len(close) < n:
            return None
        c = float(close[n - 1])
        if c != c:
            return None
        return c / o - 1.0

    n = len(fwd)
    # Path MAE/MFE from the next-open fill, using every session through n.
    mae_pct = 0.0
    mfe_pct = 0.0
    mae_i = 0
    mfe_i = 0
    hit5 = False
    hit10 = False
    for i in range(n):
        lo = float(low[i])
        hi = float(high[i])
        if lo == lo:
            adv = lo / o - 1.0
            if adv < mae_pct:
                mae_pct = adv
                mae_i = i
        if hi == hi:
            fav = hi / o - 1.0
            if fav > mfe_pct:
                mfe_pct = fav
                mfe_i = i
            if fav >= 0.05:
                hit5 = True
            if fav >= 0.10:
                hit10 = True
    dd_before_gain = bool(mae_i < mfe_i) if mfe_pct > 0 else bool(mae_pct < 0)
    return {
        "kind": "SIGNAL_STUDY",
        "not_sepa_r": True,
        "entry": o,
        "entry_index": 0,
        "exit_index": n - 1,
        "hold_sessions": n,
        "entry_date": iso_date(fwd.index[0]),
        "exit_date": iso_date(fwd.index[n - 1]),
        "fwd_5d_pct": _pct_at(min(5, n)) if n >= 5 else _pct_at(n) if n else None,
        "fwd_10d_pct": _pct_at(min(10, n)) if n >= 10 else None,
        "fwd_20d_pct": _pct_at(min(20, n)) if n >= 20 else None,
        "mae_pct": float(mae_pct),
        "mfe_pct": float(mfe_pct),
        "hit_5pct": bool(hit5),
        "hit_10pct": bool(hit10),
        "drawdown_before_gain": dd_before_gain,
    }


def summarize_signal_study(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive forward-return stats. Explicitly not an R-trade harness."""
    live = [r for r in rows if not r.get("ca_censored")]
    if not live:
        return {
            "n": 0,
            "n_raw": len(rows),
            "kind": "SIGNAL_STUDY",
            "not_sepa_r": True,
            "expectancy_r": None,
            "note": "G is a forward-% signal study. It is not SEPA R expectancy.",
        }

    def _col(key: str) -> np.ndarray:
        xs = [r.get(key) for r in live if r.get(key) is not None]
        return np.asarray(xs, dtype=float)

    def _pack(arr: np.ndarray) -> dict[str, Any]:
        if arr.size == 0:
            return {"n": 0, "mean": None, "median": None}
        return {
            "n": int(arr.size),
            "mean": round(float(arr.mean()) * 100.0, 4),
            "median": round(float(np.median(arr)) * 100.0, 4),
        }

    f20 = _col("fwd_20d_pct")
    return {
        "n": len(live),
        "n_raw": len(rows),
        "kind": "SIGNAL_STUDY",
        "not_sepa_r": True,
        "layer": "signal",
        "expectancy_r": None,
        "fwd_5d": _pack(_col("fwd_5d_pct")),
        "fwd_10d": _pack(_col("fwd_10d_pct")),
        "fwd_20d": _pack(f20),
        "mae_pct_mean": round(float(_col("mae_pct").mean()) * 100.0, 4) if live else None,
        "mfe_pct_mean": round(float(_col("mfe_pct").mean()) * 100.0, 4) if live else None,
        "hit_5pct": round(100.0 * sum(1 for r in live if r.get("hit_5pct")) / len(live), 2),
        "hit_10pct": round(100.0 * sum(1 for r in live if r.get("hit_10pct")) / len(live), 2),
        "drawdown_before_gain": round(
            100.0 * sum(1 for r in live if r.get("drawdown_before_gain")) / len(live), 2,
        ),
        "note": (
            "G answers whether Stage-2 + RS have forward predictive value "
            "without the production scanner. It is not core SEPA and is not "
            "an R-multiple expectancy."
        ),
    }
