"""NSE cross-sectional relative strength percentile (rs_cs_v1).

s = 0.40*r63 + 0.20*r126 + 0.20*r189 + 0.20*r252
Percentile = 100 * (# of valid universe scores strictly below this name) / N
Fail-closed when any horizon is missing. Not IBD proprietary. Not Nifty excess.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from research.sepa.config import SepaConfig
from research.sepa.frames import close_series, iso_date, slice_as_of


def horizon_returns(close: pd.Series, horizons: Sequence[int]) -> dict[int, float] | None:
    out: dict[int, float] = {}
    if close is None or len(close) == 0:
        return None
    last = float(close.iloc[-1])
    if last <= 0:
        return None
    for n in horizons:
        if len(close) < n + 1:
            return None
        start = float(close.iloc[-n - 1])
        if start <= 0:
            return None
        out[int(n)] = last / start - 1.0
    return out


def composite_score(returns: Mapping[int, float], config: SepaConfig) -> float | None:
    acc = 0.0
    for n, w in zip(config.rs_horizons, config.rs_weights):
        if int(n) not in returns:
            return None
        acc += float(w) * float(returns[int(n)])
    return acc


def percentile_rank(score: float, universe_scores: Sequence[float]) -> float | None:
    arr = np.asarray([s for s in universe_scores if s is not None and s == s], dtype=float)
    if arr.size == 0:
        return None
    below = float(np.sum(arr < score))
    return 100.0 * below / float(arr.size)


def score_one(frame, config: SepaConfig) -> dict[str, Any]:
    close = close_series(frame)
    rets = horizon_returns(close, config.rs_horizons) if close is not None else None
    if rets is None:
        return {"available": False, "score": None, "components": {}}
    score = composite_score(rets, config)
    return {
        "available": score is not None,
        "score": score,
        "components": {f"r{n}": rets[n] for n in config.rs_horizons if n in rets},
    }


def build_rs_table(
    frames: Mapping[str, Any],
    as_of,
    config: SepaConfig,
    *,
    universe: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Percentiles for every name in `universe` (or all frames) as-of date.

    Only uses bars <= as_of. Ranking set is names with a valid composite.
    """
    names = [str(s).upper() for s in (universe if universe is not None else frames.keys())]
    scores: dict[str, float] = {}
    components: dict[str, dict[str, float]] = {}
    for sym in names:
        frame = frames.get(sym)
        sliced = slice_as_of(frame, as_of) if frame is not None else None
        one = score_one(sliced, config)
        if one.get("available") and one.get("score") is not None:
            scores[sym] = float(one["score"])
            components[sym] = dict(one["components"])
    values = list(scores.values())
    percentiles = {sym: percentile_rank(sc, values) for sym, sc in scores.items()}
    return {
        "as_of": iso_date(as_of),
        "n_ranked": len(percentiles),
        "n_universe": len(names),
        "version": config.rs_version,
        "scores": scores,
        "percentiles": percentiles,
        "components": components,
        "formula": (
            "0.40*r63 + 0.20*r126 + 0.20*r189 + 0.20*r252 ; "
            "percentile = 100 * count(universe_score < score) / N"
        ),
    }


def lookup_rs(table: Mapping[str, Any] | None, symbol: str) -> dict[str, Any]:
    if not table:
        return {"available": False, "percentile": None, "score": None, "components": {}}
    sym = str(symbol).upper()
    pct = (table.get("percentiles") or {}).get(sym)
    score = (table.get("scores") or {}).get(sym)
    comps = (table.get("components") or {}).get(sym) or {}
    return {
        "available": pct is not None,
        "percentile": None if pct is None else float(pct),
        "score": None if score is None else float(score),
        "components": comps,
        "n_ranked": int(table.get("n_ranked") or 0),
        "n_universe": int(table.get("n_universe") or 0),
        "formula": table.get("formula") or "",
        "as_of": table.get("as_of") or "",
        "version": table.get("version") or "",
    }
