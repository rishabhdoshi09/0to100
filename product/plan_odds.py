"""Probabilistic view of a trade plan from the stock's own behaviour.

The owner asked for probabilistic views on recommendations (2026-11-12). The
evidence-based win probability on a card stays UNPROVEN (50%, 20-80% band)
until enough comparable outcomes settle, which says nothing about the plan
itself. This module answers a narrower, honest question:

    Given how THIS stock has actually moved over the last year (its own daily
    close-to-close, high and low excursions), with NO edge assumed (drift
    removed), what are the odds that this entry/stop/target plan hits its
    target before its stop within the horizon -- and where is price likely to
    be after 5/10/15 sessions?

It is a no-edge baseline, never a forecast. A setup has an edge only when its
evidence-based win rate beats ``p_target_first`` (and the breakeven rate).
Paths start at the entry (assumed filled). A day that touches both levels
counts as stop first, matching the canonical first-touch resolver.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

HORIZON_SESSIONS = 15
LOOKBACK_SESSIONS = 250
N_PATHS = 2000
MIN_SESSIONS = 120
CONE_DAYS = (5, 10, 15)

METHOD = (
    "No-edge baseline: 2,000 paths resampled from this stock's own last ~250 "
    "daily moves (close, high, low) with the drift removed; paths start at the "
    "entry; a day touching both levels counts as stop first. Not a forecast."
)


def _day_moves(bars) -> np.ndarray | None:
    """(n, 3) log moves vs the prior close: close, high, low."""
    try:
        cols = {str(c).lower(): c for c in bars.columns}
        close = bars[cols["close"]].to_numpy(dtype=float)
        high = bars[cols["high"]].to_numpy(dtype=float)
        low = bars[cols["low"]].to_numpy(dtype=float)
    except Exception:
        return None
    prev = close[:-1]
    ok = (prev > 0) & (close[1:] > 0) & (high[1:] > 0) & (low[1:] > 0)
    if int(ok.sum()) < MIN_SESSIONS:
        return None
    rc = np.log(close[1:][ok] / prev[ok])
    rh = np.log(high[1:][ok] / prev[ok])
    rl = np.log(low[1:][ok] / prev[ok])
    moves = np.column_stack([rc, np.maximum(rh, rc), np.minimum(rl, rc)])
    return moves[-LOOKBACK_SESSIONS:]


def plan_odds(
    symbol: str,
    entry: float,
    stop: float,
    target: float,
    *,
    bars=None,
    horizon: int = HORIZON_SESSIONS,
    n_paths: int = N_PATHS,
) -> dict[str, Any] | None:
    """Odds for a LONG plan (stop < entry < target). None when not computable."""
    try:
        entry, stop, target = float(entry), float(stop), float(target)
    except (TypeError, ValueError):
        return None
    if not (0 < stop < entry < target):
        return None
    if bars is None:
        try:
            from data.bhavcopy_store import get_ohlcv
            bars = get_ohlcv(str(symbol or "").upper())
        except Exception:
            bars = None
    if bars is None or len(bars) < MIN_SESSIONS + 1:
        return None
    moves = _day_moves(bars)
    if moves is None:
        return None
    moves = moves - np.array([moves[:, 0].mean()] * 3)  # remove drift: no edge assumed

    try:
        as_of = str(bars.index[-1])[:10]
    except Exception:
        as_of = ""
    seed = int(hashlib.sha256(f"{symbol}|{as_of}|{entry}|{stop}|{target}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(moves), size=(int(n_paths), int(horizon)))
    day = moves[pick]                                  # (paths, horizon, 3)
    close_path = np.cumsum(day[:, :, 0], axis=1)
    prior = np.concatenate([np.zeros((int(n_paths), 1)), close_path[:, :-1]], axis=1)
    high_path = prior + day[:, :, 1]
    low_path = prior + day[:, :, 2]

    ls, lt = np.log(stop / entry), np.log(target / entry)
    never = int(horizon) + 1
    hit_stop = np.where((low_path <= ls).any(axis=1), (low_path <= ls).argmax(axis=1), never)
    hit_target = np.where((high_path >= lt).any(axis=1), (high_path >= lt).argmax(axis=1), never)
    target_first = hit_target < hit_stop               # same-day touch -> stop first
    stop_first = (hit_stop <= hit_target) & (hit_stop < never)
    neither = ~(target_first | stop_first)

    risk = entry - stop
    reward_r = (target - entry) / risk
    final_px = entry * np.exp(close_path[:, -1])
    r_outcome = np.where(target_first, reward_r, np.where(stop_first, -1.0, (final_px - entry) / risk))

    def _pct(x: float) -> float:
        return round(float(x) * 100.0, 1)

    cone = []
    for d in CONE_DAYS:
        if d <= int(horizon):
            px = np.exp(close_path[:, d - 1])
            p10, p50, p90 = np.percentile(px, [10, 50, 90])
            cone.append({"sessions": d, "p10_pct": _pct(p10 - 1), "p50_pct": _pct(p50 - 1), "p90_pct": _pct(p90 - 1)})

    daily_vol = float(np.std(moves[:, 0]))
    return {
        "symbol": str(symbol or "").upper(),
        "as_of": as_of,
        "horizon_sessions": int(horizon),
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "reward_risk": round(reward_r, 2),
        "p_target_first": round(float(target_first.mean()), 3),
        "p_stop_first": round(float(stop_first.mean()), 3),
        "p_neither": round(float(neither.mean()), 3),
        "breakeven_win_rate": round(1.0 / (1.0 + reward_r), 3),
        "no_edge_expected_r": round(float(r_outcome.mean()), 2),
        "daily_vol_pct": round(daily_vol * 100.0, 2),
        "cone": cone,
        "sample_sessions": int(len(moves)),
        "paths": int(n_paths),
        "kind": "NO_EDGE_BASELINE",
        "method": METHOD,
    }
