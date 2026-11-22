"""Point-in-time technical features from official bars, for the learning record.

The challenger's feature schema (research.feature_schema) names 25 features,
but recommendation cards and saved scan records carry only rsi, volume_ratio
and quality_score (reward_risk is derived). The other features had no producer
anywhere, so every frozen DECISION observation stored them as missing: owner's
Mac 2026-11-18, 21 of 25 challenger features were constant zero after median
imputation, and the model learned from four inputs.

This module computes the features that are defined by a stock's own official
OHLCV, using ONLY bars that existed at decision time:

* decided at or after the 15:30 IST close  -> bars up to the decision date;
* decided before the close                  -> bars up to the previous session.

A missing or short history yields no feature (never an invented value). The
values follow research.feature_schema's definitions.

It also computes four market-context features from official data, under the
same cutoff (``CONTEXT_PRODUCED``):

* ``delivery_pct``: the stock's official bhavcopy delivery % on its last bar
  at or before the cutoff;
* ``vix``: the official India VIX close on the last index row at or before the
  cutoff, at most ``CONTEXT_MAX_AGE_DAYS`` old (the schema's own staleness);
* ``index_trend``: official Nifty 50, from the point-in-time session classifier
  of the historical curriculum: close > 50-DMA > 200-DMA is UP, close < 50-DMA
  < 200-DMA is DOWN, anything else is FLAT;
* ``breadth_pct_above_50dma``: scan.breadth's definition (close above the mean
  of the last 50 closes, a symbol needs 51 bars, at least 300 names) over the
  official store's stocks on that session. ETFs and funds are excluded by the
  scanner universe's own filters. Only sessions whose official archive is
  verified on disk count, so a live intraday overlay can never be frozen as
  breadth.

They enter the frozen LEARNING record only (evidence_intelligence.
learning_features). ``regime`` is deliberately not filled: the live analog
query weights stored rows by their regime, so filling it would change live
evidence. Cross-sectional percentiles (rel_strength, rs/sector/volatility),
base geometry, ``correlation_regime``, ``usdinr`` and ``crude_usd`` are NOT
computed here: they have no official point-in-time producer yet.
"""

from __future__ import annotations

import math
import threading
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Callable, Mapping

_IST = timezone(timedelta(hours=5, minutes=30))
_CLOSE = time(15, 30)
MIN_BARS = 21

PRODUCED = (
    "atr_pct",
    "adr_pct",
    "dist_from_high_pct",
    "clv",
    "volume_z",
    "liquidity_cr",
    "dist_20ema_pct",
    "dist_50dma_pct",
)

#: Market-context features (official sources, same cutoff).
CONTEXT_PRODUCED = (
    "delivery_pct",
    "vix",
    "index_trend",
    "breadth_pct_above_50dma",
)
#: research.feature_schema gives vix max_age_days=3; one rule for all context.
CONTEXT_MAX_AGE_DAYS = 3
#: scan.breadth.MIN_N: fewer names is not the whole market.
BREADTH_MIN_N = 300
_BREADTH_MIN_BARS = 51          # scan.breadth.breadth_from_cache: len(df) >= 51
_CATEGORICAL_UNKNOWN = {"index_trend"}
_TREND_FROM_REGIME = {"BULL_TREND": "UP", "BEAR": "DOWN"}


def cutoff_session(decided_at: Any) -> date | None:
    """Last session whose bar was complete when the decision was made."""
    text = str(decided_at or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_IST)  # naive timestamps in this codebase are IST
    local = dt.astimezone(_IST)
    return local.date() if local.time() >= _CLOSE else local.date() - timedelta(days=1)


def _finite(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def technical_features(symbol: str, decided_at: Any, frame=None) -> dict[str, float]:
    """PIT features for ``symbol`` as of ``decided_at``. {} when not computable."""
    cut = cutoff_session(decided_at)
    if cut is None:
        return {}
    if frame is None:
        try:
            from data.bhavcopy_store import get_ohlcv
            frame = get_ohlcv(str(symbol or "").upper())
        except Exception:
            return {}
    if frame is None or getattr(frame, "empty", True):
        return {}
    import pandas as pd

    cols = {c.lower(): c for c in frame.columns}
    if not {"high", "low", "close"} <= set(cols):
        return {}
    cutoff = pd.Timestamp(cut)
    tz = getattr(frame.index, "tz", None)
    if tz is not None:
        cutoff = cutoff.tz_localize(tz)
    df = frame[frame.index <= cutoff + pd.Timedelta(hours=23, minutes=59)]
    if len(df) < MIN_BARS:
        return {}
    close = df[cols["close"]].astype(float)
    high = df[cols["high"]].astype(float)
    low = df[cols["low"]].astype(float)
    volume = df[cols["volume"]].astype(float) if "volume" in cols else None
    last_c, last_h, last_l = float(close.iloc[-1]), float(high.iloc[-1]), float(low.iloc[-1])
    if last_c <= 0:
        return {}
    out: dict[str, Any] = {}
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    out["atr_pct"] = float(true_range.iloc[-14:].mean()) / last_c * 100.0
    out["adr_pct"] = float(((high - low) / close).iloc[-20:].mean()) * 100.0
    hi = float(high.iloc[-252:].max())
    if hi > 0:
        out["dist_from_high_pct"] = max(0.0, (hi - last_c) / hi * 100.0)
    rng = last_h - last_l
    out["clv"] = ((last_c - last_l) - (last_h - last_c)) / rng if rng > 0 else 0.0
    if volume is not None:
        hist = volume.iloc[-21:-1]
        sd = float(hist.std(ddof=1)) if len(hist) > 1 else 0.0
        if sd > 0:
            out["volume_z"] = (float(volume.iloc[-1]) - float(hist.mean())) / sd
        turnover = (close * volume).iloc[-20:]
        out["liquidity_cr"] = float(turnover.median()) / 1e7
    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    if ema20 > 0:
        out["dist_20ema_pct"] = (last_c / ema20 - 1.0) * 100.0
    if len(close) >= 50:
        sma50 = float(close.iloc[-50:].mean())
        if sma50 > 0:
            out["dist_50dma_pct"] = (last_c / sma50 - 1.0) * 100.0
    if "deliv_per" in cols:
        deliv = _finite(df[cols["deliv_per"]].iloc[-1])
        if deliv is not None and 0.0 <= deliv <= 100.0:
            out["delivery_pct"] = deliv
    return {k: round(v, 6) for k, v in ((k, _finite(v)) for k, v in out.items()) if v is not None}


# ── market context ────────────────────────────────────────────────────────────

def _index_loader(ticker: str, as_of: str):
    from data.index_store import get_index_ohlcv_as_of_cached
    return get_index_ohlcv_as_of_cached(ticker, as_of)


def _rows_through(frame, cut: date):
    """Rows dated on/before ``cut``; never trusts a loader to have sliced."""
    import pandas as pd

    if frame is None or getattr(frame, "empty", True):
        return None
    try:
        days = pd.DatetimeIndex(frame.index)
        if days.tz is not None:
            days = days.tz_localize(None)
        kept = frame.loc[days.normalize() <= pd.Timestamp(cut)]
    except Exception:
        return None
    return kept if len(kept) else None


def _row_date(frame) -> date | None:
    try:
        return frame.index[-1].date()
    except Exception:
        return None


def market_context(cut: date | None, *, index_fn: Callable | None = None) -> dict[str, Any]:
    """Official index context (vix, index_trend) as of ``cut``. {} when unknown."""
    if cut is None:
        return {}
    loader = index_fn or _index_loader
    out: dict[str, Any] = {}
    try:
        vix = _rows_through(loader("^INDIAVIX", cut.isoformat()), cut)
    except Exception:
        vix = None
    if vix is not None:
        cols = {c.lower(): c for c in vix.columns}
        day = _row_date(vix)
        value = _finite(vix[cols["close"]].iloc[-1]) if "close" in cols else None
        if day is not None and (cut - day).days <= CONTEXT_MAX_AGE_DAYS and value is not None \
                and 0.0 < value <= 200.0:
            out["vix"] = round(value, 6)
    try:
        nifty = _rows_through(loader("^NSEI", cut.isoformat()), cut)
    except Exception:
        nifty = None
    day = _row_date(nifty) if nifty is not None else None
    if day is not None and (cut - day).days <= CONTEXT_MAX_AGE_DAYS:
        from research.autonomy.historical_curriculum import classify_session

        state = classify_session(cut.isoformat(), index_fn=lambda _t, _d: nifty)
        if state.available:
            out["index_trend"] = _TREND_FROM_REGIME.get(state.regime, "FLAT")
    return out


def _store_symbols() -> list[str]:
    from data.bhavcopy_store import store_symbols
    return store_symbols()


def _store_ohlcv(symbol: str):
    from data.bhavcopy_store import get_ohlcv
    return get_ohlcv(symbol)


def _official_dates() -> list[str]:
    from core.session_window import official_dates
    return official_dates()


def _breadth_universe(symbols) -> list[str]:
    """The scanner universe's own stock filters, applied by symbol."""
    try:
        from data.nse_universe import _is_non_stock_fund, _is_valid_symbol
    except Exception:
        return []
    out = []
    for sym in symbols or []:
        s = str(sym or "").strip().upper()
        if s and _is_valid_symbol(s) and not _is_non_stock_fund(s):
            out.append(s)
    return out


_BREADTH_LOCK = threading.Lock()
_BREADTH_CACHE: tuple[tuple, dict[date, tuple[int, int]]] | None = None
_CONTEXT_LOCK = threading.Lock()
_CONTEXT_CACHE: dict[date, tuple[float, dict[str, Any]]] = {}


def _breadth_counts(official: tuple[date, ...], symbols: list[str], ohlcv_fn: Callable) -> dict[date, tuple[int, int]]:
    """{official session: (names above their 50-DMA, names measured)}.

    One vectorised pass per name: a 50-bar mean ending on each bar (all 50
    closes finite, as pandas rolling(50) requires), measured from the 51st bar,
    counted only on official sessions.
    """
    import numpy as np
    import pandas as pd

    days = np.asarray(sorted(official), dtype="datetime64[D]")
    above = np.zeros(len(days), dtype=np.int64)
    valid = np.zeros(len(days), dtype=np.int64)
    if not len(days):
        return {}
    for sym in symbols:
        try:
            df = ohlcv_fn(sym)
        except Exception:
            continue
        if df is None or getattr(df, "empty", True) or len(df) < _BREADTH_MIN_BARS:
            continue
        cols = {c.lower(): c for c in df.columns}
        if "close" not in cols:
            continue
        try:
            idx = pd.DatetimeIndex(df.index)
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            d = idx.normalize().values.astype("datetime64[D]")
            c = pd.to_numeric(df[cols["close"]], errors="coerce").to_numpy(dtype=float)
        except Exception:
            continue
        if len(d) > 1 and not bool(np.all(d[1:] > d[:-1])):
            order = np.argsort(d, kind="stable")
            d, c = d[order], c[order]
            keep = np.r_[d[1:] != d[:-1], True]          # a duplicated day keeps its last bar
            d, c = d[keep], c[keep]
        n = len(c)
        if n < _BREADTH_MIN_BARS:
            continue
        finite = np.isfinite(c)
        csum = np.cumsum(np.where(finite, c, 0.0))
        ccount = np.cumsum(finite.astype(np.int64))
        win_sum = csum[49:] - np.r_[0.0, csum[:-50]]
        win_n = ccount[49:] - np.r_[0, ccount[:-50]]
        with np.errstate(invalid="ignore"):
            sma = np.where(win_n == 50, win_sum / 50.0, np.nan)
        last = c[49:]
        ok = (np.arange(49, n) >= _BREADTH_MIN_BARS - 1) & np.isfinite(sma) & np.isfinite(last)
        if not ok.any():
            continue
        when = d[49:][ok]
        up = (last > sma)[ok]
        loc = np.searchsorted(days, when)
        inside = loc < len(days)
        inside[inside] = days[loc[inside]] == when[inside]
        np.add.at(valid, loc[inside], 1)
        np.add.at(above, loc[inside][up[inside]], 1)
    return {days[i].astype(object): (int(above[i]), int(valid[i])) for i in np.flatnonzero(valid)}


def breadth_pct_above_50dma(
    cut: date | None,
    *,
    symbols_fn: Callable | None = None,
    ohlcv_fn: Callable | None = None,
    official_fn: Callable | None = None,
) -> float | None:
    """Official-session breadth as of ``cut``; None when not measurable."""
    global _BREADTH_CACHE
    if cut is None:
        return None
    try:
        official = tuple(sorted(date.fromisoformat(str(d)[:10]) for d in (official_fn or _official_dates)()))
    except Exception:
        return None
    usable = [d for d in official if d <= cut]
    if not usable or (cut - usable[-1]).days > CONTEXT_MAX_AGE_DAYS:
        return None
    try:
        symbols = _breadth_universe((symbols_fn or _store_symbols)())
    except Exception:
        return None
    if len(symbols) < BREADTH_MIN_N:
        return None
    # Injected sources (tests, research) are part of the key by identity; the
    # production key is the official archive state and the universe size.
    key = (official[-1], len(official), len(symbols), symbols_fn, ohlcv_fn, official_fn)
    with _BREADTH_LOCK:
        hit = _BREADTH_CACHE if _BREADTH_CACHE is not None and _BREADTH_CACHE[0] == key else None
        if hit is None:
            counts = _breadth_counts(official, symbols, ohlcv_fn or _store_ohlcv)
            _BREADTH_CACHE = (key, counts)
        else:
            counts = hit[1]
    up, n = counts.get(usable[-1], (0, 0))
    if n < BREADTH_MIN_N:
        return None
    return round(up / n * 100.0, 6)


def reset_breadth_cache() -> None:
    global _BREADTH_CACHE
    with _BREADTH_LOCK:
        _BREADTH_CACHE = None
    with _CONTEXT_LOCK:
        _CONTEXT_CACHE.clear()


#: A board enriches every card with the same cutoff; the official index can
#: still land later in the day, so a cached read expires.
_CONTEXT_TTL_S = 300.0


def context_features(decided_at: Any) -> dict[str, Any]:
    """Market context shared by every decision with this cutoff (official sources)."""
    import time as _time

    cut = cutoff_session(decided_at)
    if cut is None:
        return {}
    now = _time.monotonic()
    with _CONTEXT_LOCK:
        hit = _CONTEXT_CACHE.get(cut)
    if hit is not None and now - hit[0] < _CONTEXT_TTL_S:
        return dict(hit[1])
    out = dict(market_context(cut))
    breadth = breadth_pct_above_50dma(cut)
    if breadth is not None:
        out["breadth_pct_above_50dma"] = breadth
    with _CONTEXT_LOCK:
        if len(_CONTEXT_CACHE) > 64:
            _CONTEXT_CACHE.clear()
        _CONTEXT_CACHE[cut] = (now, dict(out))
    return out


def _missing(name: str, value: Any) -> bool:
    if value is None:
        return True
    return name in _CATEGORICAL_UNKNOWN and str(value).strip().upper() in {"", "UNKNOWN"}


def fill_missing(
    features: Mapping[str, Any],
    symbol: str,
    decided_at: Any,
    frame=None,
    *,
    context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fill only features the decision did not carry. Returns (features, provenance).

    ``context`` lets a caller pass precomputed market context; by default it
    is read from official sources (``context_features``).
    """
    merged = dict(features)
    own = [k for k in PRODUCED + ("delivery_pct",) if _missing(k, merged.get(k))]
    shared = [k for k in CONTEXT_PRODUCED if k != "delivery_pct" and _missing(k, merged.get(k))]
    if not own and not shared:
        return merged, {"filled": [], "cutoff_session": None}
    computed = technical_features(symbol, decided_at, frame=frame) if own else {}
    if shared:
        try:
            ctx = dict(context) if context is not None else context_features(decided_at)
        except Exception:
            ctx = {}
        computed.update({k: ctx[k] for k in shared if ctx.get(k) is not None})
    filled = [k for k in own + shared if k in computed]
    for k in filled:
        merged[k] = computed[k]
    cut = cutoff_session(decided_at)
    return merged, {"filled": filled, "cutoff_session": None if cut is None else cut.isoformat(),
                    "source": "official_bhavcopy_pit",
                    "context_source": {
                        "delivery_pct": "official_bhavcopy_pit",
                        "vix": "official_index_pit",
                        "index_trend": "official_index_pit:historical_curriculum.classify_session",
                        "breadth_pct_above_50dma": "official_bhavcopy_pit:scan.breadth",
                    }}
