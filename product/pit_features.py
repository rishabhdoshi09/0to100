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

It also computes four market-context features (``CONTEXT_PRODUCED``), all
read from ONE source session: the latest official session whose bhavcopy
archive is verified on disk AND was published by the decision's own
``generated_at`` (NSE publication convention: 18:00 IST on the session day;
``published_session``). A decision at 15:31 therefore reads the previous
session's context, never that evening's archive, however late it is frozen.

* ``delivery_pct``: the stock's official delivery % on its bar dated exactly
  that session (no bar that day, e.g. a suspension, means missing);
* ``vix``: the official India VIX close dated exactly that session;
* ``index_trend``: official Nifty 50 through the historical curriculum's PIT
  classifier on rows up to that session (200 distinct sessions): close >
  50-DMA > 200-DMA is UP, close < 50-DMA < 200-DMA is DOWN, else FLAT;
* ``breadth_pct_above_50dma``: scan.breadth's definition (close strictly above
  the mean of its last 50 closes, a name needs 51 bars and a bar that
  session, at least 300 names; a bad print in the window counts as not above,
  as there) over the store's stocks, computed ONLY from bars of verified
  archives (a live overlay row is never an input). Ties are decided in exact
  arithmetic, so back-adjustment by a later corporate action cannot flip one.

It also fills ``rs_percentile`` (``CROSS_SECTIONAL_PRODUCED``), on the same
source session: research.sepa's rs_cs_v1 composite (0.40*r63 + 0.20*r126 +
0.20*r189 + 0.20*r252, returns over the name's own verified-archive bars),
ranked market-wide as 100 * (names scoring strictly below) / N over the
breadth universe with a verified bar that session (N >= 300). The shadow
observer in research.feature002 ranks within a scan's cards instead; this is
the market-wide definition. A name whose composite is not finite (a bad print
in a horizon) is missing here, where build_rs_table would rank it at 0;
composites within ``_RS_TIE`` are ties. RS is computed lazily per window of
sessions (``_RS_WINDOW``), and a name's own composite is read from the window
it is ranked in.

The provenance records the source session and this producer's version, and
the challenger scores with the producer it was trained with
(``PRODUCER_VERSION``), so a model version always names one predictor.

They enter the frozen LEARNING record only (evidence_intelligence.
learning_features); the live analog query projects them back out (it reads
the stored vector minus what this module filled). ``regime`` is deliberately
not filled. Not computed here, for want of an official point-in-time producer:
cross-sectional percentiles (rel_strength, rs/sector/volatility), base
geometry, ``correlation_regime``, ``usdinr`` and ``crude_usd``.

Known limit: the store keeps no instrument name or ISIN, so an EQ-series fund
whose symbol carries no ETF/BEES/LIQUID marker cannot be excluded from breadth
point-in-time (the scanner classifies with today's names).
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
#: Which features this module fills, by version. The challenger records the
#: version it was trained under and is scored with exactly that producer.
LEGACY_PRODUCER = "pit_v1_technical"
CONTEXT_PRODUCER = "pit_v2_context"
PRODUCER_VERSION = "pit_v3_rs"
#: Cross-sectional features (one market-wide ranking per source session).
CROSS_SECTIONAL_PRODUCED = ("rs_percentile",)
PRODUCER_FEATURES = {
    LEGACY_PRODUCER: PRODUCED,
    CONTEXT_PRODUCER: PRODUCED + CONTEXT_PRODUCED,
    PRODUCER_VERSION: PRODUCED + CONTEXT_PRODUCED + CROSS_SECTIONAL_PRODUCED,
}
#: research.sepa (rs_cs_v1): composite of bar-count horizon returns.
RS_HORIZONS = (63, 126, 189, 252)
RS_WEIGHTS = (0.40, 0.20, 0.20, 0.20)
#: research.intelligence.data.nse_calendar._PUBLISH_CUTOFF_HM
PUBLISH_CUTOFF = time(18, 0)
#: scan.breadth.MIN_N: fewer names is not the whole market.
BREADTH_MIN_N = 300
_BREADTH_MIN_BARS = 51          # scan.breadth.breadth_from_cache: len(df) >= 51
#: Relative tie band for "close above its 50-DMA". Real differences are at
#: least a tick / 50 (>= ~1e-10 of price even after a 1:100 adjustment); float
#: error of a 50-value mean is ~1e-15. Inside the band is a tie: not above.
_TIE_REL = 1e-12
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
    return {k: round(v, 6) for k, v in ((k, _finite(v)) for k, v in out.items()) if v is not None}


# ── market context ────────────────────────────────────────────────────────────

def _as_ist(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_IST)
    return dt.astimezone(_IST)


def _archive_dates() -> list[date]:
    """Sessions whose official archive is verified on disk (not the settlement calendar)."""
    from data.bhavcopy_store import _dates_on_disk
    return list(_dates_on_disk())


def published_session(decided_at: Any, *, archive_fn: Callable | None = None) -> date | None:
    """Latest verified official session published by ``decided_at`` (and complete by its cutoff)."""
    when = _as_ist(decided_at)
    cut = cutoff_session(decided_at)
    if when is None or cut is None:
        return None
    try:
        days = sorted({d if isinstance(d, date) else date.fromisoformat(str(d)[:10])
                       for d in (archive_fn or _archive_dates)()})
    except Exception:
        return None
    today, now = when.date(), when.time().replace(tzinfo=None)
    eligible = [d for d in days if d <= cut and (d < today or now >= PUBLISH_CUTOFF)]
    return eligible[-1] if eligible else None


def _daily(frame):
    """(dates as datetime64[D], frame) sorted by date, one row per date (the last)."""
    import numpy as np
    import pandas as pd

    idx = pd.DatetimeIndex(frame.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    days = idx.values.astype("datetime64[D]")
    if len(days) > 1 and not bool(np.all(days[1:] > days[:-1])):
        order = np.argsort(days, kind="stable")
        days, frame = days[order], frame.iloc[order]
        keep = np.r_[days[1:] != days[:-1], True]
        days, frame = days[keep], frame.iloc[np.flatnonzero(keep)]
    return days, frame


def _index_loader(ticker: str, as_of: str):
    from data.index_store import get_index_ohlcv_as_of_cached
    return get_index_ohlcv_as_of_cached(ticker, as_of)


def _index_through(frame, session: date):
    """Index rows on/before ``session``, sorted, one per date; None unless the last is ``session``."""
    import numpy as np

    if frame is None or getattr(frame, "empty", True):
        return None
    try:
        days, frame = _daily(frame)
    except Exception:
        return None
    keep = days <= np.datetime64(session, "D")
    if not keep.any() or days[keep][-1] != np.datetime64(session, "D"):
        return None
    return frame.iloc[np.flatnonzero(keep)]


def market_context(session: date | None, *, index_fn: Callable | None = None) -> dict[str, Any]:
    """Official index context (vix, index_trend) dated exactly ``session``. {} when unknown."""
    if session is None:
        return {}
    loader = index_fn or _index_loader
    out: dict[str, Any] = {}
    try:
        vix = _index_through(loader("^INDIAVIX", session.isoformat()), session)
    except Exception:
        vix = None
    if vix is not None:
        cols = {str(c).lower(): c for c in vix.columns}
        value = _finite(vix[cols["close"]].iloc[-1]) if "close" in cols else None
        # India VIX is strictly positive by construction: a 0 is a data fault.
        if value is not None and 0.0 < value <= 200.0:
            out["vix"] = round(value, 6)
    try:
        nifty = _index_through(loader("^NSEI", session.isoformat()), session)
    except Exception:
        nifty = None
    if nifty is not None:
        from research.autonomy.historical_curriculum import classify_session

        state = classify_session(session.isoformat(), index_fn=lambda _t, _d: nifty)
        if state.available:
            out["index_trend"] = _TREND_FROM_REGIME.get(state.regime, "FLAT")
    return out


def delivery_on(frame, session: date | None) -> float | None:
    """Official delivery % on the bar dated exactly ``session``; never a live-overlay row."""
    import numpy as np

    if session is None or frame is None or getattr(frame, "empty", True):
        return None
    cols = {str(c).lower(): c for c in frame.columns}
    if "deliv_per" not in cols:
        return None
    if str(getattr(frame, "attrs", {}).get("quantterm_live_overlay_date") or "") == session.isoformat():
        return None
    try:
        days, frame = _daily(frame)
    except Exception:
        return None
    hit = np.flatnonzero(days == np.datetime64(session, "D"))
    if not hit.size:
        return None
    value = _finite(frame[cols["deliv_per"]].iloc[int(hit[-1])])
    return value if value is not None and 0.0 <= value <= 100.0 else None


def _store_symbols() -> list[str]:
    from data.bhavcopy_store import store_symbols
    return store_symbols()


def _store_ohlcv(symbol: str):
    import data.bhavcopy_store as _bs
    return _bs.get_ohlcv(symbol)


def _store_generation() -> tuple:
    """Identity of the in-memory store: what a cached pass was computed from.

    The store's monotonic revision and the identity of its symbol map catch a
    rebuild that keeps the same last day, session count and size (GPT RS-03).
    """
    try:
        from data import bhavcopy_store as bs
        revision = bs.store_revision() if hasattr(bs, "store_revision") else (None, None)
        return (revision, id(getattr(bs, "_store", None)), getattr(bs, "_store_last_day", None),
                int(getattr(bs, "_store_sessions", 0) or 0), len(getattr(bs, "_store", {}) or {}),
                id(getattr(bs, "_CA_EVENTS", None)))
    except Exception:
        return ("unavailable",)


def _breadth_universe(symbols) -> list[str]:
    """The scanner universe's own stock filters, applied by symbol (see the module's known limit)."""
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
_BREADTH_CACHE: "dict | None" = None
#: The newest verified session is the one a store still installing its
#: archive can be missing; a cached pass that cannot measure it is rechecked
#: at most this often (older sessions are immutable once installed).
_BREADTH_RECHECK_S = 60.0
_CONTEXT_LOCK = threading.Lock()
_CONTEXT_CACHE: dict[tuple, tuple[float, tuple, dict[str, Any]]] = {}
_CONTEXT_TTL_S = 300.0
_CONTEXT_MAX = 64


def _official_closes(df, days) -> tuple | None:
    """(dates, closes, positions in ``days``) of a frame's verified-archive bars.

    The live-overlay row is dropped. One definition for the market pass and
    for a single name, so a name's own score ranks against exactly the same
    numbers.
    """
    import numpy as np

    if df is None or getattr(df, "empty", True):
        return None
    cols = {str(c).lower(): c for c in df.columns}
    if "close" not in cols:
        return None
    try:
        d, df = _daily(df)
        c = df[cols["close"]].to_numpy(dtype=float, na_value=np.nan)
    except Exception:
        return None
    loc = np.searchsorted(days, d)
    official = (loc < len(days)) & (days[np.minimum(loc, len(days) - 1)] == d)
    overlay = str(getattr(df, "attrs", {}).get("quantterm_live_overlay_date") or "")
    if overlay:
        official &= d != np.datetime64(overlay[:10], "D")
    return d[official], c[official], loc[official]


def _rs_scores(c):
    """rs_cs_v1 composite at every bar with 252 earlier bars; NaN where not finite or a price <= 0."""
    import numpy as np

    n = len(c)
    out = np.full(n, np.nan)
    span = RS_HORIZONS[-1]
    if n <= span:
        return out
    last = c[span:]
    acc = np.zeros(n - span)
    ok = np.isfinite(last) & (last > 0)
    for h, w in zip(RS_HORIZONS, RS_WEIGHTS):
        start = c[span - h: n - h]
        ok &= np.isfinite(start) & (start > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            acc = acc + w * (last / start - 1.0)
    out[span:] = np.where(ok, acc, np.nan)
    return out


class _MarketPass(dict):
    """{verified session: (above, measured)} for breadth (``archive``: the sessions it covers)."""
    archive: frozenset = frozenset()


def _market_pass(archive: tuple[date, ...], symbols: list[str], ohlcv_fn: Callable) -> "_MarketPass":
    """One pass over the store for breadth counts (no RS: breadth-only callers stay light, RS-05).

    Each name's series is first reduced to the bars of verified archives (a
    live-overlay row is dropped), then every 50-bar window is averaged on its
    own (no running sum, so no drift) and compared within ``_TIE_REL``.
    """
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view

    days = np.asarray(sorted(archive), dtype="datetime64[D]")
    result = _MarketPass()
    if not len(days):
        return result
    above = np.zeros(len(days), dtype=np.int64)
    valid = np.zeros(len(days), dtype=np.int64)
    for sym in symbols:
        try:
            df = ohlcv_fn(sym)
        except Exception:
            continue
        if df is None or getattr(df, "empty", True) or len(df) < _BREADTH_MIN_BARS:
            continue
        got = _official_closes(df, days)
        if got is None:
            continue
        _d, c, loc = got
        if len(c) < _BREADTH_MIN_BARS:
            continue
        with np.errstate(invalid="ignore"):
            sma = sliding_window_view(c, 50).mean(axis=1)[1:]      # windows ending at bar 51, 52, ...
            last = c[50:]
            up = (last - sma) > _TIE_REL * np.abs(last)            # NaN anywhere -> not above
        at = loc[50:]
        valid += np.bincount(at, minlength=len(days))
        if up.any():
            above += np.bincount(at[up], minlength=len(days))
    for i in np.flatnonzero(valid):
        result[days[i].astype(object)] = (int(above[i]), int(valid[i]))
    return result


#: RS is computed lazily for a window of sessions starting at the one asked
#: for (replay walks forward; a live board asks one session), and at most
#: ``_RS_WINDOWS_MAX`` windows are kept: ~2,000 names x 64 sessions x 12 bytes
#: is ~1.5 MB a window, instead of the whole history.
_RS_WINDOW = 64
_RS_WINDOWS_MAX = 4
#: Ranking tie band: composites closer than this are ties. A later corporate
#: action rescales a name's whole window, which leaves its return exact in
#: arithmetic but not in floating point (RS-01); real composites differ by far
#: more (a tick on a high-priced name moves r_n by ~1e-7).
_RS_TIE = 1e-12
_RS_LOCK = threading.Lock()
_RS_CACHE: "dict[tuple, dict]" = {}


def _rs_window(days, start: int, symbols: tuple, ohlcv_fn: Callable) -> dict:
    """{session: (sorted composites, symbol ids in that order)} for days[start : start + _RS_WINDOW]."""
    import numpy as np

    stop = min(len(days), start + _RS_WINDOW)
    want = days[start:stop]
    day_ids: list = []
    vals: list = []
    sym_ids: list = []
    for sid, sym in enumerate(symbols):
        try:
            df = ohlcv_fn(sym)
        except Exception:
            continue
        got = _official_closes(df, days)
        if got is None:
            continue
        _d, c, loc = got
        if len(c) <= RS_HORIZONS[-1]:
            continue
        inside = (loc >= start) & (loc < stop)
        if not inside.any():
            continue
        scores = _rs_scores(c)
        keep = inside & np.isfinite(scores)
        if keep.any():
            day_ids.append(loc[keep] - start)
            vals.append(scores[keep])
            sym_ids.append(np.full(int(keep.sum()), sid, dtype=np.int32))
    out: dict = {}
    if not vals:
        return out
    all_day = np.concatenate(day_ids)
    all_val = np.concatenate(vals)
    all_sym = np.concatenate(sym_ids)
    order = np.lexsort((all_val, all_day))
    all_day, all_val, all_sym = all_day[order], all_val[order], all_sym[order]
    cuts = np.flatnonzero(np.diff(all_day)) + 1
    for chunk_day, chunk_val, chunk_sym in zip(np.split(all_day, cuts), np.split(all_val, cuts),
                                               np.split(all_sym, cuts)):
        out[want[int(chunk_day[0])].astype(object)] = (chunk_val.copy(), chunk_sym.copy())
    return out


def _breadth_counts(archive: tuple[date, ...], symbols: list[str], ohlcv_fn: Callable) -> dict[date, tuple[int, int]]:
    """{verified session: (names above their 50-DMA, names measured)} (see ``_market_pass``)."""
    return dict(_market_pass(archive, symbols, ohlcv_fn))


def _as_date(value) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value)[:10])


def _pass_entry(
    session: date | None,
    *,
    symbols_fn: Callable | None = None,
    ohlcv_fn: Callable | None = None,
    archive_fn: Callable | None = None,
) -> dict | None:
    """The cached breadth pass covering ``session``, with its archive days and universe.

    The cache is checked on cheap identities of the raw sources (archive
    length and end points, raw symbol count, store generation, injected
    sources) BEFORE any per-symbol work: a board asks once per decision, and
    re-filtering ~3,500 symbols each time cost ~40 ms a decision.
    """
    global _BREADTH_CACHE
    import time as _time
    import numpy as np

    if session is None:
        return None
    try:
        raw_days = list((archive_fn or _archive_dates)())
        raw_syms = list((symbols_fn or _store_symbols)())
        if not raw_days:
            return None
        key = (len(raw_days), _as_date(raw_days[0]), _as_date(raw_days[-1]), len(raw_syms),
               _store_generation(), (symbols_fn, ohlcv_fn, archive_fn))
    except Exception:
        return None
    with _BREADTH_LOCK:
        now = _time.monotonic()
        entry = _BREADTH_CACHE if _BREADTH_CACHE is not None and _BREADTH_CACHE["key"] == key else None
        if entry is not None:
            market, days = entry["market"], entry["days"]
            stale_newest = (market is not None and days.size
                            and np.datetime64(session, "D") == days[-1]
                            and market.get(session, (0, 0))[1] < BREADTH_MIN_N
                            and now - entry["at"] >= _BREADTH_RECHECK_S)
            if not stale_newest:
                return entry if market is not None and session in market.archive else None
        try:
            archive = tuple(sorted({_as_date(d) for d in raw_days}))
        except Exception:
            return None
        days = np.asarray(archive, dtype="datetime64[D]")
        symbols = tuple(_breadth_universe(raw_syms))
        market = None
        if len(symbols) >= BREADTH_MIN_N:
            market = _market_pass(archive, list(symbols), ohlcv_fn or _store_ohlcv)
            market.archive = frozenset(archive)
        if key[4] != _store_generation():
            # The store changed under the pass: it is not one coherent
            # generation, so nobody gets it, not even this caller (RS-03).
            return None
        entry = {"key": key, "market": market, "at": now, "days": days, "symbols": symbols,
                 "index": {sym: i for i, sym in enumerate(symbols)},
                 "ohlcv_fn": ohlcv_fn or _store_ohlcv}
        _BREADTH_CACHE = entry
    if market is None or session not in market.archive:
        return None
    return entry


def breadth_pct_above_50dma(
    session: date | None,
    *,
    symbols_fn: Callable | None = None,
    ohlcv_fn: Callable | None = None,
    archive_fn: Callable | None = None,
) -> float | None:
    """Breadth on the verified session ``session``; None when not measurable."""
    entry = _pass_entry(session, symbols_fn=symbols_fn, ohlcv_fn=ohlcv_fn, archive_fn=archive_fn)
    if entry is None:
        return None
    up, n = entry["market"].get(session, (0, 0))
    if n < BREADTH_MIN_N:
        return None
    return round(up / n * 100.0, 6)


def rs_percentile_on(
    symbol: str,
    session: date | None,
    *,
    symbols_fn: Callable | None = None,
    ohlcv_fn: Callable | None = None,
    archive_fn: Callable | None = None,
) -> float | None:
    """Market-wide rs_cs_v1 percentile of ``symbol`` on the verified session; None when not measurable.

    The name's own composite is read from the same window it is ranked in (by
    symbol id), never recomputed from another copy of its frame (RS-03), and
    composites within ``_RS_TIE`` of it count as ties, not below (RS-01).
    """
    import numpy as np

    sym = str(symbol or "").strip().upper()
    if session is None or not sym:
        return None
    entry = _pass_entry(session, symbols_fn=symbols_fn, ohlcv_fn=ohlcv_fn, archive_fn=archive_fn)
    if entry is None or sym not in entry["index"]:
        return None
    days = entry["days"]
    pos = int(np.searchsorted(days, np.datetime64(session, "D")))
    if pos >= len(days) or days[pos] != np.datetime64(session, "D"):
        return None
    window = None
    with _RS_LOCK:
        for (key, start), cached in _RS_CACHE.items():
            if key == entry["key"] and start <= pos < start + _RS_WINDOW:
                window = cached
                break
    if window is None:
        built = _rs_window(days, pos, entry["symbols"], entry["ohlcv_fn"])
        if _store_generation() != entry["key"][4]:
            return None                       # the store moved under the window (RS-03)
        with _RS_LOCK:
            if len(_RS_CACHE) >= _RS_WINDOWS_MAX:
                _RS_CACHE.pop(next(iter(_RS_CACHE)))
            _RS_CACHE[(entry["key"], pos)] = built
        window = built
    ranked = window.get(session)
    if ranked is None:
        return None
    scores, ids = ranked
    if len(scores) < BREADTH_MIN_N:
        return None
    hit = np.flatnonzero(ids == entry["index"][sym])
    if not hit.size:
        return None
    own = float(scores[int(hit[0])])
    below = int(np.searchsorted(scores, own - _RS_TIE * max(1.0, abs(own)), side="left"))
    return round(100.0 * below / len(scores), 6)


def reset_breadth_cache() -> None:
    global _BREADTH_CACHE
    with _BREADTH_LOCK:
        _BREADTH_CACHE = None
    with _RS_LOCK:
        _RS_CACHE.clear()
    with _CONTEXT_LOCK:
        _CONTEXT_CACHE.clear()


_SHARED_CONTEXT = ("vix", "index_trend", "breadth_pct_above_50dma")


def context_features(decided_at: Any) -> dict[str, Any]:
    """Market context for a decision (official sources, one published session).

    Returns the shared features plus ``_session`` (the source session, ISO) when
    one exists. Only a COMPLETE result is cached, and only while the store
    generation it came from is current, so a source that is still loading is
    re-read on the next decision instead of being frozen as absent.
    """
    import time as _time

    session = published_session(decided_at)
    if session is None:
        return {}
    gen = _store_generation()
    key = (session, gen)
    now = _time.monotonic()
    with _CONTEXT_LOCK:
        hit = _CONTEXT_CACHE.get(key)
    if hit is not None and now - hit[0] < _CONTEXT_TTL_S:
        return dict(hit[2])
    out = dict(market_context(session))
    breadth = breadth_pct_above_50dma(session)
    if breadth is not None:
        out["breadth_pct_above_50dma"] = breadth
    out["_session"] = session.isoformat()
    if all(out.get(k) is not None for k in _SHARED_CONTEXT) and _store_generation() == gen:
        with _CONTEXT_LOCK:
            if len(_CONTEXT_CACHE) >= _CONTEXT_MAX:
                _CONTEXT_CACHE.clear()
            _CONTEXT_CACHE[key] = (now, gen, dict(out))
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
    producer: str = PRODUCER_VERSION,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fill only features the decision did not carry. Returns (features, provenance).

    ``producer`` selects which features may be filled (``PRODUCER_FEATURES``);
    an unknown producer fills nothing. ``context`` lets a caller pass
    precomputed market context (it must carry ``_session``).
    """
    allowed = PRODUCER_FEATURES.get(producer)
    merged = dict(features)
    if allowed is None:
        return merged, {"filled": [], "cutoff_session": None, "producer": producer,
                        "error": "UNKNOWN_PRODUCER"}
    own = [k for k in PRODUCED if k in allowed and _missing(k, merged.get(k))]
    want_delivery = "delivery_pct" in allowed and _missing("delivery_pct", merged.get("delivery_pct"))
    want_rs = "rs_percentile" in allowed and _missing("rs_percentile", merged.get("rs_percentile"))
    shared = [k for k in _SHARED_CONTEXT if k in allowed and _missing(k, merged.get(k))]
    cut = cutoff_session(decided_at)
    prov: dict[str, Any] = {"filled": [], "cutoff_session": None if cut is None else cut.isoformat(),
                            "source": "official_bhavcopy_pit", "producer": producer}
    if not own and not want_delivery and not want_rs and not shared:
        return merged, {"filled": [], "cutoff_session": None, "producer": producer}
    if frame is None and (own or want_delivery or want_rs):
        try:
            frame = _store_ohlcv(str(symbol or "").upper())
        except Exception:
            frame = None
    computed: dict[str, Any] = technical_features(symbol, decided_at, frame=frame) if own and frame is not None else {}
    ctx: dict[str, Any] = {}
    if want_delivery or want_rs or shared:
        try:
            ctx = dict(context) if context is not None else context_features(decided_at)
        except Exception:
            ctx = {}
        session = None
        try:
            session = date.fromisoformat(str(ctx.get("_session") or "")[:10]) if ctx.get("_session") else None
        except ValueError:
            session = None
        if want_delivery and session is not None:
            deliv = delivery_on(frame, session)
            if deliv is not None:
                computed["delivery_pct"] = round(deliv, 6)
        if want_rs and session is not None:
            try:
                rs = ctx.get("rs_percentile") if context is not None and "rs_percentile" in ctx \
                    else rs_percentile_on(symbol, session)
            except Exception:
                rs = None
            if rs is not None:
                computed["rs_percentile"] = rs
        computed.update({k: ctx[k] for k in shared if ctx.get(k) is not None})
        prov["context_session"] = None if session is None else session.isoformat()
        prov["context_source"] = {
            "delivery_pct": "official_bhavcopy_pit",
            "vix": "official_index_pit",
            "index_trend": "official_index_pit:historical_curriculum.classify_session",
            "breadth_pct_above_50dma": "official_bhavcopy_pit:scan.breadth",
            "rs_percentile": "official_bhavcopy_pit:research.sepa.rs_cs_v1:market_wide",
        }
        prov["publication_rule"] = "verified archive published by generated_at (18:00 IST convention)"
    filled = [k for k in own + (["delivery_pct"] if want_delivery else [])
              + (["rs_percentile"] if want_rs else []) + shared if k in computed]
    for k in filled:
        merged[k] = computed[k]
    prov["filled"] = filled
    return merged, prov
