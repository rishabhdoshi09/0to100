"""Market-monitor breadth + on-file news — official tape, no scrape.

Reco-style board: Nifty, advance/decline, % of names above the 20- and 40-day
averages, a short session table, and a news strip from the curator SQLite
(already fetched). Missing files stay empty. Not a QuantTerm backtest edge.
"""
from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

_TTL_S = 600.0
_cache: dict[str, Any] = {}
MIN_N = 300
SESSIONS = 8
MOVE_PCT = 4.0
BREADTH_NOTE = (
    "Advance/decline and % above 20-/40-day averages from official NSE bhavcopy. "
    "Nifty closes from the index store. Context for the tape — not a buy signal."
)
NEWS_NOTE = (
    "Headlines from the on-file news curator. POLICY/TARIFF tags are keyword "
    "lenses, not forecasts. Empty means nothing is on disk yet."
)


def _f(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        if out != out:
            return None
        return out
    except (TypeError, ValueError):
        return None


def _close_array(frame: Any):
    if frame is None or len(frame) == 0:
        return None, None
    data = frame
    try:
        data = frame.sort_index()
    except Exception:
        pass
    col = None
    for name in ("close", "Close"):
        if name in data.columns:
            col = name
            break
    if col is None:
        return None, None
    try:
        series = data[col].astype(float).dropna()
    except Exception:
        return None, None
    if len(series) < 2:
        return None, None
    return series.to_numpy(dtype=float), series.index


def _sma_ending(values, window: int, offset: int) -> float | None:
    end = len(values) - offset
    start = end - window
    if start < 0 or end <= start:
        return None
    chunk = values[start:end]
    if len(chunk) < window:
        return None
    return float(chunk.mean())


def _verdict(adv_ratio: float, pct_above_50: float) -> tuple[str, str]:
    """Same HEALTHY/MIXED/NARROW gates as the 50-DMA breadth read."""
    if adv_ratio >= 1.2 and pct_above_50 >= 55:
        return "HEALTHY", "Internals support longs — breadth is not a one-stock tape."
    if adv_ratio < 0.8 or pct_above_50 < 40:
        return "NARROW", "Internals are weak — index print can hide a thin tape."
    return "MIXED", "Internals are mixed — stay selective."


def breadth_from_closes(
    series: Sequence[tuple[Any, Any]],
    *,
    nifty_close: Any = None,
    nifty_index: Any = None,
    sessions: int = SESSIONS,
    min_n: int = MIN_N,
) -> dict[str, Any]:
    """Pure breadth over close arrays. ``series`` is [(values, index), ...]."""
    n_days = max(1, int(sessions))
    buckets = [
        {
            "adv": 0, "dec": 0, "flat": 0, "n": 0,
            "up4": 0, "down4": 0,
            "have20": 0, "above20": 0,
            "have40": 0, "above40": 0,
            "have50": 0, "above50": 0,
        }
        for _ in range(n_days)
    ]
    for values, _idx in series:
        if values is None or len(values) < 2:
            continue
        for offset in range(n_days):
            i = len(values) - 1 - offset
            if i < 1:
                continue
            last = float(values[i])
            prev = float(values[i - 1])
            if prev <= 0 or last != last or prev != prev:
                continue
            bucket = buckets[offset]
            bucket["n"] += 1
            chg = (last / prev - 1.0) * 100.0
            if last > prev:
                bucket["adv"] += 1
            elif last < prev:
                bucket["dec"] += 1
            else:
                bucket["flat"] += 1
            if chg >= MOVE_PCT:
                bucket["up4"] += 1
            elif chg <= -MOVE_PCT:
                bucket["down4"] += 1
            sma20 = _sma_ending(values, 20, offset)
            if sma20 is not None:
                bucket["have20"] += 1
                if last > sma20:
                    bucket["above20"] += 1
            sma40 = _sma_ending(values, 40, offset)
            if sma40 is not None:
                bucket["have40"] += 1
                if last > sma40:
                    bucket["above40"] += 1
            sma50 = _sma_ending(values, 50, offset)
            if sma50 is not None:
                bucket["have50"] += 1
                if last > sma50:
                    bucket["above50"] += 1

    history: list[dict[str, Any]] = []
    for offset, bucket in enumerate(buckets):
        n = int(bucket["n"])
        adv = int(bucket["adv"])
        dec = int(bucket["dec"])
        if n < min_n:
            continue
        adv_ratio = round(adv / dec, 2) if dec else 99.0
        p20 = round(bucket["above20"] / bucket["have20"] * 100.0, 1) if bucket["have20"] else None
        p40 = round(bucket["above40"] / bucket["have40"] * 100.0, 1) if bucket["have40"] else None
        p50 = round(bucket["above50"] / bucket["have50"] * 100.0, 1) if bucket["have50"] else None
        nifty = None
        day = ""
        if nifty_close is not None and len(nifty_close) > offset:
            try:
                nifty = round(float(nifty_close[-(offset + 1)]), 2)
            except (TypeError, ValueError):
                nifty = None
        if nifty_index is not None and len(nifty_index) > offset:
            try:
                stamp = nifty_index[-(offset + 1)]
                day = stamp.date().isoformat() if hasattr(stamp, "date") else str(stamp)[:10]
            except Exception:
                day = ""
        history.append({
            "offset": offset,
            "date": day,
            "n": n,
            "advancers": adv,
            "decliners": dec,
            "unchanged": int(bucket["flat"]),
            "adv_ratio": adv_ratio,
            "up_4pct": int(bucket["up4"]),
            "down_4pct": int(bucket["down4"]),
            "pct_above_20": p20,
            "pct_above_40": p40,
            "pct_above_50": p50,
            "nifty_close": nifty,
        })

    if not history:
        return {
            "available": False,
            "n": 0,
            "advancers": 0,
            "decliners": 0,
            "adv_ratio": None,
            "pct_above_20": None,
            "pct_above_40": None,
            "pct_above_50": None,
            "up_4pct": 0,
            "down_4pct": 0,
            "verdict": "",
            "line": "",
            "history": [],
            "note": BREADTH_NOTE,
            "source": "nse_bhavcopy",
        }

    today = history[0]
    p50 = float(today.get("pct_above_50") or 0.0)
    verdict, line = _verdict(float(today["adv_ratio"]), p50)
    return {
        "available": True,
        "n": today["n"],
        "advancers": today["advancers"],
        "decliners": today["decliners"],
        "adv_ratio": today["adv_ratio"],
        "pct_above_20": today["pct_above_20"],
        "pct_above_40": today["pct_above_40"],
        "pct_above_50": today["pct_above_50"],
        "up_4pct": today["up_4pct"],
        "down_4pct": today["down_4pct"],
        "verdict": verdict,
        "line": (
            f"{today['advancers']}:{today['decliners']} adv/decl · "
            f"{today['pct_above_20'] if today['pct_above_20'] is not None else '—'}% above 20-DMA · "
            f"{today['pct_above_40'] if today['pct_above_40'] is not None else '—'}% above 40-DMA"
        ),
        "advice": line,
        "history": history,
        "note": BREADTH_NOTE,
        "source": "nse_bhavcopy+index_store",
    }


def _store_series() -> list[tuple[Any, Any]]:
    out: list[tuple[Any, Any]] = []
    try:
        from data.bhavcopy_runtime import ensure_loaded
        from data import bhavcopy_store as store
        ensure_loaded(rebuild_from_local=False)
        with store._lock:
            items = list(store._store.items())
    except Exception:
        return out
    for _sym, frame in items:
        values, idx = _close_array(frame)
        if values is None:
            continue
        out.append((values, idx))
    return out


def market_breadth(*, sessions: int = SESSIONS, min_n: int = MIN_N) -> dict[str, Any]:
    """Cached breadth over the official bhavcopy store."""
    now = time.time()
    hit = _cache.get("breadth")
    ts = float(_cache.get("ts") or 0)
    if hit and (now - ts) < _TTL_S:
        return dict(hit)
    nifty_vals = nifty_idx = None
    try:
        from data.index_store import get_index_ohlcv
        frame = get_index_ohlcv("^NSEI")
        nifty_vals, nifty_idx = _close_array(frame)
    except Exception:
        nifty_vals = nifty_idx = None
    payload = breadth_from_closes(
        _store_series(),
        nifty_close=nifty_vals,
        nifty_index=nifty_idx,
        sessions=sessions,
        min_n=min_n,
    )
    _cache.update(breadth=payload, ts=now)
    return dict(payload)


def news_tape(*, hours: int = 48, limit: int = 10) -> dict[str, Any]:
    """On-file curator headlines only — never fetches."""
    empty = {
        "available": False,
        "items": [],
        "note": NEWS_NOTE,
        "source": "news_curator",
    }
    try:
        from pathlib import Path
        from news.curator_store import NewsCuratorStore
        from product.education_feed import _canon_lens
        root = Path(__file__).resolve().parents[1]
        path = root / "logs" / "news_curator.sqlite3"
        if not path.exists():
            return empty
        store = NewsCuratorStore(path)
        articles = store.recent(hours=hours, limit=limit, min_impact=20)
    except Exception:
        return empty
    items: list[dict[str, Any]] = []
    for article in articles:
        mapping: Mapping[str, Any]
        try:
            mapping = {
                "headline": article.headline,
                "summary": article.summary,
                "source": article.source,
                "category": article.category,
                "event_type": article.event_type,
                "tags": list(article.tags or ()),
                "mentioned_symbols": list(article.mentioned_symbols or ()),
            }
            lens = _canon_lens(mapping)
        except Exception:
            lens = str(getattr(article, "category", "") or "MACRO").upper()
        headline = str(getattr(article, "headline", "") or "").strip()
        if not headline:
            continue
        text = " ".join([headline, str(getattr(article, "summary", "") or "")]).lower()
        if "tariff" in text or "trade war" in text or "import duty" in text:
            lens = "TARIFF"
        items.append({
            "headline": headline,
            "source": str(getattr(article, "source", "") or ""),
            "tag": lens,
            "url": str(getattr(article, "url", "") or ""),
            "published_at": str(getattr(article, "published_at", "") or ""),
        })
        if len(items) >= limit:
            break
    return {
        "available": bool(items),
        "items": items,
        "note": NEWS_NOTE,
        "source": "news_curator",
    }
