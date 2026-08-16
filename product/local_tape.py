"""Official NSE tape for the desk — index store + bhavcopy, no Yahoo.

The first screen a trader sees must not die because `yfinance` is missing.
Nifty, VIX, sector 1-day moves and advance/decline all come from files we
already keep under logs/indices and logs/bhav.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

_BREADTH_TTL_S = 600.0
_breadth_cache: dict[str, Any] = {}


@dataclass(frozen=True)
class OfficialTape:
    as_of: str = ""
    nifty_close: float = 0.0
    nifty_change_1d: float | None = None
    nifty_change_5d: float | None = None
    vix: float | None = None
    leaders: tuple[str, ...] = ()
    laggards: tuple[str, ...] = ()
    sector_changes: dict[str, float] = field(default_factory=dict)
    breadth: dict[str, Any] = field(default_factory=dict)
    source: str = "official_nse_index+bhavcopy"

    @property
    def usable(self) -> bool:
        return bool(self.as_of) and (
            self.nifty_change_1d is not None or self.vix is not None
        )


def _index_frame(name: str):
    try:
        from data.index_store import TICKER_MAP, get_index_ohlcv
        ticker = next((t for t, n in TICKER_MAP.items() if n == name), "")
        if not ticker:
            return None
        return get_index_ohlcv(ticker)
    except Exception:
        return None


def _pct_change(frame, sessions: int) -> float | None:
    if frame is None or getattr(frame, "empty", True):
        return None
    close = frame["Close"] if "Close" in frame.columns else frame.get("close")
    if close is None or len(close) < sessions + 1:
        return None
    try:
        last = float(close.iloc[-1])
        prev = float(close.iloc[-(sessions + 1)])
        if prev == 0:
            return None
        return round((last / prev - 1.0) * 100.0, 2)
    except Exception:
        return None


def _last_close(frame) -> float:
    if frame is None or getattr(frame, "empty", True):
        return 0.0
    col = "Close" if "Close" in frame.columns else "close"
    try:
        return round(float(frame[col].iloc[-1]), 2)
    except Exception:
        return 0.0


def _as_of(frame) -> str:
    if frame is None or getattr(frame, "empty", True):
        return ""
    idx = frame.index[-1]
    try:
        return idx.date().isoformat()
    except Exception:
        return str(idx)[:10]


def session_breadth(*, min_n: int = 300) -> dict[str, Any]:
    """Advance/decline + % above 50-DMA from the in-memory bhavcopy store."""
    now = time.time()
    cached = _breadth_cache.get("payload")
    ts = float(_breadth_cache.get("ts") or 0)
    if cached and (now - ts) < _BREADTH_TTL_S:
        return dict(cached)

    empty = {
        "n": 0, "advancers": 0, "decliners": 0, "adv_ratio": 0.0,
        "pct_above_50": 0.0, "verdict": "", "line": "",
    }
    try:
        from data.bhavcopy_runtime import ensure_loaded
        from data import bhavcopy_store as store
        ensure_loaded(rebuild_from_local=False)
    except Exception:
        return empty

    adv = dec = n = above50 = have50 = 0
    with store._lock:
        items = list(store._store.items())
    for _sym, df in items:
        if df is None or len(df) < 2 or "close" not in df.columns:
            continue
        try:
            last = float(df["close"].iloc[-1])
            prev = float(df["close"].iloc[-2])
        except Exception:
            continue
        n += 1
        if last > prev:
            adv += 1
        elif last < prev:
            dec += 1
        if len(df) >= 50:
            try:
                sma = float(df["close"].iloc[-50:].mean())
            except Exception:
                continue
            have50 += 1
            if last > sma:
                above50 += 1

    if n < min_n:
        payload = {**empty, "n": n}
        _breadth_cache.update(payload=payload, ts=now)
        return payload

    adv_ratio = round(adv / dec, 2) if dec else 99.0
    p50 = round((above50 / have50) * 100.0, 1) if have50 else 0.0
    if adv_ratio >= 1.2 and p50 >= 55:
        verdict = "HEALTHY"
        line = f"{adv}:{dec} advance/decline · {p50:.0f}% above 50-DMA"
    elif adv_ratio < 0.8 or p50 < 40:
        verdict = "NARROW"
        line = f"{adv}:{dec} advance/decline · {p50:.0f}% above 50-DMA"
    else:
        verdict = "MIXED"
        line = f"{adv}:{dec} advance/decline · {p50:.0f}% above 50-DMA"
    payload = {
        "n": n, "advancers": adv, "decliners": dec, "adv_ratio": adv_ratio,
        "pct_above_50": p50, "verdict": verdict, "line": line,
    }
    _breadth_cache.update(payload=payload, ts=now)
    return payload


def read_official_tape() -> OfficialTape:
    """Nifty / VIX / sectors from the official index store + bhav breadth."""
    nifty = _index_frame("Nifty 50")
    vix_df = _index_frame("India VIX")
    as_of = _as_of(nifty) or _as_of(vix_df)

    sector_names = {
        "IT": "Nifty IT",
        "BANK": "Nifty Bank",
        "AUTO": "Nifty Auto",
        "PHARMA": "Nifty Pharma",
        "FMCG": "Nifty FMCG",
        "METAL": "Nifty Metal",
        "ENERGY": "Nifty Energy",
        "REALTY": "Nifty Realty",
    }
    changes: dict[str, float] = {}
    for label, name in sector_names.items():
        chg = _pct_change(_index_frame(name), 1)
        if chg is not None:
            changes[label] = chg
    ranked = sorted(changes.items(), key=lambda kv: kv[1], reverse=True)
    leaders = tuple(name for name, _ in ranked[:3] if ranked)
    laggards = tuple(name for name, _ in ranked[-3:] if ranked)
    laggards = tuple(x for x in laggards if x not in leaders)

    return OfficialTape(
        as_of=as_of,
        nifty_close=_last_close(nifty),
        nifty_change_1d=_pct_change(nifty, 1),
        nifty_change_5d=_pct_change(nifty, 5),
        vix=_last_close(vix_df) or None,
        leaders=leaders,
        laggards=laggards,
        sector_changes=changes,
        breadth=session_breadth(),
    )
