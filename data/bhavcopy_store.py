"""
NSE Bhavcopy Store — official whole-market OHLCV, no Yahoo.

One daily CSV from NSE covers EVERY listed stock. We keep ~260 trading
days of these on disk (logs/bhav/) and build per-symbol OHLCV frames
in memory. First build downloads history once; after that only the
newest day is fetched.

Source: https://nsearchives.nseindia.com/products/content/
        sec_bhavdata_full_DDMMYYYY.csv          (EQ series filtered)

Data is end-of-day official NSE. Live prices come from Google Finance
(data/google_finance.py) and are overlaid on results separately.
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from logger import get_logger

log = get_logger(__name__)

_BHAV_DIR = Path(__file__).resolve().parent.parent / "logs" / "bhav"
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}
_URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d}.csv"

_lock = threading.Lock()
_store: dict[str, pd.DataFrame] = {}     # symbol -> OHLCV df (lowercase cols)
_store_last_day: Optional[date] = None
_MIN_DAYS = 60                            # need at least this much history to be usable


def _day_path(d: date) -> Path:
    return _BHAV_DIR / f"{d.strftime('%d%m%Y')}.csv"


def _download_day(d: date) -> bool:
    """Fetch one bhavcopy to disk. False if unavailable (holiday/404)."""
    path = _day_path(d)
    if path.exists():
        return True
    import requests
    try:
        url = _URL.format(d=d.strftime("%d%m%Y"))
        resp = requests.get(url, headers=_HEADERS, timeout=12)
        if resp.status_code != 200 or len(resp.content) < 1000:
            return False
        _BHAV_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
        return True
    except Exception as exc:
        log.debug("bhav_download_failed", day=str(d), error=str(exc))
        return False


def _read_day(d: date) -> Optional[pd.DataFrame]:
    """Parse one day's file → DataFrame[symbol, open, high, low, close, volume]."""
    path = _day_path(d)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, dtype=str)
        df.columns = [c.strip().upper() for c in df.columns]
        df = df[df["SERIES"].str.strip() == "EQ"]
        out = pd.DataFrame({
            "symbol": df["SYMBOL"].str.strip().str.upper(),
            "open":   pd.to_numeric(df["OPEN_PRICE"], errors="coerce"),
            "high":   pd.to_numeric(df["HIGH_PRICE"], errors="coerce"),
            "low":    pd.to_numeric(df["LOW_PRICE"], errors="coerce"),
            "close":  pd.to_numeric(df["CLOSE_PRICE"], errors="coerce"),
            "volume": pd.to_numeric(df["TTL_TRD_QNTY"], errors="coerce"),
        })
        # Delivery % — institutional accumulation footprint (buy-and-hold vs intraday)
        if "DELIV_PER" in df.columns:
            out["deliv_per"] = pd.to_numeric(
                df["DELIV_PER"].str.strip().replace("-", None), errors="coerce")
        out["date"] = pd.Timestamp(d)
        return out.dropna(subset=["close"])
    except Exception as exc:
        log.debug("bhav_parse_failed", day=str(d), error=str(exc))
        return None


def _trading_days_back(n_days: int) -> list[date]:
    """Candidate weekdays going back far enough to cover n_days sessions."""
    out, d = [], date.today()
    while len(out) < int(n_days * 1.55):
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return out


def build_store(days: int = 260, progress=None) -> int:
    """
    Ensure the in-memory store covers ~`days` sessions. Downloads only
    missing files (first run: full history; later runs: today only).
    Returns the number of symbols available.
    """
    global _store, _store_last_day

    candidates = _trading_days_back(days)

    # Download whatever is missing, newest first, bounded concurrency
    missing = [d for d in candidates if not _day_path(d).exists()]
    if missing:
        log.info("bhav_download_start", missing=len(missing))
        done = 0
        with ThreadPoolExecutor(max_workers=6) as pool:
            futs = {pool.submit(_download_day, d): d for d in missing}
            for fut in as_completed(futs):
                done += 1
                if progress:
                    try:
                        progress(done, len(missing))
                    except Exception:
                        pass

    available = [d for d in candidates if _day_path(d).exists()]
    if len(available) < _MIN_DAYS:
        log.warning("bhav_insufficient_history", have=len(available))
        return 0

    newest = max(available)
    with _lock:
        if _store and _store_last_day == newest:
            return len(_store)   # already built and current

    frames = []
    for d in sorted(available):
        f = _read_day(d)
        if f is not None:
            frames.append(f)
    if not frames:
        return 0

    allday = pd.concat(frames, ignore_index=True)
    keep_cols = ["open", "high", "low", "close", "volume"]
    if "deliv_per" in allday.columns:
        keep_cols.append("deliv_per")
    new_store: dict[str, pd.DataFrame] = {}
    for sym, g in allday.groupby("symbol"):
        g = g.sort_values("date").set_index("date")
        if len(g) >= 30:
            new_store[sym] = g[[c for c in keep_cols if c in g.columns]]

    with _lock:
        _store = new_store
        _store_last_day = newest
    log.info("bhav_store_built", symbols=len(new_store), sessions=len(frames),
             latest=str(newest))
    return len(new_store)


def get_ohlcv(symbol: str) -> Optional[pd.DataFrame]:
    with _lock:
        df = _store.get(symbol.upper())
    return df.copy() if df is not None else None


def store_symbols() -> list[str]:
    with _lock:
        return list(_store.keys())


def is_ready() -> bool:
    with _lock:
        return len(_store) > 0
