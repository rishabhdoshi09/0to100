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
_store_sessions: int = 0                  # how many sessions the store covers
_MIN_DAYS = 60                            # need at least this much history to be usable
_PKL = _BHAV_DIR / "store_cache.pkl"      # consolidated cache — skip re-parsing CSVs
DEFAULT_DAYS = 500                        # ~2 years: doubles backtest evidence


def _day_path(d: date) -> Path:
    return _BHAV_DIR / f"{d.strftime('%d%m%Y')}.csv"


def _download_day(d: date, retries: int = 2) -> bool:
    """
    Fetch one bhavcopy to disk. 404 = holiday (no retry). Network errors
    retry with backoff so a flaky connection can't punch holes in history.
    """
    path = _day_path(d)
    if path.exists():
        return True
    import requests
    url = _URL.format(d=d.strftime("%d%m%Y"))
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=12)
            if resp.status_code == 404:
                return False          # holiday/weekend — retrying won't help
            if resp.status_code == 200 and len(resp.content) >= 1000:
                _BHAV_DIR.mkdir(parents=True, exist_ok=True)
                path.write_bytes(resp.content)
                return True
        except Exception as exc:
            if attempt == retries:
                log.debug("bhav_download_failed", day=str(d), error=str(exc))
        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))
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


def _save_pkl() -> None:
    """Persist the built store so restarts load in seconds, not minutes."""
    try:
        import pickle
        with _lock:
            data = {"store": _store, "last_day": _store_last_day,
                    "sessions": _store_sessions}
        _PKL.parent.mkdir(parents=True, exist_ok=True)
        tmp = _PKL.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(_PKL)                     # atomic — no half-written cache
    except Exception as exc:
        log.debug("bhav_pkl_save_failed", error=str(exc))


def _load_pkl() -> bool:
    """Load the consolidated cache. False if absent/corrupt."""
    global _store, _store_last_day, _store_sessions
    try:
        import pickle
        if not _PKL.exists():
            return False
        with open(_PKL, "rb") as f:
            data = pickle.load(f)
        if not data.get("store"):
            return False
        with _lock:
            _store = data["store"]
            _store_last_day = data["last_day"]
            _store_sessions = int(data.get("sessions") or 0)
        log.info("bhav_store_loaded_from_cache", symbols=len(_store),
                 sessions=_store_sessions, latest=str(_store_last_day))
        return True
    except Exception as exc:
        log.debug("bhav_pkl_load_failed", error=str(exc))
        try:
            _PKL.unlink()                     # corrupt cache — remove it
        except Exception:
            pass
        return False


def _full_build(available: list[date]) -> int:
    global _store, _store_last_day, _store_sessions
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
        _store_last_day = max(available)
        _store_sessions = len(frames)
    log.info("bhav_store_built", symbols=len(new_store), sessions=len(frames),
             latest=str(max(available)))
    _save_pkl()
    return len(new_store)


def _incremental_append(new_days: list[date]) -> None:
    """Append only the fresh sessions onto the loaded store."""
    global _store_last_day, _store_sessions
    frames = [f for d in sorted(new_days)
              if (f := _read_day(d)) is not None]
    if not frames:
        return
    allday = pd.concat(frames, ignore_index=True)
    with _lock:
        for sym, g in allday.groupby("symbol"):
            g = g.sort_values("date").set_index("date")
            cols = [c for c in ("open", "high", "low", "close", "volume",
                                "deliv_per") if c in g.columns]
            old = _store.get(sym)
            if old is not None:
                merged = pd.concat([old, g[[c for c in cols if c in old.columns
                                            or c in g.columns]]])
                _store[sym] = merged[~merged.index.duplicated(keep="last")]
            elif len(g) >= 1:
                _store[sym] = g[cols]
        _store_last_day = max(new_days)
        _store_sessions += len(frames)
    log.info("bhav_store_appended", days=len(frames), latest=str(max(new_days)))
    _save_pkl()


def build_store(days: int = DEFAULT_DAYS, progress=None) -> int:
    """
    Ensure the store covers ~`days` sessions (~2 years by default).
    Fast paths: in-memory → pickle cache → incremental append of new
    days only. Full CSV re-parse happens once, then never again.
    """
    candidates = _trading_days_back(days)

    # Download whatever is missing on disk, bounded concurrency
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
        have_store = bool(_store)
        last, sessions = _store_last_day, _store_sessions

    # Cold start → try the pickle cache before parsing 500 CSVs
    if not have_store and _load_pkl():
        with _lock:
            last, sessions = _store_last_day, _store_sessions
        have_store = True

    if have_store:
        deep_enough = sessions >= min(days, len(available)) * 0.9
        if last == newest and deep_enough:
            with _lock:
                return len(_store)            # current — nothing to do
        if deep_enough and last and last < newest:
            _incremental_append([d for d in available if d > last])
            with _lock:
                return len(_store)
        # store too shallow (e.g. old 260-day cache, now want 500) → rebuild

    return _full_build(available)


# ── Corporate-action adjustment, applied ON READ ─────────────────────────────
# The on-disk store stays RAW; adjustment happens here so there is no double-
# adjustment across rebuilds and an updated CA table needs no re-download. When
# no CA table exists (logs/ca_events.json absent) `_CA_EVENTS` is {} and
# get_ohlcv returns the raw frame — behaviour is byte-for-byte unchanged.
_CA_EVENTS: Optional[dict] = None      # None = not yet loaded; {} = none on file


def _ca_events() -> dict:
    global _CA_EVENTS
    if _CA_EVENTS is None:
        try:
            from data.corporate_actions import load_events
            _CA_EVENTS = load_events()
        except Exception:
            _CA_EVENTS = {}
    return _CA_EVENTS


def reload_corporate_actions() -> int:
    """Re-read the CA table from disk (after it is updated). Returns the number of
    symbols with at least one event. Cheap — adjustment is on read."""
    global _CA_EVENTS
    try:
        from data.corporate_actions import load_events
        _CA_EVENTS = load_events()
    except Exception:
        _CA_EVENTS = {}
    return len(_CA_EVENTS)


def get_ohlcv(symbol: str) -> Optional[pd.DataFrame]:
    # .copy() must happen INSIDE the lock: nse_live.apply_live_to_store()
    # mutates today's row of this same DataFrame in place (df.loc[...] = )
    # under this same lock during market hours. Copying after releasing the
    # lock could race that in-place write mid-copy — a torn read of
    # today's live-overlaid bar (e.g. new close, stale volume).
    sym = symbol.upper()
    with _lock:
        df = _store.get(sym)
        df = df.copy() if df is not None else None
    if df is None:
        return None
    events = _ca_events().get(sym)
    if events:
        try:
            from data.corporate_actions import adjust_frame
            return adjust_frame(df, events)
        except Exception:
            return df               # fail-open: raw beats a crash
    return df


def store_symbols() -> list[str]:
    with _lock:
        return list(_store.keys())


def is_ready() -> bool:
    with _lock:
        return len(_store) > 0
