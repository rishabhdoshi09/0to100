"""
NSE Index Store — official daily OHLC for NIFTY indices, no Yahoo.

Mirrors bhavcopy_store: NSE publishes ind_close_all_DDMMYYYY.csv daily
with every index's OHLC. We keep ~400 sessions on disk (logs/indices/),
consolidate to a pickle, and serve DataFrames in yfinance-compatible
shape (Title-case columns) so the regime engine plugs in unchanged.

Kite historical needs a paid subscription and Yahoo's crumb auth keeps
breaking — this is the boring, official, free source.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from logger import get_logger

log = get_logger(__name__)

_DIR = Path(__file__).resolve().parent.parent / "logs" / "indices"
_PKL = _DIR / "index_store.pkl"
_URL = "https://nsearchives.nseindia.com/content/indices/ind_close_all_{d}.csv"
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Referer": "https://www.nseindia.com/",
}

# yfinance-style ticker → NSE official index name (as in ind_close_all)
TICKER_MAP = {
    "^NSEI":      "Nifty 50",
    "^NSEBANK":   "Nifty Bank",
    "^INDIAVIX":  "India VIX",
    "^CNXIT":     "Nifty IT",
    "^CNXPHARMA": "Nifty Pharma",
    "^CNXFMCG":   "Nifty FMCG",
    "^CNXAUTO":   "Nifty Auto",
    "^CNXMETAL":  "Nifty Metal",
    "^CNXENERGY": "Nifty Energy",
    "^CNXREALTY": "Nifty Realty",
    "^CNXSC":     "Nifty Smallcap 100",
}

_lock = threading.Lock()
_store: dict[str, pd.DataFrame] = {}      # official name -> OHLC df
_last_day: Optional[date] = None


def _day_path(d: date) -> Path:
    return _DIR / f"{d.strftime('%d%m%Y')}.csv"


def _download_day(d: date, retries: int = 1) -> bool:
    path = _day_path(d)
    if path.exists():
        return True
    import requests
    url = _URL.format(d=d.strftime("%d%m%Y"))
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=12)
            if resp.status_code == 404:
                return False
            if resp.status_code == 200 and len(resp.content) > 500:
                _DIR.mkdir(parents=True, exist_ok=True)
                path.write_bytes(resp.content)
                return True
        except Exception:
            pass
        if attempt < retries:
            time.sleep(1.5)
    return False


def _read_day(d: date) -> Optional[pd.DataFrame]:
    path = _day_path(d)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, dtype=str)
        df.columns = [c.strip() for c in df.columns]
        wanted = set(TICKER_MAP.values())
        df = df[df["Index Name"].str.strip().isin(wanted)]
        out = pd.DataFrame({
            "name":  df["Index Name"].str.strip(),
            "Open":  pd.to_numeric(df["Open Index Value"], errors="coerce"),
            "High":  pd.to_numeric(df["High Index Value"], errors="coerce"),
            "Low":   pd.to_numeric(df["Low Index Value"], errors="coerce"),
            "Close": pd.to_numeric(df["Closing Index Value"], errors="coerce"),
            # Regime engine expects a Volume column (yfinance shape)
            "Volume": (pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
                       if "Volume" in df.columns else 0.0),
        })
        out["date"] = pd.Timestamp(d)
        return out.dropna(subset=["Close"])
    except Exception as exc:
        log.debug("index_day_parse_failed", day=str(d), error=str(exc))
        return None


_build_lock = threading.Lock()   # 8 parallel regime fetches must build ONCE


def build_index_store(days: int = 400) -> int:
    """Download-missing → consolidate → pickle. Returns #indices covered.
    Serialised: concurrent callers wait, then reuse the finished build."""
    with _build_lock:
        return _build_index_store_locked(days)


def _build_index_store_locked(days: int = 400) -> int:
    global _store, _last_day

    candidates = []
    d = date.today()
    while len(candidates) < int(days * 1.55):
        if d.weekday() < 5:
            candidates.append(d)
        d -= timedelta(days=1)

    # Fast path: pickle cache current?
    with _lock:
        have = bool(_store)
    if not have and _PKL.exists():
        try:
            import pickle
            with open(_PKL, "rb") as f:
                data = pickle.load(f)
            with _lock:
                _store = data["store"]
                _last_day = data["last_day"]
            log.info("index_store_loaded", indices=len(_store),
                     latest=str(_last_day))
        except Exception:
            pass

    missing = [x for x in candidates if not _day_path(x).exists()]
    if missing:
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(as_completed([pool.submit(_download_day, x) for x in missing]))

    available = [x for x in candidates if _day_path(x).exists()]
    if len(available) < 30:
        return 0
    newest = max(available)
    with _lock:
        if _store and _last_day == newest:
            return len(_store)

    frames = [f for x in sorted(available) if (f := _read_day(x)) is not None]
    if not frames:
        return 0
    allday = pd.concat(frames, ignore_index=True)
    new_store: dict[str, pd.DataFrame] = {}
    for name, g in allday.groupby("name"):
        g = g.sort_values("date").set_index("date")
        new_store[str(name)] = g[[c for c in ("Open", "High", "Low", "Close", "Volume") if c in g.columns]]
    with _lock:
        _store = new_store
        _last_day = newest
    try:
        import pickle
        _DIR.mkdir(parents=True, exist_ok=True)
        tmp = _PKL.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump({"store": new_store, "last_day": newest}, f)
        tmp.replace(_PKL)
    except Exception:
        pass
    log.info("index_store_built", indices=len(new_store), sessions=len(frames),
             latest=str(newest))
    return len(new_store)


def get_index_ohlcv(ticker: str) -> Optional[pd.DataFrame]:
    """yfinance-shaped OHLC for a ^TICKER. Builds the store on first use."""
    name = TICKER_MAP.get(ticker.upper())
    if not name:
        return None
    with _lock:
        have = name in _store
    if not have:
        build_index_store()
    with _lock:
        df = _store.get(name)
    return df.copy() if df is not None else None
