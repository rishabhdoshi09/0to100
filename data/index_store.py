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
# whole-batch budget for building the index store from the network; a dead feed gives up here
# instead of blocking the caller through hundreds of per-day request timeouts
_BUILD_BUDGET_S = 25.0
# after a build attempt that could not reach the feed, don't re-pay the network budget for every
# concurrent regime ticker; reuse the outcome for a short cooldown
_BUILD_COOLDOWN_S = 120.0
_last_build_attempt = 0.0
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
_pickup_logged = False


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
    global _store, _last_day, _pickup_logged

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
            # Schema validation — a cache built by older code (e.g. missing
            # the Volume column) must rebuild, not poison the regime engine
            _sample = next(iter(data.get("store", {}).values()), None)
            if _sample is not None and "Volume" not in _sample.columns:
                log.info("index_store_cache_outdated_rebuilding")
                _PKL.unlink()
            else:
                with _lock:
                    _store = data["store"]
                    _last_day = data["last_day"]
                if not _pickup_logged:
                    log.info("index_store_loaded", indices=len(_store),
                             latest=str(_last_day))
                    _pickup_logged = True
                else:
                    log.debug("index_store_pickup", indices=len(_store),
                              latest=str(_last_day))
        except Exception:
            pass

    global _last_build_attempt
    missing = [x for x in candidates if not _day_path(x).exists()]
    if missing and (time.time() - _last_build_attempt) < _BUILD_COOLDOWN_S:
        missing = []                     # a recent attempt already found the feed unreachable
    if missing:
        _last_build_attempt = time.time()
        # Bounded build: when the feed is down, hundreds of per-day timeouts must not block the
        # caller (e.g. the retail Market page) for many minutes. Give the whole batch a budget,
        # then give up gracefully — callers already handle an incomplete/empty store.
        pool = ThreadPoolExecutor(max_workers=6)
        futures = [pool.submit(_download_day, x) for x in missing]
        try:
            for _ in as_completed(futures, timeout=_BUILD_BUDGET_S):
                pass
        except TimeoutError:
            log.warning("index_store_build_timeout",
                        done=sum(1 for f in futures if f.done()), of=len(futures))
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    available = [x for x in candidates if _day_path(x).exists()]
    if len(available) < 30:
        return 0
    newest = max(available)
    with _lock:
        # Only short-circuit if the cache is BOTH current AND deep enough for the
        # requested `days`. Without the depth check, build_index_store(days=2500)
        # on a shallow (~400-session) cache returns immediately and never extends
        # backward — which silently caps the momentum test's history.
        cur_sessions = max((len(df) for df in _store.values()), default=0) if _store else 0
        deep_enough = cur_sessions >= min(days, len(available)) * 0.9
        if _store and _last_day == newest and deep_enough:
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


def build_from_local() -> int:
    """Build the index store from NSE index CSVs ALREADY on disk (user-supplied), with
    NO network. Local-load entry point for the SAME store, not a parallel database.
    Returns the index count (0 if none usable)."""
    from datetime import datetime as _dt
    global _store, _last_day
    if not _DIR.exists():
        return 0
    available = []
    for p in _DIR.glob("*.csv"):
        try:
            available.append(_dt.strptime(p.stem, "%d%m%Y").date())
        except Exception:
            continue
    frames = [f for x in sorted(available) if (f := _read_day(x)) is not None]
    if not frames:
        return 0
    allday = pd.concat(frames, ignore_index=True)
    new_store: dict[str, pd.DataFrame] = {}
    for name, g in allday.groupby("name"):
        g = g.sort_values("date").set_index("date")
        new_store[str(name)] = g[[c for c in ("Open", "High", "Low", "Close", "Volume")
                                  if c in g.columns]]
    with _lock:
        _store = new_store
        _last_day = max(sorted(available))
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
