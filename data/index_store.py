"""
Index OHLC store under QuantTerm's Kite-authoritative market-data policy.

When Kite is connected, index history is fetched from Zerodha Kite historical
data and persisted in a dedicated cache for the regime engine. The legacy
official NSE index CSV store remains an explicit continuity fallback only when
Kite credentials are unavailable.
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
from core.runtime_paths import logs_path

log = get_logger(__name__)

_DIR = logs_path("indices")
_PKL = _DIR / "index_store.pkl"
_KITE_PKL = _DIR / "index_store_kite.pkl"
# whole-batch budget for building the index store from the network; a dead feed gives up here
# instead of blocking the caller through hundreds of per-day request timeouts
_BUILD_BUDGET_S = 25.0
# after a build attempt that could not reach the feed, don't re-pay the network budget for every
# concurrent regime ticker; reuse the outcome for a short cooldown
_BUILD_COOLDOWN_S = 120.0
# regime_engine.compute_regime() fans out 8+ tickers in parallel; they all funnel through the
# same _build_lock, so a cold/shallow store with a dead feed would otherwise have EVERY queued
# ticker re-pay the full _BUILD_BUDGET_S in turn (8 x 25s minutes-long stall) because the
# depth-gated _BUILD_COOLDOWN_S below only protects an already-deep store. This short debounce
# — just over one build attempt's worth of time — coalesces that whole burst of lock waiters
# into the ONE attempt already paid for, while still letting a genuinely later retry (next scan
# cycle, etc.) try immediately per the cold-bootstrap comment below.
_BUILD_DEBOUNCE_S = _BUILD_BUDGET_S + 2.0
# Regime classification needs a real SMA200 / long-window market context. A
# cold bootstrap that stops below this depth is not usable and must be allowed
# to continue immediately instead of being frozen by the retry cooldown.
_REGIME_BOOTSTRAP_SESSIONS = 220
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

_KITE_INDEX_SYMBOLS = {
    "^NSEI": "NIFTY 50",
    "^NSEBANK": "NIFTY BANK",
    "^INDIAVIX": "INDIA VIX",
    "^CNXIT": "NIFTY IT",
    "^CNXPHARMA": "NIFTY PHARMA",
    "^CNXFMCG": "NIFTY FMCG",
    "^CNXAUTO": "NIFTY AUTO",
    "^CNXMETAL": "NIFTY METAL",
    "^CNXENERGY": "NIFTY ENERGY",
    "^CNXREALTY": "NIFTY REALTY",
    "^CNXSC": "NIFTY SMLCAP 100",
}

_lock = threading.Lock()
_store: dict[str, pd.DataFrame] = {}      # official NSE fallback store
_last_day: Optional[date] = None
_kite_lock = threading.Lock()
_kite_build_lock = threading.Lock()
_kite_store: dict[str, pd.DataFrame] = {}
_kite_last_day: Optional[date] = None


def _kite_authoritative() -> bool:
    try:
        from data.kite_client import kite_credentials_available
        return bool(kite_credentials_available())
    except Exception:
        return False


def _load_kite_cache() -> bool:
    global _kite_store, _kite_last_day
    with _kite_lock:
        if _kite_store:
            return True
    if not _KITE_PKL.exists():
        return False
    try:
        import pickle
        with open(_KITE_PKL, "rb") as fh:
            payload = pickle.load(fh)
        store = payload.get("store") or {}
        if not isinstance(store, dict):
            return False
        sample = next(iter(store.values()), None)
        if sample is not None and "Close" not in getattr(sample, "columns", []):
            return False
        with _kite_lock:
            _kite_store = store
            raw_day = payload.get("last_day")
            _kite_last_day = raw_day if isinstance(raw_day, date) else (
                date.fromisoformat(str(raw_day)[:10]) if raw_day else None
            )
        return bool(store)
    except Exception:
        return False


def _build_kite_index_store(days: int = 400) -> int:
    """Build all regime indices from Kite historical data in one bounded pass."""
    global _kite_store, _kite_last_day
    target = max(_REGIME_BOOTSTRAP_SESSIONS, int(days or 0))
    with _kite_build_lock:
        _load_kite_cache()
        with _kite_lock:
            depth = max((len(df) for df in _kite_store.values()), default=0)
            cached_last = _kite_last_day
        try:
            from research.intelligence.data import nse_calendar as CAL
            required = CAL.latest_required_session(CAL._now_ist(), CAL.load_holidays())
        except Exception:
            required = date.today()
        if depth >= target and cached_last is not None and cached_last >= required:
            return len(_kite_store)

        try:
            from data.kite_client import KiteClient
            from research.intelligence.data.kite_source import RateLimiter

            client = KiteClient()
            rows = client.get_instruments("NSE")
            token_by_symbol: dict[str, int] = {}
            wanted_symbols = set(_KITE_INDEX_SYMBOLS.values())
            for row in rows:
                if not isinstance(row, dict):
                    continue
                symbol = str(row.get("tradingsymbol") or "").strip().upper()
                token = row.get("instrument_token")
                if symbol in wanted_symbols and token is not None:
                    token_by_symbol[symbol] = int(token)

            if not token_by_symbol:
                raise RuntimeError("Kite NSE instrument master returned no configured indices")

            end = required
            start = end - timedelta(days=max(500, int(target * 1.8)))
            limiter = RateLimiter(max_per_sec=3.0)
            built: dict[str, pd.DataFrame] = {}
            latest: Optional[date] = None
            for ticker, official_name in TICKER_MAP.items():
                kite_symbol = _KITE_INDEX_SYMBOLS.get(ticker)
                token = token_by_symbol.get(str(kite_symbol or "").upper())
                if token is None:
                    continue
                limiter.acquire()
                frame = client.get_historical(
                    token,
                    start.isoformat(),
                    end.isoformat(),
                    "day",
                )
                if frame is None or frame.empty:
                    continue
                rename = {
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
                frame = frame.rename(columns=rename)
                keep = [col for col in ("Open", "High", "Low", "Close", "Volume") if col in frame.columns]
                frame = frame[keep].copy()
                if "Volume" not in frame.columns:
                    frame["Volume"] = 0.0
                frame = frame.dropna(subset=["Close"])
                if frame.empty:
                    continue
                frame.attrs["quantterm_source"] = "kite_index_history"
                frame.attrs["kite_tradingsymbol"] = kite_symbol
                built[official_name] = frame
                try:
                    d = pd.Timestamp(frame.index[-1]).date()
                    latest = d if latest is None or d > latest else latest
                except Exception:
                    pass

            if not built:
                raise RuntimeError("Kite historical returned no configured index history")

            with _kite_lock:
                _kite_store = built
                _kite_last_day = latest
            try:
                import pickle
                _DIR.mkdir(parents=True, exist_ok=True)
                tmp = _KITE_PKL.with_suffix(".pkl.tmp")
                with open(tmp, "wb") as fh:
                    pickle.dump({
                        "store": built,
                        "last_day": latest,
                        "source": "zerodha_kite_historical",
                    }, fh)
                tmp.replace(_KITE_PKL)
            except Exception:
                pass
            log.info(
                "kite_index_store_built",
                indices=len(built),
                latest=str(latest or ""),
            )
            return len(built)
        except Exception as exc:
            log.warning("kite_index_store_failed", error=str(exc)[:180])
            return 0


def _kite_index_frame(ticker: str, *, build: bool = True) -> Optional[pd.DataFrame]:
    name = TICKER_MAP.get((ticker or "").upper())
    if not name:
        return None
    _load_kite_cache()
    with _kite_lock:
        frame = _kite_store.get(name)
        depth = len(frame) if frame is not None else 0
    if build and depth < _REGIME_BOOTSTRAP_SESSIONS:
        _build_kite_index_store(_REGIME_BOOTSTRAP_SESSIONS)
        with _kite_lock:
            frame = _kite_store.get(name)
    return frame.copy() if frame is not None else None


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


def _days_to_download(
    candidates: list[date],
    *,
    last_day: Optional[date],
    have_store: bool,
) -> list[date]:
    """CSV days still worth fetching. Historical holes behind a deep current pickle stay on disk."""
    missing = [x for x in candidates if not _day_path(x).exists()]
    if have_store and last_day is not None:
        return [x for x in missing if x > last_day]
    return missing


def _candidate_weekdays(days: int, *, today: Optional[date] = None) -> list[date]:
    """Weekdays inside a calendar horizon sized for roughly the requested sessions.

    The 1.55 factor is a calendar-day allowance for weekends and exchange
    holidays. Counting 1.55x weekdays over-fetches years of history and can
    exhaust the cold-start network budget before the regime has enough bars.
    """
    target = max(1, int(days))
    horizon_days = max(target, int(target * 1.55))
    start = today or date.today()
    floor = start - timedelta(days=horizon_days - 1)
    out: list[date] = []
    d = start
    while d >= floor:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return out


_build_lock = threading.Lock()   # 8 parallel regime fetches must build ONCE


def build_index_store(days: int = 400) -> int:
    """Build index history from Kite when authoritative, NSE only when offline."""
    if _kite_authoritative():
        return _build_kite_index_store(days)
    with _build_lock:
        return _build_index_store_locked(days)


def _build_index_store_locked(days: int = 400) -> int:
    global _store, _last_day

    candidates = _candidate_weekdays(days)

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
                log.info("index_store_loaded", indices=len(_store),
                         latest=str(_last_day))
        except Exception:
            pass

    global _last_build_attempt
    with _lock:
        have_store = bool(_store)
        last = _last_day
        cur_sessions = max((len(df) for df in _store.values()), default=0) if _store else 0

    # A current but shallow partial bootstrap must be allowed to fill backward.
    # Once the requested depth is genuinely satisfied, historical holes behind
    # the latest session may stay untouched and only new sessions are fetched.
    requested_depth = max(
        min(max(1, int(days)), _REGIME_BOOTSTRAP_SESSIONS),
        int(max(1, int(days)) * 0.90),
    )
    depth_satisfied = have_store and cur_sessions >= requested_depth
    missing = _days_to_download(candidates, last_day=last, have_store=depth_satisfied)
    _since_last_attempt = time.time() - _last_build_attempt
    if missing and cur_sessions >= _REGIME_BOOTSTRAP_SESSIONS and _since_last_attempt < _BUILD_COOLDOWN_S:
        missing = []                     # a recent attempt already found the feed unreachable
    elif missing and _since_last_attempt < _BUILD_DEBOUNCE_S:
        missing = []                     # a sibling ticker's build is still in flight / just finished
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
            payload = {
                "done": sum(1 for f in futures if f.done()),
                "of": len(futures),
            }
            if have_store:
                log.debug("index_store_build_timeout", **payload)
            else:
                log.warning("index_store_build_timeout", **payload)
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


def load_index_store_from_cache() -> bool:
    """Load the pickle into memory. Never hits the NSE download path."""
    global _store, _last_day
    with _lock:
        if _store:
            return True
    if not _PKL.exists():
        return False
    try:
        import pickle
        with open(_PKL, "rb") as f:
            data = pickle.load(f)
        sample = next(iter(data.get("store", {}).values()), None)
        if sample is not None and "Volume" not in sample.columns:
            return False
        with _lock:
            _store = data["store"]
            _last_day = data.get("last_day")
        return bool(_store)
    except Exception:
        return False


def recent_index_closes(ticker: str, n: int = 4) -> list[float]:
    """Oldest-to-newest closes under the active index-data authority."""
    if _kite_authoritative():
        df = _kite_index_frame(ticker)
        if df is None or "Close" not in df.columns:
            return []
        values = [float(x) for x in df["Close"].dropna().tolist() if float(x) > 0]
        return values[-max(1, int(n or 1)):]

    name = TICKER_MAP.get((ticker or "").upper())
    if not name:
        return []
    load_index_store_from_cache()
    with _lock:
        df = _store.get(name)
        if df is None or "Close" not in getattr(df, "columns", []):
            return []
        values = []
        for raw in df["Close"].tolist():
            try:
                price = float(raw)
            except (TypeError, ValueError):
                continue
            if price > 0:
                values.append(price)
    return values[-max(1, int(n or 1)):]


def latest_index_print(ticker: str) -> Optional[dict]:
    """Last close + one-day change under the active index-data authority."""
    if _kite_authoritative():
        df = _kite_index_frame(ticker)
        if df is None or len(df) < 2 or "Close" not in df.columns:
            return None
        close = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2])
        try:
            as_of = str(pd.Timestamp(df.index[-1]).date())
        except Exception:
            as_of = ""
        return {
            "price": close,
            "chg_pct": round((close / prev - 1.0) * 100.0, 2) if prev else 0.0,
            "source": "kite_index_history",
            "as_of": as_of,
        }

    name = TICKER_MAP.get(ticker.upper())
    if not name:
        return None
    load_index_store_from_cache()
    with _lock:
        df = _store.get(name)
        if df is None or len(df) < 2 or "Close" not in df.columns:
            return None
        close = float(df["Close"].iloc[-1])
        prev = float(df["Close"].iloc[-2])
        try:
            as_of = str(pd.Timestamp(df.index[-1]).date())
        except Exception:
            as_of = ""
    if close <= 0:
        return None
    return {
        "price": close,
        "chg_pct": round((close / prev - 1.0) * 100.0, 2) if prev else 0.0,
        "source": "nse_index_store",
        "as_of": as_of,
    }


def get_index_ohlcv_as_of_cached(ticker: str, as_of: str) -> Optional[pd.DataFrame]:
    """Return cached index OHLC on/before as_of without network I/O."""
    if _kite_authoritative():
        frame = _kite_index_frame(ticker, build=False)
        if frame is None or frame.empty:
            return None
    else:
        name = TICKER_MAP.get((ticker or "").upper())
        if not name:
            return None
        if not load_index_store_from_cache():
            try:
                build_from_local()
            except Exception:
                return None
        with _lock:
            df = _store.get(name)
            frame = df.copy() if df is not None else None
        if frame is None or getattr(frame, "empty", True):
            return None
    try:
        cutoff = pd.Timestamp(str(as_of)[:10]).normalize()
        frame = frame.loc[frame.index.normalize() <= cutoff]
    except Exception:
        return None
    return frame.copy() if frame is not None and not frame.empty else None

def get_index_ohlcv(ticker: str) -> Optional[pd.DataFrame]:
    """Index OHLC under the active market-data authority."""
    if _kite_authoritative():
        return _kite_index_frame(ticker)

    name = TICKER_MAP.get(ticker.upper())
    if not name:
        return None
    for _ in range(2):
        with _lock:
            df = _store.get(name)
            depth = len(df) if df is not None else 0
        if depth >= _REGIME_BOOTSTRAP_SESSIONS:
            return df.copy()
        build_index_store(days=_REGIME_BOOTSTRAP_SESSIONS)
    with _lock:
        df = _store.get(name)
    return df.copy() if df is not None else None
