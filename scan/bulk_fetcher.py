"""
Bulk OHLCV prefetcher under QuantTerm's Kite-authoritative data policy.

When Kite credentials are available:
  Primary : the immutable active Kite snapshot already produced by DATA_REFRESH.
  Repair  : bounded Kite historical/day calls for only snapshot gaps.
  Rule    : NSE bhavcopy/Yahoo never silently replace Kite-covered OHLCV.

When Kite is unavailable:
  Fallback: official NSE bhavcopy, then Yahoo only as last-resort continuity.

Usage:
    prefetch(symbols)
    backfill_missing(symbols)
    df = get_cached("RELIANCE")
"""
from __future__ import annotations

import threading
import time
import json
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

import pandas as pd

_yf_cache: dict[str, pd.DataFrame] = {}
_kite_cache: dict[str, pd.DataFrame] = {}
_yf_cache_ts: float = 0.0
_lock = threading.Lock()
_TTL = 300           # seconds (yfinance backup cache)
_CHUNK = 200         # symbols per yf.download() call
_bhav_ok: bool = False

_kite_snapshot_id: str = ""
_kite_loaded_snapshot_id: str = ""
_kite_load_lock = threading.Lock()
_KITE_LOAD_CHUNK_ROWS = 100_000


def _kite_authoritative() -> bool:
    try:
        from data.kite_client import kite_credentials_available
        return bool(kite_credentials_available())
    except Exception:
        return False


def _current_kite_snapshot():
    """Invalidate old frames when DATA_REFRESH replaces or removes authority.

    Checking the atomic pointer is cheap; full snapshot verification and CSV
    loading remain on cache misses. Broker repairs without a snapshot retain
    the empty identity until a snapshot is activated.
    """
    global _kite_snapshot_id, _kite_loaded_snapshot_id
    try:
        from research.intelligence.data.snapshot_store import SnapshotStore

        store = SnapshotStore()
        sid = str(store.get_active_snapshot() or "")
    except Exception:
        store, sid = None, ""
    with _lock:
        if sid != _kite_snapshot_id:
            _kite_cache.clear()
            _kite_snapshot_id = sid
            _kite_loaded_snapshot_id = ""
    return store, sid


def _read_kite_snapshot_frames(path, sid: str) -> dict[str, pd.DataFrame]:
    """Read once in chunks, without also materializing Snapshot's row objects."""
    parts: dict[str, list[pd.DataFrame]] = {}
    with pd.read_csv(path, chunksize=_KITE_LOAD_CHUNK_ROWS) as chunks:
        for raw in chunks:
            if raw.empty:
                continue
            raw["symbol"] = raw["symbol"].astype(str).str.upper()
            keep = [col for col in ("open", "high", "low", "close", "volume") if col in raw.columns]
            for symbol, frame in raw.groupby("symbol", sort=False):
                parts.setdefault(str(symbol), []).append(frame[["date", *keep]].copy())

    loaded: dict[str, pd.DataFrame] = {}
    while parts:
        symbol, frames = parts.popitem()
        frame = pd.concat(frames, ignore_index=True)
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values("date").drop_duplicates(subset=["date"], keep="last").set_index("date")
        for col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=[col for col in ("open", "high", "low", "close") if col in frame.columns])
        if len(frame) >= 30:
            frame.attrs["quantterm_source"] = "kite_snapshot"
            frame.attrs["quantterm_snapshot_id"] = sid
            loaded[symbol] = frame
    return loaded


def _adopt_active_kite_snapshot(symbols: list[str] | None = None) -> int:
    """Prepare one verified snapshot for all readers, including known gaps.

    Loading only a requested symbol reread the entire CSV on every cold name
    and every genuine gap. F&O and the cash scan share one load; pointer changes
    invalidate both frames and the completed-load marker. None means all names.
    """
    global _kite_loaded_snapshot_id
    wanted = None if symbols is None else {
        str(symbol or "").strip().upper() for symbol in symbols if str(symbol or "").strip()
    }
    if wanted == set():
        return 0

    try:
        with _kite_load_lock:
            store, sid = _current_kite_snapshot()
            if not sid:
                return 0
            with _lock:
                complete = _kite_loaded_snapshot_id == sid
            if not complete:
                started = time.monotonic()
                directory = store.root / sid
                manifest = json.loads((directory / "manifest.json").read_text())
                if str(manifest.get("source") or "").lower() != "kite":
                    return 0
                verified, failures = store.verify_snapshot(sid)
                if not verified:
                    raise ValueError(f"Kite snapshot verification failed: {failures}")
                log = __import__("logger").get_logger(__name__)
                log.info("kite_snapshot_cache_loading", snapshot_id=sid,
                         instruments=manifest.get("instrument_count"))
                loaded = _read_kite_snapshot_frames(directory / "bars_equity.csv", sid)
                # Recheck the actual pointer after IO, not only what another
                # thread last observed. A successor must never receive old bars.
                _, active = _current_kite_snapshot()
                if active != sid:
                    return 0
                with _lock:
                    if sid != _kite_snapshot_id:
                        return 0
                    _kite_cache.update(loaded)
                    _kite_loaded_snapshot_id = sid
                log.info("kite_snapshot_cache_loaded", snapshot_id=sid,
                         symbols=len(loaded), elapsed_s=round(time.monotonic() - started, 2))
            with _lock:
                return len(_kite_cache) if wanted is None else sum(s in _kite_cache for s in wanted)
    except Exception:
        return 0


def _overlay_live_quiet() -> None:
    try:
        from data.nse_live import apply_live_to_store
        apply_live_to_store()
    except Exception:
        pass


def _bhav_symbols() -> set[str]:
    try:
        from data.bhavcopy_store import store_symbols
        return {str(s).strip().upper() for s in (store_symbols() or []) if str(s).strip()}
    except Exception:
        return set()


def adopt_ready_store(*, overlay_live: bool = True) -> int:
    """Adopt cached history from the active authority without fetching a feed.

    With Kite configured, the bhavcopy count is not scanner readiness: only the
    active Kite snapshot can prepare that cache. Offline operation retains the
    official bhavcopy path and optional background live overlay.
    """
    global _bhav_ok
    if _kite_authoritative():
        return _adopt_active_kite_snapshot()
    try:
        from data.bhavcopy_runtime import status as history_status

        info = history_status(load_cache=True)
        n = int(info.get("symbols") or 0)
        sessions = int(info.get("sessions") or 0)
        if not info.get("ready") or n < 200 or sessions < 60:
            return 0
        with _lock:
            _bhav_ok = True
        if overlay_live:
            threading.Thread(
                target=_overlay_live_quiet, name="live-overlay", daemon=True,
            ).start()
        covered = _bhav_symbols()
        return len(covered) if covered else n
    except Exception:
        return 0


def prefetch(
    symbols: list[str],
    period: str = "260d",
    progress: Optional[Callable[[int, int], None]] = None,
) -> int:
    """Make OHLCV available under the Kite-authoritative source policy."""
    global _bhav_ok
    requested = list(dict.fromkeys(
        str(symbol or "").strip().upper()
        for symbol in (symbols or [])
        if str(symbol or "").strip()
    ))
    if not requested:
        return 0

    if _kite_authoritative():
        _adopt_active_kite_snapshot(requested)
        with _lock:
            missing = [symbol for symbol in requested if symbol not in _kite_cache]
        if missing:
            try:
                loaded, _stats = _repair_via_kite(missing)
            except Exception as exc:
                __import__("logger").get_logger(__name__).warning(
                    "kite_prefetch_failed",
                    missing=len(missing),
                    error=str(exc)[:160],
                )
                loaded = {}
            if loaded:
                with _lock:
                    _kite_cache.update(loaded)
        with _lock:
            covered = sum(1 for symbol in requested if symbol in _kite_cache)
        if progress:
            try:
                progress(covered, len(requested))
            except Exception:
                pass
        return covered

    ready = adopt_ready_store(overlay_live=True)
    if ready >= 200:
        have = _bhav_symbols()
        return sum(1 for s in requested if s in have) or ready

    # Kite unavailable: official NSE history is the first continuity fallback.
    try:
        from data.bhavcopy_store import build_store
        n = build_store(days=260, progress=progress)
        if n >= 200:
            try:
                from data.nse_live import apply_live_to_store
                apply_live_to_store()
            except Exception:
                pass
            with _lock:
                _bhav_ok = True
            covered = _bhav_symbols()
            return sum(1 for s in requested if s in covered)
    except Exception as exc:
        __import__("logger").get_logger(__name__).warning(
            "bhav_prefetch_failed", error=str(exc)
        )
    with _lock:
        _bhav_ok = False

    return _prefetch_yf(requested, period=period, progress=progress)


def _frame_from_kite(candles: list[dict]) -> Optional[pd.DataFrame]:
    rows = []
    for candle in candles or []:
        try:
            o = float(candle["open"])
            h = float(candle["high"])
            l = float(candle["low"])
            c = float(candle["close"])
            v = float(candle.get("volume", 0) or 0)
            stamp = pd.to_datetime(candle.get("date"))
        except Exception:
            continue
        if min(o, h, l, c) <= 0 or l > o or o > h or l > c or c > h or v < 0:
            continue
        rows.append((stamp, o, h, l, c, v))
    if not rows:
        return None
    frame = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    frame = frame.drop_duplicates(subset=["date"], keep="last").sort_values("date").set_index("date")
    return frame


def _repair_via_bhavcopy(missing: list[str]) -> dict:
    """Official route: bring the store up to date, then re-read the gaps.

    A symbol is usually absent because it listed after the store was last
    built, so the fix is sessions rather than symbols. When the store is
    already current these names are genuinely not in the official record, and
    saying that quickly is better than downloading the same sessions again.
    """
    from data import bhavcopy_store as BS

    stored = set(BS.store_symbols() or [])
    still = [s for s in missing if s not in stored]
    if not still:
        return {}

    if not BS.is_ready():
        # An empty store is a bootstrap problem, and bootstrapping it means
        # hundreds of session downloads. A scan asking for three missing
        # symbols must not turn into that: the data layer owns the first build.
        raise LookupError("official store is not built yet; bootstrap owns that")

    current = False
    try:
        from research.intelligence.data import nse_calendar as CAL

        required = CAL.latest_required_session(CAL._now_ist(), CAL.load_holidays())
        latest = BS.latest_two_eq_sessions()
        newest = max((d for d in (latest or []) if d), default=None)
        current = bool(newest and str(newest)[:10] >= required.isoformat())
    except Exception:
        current = False

    if current:
        raise LookupError(
            f"{len(still)} symbols absent from an up-to-date official store"
        )

    BS.build_store()
    out: dict[str, pd.DataFrame] = {}
    for symbol in still:
        try:
            frame = BS.get_ohlcv(symbol)
        except Exception:
            continue
        if frame is not None and len(frame) >= 30:
            out[symbol] = frame
    return out


def _repair_via_yfinance(missing: list[str]) -> dict:
    """Last resort, and labelled as such in the provenance."""
    _prefetch_yf(list(missing))
    with _lock:
        return {s: _yf_cache[s] for s in missing if s in _yf_cache}


def _repair_frames_validator(payload):
    from data.acquisition import EMPTY, PARSER_CHANGED, Validation

    if not isinstance(payload, dict):
        return Validation.bad(PARSER_CHANGED, "repair source returned an unexpected shape")
    if not payload:
        return Validation.bad(EMPTY, "source supplied none of the missing symbols")
    return Validation.good(record_count=len(payload))


def backfill_missing(symbols: list[str], *, client=None, now: datetime | None = None) -> dict:
    """Repair scan history without crossing the active source authority."""
    from data.acquisition import Source, SourceTier, acquire

    requested = list(dict.fromkeys(
        str(s).strip().upper() for s in (symbols or []) if str(s).strip()
    ))
    if _kite_authoritative():
        _adopt_active_kite_snapshot(requested)
        with _lock:
            missing = [s for s in requested if s not in _kite_cache]
        if not missing:
            return {
                "requested": len(requested), "missing": 0, "attempted": 0,
                "attempted_total": 0, "loaded": 0, "unresolved": 0,
                "failed": 0, "state": "KITE_SNAPSHOT", "sources": ["kite_snapshot"],
                "source": "kite_snapshot", "attempts": [],
            }

        stats: dict[str, Any] = {"attempted": 0, "failed": 0}

        def kite_rung() -> dict:
            frames, outcome = _repair_via_kite(missing, client=client, now=now)
            stats.update(outcome)
            return frames

        result = acquire(
            "scan_history_repair",
            [
                Source(
                    "zerodha_kite_data_only",
                    SourceTier.BROKER,
                    kite_rung,
                    _repair_frames_validator,
                    identifier="kite historical/day",
                ),
            ],
            accumulate=lambda acc, new: {**acc, **new},
            complete=lambda acc: set(missing) <= set(acc),
            parser_version="2-kite-authoritative",
        )
        loaded = dict(result.value or {}) if result.ok else {}
        if loaded:
            with _lock:
                _kite_cache.update(loaded)
        report = {
            "requested": len(requested),
            "missing": len(missing),
            "attempted": int(stats.get("attempted") or 0),
            "attempted_total": len(missing),
            "loaded": len(loaded),
            "unresolved": len(set(missing) - set(loaded)),
            "failed": int(stats.get("failed") or 0),
            "state": result.state,
            "sources": list(result.contributing_sources),
            "source": result.source or "",
            "attempts": [
                {
                    "source": a.source,
                    "outcome": a.outcome,
                    "detail": a.detail[:160],
                    "loaded": a.record_count,
                }
                for a in result.attempts
            ],
        }
        if stats.get("error_code"):
            report["error_code"] = stats["error_code"]
            report["error"] = stats.get("error", "")
        return report

    # No Kite session: retain the continuity ladder for offline/public operation.
    with _lock:
        fallback_have = set(_yf_cache)
    have = _bhav_symbols() | fallback_have
    missing = [s for s in requested if s not in have]
    if not missing:
        return {
            "requested": len(requested), "missing": 0, "attempted": 0,
            "attempted_total": 0, "loaded": 0, "unresolved": 0,
            "failed": 0,
        }

    wanted = set(missing)
    result = acquire(
        "scan_history_repair",
        [
            Source(
                "nse_bhavcopy",
                SourceTier.OFFICIAL_FILE,
                lambda: _repair_via_bhavcopy(missing),
                _repair_frames_validator,
                identifier="NSE bhavcopy store",
            ),
            Source(
                "yfinance_daily",
                SourceTier.REPUTABLE_PUBLIC,
                lambda: _repair_via_yfinance(missing),
                _repair_frames_validator,
                identifier="yfinance 1d",
            ),
        ],
        accumulate=lambda acc, new: {**acc, **new},
        complete=lambda acc: wanted <= set(acc),
        parser_version="2-offline-fallback",
    )
    loaded = dict(result.value or {}) if result.ok else {}
    if loaded:
        with _lock:
            _yf_cache.update(loaded)
    return {
        "requested": len(requested),
        "missing": len(missing),
        "attempted": 0,
        "attempted_total": len(missing),
        "loaded": len(loaded),
        "unresolved": len(wanted - set(loaded)),
        "failed": 0,
        "state": result.state,
        "sources": list(result.contributing_sources),
        "source": result.source or "",
        "attempts": [
            {
                "source": a.source,
                "outcome": a.outcome,
                "detail": a.detail[:160],
                "loaded": a.record_count,
            }
            for a in result.attempts
        ],
    }


def _repair_via_kite(missing: list[str], *, client=None, now: datetime | None = None):
    """The broker rung. Returns (frames, stats); raises when the session is unusable."""
    stats: dict[str, Any] = {"attempted": 0, "failed": 0}
    try:
        if client is None:
            from research.intelligence.data.kite_activation import KiteDataClient
            client = KiteDataClient.from_config()
        # profile is the cheapest explicit session validity check and contains no
        # secret in our logs; only success/failure counts are returned below.
        if not client.profile():
            raise RuntimeError("empty Kite profile")
        instruments = list(client.instruments("NSE"))
    except Exception as exc:
        stats["error_code"] = "KITE_HISTORY_UNAVAILABLE"
        stats["error"] = f"{type(exc).__name__}: {exc}"
        raise

    token_by_symbol: dict[str, object] = {}
    for row in instruments:
        if not isinstance(row, dict):
            continue
        if str(row.get("exchange") or "NSE").upper() != "NSE":
            continue
        if str(row.get("instrument_type") or "").upper() != "EQ":
            continue
        sym = str(row.get("tradingsymbol") or "").strip().upper()
        token = row.get("instrument_token")
        if sym and token is not None:
            token_by_symbol[sym] = token

    resolvable = [s for s in missing if s in token_by_symbol]
    if not resolvable:
        return {}, stats

    try:
        from research.intelligence.data import nse_calendar as CAL
        from research.intelligence.data.kite_source import RateLimiter
        required = CAL.latest_required_session(now or CAL._now_ist(), CAL.load_holidays())
        want_to = required.isoformat()
        want_from = (required - timedelta(days=400)).isoformat()
        limiter = RateLimiter(max_per_sec=3.0)
    except Exception:
        end = (now or datetime.now()).date()
        want_to = end.isoformat()
        want_from = (end - timedelta(days=400)).isoformat()
        from research.intelligence.data.kite_source import RateLimiter
        limiter = RateLimiter(max_per_sec=3.0)

    loaded: dict[str, pd.DataFrame] = {}
    failed = 0
    for symbol in resolvable:
        try:
            limiter.acquire()
            candles = list(client.historical(token_by_symbol[symbol], want_from, want_to, "day"))
            frame = _frame_from_kite(candles)
            if frame is None:
                failed += 1
                continue
            loaded[symbol] = frame
        except Exception:
            failed += 1

    stats["attempted"] = len(resolvable)
    stats["failed"] = failed
    return loaded, stats


def get_cached(symbol: str) -> Optional[pd.DataFrame]:
    """Cached OHLCV respecting the active source authority."""
    clean = str(symbol or "").strip().upper()
    if _kite_authoritative():
        _current_kite_snapshot()
        with _lock:
            df = _kite_cache.get(clean)
        if df is None:
            _adopt_active_kite_snapshot([clean])
            with _lock:
                df = _kite_cache.get(clean)
        return df.copy() if df is not None else None

    with _lock:
        use_bhav = _bhav_ok
    if use_bhav:
        try:
            from data.bhavcopy_store import get_ohlcv
            df = get_ohlcv(clean)
            if df is not None:
                return df
        except Exception:
            pass
    with _lock:
        df = _yf_cache.get(clean)
    return df.copy() if df is not None else None


def cached_symbols() -> list[str]:
    if _kite_authoritative():
        _current_kite_snapshot()
        with _lock:
            return sorted(_kite_cache)

    symbols: set[str] = set()
    with _lock:
        use_bhav = _bhav_ok
        symbols.update(_yf_cache)
    if use_bhav:
        symbols.update(_bhav_symbols())
    return sorted(symbols)


def is_warm() -> bool:
    if _kite_authoritative():
        _current_kite_snapshot()
        with _lock:
            return bool(_kite_cache)
    with _lock:
        if _bhav_ok:
            return True
        return bool(_yf_cache) and (time.time() - _yf_cache_ts < _TTL)


# ── yfinance backup path ──────────────────────────────────────────────────────

def _prefetch_yf(
    symbols: list[str],
    period: str = "260d",
    progress: Optional[Callable[[int, int], None]] = None,
) -> int:
    global _yf_cache, _yf_cache_ts

    now = time.time()
    with _lock:
        if now - _yf_cache_ts < _TTL and _yf_cache:
            missing = [s for s in symbols if s not in _yf_cache]
            if len(missing) < max(10, len(symbols) * 0.05):
                return len(_yf_cache)

    wanted = [s for s in symbols if not s.startswith("^")]
    if not wanted:
        return 0

    import yfinance as yf
    log = __import__("logger").get_logger(__name__)
    t0 = time.time()
    log.info("yf_bulk_fetch_start", count=len(wanted))

    result: dict[str, pd.DataFrame] = {}
    for i in range(0, len(wanted), _CHUNK):
        chunk = wanted[i:i + _CHUNK]
        ns_syms = [f"{s}.NS" for s in chunk]
        try:
            raw = yf.download(
                ns_syms, period=period, interval="1d", group_by="ticker",
                threads=8, progress=False, auto_adjust=True,
            )
        except Exception as exc:
            log.warning("yf_bulk_chunk_failed", chunk=i // _CHUNK, error=str(exc))
            continue
        if raw is None or raw.empty:
            continue
        single = len(ns_syms) == 1
        for sym, ns in zip(chunk, ns_syms):
            try:
                df = raw.copy() if single else raw[ns].copy()
                df.columns = [str(c).lower() for c in df.columns]
                df = df.dropna(subset=["close"])
                if len(df) >= 30:
                    result[sym] = df
            except Exception:
                pass
        if progress:
            try:
                progress(min(i + _CHUNK, len(wanted)), len(wanted))
            except Exception:
                pass

    with _lock:
        _yf_cache = result
        _yf_cache_ts = time.time()
    log.info("yf_bulk_fetch_done", loaded=len(result), of=len(wanted),
             elapsed_s=round(time.time() - t0, 1))
    return len(result)
