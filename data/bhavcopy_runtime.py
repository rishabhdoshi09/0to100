"""Cross-process runtime access to the canonical NSE bhavcopy store.

``data.bhavcopy_store`` keeps an in-memory symbol map for speed and persists the
same map to ``logs/bhav/store_cache.pkl``. QuantTerm runs the autonomy supervisor
and FastAPI bridge in separate processes, so every reader must load that persisted
cache before treating the store as unavailable.

This module does not create a second database and never fetches live data. It only
loads the canonical persisted cache, optionally rebuilding it from already-downloaded
CSV files when explicitly requested by a worker.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import threading
from typing import Any


_CACHE_RELOAD_LOCK = threading.Lock()
_LAST_CACHE_RELOAD_ATTEMPT: tuple[Any, ...] | None = None


def _store_module():
    from data import bhavcopy_store as store
    return store


def _cache_signature(path: Path) -> tuple[int, int, int] | None:
    """Cheap identity for the persisted pickle currently visible on disk."""
    try:
        stat = path.stat()
        return (int(stat.st_mtime_ns), int(stat.st_size), int(stat.st_ino))
    except Exception:
        return None


def _sync_persisted_cache(store) -> None:
    """Reload a newer persisted cache at most once per unchanged cache file.

    Raw CSVs can appear before another process has finished rebuilding/writing the
    canonical pickle. Comparing only ``csv_latest > memory_latest`` therefore used
    to reload the same large pickle on every health poll. Remember the exact cache
    file identity attempted for a given memory/disk date pair: a failed-to-advance
    reload is suppressed until the atomic pickle file actually changes.
    """
    global _LAST_CACHE_RELOAD_ATTEMPT

    with _CACHE_RELOAD_LOCK:
        with store._lock:
            empty = not bool(store._store)
            memory_latest = store._store_last_day

        if empty:
            store._load_pkl()
            return

        dates = store._dates_on_disk()
        disk_latest = dates[-1] if dates else None
        if disk_latest is None or (
            isinstance(memory_latest, date) and disk_latest <= memory_latest
        ):
            return

        attempt = (memory_latest, disk_latest, _cache_signature(store._PKL))
        if attempt == _LAST_CACHE_RELOAD_ATTEMPT:
            return

        # Mark before loading. _load_pkl() is fail-safe and may legitimately leave
        # memory on the older completed session; repeated requests must not spin on
        # that same unchanged pickle. An atomic rewrite changes the signature and
        # permits the next coherence reload immediately.
        _LAST_CACHE_RELOAD_ATTEMPT = attempt
        store._load_pkl()


def _snapshot(store) -> dict[str, Any]:
    with store._lock:
        symbols = len(store._store)
        sessions = int(store._store_sessions or 0)
        latest = store._store_last_day
    dates = store._dates_on_disk()
    return {
        "ready": symbols > 0,
        "symbols": symbols,
        "sessions": sessions,
        "latest_date": latest.isoformat() if isinstance(latest, date) else str(latest or ""),
        "csv_files": len(dates),
        "csv_latest_date": dates[-1].isoformat() if dates else "",
        "cache_exists": bool(store._PKL.exists()),
        "cache_path": str(store._PKL),
        "bhavcopy_dir": str(store._BHAV_DIR),
        "minimum_sessions": int(store._MIN_DAYS),
        "source": "official_nse_bhavcopy",
    }


def status(*, load_cache: bool = False) -> dict[str, Any]:
    """Return canonical history readiness without network access.

    ``load_cache=True`` keeps this process coherent with the persisted canonical
    cache. It loads ``store_cache.pkl`` when memory is empty and reloads when a
    newer raw session exists, but never reloads the same unchanged pickle in a
    tight loop while another process is still publishing that session.
    """
    store = _store_module()
    if load_cache:
        _sync_persisted_cache(store)
    return _snapshot(store)


def ensure_loaded(*, rebuild_from_local: bool = False) -> dict[str, Any]:
    """Load the persisted canonical cache into the current process.

    When ``rebuild_from_local`` is true, a worker may rebuild the same cache from
    CSV files already present under ``logs/bhav``. No network call occurs here.
    """
    store = _store_module()
    current = status(load_cache=True)
    if current["ready"] or not rebuild_from_local:
        return current
    if current["csv_files"] >= int(current["minimum_sessions"]):
        store.build_from_local()
    return status(load_cache=False)


def get_ohlcv(symbol: str):
    """Return canonical OHLCV after lazily loading the persisted cache."""
    ensure_loaded(rebuild_from_local=False)
    return _store_module().get_ohlcv(symbol)


def expected_latest_completed_session(
    *,
    now: datetime | None = None,
    holidays: set | None = None,
) -> date:
    """Most recent completed-and-published NSE session.

    Uses the existing exchange calendar: weekends and holidays walk back, and
    today's file is required only after the official publication cutoff.
    """
    from research.intelligence.data.nse_calendar import latest_required_session, load_holidays, _now_ist

    clock = now or _now_ist()
    hols = holidays if holidays is not None else load_holidays()
    return latest_required_session(clock, hols)


def official_history_freshness(
    history: dict[str, Any] | None = None,
    *,
    now: datetime | None = None,
    holidays: set | None = None,
    load_cache: bool = True,
) -> dict[str, Any]:
    """Compare official bhavcopy ``latest_date`` to the expected completed session.

    A large store that ends on an old session is stale. Missing dates stay
    missing — this never invents bars.
    """
    from research.intelligence.data.nse_calendar import (
        load_holidays, publication_window, sessions_gap,
    )

    current = dict(history) if history is not None else status(load_cache=load_cache)
    hols = holidays if holidays is not None else load_holidays()
    window = publication_window(now, hols)
    completed = window["completed_session"]
    minimum_required = window["minimum_required_official_session"]
    expected = completed
    latest_raw = str(current.get("latest_date") or current.get("csv_latest_date") or "")[:10]
    try:
        latest = date.fromisoformat(latest_raw) if latest_raw else None
    except ValueError:
        latest = None
    sessions = int(current.get("sessions", 0) or 0)
    ready = bool(current.get("ready"))
    stale_sessions = sessions_gap(latest, completed, hols) if latest is not None else None
    try:
        from research.intelligence.data.nse_calendar import _now_ist
        today = (now or _now_ist()).date()
    except Exception:
        today = completed

    # Four truths, deliberately not collapsed into one boolean:
    #   current           -- we hold the latest CLOSED session. Never true just
    #                        because a clock crossed the publication cutoff.
    #   usable_for_scan   -- we hold everything that is MANDATORY right now.
    #   publication_pending -- the completed session's archive has not landed,
    #                        and is not yet mandatory.
    #   stale             -- something mandatory is missing. Fail closed.
    publication_pending = False
    if not ready:
        reason_code, current_ok, usable = "HISTORY_NOT_READY", False, False
    elif sessions < 60:
        reason_code, current_ok, usable = "HISTORY_TOO_SHALLOW", False, False
    elif latest is None:
        reason_code, current_ok, usable = "HISTORY_DATE_MISSING", False, False
    elif latest > today:
        reason_code, current_ok, usable = "HISTORY_FUTURE_DATED", False, False
    elif latest >= completed:
        reason_code, current_ok, usable = "HISTORY_CURRENT", True, True
    elif latest >= minimum_required:
        reason_code, current_ok, usable = "HISTORY_PUBLICATION_PENDING", False, True
        publication_pending = True
    else:
        reason_code, current_ok, usable = "HISTORY_STALE", False, False
    return {
        **current,
        "current": current_ok,
        "usable_for_scan": usable,
        "publication_pending": publication_pending,
        "expected_latest_completed_session": expected.isoformat(),
        "completed_session": completed.isoformat(),
        "minimum_required_official_session": minimum_required.isoformat(),
        "publication_deadline": window["publication_deadline"].isoformat(),
        "in_publication_grace": bool(window["in_publication_grace"]),
        "available_session": latest.isoformat() if latest is not None else latest_raw,
        "stale_sessions": 0 if current_ok else (stale_sessions if stale_sessions is not None else None),
        "reason_code": reason_code,
        "history": current,
    }
