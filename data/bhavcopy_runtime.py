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

from datetime import date
from pathlib import Path
from typing import Any


def _store_module():
    from data import bhavcopy_store as store
    return store


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

    ``load_cache=True`` loads ``store_cache.pkl`` into this process when its
    in-memory map is empty. This is safe for API/read-only processes.
    """
    store = _store_module()
    if load_cache:
        with store._lock:
            empty = not bool(store._store)
        if empty:
            store._load_pkl()
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
