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

_last_network_build_ts = 0.0
_NETWORK_COOLDOWN_S = 600.0


def _store_module():
    from data import bhavcopy_store as store
    return store


def _required_session() -> str:
    try:
        from core.market_clock import now_ist_naive
        from research.autonomy.schedules import required_completed_session
        return required_completed_session(now_ist_naive())
    except Exception:
        return ""


def _freshness_fields(latest: str) -> dict[str, Any]:
    required = _required_session()
    latest_s = str(latest or "")
    if required and (not latest_s or latest_s < required):
        freshness, stale = "STALE", True
    elif latest_s:
        freshness, stale = "READY", False
    else:
        freshness, stale = "MISSING", True
    return {
        "required_session": required,
        "is_stale": stale,
        "freshness": freshness,
    }


def _snapshot(store) -> dict[str, Any]:
    with store._lock:
        symbols = len(store._store)
        sessions = int(store._store_sessions or 0)
        latest = store._store_last_day
    dates = store._dates_on_disk()
    latest_s = latest.isoformat() if isinstance(latest, date) else str(latest or "")
    payload = {
        "ready": symbols > 0,
        "symbols": symbols,
        "sessions": sessions,
        "latest_date": latest_s,
        "csv_files": len(dates),
        "csv_latest_date": dates[-1].isoformat() if dates else "",
        "cache_exists": bool(store._PKL.exists()),
        "cache_path": str(store._PKL),
        "bhavcopy_dir": str(store._BHAV_DIR),
        "minimum_sessions": int(store._MIN_DAYS),
        "source": "official_nse_bhavcopy",
    }
    payload.update(_freshness_fields(latest_s))
    return payload


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


def ensure_current_session(*, allow_network: bool = False) -> dict[str, Any]:
    """Bring the in-process store up to the last completed NSE session.

    Local CSVs already on disk are appended immediately. Network download is
    optional and rate-limited so a dashboard poll cannot hammer NSE.
    """
    global _last_network_build_ts
    current = status(load_cache=True)
    if not current.get("is_stale"):
        return current
    store = _store_module()
    csv_latest = str(current.get("csv_latest_date") or "")
    latest = str(current.get("latest_date") or "")
    if csv_latest and (not latest or csv_latest > latest):
        try:
            store.build_store()
        except Exception:
            pass
        return status(load_cache=False)
    if allow_network:
        import time
        now = time.time()
        if now - _last_network_build_ts >= _NETWORK_COOLDOWN_S:
            _last_network_build_ts = now
            try:
                store.build_store()
            except Exception:
                pass
            return status(load_cache=False)
    return status(load_cache=False)
