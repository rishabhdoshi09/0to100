"""
Auto-Scan — background full-market scanner.

Starts once per process. A daemon thread scans the ENTIRE NSE universe
with the UnifiedScanner, stores results in a module-level store, and
refreshes every 15 minutes during market hours (hourly otherwise).

The UI reads the store instantly — no waiting. Every BUY signal is
logged to the signal-outcome tracker so accuracy is measured on real
outcomes, not opinions.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Optional

from logger import get_logger

log = get_logger(__name__)

_lock = threading.Lock()
_results: list[dict] = []
_scanned_count: int = 0
_last_scan_ts: float = 0.0
_status: str = "idle"          # idle | scanning | ready | error
_thread_started: bool = False

_MARKET_REFRESH_S = 900        # 15 min during market hours
_OFFHOURS_REFRESH_S = 3600     # hourly otherwise


def _is_market_hours() -> bool:
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return 9 * 60 + 15 <= minutes <= 15 * 60 + 30


def _serialize(r) -> dict:
    return {
        "symbol": r.symbol, "price": r.price, "change_pct": r.change_pct,
        "momentum_5d": r.momentum_5d, "volume_ratio": r.volume_ratio,
        "signals": r.signal_labels, "categories": sorted(r.categories),
        "reasons": r.reasons, "score": r.score, "verdict": r.verdict,
        "entry": r.entry, "stop": r.stop, "target": r.target,
        "rr": round(r.risk_reward, 1),
    }


def _log_buys_for_tracking(results) -> None:
    """Feed BUY signals into the outcome tracker (dedupes per day itself)."""
    try:
        from core.signal_outcome_tracker import log_signal
        for r in results:
            if r.verdict != "BUY":
                continue
            log_signal(
                symbol=r.symbol, signal_type="UNIFIED_BUY",
                entry_price=r.entry, pivot_price=r.entry,
                stop_price=r.stop, target_price=r.target,
                quality_score=r.score, accum_score=0.0,
                archetype="|".join(r.signals), regime="",
            )
    except Exception as exc:
        log.debug("auto_scan_tracking_skip", error=str(exc))


def _scan_once() -> None:
    global _results, _scanned_count, _last_scan_ts, _status
    with _lock:
        _status = "scanning"
    try:
        from data.nse_universe import get_nse_universe
        from scan.unified_scanner import UnifiedScanner
        universe = get_nse_universe()
        raw = UnifiedScanner(max_workers=8).scan(universe)
        _log_buys_for_tracking(raw)
        serialized = [_serialize(r) for r in raw]
        # JARVIS conviction layer — news buzz + earnings evidence on top picks
        try:
            from scan.conviction import build_conviction
            serialized = build_conviction(serialized)
        except Exception as exc:
            log.debug("conviction_skip", error=str(exc))
        with _lock:
            _results = serialized
            _scanned_count = len(universe)
            _last_scan_ts = time.time()
            _status = "ready"
        log.info("auto_scan_complete", universe=len(universe), signals=len(serialized))
    except Exception as exc:
        with _lock:
            _status = "error" if not _results else "ready"
        log.warning("auto_scan_failed", error=str(exc))


def _worker() -> None:
    while True:
        _scan_once()
        time.sleep(_MARKET_REFRESH_S if _is_market_hours() else _OFFHOURS_REFRESH_S)


def start_background_scan() -> None:
    """Idempotent — starts the daemon scanner thread once per process."""
    global _thread_started
    with _lock:
        if _thread_started:
            return
        _thread_started = True
    t = threading.Thread(target=_worker, name="auto-scan", daemon=True)
    t.start()
    log.info("auto_scan_started")


def force_rescan() -> None:
    """Trigger an immediate rescan in the background (non-blocking)."""
    threading.Thread(target=_scan_once, name="auto-scan-force", daemon=True).start()


def get_results() -> tuple[list[dict], int, float, str]:
    """Returns (results, universe_size, last_scan_unix_ts, status)."""
    with _lock:
        return list(_results), _scanned_count, _last_scan_ts, _status
