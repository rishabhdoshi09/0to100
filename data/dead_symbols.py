"""
Dead-symbol registry — delisted/renamed symbols se baar-baar poochna band.

A symbol lands here only on STRONG evidence: Kite's instrument list
doesn't know it AND yfinance has no data either. Once registered it is
skipped everywhere (history fetch, outcome tracker, per-symbol scrape
passes) for TTL_DAYS, then retried once — so a relisting or an NSE
rename heals automatically instead of needing manual cleanup.

Persistent (logs/dead_symbols.json), thread-safe, zero network.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from logger import get_logger

log = get_logger(__name__)

_FILE = Path(__file__).resolve().parent.parent / "logs" / "dead_symbols.json"
_TTL_DAYS = 7

_lock = threading.Lock()
_cache: dict[str, dict] | None = None


def _load() -> dict[str, dict]:
    global _cache
    with _lock:
        if _cache is None:
            _cache = {}
            try:
                if _FILE.exists():
                    _cache = json.loads(_FILE.read_text())
            except Exception:
                _cache = {}
        return _cache


def _save() -> None:
    try:
        with _lock:
            data = dict(_cache or {})
        _FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(_FILE)
    except Exception as exc:
        log.debug("dead_symbols_save_failed", error=str(exc))


def is_dead(symbol: str) -> bool:
    """True while the symbol's registration is fresh (< TTL). Expired
    entries return False so the symbol gets ONE fresh chance."""
    entry = _load().get(symbol.upper())
    if not entry:
        return False
    if time.time() - float(entry.get("ts", 0)) > _TTL_DAYS * 86400:
        return False
    return True


def mark_dead(symbol: str, reason: str = "") -> None:
    sym = symbol.upper()
    with _lock:
        cache = _cache if _cache is not None else {}
        cache[sym] = {"ts": time.time(), "reason": reason[:120]}
    _save()
    log.info("symbol_marked_dead", symbol=sym, reason=reason[:80],
             ttl_days=_TTL_DAYS)


def dead_list() -> dict[str, dict]:
    """Snapshot for diagnostics/UI."""
    return {k: dict(v) for k, v in _load().items()
            if time.time() - float(v.get("ts", 0)) <= _TTL_DAYS * 86400}
