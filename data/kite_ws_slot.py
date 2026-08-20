"""One KiteTicker per process.

Zerodha allows a handful of WebSockets per API key. The autonomy live feed
and the breakout sniper used to each call ``connect()``. The second upgrade
fails 403 Forbidden, ``on_close`` marks the sniper dead, and the next tick
opens another doomed socket — ``live_feed_stale`` plus a reconnect storm.

First owner keeps the socket. Everyone else attaches and reads the same ticks.
"""
from __future__ import annotations

import threading

_lock = threading.Lock()
_owner: str | None = None


def ticker_owner() -> str | None:
    with _lock:
        return _owner


def claim_ticker(owner: str) -> bool:
    """True when this owner may open (or already owns) the process ticker."""
    name = str(owner or "").strip() or "unknown"
    with _lock:
        global _owner
        if _owner in (None, name):
            _owner = name
            return True
        return False


def release_ticker(owner: str) -> None:
    name = str(owner or "").strip()
    with _lock:
        global _owner
        if _owner == name:
            _owner = None


def reset_ticker_slot() -> None:
    """Tests only."""
    with _lock:
        global _owner
        _owner = None
