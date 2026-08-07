"""
LLM health gate — stop hammering a dead API.

When DeepSeek is out of balance (402) or unauthorised, every brief/coach
cycle was still calling it, failing, and log-spamming. This tiny circuit
breaker remembers the failure and backs off, so call sites skip the LLM
(and use their rule-based fallbacks) until the cooldown passes. One
failure, one quiet cooldown — not a retry every cycle.
"""
from __future__ import annotations

import threading
import time

_lock = threading.Lock()
_down_until: float = 0.0
_reason: str = ""

# Balance/auth problems won't fix themselves in minutes → long cooldown.
# Transient errors (timeout, network) → short cooldown.
_COOLDOWN_HARD = 6 * 3600
_COOLDOWN_SOFT = 300


def _classify(err: str) -> float:
    e = (err or "").lower()
    if any(k in e for k in ("insufficient balance", "402", "401",
                            "unauthorized", "invalid api", "quota",
                            "expired")):
        return _COOLDOWN_HARD
    return _COOLDOWN_SOFT


def note_failure(err: str) -> None:
    global _down_until, _reason
    with _lock:
        _down_until = time.time() + _classify(str(err))
        _reason = str(err)[:120]


def note_success() -> None:
    global _down_until, _reason
    with _lock:
        _down_until = 0.0
        _reason = ""


def available() -> bool:
    with _lock:
        return time.time() >= _down_until


def status() -> dict:
    with _lock:
        secs = max(0.0, _down_until - time.time())
    return {"available": secs <= 0, "cooldown_min": round(secs / 60, 1),
            "reason": _reason}
