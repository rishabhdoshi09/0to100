"""
System Pulse — heartbeat + latency telemetry for every daemon.

Premium products don't guess whether the machinery is alive — they show
it. Every background loop calls beat(name) each cycle; hot paths wrap
themselves in timed(name). The UI renders one strip: green/amber/red
per daemon, p50/p95 latency per path, quote-cache hit rate.

In-memory only (all daemons share the Streamlit process), thread-safe,
zero I/O — recording a beat or a timing costs nanoseconds, so this can
never become the latency it is meant to measure.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager

_lock = threading.Lock()
_beats: dict[str, dict] = {}                 # name -> {ts, note}
_lat: dict[str, deque] = {}                  # name -> rolling seconds
_counters: dict[str, dict[str, int]] = {}    # name -> {key: count}

_LAT_WINDOW = 200

# Expected cadence per daemon: (warn_after_s, dead_after_s).
# auto_scan sleeps 15 min in market hours / 60 min off-hours, so "dead"
# only past the LONGEST legitimate gap — no false alarms overnight.
_CADENCE = {
    "auto_scan":          (1200, 4500),
    "telegram_listener":  (120, 600),
    "sniper":             (300, 1800),
    "live_ticker":        (300, 1800),
    "autopilot":          (1200, 4500),
}
_DEFAULT_CADENCE = (900, 3600)


# ── Recording (called from daemons/hot paths) ─────────────────────────────────

def beat(name: str, note: str = "") -> None:
    with _lock:
        _beats[name] = {"ts": time.time(), "note": note}


def record_latency(name: str, seconds: float) -> None:
    with _lock:
        _lat.setdefault(name, deque(maxlen=_LAT_WINDOW)).append(float(seconds))


@contextmanager
def timed(name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        record_latency(name, time.perf_counter() - t0)


def count(name: str, key: str, n: int = 1) -> None:
    with _lock:
        _counters.setdefault(name, {})
        _counters[name][key] = _counters[name].get(key, 0) + n


# ── Reading (called from the UI) ──────────────────────────────────────────────

def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = min(len(vals) - 1, max(0, int(round(p / 100 * (len(vals) - 1)))))
    return vals[idx]


def daemon_status(name: str, now: float | None = None) -> tuple[str, float]:
    """('OK'|'SLOW'|'DEAD'|'NEVER', age_seconds)."""
    now = now or time.time()
    with _lock:
        b = _beats.get(name)
    if not b:
        return "NEVER", -1.0
    age = now - b["ts"]
    warn_s, dead_s = _CADENCE.get(name, _DEFAULT_CADENCE)
    if age >= dead_s:
        return "DEAD", age
    if age >= warn_s:
        return "SLOW", age
    return "OK", age


def pulse() -> dict:
    """Full snapshot for the UI: daemons, latency stats, counters."""
    now = time.time()
    with _lock:
        beats = {k: dict(v) for k, v in _beats.items()}
        lat = {k: list(v) for k, v in _lat.items()}
        counters = {k: dict(v) for k, v in _counters.items()}
    daemons = {}
    for name, b in beats.items():
        status, age = daemon_status(name, now)
        daemons[name] = {"status": status, "age_s": round(age, 1),
                         "note": b.get("note", "")}
    latency = {}
    for name, vals in lat.items():
        latency[name] = {"n": len(vals),
                         "p50_ms": round(_pct(vals, 50) * 1000, 1),
                         "p95_ms": round(_pct(vals, 95) * 1000, 1),
                         "last_ms": round(vals[-1] * 1000, 1) if vals else 0.0}
    return {"daemons": daemons, "latency": latency, "counters": counters}
