#!/usr/bin/env python3
"""Read-only QuantTerm runtime status for operators and automation.

The probe reports only observable process/HTTP/runtime-file facts.  It never
turns absence into success and never mutates product state.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime_paths import logs_path


def _pid_alive(pid: int) -> bool:
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _http_ok(url: str, timeout: float = 1.5, attempts: int = 2) -> bool:
    """Bounded liveness probe tolerant of one transient busy response.

    QuantTerm can be CPU/I/O busy during first-run history preparation.  A single
    one-second timeout made the documented operator status command report DOWN
    even while /api/health simultaneously reported the stack READY.  Retry once,
    bounded, so a genuinely hung endpoint still fails closed within a few seconds.
    """
    tries = max(1, int(attempts))
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return int(response.status) == 200
        except Exception:
            if attempt + 1 < tries:
                time.sleep(0.1)
    return False


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _safe_pid(value: Any) -> int:
    """Parse persisted PID evidence without allowing malformed state to crash the probe."""
    try:
        pid = int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0
    return pid if pid > 1 else 0


def _safe_epoch(value: Any) -> float:
    """Parse a finite positive epoch; invalid persisted evidence means unknown."""
    try:
        epoch = float(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return epoch if math.isfinite(epoch) and epoch > 0 else 0.0


def runtime_status(now: float | None = None) -> dict[str, Any]:
    """Return deterministic status derived from live endpoints and persisted state."""
    now = time.time() if now is None else float(now)
    market_runtime = _read_json(logs_path("market_ops", "runtime.json"))
    worker_pid = _safe_pid(market_runtime.get("worker_pid"))
    heartbeat = _safe_epoch(market_runtime.get("heartbeat_epoch"))
    heartbeat_age = max(0.0, now - heartbeat) if heartbeat > 0 else None
    worker_alive = _pid_alive(worker_pid)
    # Canonical terminal_api uses a 10s freshness window. Keep the operator
    # probe on the same contract so two official status surfaces cannot disagree.
    worker_fresh = bool(worker_alive and heartbeat_age is not None and heartbeat_age <= 10.0)

    api_ok = _http_ok("http://127.0.0.1:8765/api/health")
    desk_ok = _http_ok("http://127.0.0.1:5173/")
    report_ok = _http_ok("http://127.0.0.1:8766/health")

    return {
        "status_schema": 1,
        "observed_at_epoch": now,
        "market_api": {"healthy": api_ok, "source": "http://127.0.0.1:8765/api/health"},
        "desk": {"healthy": desk_ok, "source": "http://127.0.0.1:5173/"},
        "report_api": {"healthy": report_ok, "source": "http://127.0.0.1:8766/health"},
        "market_ops": {
            "healthy": worker_fresh,
            "pid": worker_pid or None,
            "pid_alive": worker_alive,
            "heartbeat_age_seconds": round(heartbeat_age, 3) if heartbeat_age is not None else None,
            "source": str(logs_path("market_ops", "runtime.json")),
        },
        "ready": bool(api_ok and desk_ok and worker_fresh),
    }


def main() -> int:
    status = runtime_status()
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0 if status["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
