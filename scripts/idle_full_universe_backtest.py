#!/usr/bin/env python3
"""Idle laptop watcher → enqueue full-universe signal backtest (research only).

When the machine has been idle for N seconds (default 600 = 10 minutes), or when
``--now`` is passed, enqueue ``FULL_UNIVERSE_BACKTEST`` on the market-ops worker.

Never places LIVE or paper orders. Skips if a backtest is already pending/running,
or if a successful run finished inside the cooldown window.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "logs" / "market_ops" / "idle_backtest.lock"
STATE = ROOT / "logs" / "market_ops" / "idle_backtest_state.json"
DEFAULT_IDLE_S = 600
DEFAULT_COOLDOWN_S = 6 * 60 * 60


def _idle_seconds() -> float | None:
    """Best-effort OS idle seconds. None if undetectable (headless / no GUI)."""
    # Linux: xprintidle (ms)
    try:
        import subprocess

        out = subprocess.check_output(["xprintidle"], stderr=subprocess.DEVNULL, text=True).strip()
        return float(out) / 1000.0
    except Exception:
        pass
    # macOS: HIDIdleTime nanoseconds via ioreg
    try:
        import re
        import subprocess

        out = subprocess.check_output(
            ["ioreg", "-c", "IOHIDSystem"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        match = re.search(r'"HIDIdleTime"\s*=\s*(\d+)', out)
        if match:
            return float(match.group(1)) / 1_000_000_000.0
    except Exception:
        pass
    # Optional override for testing / CI
    env = os.environ.get("QT_FAKE_IDLE_SECONDS", "").strip()
    if env:
        try:
            return float(env)
        except ValueError:
            return None
    return None


def _read_state() -> dict:
    try:
        if STATE.exists():
            return json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _write_state(payload: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _acquire_lock() -> bool:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl

        handle = LOCK.open("w")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return False
        handle.write(str(os.getpid()))
        handle.flush()
        # Keep handle alive for process lifetime
        globals()["_lock_handle"] = handle
        return True
    except Exception:
        return False


def enqueue_full_universe_backtest(*, requested_by: str) -> dict:
    if os.environ.get("QT_ALLOW_LIVE", "").strip() in {"1", "true", "TRUE", "yes"}:
        return {
            "accepted": False,
            "reason": "QT_ALLOW_LIVE is set — idle research watcher refuses to run near live mode",
        }
    sys.path.insert(0, str(ROOT))
    from operations.market_ops import FULL_UNIVERSE_BACKTEST, LANES
    from operations.store import OperationStore

    store = OperationStore(ROOT / "logs" / "market_ops" / "jobs.db")
    # OperationStore deduplicates PENDING/RUNNING of the same kind.
    operation, created = store.enqueue(
        FULL_UNIVERSE_BACKTEST,
        lane=LANES[FULL_UNIVERSE_BACKTEST],
        requested_by=requested_by,
    )
    return {
        "accepted": True,
        "created": created,
        "operation_id": operation.get("operation_id"),
        "status": operation.get("status"),
        "places_orders": False,
        "live_locked": True,
        "note": None if created else "Reused existing pending/running full-universe backtest",
    }


def maybe_trigger(*, idle_seconds: float, cooldown_s: float, force: bool = False) -> dict:
    state = _read_state()
    now = time.time()
    last = float(state.get("last_triggered_at") or 0)
    if not force and last and (now - last) < cooldown_s:
        return {
            "triggered": False,
            "reason": f"cooldown active · {(cooldown_s - (now - last)) / 60:.0f} min left",
        }

    idle = _idle_seconds()
    if force:
        reason = "manual --now"
    elif idle is None:
        return {"triggered": False, "reason": "idle time undetectable on this host (install xprintidle or run on laptop GUI)"}
    elif idle < idle_seconds:
        return {"triggered": False, "reason": f"idle {idle:.0f}s < threshold {idle_seconds:.0f}s", "idle_s": idle}
    else:
        reason = f"idle {idle:.0f}s >= {idle_seconds:.0f}s"

    result = enqueue_full_universe_backtest(requested_by=f"idle_watcher:{reason}")
    if result.get("accepted"):
        _write_state({
            "last_triggered_at": now,
            "last_reason": reason,
            "last_operation_id": result.get("operation_id"),
        })
    return {"triggered": bool(result.get("accepted")), "reason": reason, **result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idle-seconds", type=float, default=DEFAULT_IDLE_S,
                        help="Idle threshold before auto enqueue (default 600 = 10 min)")
    parser.add_argument("--cooldown-seconds", type=float, default=DEFAULT_COOLDOWN_S,
                        help="Minimum gap between auto runs (default 6h)")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--now", action="store_true", help="Enqueue immediately (you asked)")
    parser.add_argument("--once", action="store_true", help="Check once and exit")
    args = parser.parse_args()

    if not _acquire_lock() and not args.now:
        print("idle_backtest: another watcher already holds the lock", flush=True)
        return 1

    print(
        "idle_full_universe_backtest online · "
        f"idle>={args.idle_seconds:.0f}s · cooldown={args.cooldown_seconds:.0f}s · "
        "LIVE locked · research only",
        flush=True,
    )

    while True:
        payload = maybe_trigger(
            idle_seconds=float(args.idle_seconds),
            cooldown_s=float(args.cooldown_seconds),
            force=bool(args.now),
        )
        print(json.dumps(payload, default=str), flush=True)
        if args.now or args.once:
            return 0 if payload.get("triggered") or payload.get("accepted") else 0
        # --now only fires once
        args.now = False
        time.sleep(max(5.0, float(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
