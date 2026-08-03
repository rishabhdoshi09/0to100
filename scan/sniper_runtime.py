"""Stack-owned breakout sniper loop (Telegram alerts only — no broker orders).

The React/complete stack does not run Streamlit ``auto_scan``, so this process
keeps the legacy tick-stream sniper alive: start WebSocket when Kite is ready,
refresh watch names from ``latest_momentum_scan.json``, and alert on confirmed
single-stock breakouts.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "logs" / "sniper" / "runtime.json"


def _emit(kind: str, message: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    print(f"[{stamp}] SNIPER {kind:<9} {message}", flush=True)


def _market_open() -> bool:
    try:
        from research.autonomy import schedules as SCH
        from core.market_clock import now_ist
        from research.intelligence.data import nse_calendar as CAL

        return bool(SCH.market_is_open(now_ist(), CAL.load_holidays() or set()))
    except Exception:
        try:
            from core.market_clock import now_ist

            now = now_ist()
            mins = now.hour * 60 + now.minute
            return now.weekday() < 5 and (9 * 60 + 15) <= mins <= (15 * 60 + 30)
        except Exception:
            return False


def _write_status(extra: dict[str, Any] | None = None) -> None:
    try:
        from scan.breakout_sniper import sniper_status

        payload = {
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "market_open": _market_open(),
            **sniper_status(),
            **(extra or {}),
        }
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATUS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, STATUS_PATH)
    except Exception:
        pass


def run_loop(*, poll_seconds: float = 20.0) -> int:
    """Poll forever: start/refresh sniper in market hours, idle otherwise."""
    poll_seconds = max(5.0, float(poll_seconds))
    _emit("ONLINE", f"breakout sniper runtime · poll={poll_seconds:.0f}s")
    last_watch = -1
    while True:
        try:
            if not _market_open():
                _write_status({"phase": "off_session", "note": "market closed — sniper idle"})
                time.sleep(poll_seconds)
                continue

            from scan.breakout_sniper import refresh_watch_from_scan_store, start_sniper

            started = bool(start_sniper())
            watching = int(refresh_watch_from_scan_store(limit=80) or 0)
            if watching != last_watch:
                _emit(
                    "WATCH",
                    f"started={started} · watching={watching} "
                    f"(from latest_momentum_scan + watchlist)",
                )
                last_watch = watching
            if not started:
                _write_status(
                    {
                        "phase": "waiting_kite",
                        "note": "Kite login / ticker unavailable — run: python main.py login",
                    }
                )
            elif watching <= 0:
                _write_status(
                    {
                        "phase": "waiting_scan",
                        "note": "No sniper candidates yet — run Scan now in the terminal",
                    }
                )
            else:
                _write_status({"phase": "armed", "note": "tick stream watching breakout levels"})
        except Exception as exc:
            _emit("WARN", f"{type(exc).__name__}: {exc}")
            _write_status({"phase": "error", "note": str(exc)[:200]})
        time.sleep(poll_seconds)


def main() -> int:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
    except Exception:
        pass
    poll = float(os.getenv("QT_SNIPER_POLL_SECONDS", "20") or 20)
    return run_loop(poll_seconds=poll)


if __name__ == "__main__":
    raise SystemExit(main() or 0)
