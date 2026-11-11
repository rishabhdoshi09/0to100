"""Warn the owner before battery drain takes the USB runtime down.

Owner's Mac, 2026-11-11 ~09:43 IST: the laptop ran on battery to ~12%, the
USB "Expansion" drive lost power, the runtime sparsebundle detached and the
desk stopped mid-session. Nothing told the owner the charger was out.

This watcher reads ``pmset -g batt`` once a minute (macOS only) and sends a
Telegram alert when the host switches to battery, repeats every 15 minutes
while on battery at or below LOW_BATTERY_PCT, and confirms when AC returns.
Read-only: it never changes power settings.
"""
from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Sequence

LOW_BATTERY_PCT = 40
REPEAT_S = 15 * 60
POLL_S = 60.0


def _pmset(argv: Sequence[str]) -> str:
    return subprocess.run(list(argv), capture_output=True, text=True, timeout=10).stdout


def read_power(run: Callable[[Sequence[str]], str] | None = None) -> dict[str, Any] | None:
    """{"source": "AC"|"BATTERY", "percent": int|None} or None when unknown."""
    if run is None:
        if sys.platform != "darwin":
            return None
        run = _pmset
    try:
        text = str(run(["pmset", "-g", "batt"]) or "")
    except Exception:
        return None
    first = text.splitlines()[0] if text else ""
    if "AC Power" in first:
        source = "AC"
    elif "Battery Power" in first:
        source = "BATTERY"
    else:
        return None
    match = re.search(r"(\d{1,3})%", text)
    return {"source": source, "percent": int(match.group(1)) if match else None}


class PowerWatch:
    def __init__(self, send: Callable[[str], Any], *, now: Callable[[], float] = time.time,
                 low_pct: int = LOW_BATTERY_PCT, repeat_s: float = REPEAT_S) -> None:
        self.send = send
        self.now = now
        self.low_pct = low_pct
        self.repeat_s = repeat_s
        self.state: str | None = None
        self.last_alert = 0.0

    def observe(self, power: dict[str, Any] | None) -> str | None:
        if not power:
            return None
        source = power.get("source")
        pct = power.get("percent")
        pct_text = f"{pct}%" if pct is not None else "unknown %"
        message = None
        if source == "BATTERY":
            low = pct is not None and pct <= self.low_pct
            if self.state != "BATTERY" or (low and self.now() - self.last_alert >= self.repeat_s):
                message = (
                    f"⚠️ QuantTerm host is on BATTERY ({pct_text}). Plug in the charger: "
                    "at low battery the USB runtime disk loses power and the desk stops."
                )
        elif source == "AC" and self.state == "BATTERY":
            message = f"✅ QuantTerm host is back on AC power ({pct_text})."
        if message:
            try:
                self.send(message)
            except Exception:
                pass
            self.last_alert = self.now()
            print(f"[HOST POWER] {message}", file=sys.stderr, flush=True)
        self.state = source or self.state
        return message


def _telegram_send(message: str) -> None:
    from alerts.telegram_alerts import AlertEngine

    AlertEngine().send(message)


def run_power_watch(stop: threading.Event, *, send: Callable[[str], Any] | None = None,
                    run: Callable[[Sequence[str]], str] | None = None,
                    poll_s: float = POLL_S) -> None:
    watch = PowerWatch(send or _telegram_send)
    while not stop.is_set():
        try:
            watch.observe(read_power(run))
        except Exception:
            pass
        stop.wait(poll_s)


def start_power_watch(stop: threading.Event) -> threading.Thread | None:
    """Daemon thread on macOS hosts; a no-op elsewhere."""
    if sys.platform != "darwin":
        return None
    thread = threading.Thread(target=run_power_watch, args=(stop,),
                              name="quantterm-power-watch", daemon=True)
    thread.start()
    return thread
