"""Tell the owner when the desk is paused because runtime storage is gone.

Owner's Mac, 16-18 Nov 2026: the USB drive that holds the runtime was not
mounted for about two days. Every launchd start failed the storage preflight
("external volume is not mounted") before the host supervisor existed, and a
loss while running only printed "[HOST STORAGE] LOST" to the launchd log. The
restart-limit alert keeps its dedupe state on the runtime disk, which is the
disk that was missing. Nobody was told; two sessions were lost.

This module is the missing notification. Its state lives on the internal disk
(next to the launchd logs), never on the runtime it reports about.

- An outage is alerted only after ``GRACE_S``. A lid-close wake often gives a
  few seconds of ``Errno 5`` that self-recovery fixes; that is not news.
- While the outage lasts, a reminder goes out every ``REMIND_S``.
- When storage comes back after an alerted outage, one "back" message is sent.
- It is active only for the external-storage contract
  (``QT_STORAGE_PREFLIGHT_REQUIRED``) or when ``QT_STORAGE_ALERT_STATE`` names
  a state file explicitly, so tests and other hosts never touch the owner's
  file.
- Nothing here raises, and an alert never changes trading or desk state.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

GRACE_S = 180.0
REMIND_S = 3 * 60 * 60
RETRY_S = 5 * 60            # after an undelivered alert (e.g. the network is down too)
_LOCK = threading.Lock()


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def state_path() -> Path | None:
    explicit = str(os.environ.get("QT_STORAGE_ALERT_STATE") or "").strip()
    if explicit:
        return Path(explicit).expanduser()
    if not _truthy(os.environ.get("QT_STORAGE_PREFLIGHT_REQUIRED")):
        return None
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "QuantTerm" / "storage_alert.json"
    return Path.home() / ".local" / "state" / "quantterm" / "storage_alert.json"


def _read(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _clear(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _default_sender(message: str):
    from product.host_alerts import send_operational_alert

    return send_operational_alert(message)


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _clock_text(epoch: float) -> str:
    try:
        from zoneinfo import ZoneInfo
        from datetime import datetime

        return datetime.fromtimestamp(epoch, ZoneInfo("Asia/Kolkata")).strftime("%a %d %b %H:%M IST")
    except Exception:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(epoch))


def _delivered(result: Any) -> bool:
    return bool(getattr(result, "delivered", False))


def note_storage_down(
    reason: str,
    *,
    sender: Callable[[str], Any] | None = None,
    now: Callable[[], float] = time.time,
) -> str:
    """Record that runtime storage is unavailable; alert when it is news.

    Returns what happened: DISABLED, RECORDED, ALERTED, REMINDED, WAITING or
    ERROR. Safe to call on every failed attempt (each launchd restart, each
    recovery poll): it only reads and writes one small file.
    """
    path = state_path()
    if path is None:
        return "DISABLED"
    try:
        with _LOCK:
            state = _read(path)
            t = float(now())
            text = str(reason or "runtime storage unavailable").strip()[:300]
            since = float(state.get("down_since") or 0.0)
            if since <= 0 or since > t:
                _write(path, {"schema_version": 1, "down_since": t, "reason": text,
                              "alerted_epoch": 0.0, "alerts": 0})
                return "RECORDED"
            alerted = float(state.get("alerted_epoch") or 0.0)
            if t - since < GRACE_S:
                return "WAITING"
            if alerted and t - alerted < REMIND_S:
                return "WAITING"
            attempted = float(state.get("attempt_epoch") or 0.0)
            if attempted and t - attempted < RETRY_S:
                return "WAITING"
            first = not alerted
            if first:
                message = (
                    "QuantTerm PAUSED: runtime storage unavailable since "
                    f"{_clock_text(since)} ({_duration(t - since)}).\n"
                    f"Detail: {text}\n"
                    "No scan, paper cycle or EOD data until it is back. The desk "
                    "resumes by itself: check that the external runtime drive is "
                    "connected and powered, and the Mac is awake on AC."
                )
            else:
                message = (
                    "QuantTerm STILL PAUSED: runtime storage unavailable for "
                    f"{_duration(t - since)} (since {_clock_text(since)}).\n"
                    f"Detail: {text}"
                )
            result = (sender or _default_sender)(message)
            state.update({
                "reason": text,
                "attempt_epoch": t,
                "last_alert": getattr(result, "to_dict", lambda: {})(),
            })
            if not _delivered(result):
                _write(path, state)
                return "ERROR"
            state.update({"alerted_epoch": t, "alerts": int(state.get("alerts") or 0) + 1})
            _write(path, state)
            return "ALERTED" if first else "REMINDED"
    except Exception:
        return "ERROR"


def note_storage_recovered(
    *,
    sender: Callable[[str], Any] | None = None,
    now: Callable[[], float] = time.time,
) -> str:
    """Close an outage. Sends one "back" message only if the owner was alerted."""
    path = state_path()
    if path is None:
        return "DISABLED"
    try:
        with _LOCK:
            state = _read(path)
            if not state:
                return "NONE"
            since = float(state.get("down_since") or 0.0)
            alerted = float(state.get("alerted_epoch") or 0.0)
            _clear(path)
            if not alerted:
                return "CLEARED"
            message = (
                "QuantTerm storage is back after "
                f"{_duration(float(now()) - since)}; the desk is starting again "
                "(the startup scan takes about 5 minutes)."
            )
            result = (sender or _default_sender)(message)
            return "ANNOUNCED" if _delivered(result) else "ERROR"
    except Exception:
        return "ERROR"


def watch_outage(
    is_lost: Callable[[], bool],
    reason: Callable[[], str],
    *,
    stop: threading.Event,
    poll_s: float = 30.0,
) -> threading.Thread:
    """While the running desk is paused on lost storage, keep the outage noted."""

    def run() -> None:
        while not stop.is_set() and is_lost():
            note_storage_down(reason())
            stop.wait(poll_s)

    thread = threading.Thread(target=run, name="quantterm-storage-outage-alert", daemon=True)
    thread.start()
    return thread
