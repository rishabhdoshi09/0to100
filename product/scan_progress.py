"""Shared scan/Top-Stocks progress so the UI and console can show a real ETA.

Both the autonomy job and the market-operations scan write here. The desk
reads it without waiting for a finished scan file.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def default_progress_path() -> Path:
    return Path(__file__).resolve().parents[1] / "logs" / "product" / "scan_progress.json"


DEFAULT_PROGRESS_PATH = default_progress_path()

_started_at: float | None = None
_last_write = 0.0
_WRITE_GAP_S = 0.4


def eta_seconds(current: int, total: int, started_at: float | None, *, now: float | None = None) -> float | None:
    """Linear ETA from observed rate. None until there is a real denominator and pace."""
    total = int(total or 0)
    current = int(current or 0)
    if total <= 0 or current <= 0 or not started_at:
        return None
    elapsed = float(now if now is not None else time.time()) - float(started_at)
    if elapsed < 0.4:
        return None
    rate = current / elapsed
    if rate <= 0:
        return None
    remain = (total - current) / rate
    return max(0.0, float(remain))


def eta_label(seconds: float | None) -> str:
    if seconds is None:
        return ""
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return ""
    if value < 0:
        return ""
    if value < 15:
        return "under 15s"
    if value < 60:
        return f"about {int(round(value / 5.0) * 5)}s"
    minutes = max(1, int(round(value / 60.0)))
    if minutes == 1:
        return "about 1 min"
    return f"about {minutes} min"


def write_progress(
    *,
    current: int = 0,
    total: int = 0,
    stage: str = "SCANNING",
    source: str = "",
    path: str | Path | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Persist a throttled progress snapshot. Safe to call from a scan thread."""
    global _started_at, _last_write
    stamp = float(now if now is not None else time.time())
    stage = str(stage or "SCANNING").upper()
    if _started_at is None or stage in {"STARTING", "LOADING_UNIVERSE"}:
        _started_at = stamp
    current = max(0, int(current or 0))
    total = max(0, int(total or 0))
    remaining = eta_seconds(current, total, _started_at, now=stamp)
    payload = {
        "active": True,
        "stage": stage,
        "current": current,
        "total": total,
        "pct": round((current / total) * 100, 1) if total > 0 else None,
        "eta_s": None if remaining is None else round(remaining, 1),
        "eta_label": eta_label(remaining),
        "elapsed_s": round(max(0.0, stamp - float(_started_at)), 1),
        "started_at": _started_at,
        "updated_at": stamp,
        "source": str(source or ""),
        "error": "",
    }
    if current < 3 or current == total or stamp - _last_write >= _WRITE_GAP_S:
        _atomic_json(path or DEFAULT_PROGRESS_PATH, payload)
        _last_write = stamp
    return payload


def finish_progress(
    *,
    records: int = 0,
    setups: int = 0,
    error: str = "",
    path: str | Path | None = None,
) -> dict[str, Any]:
    global _started_at, _last_write
    payload = {
        "active": False,
        "stage": "FAILED" if error else "DONE",
        "current": 0,
        "total": 0,
        "pct": 100.0 if not error else None,
        "eta_s": 0.0,
        "eta_label": "",
        "elapsed_s": 0.0,
        "started_at": _started_at,
        "updated_at": time.time(),
        "source": "",
        "error": str(error or ""),
        "records": int(records or 0),
        "setups": int(setups or 0),
    }
    _atomic_json(path or DEFAULT_PROGRESS_PATH, payload)
    _started_at = None
    _last_write = 0.0
    return payload


def read_progress(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path or DEFAULT_PROGRESS_PATH)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if data.get("active"):
                try:
                    age = time.time() - float(data.get("updated_at") or 0)
                except (TypeError, ValueError):
                    age = 9999.0
                # A crashed scan used to leave active=true and a leftover ETA.
                if age > 90:
                    data = {
                        **data,
                        "active": False,
                        "eta_s": None,
                        "eta_label": "",
                        "stale": True,
                    }
            return data
    except Exception:
        pass
    return {
        "active": False,
        "stage": "",
        "current": 0,
        "total": 0,
        "pct": None,
        "eta_s": None,
        "eta_label": "",
        "error": "",
    }


def _atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, target)
        tmp.unlink(missing_ok=True)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
