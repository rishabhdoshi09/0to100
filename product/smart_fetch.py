"""Load-safe symbol research auto-download scheduler.

Design goals (old MacBook Air / QT_LOW_POWER):
  • Never scrape a whole universe from the UI path
  • One symbol job at a time (global single worker)
  • Per-symbol single-flight + cooldown
  • Cache-first: only fill MISSING evidence kinds
  • FastAPI request threads only enqueue — work runs in a daemon thread
  • Bounded queue so opening many symbols cannot pile up HTTP storms
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "logs" / "product" / "smart_fetch_status.json"

# Safe auto kinds: Screener exports + annual PDF attach.
# Guidance kinds stay manual/structured — auto may attach unparsed sources later
# only when the user explicitly asks (force full kinds).
DEFAULT_AUTO_KINDS = (
    "financial_history",
    "business_profile",
    "shareholding_history",
    "annual_report",
)

_COOLDOWN_S = 20 * 60
_MAX_QUEUE = 6
_STATUS_TTL_S = 6 * 60 * 60

_lock = threading.RLock()
_worker_started = False
_wake = threading.Event()
_queue: deque[str] = deque()
_jobs: dict[str, dict[str, Any]] = {}


def _low_power() -> bool:
    return str(os.getenv("QT_LOW_POWER", "") or "").strip().lower() in {"1", "true", "yes"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _persist() -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - _STATUS_TTL_S
    slim = {
        symbol: job
        for symbol, job in _jobs.items()
        if float(job.get("updated_at_epoch") or 0) >= cutoff
        or job.get("status") in {"QUEUED", "RUNNING"}
    }
    payload = {
        "updated_at": _now_iso(),
        "low_power": _low_power(),
        "queue": list(_queue),
        "jobs": slim,
    }
    tmp = STATUS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(STATUS_PATH)


def _load_disk() -> None:
    if not STATUS_PATH.exists():
        return
    try:
        payload = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return
    jobs = payload.get("jobs") or {}
    if isinstance(jobs, dict):
        for symbol, job in jobs.items():
            if not isinstance(job, dict):
                continue
            # Do not resume RUNNING after process restart — mark interrupted.
            if job.get("status") == "RUNNING":
                job = {
                    **job,
                    "status": "INTERRUPTED",
                    "message": "Worker restarted before this symbol finished",
                    "updated_at": _now_iso(),
                    "updated_at_epoch": time.time(),
                }
            _jobs[str(symbol).upper()] = job


def _ensure_worker() -> None:
    global _worker_started
    with _lock:
        if _worker_started:
            return
        _load_disk()
        thread = threading.Thread(target=_worker_loop, name="qt-smart-fetch", daemon=True)
        thread.start()
        _worker_started = True


def _missing_kinds(symbol: str, wanted: list[str]) -> list[str]:
    from reporting.evidence_intake import evidence_requirements

    status = evidence_requirements(symbol)
    missing: list[str] = []
    for item in status.get("requirements") or []:
        key = str(item.get("key") or "")
        if key not in wanted:
            continue
        # Skip when already analytically available OR annual report already attached.
        if item.get("available"):
            continue
        if key == "annual_report" and item.get("source_attached"):
            continue
        missing.append(key)
    return missing


def _run_job(symbol: str, job: dict[str, Any]) -> dict[str, Any]:
    from reporting.evidence_autofetch import autofetch_evidence

    wanted = list(job.get("kinds") or DEFAULT_AUTO_KINDS)
    force = bool(job.get("force"))
    refresh_screener = bool(job.get("refresh_screener", True))

    if force:
        kinds = wanted
    else:
        kinds = _missing_kinds(symbol, wanted)
        if not kinds:
            return {
                "accepted": True,
                "symbol": symbol,
                "attached_count": 0,
                "failed_count": 0,
                "results": [],
                "skipped": True,
                "message": "Nothing missing — cache/evidence already cover safe auto kinds",
                "honesty": "Smart fetch skips work that is already present.",
            }

    # Under low power, prefer Screener exports and at most one PDF hunt.
    if _low_power() and not force:
        preferred = [k for k in kinds if k in {"financial_history", "business_profile", "shareholding_history"}]
        optional = [k for k in kinds if k not in preferred][:1]
        kinds = preferred + optional

    report = autofetch_evidence(
        symbol,
        kinds=kinds,
        refresh_screener=refresh_screener,
        only_missing=not force,
        max_link_downloads=2 if _low_power() else 3,
    )
    report["message"] = (
        f"Attached {report.get('attached_count', 0)}, failed {report.get('failed_count', 0)} "
        f"for kinds: {', '.join(kinds)}"
    )
    return report


def _worker_loop() -> None:
    while True:
        _wake.wait(timeout=2.0)
        _wake.clear()
        while True:
            with _lock:
                if not _queue:
                    break
                symbol = _queue.popleft()
                job = _jobs.get(symbol)
                if not job or job.get("status") != "QUEUED":
                    continue
                job["status"] = "RUNNING"
                job["started_at"] = _now_iso()
                job["updated_at"] = job["started_at"]
                job["updated_at_epoch"] = time.time()
                job["message"] = "Downloading/exporting missing research sources…"
                _persist()
            try:
                report = _run_job(symbol, job)
                with _lock:
                    current = _jobs.get(symbol) or job
                    if current.get("job_id") != job.get("job_id"):
                        continue
                    current.update(
                        {
                            "status": "SUCCEEDED",
                            "finished_at": _now_iso(),
                            "updated_at": _now_iso(),
                            "updated_at_epoch": time.time(),
                            "message": report.get("message") or "Smart fetch completed",
                            "report": {
                                "attached_count": report.get("attached_count"),
                                "failed_count": report.get("failed_count"),
                                "results": report.get("results"),
                                "screener_note": report.get("screener_note"),
                                "honesty": report.get("honesty"),
                                "skipped": report.get("skipped"),
                            },
                        }
                    )
                    _persist()
            except Exception as exc:
                with _lock:
                    current = _jobs.get(symbol) or job
                    if current.get("job_id") != job.get("job_id"):
                        continue
                    current.update(
                        {
                            "status": "FAILED",
                            "finished_at": _now_iso(),
                            "updated_at": _now_iso(),
                            "updated_at_epoch": time.time(),
                            "message": f"Smart fetch failed: {exc}",
                            "error": str(exc),
                        }
                    )
                    _persist()


def schedule_symbol_fetch(
    symbol: str,
    *,
    force: bool = False,
    kinds: list[str] | None = None,
    refresh_screener: bool = True,
    requested_by: str = "ui",
) -> dict[str, Any]:
    """Enqueue a load-safe auto-download for one symbol. Never blocks on network."""
    from reporting.evidence_intake import clean_symbol

    symbol = clean_symbol(symbol)
    _ensure_worker()
    wanted = [str(k).strip() for k in (kinds or DEFAULT_AUTO_KINDS) if str(k).strip()]
    now = time.time()

    with _lock:
        existing = _jobs.get(symbol)
        if existing and existing.get("status") in {"QUEUED", "RUNNING"}:
            return {
                "accepted": True,
                "created": False,
                "symbol": symbol,
                "job": existing,
                "message": "Smart fetch already in flight for this symbol",
            }
        if (
            existing
            and not force
            and existing.get("status") == "SUCCEEDED"
            and now - float(existing.get("updated_at_epoch") or 0) < _COOLDOWN_S
        ):
            return {
                "accepted": True,
                "created": False,
                "symbol": symbol,
                "job": existing,
                "message": "Recent smart fetch still within cooldown — not re-queued",
                "cooldown_s": _COOLDOWN_S,
            }

        # Bound queue depth: drop oldest queued (not running) if full.
        while len(_queue) >= _MAX_QUEUE:
            dropped = _queue.popleft()
            dropped_job = _jobs.get(dropped)
            if dropped_job and dropped_job.get("status") == "QUEUED":
                dropped_job.update(
                    {
                        "status": "DROPPED",
                        "message": "Dropped — queue full (open fewer symbols or wait)",
                        "updated_at": _now_iso(),
                        "updated_at_epoch": now,
                    }
                )

        job = {
            "job_id": uuid.uuid4().hex[:16],
            "symbol": symbol,
            "status": "QUEUED",
            "kinds": wanted,
            "force": bool(force),
            "refresh_screener": bool(refresh_screener),
            "requested_by": requested_by,
            "queued_at": _now_iso(),
            "updated_at": _now_iso(),
            "updated_at_epoch": now,
            "message": "Queued behind other symbol research downloads",
            "low_power": _low_power(),
        }
        _jobs[symbol] = job
        if symbol not in _queue:
            _queue.append(symbol)
        _persist()

    _wake.set()
    return {
        "accepted": True,
        "created": True,
        "symbol": symbol,
        "job": job,
        "message": "Smart fetch queued (single-flight worker — will not overwhelm the backend)",
    }


def symbol_fetch_status(symbol: str) -> dict[str, Any]:
    from reporting.evidence_intake import clean_symbol

    symbol = clean_symbol(symbol)
    _ensure_worker()
    with _lock:
        job = _jobs.get(symbol)
        return {
            "symbol": symbol,
            "job": job,
            "queue_depth": len(_queue),
            "low_power": _low_power(),
            "cooldown_s": _COOLDOWN_S,
        }


def scheduler_snapshot() -> dict[str, Any]:
    _ensure_worker()
    with _lock:
        return {
            "queue": list(_queue),
            "queue_depth": len(_queue),
            "active": [j for j in _jobs.values() if j.get("status") in {"QUEUED", "RUNNING"}],
            "low_power": _low_power(),
            "max_queue": _MAX_QUEUE,
            "cooldown_s": _COOLDOWN_S,
            "default_kinds": list(DEFAULT_AUTO_KINDS),
        }
