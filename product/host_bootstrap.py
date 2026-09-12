"""Fresh-host bootstrap for durable QuantTerm data stores.

The normal repair ladder assumes there is already a canonical history store to
repair.  That is the correct behaviour during a session (a request for three
missing symbols must not unexpectedly download hundreds of sessions), but a
brand-new installed host needs one explicit owner for the *first* build.

This module is that owner.  It is called once by the installed host supervisor
before user-facing services start.  It never fabricates data and it does not
turn an unavailable official archive into a green result: failure is persisted
and the rest of the desk may still start in a truthful DEGRADED state so its
normal fallback machinery can operate.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from core.runtime_paths import runtime_path

STATUS_PATH = Path("state") / "host_bootstrap.json"
MIN_WHOLE_MARKET_SYMBOLS = 200


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def bootstrap_host_state(*, should_stop: Callable[[], bool] | None = None) -> dict[str, Any]:
    """Prepare durable stores and attempt the first canonical history build.

    Returns a truthful report rather than raising on a market-data outage.  A
    safety/bootstrap programming error is recorded as FAILED; an unavailable
    official history source is DEGRADED so the normal scan/data fallbacks still
    get a chance after the desk starts.
    """
    report: dict[str, Any] = {
        "schema_version": 1,
        "started_at": _now(),
        "state": "STARTING",
        "sqlite": {},
        "history_before": {},
        "history_after": {},
        "history_build_attempted": False,
        "error": "",
    }
    sqlite_failed = False
    try:
        try:
            from product.sqlite_runtime import bootstrap_product_stores

            report["sqlite"] = bootstrap_product_stores() or {}
        except Exception as exc:
            # Durable product stores are required for an unattended desk.  A
            # history fallback cannot compensate for a state store we cannot
            # initialise, so keep the evidence and fail the supervisor closed.
            sqlite_failed = True
            report["sqlite"] = {"state": "FAILED", "error": f"{type(exc).__name__}: {exc}"[:300]}

        from data.bhavcopy_runtime import status as history_status

        before = history_status(load_cache=True)
        report["history_before"] = before
        ready = bool(before.get("ready")) and int(before.get("sessions") or 0) >= int(before.get("minimum_sessions") or 60)
        if not ready:
            report["history_build_attempted"] = True
            try:
                from data.bhavcopy_store import DEFAULT_DAYS, HistoryAcquisitionCancelled, build_store

                build_store(days=DEFAULT_DAYS, should_stop=should_stop)
            except HistoryAcquisitionCancelled:
                report["state"] = "CANCELLED"
                report["error"] = "history bootstrap cancelled during shutdown"
                report["finished_at"] = _now()
                _atomic_json(runtime_path(STATUS_PATH), report)
                return report
            except Exception as exc:
                # Do not hide the outage; scanner/provider fallbacks may still be
                # useful after startup, so this is DEGRADED rather than a fatal
                # host-supervisor exception.
                report["error"] = f"{type(exc).__name__}: {exc}"[:300]

        after = history_status(load_cache=True)
        report["history_after"] = after
        symbols = int(after.get("symbols") or 0)
        sessions = int(after.get("sessions") or 0)
        min_sessions = int(after.get("minimum_sessions") or 60)
        if sqlite_failed:
            report["state"] = "FAILED"
            if not report["error"]:
                report["error"] = str(report["sqlite"].get("error") or "durable product-store bootstrap failed")
        elif bool(after.get("ready")) and symbols >= MIN_WHOLE_MARKET_SYMBOLS and sessions >= min_sessions:
            report["state"] = "READY"
        else:
            report["state"] = "DEGRADED"
            if not report["error"]:
                report["error"] = (
                    f"canonical official history is not ready: symbols={symbols}, "
                    f"sessions={sessions}, minimum_sessions={min_sessions}"
                )
    except Exception as exc:
        report["state"] = "FAILED"
        report["error"] = f"{type(exc).__name__}: {exc}"[:300]
    report["finished_at"] = _now()
    try:
        _atomic_json(runtime_path(STATUS_PATH), report)
    except Exception:
        pass
    return report
