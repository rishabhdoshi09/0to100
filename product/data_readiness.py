"""Single official-history readiness contract for desk, API, and scheduler.

The DATA lane is current only when official NSE history is current. Worker
liveness, broker login, and snapshot identity are separate planes. Mixing them
into one ``ready`` boolean produced contradictory Home states: HISTORY_CURRENT
with ``data_ready=false`` and a Waiting / "Getting the latest market data" UI.
"""
from __future__ import annotations

from typing import Any, Mapping

HISTORY_CURRENT = "HISTORY_CURRENT"
HISTORY_STALE = "HISTORY_STALE"
HISTORY_NOT_READY = "HISTORY_NOT_READY"
HISTORY_TOO_SHALLOW = "HISTORY_TOO_SHALLOW"
HISTORY_DATE_MISSING = "HISTORY_DATE_MISSING"
HISTORY_FUTURE_DATED = "HISTORY_FUTURE_DATED"
HISTORY_PUBLICATION_PENDING = "HISTORY_PUBLICATION_PENDING"
HISTORY_UNKNOWN = "HISTORY_UNKNOWN"

_NOT_CURRENT = frozenset({
    HISTORY_STALE,
    HISTORY_NOT_READY,
    HISTORY_TOO_SHALLOW,
    HISTORY_DATE_MISSING,
    HISTORY_FUTURE_DATED,
    HISTORY_PUBLICATION_PENDING,
    HISTORY_UNKNOWN,
})


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def project_official_data_readiness(
    *,
    freshness: Mapping[str, Any] | None = None,
    data: Mapping[str, Any] | None = None,
    bhav: Mapping[str, Any] | None = None,
    operations_running: bool | None = None,
    preparing: bool = False,
    data_failed: bool = False,
) -> dict[str, Any]:
    """Project one DATA-lane truth from official history freshness.

    ``data_ready`` equals ``history_current``. A down market-ops worker is an
    automation/runtime problem, not "market data is missing".
    """
    data_d = _as_dict(data)
    bhav_d = _as_dict(bhav if bhav is not None else data_d.get("bhavcopy"))
    fresh = _as_dict(freshness if freshness is not None else data_d.get("history_freshness"))
    if not fresh:
        fresh = {
            "current": bhav_d.get("current"),
            "reason_code": bhav_d.get("reason_code") or "",
            "available_session": bhav_d.get("available_session") or bhav_d.get("latest_date") or "",
            "expected_latest_completed_session": bhav_d.get("expected_latest_completed_session") or "",
            "stale_sessions": bhav_d.get("stale_sessions"),
        }

    reason = str(fresh.get("reason_code") or bhav_d.get("reason_code") or "").strip().upper()
    if reason in _NOT_CURRENT:
        history_current = False
    elif reason == HISTORY_CURRENT:
        history_current = True
    elif "current" in fresh:
        history_current = bool(fresh.get("current"))
    elif "current" in bhav_d:
        history_current = bool(bhav_d.get("current"))
    else:
        history_current = False
        reason = reason or HISTORY_UNKNOWN

    store_loaded = bool(data_d.get("ready") or bhav_d.get("ready"))
    # Official current history is the DATA lane. Do not AND with an unloaded
    # API-process pickle or a dead worker — those belong on other lanes.
    data_ready = bool(history_current)
    ops_running = None if operations_running is None else bool(operations_running)

    if data_failed and not data_ready and not preparing:
        lane_status, lane_code = "Problem", "FAILED"
    elif preparing and not history_current:
        lane_status, lane_code = "Working", "WORKING"
    elif history_current:
        lane_status, lane_code = "Ready", "READY"
    elif reason == HISTORY_STALE or reason == HISTORY_PUBLICATION_PENDING:
        lane_status, lane_code = "Waiting", "WAITING_DEPENDENCY"
    else:
        lane_status, lane_code = "Waiting", "WAITING"

    return {
        "data_ready": data_ready,
        "history_current": history_current,
        "reason_code": reason or (HISTORY_CURRENT if history_current else HISTORY_UNKNOWN),
        "store_loaded": store_loaded,
        "operations_running": ops_running,
        "lane_status": lane_status,
        "lane_status_code": lane_code,
        "available_session": fresh.get("available_session") or bhav_d.get("latest_date") or "",
        "expected_latest_completed_session": fresh.get("expected_latest_completed_session") or "",
        "stale_sessions": fresh.get("stale_sessions"),
        "preparing": bool(preparing),
        "data_failed": bool(data_failed),
        "source": "official_nse_bhavcopy",
    }


def official_history_is_data_ready(freshness: Mapping[str, Any] | None) -> bool:
    return bool(project_official_data_readiness(freshness=freshness)["data_ready"])
