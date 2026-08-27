"""Observable, resumable data refresh job registry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from data_platform.contracts import utc_now_iso


@dataclass(frozen=True)
class DataJobSpec:
    id: str
    label: str
    control: str | None
    description: str


JOBS: tuple[DataJobSpec, ...] = (
    DataJobSpec("security_master", "Security master refresh", None, "Rebuild symbol profiles from universe maps"),
    DataJobSpec("daily_prices", "Daily price update", "PREPARE_MARKET_DATA", "Bhavcopy build from local CSVs"),
    DataJobSpec("historical_backfill", "Historical price backfill", "PREPARE_MARKET_DATA", "Extend official EOD history"),
    DataJobSpec("corporate_actions", "Corporate action refresh", None, "Reload logs/ca_events.json"),
    DataJobSpec("fundamentals", "Fundamentals refresh", "REFRESH_STOCK_FUNDAMENTALS", "Per-symbol fundamentals fetch"),
    DataJobSpec("ownership", "Ownership refresh", None, "Ownership via fundamentals provider when available"),
    DataJobSpec("derived_ratios", "Derived ratio rebuild", None, "Recompute ratios from cached fundamentals"),
    DataJobSpec("market_scan", "Market scan rebuild", "MARKET_SCAN", "Whole-market scanner pass"),
    DataJobSpec("long_term_scan", "Long-term scan", "LONG_TERM_SCAN", "Long-term quality screen"),
    DataJobSpec("coverage_audit", "Coverage audit", None, "Universe coverage and remediation queue"),
)


def _bhav_status() -> dict[str, Any]:
    try:
        from data.bhavcopy_runtime import status
        return status(load_cache=False)
    except Exception:
        return {}


def jobs_payload() -> dict[str, Any]:
    bhav = _bhav_status()
    return {
        "generated_at": utc_now_iso(),
        "bhavcopy": bhav,
        "jobs": [
            {
                "id": j.id,
                "label": j.label,
                "control": j.control,
                "description": j.description,
                "trigger": "operations_control" if j.control else "manual_or_worker",
            }
            for j in JOBS
        ],
    }


def _enqueue_operation_kind(kind: str) -> dict[str, Any]:
    """Queue a market-operations lane job (same paths as terminal /api/controls)."""
    try:
        from operations.market_ops import LANES
        from operations.store import OperationStore

        import terminal_api as core

        core._ensure_ops_worker()
        operation, created = OperationStore(core.OPS_DB).enqueue(
            kind,
            lane=LANES[kind],
            requested_by="data_platform_job",
        )
        return {
            "ok": True,
            "operation_id": operation.get("operation_id"),
            "created": created,
            "kind": kind,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


_CONTROL_KIND_MAP = {
    "PREPARE_MARKET_DATA": "DATA_PREPARE",
    "MARKET_SCAN": "MARKET_SCAN",
    "LONG_TERM_SCAN": "LONG_TERM_SCAN",
}


def run_job(job_id: str) -> dict[str, Any]:
    spec = next((j for j in JOBS if j.id == job_id), None)
    if spec is None:
        return {"ok": False, "error": "unknown job", "job_id": job_id}
    if job_id == "fundamentals":
        return {
            "ok": True,
            "job_id": job_id,
            "message": "Fundamentals load lazily per symbol when you open Stock Intelligence.",
            "stats": __import__("fundamentals.lazy", fromlist=["cache_status"]).cache_status(),
        }
    if job_id == "corporate_actions":
        try:
            from data.corporate_actions import load_events
            n = len(load_events())
            return {"ok": True, "job_id": job_id, "loaded_events": n}
        except Exception as exc:
            return {"ok": False, "job_id": job_id, "error": str(exc)}
    if job_id == "coverage_audit":
        try:
            from data_platform.coverage import audit_universe
            from data_platform.security_master import supported_symbols
            return {"ok": True, "job_id": job_id, "report": audit_universe(supported_symbols(limit=60), limit=60)}
        except Exception as exc:
            return {"ok": False, "job_id": job_id, "error": str(exc)}
    if job_id == "derived_ratios":
        return {"ok": True, "job_id": job_id, "note": "Ratios computed on read via data_platform.ratios"}
    if job_id == "security_master":
        try:
            from data_platform.security_master import security_master_payload
            payload = security_master_payload(limit=50)
            return {"ok": True, "job_id": job_id, "count": payload.get("count", 0)}
        except Exception as exc:
            return {"ok": False, "job_id": job_id, "error": str(exc)}
    if job_id == "ownership":
        return {"ok": True, "job_id": job_id, "note": "Ownership loads with fundamentals provider when available"}
    if spec.control:
        kind = _CONTROL_KIND_MAP.get(spec.control)
        if not kind:
            return {
                "ok": False,
                "job_id": job_id,
                "error": "control not mapped to operations lane",
                "control": spec.control,
            }
        payload = _enqueue_operation_kind(kind)
        return {"job_id": job_id, **payload}
    return {"ok": False, "job_id": job_id, "error": "job has no runnable action"}
