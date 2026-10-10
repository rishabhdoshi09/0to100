"""Local API bridge for the dedicated QuantTerm terminal.

Authoritative market/research stores remain in Python. User-requested market
operations are dispatched to a dedicated worker plane; PAPER autonomy remains a
separate execution/learning lane and is never allowed to block scans.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from logger import quiet_uvicorn_health_access
from core.runtime_paths import logs_dir, logs_path

ROOT = Path(__file__).resolve().parent
OPS_ROOT = logs_dir() / "market_ops"
OPS_RUNTIME = OPS_ROOT / "runtime.json"
OPS_DB = OPS_ROOT / "jobs.db"

app = FastAPI(title="QuantTerm Terminal API", version="0.4.0")
quiet_uvicorn_health_access()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def _log_unhandled(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        raise exc
    print(f"[API] unhandled {request.method} {request.url.path}: {exc}", flush=True)
    return JSONResponse(
        {"ok": False, "error": str(exc)[:300], "path": request.url.path},
        status_code=500,
    )

@app.middleware("http")
async def _public_mutation_boundary(request: Request, call_next):
    """Require explicit operator authorization for mutations in public mode.

    The guard covers every POST/PUT/PATCH/DELETE route registered on this
    FastAPI application, including product-extension routes. It deliberately
    does not trust client IPs or forwarding headers.
    """
    from product.public_access import authorize_mutation

    access = authorize_mutation(request.method, request.headers)
    if not access.allowed:
        return JSONResponse(
            {
                "detail": access.detail,
                "code": access.code,
                "public_read_only": True,
            },
            status_code=403,
            headers={"Cache-Control": "no-store"},
        )
    return await call_next(request)


@app.get("/api/access")
def access_mode() -> dict[str, Any]:
    """Expose the non-secret access policy for the public desk."""
    from product.public_access import access_projection

    return access_projection()


_ops_process: subprocess.Popen | None = None


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
        if result != result:
            return None
        return result
    except Exception:
        return None


def _json_file(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _json_safe(value: Any) -> Any:
    """JSON-encode without NaN/Inf so the RecoWealth desk never 500s on a float."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    return value


DASHBOARD_SCAN_RECORD_LIMIT = 80

_warm_threads: dict[str, threading.Thread] = {}
_warm_guard = threading.Lock()


def _schedule_warm(name: str, target) -> None:
    """Run a one-shot warmer without holding the desk request."""
    with _warm_guard:
        current = _warm_threads.get(name)
        if current is not None and current.is_alive():
            return
        thread = threading.Thread(target=target, name=f"quantterm-{name}", daemon=True)
        _warm_threads[name] = thread
        thread.start()


def _warm_regime() -> None:
    try:
        from core.regime_engine import compute_regime
        compute_regime()
    except Exception:
        return


def _warm_bhavcopy_cache() -> None:
    try:
        from data.bhavcopy_runtime import status as bhavcopy_status
        bhavcopy_status(load_cache=True)
    except Exception:
        return


def _dashboard_num(row: dict[str, Any], *keys: str) -> float:
    for key in keys:
        try:
            value = float(row.get(key) or 0.0)
        except (TypeError, ValueError):
            continue
        if value == value:
            return value
    return 0.0


def _dashboard_record_rank(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        _dashboard_num(row, "composite", "sepa_score", "score"),
        _dashboard_num(row, "sepa_score", "score"),
        _dashboard_num(row, "score"),
    )


def _slim_ranked_records(
    payload: dict[str, Any],
    *,
    limit: int = DASHBOARD_SCAN_RECORD_LIMIT,
) -> dict[str, Any]:
    """Keep Home fast: top-ranked rows only. Universe size stays the real count."""
    if not isinstance(payload, dict):
        return payload
    records = [row for row in (payload.get("records") or []) if isinstance(row, dict)]
    cap = max(1, int(limit))
    ranked = sorted(records, key=_dashboard_record_rank, reverse=True)[:cap]
    out = dict(payload)
    out["records"] = ranked
    if "universe_size" in payload:
        out["universe_size"] = int(payload.get("universe_size") or 0) or len(records)
    out["dashboard_record_limit"] = cap
    out["dashboard_records_shown"] = len(ranked)
    return out


def _empty_dashboard(error: str, scan: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = scan if isinstance(scan, dict) else _scan_payload()
    payload = _slim_ranked_records(payload)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "market": {
            "available": False,
            "health": "Unavailable",
            "summary": "Market API is degraded; cards use the last readable scan.",
            "trade_stance": "Do not infer a market stance from missing data.",
            "breadth": "—",
            "leaders": [],
            "laggards": [],
            "nifty_change_1d": None,
            "nifty_change_5d": None,
            "vix": None,
            "nifty_price": None,
            "technical_details": {},
        },
        "scan": payload,
        "long_term": {"available": False, "summary": {}, "records": [], "job": {}},
        "paper": {
            "available": False,
            "enabled": False,
            "supervisor_running": False,
            "capital": 0.0,
            "equity": 0.0,
            "equity_curve": [],
            "open_risk": 0.0,
            "risk_per_trade_pct": 0.01,
            "max_positions": 0,
            "open_positions": [],
            "closed_trades": [],
            "refusals": [],
            "last_cycle": {},
        },
        "autonomy": {
            "available": False,
            "running": False,
            "process_running": False,
            "state": "UNKNOWN",
            "plain_state": "Autonomy status unavailable.",
            "explanation": error,
            "heartbeat_ist": "",
            "scheduler_owner_pid": None,
            "active_job": {},
            "new_paper_entries": False,
            "existing_exits": False,
            "research_enabled": False,
            "capability_notes": [],
            "active_failures": [],
            "recent_dialogue": [],
            "recent_transitions": [],
            "jobs": {},
            "jobs_recent": [],
            "owner_state": {},
            "live_feed": {},
            "last_cycle": {},
        },
        "operations": {
            "available": False,
            "running": False,
            "worker_pid": None,
            "heartbeat": "",
            "active_lanes": {},
            "counts": {},
            "active": [],
            "recent": [],
            "latest": {},
        },
        "news": {
            "available": False,
            "stats": {"total": 0, "important": 0, "fno_linked": 0, "macro": 0, "sources": 0},
            "articles": [],
            "source_health": [],
            "latest_refresh": {},
        },
        "fno": {"available": False, "source": "unavailable", "mapped_underlyings": 0, "underlyings": [], "exclusions": []},
        "data": {
            "ready": False,
            "snapshot": {"ready": False, "snapshot_id": "", "latest_date": "", "source": ""},
            "bhavcopy": {"ready": False, "symbols": 0, "sessions": 0, "latest_date": "", "csv_files": 0, "cache_exists": False},
            "scan_saved": bool(payload.get("available")),
            "scan_records": len(payload.get("records", []) or []),
            "long_term_saved": False,
            "long_term_records": 0,
            "blockers": [error] if error else [],
        },
        "conviction": [],
        "scan_progress": {"active": False, "eta_label": "", "current": 0, "total": 0},
        "daily_wrap": [],
        "error": error,
    }


def _fresh_epoch(value: Any, max_age_s: float = 10.0) -> bool:
    try:
        age = time.time() - float(value)
        return 0 <= age <= max_age_s
    except Exception:
        return False


def _ops_runtime_payload() -> dict[str, Any]:
    runtime = _json_file(OPS_RUNTIME, {})
    running = bool(runtime.get("process_running")) and _fresh_epoch(runtime.get("heartbeat_epoch"))
    lock_pid = 0
    if not running:
        try:
            from operations.store import live_lock_owner_pid
            lock_pid = live_lock_owner_pid(OPS_ROOT / "worker.lock")
        except Exception:
            lock_pid = 0
        if lock_pid:
            running = True
            runtime = {
                **runtime,
                "worker_pid": lock_pid,
                "process_running": True,
                "recovering": True,
            }
    return {
        **runtime,
        "running": running,
        "process_running": bool(runtime.get("process_running")) or bool(lock_pid),
    }


def _ensure_ops_worker(*, wait: bool = True) -> dict[str, Any]:
    """Start the dedicated market-operations worker when it is not healthy."""
    global _ops_process
    from operations.store import pid_is_alive

    runtime = _ops_runtime_payload()
    pid = runtime.get("worker_pid")
    if runtime.get("running") and pid_is_alive(pid):
        return runtime
    if _ops_process is not None and _ops_process.poll() is None and pid_is_alive(_ops_process.pid):
        return runtime
    env = os.environ.copy()
    existing = str(env.get("PYTHONPATH") or "").strip()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT)] + ([existing] if existing else []))
    _ops_process = subprocess.Popen(
        [sys.executable, "-u", "-m", "operations.market_ops"],
        cwd=str(ROOT),
        env=env,
    )
    if not wait:
        return _ops_runtime_payload()
    deadline = time.time() + 2.5
    while time.time() < deadline:
        time.sleep(0.1)
        runtime = _ops_runtime_payload()
        if runtime.get("running"):
            break
        if _ops_process.poll() is not None:
            break
    return runtime


@app.on_event("startup")
def _startup() -> None:
    try:
        _ensure_ops_worker()
    except RuntimeError:
        # Market Operations is launcher-owned. A late first heartbeat must not
        # take the desk API down. Home shows WAITING / PREPARING instead.
        return


@app.on_event("shutdown")
def _shutdown() -> None:
    global _ops_process
    if _ops_process is not None and _ops_process.poll() is None:
        _ops_process.terminate()
        try:
            _ops_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _ops_process.kill()
    _ops_process = None


def _market_payload() -> dict:
    try:
        from product.market_view import peek_cached_market_view
        market = peek_cached_market_view()
        _schedule_warm("regime", _warm_regime)
        if market is None:
            return {
                "available": False,
                "health": "Unavailable",
                "summary": "Market regime is still assembling from index history.",
                "trade_stance": "Do not infer a market stance from missing data.",
                "breadth": "—",
                "leaders": [],
                "laggards": [],
                "nifty_change_1d": None,
                "nifty_change_5d": None,
                "vix": None,
                "nifty_price": None,
                "technical_details": {},
            }
        available = bool(getattr(market, "available", True)) and str(market.health or "") != "Unavailable"
        return {
            "available": available,
            "health": market.health,
            "summary": market.summary,
            "trade_stance": market.trade_stance,
            "breadth": market.breadth,
            "leaders": list(market.leaders) if available else [],
            "laggards": list(market.laggards) if available else [],
            "nifty_change_1d": _safe_float(market.nifty_change_1d) if available else None,
            "nifty_change_5d": _safe_float(market.nifty_change_5d) if available else None,
            "vix": _safe_float(market.vix) if available else None,
            "nifty_price": _safe_float(getattr(market, "nifty_price", None)) if available else None,
            "technical_details": dict(getattr(market, "technical_details", {}) or {}),
        }
    except Exception as exc:
        return {
            "available": False,
            "health": "Unavailable",
            "summary": "Market regime projection is unavailable.",
            "trade_stance": "Do not infer a market stance from missing data.",
            "breadth": "—",
            "leaders": [],
            "laggards": [],
            "nifty_change_1d": None,
            "nifty_change_5d": None,
            "vix": None,
            "nifty_price": None,
            "technical_details": {},
            "error": str(exc),
        }


def _scan_payload() -> dict:
    try:
        from product.scan_store import load_scan, resolved_scan_path
        payload = load_scan(resolved_scan_path()) or {}
        records = [dict(row) for row in (payload.get("records", []) or []) if isinstance(row, dict)]
        provenance = payload.get("provenance")
        return {
            "available": bool(payload),
            "scanned_at": payload.get("scanned_at", ""),
            "universe_size": int(payload.get("universe_size", 0) or 0),
            "summary": dict(payload.get("summary", {}) or {}),
            "records": records,
            # WHEN THE SCAN RAN vs WHICH SESSION IT READ are different facts.
            # The desk must be able to render them separately.
            "provenance": dict(provenance) if isinstance(provenance, dict) else {},
            # Preserve canonical coverage from this same atomic saved scan.
            # Previously api.runtime reparsed the entire scan file just to
            # recover these four metadata fields on every dashboard request.
            "requested_universe": int(payload.get("requested_universe", payload.get("universe_size", 0)) or 0),
            "coverage_state": str(payload.get("coverage_state") or "UNKNOWN"),
            "coverage_warning": str(payload.get("coverage_warning") or ""),
            "coverage": dict(payload.get("coverage") or {}),
        }
    except Exception as exc:
        return {
            "available": False,
            "scanned_at": "",
            "universe_size": 0,
            "summary": {},
            "records": [],
            "provenance": {},
            "requested_universe": 0,
            "coverage_state": "UNKNOWN",
            "coverage_warning": "",
            "coverage": {},
            "error": str(exc),
        }


def _recent_autonomy_jobs(limit: int = 60) -> list[dict]:
    try:
        from research.autonomy import default_root
        db_path = default_root() / "jobs.db"
        if not db_path.exists():
            return []
        connection = sqlite3.connect(str(db_path), timeout=2.0)
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                "SELECT job_id,job_type,status,attempt,critical,scheduled_for,started_at,finished_at,"
                "result_summary,error_code,error_message,blocked_on,blocked_reason "
                "FROM jobs ORDER BY created_at DESC LIMIT ?",
                (max(1, min(int(limit), 200)),),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            connection.close()
    except Exception:
        return []


def _latest_autonomy_job(job_types: set[str]) -> dict:
    for job in _recent_autonomy_jobs(limit=200):
        if str(job.get("job_type", "")) in job_types:
            return job
    return {}


def _long_term_payload() -> dict:
    try:
        from product.long_term_store import load_long_term_scan
        payload = load_long_term_scan() or {}
        return {
            "available": bool(payload),
            "scanned_at": payload.get("scanned_at", ""),
            "fundamentals_source": payload.get("fundamentals_source", ""),
            "summary": dict(payload.get("summary", {}) or {}),
            "records": [dict(row) for row in (payload.get("records", []) or []) if isinstance(row, dict)],
            "job": _latest_autonomy_job({"long_term_scan", "long_term_refresh"}),
        }
    except Exception as exc:
        return {
            "available": False,
            "scanned_at": "",
            "fundamentals_source": "",
            "summary": {},
            "records": [],
            "job": {},
            "error": str(exc),
        }


def _paper_equity_curve() -> list[float]:
    raw = _json_file(logs_dir() / "intelligence" / "intel_book.json", {})
    curve: list[float] = []
    for value in raw.get("equity_curve", []) or []:
        parsed = _safe_float(value)
        if parsed is not None:
            curve.append(parsed)
    return curve[-240:]


def _paper_learning_payload() -> dict:
    """Daily paper-memory overlay for the bash terminal. Never raises."""
    try:
        from product.paper_learning import public_memory
        return public_memory()
    except Exception as exc:
        return {
            "available": False,
            "as_of": "",
            "closed_trades": 0,
            "cooldown": [],
            "prefer": [],
            "shadow_prefer": [],
            "self_feed": {},
            "summary": "Paper memory unavailable.",
            "live_locked": None,
            "live_lock_verified": False,
            "live_lock_status": "UNVERIFIED",
            "live_lock_source": "product.live_execution_interlock",
            "disclaimer": str(exc),
            "ladder": "",
        }


def _project_open_position(raw: dict) -> dict:
    """research.auto_research.paper_book.PaperPosition.as_dict() uses its own
    internal field names (qty/stop_price/target_price/strategy_id/bars_held)
    -- correct for the sizing/mark-to-market math that reads them elsewhere,
    but not what the Paper Portfolio frontend (frontend/src/types.ts's
    PaperPosition, rendered by components.tsx's PositionsTable) expects. Add
    the frontend's names alongside the originals (never remove the
    originals -- other consumers of this same payload may still read them)
    so the table stops silently showing QTY 0 / stop-target blank for data
    that was there all along, just under a different key. There is no live
    quote fetched in this read-only status path (deliberately -- this
    endpoint is polled every few seconds and must stay network-free), so
    current_price/pnl/pnl_pct for an OPEN position stay absent rather than
    fabricated; the frontend already renders that as '-' honestly.
    """
    row = dict(raw)
    row.setdefault("quantity", row.get("qty"))
    row.setdefault("stop", row.get("stop_price"))
    row.setdefault("target", row.get("target_price"))
    row.setdefault("strategy", row.get("strategy_id"))
    row.setdefault("days_held", row.get("bars_held"))
    return row


def _project_closed_trade(raw: dict) -> dict:
    """Mirrors _project_open_position for research.auto_research.paper_book
    .ClosedTrade.as_dict() -- current_price maps to the real exit_price (a
    closed trade has no "current" price, its exit price IS what the
    'EXIT / CURRENT' column header means), result_r maps to the real
    realized_R. pnl/exit_reason already match the frontend's field names.
    """
    row = dict(raw)
    row.setdefault("quantity", row.get("qty"))
    row.setdefault("stop", row.get("stop_price"))
    row.setdefault("current_price", row.get("exit_price"))
    row.setdefault("strategy", row.get("strategy_id"))
    row.setdefault("result_r", row.get("realized_R"))
    return row


def _paper_payload() -> dict:
    try:
        from product.paper_status import read_paper_status
        paper = read_paper_status()
        return {
            "available": True,
            "enabled": paper.enabled,
            "supervisor_running": paper.supervisor_running,
            "capital": paper.capital,
            "equity": paper.equity,
            "equity_curve": _paper_equity_curve(),
            "open_risk": paper.open_risk,
            "risk_per_trade_pct": paper.risk_per_trade_pct,
            "max_positions": paper.max_positions,
            "open_positions": [
                _project_open_position(p) for p in paper.open_positions if isinstance(p, dict)
            ],
            "closed_trades": [
                _project_closed_trade(t) for t in list(paper.closed_trades)[-100:] if isinstance(t, dict)
            ],
            "refusals": list(paper.refusals)[-50:],
            "last_cycle": dict(paper.last_cycle or {}),
            "last_error": paper.last_error,
            "learning": _paper_learning_payload(),
        }
    except Exception as exc:
        return {
            "available": False,
            "enabled": False,
            "supervisor_running": False,
            "capital": 0.0,
            "equity": 0.0,
            "equity_curve": [],
            "open_risk": 0.0,
            "risk_per_trade_pct": 0.01,
            "max_positions": 0,
            "open_positions": [],
            "closed_trades": [],
            "refusals": [],
            "last_cycle": {},
            "last_error": str(exc),
            "error": str(exc),
            "learning": _paper_learning_payload(),
        }


def _capability(value: Any) -> str:
    text = str(value or "blocked").strip().lower()
    return text if text in {"allowed", "limited", "blocked", "read_only"} else "blocked"


def _autonomy_payload() -> dict:
    try:
        from product.autonomy_status import read_autonomy_status
        from research.autonomy import default_root
        root = default_root()
        status = read_autonomy_status()
        raw = _json_file(root / "status.json", {})
        runtime = _json_file(root / "runtime.json", {})
        entry_capability = _capability(status.get("new_paper_entries"))
        exit_capability = _capability(status.get("existing_exits"))
        research_capability = _capability(status.get("research"))
        live_feed = dict(raw.get("live_feed", {}) or {})
        telegram = {}
        try:
            from product.telegram_delivery import delivery_status
            telegram = delivery_status()
        except Exception as exc:
            telegram = {"configured": False, "state": "unavailable", "detail": str(exc)}
        explanation = str(raw.get("explanation") or "") or str(status.get("explanation") or "")
        reason_code = str(raw.get("reason_code") or status.get("reason_code") or "")
        return {
            "available": True,
            "running": bool(status.get("running")),
            "process_running": bool(runtime.get("process_running", raw.get("process_running", False))),
            "state": str(status.get("state", "UNKNOWN")),
            "plain_state": str(status.get("plain_state", "")),
            "explanation": explanation,
            "reason_code": reason_code,
            "heartbeat_ist": str(runtime.get("heartbeat_ist") or status.get("heartbeat_ist", "")),
            "scheduler_owner_pid": runtime.get("scheduler_owner_pid", raw.get("scheduler_owner_pid")),
            "active_job": dict(runtime.get("active_job", {}) or {}),
            "current_activity": str(status.get("current_activity") or "UNKNOWN"),
            "activity_truth": dict(status.get("activity_truth", {}) or {}),
            "resource_governor": dict(status.get("resource_governor", {}) or {}),
            "operational_incidents": dict(status.get("operational_incidents", {}) or {}),
            "new_entry_capability": entry_capability,
            "existing_exit_capability": exit_capability,
            "research_capability": research_capability,
            "new_paper_entries": entry_capability == "allowed",
            "existing_exits": exit_capability != "blocked",
            "research_enabled": research_capability != "blocked",
            "capability_notes": list(status.get("capability_notes", []) or []),
            "active_failures": list(
                status.get("active_failures")
                or raw.get("active_failures")
                or []
            ),
            "recent_dialogue": list(status.get("recent_dialogue", []) or [])[-40:],
            "recent_transitions": list(status.get("recent_transitions", []) or [])[-30:],
            "jobs": dict(status.get("jobs", {}) or {}),
            "jobs_recent": _recent_autonomy_jobs(),
            "owner_state": dict(status.get("owner_state", {}) or {}),
            "live_feed": live_feed,
            "telegram": telegram,
            "last_cycle": dict(status.get("last_cycle", {}) or {}),
        }
    except Exception as exc:
        return {
            "available": False,
            "running": False,
            "process_running": False,
            "state": "UNKNOWN",
            "plain_state": "Autonomy status unavailable.",
            "explanation": str(exc),
            "reason_code": "",
            "heartbeat_ist": "",
            "scheduler_owner_pid": None,
            "active_job": {},
            "current_activity": "UNKNOWN",
            "activity_truth": {},
            "resource_governor": {},
            "operational_incidents": {"open_count": 0, "open": [], "recent": []},
            "new_entry_capability": "blocked",
            "existing_exit_capability": "blocked",
            "research_capability": "blocked",
            "new_paper_entries": False,
            "existing_exits": False,
            "research_enabled": False,
            "capability_notes": [],
            "active_failures": [],
            "recent_dialogue": [],
            "recent_transitions": [],
            "jobs": {},
            "jobs_recent": [],
            "owner_state": {},
            "live_feed": {},
            "telegram": {"configured": False, "state": "unavailable"},
            "last_cycle": {},
            "error": str(exc),
        }


def _snapshot_payload() -> dict:
    try:
        from research.intelligence.data.snapshot_store import SnapshotStore
        root = logs_dir() / "snapshots"
        store = SnapshotStore(root)
        snapshot_id = store.get_active_snapshot()
        if not snapshot_id:
            return {
                "ready": False,
                "snapshot_id": "",
                "latest_date": "",
                "source": "",
                "error": "No active verified snapshot",
            }
        manifest = _json_file(root / str(snapshot_id) / "manifest.json", {})
        return {
            "ready": True,
            "snapshot_id": str(snapshot_id),
            "latest_date": str(manifest.get("last_trading_date") or ""),
            "source": str(manifest.get("source") or ""),
        }
    except Exception as exc:
        return {"ready": False, "snapshot_id": "", "latest_date": "", "source": "", "error": str(exc)}


def _scan_progress_payload() -> dict[str, Any]:
    try:
        from product.scan_progress import read_progress
        return read_progress()
    except Exception:
        return {"active": False, "eta_label": "", "current": 0, "total": 0}


def _operations_payload() -> dict[str, Any]:
    """Only query lightweight operation metadata for polling dashboard clients.

    Detailed results remain available from /api/operations/{operation_id}.
    Never decode large completed result_json blobs on the 8-12s UI poll.
    """
    try:
        from operations.market_ops import LANES
        from operations.store import OperationStore
        store = OperationStore(OPS_DB)
        runtime = _ops_runtime_payload()
        recent = store.recent_summary(100)
        latest = {}
        for kind in LANES:
            item = store.latest_summary(kind)
            if item:
                latest[kind] = item
        return {
            "available": True,
            "running": bool(runtime.get("running")),
            "worker_pid": runtime.get("worker_pid"),
            "heartbeat": runtime.get("heartbeat", ""),
            "active_lanes": dict(runtime.get("active", {}) or {}),
            "counts": store.counts(),
            "active": store.active_summary(),
            "recent": recent,
            "latest": latest,
        }
    except Exception as exc:
        return {
            "available": False,
            "running": False,
            "worker_pid": None,
            "heartbeat": "",
            "active_lanes": {},
            "counts": {},
            "active": [],
            "recent": [],
            "latest": {},
            "error": str(exc),
        }


def _news_payload() -> dict[str, Any]:
    try:
        from news.curator_store import NewsCuratorStore
        store = NewsCuratorStore(logs_dir() / "news_curator.sqlite3")
        try:
            articles = [item.as_dict() for item in store.recent(hours=168, limit=120)]
            health = [item.as_dict() for item in store.source_health()]
            stats = store.stats(hours=24)
        finally:
            store.close()
        # News needs one operation's status, not an entire second operations
        # dashboard (which would repeatedly load historical result blobs).
        from operations.store import OperationStore
        latest_refresh = OperationStore(OPS_DB).latest_summary("NEWS_REFRESH") or {}
        return {
            "available": bool(articles or health),
            "stats": stats,
            "articles": articles,
            "source_health": health,
            "latest_refresh": latest_refresh,
        }
    except Exception as exc:
        return {
            "available": False,
            "stats": {"total": 0, "important": 0, "fno_linked": 0, "macro": 0, "sources": 0},
            "articles": [],
            "source_health": [],
            "latest_refresh": {},
            "error": str(exc),
        }


def _fo_forward_evidence_overlay(
    directional: dict[str, Any],
    outcomes: list[dict[str, Any]],
    *,
    min_n: int = 30,
) -> dict[str, Any]:
    """Attach read-only forward-paper evidence to persisted F&O candidates.

    This projection never changes setup/contract scores, candidate ordering, or
    execution authority. A win-probability/EV claim is surfaced only when the
    evidence module says the exact candidate context has enough fully-costed
    settled FORWARD_PAPER observations.
    """
    from product.fo_evidence import (
        FO_CONTEXT_SCHEMA_VERSION,
        fo_evidence_coverage,
        summarize_fo_outcomes,
    )

    payload = dict(directional or {})
    candidates: list[dict[str, Any]] = []
    for raw in payload.get("candidates") or []:
        if not isinstance(raw, dict):
            continue
        row = dict(raw)
        selected = row.get("selected_contract")
        selected = selected if isinstance(selected, dict) else {}
        context_key = str(selected.get("context_key") or "").strip()
        if not context_key:
            evidence = {
                "context_key": "",
                "context_schema_version": FO_CONTEXT_SCHEMA_VERSION,
                "valid_context": False,
                "evidence_lane": "FORWARD_PAPER",
                "status": "NO_CONTEXT_KEY",
                "observed_n": 0,
                "n": 0,
                "probability_claim_available": False,
                "win_probability_pct": None,
                "win_probability_wilson_lb_pct": None,
                "expectancy_pct": None,
                "conservative_ev_pct": None,
                "insufficient_evidence": True,
                "minimum_required_n": max(1, int(min_n)),
                "production_influence_allowed": False,
                "coverage": fo_evidence_coverage(
                    outcomes,
                    context_key="",
                    min_n=min_n,
                ),
            }
        else:
            evidence = summarize_fo_outcomes(
                outcomes,
                context_key=context_key,
                min_n=min_n,
            )
            evidence["coverage"] = fo_evidence_coverage(
                outcomes,
                context_key=context_key,
                min_n=min_n,
            )
            if not bool(evidence.get("valid_context")):
                status = "CONTEXT_VERSION_REQUIRED"
            elif evidence.get("probability_claim_available"):
                status = "EVIDENCE_READY"
            elif int(evidence.get("n") or 0) == 0:
                cost_excluded = int(evidence.get("excluded_unpriced_costs") or 0)
                path_excluded = int(evidence.get("excluded_path_observation") or 0)
                if cost_excluded > 0 and path_excluded > 0:
                    status = "EVIDENCE_INPUTS_REQUIRED"
                elif path_excluded > 0:
                    status = "PATH_OBSERVATION_REQUIRED"
                elif cost_excluded > 0:
                    status = "COST_MODEL_REQUIRED"
                elif int(evidence.get("observed_n") or 0) > 0:
                    status = "ACCUMULATING"
                else:
                    status = "NO_FORWARD_OUTCOMES"
            elif int(evidence.get("observed_n") or 0) > 0:
                status = "ACCUMULATING"
            else:
                status = "NO_FORWARD_OUTCOMES"
            evidence["status"] = status
        row["forward_evidence"] = evidence
        candidates.append(row)

    payload["candidates"] = candidates
    payload["candidate_evidence_policy"] = {
        "lane": "FORWARD_PAPER",
        "minimum_fully_costed_n": max(1, int(min_n)),
        "context_schema_version": FO_CONTEXT_SCHEMA_VERSION,
        "probability_requires_current_context_version": True,
        "probability_requires_exact_context": True,
        "probability_requires_complete_observed_path": True,
        "broader_context_counts_research_only": True,
        "paper_only": True,
        "live_execution_allowed": False,
    }
    return payload


def _fo_directional_payload() -> dict[str, Any]:
    path = logs_dir() / "product" / "fo_directional.json"
    payload = _json_file(path, {})
    if not payload:
        return {
            "available": False,
            "status": "NOT_RUN",
            "decision": None,
            "candidate_count": 0,
            "candidates": [],
            "paper_only": True,
            "live_execution_allowed": False,
        }
    payload["cache_mtime"] = path.stat().st_mtime if path.exists() else None
    payload["paper_only"] = True
    payload["live_execution_allowed"] = False
    try:
        from product.fo_paper_store import FoPaperStore

        with FoPaperStore(read_only=True) as store:
            # Bounded latest window keeps the dashboard read cheap while still
            # providing ample context evidence for the minimum-N promotion gate.
            outcomes = store.load_trades(limit=5000)
        payload = _fo_forward_evidence_overlay(payload, outcomes, min_n=30)
        payload["candidate_evidence_status"] = "AVAILABLE"
    except Exception as exc:
        # Evidence projection failure must never erase a valid persisted scan.
        # It also must not carry saved probability claims forward when the
        # durable observations cannot be read (including an absent ledger).
        payload["candidates"] = [
            {
                **row,
                "forward_evidence": {
                    "status": "UNAVAILABLE",
                    "evidence_lane": "FORWARD_PAPER",
                    "observed_n": None,
                    "n": None,
                    "minimum_required_n": 30,
                    "probability_claim_available": False,
                    "win_probability_pct": None,
                    "win_probability_wilson_lb_pct": None,
                    "expectancy_pct": None,
                    "conservative_ev_pct": None,
                    "insufficient_evidence": True,
                    "production_influence_allowed": False,
                    "coverage": {},
                },
            }
            for row in (payload.get("candidates") or []) if isinstance(row, dict)
        ]
        payload["candidate_evidence_status"] = "UNAVAILABLE"
        payload["candidate_evidence_error"] = str(exc)[:240]
    return payload


def _fo_paper_payload() -> dict[str, Any]:
    """Read durable NSE F&O paper state without creating market activity."""
    try:
        from product.fo_paper_store import FoPaperStore

        with FoPaperStore(read_only=True) as store:
            status = store.status()
            open_positions = store.load_positions()
            recent_closed = store.load_trades(limit=50)
        return {
            "available": True,
            "status": status,
            "open_positions": open_positions,
            "recent_closed_trades": recent_closed,
            "production_evidence_enabled": int(
                status.get("production_evidence_trades") or 0
            ) > 0,
            "paper_only": True,
            "live_execution_allowed": False,
        }
    except Exception as exc:
        return {
            "available": False,
            "status": {},
            "open_positions": [],
            "recent_closed_trades": [],
            "production_evidence_enabled": False,
            "paper_only": True,
            "live_execution_allowed": False,
            "error": str(exc)[:240],
        }


def _fno_learning_impact_payload(directional: dict[str, Any]) -> dict[str, Any]:
    try:
        from product.fno_learning_impact import build_fno_learning_impact
        return dict(build_fno_learning_impact(directional) or {})
    except Exception as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}"[:240],
            "historical": {"available": False, "can_affect_ranking": False},
            "forward": {"available": False, "can_affect_ranking": False},
            "ranking_impact": {"status": "UNAVAILABLE", "plain": "Learning-impact projection unavailable."},
            "live_locked": True,
        }


def _fno_payload() -> dict[str, Any]:
    path = logs_dir() / "product" / "fno_universe.json"
    persisted = _json_file(path, {})
    directional = _fo_directional_payload()
    paper = _fo_paper_payload()
    learning_impact = _fno_learning_impact_payload(directional)
    from product.fo_desk import build_fo_desk
    desk = build_fo_desk(directional, paper)
    if persisted:
        persisted["available"] = int(persisted.get("mapped_underlyings", 0) or 0) > 0
        persisted["cache_mtime"] = path.stat().st_mtime if path.exists() else None
        persisted["directional"] = directional
        persisted["paper"] = paper
        persisted["learning_impact"] = learning_impact
        persisted["desk"] = desk
        return persisted
    try:
        from data.fno_universe import current_fno_universe
        report = current_fno_universe()
        return {
            "available": report.mapped_underlyings > 0,
            "generated_at": None,
            "source": report.source,
            "total_instrument_rows": report.total_instrument_rows,
            "total_future_contracts": report.total_future_contracts,
            "index_future_contracts": report.index_future_contracts,
            "unique_stock_underlyings": report.unique_stock_underlyings,
            "mapped_underlyings": report.mapped_underlyings,
            "underlyings": [item.__dict__ for item in report.underlyings],
            "exclusions": [item.__dict__ for item in report.exclusions],
            "directional": directional,
            "paper": paper,
            "learning_impact": learning_impact,
            "desk": desk,
        }
    except Exception as exc:
        return {
            "available": False,
            "source": "unavailable",
            "mapped_underlyings": 0,
            "underlyings": [],
            "exclusions": [],
            "directional": directional,
            "paper": paper,
            "learning_impact": learning_impact,
            "desk": desk,
            "error": str(exc),
        }


def _data_payload(scan: dict, long_term: dict, operations: dict, fno: dict, news: dict) -> dict:
    try:
        from data.bhavcopy_runtime import status as bhavcopy_status
        # Do not unpickle store_cache.pkl on the Home request. A cold API
        # process can spend longer than the page timeout loading it.
        bhavcopy = bhavcopy_status(load_cache=False)
        if bhavcopy.get("cache_exists") and not bhavcopy.get("ready"):
            _schedule_warm("bhavcopy-cache", _warm_bhavcopy_cache)
    except Exception as exc:
        bhavcopy = {
            "ready": False,
            "symbols": 0,
            "sessions": 0,
            "latest_date": "",
            "csv_files": 0,
            "cache_exists": False,
            "error": str(exc),
        }
    snapshot = _snapshot_payload()
    blockers: list[str] = []
    try:
        from data.bhavcopy_runtime import official_history_freshness

        freshness = official_history_freshness(
            bhavcopy, load_cache=False, require_store=False,
        )
        for key in (
            "current",
            "expected_latest_completed_session",
            "available_session",
            "stale_sessions",
            "reason_code",
            "store_loaded",
        ):
            if key in freshness:
                bhavcopy[key] = freshness[key]
    except Exception:
        freshness = {}
    from product.data_readiness import project_official_data_readiness
    data_truth = project_official_data_readiness(
        freshness=freshness,
        data={"ready": bhavcopy.get("ready"), "bhavcopy": bhavcopy},
        bhav=bhavcopy,
        operations_running=bool(operations.get("running")),
    )
    if not bhavcopy.get("ready"):
        if data_truth.get("history_current"):
            blockers.append(
                "Official session files are current; this API process has not loaded the bhavcopy store yet. Scans will use the same official files once the pickle is in memory."
            )
        elif bhavcopy.get("cache_exists"):
            blockers.append("Official NSE bhavcopy cache is on disk and still loading into the desk API.")
        else:
            blockers.append("Official NSE bhavcopy history is not ready; direct scans will prepare it first.")
    elif int(bhavcopy.get("sessions", 0) or 0) < int(bhavcopy.get("minimum_sessions", 60) or 60):
        blockers.append("Official bhavcopy history is shallower than the minimum screen requirement.")
    elif freshness and not freshness.get("current", True):
        blockers.append(
            "Official NSE bhavcopy is behind the latest completed session "
            f"({freshness.get('available_session') or 'unknown'} < "
            f"{freshness.get('expected_latest_completed_session') or 'unknown'})."
        )
    if not snapshot.get("ready"):
        blockers.append("Verified snapshot is missing; PAPER autonomy is limited, but direct cash scans can still use official bhavcopy history.")
    if not operations.get("running"):
        blockers.append("Dedicated market-operations worker is not online.")
    if not fno.get("available"):
        blockers.append("Current F&O instrument universe is unavailable; refresh instruments after Zerodha login.")
    if not news.get("available"):
        blockers.append("Curated news store is empty; run a news refresh to inspect source health.")
    return {
        "ready": bool(data_truth["data_ready"]),
        "history_current": bool(data_truth["history_current"]),
        "store_loaded": bool(data_truth["store_loaded"]),
        "operations_running": bool(operations.get("running")),
        "lane_status": data_truth["lane_status"],
        "lane_status_code": data_truth["lane_status_code"],
        "snapshot": snapshot,
        "bhavcopy": bhavcopy,
        "scan_saved": bool(scan.get("available")),
        "scan_records": len(scan.get("records", []) or []),
        "long_term_saved": bool(long_term.get("available")),
        "long_term_records": len(long_term.get("records", []) or []),
        "blockers": list(dict.fromkeys(blockers)),
    }


def _conviction(scan: dict, market: dict) -> list[dict]:
    if not scan.get("available") or not market.get("available"):
        return []
    try:
        from product.conviction import build_conviction_shortlist
        from product.market_view import RetailMarketView
        view = RetailMarketView(
            health=str(market["health"]),
            summary=str(market["summary"]),
            trade_stance=str(market["trade_stance"]),
            breadth=str(market["breadth"]),
            leaders=tuple(market.get("leaders", [])),
            laggards=tuple(market.get("laggards", [])),
            nifty_change_1d=float(market.get("nifty_change_1d") or 0.0),
            nifty_change_5d=float(market.get("nifty_change_5d") or 0.0),
            vix=float(market.get("vix") or 0.0),
            nifty_price=float(market.get("nifty_price") or 0.0),
            technical_details=dict(market.get("technical_details", {}) or {}),
        )
        return build_conviction_shortlist(
            {"records": scan.get("records", []), "summary": scan.get("summary", {})},
            view,
        )
    except Exception:
        return []


@app.get("/api/health")
def health() -> dict:
    """Liveness plus the cheap STARTING/READY/DEGRADED/FAILED/RECOVERING probe.

    File, PID and port checks only. Autonomy SQLite and live scans stay off
    this path so the launcher can start the desk.
    """
    payload = {
        "ok": True,
        "service": "quantterm-terminal-api",
        "version": app.version,
        "lifecycle": None,
        "reason": "Terminal API is serving",
        "reasons": [],
        "components": [],
    }
    try:
        from product.public_access import access_projection
        payload["access"] = access_projection()
    except Exception:
        payload["access"] = {
            "public_read_only": None,
            "mutation_policy": "UNVERIFIED",
            "operator_token_configured": False,
            "live_money_unlocked": False,
        }
    try:
        from product.runtime_lifecycle import inspect_runtime

        runtime = inspect_runtime(api_serving=True)
        payload.update({
            "lifecycle": runtime.get("lifecycle"),
            "reason": runtime.get("reason") or payload["reason"],
            "reasons": runtime.get("reasons") or [],
            "components": runtime.get("components") or [],
            "history": runtime.get("history") or {},
            "resources": runtime.get("resources") or {},
            "checked_at": runtime.get("checked_at"),
        })
        # Copy inspect_runtime safety/readiness only. Never invent a positive value.
        for key in (
            "operational_ready",
            "evidence_ready",
            "live_locked",
            "live_lock_verified",
            "live_execution_authorized",
            "live_lock_status",
            "live_lock_reason",
            "live_lock_source",
        ):
            if key in runtime:
                payload[key] = runtime[key]
        payload["ok"] = payload["lifecycle"] != "FAILED"
    except Exception as exc:
        payload.update({
            "ok": True,
            "lifecycle": "DEGRADED",
            "reason": f"Runtime probe failed: {exc}"[:240],
            "reasons": [str(exc)[:240]],
            "live_locked": None,
            "live_lock_verified": False,
            "live_execution_authorized": None,
            "live_lock_status": "UNVERIFIED",
            "live_lock_reason": f"Runtime probe failed before live-execution safety could be verified: {exc}"[:240],
            "live_lock_source": "product.live_execution_interlock",
        })
    return payload


@app.get("/api/dashboard")
def dashboard() -> dict:
    """RecoWealth desk bootstrap. Last readable scan survives a subsystem failure."""
    started = time.monotonic()
    timings: dict[str, float] = {}

    def _timed(label: str, fn):
        step_started = time.monotonic()
        try:
            return fn()
        finally:
            timings[label] = round(time.monotonic() - step_started, 3)

    try:
        try:
            scan = _timed("scan", _scan_payload)
        except Exception as exc:
            scan = {
                "available": False,
                "scanned_at": "",
                "universe_size": 0,
                "summary": {},
                "records": [],
                "provenance": {},
                "error": str(exc),
            }
        try:
            market = _timed("market", _market_payload)
            long_term = _timed("long_term", _long_term_payload)
            paper = _timed("paper", _paper_payload)
            autonomy = _timed("autonomy", _autonomy_payload)
            operations = _timed("operations", _operations_payload)
            news = _timed("news", _news_payload)
            fno = _timed("fno", _fno_payload)
            data = _timed("data", lambda: _data_payload(scan, long_term, operations, fno, news))
            conviction = _timed("conviction", lambda: _conviction(scan, market))
            daily_wrap: list = []
            try:
                from product.desk_note import daily_wrap as build_daily_wrap
                daily_wrap = build_daily_wrap(
                    articles=list(news.get("articles") or []),
                    scan_payload=scan,
                )
            except Exception:
                daily_wrap = []
            return _json_safe({
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "market": market,
                "scan": _slim_ranked_records(scan),
                "long_term": _slim_ranked_records(long_term),
                "paper": paper,
                "autonomy": autonomy,
                "operations": operations,
                "news": news,
                "fno": fno,
                "data": data,
                "conviction": conviction,
                "scan_progress": _scan_progress_payload(),
                "daily_wrap": daily_wrap,
            })
        except Exception as exc:
            degraded = _empty_dashboard(f"Dashboard degraded: {exc}", scan)
            try:
                degraded["market"] = _market_payload()
            except Exception:
                pass
            try:
                degraded["long_term"] = _long_term_payload()
            except Exception:
                pass
            try:
                degraded["operations"] = _operations_payload()
            except Exception:
                pass
            return _json_safe(degraded)
    finally:
        total = time.monotonic() - started
        if total >= 2.0:
            # No symbols, positions, credentials or other sensitive payloads.
            # Only slow requests are logged; external response JSON is unchanged.
            lanes = " ".join(f"{name}={seconds:.3f}s" for name, seconds in timings.items())
            print(f"[API SLOW] GET /api/dashboard total={total:.3f}s {lanes}", flush=True)


@app.get("/api/operations")
def operations_status() -> dict:
    return _operations_payload()


@app.get("/api/operations/{operation_id}")
def operation_status(operation_id: str) -> dict:
    from operations.store import OperationStore
    item = OperationStore(OPS_DB).get(operation_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Operation not found")
    return item


@app.get("/api/news")
def news_status() -> dict:
    return _news_payload()


@app.get("/api/education")
def education_feed(min_impact: int = 40, limit: int = 40) -> dict:
    """Educational cards projected from curated news — never invents articles."""
    from product.education_feed import build_education_feed

    news = _news_payload()
    return build_education_feed(
        articles=list(news.get("articles") or []),
        min_impact=max(0, min(int(min_impact or 40), 100)),
        limit=max(1, min(int(limit or 40), 100)),
    )


@app.get("/api/fno")
def fno_status() -> dict:
    return _fno_payload()


@app.get("/api/fno-directional")
def fno_directional_status() -> dict:
    return _fo_directional_payload()


@app.get("/api/data-readiness")
def data_readiness() -> dict:
    scan = _scan_payload()
    long_term = _long_term_payload()
    operations = _operations_payload()
    news = _news_payload()
    fno = _fno_payload()
    return _data_payload(scan, long_term, operations, fno, news)


@app.get("/api/chart/{symbol}")
def chart(symbol: str, limit: int = 220) -> dict:
    clean_symbol = symbol.strip().upper()
    if not clean_symbol or len(clean_symbol) > 32:
        raise HTTPException(status_code=400, detail="Invalid symbol")
    try:
        from data.bhavcopy_runtime import get_ohlcv, status as bhavcopy_status
        frame = get_ohlcv(clean_symbol)
        readiness = bhavcopy_status(load_cache=False)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Price history unavailable: {exc}") from exc
    if frame is None or len(frame) == 0:
        return {"symbol": clean_symbol, "bars": [], "history": readiness}
    frame = frame.tail(max(20, min(int(limit), 500))).copy()
    bars = []
    for index, row in frame.iterrows():
        stamp = getattr(index, "date", lambda: index)()
        bars.append({
            "time": str(stamp),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume", 0.0) or 0.0),
        })
    return {"symbol": clean_symbol, "bars": bars, "history": readiness}


_OPERATION_CONTROLS = {
    "RUN_SCAN_NOW": "MARKET_SCAN",
    "RUN_LONG_TERM_SCAN_NOW": "MARKET_SCAN",
    "REFRESH_LONG_TERM_NOW": "LONG_TERM_REFRESH",
    "REFRESH_NEWS_NOW": "NEWS_REFRESH",
    "REFRESH_MARKET_REPORT_NOW": "MARKET_REPORT",
    "REFRESH_FNO_NOW": "FNO_REFRESH",
    "REFRESH_DATA_NOW": "DATA_PREPARE",
}
_AUTONOMY_CONTROLS = {
    "RUN_CYCLE_NOW",
    "PAUSE_NEW_PAPER_ENTRIES",
    "RESUME_NEW_PAPER_ENTRIES",
    "OBSERVE_ONLY_TODAY",
    "CLEAR_OBSERVE_ONLY",
    "RUN_HISTORICAL_REPLAY",
    "RUN_LEARNING_NOW",
}
_ALLOWED_CONTROLS = set(_OPERATION_CONTROLS) | _AUTONOMY_CONTROLS
_USER_OPERATION_PRIORITY = 100


@app.post("/api/controls/{control_name}")
def control(control_name: str) -> dict:
    name = control_name.strip().upper()
    if name not in _ALLOWED_CONTROLS:
        raise HTTPException(status_code=400, detail="Control is not allowed through the terminal API")
    if name in _OPERATION_CONTROLS:
        from operations.market_ops import LANES
        from operations.store import OperationStore
        store = OperationStore(OPS_DB)
        try:
            store.recover_dead_running()
        except Exception:
            pass
        kind = _OPERATION_CONTROLS[name]
        if name in {"RUN_SCAN_NOW", "RUN_LONG_TERM_SCAN_NOW"}:
            try:
                from data.bhavcopy_runtime import official_history_freshness
                from operations.market_ops import DATA_PREPARE

                if not official_history_freshness().get("current"):
                    store.enqueue(
                        DATA_PREPARE,
                        lane=LANES[DATA_PREPARE],
                        requested_by="terminal",
                        priority=_USER_OPERATION_PRIORITY,
                    )
            except Exception:
                pass
        operation, created = store.enqueue(
            kind,
            lane=LANES[kind],
            requested_by="terminal",
            priority=_USER_OPERATION_PRIORITY,
        )
        try:
            worker = _ensure_ops_worker(wait=False)
        except RuntimeError as exc:
            from operations.store import live_lock_owner_pid
            lock_pid = live_lock_owner_pid(OPS_ROOT / "worker.lock")
            if not lock_pid:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            worker = {"running": True, "worker_pid": lock_pid, "recovering": True}
        return {
            "accepted": True,
            "control": name,
            "operation_id": operation.get("operation_id"),
            "operation_status": operation.get("status"),
            "created": created,
            "priority": operation.get("priority"),
            "worker_recovering": bool((worker or {}).get("recovering")),
        }
    from research.autonomy.controls import request_control
    queued = request_control(name, reason="owner requested control from dedicated terminal frontend")
    return {
        "accepted": True,
        "control": name,
        "control_id": getattr(queued, "control_id", ""),
    }
