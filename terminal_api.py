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

ROOT = Path(__file__).resolve().parent
OPS_ROOT = ROOT / "logs" / "market_ops"
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
    return {
        **runtime,
        "running": running,
        "process_running": bool(runtime.get("process_running")),
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
        return {
            "available": True,
            "health": market.health,
            "summary": market.summary,
            "trade_stance": market.trade_stance,
            "breadth": market.breadth,
            "leaders": list(market.leaders),
            "laggards": list(market.laggards),
            "nifty_change_1d": _safe_float(market.nifty_change_1d),
            "nifty_change_5d": _safe_float(market.nifty_change_5d),
            "vix": _safe_float(market.vix),
            "nifty_price": _safe_float(getattr(market, "nifty_price", None)),
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
        from product.scan_store import load_scan
        payload = load_scan() or {}
        records = [dict(row) for row in (payload.get("records", []) or []) if isinstance(row, dict)]
        return {
            "available": bool(payload),
            "scanned_at": payload.get("scanned_at", ""),
            "universe_size": int(payload.get("universe_size", 0) or 0),
            "summary": dict(payload.get("summary", {}) or {}),
            "records": records,
        }
    except Exception as exc:
        return {
            "available": False,
            "scanned_at": "",
            "universe_size": 0,
            "summary": {},
            "records": [],
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
    raw = _json_file(ROOT / "logs" / "intelligence" / "intel_book.json", {})
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
            "live_locked": True,
            "disclaimer": str(exc),
            "ladder": "",
        }


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
            "open_positions": list(paper.open_positions),
            "closed_trades": list(paper.closed_trades)[-100:],
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
        return {
            "available": True,
            "running": bool(status.get("running")),
            "process_running": bool(runtime.get("process_running", raw.get("process_running", False))),
            "state": str(status.get("state", "UNKNOWN")),
            "plain_state": str(status.get("plain_state", "")),
            "explanation": str(status.get("explanation", "")),
            "heartbeat_ist": str(runtime.get("heartbeat_ist") or status.get("heartbeat_ist", "")),
            "scheduler_owner_pid": runtime.get("scheduler_owner_pid", raw.get("scheduler_owner_pid")),
            "active_job": dict(runtime.get("active_job", {}) or {}),
            "new_entry_capability": entry_capability,
            "existing_exit_capability": exit_capability,
            "research_capability": research_capability,
            "new_paper_entries": entry_capability == "allowed",
            "existing_exits": exit_capability != "blocked",
            "research_enabled": research_capability != "blocked",
            "capability_notes": list(status.get("capability_notes", []) or []),
            "active_failures": list(raw.get("active_failures", []) or []),
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
            "heartbeat_ist": "",
            "scheduler_owner_pid": None,
            "active_job": {},
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
        root = ROOT / "logs" / "snapshots"
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
    try:
        from operations.market_ops import LANES
        from operations.store import OperationStore
        store = OperationStore(OPS_DB)
        runtime = _ops_runtime_payload()
        recent = store.recent(100)
        latest = {}
        for kind in LANES:
            item = store.latest(kind)
            if item:
                latest[kind] = item
        return {
            "available": True,
            "running": bool(runtime.get("running")),
            "worker_pid": runtime.get("worker_pid"),
            "heartbeat": runtime.get("heartbeat", ""),
            "active_lanes": dict(runtime.get("active", {}) or {}),
            "counts": store.counts(),
            "active": store.active(),
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
        store = NewsCuratorStore(ROOT / "logs" / "news_curator.sqlite3")
        try:
            articles = [item.as_dict() for item in store.recent(hours=168, limit=120)]
            health = [item.as_dict() for item in store.source_health()]
            stats = store.stats(hours=24)
        finally:
            store.close()
        latest_refresh = _operations_payload().get("latest", {}).get("NEWS_REFRESH", {})
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


def _fno_payload() -> dict[str, Any]:
    path = ROOT / "logs" / "product" / "fno_universe.json"
    persisted = _json_file(path, {})
    if persisted:
        persisted["available"] = int(persisted.get("mapped_underlyings", 0) or 0) > 0
        persisted["cache_mtime"] = path.stat().st_mtime if path.exists() else None
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
        }
    except Exception as exc:
        return {
            "available": False,
            "source": "unavailable",
            "mapped_underlyings": 0,
            "underlyings": [],
            "exclusions": [],
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

        freshness = official_history_freshness(bhavcopy, load_cache=False)
        for key in (
            "current",
            "expected_latest_completed_session",
            "available_session",
            "stale_sessions",
            "reason_code",
        ):
            if key in freshness:
                bhavcopy[key] = freshness[key]
    except Exception:
        freshness = {}
    if not bhavcopy.get("ready"):
        if bhavcopy.get("cache_exists"):
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
        "ready": bool(bhavcopy.get("ready") and bhavcopy.get("current", True) and operations.get("running")),
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
        "lifecycle": "READY",
        "reason": "Terminal API is serving",
        "reasons": [],
        "components": [],
        "live_locked": True,
    }
    try:
        from product.runtime_lifecycle import inspect_runtime

        runtime = inspect_runtime(api_serving=True)
        payload.update({
            "lifecycle": runtime.get("lifecycle") or "READY",
            "reason": runtime.get("reason") or payload["reason"],
            "reasons": runtime.get("reasons") or [],
            "components": runtime.get("components") or [],
            "history": runtime.get("history") or {},
            "checked_at": runtime.get("checked_at"),
            "live_locked": True,
        })
        payload["ok"] = payload["lifecycle"] != "FAILED"
    except Exception as exc:
        payload.update({
            "ok": True,
            "lifecycle": "DEGRADED",
            "reason": f"Runtime probe failed: {exc}"[:240],
            "reasons": [str(exc)[:240]],
        })
    return payload


@app.get("/api/dashboard")
def dashboard() -> dict:
    """RecoWealth desk bootstrap. Last readable scan survives a subsystem failure."""
    try:
        scan = _scan_payload()
    except Exception as exc:
        scan = {
            "available": False,
            "scanned_at": "",
            "universe_size": 0,
            "summary": {},
            "records": [],
            "error": str(exc),
        }
    try:
        market = _market_payload()
        long_term = _long_term_payload()
        paper = _paper_payload()
        autonomy = _autonomy_payload()
        operations = _operations_payload()
        news = _news_payload()
        fno = _fno_payload()
        data = _data_payload(scan, long_term, operations, fno, news)
        conviction = _conviction(scan, market)
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
        _ensure_ops_worker(wait=False)
        return {
            "accepted": True,
            "control": name,
            "operation_id": operation.get("operation_id"),
            "operation_status": operation.get("status"),
            "created": created,
            "priority": operation.get("priority"),
        }
    from research.autonomy.controls import request_control
    queued = request_control(name, reason="owner requested control from dedicated terminal frontend")
    return {
        "accepted": True,
        "control": name,
        "control_id": getattr(queued, "control_id", ""),
    }
