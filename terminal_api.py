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
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent
OPS_ROOT = ROOT / "logs" / "market_ops"
OPS_RUNTIME = OPS_ROOT / "runtime.json"
OPS_DB = OPS_ROOT / "jobs.db"

app = FastAPI(title="QuantTerm Terminal API", version="0.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
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


def _ensure_ops_worker(*, wait_s: float = 8.0) -> dict[str, Any]:
    """Start the dedicated market-operations worker when it is not healthy."""
    global _ops_process
    runtime = _ops_runtime_payload()
    if runtime.get("running"):
        runtime["ensure_attempted"] = False
        runtime["ensure_ok"] = True
        return runtime

    # If a prior worker died without releasing diagnostics, clear a dead lock file
    # so a fresh spawn can take ownership.
    try:
        from operations.market_ops import LOCK_PATH, SingleWorkerLock

        SingleWorkerLock(LOCK_PATH).reclaim_if_dead()
    except Exception:
        pass

    def _wait_for_online(deadline: float) -> dict[str, Any]:
        while time.time() < deadline:
            time.sleep(0.15)
            current = _ops_runtime_payload()
            if current.get("running"):
                return current
            if _ops_process is not None and _ops_process.poll() is not None:
                break
        return _ops_runtime_payload()

    if _ops_process is not None and _ops_process.poll() is None:
        # Child still starting — wait for heartbeat.
        runtime = _wait_for_online(time.time() + max(1.0, float(wait_s)))
        runtime["ensure_attempted"] = True
        runtime["ensure_ok"] = bool(runtime.get("running"))
        if not runtime["ensure_ok"]:
            runtime["ensure_error"] = (
                "Market-ops child process is running but has not published a fresh heartbeat yet"
            )
        return runtime

    _ops_process = subprocess.Popen(
        [sys.executable, "-u", "-m", "operations.market_ops"],
        cwd=str(ROOT),
        env=os.environ.copy(),
    )
    runtime = _wait_for_online(time.time() + max(1.0, float(wait_s)))

    # Spawn exited immediately (often lock conflict). Reclaim dead owner and retry once.
    if not runtime.get("running") and _ops_process.poll() is not None:
        try:
            from operations.market_ops import LOCK_PATH, SingleWorkerLock

            SingleWorkerLock(LOCK_PATH).reclaim_if_dead()
        except Exception:
            pass
        _ops_process = subprocess.Popen(
            [sys.executable, "-u", "-m", "operations.market_ops"],
            cwd=str(ROOT),
            env=os.environ.copy(),
        )
        runtime = _wait_for_online(time.time() + max(1.0, float(wait_s)))

    runtime["ensure_attempted"] = True
    runtime["ensure_ok"] = bool(runtime.get("running"))
    runtime["spawn_pid"] = getattr(_ops_process, "pid", None)
    if not runtime["ensure_ok"]:
        exit_code = _ops_process.poll() if _ops_process is not None else None
        if exit_code is not None:
            runtime["ensure_error"] = (
                f"Market-ops worker exited before heartbeat (exit={exit_code}). "
                "Another worker may own the lock, or startup crashed — "
                "run: bash scripts/stop_quantterm.sh && bash scripts/run_quantterm_complete.sh"
            )
        else:
            runtime["ensure_error"] = (
                "Market-ops worker was spawned but no heartbeat yet — "
                "wait a few seconds or restart the stack"
            )
    return runtime


def _queue_message_for_control(kind: str, runtime: dict[str, Any]) -> str:
    lane = ""
    try:
        from operations.market_ops import LANES

        lane = str(LANES.get(kind) or "")
    except Exception:
        lane = ""
    active = dict((runtime.get("active") or {}).get(lane) or {})
    if not runtime.get("running"):
        detail = str(runtime.get("ensure_error") or "").strip()
        base = (
            "Queued, but market-ops worker is OFFLINE — "
            "restart with bash scripts/stop_quantterm.sh && bash scripts/run_quantterm_complete.sh"
        )
        return f"{base}. {detail}" if detail else (
            base + " (scans cannot start without the worker)"
        )
    if active:
        return (
            f"Queued behind active {active.get('kind') or 'job'} on the {lane or 'same'} lane — "
            "this scan starts when that job finishes"
        )
    return (
        f"Queued — market-ops worker ONLINE (pid {runtime.get('worker_pid') or '—'}); "
        "should lease this job within a few seconds"
    )


@app.on_event("startup")
def _startup() -> None:
    _ensure_ops_worker()


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
        from product.market_view import current_market_view
        market = current_market_view()
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
            "as_of": str((getattr(market, "technical_details", {}) or {}).get("as_of") or ""),
            "source": str((getattr(market, "technical_details", {}) or {}).get("source") or ""),
            "quote_source": str((getattr(market, "technical_details", {}) or {}).get("quote_source") or ""),
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
            "live_feed": dict(raw.get("live_feed", {}) or {}),
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


def _operations_payload() -> dict[str, Any]:
    try:
        from operations.market_ops import LANES
        from operations.store import OperationStore
        store = OperationStore(OPS_DB)
        runtime = _ops_runtime_payload()
        # Self-heal: UI polls this while a scan is PENDING. If the worker died,
        # try to bring it back so MARKET_SCAN is not stuck forever.
        if not runtime.get("running") and store.active():
            runtime = _ensure_ops_worker(wait_s=2.5)
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
            "ensure_ok": runtime.get("ensure_ok", bool(runtime.get("running"))),
            "ensure_error": runtime.get("ensure_error", ""),
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
            "ensure_ok": False,
            "ensure_error": str(exc),
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


def _institutional_payload() -> dict[str, Any]:
    try:
        from data.fii_dii_store import workspace_payload

        return workspace_payload(days=30)
    except Exception as exc:
        return {"available": False, "error": str(exc)}


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
        from data.bhavcopy_runtime import ensure_current_session
        bhavcopy = ensure_current_session(allow_network=True)
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
    try:
        from options.eod_store import store_status as options_eod_status

        options_eod = options_eod_status()
    except Exception as exc:
        options_eod = {
            "available": False,
            "path": "",
            "symbols": 0,
            "snapshots": 0,
            "latest_as_of": "",
            "error": str(exc),
        }
    blockers: list[str] = []
    history_current = bool(bhavcopy.get("ready")) and not bool(bhavcopy.get("is_stale"))
    if not bhavcopy.get("ready"):
        blockers.append("Official NSE bhavcopy history is not ready; direct scans will prepare it first.")
    elif bhavcopy.get("is_stale"):
        blockers.append(
            f"Official bhavcopy last bar is {bhavcopy.get('latest_date') or 'missing'}; "
            f"required session is {bhavcopy.get('required_session') or 'latest completed'}."
        )
    elif int(bhavcopy.get("sessions", 0) or 0) < int(bhavcopy.get("minimum_sessions", 60) or 60):
        blockers.append("Official bhavcopy history is shallower than the minimum screen requirement.")
    if not snapshot.get("ready"):
        blockers.append("Verified snapshot is missing; PAPER autonomy is limited, but direct cash scans can still use official bhavcopy history.")
    if not options_eod.get("available"):
        blockers.append("Options EOD OI/IV history is empty; run options-eod capture or wait for the EOD autonomy job.")
    if not operations.get("running"):
        blockers.append("Dedicated market-operations worker is not online.")
    if not fno.get("available"):
        blockers.append("Current F&O instrument universe is unavailable; refresh instruments after Zerodha login.")
    if not news.get("available"):
        blockers.append("Curated news store is empty; run a news refresh to inspect source health.")
    try:
        from data.live_quotes import kite_quote_health
        kite = kite_quote_health()
    except Exception as exc:
        kite = {"ok": False, "status": "error", "note": str(exc)[:160]}
    return {
        "ready": bool(history_current and operations.get("running")),
        "snapshot": snapshot,
        "bhavcopy": bhavcopy,
        "kite": kite,
        "options_eod": options_eod,
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
    """Pure liveness probe for Vite/stack monitors — must stay cheap and lock-free.

    Rich autonomy/ops status belongs on /api/dashboard (and /api/health/detail).
    Bundling it here made curl health checks time out while data_refresh or
    dashboard work held the API busy, and the stack launcher then killed Vite.
    """
    return {
        "ok": True,
        "service": "quantterm-terminal-api",
        "version": app.version,
    }


@app.get("/api/health/detail")
def health_detail() -> dict:
    """Optional richer status for debugging — not used by stack liveness checks."""
    autonomy = _autonomy_payload()
    operations = _operations_payload()
    return {
        "ok": True,
        "service": "quantterm-terminal-api",
        "version": app.version,
        "autonomy_running": autonomy.get("running", False),
        "autonomy_state": autonomy.get("state", "UNKNOWN"),
        "market_operations_running": operations.get("running", False),
    }


@app.get("/api/dashboard")
def dashboard() -> dict:
    market = _market_payload()
    scan = _scan_payload()
    long_term = _long_term_payload()
    paper = _paper_payload()
    autonomy = _autonomy_payload()
    operations = _operations_payload()
    news = _news_payload()
    fno = _fno_payload()
    data = _data_payload(scan, long_term, operations, fno, news)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "market": market,
        "scan": scan,
        "long_term": long_term,
        "paper": paper,
        "autonomy": autonomy,
        "operations": operations,
        "news": news,
        "fno": fno,
        "institutional": _institutional_payload(),
        "data": data,
        "conviction": _conviction(scan, market),
    }


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
    live_meta = {
        "live": False,
        "source": "",
        "eod_as_of": str(readiness.get("latest_date") or ""),
        "price_tag": "EOD",
        "sessions_behind": None,
    }
    if frame is None or len(frame) == 0:
        return {
            "symbol": clean_symbol,
            "bars": [],
            "history": readiness,
            "freshness": live_meta,
        }
    try:
        from research.intelligence.data import nse_calendar as CAL
        latest = str(readiness.get("latest_date") or "")
        if latest:
            verdict = CAL.snapshot_freshness(latest, allowance_sessions=0)
            live_meta["sessions_behind"] = verdict.get("sessions_behind")
            live_meta["history_fresh"] = bool(verdict.get("fresh"))
            live_meta["required_session"] = verdict.get("required")
    except Exception:
        live_meta["history_fresh"] = None
    try:
        from data.nse_live import overlay_live_on_frame
        frame, overlay = overlay_live_on_frame(frame, clean_symbol)
        live_meta.update({k: overlay.get(k, live_meta.get(k)) for k in overlay})
    except Exception:
        pass
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
    last_close = float(bars[-1]["close"]) if bars else None
    return {
        "symbol": clean_symbol,
        "bars": bars,
        "history": readiness,
        "freshness": live_meta,
        "last_close": last_close,
        "price_tag": live_meta.get("price_tag") or "EOD",
    }


_OPERATION_CONTROLS = {
    "RUN_SCAN_NOW": "MARKET_SCAN",
    "RUN_LONG_TERM_SCAN_NOW": "LONG_TERM_SCAN",
    "REFRESH_LONG_TERM_NOW": "LONG_TERM_REFRESH",
    "REFRESH_NEWS_NOW": "NEWS_REFRESH",
    "REFRESH_FNO_NOW": "FNO_REFRESH",
    "REFRESH_DATA_NOW": "DATA_PREPARE",
    "RUN_FULL_UNIVERSE_BACKTEST_NOW": "FULL_UNIVERSE_BACKTEST",
}
_AUTONOMY_CONTROLS = {
    "RUN_CYCLE_NOW",
    "PAUSE_NEW_PAPER_ENTRIES",
    "RESUME_NEW_PAPER_ENTRIES",
}
_ALLOWED_CONTROLS = set(_OPERATION_CONTROLS) | _AUTONOMY_CONTROLS


@app.post("/api/controls/{control_name}")
def control(control_name: str) -> dict:
    name = control_name.strip().upper()
    if name not in _ALLOWED_CONTROLS:
        raise HTTPException(status_code=400, detail="Control is not allowed through the terminal API")
    if name in _OPERATION_CONTROLS:
        from operations.market_ops import LANES
        from operations.store import OperationStore
        runtime = _ensure_ops_worker(wait_s=8.0)
        kind = _OPERATION_CONTROLS[name]
        queue_message = _queue_message_for_control(kind, runtime)
        operation, created = OperationStore(OPS_DB).enqueue(
            kind,
            lane=LANES[kind],
            requested_by="terminal",
            message=queue_message,
        )
        return {
            "accepted": True,
            "control": name,
            "operation_id": operation.get("operation_id"),
            "operation_status": operation.get("status"),
            "operation_message": operation.get("message") or queue_message,
            "created": created,
            "worker": {
                "running": bool(runtime.get("running")),
                "worker_pid": runtime.get("worker_pid"),
                "heartbeat": runtime.get("heartbeat"),
                "active_lanes": dict(runtime.get("active") or {}),
                "ensure_ok": runtime.get("ensure_ok", bool(runtime.get("running"))),
                "ensure_error": runtime.get("ensure_error", ""),
            },
            "transparency": queue_message,
            "blocker": (
                None
                if runtime.get("running")
                else (runtime.get("ensure_error") or "Market-ops worker is OFFLINE")
            ),
        }
    from research.autonomy.controls import request_control
    queued = request_control(name, reason="owner requested control from dedicated terminal frontend")
    return {
        "accepted": True,
        "control": name,
        "control_id": getattr(queued, "control_id", ""),
    }
