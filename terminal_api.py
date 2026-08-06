"""Local API bridge for the dedicated QuantTerm terminal.

Authoritative market/research stores remain in Python. User-requested market
operations are dispatched to a dedicated worker plane; PAPER autonomy remains a
separate execution/learning lane and is never allowed to block scans.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time
from typing import Any

from fastapi import Body, FastAPI, HTTPException
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


class _QuietHealthAccess(logging.Filter):
    """Stack watch loops hit /api/health often — keep the console readable."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "/api/health" not in msg and ' "GET /health' not in msg


logging.getLogger("uvicorn.access").addFilter(_QuietHealthAccess())

_ops_process: subprocess.Popen | None = None
_ops_ensure_last_attempt: float = 0.0
_bhav_status_cache: dict[str, Any] = {"ts": 0.0, "payload": None}
_dashboard_cache: dict[str, Any] = {"ts": 0.0, "payload": None}
_DASHBOARD_CACHE_TTL_S = 8.0
_DASHBOARD_STALE_S = 90.0
_dashboard_rebuild_lock = threading.Lock()
_dashboard_rebuild_started = False
_institutional_refresh_started = False


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


def _ops_pid_alive(pid: Any) -> bool:
    try:
        pid_i = int(pid)
    except Exception:
        return False
    if pid_i <= 0:
        return False
    try:
        os.kill(pid_i, 0)
        return True
    except OSError:
        return False


def _ops_runtime_payload() -> dict[str, Any]:
    runtime = _json_file(OPS_RUNTIME, {})
    # Heartbeat thread writes every ~2s, but heavy scans on old Macs can starve
    # the GIL long enough to miss a 10s window — that used to flip the worker
    # "OFFLINE" and trigger a terminate/respawn storm mid-scan.
    hb_fresh = _fresh_epoch(runtime.get("heartbeat_epoch"), max_age_s=60.0)
    pid = runtime.get("worker_pid")
    pid_alive = _ops_pid_alive(pid)
    process_flag = bool(runtime.get("process_running"))
    running = bool(process_flag and pid_alive and hb_fresh)
    # Busy-but-alive: PID still up, heartbeat only slightly late — stay ONLINE.
    if not running and process_flag and pid_alive and _fresh_epoch(runtime.get("heartbeat_epoch"), max_age_s=120.0):
        running = True
    return {
        **runtime,
        "running": running,
        "process_running": process_flag,
        "pid_alive": pid_alive,
        "heartbeat_fresh": hb_fresh,
    }


def _reclaim_stale_ops_lock(
    *,
    max_heartbeat_age_s: float = 90.0,
    allow_terminate: bool = True,
) -> str:
    """Clear dead/stale market-ops lock holders so a new worker can start."""
    try:
        from operations.market_ops import LOCK_PATH, RUNTIME_PATH, SingleWorkerLock
    except Exception as exc:
        return f"lock_import_failed:{exc}"

    lock = SingleWorkerLock(LOCK_PATH)
    if lock.reclaim_if_dead():
        return "reclaimed_dead_lock"

    runtime = _json_file(RUNTIME_PATH, {})
    heartbeat = runtime.get("heartbeat_epoch")
    stale = not _fresh_epoch(heartbeat, max_age_s=max_heartbeat_age_s)
    holder = lock.holder_pid() or runtime.get("worker_pid")
    if holder and stale:
        if not allow_terminate:
            return f"stale_holder_kept:{holder}"
        # Orphaned worker after API-only stop: alive PID, dead heartbeat.
        try:
            lock.path.write_text(str(int(holder)), encoding="utf-8")
        except Exception:
            pass
        if lock.terminate_holder(reason="stale_heartbeat"):
            try:
                RUNTIME_PATH.unlink(missing_ok=True)
            except Exception:
                pass
            return f"terminated_stale_holder:{holder}"
    return "lock_held_or_clear"


def _ensure_ops_worker(
    *,
    wait_s: float = 8.0,
    force: bool = False,
    allow_terminate: bool = True,
) -> dict[str, Any]:
    """Start the dedicated market-operations worker when it is not healthy."""
    global _ops_process, _ops_ensure_last_attempt
    runtime = _ops_runtime_payload()
    if runtime.get("running"):
        runtime["ensure_attempted"] = False
        runtime["ensure_ok"] = True
        return runtime

    now = time.time()
    if not force and (now - float(_ops_ensure_last_attempt or 0.0)) < 3.0:
        runtime["ensure_attempted"] = True
        runtime["ensure_ok"] = False
        runtime["ensure_error"] = runtime.get("ensure_error") or (
            "Market-ops worker offline — retry already in flight"
        )
        return runtime
    _ops_ensure_last_attempt = now

    reclaim_note = _reclaim_stale_ops_lock(allow_terminate=allow_terminate)

    def _wait_for_online(deadline: float) -> dict[str, Any]:
        while time.time() < deadline:
            time.sleep(0.15)
            current = _ops_runtime_payload()
            if current.get("running"):
                return current
            if _ops_process is not None and _ops_process.poll() is not None:
                break
        return _ops_runtime_payload()

    wait = max(0.0, float(wait_s))

    if _ops_process is not None and _ops_process.poll() is None:
        # Child still starting — wait for heartbeat only when requested.
        runtime = _wait_for_online(time.time() + wait) if wait > 0 else _ops_runtime_payload()
        runtime["ensure_attempted"] = True
        runtime["ensure_ok"] = bool(runtime.get("running"))
        if not runtime["ensure_ok"]:
            runtime["ensure_error"] = (
                "Market-ops child process is running but has not published a fresh heartbeat yet"
            )
        runtime["reclaim"] = reclaim_note
        return runtime

    _ops_process = subprocess.Popen(
        [sys.executable, "-u", "-m", "operations.market_ops"],
        cwd=str(ROOT),
        env=os.environ.copy(),
    )
    runtime = _wait_for_online(time.time() + wait) if wait > 0 else _ops_runtime_payload()

    # Spawn exited immediately (often lock conflict). Force-reclaim stale holder and retry.
    if not runtime.get("running") and _ops_process.poll() is not None:
        reclaim_note = _reclaim_stale_ops_lock(
            max_heartbeat_age_s=5.0,
            allow_terminate=allow_terminate,
        )
        _ops_process = subprocess.Popen(
            [sys.executable, "-u", "-m", "operations.market_ops"],
            cwd=str(ROOT),
            env=os.environ.copy(),
        )
        runtime = _wait_for_online(time.time() + wait) if wait > 0 else _ops_runtime_payload()

    runtime["ensure_attempted"] = True
    runtime["ensure_ok"] = bool(runtime.get("running"))
    runtime["spawn_pid"] = getattr(_ops_process, "pid", None)
    runtime["reclaim"] = reclaim_note
    if not runtime["ensure_ok"]:
        exit_code = _ops_process.poll() if _ops_process is not None else None
        if exit_code is not None:
            runtime["ensure_error"] = (
                f"Market-ops worker exited before heartbeat (exit={exit_code}, {reclaim_note}). "
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
    # Do not block API bind on worker heartbeat — old Macs need /api/health fast.
    # Never terminate an existing ops PID from startup; only spawn if missing.
    _ensure_ops_worker(wait_s=0.0, allow_terminate=False)
    _schedule_regime_refresh()
    _schedule_institutional_refresh()
    _schedule_active_buy_alerts()


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


_regime_refresh_started = False


def _schedule_regime_refresh() -> None:
    """Warm/refresh Kite-first regime off the dashboard request path."""
    global _regime_refresh_started
    if _regime_refresh_started:
        return
    _regime_refresh_started = True

    def _worker() -> None:
        while True:
            try:
                from core.regime_engine import compute_regime

                compute_regime(allow_network=True)
            except Exception:
                pass
            # Keep the Terminal API free; refresh in the background only.
            time.sleep(120.0 if str(os.getenv("QT_LOW_POWER", "")).strip() in {"1", "true", "TRUE", "yes"} else 60.0)

    try:
        import threading

        threading.Thread(target=_worker, name="regime-refresh", daemon=True).start()
    except Exception:
        _regime_refresh_started = False


def _schedule_institutional_refresh() -> None:
    """Warm FII/DII + option-chain caches off the dashboard request path.

    The previous dashboard path called NSE (cookie prime + FII + bulk deals +
    NIFTY chain) on every poll — that routinely exceeded the UI's 60s timeout
    while a market scan was already saturating an old Mac.
    """
    global _institutional_refresh_started
    if _institutional_refresh_started:
        return
    _institutional_refresh_started = True

    def _worker() -> None:
        while True:
            try:
                from data.fii_dii_store import workspace_payload

                workspace_payload(days=30, include_nifty_options=True, allow_network=True)
            except Exception:
                pass
            low = str(os.getenv("QT_LOW_POWER", "")).strip() in {"1", "true", "TRUE", "yes"}
            time.sleep(300.0 if low else 180.0)

    try:
        import threading

        threading.Thread(target=_worker, name="institutional-refresh", daemon=True).start()
    except Exception:
        _institutional_refresh_started = False


_active_buy_alerts_started = False


def _schedule_active_buy_alerts() -> None:
    """Push Active Buys tech/fund warnings to Telegram without needing market scan.

    Low-power mode disables auto MARKET_SCAN, which used to be the only path that
    called ``push_buy_book_alerts``. This dedicated loop keeps retail alerts alive.
    """
    global _active_buy_alerts_started
    if _active_buy_alerts_started:
        return
    _active_buy_alerts_started = True

    def _worker() -> None:
        # First pass after a short delay so API bind is not competing with Kite.
        time.sleep(45.0)
        while True:
            try:
                from risk.buy_book_watcher import push_buy_book_alerts

                push_buy_book_alerts()
            except Exception:
                pass
            low = str(os.getenv("QT_LOW_POWER", "")).strip() in {"1", "true", "TRUE", "yes"}
            time.sleep(900.0 if low else 420.0)

    try:
        import threading

        threading.Thread(target=_worker, name="active-buy-alerts", daemon=True).start()
    except Exception:
        _active_buy_alerts_started = False


def _market_payload(*, allow_network: bool = False) -> dict:
    try:
        from product.market_view import current_market_view
        market = current_market_view(allow_network=allow_network)
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
            "technical_details": dict(getattr(market, "technical_details", {}) or {}),
        }
    except Exception as exc:
        return {
            "available": False,
            "health": "Unavailable",
            "summary": "Market regime unavailable — needs Kite login / index history.",
            "trade_stance": "Do not infer a market stance from missing data.",
            "breadth": "—",
            "leaders": [],
            "laggards": [],
            "nifty_change_1d": None,
            "nifty_change_5d": None,
            "vix": None,
            "technical_details": {"primary_source": "kite", "yahoo_fallback": False},
            "error": str(exc),
        }


def _scan_payload(*, record_limit: int = 150) -> dict:
    try:
        from product.scan_store import load_scan, watchlist_rows
        payload = load_scan() or {}
        all_records = [dict(row) for row in (payload.get("records", []) or []) if isinstance(row, dict)]
        # Dashboard must stay small — shipping 2k+ rows freezes older Macs.
        limit = max(20, min(int(record_limit), 400))
        if len(all_records) > limit:
            records = [dict(row) for row in watchlist_rows(payload, limit=limit)]
        else:
            records = all_records
        return {
            "available": bool(payload),
            "scanned_at": payload.get("scanned_at", ""),
            "universe_size": int(payload.get("universe_size", 0) or 0),
            "summary": dict(payload.get("summary", {}) or {}),
            "records": records,
            "records_truncated": len(all_records) > len(records),
            "records_total": len(all_records),
        }
    except Exception as exc:
        return {
            "available": False,
            "scanned_at": "",
            "universe_size": 0,
            "summary": {},
            "records": [],
            "records_truncated": False,
            "records_total": 0,
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


def _long_term_payload(*, record_limit: int = 80) -> dict:
    try:
        from product.long_term_store import load_long_term_scan
        payload = load_long_term_scan() or {}
        records = [dict(row) for row in (payload.get("records", []) or []) if isinstance(row, dict)]
        limit = max(10, min(int(record_limit), 200))
        return {
            "available": bool(payload),
            "scanned_at": payload.get("scanned_at", ""),
            "fundamentals_source": payload.get("fundamentals_source", ""),
            "summary": dict(payload.get("summary", {}) or {}),
            "records": records[:limit],
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
            "jobs_recent": _recent_autonomy_jobs(limit=20),
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


def _operations_payload(*, wait_s: float = 0.0, recent_limit: int = 40) -> dict[str, Any]:
    try:
        from operations.market_ops import LANES
        from operations.store import OperationStore
        store = OperationStore(OPS_DB)
        runtime = _ops_runtime_payload()
        # Self-heal whenever the worker is offline — not only when a job is
        # already queued. Otherwise Scan now queues PENDING and never leases.
        # Default wait_s=0 keeps /api/dashboard non-blocking on old Macs.
        if not runtime.get("running"):
            # Dashboard/ops polls: spawn only — never terminate a busy worker.
            runtime = _ensure_ops_worker(
                wait_s=max(0.0, float(wait_s)),
                allow_terminate=float(wait_s) > 0,
            )
        snap = store.dashboard_snapshot(kinds=LANES.keys(), recent_limit=recent_limit)
        return {
            "available": True,
            "running": bool(runtime.get("running")),
            "worker_pid": runtime.get("worker_pid"),
            "heartbeat": runtime.get("heartbeat", ""),
            "active_lanes": dict(runtime.get("active", {}) or {}),
            "ensure_ok": runtime.get("ensure_ok", bool(runtime.get("running"))),
            "ensure_error": runtime.get("ensure_error", ""),
            "reclaim": runtime.get("reclaim", ""),
            "counts": dict(snap.get("counts") or {}),
            "active": list(snap.get("active") or []),
            "recent": list(snap.get("recent") or []),
            "latest": dict(snap.get("latest") or {}),
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


def _news_payload(*, latest_refresh: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        from news.curator_store import NewsCuratorStore
        store = NewsCuratorStore(ROOT / "logs" / "news_curator.sqlite3")
        try:
            articles = [item.as_dict() for item in store.recent(hours=168, limit=120)]
            health = [item.as_dict() for item in store.source_health()]
            stats = store.stats(hours=24)
        finally:
            store.close()
        # Caller should pass operations.latest — never re-open the ops DB here.
        refresh = dict(latest_refresh or {})
        return {
            "available": bool(articles or health),
            "stats": stats,
            "articles": articles,
            "source_health": health,
            "latest_refresh": refresh,
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


def _institutional_payload(*, allow_network: bool = False) -> dict[str, Any]:
    """Dashboard default is cache-only; background thread warms NSE separately."""
    try:
        from data.fii_dii_store import workspace_payload

        return workspace_payload(
            days=30,
            include_nifty_options=True,
            allow_network=bool(allow_network),
        )
    except Exception as exc:
        return {"available": False, "error": str(exc), "network_used": bool(allow_network)}


def _fno_payload() -> dict[str, Any]:
    """Persisted F&O universe only — never rebuild instruments on dashboard path."""
    path = ROOT / "logs" / "product" / "fno_universe.json"
    persisted = _json_file(path, {})
    if persisted:
        persisted["available"] = int(persisted.get("mapped_underlyings", 0) or 0) > 0
        persisted["cache_mtime"] = path.stat().st_mtime if path.exists() else None
        return persisted
    return {
        "available": False,
        "source": "unavailable",
        "mapped_underlyings": 0,
        "underlyings": [],
        "exclusions": [],
        "error": "No persisted fno_universe.json — run REFRESH_FNO_NOW / market-ops bootstrap",
    }


def _bhavcopy_status_fast() -> dict[str, Any]:
    """Dashboard hot-path status — never blocks on a full pickle reload.

    Loading ``store_cache.pkl`` into the API process can take tens of seconds and
    made the React shell show "backend unavailable". Scans still load history
    inside the market-ops worker via ``_ensure_history``.
    """
    global _bhav_status_cache
    now = time.time()
    cached = _bhav_status_cache.get("payload")
    if cached is not None and (now - float(_bhav_status_cache.get("ts") or 0.0)) < 30.0:
        return dict(cached)
    try:
        from data.bhavcopy_runtime import status as bhavcopy_status

        # Disk metadata only — never load_cache=True on the dashboard path.
        bhavcopy = dict(bhavcopy_status(load_cache=False))
        disk_ready = bool(
            bhavcopy.get("ready")
            or bhavcopy.get("cache_exists")
            or int(bhavcopy.get("csv_files", 0) or 0) >= int(bhavcopy.get("minimum_sessions", 60) or 60)
        )
        bhavcopy["disk_ready"] = disk_ready
        if not bhavcopy.get("ready") and disk_ready:
            bhavcopy["message"] = (
                "History is on disk; market-ops loads it for scans. "
                "API skips full pickle reload so the dashboard stays responsive."
            )
        _bhav_status_cache = {"ts": now, "payload": dict(bhavcopy)}
        return dict(bhavcopy)
    except Exception as exc:
        return {
            "ready": False,
            "disk_ready": False,
            "symbols": 0,
            "sessions": 0,
            "latest_date": "",
            "csv_files": 0,
            "cache_exists": False,
            "error": str(exc),
        }


def _data_payload(scan: dict, long_term: dict, operations: dict, fno: dict, news: dict) -> dict:
    bhavcopy = _bhavcopy_status_fast()
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
    if not bhavcopy.get("ready") and not bhavcopy.get("cache_exists"):
        blockers.append("Official NSE bhavcopy history is not ready; direct scans will prepare it first.")
    elif not bhavcopy.get("ready") and bhavcopy.get("cache_exists"):
        blockers.append("Bhavcopy cache exists but is not loaded in the API process yet; scans can still prepare it.")
    elif int(bhavcopy.get("sessions", 0) or 0) < int(bhavcopy.get("minimum_sessions", 60) or 60):
        if int(bhavcopy.get("csv_files", 0) or 0) < 60:
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
    history_ok = bool(
        bhavcopy.get("ready")
        or bhavcopy.get("cache_exists")
        or int(bhavcopy.get("csv_files", 0) or 0) >= 60
    )
    return {
        "ready": bool(history_ok and operations.get("running")),
        "snapshot": snapshot,
        "bhavcopy": bhavcopy,
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





def _dashboard_cache_ttl() -> float:
    low = str(os.getenv("QT_LOW_POWER", "")).strip() in {"1", "true", "TRUE", "yes"}
    return 15.0 if low else _DASHBOARD_CACHE_TTL_S


def _build_dashboard_payload() -> dict:
    try:
        # Hot path: never terminate a busy market-ops worker just because its
        # heartbeat lagged under GIL pressure — that killed scans and jammed :8765.
        _ensure_ops_worker(wait_s=0.0, allow_terminate=False)
    except Exception:
        pass
    _schedule_regime_refresh()
    _schedule_institutional_refresh()
    _schedule_active_buy_alerts()
    market = _market_payload(allow_network=False)
    scan = _scan_payload(record_limit=80)
    long_term = _long_term_payload(record_limit=40)
    paper = _paper_payload()
    autonomy = _autonomy_payload()
    operations = _operations_payload(wait_s=0.0, recent_limit=20)
    news = _news_payload(
        latest_refresh=dict((operations.get("latest") or {}).get("NEWS_REFRESH") or {}),
    )
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
        # Cache-only — NSE FII/options warmed by background thread.
        "institutional": _institutional_payload(allow_network=False),
        "data": data,
        "conviction": _conviction(scan, market),
        "cache_hit": False,
    }


def _schedule_dashboard_rebuild() -> None:
    """Refresh dashboard cache off the request path when serving stale."""
    global _dashboard_rebuild_started
    if not _dashboard_rebuild_lock.acquire(blocking=False):
        return
    if _dashboard_rebuild_started:
        _dashboard_rebuild_lock.release()
        return
    _dashboard_rebuild_started = True
    _dashboard_rebuild_lock.release()

    def _worker() -> None:
        global _dashboard_rebuild_started, _dashboard_cache
        try:
            payload = _build_dashboard_payload()
            _dashboard_cache = {"ts": time.time(), "payload": dict(payload)}
        except Exception:
            pass
        finally:
            _dashboard_rebuild_started = False

    try:
        threading.Thread(target=_worker, name="dashboard-rebuild", daemon=True).start()
    except Exception:
        _dashboard_rebuild_started = False


@app.get("/api/dashboard")
def dashboard() -> dict:
    # Never block the dashboard on Yahoo/NSE fetches or worker heartbeat waits.
    # Those freezes timed out the UI (25–60s) on older MacBooks — especially while
    # a market scan was already saturating CPU/disk.
    global _dashboard_cache
    now = time.time()
    cached = _dashboard_cache.get("payload")
    cache_age = now - float(_dashboard_cache.get("ts") or 0.0)
    ttl = _dashboard_cache_ttl()
    if cached is not None and cache_age < ttl:
        out = dict(cached)
        out["generated_at"] = datetime.now(timezone.utc).isoformat()
        out["cache_hit"] = True
        out["cache_age_s"] = round(cache_age, 2)
        return out
    if cached is not None and cache_age < _DASHBOARD_STALE_S:
        # Stale-while-revalidate: answer immediately, rebuild in background.
        _schedule_dashboard_rebuild()
        out = dict(cached)
        out["generated_at"] = datetime.now(timezone.utc).isoformat()
        out["cache_hit"] = True
        out["cache_stale"] = True
        out["cache_age_s"] = round(cache_age, 2)
        return out

    payload = _build_dashboard_payload()
    _dashboard_cache = {"ts": now, "payload": dict(payload)}
    return payload


@app.get("/api/sniper-board")
def sniper_board_status() -> dict:
    """Confirmed sniper breakouts + optional focused evaluation ranking."""
    try:
        from product.sniper_board import board_api_payload

        return board_api_payload()
    except Exception as exc:
        return {
            "available": False,
            "hits": [],
            "hit_count": 0,
            "symbols": [],
            "evaluation_records": [],
            "evaluation_summary": {},
            "places_orders": False,
            "live_locked": True,
            "error": str(exc),
        }


@app.get("/api/quotes/heartbeat")
def quotes_heartbeat(symbols: str = "", limit: int = 40) -> dict:
    """Live LTP heartbeat for visible symbols — Kite WS preferred, REST fallback.

    Charts / dossiers stay on EOD. Off-session returns honest session_open=false.
    """
    try:
        from product.quote_heartbeat import build_quote_heartbeat

        parts = [p.strip() for p in str(symbols or "").split(",") if p.strip()]
        return build_quote_heartbeat(parts, limit=max(1, min(int(limit or 40), 80)))
    except Exception as exc:
        return {
            "available": False,
            "session_open": False,
            "streaming": False,
            "quotes": {},
            "rows": [],
            "missing": [],
            "places_orders": False,
            "live_locked": True,
            "honesty": "Live quote heartbeat unavailable — no invented ticks.",
            "error": str(exc),
        }


@app.get("/api/wrap-of-the-day")
def wrap_of_the_day_get() -> dict:
    """User-authored Wrap of the Day — never invents bullets."""
    try:
        from product.wrap_of_the_day import load_wrap

        return load_wrap()
    except Exception as exc:
        return {
            "available": False,
            "bullets": [],
            "message": str(exc),
            "places_orders": False,
            "honesty": "Wrap of the Day unavailable.",
        }


@app.post("/api/wrap-of-the-day")
def wrap_of_the_day_save(body: dict[str, Any] = Body(default_factory=dict)) -> dict:
    """Save today's Wrap of the Day from pasted text or bullet list."""
    payload = body or {}
    try:
        from product.wrap_of_the_day import notify_wrap_telegram, save_wrap

        wrap = save_wrap(
            payload.get("bullets") or [],
            text=str(payload.get("text") or payload.get("raw_text") or ""),
            date=str(payload.get("date") or "") or None,
            source=str(payload.get("source") or "paste"),
        )
        if bool(payload.get("notify", False)) and wrap.get("available"):
            wrap["telegram"] = notify_wrap_telegram(wrap)
        return wrap
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/wrap-of-the-day/notify")
def wrap_of_the_day_notify() -> dict:
    """Send the current Wrap of the Day to Telegram."""
    try:
        from product.wrap_of_the_day import load_wrap, notify_wrap_telegram

        wrap = load_wrap()
        telegram = notify_wrap_telegram(wrap)
        return {
            "accepted": True,
            "telegram": telegram,
            "available": bool(wrap.get("available")),
            "count": len(wrap.get("bullets") or []),
            "date": wrap.get("date") or "",
            "places_orders": False,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/street-pulse")
def street_pulse_status(force: bool = False) -> dict:
    """Daily Street Pulse research digest — composes stores; never invents fills."""
    try:
        from reports.street_pulse import pulse_api_payload

        return pulse_api_payload(force=bool(force))
    except Exception as exc:
        return {
            "available": False,
            "report_type": "DAILY_STREET_PULSE",
            "takeaways": [],
            "gaps": [str(exc)],
            "places_orders": False,
            "live_locked": True,
            "signal_desk": False,
            "honesty": "Street pulse failed to assemble — no invented market narrative.",
            "error": str(exc),
        }


@app.post("/api/street-pulse/telegram")
def street_pulse_telegram(force: bool = True) -> dict:
    """Send Daily Pulse to Telegram (research digest only — never places orders)."""
    try:
        from reports.street_pulse import send_pulse_telegram

        result = send_pulse_telegram(force_build=bool(force))
        result["places_orders"] = False
        result["live_locked"] = True
        return result
    except Exception as exc:
        return {
            "sent": False,
            "configured": False,
            "places_orders": False,
            "live_locked": True,
            "error": str(exc),
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


@app.get("/api/us/dashboard")
def us_dashboard() -> dict:
    """US retail dashboard — listings, Yahoo EOD, scan, paper autopilot."""
    from product import us_retail

    return us_retail.dashboard()


@app.get("/api/us/readiness")
def us_readiness() -> dict:
    from product import us_retail

    return us_retail.readiness()


@app.get("/api/us/overview")
def us_overview() -> dict:
    from product import us_retail

    return us_retail.overview()


@app.get("/api/us/scan")
def us_scan() -> dict:
    from product import us_retail

    return us_retail.scan_payload()


@app.get("/api/us/paper")
def us_paper() -> dict:
    from product import us_retail

    return us_retail.paper_status()


@app.get("/api/us/stock/{symbol}")
def us_stock(symbol: str) -> dict:
    from product import us_retail

    return us_retail.stock_workspace(symbol)


@app.get("/api/us/chart/{symbol}")
def us_chart(symbol: str, limit: int = 220) -> dict:
    """US OHLCV chart — disk cache first, Yahoo on miss. Never invents bars."""
    from data import us_history_store as hist

    clean = symbol.strip().upper()
    if not clean or len(clean) > 16:
        raise HTTPException(status_code=400, detail="Invalid US symbol")
    frame = hist.get_ohlcv(clean, allow_network=True)
    readiness = hist.status()
    if frame is None or len(frame) == 0:
        return {
            "symbol": clean,
            "market": "US",
            "bars": [],
            "history": readiness,
            "source": "yfinance",
            "message": "No US history available for this symbol",
        }
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
    return {
        "symbol": clean,
        "market": "US",
        "bars": bars,
        "history": readiness,
        "source": "yfinance",
    }


@app.get("/api/us/education")
def us_education() -> dict:
    from product import us_retail

    cards = us_retail.education_concepts()
    return {
        "schema_version": 1,
        "market": "US",
        "available": True,
        "places_orders": False,
        "honesty": "Fixed US teach-ins — never invented articles.",
        "cards": cards,
    }


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
    "RUN_LONG_TERM_SCAN_NOW": "LONG_TERM_SCAN",
    "REFRESH_LONG_TERM_NOW": "LONG_TERM_REFRESH",
    "REFRESH_NEWS_NOW": "NEWS_REFRESH",
    "REFRESH_FNO_NOW": "FNO_REFRESH",
    "REFRESH_DATA_NOW": "DATA_PREPARE",
    "RUN_FULL_UNIVERSE_BACKTEST_NOW": "FULL_UNIVERSE_BACKTEST",
    "RUN_US_DATA_PREPARE_NOW": "US_DATA_PREPARE",
    "RUN_US_SCAN_NOW": "US_MARKET_SCAN",
    "RUN_SNIPER_BOARD_EVAL_NOW": "SNIPER_BOARD_EVAL",
}
_CANCEL_CONTROLS = {
    "CANCEL_SCAN_NOW": ("MARKET_SCAN",),
    "CANCEL_LONG_TERM_SCAN_NOW": ("LONG_TERM_SCAN", "LONG_TERM_REFRESH"),
    "CANCEL_SNIPER_BOARD_EVAL_NOW": ("SNIPER_BOARD_EVAL",),
    "CANCEL_US_SCAN_NOW": ("US_MARKET_SCAN",),
}
_AUTONOMY_CONTROLS = {
    "RUN_CYCLE_NOW",
    "PAUSE_NEW_PAPER_ENTRIES",
    "RESUME_NEW_PAPER_ENTRIES",
}
_ALLOWED_CONTROLS = set(_OPERATION_CONTROLS) | set(_CANCEL_CONTROLS) | _AUTONOMY_CONTROLS


@app.post("/api/controls/{control_name}")
def control(control_name: str) -> dict:
    name = control_name.strip().upper()
    if name not in _ALLOWED_CONTROLS:
        raise HTTPException(status_code=400, detail="Control is not allowed through the terminal API")
    if name in _CANCEL_CONTROLS:
        from operations.store import OperationStore

        kinds = _CANCEL_CONTROLS[name]
        report = OperationStore(OPS_DB).request_cancel_kinds(kinds)
        return {
            "accepted": True,
            "control": name,
            "cancelled": True,
            "kinds": list(report.get("kinds") or kinds),
            "cancelled_pending": int(report.get("cancelled_pending") or 0),
            "cancel_requested_running": int(report.get("cancel_requested_running") or 0),
            "operation_ids": list(report.get("operation_ids") or []),
            "transparency": (
                "Stop requested — queued scans cancelled immediately; "
                "a running scan stops after the current batch."
            ),
        }
    if name in _OPERATION_CONTROLS:
        from operations.market_ops import LANES
        from operations.store import OperationStore
        # Keep control clicks snappy on old Macs — do not sit 10s waiting for heartbeat.
        runtime = _ensure_ops_worker(wait_s=2.0, force=True)
        if not runtime.get("running"):
            # One hard reclaim+retry before accepting a PENDING forever queue.
            _reclaim_stale_ops_lock(max_heartbeat_age_s=5.0)
            runtime = _ensure_ops_worker(wait_s=2.0, force=True)
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
