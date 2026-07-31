"""Read-only API bridge for the dedicated QuantTerm terminal frontend.

The existing Python product, research, scanner, paper-book and autonomy stores remain authoritative.
This module projects that state to local HTTP and forwards a small whitelist of owner controls to the
single autonomy supervisor. It never scans, trades, or mutates broker state directly.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="QuantTerm Terminal API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


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
        }
    except Exception as exc:
        return {
            "available": False,
            "scanned_at": "",
            "fundamentals_source": "",
            "summary": {},
            "records": [],
            "error": str(exc),
        }


def _paper_equity_curve() -> list[float]:
    path = Path(__file__).resolve().parent / "logs" / "intelligence" / "intel_book.json"
    raw = _json_file(path, {})
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


def _recent_jobs(limit: int = 60) -> list[dict]:
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


def _autonomy_payload() -> dict:
    try:
        from product.autonomy_status import read_autonomy_status
        from research.autonomy import default_root
        status = read_autonomy_status()
        raw = _json_file(default_root() / "status.json", {})
        return {
            "available": True,
            "running": bool(status.get("running")),
            "process_running": bool(raw.get("process_running", False)),
            "state": str(status.get("state", "UNKNOWN")),
            "plain_state": str(status.get("plain_state", "")),
            "explanation": str(status.get("explanation", "")),
            "heartbeat_ist": str(status.get("heartbeat_ist", "")),
            "scheduler_owner_pid": raw.get("scheduler_owner_pid"),
            "new_paper_entries": bool(status.get("new_paper_entries")),
            "existing_exits": bool(status.get("existing_exits")),
            "research_enabled": bool(status.get("research")),
            "capability_notes": list(status.get("capability_notes", []) or []),
            "active_failures": list(raw.get("active_failures", []) or []),
            "recent_dialogue": list(status.get("recent_dialogue", []) or [])[-40:],
            "recent_transitions": list(status.get("recent_transitions", []) or [])[-30:],
            "jobs": dict(status.get("jobs", {}) or {}),
            "jobs_recent": _recent_jobs(),
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
            {"records": scan.get("records", []), "summary": scan.get("summary", {})}, view
        )
    except Exception:
        return []


@app.get("/api/health")
def health() -> dict:
    autonomy = _autonomy_payload()
    return {
        "ok": True,
        "service": "quantterm-terminal-api",
        "version": app.version,
        "autonomy_running": autonomy.get("running", False),
        "autonomy_state": autonomy.get("state", "UNKNOWN"),
    }


@app.get("/api/dashboard")
def dashboard() -> dict:
    market = _market_payload()
    scan = _scan_payload()
    long_term = _long_term_payload()
    paper = _paper_payload()
    autonomy = _autonomy_payload()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "market": market,
        "scan": scan,
        "long_term": long_term,
        "paper": paper,
        "autonomy": autonomy,
        "conviction": _conviction(scan, market),
    }


@app.get("/api/chart/{symbol}")
def chart(symbol: str, limit: int = 220) -> dict:
    clean_symbol = symbol.strip().upper()
    if not clean_symbol or len(clean_symbol) > 32:
        raise HTTPException(status_code=400, detail="Invalid symbol")
    try:
        from data.bhavcopy_store import get_ohlcv
        frame = get_ohlcv(clean_symbol)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Price history unavailable: {exc}") from exc
    if frame is None or len(frame) == 0:
        return {"symbol": clean_symbol, "bars": []}
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
    return {"symbol": clean_symbol, "bars": bars}


_ALLOWED_CONTROLS = {
    "RUN_SCAN_NOW",
    "RUN_LONG_TERM_SCAN_NOW",
    "REFRESH_LONG_TERM_NOW",
    "RUN_CYCLE_NOW",
    "REFRESH_DATA_NOW",
    "PAUSE_NEW_PAPER_ENTRIES",
    "RESUME_NEW_PAPER_ENTRIES",
}


@app.post("/api/controls/{control_name}")
def control(control_name: str) -> dict:
    name = control_name.strip().upper()
    if name not in _ALLOWED_CONTROLS:
        raise HTTPException(status_code=400, detail="Control is not allowed through the terminal API")
    from research.autonomy.controls import request_control
    queued = request_control(name, reason="owner requested control from dedicated terminal frontend")
    return {"accepted": True, "control": name, "control_id": getattr(queued, "control_id", "")}
