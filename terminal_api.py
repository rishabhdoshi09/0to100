"""Read-only API bridge for the dedicated QuantTerm terminal frontend.

The existing Python product, research, scanner, paper-book and autonomy stores remain authoritative.
This module only projects that state to HTTP and forwards a tiny whitelist of owner controls to the
single autonomy supervisor. It never scans, trades, or mutates broker state directly.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="QuantTerm Terminal API", version="0.1.0")
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
            "error": str(exc),
        }


def _scan_payload() -> dict:
    try:
        from product.scan_store import load_scan
        payload = load_scan() or {}
        return {
            "available": bool(payload),
            "scanned_at": payload.get("scanned_at", ""),
            "universe_size": int(payload.get("universe_size", 0) or 0),
            "summary": dict(payload.get("summary", {}) or {}),
            "records": list(payload.get("records", []) or []),
        }
    except Exception as exc:
        return {"available": False, "universe_size": 0, "summary": {}, "records": [], "error": str(exc)}


def _long_term_payload() -> dict:
    try:
        from product.long_term_store import load_long_term_scan
        payload = load_long_term_scan() or {}
        return {
            "available": bool(payload),
            "scanned_at": payload.get("scanned_at", ""),
            "summary": dict(payload.get("summary", {}) or {}),
            "records": list(payload.get("records", []) or []),
        }
    except Exception as exc:
        return {"available": False, "summary": {}, "records": [], "error": str(exc)}


def _paper_equity_curve() -> list[float]:
    path = Path(__file__).resolve().parent / "logs" / "intelligence" / "intel_book.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
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
            "enabled": paper.enabled,
            "supervisor_running": paper.supervisor_running,
            "capital": paper.capital,
            "equity": paper.equity,
            "equity_curve": _paper_equity_curve(),
            "open_risk": paper.open_risk,
            "risk_per_trade_pct": paper.risk_per_trade_pct,
            "max_positions": paper.max_positions,
            "open_positions": list(paper.open_positions),
            "closed_trades": list(paper.closed_trades)[-50:],
        }
    except Exception as exc:
        return {
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
            "error": str(exc),
        }


def _autonomy_payload() -> dict:
    try:
        from product.autonomy_status import read_autonomy_status
        status = read_autonomy_status()
        return {
            "running": bool(status.get("running")),
            "state": str(status.get("state", "UNKNOWN")),
            "plain_state": str(status.get("plain_state", "")),
            "explanation": str(status.get("explanation", "")),
            "heartbeat_ist": str(status.get("heartbeat_ist", "")),
            "new_paper_entries": bool(status.get("new_paper_entries")),
            "recent_dialogue": list(status.get("recent_dialogue", []) or [])[-20:],
            "jobs": dict(status.get("jobs", {}) or {}),
        }
    except Exception as exc:
        return {
            "running": False,
            "state": "UNKNOWN",
            "plain_state": "Autonomy status unavailable.",
            "explanation": str(exc),
            "heartbeat_ist": "",
            "new_paper_entries": False,
            "recent_dialogue": [],
            "jobs": {},
        }


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "service": "quantterm-terminal-api"}


@app.get("/api/dashboard")
def dashboard() -> dict:
    market = _market_payload()
    scan = _scan_payload()
    long_term = _long_term_payload()
    paper = _paper_payload()
    autonomy = _autonomy_payload()
    conviction: list[dict] = []

    if scan.get("available") and market.get("available"):
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
                technical_details={},
            )
            conviction = build_conviction_shortlist(
                {"records": scan.get("records", []), "summary": scan.get("summary", {})},
                view,
            )
        except Exception:
            conviction = []

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "market": market,
        "scan": scan,
        "long_term": long_term,
        "paper": paper,
        "autonomy": autonomy,
        "conviction": conviction,
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


_ALLOWED_CONTROLS = {"RUN_SCAN_NOW", "RUN_LONG_TERM_SCAN_NOW", "RUN_CYCLE_NOW"}


@app.post("/api/controls/{control_name}")
def control(control_name: str) -> dict:
    name = control_name.strip().upper()
    if name not in _ALLOWED_CONTROLS:
        raise HTTPException(status_code=400, detail="Control is not allowed through the terminal API")
    from research.autonomy.controls import request_control
    queued = request_control(name, reason="owner requested control from dedicated terminal frontend")
    return {"accepted": True, "control": name, "control_id": getattr(queued, "control_id", "")}
