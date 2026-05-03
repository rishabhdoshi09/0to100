"""FastAPI cockpit – read-only views + manual `/api/trade`.

Routes
~~~~~~
* ``GET  /api/health``            — service health
* ``GET  /api/portfolio``         — snapshot
* ``GET  /api/positions``         — open positions
* ``GET  /api/signals/latest``    — last N signal rows
* ``GET  /api/cycle/status``      — last decision cycle
* ``GET  /api/cycle/last``        — alias of /cycle/status
* ``POST /api/cycle/run``         — force one cycle
* ``GET  /api/screener/latest``   — last screener result
* ``POST /api/screener/run``      — force one screen
* ``GET  /api/disagreements``     — last ensemble disagreements
* ``GET  /api/universe``          — cached instrument list
* ``POST /api/universe/refresh``  — force Kite refresh
* ``POST /api/trade``             — manual BUY/SELL
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sq_ai.backend.executor import Order
from sq_ai.backend.report_scheduler import ReportGenerator, DEFAULT_REPORTS_DIR
from sq_ai.backend.scheduler import TradingScheduler
from sq_ai.backend.screener_engine import run_screener
from sq_ai.backend.stock_research import (
    full_profile, header as stock_header, news_section, peers_compare,
    technicals as stock_technicals,
)
from sq_ai.backend.financials import get_financials, get_quarterly, get_ratios
from sq_ai.backend.analyst_estimates import get_estimates
from sq_ai.backend.shareholding import get_shareholding
from sq_ai.backend.corporate_actions import get_actions
from sq_ai.backend.earnings_analyzer import analyse_call, list_calls
from sq_ai.backend.watchlist import WatchlistService


class ManualTrade(BaseModel):
    symbol: str
    action: str
    qty: int
    price: float
    stop: float = 0.0
    target: float = 0.0


class ScreenerRequest(BaseModel):
    symbols: list[str] | None = None
    filters: dict = {}
    include_fundamentals: bool = False
    max_results: int = 50


class PresetRequest(BaseModel):
    name: str
    filters: dict


class WatchlistAddRequest(BaseModel):
    symbol: str
    note: str = ""


class EarningsAnalyseRequest(BaseModel):
    symbol: str
    quarter: str
    transcript_url: str | None = None
    call_date: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    sched = TradingScheduler()
    if os.environ.get("SQ_AUTOSTART_SCHEDULER", "true").lower() == "true":
        sched.start()
    app.state.sched = sched
    try:
        yield
    finally:
        sched.shutdown()


app = FastAPI(title="sq_ai cockpit", version="0.2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ─── health ────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "service": "sq_ai", "version": app.version}


# ─── portfolio / positions / trades ────────────────────────────────────────
@app.get("/api/portfolio")
async def portfolio() -> dict[str, Any]:
    sched: TradingScheduler = app.state.sched
    snap = sched.last_cycle.get("snapshot")
    return snap or sched._snapshot({})


@app.get("/api/positions")
async def positions() -> list[dict]:
    return app.state.sched.tracker.open_positions()


@app.get("/api/trades")
async def trades(limit: int = 100) -> list[dict]:
    return app.state.sched.tracker.closed_trades(limit=limit)


# ─── signals & cycle ───────────────────────────────────────────────────────
@app.get("/api/signals/latest")
async def signals_latest(limit: int = 20) -> list[dict]:
    return app.state.sched.tracker.latest_signals(limit=limit)


@app.get("/api/cycle/status")
async def cycle_status() -> dict[str, Any]:
    return app.state.sched.last_cycle or {"note": "no cycle yet"}


@app.get("/api/cycle/last")
async def cycle_last() -> dict[str, Any]:
    return app.state.sched.last_cycle or {"note": "no cycle yet"}


@app.post("/api/cycle/run")
async def cycle_run() -> dict[str, Any]:
    return app.state.sched.run_cycle()


# ─── screener ──────────────────────────────────────────────────────────────
@app.get("/api/screener")
async def screener_results() -> list[dict]:
    return app.state.sched.tracker.latest_screener()


@app.get("/api/screener/latest")
async def screener_latest() -> dict[str, Any]:
    sched: TradingScheduler = app.state.sched
    out = sched.last_screener or {}
    out["persisted"] = sched.tracker.latest_screener()
    return out


@app.post("/api/screener/run")
async def screener_run() -> dict[str, Any]:
    return app.state.sched.run_screener()


# ─── universe ──────────────────────────────────────────────────────────────
@app.get("/api/universe")
async def universe_list() -> list[dict]:
    return app.state.sched.tracker.get_cached_instruments()


@app.post("/api/universe/refresh")
async def universe_refresh() -> dict[str, Any]:
    return app.state.sched._refresh_universe()


# ─── disagreements ─────────────────────────────────────────────────────────
@app.get("/api/disagreements")
async def disagreements(limit: int = 50) -> list[dict]:
    return app.state.sched.tracker.latest_disagreements(limit=limit)


# ─── manual trade ──────────────────────────────────────────────────────────
@app.post("/api/trade")
async def trade(t: ManualTrade) -> dict[str, Any]:
    sched: TradingScheduler = app.state.sched
    if t.action.upper() == "BUY":
        return sched.executor.buy(
            Order(t.symbol, "BUY", t.qty, t.price, t.stop, t.target),
            reasoning="manual", regime="manual",
        )
    if t.action.upper() == "SELL":
        for pos in sched.tracker.open_positions():
            if pos["symbol"] == t.symbol:
                return sched.executor.exit_position(pos["id"], t.price)
        raise HTTPException(404, f"no open position for {t.symbol}")
    raise HTTPException(400, f"unknown action {t.action}")


# ───────────────────────────────────────────────────────── stock research
@app.get("/api/stock/profile/{symbol}")
async def stock_profile(symbol: str) -> dict[str, Any]:
    return full_profile(symbol)


@app.get("/api/stock/header/{symbol}")
async def stock_header_route(symbol: str) -> dict[str, Any]:
    return stock_header(symbol)


@app.get("/api/stock/technicals/{symbol}")
async def stock_technicals_route(symbol: str) -> dict[str, Any]:
    return stock_technicals(symbol)


@app.get("/api/stock/financials/{symbol}")
async def stock_financials(symbol: str) -> dict[str, Any]:
    return {
        "ratios": get_ratios(symbol),
        "annual": get_financials(symbol),
        "quarterly": get_quarterly(symbol),
    }


@app.get("/api/stock/earnings/{symbol}")
async def stock_earnings(symbol: str) -> list[dict]:
    sched: TradingScheduler = app.state.sched
    return list_calls(symbol, tracker=sched.tracker)


@app.post("/api/stock/earnings/analyse")
async def stock_earnings_analyse(req: EarningsAnalyseRequest) -> dict[str, Any]:
    sched: TradingScheduler = app.state.sched
    return analyse_call(req.symbol, req.quarter,
                        transcript_url=req.transcript_url,
                        call_date=req.call_date,
                        tracker=sched.tracker)


@app.get("/api/stock/estimates/{symbol}")
async def stock_estimates(symbol: str) -> dict[str, Any]:
    return get_estimates(symbol)


@app.get("/api/stock/shareholding/{symbol}")
async def stock_shareholding(symbol: str) -> dict[str, Any]:
    return get_shareholding(symbol)


@app.get("/api/stock/actions/{symbol}")
async def stock_actions(symbol: str) -> dict[str, Any]:
    return get_actions(symbol)


@app.get("/api/stock/news/{symbol}")
async def stock_news(symbol: str, top_n: int = 10) -> list[dict]:
    return news_section(symbol, top_n=top_n)


@app.get("/api/stock/peers/{symbol}")
async def stock_peers(symbol: str) -> list[dict]:
    return peers_compare(symbol)


# ───────────────────────────────────────────────────────── dynamic screener
@app.post("/api/screener/run")
async def dynamic_screener(req: ScreenerRequest) -> list[dict]:
    sched: TradingScheduler = app.state.sched
    if req.symbols:
        syms = req.symbols
    else:
        from sq_ai.backend.universe import get_active_universe
        syms = get_active_universe(
            max_symbols=200, tracker=sched.tracker,
            fallback_yaml=sched.cfg.get("universe") or [],
        )
    return run_screener(
        syms, filters=req.filters,
        include_fundamentals=req.include_fundamentals,
        max_results=req.max_results,
    )


@app.get("/api/screener/presets")
async def screener_presets() -> list[dict]:
    return app.state.sched.tracker.preset_list()


@app.post("/api/screener/presets")
async def screener_preset_save(req: PresetRequest) -> dict[str, Any]:
    app.state.sched.tracker.preset_save(req.name, req.filters)
    return {"status": "ok", "name": req.name}


@app.delete("/api/screener/presets/{name}")
async def screener_preset_delete(name: str) -> dict[str, Any]:
    n = app.state.sched.tracker.preset_delete(name)
    return {"status": "ok", "deleted": n}


# ───────────────────────────────────────────────────────── watchlist
@app.get("/api/watchlist")
async def watchlist() -> list[dict]:
    return WatchlistService(app.state.sched.tracker).list()


@app.post("/api/watchlist")
async def watchlist_add(req: WatchlistAddRequest) -> dict[str, Any]:
    return WatchlistService(app.state.sched.tracker).add(req.symbol, req.note)


@app.delete("/api/watchlist/{symbol}")
async def watchlist_remove(symbol: str) -> dict[str, Any]:
    return WatchlistService(app.state.sched.tracker).remove(symbol)


# ───────────────────────────────────────────────────────── equity curve
@app.get("/api/equity")
async def equity_curve() -> list[dict]:
    """Daily equity + cash series for charting the portfolio curve."""
    return app.state.sched.tracker.equity_curve()


# ───────────────────────────────────────────────────────── reports
@app.post("/api/reports/generate")
async def report_generate() -> dict[str, Any]:
    gen = ReportGenerator(tracker=app.state.sched.tracker)
    return gen.generate()


@app.get("/api/reports/list")
async def report_list() -> list[dict]:
    return app.state.sched.tracker.report_list()


@app.get("/api/reports/download/{filename}")
async def report_download(filename: str):
    from fastapi.responses import FileResponse
    p = Path(DEFAULT_REPORTS_DIR) / filename
    if not p.exists():
        raise HTTPException(404, f"report {filename} not found")
    return FileResponse(str(p), media_type="application/pdf",
                        filename=filename)
