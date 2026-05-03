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
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sq_ai.backend.executor import Order
from sq_ai.backend.scheduler import TradingScheduler


class ManualTrade(BaseModel):
    symbol: str
    action: str
    qty: int
    price: float
    stop: float = 0.0
    target: float = 0.0


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
