"""FastAPI app – read-only cockpit endpoints + manual trade trigger.

All routes are ``/api/*`` (matches the platform routing convention).
Uses an APScheduler-backed ``TradingScheduler`` started in the lifespan.
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


# ─────────────────────────────────────────────────────────────────────────────
class ManualTrade(BaseModel):
    symbol: str
    action: str          # BUY / SELL
    qty: int
    price: float
    stop: float = 0.0
    target: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
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


app = FastAPI(title="sq_ai cockpit", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────── health
@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "service": "sq_ai", "version": app.version}


# ─────────────────────────────────────────────────────────────────────── portfolio
@app.get("/api/portfolio")
async def portfolio() -> dict[str, Any]:
    sched: TradingScheduler = app.state.sched
    snap = sched.last_cycle.get("snapshot")
    if not snap:
        # cold-start: best-effort snapshot with no live prices
        snap = sched._snapshot({})
    return snap


@app.get("/api/positions")
async def positions() -> list[dict]:
    sched: TradingScheduler = app.state.sched
    return sched.tracker.open_positions()


@app.get("/api/trades")
async def trades(limit: int = 100) -> list[dict]:
    sched: TradingScheduler = app.state.sched
    return sched.tracker.closed_trades(limit=limit)


# ───────────────────────────────────────────────────────────────────────── signals
@app.get("/api/signals/latest")
async def signals_latest(limit: int = 20) -> list[dict]:
    sched: TradingScheduler = app.state.sched
    return sched.tracker.latest_signals(limit=limit)


@app.get("/api/cycle/last")
async def last_cycle() -> dict[str, Any]:
    sched: TradingScheduler = app.state.sched
    return sched.last_cycle or {"note": "no cycle yet"}


@app.post("/api/cycle/run")
async def run_cycle_now() -> dict[str, Any]:
    sched: TradingScheduler = app.state.sched
    return sched.run_cycle()


# ───────────────────────────────────────────────────────────────────────── screener
@app.get("/api/screener")
async def screener_results() -> list[dict]:
    sched: TradingScheduler = app.state.sched
    return sched.tracker.latest_screener()


@app.post("/api/screener/run")
async def screener_run() -> dict[str, Any]:
    sched: TradingScheduler = app.state.sched
    return sched.run_screener()


# ─────────────────────────────────────────────────────────────────────────── trade
@app.post("/api/trade")
async def trade(t: ManualTrade) -> dict[str, Any]:
    sched: TradingScheduler = app.state.sched
    if t.action.upper() == "BUY":
        return sched.executor.buy(Order(t.symbol, "BUY", t.qty, t.price, t.stop, t.target))
    if t.action.upper() == "SELL":
        # find the open trade for this symbol (FIFO)
        for pos in sched.tracker.open_positions():
            if pos["symbol"] == t.symbol:
                return sched.executor.exit_position(pos["id"], t.price)
        raise HTTPException(404, f"no open position for {t.symbol}")
    raise HTTPException(400, f"unknown action {t.action}")
