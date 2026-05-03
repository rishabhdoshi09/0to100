"""
FastAPI application — REST + WebSocket interface.

Endpoints:
  GET  /health          — liveness check
  GET  /portfolio       — current positions and cash
  GET  /signals         — latest signal per symbol
  GET  /signals/{sym}   — signal for a specific symbol
  GET  /trades          — recent trade history
  GET  /stats           — performance statistics
  POST /signal/force    — manually trigger signal generation for a symbol
  POST /train           — trigger model retraining (async background task)
  POST /backtest        — run a backtest (async)
  WS   /ws              — real-time signal / fill stream

Run: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from api.websocket_manager import WebSocketManager
from config.settings import settings
from data.ingestion.historical import HistoricalIngestion
from data.storage.redis_client import RedisClient


# ── Prometheus metrics ────────────────────────────────────────────────────────
signal_counter = Counter("sqai_signals_total", "Total signals generated", ["action"])
trade_counter = Counter("sqai_trades_total", "Total trades executed", ["side"])
equity_gauge = Gauge("sqai_equity", "Current portfolio equity")
sharpe_gauge = Gauge("sqai_sharpe", "Rolling Sharpe ratio")
win_rate_gauge = Gauge("sqai_win_rate", "Rolling win rate")
request_latency = Histogram("sqai_request_latency_seconds", "API request latency")


# ── App lifecycle ────────────────────────────────────────────────────────────

ws_manager = WebSocketManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start Redis listener in background
    asyncio.create_task(ws_manager.listen_redis())
    logger.info("FastAPI started")
    yield
    logger.info("FastAPI shutting down")


app = FastAPI(
    title="SimpleQuant AI v2",
    description="Probabilistic trading system API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

redis = RedisClient()


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "redis": redis.is_available(),
    }


# ── Portfolio ─────────────────────────────────────────────────────────────────

@app.get("/portfolio")
async def portfolio():
    data = redis.get_portfolio()
    if data is None:
        return {"message": "No portfolio data yet. Start the trading engine first."}
    equity_gauge.set(data.get("total_value", 0))
    return data


# ── Signals ───────────────────────────────────────────────────────────────────

@app.get("/signals")
async def all_signals():
    return redis.get_all_signals()


@app.get("/signals/{symbol}")
async def symbol_signal(symbol: str):
    sig = redis.get_signal(symbol.upper())
    if sig is None:
        raise HTTPException(status_code=404, detail=f"No signal cached for {symbol}")
    return sig


@app.post("/signal/force")
async def force_signal(symbol: str, background_tasks: BackgroundTasks):
    """Trigger on-demand signal generation for a symbol."""
    background_tasks.add_task(_generate_signal_bg, symbol.upper())
    return {"message": f"Signal generation queued for {symbol}"}


async def _generate_signal_bg(symbol: str) -> None:
    try:
        from signals.composite_signal import CompositeSignal
        from data.ingestion.historical import HistoricalIngestion
        from datetime import date
        ingest = HistoricalIngestion()
        from config.settings import settings
        from datetime import datetime
        start = datetime.strptime(settings.backtest_start_date, "%Y-%m-%d").date()
        data = ingest.ingest_symbol(symbol, start, date.today())
        if data is None or data.empty:
            return
        engine = CompositeSignal()
        result = engine.generate(symbol, data)
        sig_dict = {
            "symbol": result.symbol,
            "action": result.action,
            "probability": result.probability,
            "confidence": result.confidence,
            "regime": result.regime,
            "sentiment_score": result.sentiment_score,
            "fundamental_score": result.fundamental_score,
            "component_probs": result.component_probs,
            "timestamp": datetime.utcnow().isoformat(),
        }
        redis.cache_signal(symbol, sig_dict)
        redis.publish_signal({"type": "signal", **sig_dict})
        signal_counter.labels(action=result.action).inc()
        logger.info(f"Force signal: {symbol} → {result.action}")
    except Exception as exc:
        logger.error(f"Force signal failed: {exc}")


# ── Trades ────────────────────────────────────────────────────────────────────

@app.get("/trades")
async def recent_trades(limit: int = 50):
    # Read from Redis stream (populated by OrderManager)
    try:
        entries = redis.client.xrevrange("fills", count=limit)
        return [dict(e[1]) for e in entries]
    except Exception:
        return []


# ── Performance stats ─────────────────────────────────────────────────────────

@app.get("/stats")
async def performance_stats():
    portfolio = redis.get_portfolio()
    return {
        "total_value": portfolio.get("total_value") if portfolio else None,
        "daily_pnl": portfolio.get("daily_pnl") if portfolio else None,
        "n_positions": len(portfolio.get("positions", {})) if portfolio else 0,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ── Training trigger ──────────────────────────────────────────────────────────

@app.post("/train")
async def trigger_training(background_tasks: BackgroundTasks):
    background_tasks.add_task(_train_bg)
    return {"message": "Model training queued. Check logs for progress."}


async def _train_bg() -> None:
    from models.train_pipeline import TrainingPipeline
    from data.ingestion.historical import HistoricalIngestion
    from datetime import datetime
    start = datetime.strptime(settings.backtest_start_date, "%Y-%m-%d").date()
    end = datetime.strptime(settings.backtest_end_date, "%Y-%m-%d").date()
    ingest = HistoricalIngestion()
    data = ingest.ingest_all(from_date=start, to_date=end)
    pipeline = TrainingPipeline()
    pipeline.run(data)


# ── Backtest trigger ──────────────────────────────────────────────────────────

@app.post("/backtest")
async def trigger_backtest(
    start_date: str = settings.backtest_start_date,
    end_date: str = settings.backtest_end_date,
    background_tasks: BackgroundTasks = None,
):
    background_tasks.add_task(_backtest_bg, start_date, end_date)
    return {"message": f"Backtest queued: {start_date} → {end_date}"}


async def _backtest_bg(start_date: str, end_date: str) -> None:
    from backtest.engine import BacktestEngine
    from datetime import datetime
    from data.ingestion.historical import HistoricalIngestion
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    ingest = HistoricalIngestion()
    data = ingest.load_from_cache(from_date=start, to_date=end)
    engine = BacktestEngine(data)
    results = engine.run()
    stats = results.get("stats", {})
    logger.info(f"Backtest result: {stats}")
    redis.client.set("backtest:latest", str(stats), ex=86400)


# ── Prometheus metrics ────────────────────────────────────────────────────────

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            # Keep connection alive; broadcast is handled by listen_redis()
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
