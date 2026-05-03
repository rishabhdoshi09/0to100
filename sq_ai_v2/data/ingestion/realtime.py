"""
Real-time tick ingestion via Kite Connect WebSocket.
Publishes incoming ticks to Redis Streams for downstream consumers.
Gracefully degrades to a synthetic tick generator when no API key exists.
"""

from __future__ import annotations

import asyncio
import json
import random
import threading
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional

from loguru import logger

from config.settings import settings
from data.storage.redis_client import RedisClient


_STREAM_KEY_PREFIX = "ticks:"   # Redis stream key per symbol


class RealTimeIngestion:
    """
    Subscribes to Kite WebSocket ticks for a list of symbols and
    publishes each tick as a Redis Stream entry.
    """

    def __init__(self, symbols: Optional[List[str]] = None) -> None:
        self._symbols = symbols or settings.symbol_list
        self._redis = RedisClient()
        self._running = False
        self._kite_ticker = None
        self._thread: Optional[threading.Thread] = None

    # ── Start / Stop ──────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True

        if settings.kite_api_key and settings.kite_access_token:
            self._start_kite_ws()
        else:
            logger.warning("No Kite credentials — starting synthetic tick generator")
            self._thread = threading.Thread(target=self._synthetic_ticker, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._kite_ticker is not None:
            try:
                self._kite_ticker.stop()
            except Exception:
                pass
        logger.info("RealTimeIngestion stopped")

    # ── Kite WebSocket ────────────────────────────────────────────────────

    def _start_kite_ws(self) -> None:
        try:
            from kiteconnect import KiteConnect, KiteTicker

            kite = KiteConnect(api_key=settings.kite_api_key)
            kite.set_access_token(settings.kite_access_token)

            # Resolve instrument tokens
            instruments = kite.instruments("NSE")
            token_map: Dict[int, str] = {}
            for inst in instruments:
                if inst["tradingsymbol"] in self._symbols:
                    token_map[inst["instrument_token"]] = inst["tradingsymbol"]

            if not token_map:
                logger.warning("No instrument tokens resolved; falling back to synthetic")
                self._thread = threading.Thread(
                    target=self._synthetic_ticker, daemon=True
                )
                self._thread.start()
                return

            self._kite_ticker = KiteTicker(
                api_key=settings.kite_api_key,
                access_token=settings.kite_access_token,
            )

            def on_ticks(ws, ticks):
                for tick in ticks:
                    sym = token_map.get(tick["instrument_token"], "UNKNOWN")
                    self._publish_tick(sym, tick)

            def on_connect(ws, response):
                ws.subscribe(list(token_map.keys()))
                ws.set_mode(ws.MODE_FULL, list(token_map.keys()))
                logger.info(f"Kite WebSocket connected, subscribed to {len(token_map)} symbols")

            def on_error(ws, code, reason):
                logger.error(f"Kite WS error {code}: {reason}")

            def on_close(ws, code, reason):
                logger.warning(f"Kite WS closed {code}: {reason}")

            self._kite_ticker.on_ticks = on_ticks
            self._kite_ticker.on_connect = on_connect
            self._kite_ticker.on_error = on_error
            self._kite_ticker.on_close = on_close

            self._thread = threading.Thread(
                target=self._kite_ticker.connect, kwargs={"threaded": False}, daemon=True
            )
            self._thread.start()
            logger.info("Kite WebSocket thread started")

        except ImportError:
            logger.error("kiteconnect not installed; falling back to synthetic")
            self._thread = threading.Thread(target=self._synthetic_ticker, daemon=True)
            self._thread.start()

    # ── Redis publish ─────────────────────────────────────────────────────

    def _publish_tick(self, symbol: str, tick: dict) -> None:
        stream_key = f"{_STREAM_KEY_PREFIX}{symbol}"
        payload = {
            "symbol": symbol,
            "timestamp": tick.get("timestamp", datetime.utcnow()).isoformat()
            if hasattr(tick.get("timestamp"), "isoformat")
            else datetime.utcnow().isoformat(),
            "last_price": str(tick.get("last_price", 0)),
            "volume": str(tick.get("volume", 0)),
            "bid": str(tick.get("depth", {}).get("buy", [{}])[0].get("price", 0)),
            "ask": str(tick.get("depth", {}).get("sell", [{}])[0].get("price", 0)),
        }
        try:
            self._redis.client.xadd(stream_key, payload, maxlen=10_000)
        except Exception as exc:
            logger.debug(f"Redis xadd failed: {exc}")

    # ── Synthetic tick generator (offline mode) ────────────────────────────

    def _synthetic_ticker(self) -> None:
        prices = {s: random.uniform(500, 3000) for s in self._symbols}
        logger.info(f"Synthetic ticker running for {self._symbols}")

        while self._running:
            for sym in self._symbols:
                pct = random.gauss(0, 0.0002)
                prices[sym] *= 1 + pct
                fake_tick = {
                    "last_price": round(prices[sym], 2),
                    "volume": random.randint(100, 5000),
                    "timestamp": datetime.utcnow(),
                }
                self._publish_tick(sym, fake_tick)
            time.sleep(1.0)   # one tick per second per symbol in synthetic mode

    # ── Async reader ──────────────────────────────────────────────────────

    async def read_latest(self, symbol: str, count: int = 100) -> list:
        """Read latest ticks from Redis Stream for a symbol."""
        stream_key = f"{_STREAM_KEY_PREFIX}{symbol}"
        try:
            entries = self._redis.client.xrevrange(stream_key, count=count)
            return [dict(entry[1]) for entry in entries]
        except Exception as exc:
            logger.debug(f"Redis xrevrange failed: {exc}")
            return []
