"""
WebSocket connection manager for real-time signal / fill broadcasting.
Subscribes to Redis pub/sub and fans out to all connected WS clients.
"""

from __future__ import annotations

import asyncio
import json
from typing import List

from fastapi import WebSocket
from loguru import logger

from data.storage.redis_client import RedisClient


class WebSocketManager:
    def __init__(self) -> None:
        self._connections: List[WebSocket] = []
        self._redis = RedisClient()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)
        logger.info(f"WS connected: {len(self._connections)} total")

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)
        logger.info(f"WS disconnected: {len(self._connections)} remaining")

    async def broadcast(self, message: dict) -> None:
        dead = []
        for ws in self._connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    async def listen_redis(self) -> None:
        """
        Background coroutine: subscribe to Redis pub/sub and broadcast to WS clients.
        """
        if not self._redis.is_available():
            logger.warning("Redis unavailable — WS real-time feed disabled")
            return

        pubsub = self._redis.pubsub()
        pubsub.subscribe(self._redis._CHANNEL_SIGNALS)
        logger.info("Redis pub/sub listener started")

        loop = asyncio.get_event_loop()
        while True:
            try:
                msg = await loop.run_in_executor(None, pubsub.get_message, True, 0.1)
                if msg and msg["type"] == "message":
                    data = json.loads(msg["data"])
                    await self.broadcast(data)
            except Exception as exc:
                logger.debug(f"WS Redis listener error: {exc}")
            await asyncio.sleep(0.05)
