"""
Redis client — used for:
  • Live tick streams (Redis Streams)
  • Signal cache (latest signal per symbol)
  • Pub/Sub for API websocket broadcast
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import redis
from loguru import logger

from config.settings import settings


class RedisClient:
    """
    Lazy-connect Redis client.  All methods silently degrade when Redis
    is not reachable (returns None / empty dict / False).
    """

    _SIGNAL_KEY_PREFIX = "signal:"
    _PRICE_KEY_PREFIX = "price:"
    _CHANNEL_SIGNALS = "channel:signals"

    def __init__(self) -> None:
        self._client: Optional[redis.Redis] = None

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            kwargs: Dict[str, Any] = {
                "host": settings.redis_host,
                "port": settings.redis_port,
                "decode_responses": True,
            }
            if settings.redis_password:
                kwargs["password"] = settings.redis_password
            self._client = redis.Redis(**kwargs)
        return self._client

    def is_available(self) -> bool:
        try:
            return self.client.ping()
        except Exception:
            return False

    # ── Signal cache ──────────────────────────────────────────────────────

    def cache_signal(self, symbol: str, signal: Dict, ttl: int = 3600) -> None:
        key = f"{self._SIGNAL_KEY_PREFIX}{symbol}"
        try:
            self.client.setex(key, ttl, json.dumps(signal))
        except Exception as exc:
            logger.debug(f"Redis cache_signal failed: {exc}")

    def get_signal(self, symbol: str) -> Optional[Dict]:
        key = f"{self._SIGNAL_KEY_PREFIX}{symbol}"
        try:
            val = self.client.get(key)
            return json.loads(val) if val else None
        except Exception:
            return None

    def get_all_signals(self) -> Dict[str, Dict]:
        result: Dict[str, Dict] = {}
        try:
            keys = self.client.keys(f"{self._SIGNAL_KEY_PREFIX}*")
            for key in keys:
                sym = key.replace(self._SIGNAL_KEY_PREFIX, "")
                sig = self.get_signal(sym)
                if sig:
                    result[sym] = sig
        except Exception as exc:
            logger.debug(f"Redis get_all_signals failed: {exc}")
        return result

    # ── Price cache ───────────────────────────────────────────────────────

    def cache_price(self, symbol: str, price: float, ttl: int = 60) -> None:
        key = f"{self._PRICE_KEY_PREFIX}{symbol}"
        try:
            self.client.setex(key, ttl, str(price))
        except Exception as exc:
            logger.debug(f"Redis cache_price failed: {exc}")

    def get_price(self, symbol: str) -> Optional[float]:
        key = f"{self._PRICE_KEY_PREFIX}{symbol}"
        try:
            val = self.client.get(key)
            return float(val) if val else None
        except Exception:
            return None

    # ── Pub/Sub for websocket fan-out ─────────────────────────────────────

    def publish_signal(self, payload: Dict) -> None:
        try:
            self.client.publish(self._CHANNEL_SIGNALS, json.dumps(payload))
        except Exception as exc:
            logger.debug(f"Redis publish failed: {exc}")

    def pubsub(self):
        return self.client.pubsub()

    # ── Portfolio state ───────────────────────────────────────────────────

    def cache_portfolio(self, portfolio: Dict, ttl: int = 300) -> None:
        try:
            self.client.setex("portfolio:state", ttl, json.dumps(portfolio))
        except Exception as exc:
            logger.debug(f"Redis portfolio cache failed: {exc}")

    def get_portfolio(self) -> Optional[Dict]:
        try:
            val = self.client.get("portfolio:state")
            return json.loads(val) if val else None
        except Exception:
            return None

    # ── Tick stream reader ────────────────────────────────────────────────

    def read_ticks(self, symbol: str, count: int = 100) -> list:
        try:
            entries = self.client.xrevrange(f"ticks:{symbol}", count=count)
            return [dict(e[1]) for e in entries]
        except Exception:
            return []
