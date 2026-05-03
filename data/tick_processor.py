"""
Live tick data processor.

Subscribes to Kite WebSocket ticks and maintains an in-memory
OHLCV bar builder (1-minute bars by default).
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from logger import get_logger

log = get_logger(__name__)


class TickBar:
    """Accumulates ticks into a single OHLCV bar."""

    __slots__ = ("symbol", "open", "high", "low", "close", "volume", "timestamp", "ticks")

    def __init__(self, symbol: str, price: float, volume: int, ts: datetime) -> None:
        self.symbol = symbol
        self.open = price
        self.high = price
        self.low = price
        self.close = price
        self.volume = volume
        self.timestamp = ts
        self.ticks = 1

    def update(self, price: float, volume: int) -> None:
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += volume
        self.ticks += 1

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


class TickProcessor:
    """
    Collects raw ticks from KiteTicker callbacks,
    builds rolling 1-minute OHLCV bars, and fires bar_callbacks
    whenever a bar is completed.
    """

    def __init__(self, bar_interval_seconds: int = 60) -> None:
        self._interval = bar_interval_seconds
        self._lock = threading.Lock()
        self._current_bars: Dict[int, TickBar] = {}      # token → current bar
        self._bar_start: Dict[int, int] = {}              # token → epoch bucket
        self._token_symbol: Dict[int, str] = {}           # token → symbol
        self._bar_callbacks: List[Callable] = []
        self._latest_ltp: Dict[int, float] = {}

    def register_token(self, token: int, symbol: str) -> None:
        with self._lock:
            self._token_symbol[token] = symbol

    def add_bar_callback(self, fn: Callable[[Dict], None]) -> None:
        self._bar_callbacks.append(fn)

    def on_ticks(self, ws, ticks: List[Dict]) -> None:  # noqa: ARG002
        now = datetime.now(timezone.utc)
        bucket = int(now.timestamp()) // self._interval * self._interval
        with self._lock:
            for tick in ticks:
                token: int = tick.get("instrument_token", 0)
                price: float = tick.get("last_price", 0.0)
                volume: int = tick.get("last_traded_quantity", 0)
                self._latest_ltp[token] = price
                symbol = self._token_symbol.get(token, str(token))

                if token not in self._bar_start:
                    self._bar_start[token] = bucket
                    self._current_bars[token] = TickBar(symbol, price, volume, now)
                    continue

                if bucket != self._bar_start[token]:
                    # Completed bar — fire callbacks
                    completed = self._current_bars[token]
                    log.debug("bar_completed", symbol=symbol, bar=completed.to_dict())
                    for cb in self._bar_callbacks:
                        try:
                            cb(completed.to_dict())
                        except Exception as exc:
                            log.error("bar_callback_error", error=str(exc))
                    # Start new bar
                    self._bar_start[token] = bucket
                    self._current_bars[token] = TickBar(symbol, price, volume, now)
                else:
                    self._current_bars[token].update(price, volume)

    def get_ltp(self, token: int) -> Optional[float]:
        return self._latest_ltp.get(token)

    def get_current_bar(self, token: int) -> Optional[Dict]:
        bar = self._current_bars.get(token)
        return bar.to_dict() if bar else None
