"""
📡 Kite WebSocket live overlay — live quotes for paper entry-eligibility and open-position
management during market hours. DATA ONLY (no orders).

It rejects out-of-order and future-dated ticks, tracks per-symbol staleness, and restores
subscriptions after a reconnect (bounded backoff). It NEVER finalizes an intraday bar as daily
evidence — finalized daily history always flows through the validated snapshot process. When a
symbol's feed is stale, new entries for it are blocked while risk-reducing exits may continue on
the last valid price; fabricated prices are never produced.

The feed is injected (duck-typed) so this is deterministic offline; production wires KiteTicker.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class _Tick:
    price: float
    ts: float                                # epoch seconds (monotonic-ish); provenance


def _feed_socket_pending(feed) -> bool:
    """True for a real KiteTicker whose websocket is not open yet.

    Duck-typed test feeds have no ``ws`` attribute, so they stay synchronous.
    """
    ticker = getattr(feed, "_t", None)
    if ticker is None:
        return False
    return hasattr(ticker, "ws") and getattr(ticker, "ws", None) is None


class KiteLiveOverlay:
    def __init__(self, feed=None, *, max_stale_s: float = 30.0, clock=time.time,
                 sleep_fn=time.sleep, max_reconnect_backoff: float = 8.0):
        self.feed = feed
        self.max_stale_s = max_stale_s
        self._clock = clock
        self._sleep = sleep_fn
        self._max_backoff = max_reconnect_backoff
        self._subs: set = set()
        self._ticks: dict[str, _Tick] = {}
        self._rejected = {"out_of_order": 0, "future": 0}
        self.connected = False
        self.reconnects = 0
        self.last_connect_ts = 0.0

    # ── subscription lifecycle ────────────────────────────────────────────────────
    def connect(self) -> None:
        if self.feed is not None:
            self.feed.connect(self._subs)
            if _feed_socket_pending(self.feed):
                # Threaded KiteTicker.connect() returns before ws exists. Claiming
                # connected here made subscribe() call ws.sendMessage on None.
                self.last_connect_ts = self._clock()
                return
        self.connected = True
        self.last_connect_ts = self._clock()

    def subscribe(self, symbols) -> None:
        self._subs |= {s.upper() for s in symbols}
        if self.connected and self.feed is not None:
            self.feed.subscribe(list(self._subs))

    def on_reconnect(self) -> None:
        """Bounded-backoff reconnect that RESTORES all subscriptions."""
        self.reconnects += 1
        backoff = min(self._max_backoff, 0.05 * (2 ** min(self.reconnects, 8)))
        self._sleep(backoff)
        self.connected = True
        self.last_connect_ts = self._clock()
        if self.feed is not None:
            self.feed.subscribe(list(self._subs))       # subscriptions restored after reconnect

    # ── tick ingestion (reject bad ticks) ─────────────────────────────────────────
    def on_tick(self, symbol: str, price: float, ts: float) -> bool:
        symbol = symbol.upper()
        now = self._clock()
        if ts > now + 1.0:                               # future-dated tick
            self._rejected["future"] += 1; return False
        prev = self._ticks.get(symbol)
        if prev is not None and ts < prev.ts:            # out-of-order tick
            self._rejected["out_of_order"] += 1; return False
        if price <= 0:
            return False
        self._ticks[symbol] = _Tick(price=float(price), ts=float(ts))
        return True

    # ── reads for the loop ─────────────────────────────────────────────────────────
    def price(self, symbol: str):
        t = self._ticks.get(symbol.upper())
        return t.price if t else None

    def last_tick_ts(self, symbol: str):
        t = self._ticks.get(symbol.upper())
        return t.ts if t else None

    def is_stale(self, symbol: str, *, now: float | None = None) -> bool:
        t = self._ticks.get(symbol.upper())
        if t is None:
            return True                                  # never ticked ⇒ stale
        return ((now or self._clock()) - t.ts) > self.max_stale_s

    def entry_allowed(self, symbol: str, *, now: float | None = None) -> bool:
        """New paper entries need a fresh, valid live price for the symbol."""
        return (not self.is_stale(symbol, now=now)) and (self.price(symbol) or 0) > 0

    def health(self) -> dict:
        return {"connected": self.connected, "subscriptions": len(self._subs),
                "reconnects": self.reconnects, "symbols_ticking": len(self._ticks),
                "rejected": dict(self._rejected), "last_connect_ts": self.last_connect_ts}
