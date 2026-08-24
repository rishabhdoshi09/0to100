"""Supervisor-owned, data-only Kite live-feed controller."""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path


class LiveFeedController:
    def __init__(self, status_path=None, *, overlay=None):
        self.status_path = Path(status_path) if status_path else None
        self.overlay = overlay
        self.feed = None
        self.ticker = None
        self.last_error = ""
        self.subscribed: set[str] = set()
        self._quote_at = 0.0
        self._quote_log_at = 0.0

    def _tokens(self, symbols) -> dict[int, str]:
        cache = Path(__file__).resolve().parents[2] / "logs" / "instruments_cache.csv"
        if not cache.exists():
            return {}
        wanted = {str(s).upper() for s in symbols}
        out = {}
        with open(cache, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                sym = str(row.get("tradingsymbol") or "").upper()
                if sym not in wanted or str(row.get("exchange") or "").upper() != "NSE":
                    continue
                try:
                    out[int(float(row["instrument_token"]))] = sym
                except Exception:
                    continue
        return out

    def _ticking(self) -> int:
        if self.overlay is None:
            return 0
        try:
            return int((self.overlay.health() or {}).get("symbols_ticking") or 0)
        except Exception:
            return 0

    def _quote_overlay(self, symbols) -> int:
        """REST LTP into the overlay when the websocket has no ticks. Data only."""
        if self.overlay is None or not callable(getattr(self.overlay, "on_tick", None)):
            return 0
        if self._ticking() > 0:
            return self._ticking()
        now = time.time()
        if now - float(self._quote_at or 0.0) < 8.0:
            return self._ticking()
        try:
            from data.kite_client import KiteClient, _fresh_env
            if not _fresh_env("KITE_API_KEY") or not _fresh_env("KITE_ACCESS_TOKEN"):
                return 0
            prices = KiteClient().get_ltp(sorted(symbols)[:80]) or {}
        except Exception as exc:
            if not self.last_error:
                self.last_error = str(exc)[:240]
            return 0
        stamped = time.time()
        filled = 0
        for sym, raw in prices.items():
            try:
                price = float(raw)
            except (TypeError, ValueError):
                continue
            if price <= 0:
                continue
            self.overlay.on_tick(str(sym).upper(), price, stamped)
            filled += 1
        self._quote_at = stamped
        if filled:
            self.last_error = ""
            self.overlay.connected = True
            if stamped - float(self._quote_log_at or 0.0) >= 60.0:
                self._quote_log_at = stamped
                print(f"[KITE] quote overlay · {filled} symbols · sniper uses LTP until websocket ticks", flush=True)
        return filled

    def start(self, symbols) -> dict:
        symbols = {str(s).upper() for s in symbols if str(s).strip()}
        if not symbols:
            return self.health()
        try:
            token_to_symbol = self._tokens(symbols)
            if self.overlay is None:
                from data.kite_client import _fresh_env
                api_key = _fresh_env("KITE_API_KEY")
                token = _fresh_env("KITE_ACCESS_TOKEN")
                if not api_key or not token:
                    raise RuntimeError("valid Kite credentials are required for live ticks")
                if not token_to_symbol:
                    raise RuntimeError("no approved subscription tokens resolved")
                from kiteconnect import KiteTicker
                from research.intelligence.data.kite_activation import KiteTickerFeed
                from research.intelligence.data.kite_live import KiteLiveOverlay
                self.ticker = KiteTicker(api_key=api_key, access_token=token)
                self.overlay = KiteLiveOverlay()
                self.feed = KiteTickerFeed(self.ticker, token_to_symbol=token_to_symbol,
                                           overlay=self.overlay)
                self.overlay.feed = self.feed
            elif self.feed is not None and token_to_symbol:
                self.feed.add_mappings(token_to_symbol)
            if not getattr(self.overlay, "connected", False):
                self.overlay.connect()
            self.overlay.subscribe(symbols)
            self.subscribed |= symbols
            if "sendMessage" not in str(self.last_error or ""):
                self.last_error = ""
        except Exception as exc:
            self.last_error = str(exc)[:240]
        try:
            self._quote_overlay(self.subscribed or symbols)
        except Exception:
            pass
        self._persist()
        return self.health()

    def stop(self) -> None:
        try:
            if self.ticker is not None and hasattr(self.ticker, "close"):
                self.ticker.close()
        except Exception:
            pass
        if self.overlay is not None:
            self.overlay.connected = False
        self._persist()

    def fresh_symbols(self) -> frozenset[str]:
        if self.overlay is None:
            return frozenset()
        return frozenset(s for s in self.subscribed if self.overlay.entry_allowed(s))

    def entry_allowed(self, symbol: str) -> bool:
        return bool(self.overlay and self.overlay.entry_allowed(symbol))

    def price(self, symbol: str):
        """Latest validated live price for alerting/read-only observation."""
        return self.overlay.price(symbol) if self.overlay is not None else None

    def health(self) -> dict:
        base = self.overlay.health() if self.overlay is not None else {
            "connected": False, "subscriptions": 0, "reconnects": 0,
            "symbols_ticking": 0, "rejected": {}, "last_connect_ts": 0.0,
        }
        stale = []
        if self.overlay is not None:
            stale = sorted(s for s in self.subscribed if self.overlay.is_stale(s))
        return {**base, "subscribed_symbols": sorted(self.subscribed),
                "fresh_symbols": sorted(self.fresh_symbols()), "stale_symbols": stale,
                "last_error": self.last_error}

    def _persist(self) -> None:
        if self.status_path is None:
            return
        try:
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.status_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.health(), indent=2, default=str), encoding="utf-8")
            os.replace(tmp, self.status_path)
        except Exception:
            pass
