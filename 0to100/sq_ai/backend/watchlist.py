"""Watchlist CRUD – thin wrapper over PortfolioTracker."""
from __future__ import annotations

from sq_ai.portfolio.tracker import PortfolioTracker


class WatchlistService:
    def __init__(self, tracker: PortfolioTracker | None = None) -> None:
        self.tracker = tracker or PortfolioTracker()

    def list(self) -> list[dict]:
        return self.tracker.watchlist_list()

    def add(self, symbol: str, note: str = "") -> dict:
        symbol = symbol.upper().strip()
        if not symbol:
            raise ValueError("empty symbol")
        if not symbol.endswith(".NS") and "." not in symbol:
            symbol = f"{symbol}.NS"
        self.tracker.watchlist_add(symbol, note)
        return {"status": "ok", "symbol": symbol}

    def remove(self, symbol: str) -> dict:
        n = self.tracker.watchlist_remove(symbol)
        return {"status": "ok" if n else "not_found", "removed": n}


__all__ = ["WatchlistService"]
