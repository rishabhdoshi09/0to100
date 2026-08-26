"""Named StockResearchEngine — the single research entry point.

Scanners and the manual Stock Investigator call this. It never scans the market.
GET /api/due-diligence must keep using cache-only builders (this module does not fetch).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Sequence

from product.due_diligence.engine import build_due_diligence
from product.due_diligence.suggest import suggest_tickers


class StockResearchEngine:
    """Ticker → company master → sector framework → rule-based research dashboard."""

    def investigate(
        self,
        symbol: str,
        *,
        scan_payload: Mapping[str, Any] | None = None,
        long_term_payload: Mapping[str, Any] | None = None,
        raw_fundamentals: Mapping[str, Any] | None = None,
        news: Sequence[Mapping[str, Any]] | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        return build_due_diligence(
            symbol,
            scan_payload=scan_payload,
            long_term_payload=long_term_payload,
            raw_fundamentals=raw_fundamentals,
            news=news,
            now=now,
        )

    def suggest(self, query: str, *, limit: int = 8) -> list[dict[str, Any]]:
        return suggest_tickers(query, limit=limit)


def investigate_stock(symbol: str, **kwargs: Any) -> dict[str, Any]:
    return StockResearchEngine().investigate(symbol, **kwargs)
