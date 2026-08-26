"""Screener.in adapter. Acquire calls this; GET never does."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from product.due_diligence.providers.base import FetchResult, ProviderError, empty_normalized


class ScreenerAdapter:
    name = "screener.in"
    source_type = "established_financial_data"

    def fetch(self, symbol: str, *, force: bool = False) -> FetchResult:
        from fundamentals.fetcher import get_deep_fundamentals

        url = f"https://www.screener.in/company/{symbol.upper()}/"
        retrieved = datetime.now(timezone.utc).isoformat()
        try:
            payload = get_deep_fundamentals(symbol, force_refresh=force)
        except Exception as exc:
            return FetchResult(
                ok=False,
                provider=self.name,
                url=url,
                retrieved_at=retrieved,
                error=str(exc)[:240],
            )
        return FetchResult(
            ok=True,
            provider=self.name,
            url=str((payload or {}).get("url") or url),
            retrieved_at=retrieved,
            status_code=200,
            content_type="application/json",
            body=b"",
            error="",
        )

    def parse(self, result: FetchResult) -> dict[str, Any]:
        if not result.ok:
            raise ProviderError(result.error or "screener.in unavailable")
        return empty_normalized()
