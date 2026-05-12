from __future__ import annotations
import structlog
from app.scrapers.base import BaseScraper, BrowserPool, ScrapeResult
from app.core.config import settings

log = structlog.get_logger(__name__)

_ENDPOINTS = {
    "quote": "/api/quote-equity?symbol={symbol}",
    "corp_info": "/api/corp-info?symbol={symbol}",
    "annual_reports": "/api/annual-reports?symbol={symbol}&industry=",
    "shareholding": "/api/corporate-shareholding-patterns?symbol={symbol}&from=2023-01-01&to=2024-12-31",
    "filings": "/api/corp-announcements?symbol={symbol}&subject=",
}


class NSEScraper(BaseScraper):
    def __init__(self, browser_pool: BrowserPool) -> None:
        super().__init__(browser_pool)

    async def scrape_company(self, symbol: str) -> ScrapeResult:
        ctx = await self._pool.new_context()
        try:
            page = await ctx.new_page()
            self._setup_interceptor(page)
            await self._establish_session(page)
            data = {}
            for key, endpoint_tpl in _ENDPOINTS.items():
                endpoint = endpoint_tpl.format(symbol=symbol)
                try:
                    api_data = await self._call_internal_api(page, endpoint)
                    if api_data:
                        data[key] = api_data
                except Exception as exc:
                    log.warning("nse_endpoint_failed", symbol=symbol, endpoint=key, error=str(exc))
            return ScrapeResult(symbol=symbol, success=bool(data), data=data, intercepted=self._intercepted)
        except Exception as exc:
            log.error("nse_scrape_failed", symbol=symbol, error=str(exc))
            return ScrapeResult(symbol=symbol, success=False, error=str(exc))
        finally:
            await ctx.close()

    async def _establish_session(self, page) -> None:
        await self._navigate_with_retry(page, settings.nse_base_url)
        await self._human_delay(1000, 2000)

    async def _call_internal_api(self, page, endpoint: str) -> dict | None:
        url = f"{settings.nse_base_url}{endpoint}"
        return await page.evaluate(
            f"""async () => {{
                const r = await fetch('{url}', {{ headers: {{ 'Accept': 'application/json' }} }});
                return r.ok ? await r.json() : null;
            }}"""
        )
