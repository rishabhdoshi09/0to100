from __future__ import annotations
import asyncio
import structlog
from app.scrapers.base import BrowserPool, ScrapeResult
from app.scrapers.nse import NSEScraper
from app.scrapers.screener import ScreenerScraper
from app.core.config import settings

log = structlog.get_logger(__name__)

_browser_pool: BrowserPool | None = None


async def get_browser_pool() -> BrowserPool:
    global _browser_pool
    if _browser_pool is None:
        _browser_pool = BrowserPool(pool_size=2)
        await _browser_pool.start()
    return _browser_pool


class ScrapingCoordinator:
    def __init__(self, browser_pool: BrowserPool) -> None:
        self._pool = browser_pool

    async def scrape_company_full(self, symbol: str) -> ScrapeResult:
        nse_scraper = NSEScraper(self._pool)
        screener_scraper = ScreenerScraper(self._pool)

        nse_task = asyncio.create_task(nse_scraper.scrape_company(symbol))
        screener_task = asyncio.create_task(screener_scraper.scrape_company(symbol))

        nse_result, screener_result = await asyncio.gather(nse_task, screener_task, return_exceptions=True)

        merged_data = {}
        if isinstance(nse_result, ScrapeResult) and nse_result.success:
            merged_data["nse"] = nse_result.data
        if isinstance(screener_result, ScrapeResult) and screener_result.success:
            merged_data["screener"] = screener_result.data

        log.info("scrape_complete", symbol=symbol, sources=list(merged_data.keys()))
        return ScrapeResult(symbol=symbol, success=bool(merged_data), data=merged_data)

    async def scrape_filings(self, symbol: str) -> list[dict]:
        nse_scraper = NSEScraper(self._pool)
        result = await nse_scraper.scrape_company(symbol)
        filings_raw = result.data.get("filings", {}).get("data", [])

        seen, filings = set(), []
        for f in filings_raw:
            key = (f.get("subject", ""), f.get("an_dt", ""))
            if key not in seen:
                seen.add(key)
                filings.append(f)
        return filings

    async def scrape_shareholding(self, symbol: str) -> dict:
        screener_scraper = ScreenerScraper(self._pool)
        result = await screener_scraper.scrape_company(symbol)
        return result.data.get("shareholding", {})


async def get_coordinator() -> ScrapingCoordinator:
    pool = await get_browser_pool()
    return ScrapingCoordinator(pool)
