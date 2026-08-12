from __future__ import annotations
import asyncio
import structlog
from app.workers.celery_app import celery_app

log = structlog.get_logger(__name__)


@celery_app.task(name="app.workers.tasks.scraping.scrape_company", bind=True, max_retries=3)
def scrape_company(self, symbol: str):
    try:
        asyncio.run(_scrape_company_async(symbol))
    except Exception as exc:
        log.error("scrape_company_failed", symbol=symbol, error=str(exc))
        raise self.retry(exc=exc, countdown=300)


async def _scrape_company_async(symbol: str):
    from app.scrapers.coordinator import get_coordinator
    coordinator = await get_coordinator()
    result = await coordinator.scrape_company_full(symbol)
    log.info("scrape_company_complete", symbol=symbol, success=result.success)


@celery_app.task(name="app.workers.tasks.scraping.scrape_filings", bind=True, max_retries=3)
def scrape_filings(self, symbol: str):
    try:
        asyncio.run(_scrape_filings_async(symbol))
    except Exception as exc:
        log.error("scrape_filings_failed", symbol=symbol, error=str(exc))
        raise self.retry(exc=exc, countdown=300)


async def _scrape_filings_async(symbol: str):
    from app.scrapers.coordinator import get_coordinator
    coordinator = await get_coordinator()
    await coordinator.scrape_filings(symbol)


@celery_app.task(name="app.workers.tasks.scraping.scrape_shareholding", bind=True, max_retries=3)
def scrape_shareholding(self, symbol: str):
    try:
        asyncio.run(_scrape_shareholding_async(symbol))
    except Exception as exc:
        log.error("scrape_shareholding_failed", symbol=symbol, error=str(exc))
        raise self.retry(exc=exc, countdown=300)


async def _scrape_shareholding_async(symbol: str):
    from app.scrapers.coordinator import get_coordinator
    coordinator = await get_coordinator()
    await coordinator.scrape_shareholding(symbol)
