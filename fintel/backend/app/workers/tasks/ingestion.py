from __future__ import annotations
import structlog
from app.workers.celery_app import celery_app

log = structlog.get_logger(__name__)


@celery_app.task(name="app.workers.tasks.ingestion.sync_nse_filings", bind=True, max_retries=3)
def sync_nse_filings(self):
    import asyncio
    try:
        asyncio.run(_sync_nse_filings_async())
    except Exception as exc:
        log.error("sync_nse_filings_failed", error=str(exc))
        raise self.retry(exc=exc, countdown=60)


async def _sync_nse_filings_async():
    log.info("syncing_nse_filings")
    # TODO: implement with NSEScraper


@celery_app.task(name="app.workers.tasks.ingestion.update_price_snapshots", bind=True, max_retries=3)
def update_price_snapshots(self):
    import asyncio
    try:
        asyncio.run(_update_prices_async())
    except Exception as exc:
        log.error("update_prices_failed", error=str(exc))
        raise self.retry(exc=exc, countdown=30)


async def _update_prices_async():
    log.info("updating_price_snapshots")
    # TODO: implement with NSE price API
