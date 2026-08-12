from __future__ import annotations
import asyncio
import structlog
from app.workers.celery_app import celery_app

log = structlog.get_logger(__name__)


@celery_app.task(name="app.workers.tasks.ai_extraction.extract_filing", bind=True, max_retries=2)
def extract_filing(self, filing_id: str):
    try:
        asyncio.run(_extract_filing_async(filing_id))
    except Exception as exc:
        log.error("extract_filing_failed", filing_id=filing_id, error=str(exc))
        raise self.retry(exc=exc, countdown=120)


async def _extract_filing_async(filing_id: str):
    import uuid
    import httpx
    log.info("extracting_filing", filing_id=filing_id)
    # TODO: Download PDF, extract text with pypdf, call LLM for structured extraction
    # Then queue embedding generation
    from app.workers.tasks.embeddings import generate_filing_embeddings
    generate_filing_embeddings.delay(filing_id)


@celery_app.task(name="app.workers.tasks.ai_extraction.process_pending_filings")
def process_pending_filings():
    asyncio.run(_process_pending_async())


async def _process_pending_async():
    log.info("processing_pending_filings")
    # TODO: Query DB for pending filings and dispatch extract_filing tasks
