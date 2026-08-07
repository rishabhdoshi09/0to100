from __future__ import annotations
import asyncio
import structlog
from app.workers.celery_app import celery_app

log = structlog.get_logger(__name__)

_CHUNK_SIZE = 512
_CHUNK_OVERLAP = 64


def _chunk_text(text: str) -> list[str]:
    sentences = text.replace("\n", " ").split(". ")
    chunks, current, current_len = [], [], 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        words = sentence.split()
        if current_len + len(words) > _CHUNK_SIZE:
            if current:
                chunks.append(". ".join(current) + ".")
            # Overlap: keep last N words worth of sentences
            overlap, overlap_len = [], 0
            for s in reversed(current):
                s_words = s.split()
                if overlap_len + len(s_words) <= _CHUNK_OVERLAP:
                    overlap.insert(0, s)
                    overlap_len += len(s_words)
                else:
                    break
            current = overlap
            current_len = overlap_len
        current.append(sentence)
        current_len += len(words)

    if current:
        chunks.append(". ".join(current) + ".")
    return chunks


@celery_app.task(name="app.workers.tasks.embeddings.generate_filing_embeddings", bind=True, max_retries=3)
def generate_filing_embeddings(self, filing_id: str):
    try:
        asyncio.run(_generate_embeddings_async(filing_id))
    except Exception as exc:
        log.error("generate_embeddings_failed", filing_id=filing_id, error=str(exc))
        raise self.retry(exc=exc, countdown=60)


async def _generate_embeddings_async(filing_id: str):
    log.info("generating_embeddings", filing_id=filing_id)
    # TODO: Load filing text, chunk it, call OpenAI embeddings, store in Qdrant
