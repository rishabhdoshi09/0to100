"""
Qdrant local vector index for financial news articles.

Uses OpenAI text-embedding-3-small (1536-dim, $0.00002/1K tokens — ~$0.0002
per full news cycle) for embedding, stored in a local on-disk Qdrant instance
(~150MB RAM, no server process needed).

Graceful degradation: if OPENAI_API_KEY is absent or Qdrant is unavailable,
search() falls back to keyword matching so nothing breaks downstream.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, List

log = logging.getLogger(__name__)

_COLLECTION = "devbloom_news"
_DIM = 1536
_DB_PATH = "data/news_vectors"


class SemanticNewsIndex:
    def __init__(self) -> None:
        os.makedirs(_DB_PATH, exist_ok=True)
        self._ready = False
        self._q = None
        self._oai = None
        self._init()

    def _init(self) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self._q = QdrantClient(path=_DB_PATH)
            existing = [c.name for c in self._q.get_collections().collections]
            if _COLLECTION not in existing:
                self._q.create_collection(
                    _COLLECTION,
                    vectors_config=VectorParams(size=_DIM, distance=Distance.COSINE),
                )
        except Exception as exc:
            log.warning("qdrant_init_failed: %s", exc)
            return

        try:
            from openai import OpenAI
            from config import settings
            api_key = settings.openai_api_key if hasattr(settings, "openai_api_key") else ""
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                log.warning("semantic_index: OPENAI_API_KEY not set; falling back to keyword search")
                return
            self._oai = OpenAI(api_key=api_key)
        except Exception as exc:
            log.warning("openai_init_failed: %s", exc)
            return

        self._ready = True

    # ── Public API ─────────────────────────────────────────────────────────

    def index(self, articles) -> int:
        """
        Embed and upsert articles into Qdrant.
        articles: list of RawArticle objects (from news.fetcher).
        Returns number of points upserted (0 on failure).
        """
        if not self._ready or not articles:
            return 0
        try:
            from qdrant_client.models import PointStruct

            texts = [f"{a.headline}. {a.summary[:300]}" for a in articles]
            resp = self._oai.embeddings.create(
                model="text-embedding-3-small", input=texts
            )
            points = []
            for a, emb_obj in zip(articles, resp.data):
                pid = int(hashlib.sha1(a.id.encode()).hexdigest()[:8], 16)
                points.append(
                    PointStruct(
                        id=pid,
                        vector=emb_obj.embedding,
                        payload=a.to_dict(),
                    )
                )
            self._q.upsert(_COLLECTION, points)
            log.debug("semantic_index: upserted %d articles", len(points))
            return len(points)
        except Exception as exc:
            log.warning("semantic_index_upsert_failed: %s", exc)
            return 0

    def search(self, symbol: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """
        Return top-k articles most semantically relevant to the symbol.
        Falls back to empty list (caller handles gracefully) on any failure.
        """
        if not self._ready:
            return self._keyword_fallback(symbol, top_k)
        try:
            query = f"NSE {symbol} stock earnings revenue results guidance outlook"
            emb = self._oai.embeddings.create(
                model="text-embedding-3-small", input=[query]
            ).data[0].embedding
            hits = self._q.search(_COLLECTION, emb, limit=top_k)
            return [h.payload for h in hits]
        except Exception as exc:
            log.warning("semantic_search_failed: %s", exc)
            return self._keyword_fallback(symbol, top_k)

    # ── Fallback ───────────────────────────────────────────────────────────

    @staticmethod
    def _keyword_fallback(symbol: str, top_k: int) -> List[Dict[str, Any]]:
        """Substring keyword match — same logic as existing normalizer."""
        try:
            from news.fetcher import NewsFetcher
            articles = NewsFetcher().fetch_all()
            sym_lower = symbol.lower()
            relevant = [
                a.to_dict() for a in articles
                if sym_lower in a.headline.lower() or sym_lower in a.summary.lower()
            ]
            return relevant[:top_k] if relevant else [a.to_dict() for a in articles[:top_k]]
        except Exception:
            return []
