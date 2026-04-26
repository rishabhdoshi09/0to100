"""
News normalizer: filters articles by relevance to the trading universe
and normalizes into a consistent structured format for downstream use.
"""

from __future__ import annotations

import re
from typing import List

from config import settings
from news.fetcher import RawArticle
from logger import get_logger

log = get_logger(__name__)


class NormalizedArticle:
    """Structured news item ready for LLM context injection."""

    __slots__ = ("id", "headline", "summary", "source", "published_at", "mentioned_symbols")

    def __init__(
        self,
        id: str,
        headline: str,
        summary: str,
        source: str,
        published_at: str,
        mentioned_symbols: List[str],
    ) -> None:
        self.id = id
        self.headline = headline
        self.summary = summary
        self.source = source
        self.published_at = published_at
        self.mentioned_symbols = mentioned_symbols

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "headline": self.headline,
            "summary": self.summary,
            "source": self.source,
            "published_at": self.published_at,
            "mentioned_symbols": self.mentioned_symbols,
        }


class NewsNormalizer:
    def __init__(self) -> None:
        self._universe = set(settings.symbol_list)

    def normalize(self, articles: List[RawArticle]) -> List[NormalizedArticle]:
        """
        Filter by universe relevance and clean.
        Articles with no symbol mention are kept as macro context.
        """
        normalized: List[NormalizedArticle] = []
        for art in articles:
            text = f"{art.headline} {art.summary}".upper()
            mentioned = [s for s in self._universe if s in text]
            normalized.append(
                NormalizedArticle(
                    id=art.id,
                    headline=art.headline,
                    summary=art.summary[:400],
                    source=art.source,
                    published_at=art.published_at.isoformat(),
                    mentioned_symbols=mentioned,
                )
            )

        log.debug("news_normalized", count=len(normalized))
        return normalized
