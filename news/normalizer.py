"""Normalize news and map it to the full NSE/F&O universe."""
from __future__ import annotations

from typing import List

from news.fetcher import RawArticle
from logger import get_logger

log = get_logger(__name__)


class NormalizedArticle:
    """Structured news item ready for LLM context injection."""

    __slots__ = (
        "id", "headline", "summary", "source", "published_at",
        "mentioned_symbols", "url", "fno_symbols",
    )

    def __init__(
        self,
        id: str,
        headline: str,
        summary: str,
        source: str,
        published_at: str,
        mentioned_symbols: List[str],
        url: str = "",
        fno_symbols: List[str] | None = None,
    ) -> None:
        self.id = id
        self.headline = headline
        self.summary = summary
        self.source = source
        self.published_at = published_at
        self.mentioned_symbols = mentioned_symbols
        self.url = url
        self.fno_symbols = list(fno_symbols or [])

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "headline": self.headline,
            "summary": self.summary,
            "source": self.source,
            "published_at": self.published_at,
            "mentioned_symbols": self.mentioned_symbols,
            "fno_symbols": self.fno_symbols,
            "url": self.url,
        }


class NewsNormalizer:
    def __init__(self) -> None:
        try:
            from news.curator import get_entity_resolver
            self._resolver = get_entity_resolver()
        except Exception:
            self._resolver = None

    def normalize(self, articles: List[RawArticle]) -> List[NormalizedArticle]:
        """Keep macro news and map company news across the full NSE universe."""
        normalized: List[NormalizedArticle] = []
        for article in articles:
            text = f"{article.headline} {article.summary}"
            mentioned: tuple[str, ...] = ()
            fno: tuple[str, ...] = ()
            if self._resolver is not None:
                try:
                    mentioned, fno = self._resolver.resolve(text)
                except Exception:
                    mentioned, fno = (), ()
            normalized.append(
                NormalizedArticle(
                    id=article.id,
                    headline=article.headline,
                    summary=article.summary[:400],
                    source=article.source,
                    published_at=article.published_at.isoformat(),
                    mentioned_symbols=list(mentioned),
                    fno_symbols=list(fno),
                    url=article.url,
                )
            )

        log.debug("news_normalized", count=len(normalized))
        return normalized
