"""
News fetcher: pulls articles from RSS feeds (configurable) and
optionally a REST news API endpoint.

⚠️  News is context for the LLM — NOT ground truth for trade decisions.
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from typing import List

import feedparser

from config import settings
from logger import get_logger

log = get_logger(__name__)

_FETCH_TIMEOUT = 15  # seconds per RSS request
_MAX_AGE_HOURS = 6   # ignore articles older than this


class RawArticle:
    __slots__ = ("id", "headline", "summary", "source", "url", "published_at")

    def __init__(
        self,
        headline: str,
        summary: str,
        source: str,
        url: str,
        published_at: datetime,
    ) -> None:
        self.headline = headline.strip()
        self.summary = (summary or "").strip()
        self.source = source
        self.url = url
        self.published_at = published_at
        # Stable dedup key — based on headline text
        self.id = hashlib.sha1(headline.lower().encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "headline": self.headline,
            "summary": self.summary,
            "source": self.source,
            "url": self.url,
            "published_at": self.published_at.isoformat(),
        }


class NewsFetcher:
    def __init__(self) -> None:
        self._seen_ids: set[str] = set()   # dedup across cycles

    def fetch_all(self, max_age_hours: int = _MAX_AGE_HOURS) -> List[RawArticle]:
        """Pull news from RSS feeds + Marketaux (if key is set). Returns deduplicated list."""
        articles: List[RawArticle] = []
        for url in settings.rss_feed_list:
            try:
                new = self._fetch_rss(url, max_age_hours)
                articles.extend(new)
            except Exception as exc:
                log.warning("rss_fetch_failed", url=url, error=str(exc))

        # Marketaux enrichment — one API call for the full universe
        try:
            from news.marketaux_news import get_marketaux_client
            mx = get_marketaux_client()
            if mx is not None:
                mx_articles = mx.fetch_as_raw_articles(
                    symbols=settings.symbol_list, limit=10, days_back=2
                )
                articles.extend(mx_articles)
                log.info("marketaux_articles_merged", count=len(mx_articles))
        except Exception as exc:
            log.warning("marketaux_fetch_failed", error=str(exc))

        # Deduplicate
        fresh: List[RawArticle] = []
        for a in articles:
            if a.id not in self._seen_ids:
                self._seen_ids.add(a.id)
                fresh.append(a)

        # Cap seen_ids memory growth
        if len(self._seen_ids) > 5000:
            self._seen_ids.clear()

        log.info("news_fetched", total=len(articles), fresh=len(fresh))
        return sorted(fresh, key=lambda x: x.published_at, reverse=True)

    def _fetch_rss(self, feed_url: str, max_age_hours: int) -> List[RawArticle]:
        feed = feedparser.parse(feed_url, request_headers={"User-Agent": "SimpleQuantAI/1.0"})
        now = datetime.now(timezone.utc)
        cutoff_ts = now.timestamp() - max_age_hours * 3600
        articles: List[RawArticle] = []

        for entry in feed.entries:
            published_at = self._parse_entry_time(entry, now)
            if published_at.timestamp() < cutoff_ts:
                continue
            headline = entry.get("title", "")
            summary = entry.get("summary", entry.get("description", ""))
            # Strip HTML tags naively (no external dep)
            summary = self._strip_tags(summary)[:600]
            source = feed.feed.get("title", feed_url)
            url = entry.get("link", "")
            if headline:
                articles.append(
                    RawArticle(headline, summary, source, url, published_at)
                )
        return articles

    @staticmethod
    def _parse_entry_time(entry, fallback: datetime) -> datetime:
        try:
            struct = entry.get("published_parsed") or entry.get("updated_parsed")
            if struct:
                ts = time.mktime(struct)
                return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            pass
        return fallback

    @staticmethod
    def _strip_tags(text: str) -> str:
        import re
        return re.sub(r"<[^>]+>", " ", text).strip()
