"""
News fetcher: pulls articles from RSS feeds (configurable), optionally Marketaux,
and the durable retail news-curator cache.

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

_FETCH_TIMEOUT = 15
_MAX_AGE_HOURS = 6


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
        self._seen_ids: set[str] = set()

    def fetch_all(self, max_age_hours: int = _MAX_AGE_HOURS) -> List[RawArticle]:
        """Pull configured feeds, paid enrichment, and curator cache; deduplicate."""
        articles: List[RawArticle] = []
        for url in settings.rss_feed_list:
            try:
                new = self._fetch_rss(url, max_age_hours)
                articles.extend(new)
                log.info("rss_feed_result", feed=url.split("/")[2], articles=len(new))
            except Exception as exc:
                log.warning("rss_fetch_failed", url=url, error=str(exc))

        from datetime import date as _date
        _mx_day = str(_date.today())
        if getattr(self, "_mx_fail_day", "") != _mx_day:
            self._mx_fail_day = _mx_day
            self._mx_fails = 0
        if self._mx_fails < 3:
            try:
                from news.marketaux_news import get_marketaux_client
                mx = get_marketaux_client()
                if mx is not None:
                    mx_articles = mx.fetch_as_raw_articles(
                        symbols=settings.symbol_list, limit=10, days_back=2
                    )
                    articles.extend(mx_articles)
                    log.info("marketaux_articles_merged", count=len(mx_articles))
                    self._mx_fails = self._mx_fails + 1 if not mx_articles else 0
            except Exception as exc:
                self._mx_fails += 1
                log.warning("marketaux_fetch_failed", error=str(exc), fails_today=self._mx_fails)

        # The automated curator has much broader official/media coverage and a
        # durable deduplicated cache. This bridge enriches existing LLM/JARVIS
        # contexts without making the curator a trading authority.
        try:
            from news.curator_service import get_news_curator_service
            curated = get_news_curator_service().curator.raw_articles(
                hours=max_age_hours,
                limit=500,
            )
            articles.extend(curated)
            log.info("curated_news_context_merged", count=len(curated))
        except Exception as exc:
            log.debug("curated_news_context_unavailable", error=str(exc))

        fresh: List[RawArticle] = []
        for article in articles:
            if article.id not in self._seen_ids:
                self._seen_ids.add(article.id)
                fresh.append(article)

        if len(self._seen_ids) > 5000:
            self._seen_ids.clear()

        log.info("news_fetched", total=len(articles), fresh=len(fresh))
        return sorted(fresh, key=lambda x: x.published_at, reverse=True)

    def _fetch_rss(self, feed_url: str, max_age_hours: int) -> List[RawArticle]:
        _browser_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
        }
        try:
            import requests
            resp = requests.get(feed_url, headers=_browser_headers, timeout=10)
            feed = feedparser.parse(resp.content)
            if not feed.entries:
                log.warning(
                    "rss_feed_empty",
                    url=feed_url,
                    http_status=resp.status_code,
                    bozo=bool(getattr(feed, "bozo", False)),
                )
        except Exception:
            feed = feedparser.parse(feed_url, request_headers=_browser_headers)
        now = datetime.now(timezone.utc)
        cutoff_ts = now.timestamp() - max_age_hours * 3600
        articles: List[RawArticle] = []

        for entry in feed.entries:
            published_at = self._parse_entry_time(entry, now)
            if published_at.timestamp() < cutoff_ts:
                continue
            headline = entry.get("title", "")
            summary = entry.get("summary", entry.get("description", ""))
            summary = self._strip_tags(summary)[:600]
            source = feed.feed.get("title", feed_url)
            url = entry.get("link", "")
            if headline:
                articles.append(RawArticle(headline, summary, source, url, published_at))
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
