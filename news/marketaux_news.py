"""
Marketaux news integration — fetches AI-tagged, sentiment-scored stock news.

Produces RawArticle objects compatible with the existing NewsFetcher pipeline,
so Marketaux results flow through NewsNormalizer → NewsSummarizer → LLM context
exactly like RSS articles.

Free tier: 100 requests/day. Use fetch_for_symbols() with comma-separated
symbols to fetch many tickers in a single request (up to 25 per call).
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests

from news.fetcher import RawArticle
from logger import get_logger

log = get_logger(__name__)

_BASE_URL = "https://api.marketaux.com/v1/news/all"
_DEFAULT_DAYS_BACK = 7
_REQUEST_TIMEOUT = 10


class MarketauxNews:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("MARKETAUX_API_KEY", "")
        if not self.api_key:
            raise ValueError("MARKETAUX_API_KEY not set. Add it to .env.")

    # ── Core fetch ────────────────────────────────────────────────────────────

    def fetch_news(
        self,
        tickers: str | List[str],
        limit: int = 10,
        days_back: int = _DEFAULT_DAYS_BACK,
        sentiment_filter: str | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch news articles from Marketaux.

        Returns a list of raw dicts (title, description, url, source,
        published_at, sentiment_score, sentiment_label, tickers).
        Raises on HTTP error. Returns [] on empty result.
        """
        symbols = ",".join(tickers) if isinstance(tickers, list) else tickers
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=days_back)

        params: Dict[str, Any] = {
            "api_token": self.api_key,
            "symbols": symbols,
            "limit": min(limit, 50),
            "published_after": start_date.isoformat(),
            "published_before": end_date.isoformat(),
            "sort": "published_at",
            "sort_order": "desc",
            "language": "en",
        }
        if sentiment_filter:
            params["filter_entities_sentiment"] = sentiment_filter

        resp = requests.get(_BASE_URL, params=params, timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        articles = []
        for item in data.get("data", []):
            # Entity-level sentiment (per-ticker, more precise)
            entities = item.get("entities", [])
            sym_entity = next(
                (e for e in entities if e.get("symbol", "").upper() in symbols.upper()),
                entities[0] if entities else {},
            )
            articles.append({
                "title":           item.get("title", ""),
                "description":     item.get("description", ""),
                "url":             item.get("url", ""),
                "source":          item.get("source", "marketaux"),
                "published_at":    item.get("published_at", ""),
                "sentiment_score": sym_entity.get("sentiment_score"),
                "sentiment_label": sym_entity.get("sentiment", "neutral"),
                "tickers":         [e.get("symbol") for e in entities if e.get("symbol")],
            })

        log.info("marketaux_fetched", symbols=symbols, count=len(articles))
        return articles

    def fetch_for_symbols(
        self,
        symbols: List[str],
        limit: int = 10,
        days_back: int = _DEFAULT_DAYS_BACK,
    ) -> List[Dict[str, Any]]:
        """Fetch news for up to 25 symbols in a single API call (rate-limit friendly)."""
        chunks = [symbols[i:i + 25] for i in range(0, len(symbols), 25)]
        all_articles: List[Dict[str, Any]] = []
        for chunk in chunks:
            try:
                all_articles.extend(self.fetch_news(chunk, limit=limit, days_back=days_back))
            except Exception as exc:
                log.warning("marketaux_chunk_failed", symbols=chunk, error=str(exc))
        return all_articles

    # ── Pipeline integration ──────────────────────────────────────────────────

    def fetch_as_raw_articles(
        self,
        symbols: List[str],
        limit: int = 10,
        days_back: int = _DEFAULT_DAYS_BACK,
    ) -> List[RawArticle]:
        """
        Fetch Marketaux news and return RawArticle objects compatible with
        NewsFetcher output — can be merged directly into the trading engine pipeline.
        """
        raw = self.fetch_for_symbols(symbols, limit=limit, days_back=days_back)
        articles: List[RawArticle] = []
        for item in raw:
            if not item.get("title"):
                continue
            try:
                pub = datetime.fromisoformat(
                    item["published_at"].replace("Z", "+00:00")
                )
            except Exception:
                pub = datetime.now(timezone.utc)

            # Embed sentiment label into summary so it flows into LLM context
            label = (item.get("sentiment_label") or "neutral").upper()
            score = item.get("sentiment_score")
            score_str = f" [{label}" + (f" {score:+.2f}]" if score is not None else "]")
            summary = (item.get("description") or "") + score_str

            a = RawArticle(
                headline=item["title"],
                summary=summary[:600],
                source=f"marketaux/{item.get('source', 'unknown')}",
                url=item.get("url", ""),
                published_at=pub,
            )
            articles.append(a)
        return articles


def get_marketaux_client() -> Optional[MarketauxNews]:
    """Return a MarketauxNews client if key is set, else None (graceful)."""
    try:
        return MarketauxNews()
    except ValueError:
        return None
