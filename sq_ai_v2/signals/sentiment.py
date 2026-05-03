"""
Sentiment module.

Uses FinBERT (ProsusAI/finbert) to score news headlines / summaries.
Score: P(positive) - P(negative) ∈ [-1, 1].
Aggregated over a symbol's recent news to produce a sentiment signal ∈ [0, 1].

Falls back to 0.5 (neutral) when no news or model unavailable.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
import requests
from loguru import logger

from config.settings import settings


class SentimentAnalyser:
    """
    Wraps FinBERT for financial news sentiment scoring.
    Lazy-loads the model on first use.
    """

    def __init__(self) -> None:
        self._pipeline = None
        self._model_loaded = False

    def _load_model(self) -> None:
        if self._model_loaded:
            return
        try:
            from transformers import pipeline as hf_pipeline
            logger.info(f"Loading FinBERT ({settings.hf_model_name})...")
            self._pipeline = hf_pipeline(
                "text-classification",
                model=settings.hf_model_name,
                top_k=None,
                device=-1,              # CPU; change to 0 for GPU
                cache_dir=settings.hf_cache_dir,
                truncation=True,
                max_length=512,
            )
            self._model_loaded = True
            logger.info("FinBERT loaded")
        except Exception as exc:
            logger.warning(f"FinBERT unavailable: {exc} — using neutral fallback")
            self._pipeline = None
            self._model_loaded = True

    # ── Core scoring ──────────────────────────────────────────────────────

    def score_text(self, text: str) -> float:
        """
        Returns sentiment score ∈ [-1, 1].
          +1 = very positive, 0 = neutral, -1 = very negative.
        """
        self._load_model()
        if self._pipeline is None or not text.strip():
            return 0.0

        try:
            results = self._pipeline(text[:512])  # FinBERT max 512 tokens
            label_scores = {r["label"].lower(): r["score"] for r in results[0]}
            pos = label_scores.get("positive", 0.0)
            neg = label_scores.get("negative", 0.0)
            return float(pos - neg)
        except Exception as exc:
            logger.debug(f"Sentiment scoring failed: {exc}")
            return 0.0

    def score_batch(self, texts: List[str]) -> List[float]:
        if not texts:
            return []
        self._load_model()
        if self._pipeline is None:
            return [0.0] * len(texts)
        try:
            cleaned = [t[:512] for t in texts if t.strip()]
            results = self._pipeline(cleaned, batch_size=8)
            scores = []
            for res in results:
                label_scores = {r["label"].lower(): r["score"] for r in res}
                scores.append(label_scores.get("positive", 0.0) - label_scores.get("negative", 0.0))
            return scores
        except Exception as exc:
            logger.debug(f"Batch scoring failed: {exc}")
            return [0.0] * len(texts)

    # ── Aggregate to signal [0, 1] ────────────────────────────────────────

    def aggregate_signal(
        self,
        texts: List[str],
        decay: float = 0.95,
    ) -> float:
        """
        Aggregate list of recent news texts (most recent first) to a
        probability-like signal ∈ [0, 1].
        Applies exponential decay so older articles have less weight.
        """
        if not texts:
            return 0.5

        scores = self.score_batch(texts[:20])  # cap at 20 articles

        # Exponential decay weights (most recent = weight 1.0)
        weights = np.array([decay**i for i in range(len(scores))])
        weights /= weights.sum()

        weighted_score = float(np.dot(scores, weights))

        # Map [-1, 1] → [0, 1]
        return float(np.clip((weighted_score + 1) / 2, 0.0, 1.0))

    # ── News fetching ─────────────────────────────────────────────────────

    def fetch_newsapi(self, symbol: str, n: int = 10) -> List[str]:
        """Fetch news from NewsAPI for a symbol. Returns list of headlines."""
        if not settings.newsapi_key:
            return []
        try:
            r = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": symbol,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": n,
                    "apiKey": settings.newsapi_key,
                },
                timeout=10,
            )
            data = r.json()
            return [
                f"{a.get('title', '')}. {a.get('description', '')}".strip()
                for a in data.get("articles", [])
            ]
        except Exception as exc:
            logger.debug(f"NewsAPI fetch failed for {symbol}: {exc}")
            return []

    def fetch_finnhub(self, symbol: str, n: int = 10) -> List[str]:
        """Fetch news from Finnhub for a symbol."""
        if not settings.finnhub_key:
            return []
        try:
            r = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params={
                    "symbol": symbol,
                    "from": "2024-01-01",
                    "to": "2099-12-31",
                    "token": settings.finnhub_key,
                },
                timeout=10,
            )
            articles = r.json()
            return [a.get("headline", "") for a in articles[:n]]
        except Exception as exc:
            logger.debug(f"Finnhub fetch failed for {symbol}: {exc}")
            return []

    def get_sentiment_signal(self, symbol: str) -> float:
        """
        Fetch news from available sources, score, and return signal ∈ [0, 1].
        """
        texts: List[str] = []
        texts.extend(self.fetch_newsapi(symbol, n=10))
        texts.extend(self.fetch_finnhub(symbol, n=10))

        if not texts:
            logger.debug(f"No news found for {symbol}; returning neutral 0.5")
            return 0.5

        signal = self.aggregate_signal(texts)
        logger.debug(f"Sentiment signal for {symbol}: {signal:.3f} ({len(texts)} articles)")
        return signal
