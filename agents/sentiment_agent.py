"""
Sentiment analysis agent – uses DeepSeek V3 (fast).

Pipeline:
  1. Fetch raw articles via NewsFetcher (RSS).
  2. Embed + upsert into local Qdrant via SemanticNewsIndex.
  3. Retrieve top-k semantically relevant articles (falls back to keyword match).
  4. Pre-score each article with VADER (rule-based, zero-cost).
  5. Send VADER-enriched articles to DeepSeek V3 for LLM analysis.
  6. Return structured JSON with both LLM and VADER scores for cross-validation.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from agents.prompts import SENTIMENT_AGENT_PROMPT
from llm.deepseek_client import DeepSeekDual


class SentimentAgent:
    def __init__(self) -> None:
        self.llm = DeepSeekDual()

    def analyze(self, symbol: str) -> Dict[str, Any]:
        # ── Step 1: fetch raw articles ────────────────────────────────────────
        raw_articles = []
        try:
            from news.fetcher import NewsFetcher
            raw_articles = NewsFetcher().fetch_all()
        except Exception as exc:
            pass

        # ── Step 2: semantic indexing (upsert to Qdrant) ──────────────────────
        try:
            from news.semantic_index import SemanticNewsIndex
            idx = SemanticNewsIndex()
            if raw_articles:
                idx.index(raw_articles)
            relevant = idx.search(symbol, top_k=8)
        except Exception:
            # Fall back to simple keyword search over raw articles
            sym_lower = symbol.lower()
            relevant = [
                a.to_dict() for a in raw_articles
                if sym_lower in a.headline.lower() or sym_lower in a.summary.lower()
            ][:8]
            if not relevant:
                relevant = [a.to_dict() for a in raw_articles[:5]]

        # ── Step 3: VADER pre-scoring ─────────────────────────────────────────
        avg_vader = 0.0
        try:
            from news.vader_scorer import batch_score
            relevant = batch_score(relevant)
            scores = [a.get("vader_score", 0.0) for a in relevant]
            avg_vader = round(sum(scores) / max(len(scores), 1), 4)
        except Exception:
            pass

        # ── Step 4: LLM analysis ──────────────────────────────────────────────
        news_text = json.dumps(relevant, indent=2) if relevant else "No recent news found."
        prompt = (
            f"{SENTIMENT_AGENT_PROMPT}\n\n"
            f"Independent VADER pre-score for {symbol}: {avg_vader:+.2f} "
            f"({'POSITIVE' if avg_vader > 0.05 else 'NEGATIVE' if avg_vader < -0.05 else 'NEUTRAL'})\n\n"
            f"Semantically retrieved articles for {symbol}:\n{news_text}\n\n"
            "Return your analysis as a JSON object. "
            "Note: if your overall_sentiment strongly disagrees with the VADER pre-score, "
            "explain the discrepancy in the summary field."
        )
        result = self.llm.structured_response(prompt, reasoning=False)
        result.setdefault("symbol", symbol)
        result["vader_avg"] = avg_vader
        result["article_count"] = len(relevant)
        return result
