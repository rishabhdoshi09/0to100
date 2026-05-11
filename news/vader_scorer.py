"""
VADER sentiment scoring for financial news headlines.

Rule-based, zero-cost, no API calls, < 10ms per article.
Used as an independent pre-signal before the LLM sentiment agent runs.
"""

from __future__ import annotations

from typing import Any, Dict, List

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_sia = SentimentIntensityAnalyzer()


def score_article(headline: str, summary: str = "") -> float:
    """Return VADER compound score: -1.0 (very negative) to +1.0 (very positive)."""
    text = f"{headline}. {summary[:300]}"
    return round(_sia.polarity_scores(text)["compound"], 4)


def label(score: float) -> str:
    if score >= 0.05:
        return "POSITIVE"
    if score <= -0.05:
        return "NEGATIVE"
    return "NEUTRAL"


def batch_score(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add 'vader_score' and 'vader_label' keys to each article dict in-place.
    Returns the same list (mutated) sorted by absolute score descending.
    """
    for a in articles:
        score = score_article(a.get("headline", ""), a.get("summary", ""))
        a["vader_score"] = score
        a["vader_label"] = label(score)
    return sorted(articles, key=lambda x: abs(x.get("vader_score", 0.0)), reverse=True)
