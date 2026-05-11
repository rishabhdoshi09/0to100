"""Sentiment analysis agent – uses DeepSeek V3 (fast)."""

from __future__ import annotations

import json
from typing import Any, Dict

from agents.prompts import SENTIMENT_AGENT_PROMPT
from agents.tools import get_recent_news
from ai.deepseek_dual import DeepSeekDual


class SentimentAgent:
    def __init__(self) -> None:
        self.llm = DeepSeekDual()

    def analyze(self, symbol: str) -> Dict[str, Any]:
        news = get_recent_news(symbol)
        news_text = json.dumps(news, indent=2) if news else "No recent news found."
        prompt = (
            f"{SENTIMENT_AGENT_PROMPT}\n\n"
            f"Recent news for {symbol}:\n{news_text}\n\n"
            "Return your analysis as a JSON object."
        )
        result = self.llm.structured_response(prompt, reasoning=False)
        result.setdefault("symbol", symbol)
        return result
