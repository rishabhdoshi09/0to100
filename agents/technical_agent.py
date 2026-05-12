"""Technical analysis agent – uses DeepSeek V3 (fast)."""

from __future__ import annotations

import json
from typing import Any, Dict

from agents.prompts import TECHNICAL_AGENT_PROMPT
from agents.tools import get_technical_indicators
from llm.deepseek_client import DeepSeekDual


class TechnicalAgent:
    def __init__(self) -> None:
        self.llm = DeepSeekDual()

    def analyze(self, symbol: str) -> Dict[str, Any]:
        indicators = get_technical_indicators(symbol)
        prompt = (
            f"{TECHNICAL_AGENT_PROMPT}\n\n"
            f"Indicator snapshot for {symbol}:\n"
            f"{json.dumps(indicators, indent=2)}\n\n"
            "Return your analysis as a JSON object."
        )
        result = self.llm.structured_response(prompt, reasoning=False)
        result.setdefault("symbol", symbol)
        return result
