"""
Context builder: assembles the structured prompt sent to DeepSeek.

Everything the LLM needs to make a decision — no more, no less.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from logger import get_logger

log = get_logger(__name__)

_MAX_PROMPT_CHARS = 12_000  # conservative limit for DeepSeek context


class ContextBuilder:
    def build(
        self,
        symbol: str,
        market_snapshot: Dict[str, Any],
        indicators: Dict[str, Any],
        news_block: str,
        portfolio_state: Dict[str, Any],
        risk_limits: Dict[str, Any],
    ) -> str:
        """
        Assemble the full context string to send to the LLM.

        Returns a prompt that fits within _MAX_PROMPT_CHARS.
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        market_json = json.dumps(self._clean(market_snapshot), indent=2)
        indicator_json = json.dumps(self._clean(indicators), indent=2)
        portfolio_json = json.dumps(self._clean(portfolio_state), indent=2)
        risk_json = json.dumps(self._clean(risk_limits), indent=2)

        prompt = f"""=== SIMPLEQUANT AI — DECISION REQUEST ===
Timestamp: {now}
Symbol to evaluate: {symbol}

--- MARKET SNAPSHOT ---
{market_json}

--- TECHNICAL INDICATORS ---
{indicator_json}

--- NEWS CONTEXT (use as context only, not ground truth) ---
{news_block}

--- CURRENT PORTFOLIO STATE ---
{portfolio_json}

--- RISK LIMITS (hard constraints) ---
{risk_json}

Based on the above data, generate a trading signal for {symbol}.
Output ONLY valid JSON. No text before or after the JSON object.
"""

        if len(prompt) > _MAX_PROMPT_CHARS:
            prompt = prompt[:_MAX_PROMPT_CHARS] + "\n[…prompt truncated]"
            log.warning("context_prompt_truncated", symbol=symbol, length=len(prompt))

        log.debug("context_built", symbol=symbol, chars=len(prompt))
        return prompt

    @staticmethod
    def _clean(d: Dict) -> Dict:
        """Remove None values and numpy types for JSON serialization."""
        import math
        out = {}
        for k, v in d.items():
            if v is None:
                continue
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                continue
            out[k] = v
        return out
