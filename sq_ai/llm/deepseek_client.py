"""
DeepSeek LLM client.

Supports two models (set DEEPSEEK_MODEL in .env):
  deepseek-reasoner  (R1) — chain-of-thought reasoning. Best for trading decisions.
                            Internally thinks before answering; reasoning_content is
                            logged for audit. No temperature param, no JSON mode.
  deepseek-chat      (V3) — faster, supports JSON mode. Good for high-freq cycles.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from openai import OpenAI

from config import settings
from logger import get_logger

log = get_logger(__name__)

_SYSTEM_PROMPT = """You are a quantitative trading signal generator for Indian equities (NSE).

Your role:
- Analyze the provided market data, technical indicators, and news context.
- Reason carefully about risk, momentum, and macro context before deciding.
- Generate a single trading signal as a strict JSON object.
- Output ONLY the JSON. No text before or after it.

Output format (EXACT — no deviation):
{
  "symbol": "<NSE symbol>",
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": <float 0.0-1.0>,
  "time_horizon": "intraday" | "swing" | "positional",
  "position_size": <float, fraction of capital 0.0-1.0>,
  "reasoning": "<2-3 sentence explanation integrating technicals + news>",
  "risk_level": "low" | "medium" | "high"
}

CRITICAL RULES:
1. Output ONLY the JSON object. Nothing before or after.
2. If confidence < 0.60, set action to HOLD.
3. Never fabricate data. Use only what is provided.
4. News is context only — not confirmation of a trade.
5. If information is insufficient, output action=HOLD, confidence=0.50.
6. Be conservative. Missing a trade is better than a wrong trade.
"""


class DeepSeekClient:
    def __init__(self) -> None:
        self._client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
        self._model = settings.deepseek_model
        self._is_reasoner = "reasoner" in self._model.lower()
        log.info("deepseek_client_init", model=self._model, reasoner=self._is_reasoner)

    def get_signal(self, context_prompt: str) -> Optional[Dict[str, Any]]:
        """
        Send context to DeepSeek and return parsed JSON signal.
        Returns None if model output is invalid or unparseable.
        """
        try:
            kwargs: Dict[str, Any] = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": context_prompt},
                ],
                "max_tokens": 8000 if self._is_reasoner else 1024,
            }

            if self._is_reasoner:
                # R1: no temperature/response_format — reasoning fills tokens first,
                # needs large max_tokens so JSON output is not truncated
                pass
            else:
                # V3: deterministic + enforce JSON output
                kwargs["temperature"] = 0.1
                kwargs["response_format"] = {"type": "json_object"}

            response = self._client.chat.completions.create(**kwargs)
            message = response.choices[0].message

            # R1 exposes chain-of-thought in reasoning_content — log for audit trail
            reasoning = getattr(message, "reasoning_content", None)
            if reasoning:
                log.debug(
                    "llm_chain_of_thought",
                    chars=len(reasoning),
                    preview=reasoning[:300],
                )

            raw_text = message.content or ""
            log.debug("llm_raw_response", model=self._model, text=raw_text[:300])
            return self._parse_json(raw_text)

        except Exception as exc:
            log.error("llm_call_failed", error=str(exc))
            return None

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse the JSON object from model output."""
        text = text.strip()

        # Strip markdown code fences
        if "```" in text:
            text = re.sub(r"```(?:json)?", "", text).strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fall back: extract first { ... } block (handles prose around JSON)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        log.warning("llm_json_parse_failed", text=text[:300])
        return None
