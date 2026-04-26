"""
DeepSeek LLM client.

Uses the OpenAI-compatible API exposed by DeepSeek.
The client is STRICTLY read-only from the trading perspective:
  - It receives a structured context packet.
  - It returns a structured JSON signal.
  - It NEVER calls any execution API.

Hard rules enforced here:
  - All prompts demand JSON-only output.
  - Non-JSON responses are rejected.
  - Responses missing required fields are rejected.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from openai import OpenAI

from config import settings
from logger import get_logger

log = get_logger(__name__)

_SYSTEM_PROMPT = """You are a quantitative trading signal generator for Indian equities.

Your role:
- Analyze the provided market data, technical indicators, and news context.
- Generate a single trading signal in strict JSON format.
- Do NOT output any text outside the JSON object.
- Do NOT explain your reasoning outside the JSON.

Output format (EXACT — no deviation allowed):
{
  "symbol": "<NSE symbol>",
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": <float 0.0-1.0>,
  "time_horizon": "intraday" | "swing" | "positional",
  "position_size": <float, fraction of capital 0.0-1.0>,
  "reasoning": "<1-2 sentence explanation>",
  "risk_level": "low" | "medium" | "high"
}

CRITICAL RULES:
1. Output ONLY the JSON object. Nothing before or after.
2. confidence < 0.60 means you are uncertain — set action to HOLD.
3. Never fabricate data. Use only what is given.
4. News is context, not confirmation. Be conservative.
5. If information is insufficient, set action=HOLD, confidence=0.50.
"""


class DeepSeekClient:
    def __init__(self) -> None:
        self._client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
        self._model = settings.deepseek_model

    def get_signal(self, context_prompt: str) -> Optional[Dict[str, Any]]:
        """
        Send context to DeepSeek and return parsed JSON signal.
        Returns None if the model output is invalid.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": context_prompt},
                ],
                temperature=0.1,        # low temperature → deterministic
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            raw_text = response.choices[0].message.content or ""
            log.debug("llm_raw_response", text=raw_text[:300])
            return self._parse_json(raw_text)
        except Exception as exc:
            log.error("llm_call_failed", error=str(exc))
            return None

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse the JSON object from model output."""
        text = text.strip()
        # Some models wrap in markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(
                l for l in lines if not l.startswith("```")
            ).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            log.warning("llm_json_parse_failed", error=str(exc), text=text[:200])
            return None
