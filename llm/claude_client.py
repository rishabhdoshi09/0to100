"""
Claude (Anthropic) LLM client — final decision layer.

Guards gracefully when ANTHROPIC_API_KEY is unset: all methods return None
so callers can treat a missing key as a HOLD / neutral signal.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

from logger import get_logger

log = get_logger(__name__)

try:
    import anthropic as _anthropic_sdk
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False
    log.warning("anthropic_sdk_missing", hint="pip install anthropic")


class ClaudeClient:
    _MODEL = "claude-sonnet-4-6"
    _MAX_TOKENS = 1024

    _SYSTEM = """You are a senior quantitative equity analyst for Indian markets (NSE/BSE).
You receive pre-screened technical + news context and must output ONLY a JSON trading signal.

Output format (EXACT):
{
  "symbol": "<NSE symbol>",
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": <float 0.0-1.0>,
  "sentiment_score": <float -1.0 to 1.0>,
  "reasoning": "<2-3 sentences>"
}

Rules:
1. Output ONLY the JSON — no prose before or after.
2. If confidence < 0.60 set action = HOLD.
3. Use only the data provided; never fabricate prices or news.
4. sentiment_score: +1 = very bullish, -1 = very bearish, 0 = neutral.
"""

    def __init__(self) -> None:
        self._key = os.getenv("ANTHROPIC_API_KEY")
        self._client: Any = None
        if self._key and _ANTHROPIC_AVAILABLE:
            self._client = _anthropic_sdk.Anthropic(api_key=self._key)
            log.info("claude_client_ready", model=self._MODEL)
        else:
            log.warning(
                "claude_client_disabled",
                has_key=bool(self._key),
                sdk_installed=_ANTHROPIC_AVAILABLE,
            )

    @property
    def available(self) -> bool:
        return self._client is not None

    def get_signal(self, context_prompt: str) -> Optional[Dict[str, Any]]:
        """Return parsed JSON signal dict, or None if unavailable / parse error."""
        if not self.available:
            return None
        try:
            message = self._client.messages.create(
                model=self._MODEL,
                max_tokens=self._MAX_TOKENS,
                system=self._SYSTEM,
                messages=[{"role": "user", "content": context_prompt}],
            )
            raw = message.content[0].text if message.content else ""
            log.debug("claude_raw_response", chars=len(raw), preview=raw[:200])
            return self._parse_json(raw)
        except Exception as exc:
            log.error("claude_call_failed", error=str(exc))
            return None

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict[str, Any]]:
        text = text.strip()
        if "```" in text:
            text = re.sub(r"```(?:json)?", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        log.warning("claude_json_parse_failed", text=text[:300])
        return None
