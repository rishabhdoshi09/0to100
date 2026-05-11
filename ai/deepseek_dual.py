"""
Dual DeepSeek client – uses V3 (deepseek-chat) for fast tasks and
R1 (deepseek-reasoner) for complex multi-step reasoning.

This is a new, additive file. The existing dual_llm_service.py is untouched.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI


class DeepSeekDual:
    """Thin wrapper around the DeepSeek OpenAI-compatible API."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        self.model_fast = "deepseek-chat"       # V3 – fast, JSON-mode capable
        self.model_reason = "deepseek-reasoner"  # R1 – chain-of-thought

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        reasoning: bool = False,
    ) -> str:
        """Send a chat request and return the response text."""
        model = self.model_reason if reasoning else self.model_fast
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": 8000 if reasoning else 2048,
        }
        if not reasoning:
            kwargs["temperature"] = temperature

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def structured_response(
        self,
        prompt: str,
        system: Optional[str] = None,
        reasoning: bool = False,
    ) -> Dict[str, Any]:
        """Ask for JSON output, return parsed dict (or {"raw": ...} on failure)."""
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = self.chat(messages, reasoning=reasoning)

        # Strip markdown code fences
        if "```" in resp:
            resp = re.sub(r"```(?:json)?", "", resp).strip().rstrip("`").strip()

        try:
            return json.loads(resp)
        except json.JSONDecodeError:
            pass

        # Extract first {...} block
        start = resp.find("{")
        end = resp.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(resp[start:end])
            except json.JSONDecodeError:
                pass

        return {"raw": resp}
