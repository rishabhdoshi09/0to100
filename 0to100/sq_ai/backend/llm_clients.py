"""Low-level LLM client wrappers with a unified ``generate()`` API.

* ``ClaudeClient``   – Anthropic SDK
* ``DeepSeekClient`` – DeepSeek via OpenAI-compatible HTTP

Both clients degrade gracefully (return ``None``) when their API key is
absent or the call fails – the higher-level decision pipeline then falls
back to ML+regime.
"""
from __future__ import annotations

import os
import time
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
class ClaudeClient:
    """Anthropic Claude wrapper."""

    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(self, api_key: Optional[str] = None,
                 model: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model or os.environ.get("CLAUDE_MODEL", self.DEFAULT_MODEL)
        self._client = None
        if self.api_key and "REPLACE" not in self.api_key:
            try:
                import anthropic  # noqa: WPS433
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except Exception as exc:                       # pragma: no cover
                print(f"[ClaudeClient] init failed: {exc}")

    @property
    def available(self) -> bool:
        return self._client is not None

    def generate(self, prompt: str, max_tokens: int = 300,
                 temperature: float = 0.2,
                 system: Optional[str] = None) -> Optional[str]:
        if not self.available:
            return None
        backoff = 1.0
        for _ in range(3):
            try:
                kwargs = {
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if system:
                    kwargs["system"] = system
                resp = self._client.messages.create(**kwargs)
                return resp.content[0].text if resp.content else None
            except Exception as exc:                       # pragma: no cover
                msg = str(exc).lower()
                if any(s in msg for s in ("rate", "overloaded", "529", "timeout")):
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                print(f"[ClaudeClient.generate] {exc}")
                return None
        return None


# ─────────────────────────────────────────────────────────────────────────────
class DeepSeekClient:
    """DeepSeek via OpenAI-compatible API."""

    BASE_URL = "https://api.deepseek.com/v1"
    DEFAULT_MODEL = "deepseek-chat"

    def __init__(self, api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 base_url: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.model = model or os.environ.get("DEEPSEEK_MODEL", self.DEFAULT_MODEL)
        self.base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", self.BASE_URL)
        self._client = None
        if self.api_key and "REPLACE" not in self.api_key:
            try:
                from openai import OpenAI  # noqa: WPS433
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            except Exception as exc:                       # pragma: no cover
                print(f"[DeepSeekClient] init failed: {exc}")

    @property
    def available(self) -> bool:
        return self._client is not None

    def generate(self, prompt: str, max_tokens: int = 300,
                 temperature: float = 0.2,
                 system: Optional[str] = None) -> Optional[str]:
        if not self.available:
            return None
        backoff = 1.0
        for _ in range(3):
            try:
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": prompt})
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return resp.choices[0].message.content if resp.choices else None
            except Exception as exc:                       # pragma: no cover
                msg = str(exc).lower()
                if any(s in msg for s in ("rate", "429", "503", "timeout")):
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                print(f"[DeepSeekClient.generate] {exc}")
                return None
        return None


__all__ = ["ClaudeClient", "DeepSeekClient"]
