"""
DualChatEngine — conversational dual-LLM for DevBloom Co-Pilot and analytics.

Pipeline (mirrors the signal pipeline but for free-form chat):
  1. DeepSeek generates a first-pass analysis — fast, data-driven quant reasoning.
  2. Claude reviews DeepSeek's answer AND the original query, then produces the
     final response: agrees, enriches, or overrides.
  3. Returns (final_text, decision_maker) where decision_maker is one of:
       "deepseek"         — Claude unavailable; DeepSeek answer used as-is.
       "claude_validated" — Claude agreed with and refined DeepSeek's view.
       "claude_override"  — Claude materially disagreed and replaced the answer.

Graceful degradation:
  - DeepSeek down → Claude only (single-LLM mode, labelled "claude_solo").
  - Both down      → offline message.
  - Any timeout    → returns what's available.
"""

from __future__ import annotations

import os
from typing import Generator, Iterator

_DS_URL  = "https://api.deepseek.com/v1/chat/completions"
_DS_MODEL = "deepseek-chat"

_DEEPSEEK_SYSTEM = """You are a senior quant analyst specialising in Indian equity markets (NSE/BSE).
Give a thorough, data-driven first-pass analysis. Use bullet points where helpful.
Be concrete — cite indicator values, price levels, and sector context when given.
This is Stage 1 of a two-stage review; a senior analyst (Claude) will validate your output."""

_CLAUDE_REVIEW_SYSTEM = """You are the senior analyst in a two-stage AI review pipeline for DevBloom Terminal.

Your role:
1. Read the user's original question.
2. Read the Stage-1 analysis from DeepSeek (your junior analyst).
3. If DeepSeek is correct and thorough → refine and present the final answer cleanly.
4. If DeepSeek made errors, missed key risks, or gave weak reasoning → override with your own analysis.
5. Always label your response with one of:
   [VALIDATED] — you agree with DeepSeek's core view (add any refinements below)
   [OVERRIDE]  — you disagree on a material point (state why, give your view)

Keep responses concise and actionable. Use bullet points for multi-part answers.
Market context: Indian equities (NSE/BSE), INR-denominated, IST timezone."""


def _deepseek_chat(prompt: str, system: str = _DEEPSEEK_SYSTEM, max_tokens: int = 600) -> str | None:
    """Call DeepSeek chat API (free-form, no JSON enforcement)."""
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        return None
    try:
        import requests
        resp = requests.post(
            _DS_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": _DS_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens":  max_tokens,
            },
            timeout=25,
        )
        return resp.json()["choices"][0]["message"]["content"]
    except Exception:
        return None


def _claude_stream(messages: list[dict], system: str) -> Iterator[str]:
    """Stream Claude response tokens; yields empty iterator if unavailable."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        with client.messages.stream(
            model="claude-haiku-4-5-20251001",
            max_tokens=900,
            system=system,
            messages=messages,
        ) as stream:
            yield from stream.text_stream
    except Exception:
        return


def _claude_sync(messages: list[dict], system: str, max_tokens: int = 700) -> str | None:
    """Synchronous (non-streaming) Claude call."""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return resp.content[0].text if resp.content else None
    except Exception:
        return None


class DualChatEngine:
    """
    Conversational dual-LLM engine for the DevBloom Co-Pilot.

    Usage (streaming):
        engine = DualChatEngine()
        for chunk, decision_maker in engine.stream(user_msg, history, context):
            ui.write(chunk)

    Usage (sync, e.g. bias detector):
        text, decision_maker = engine.ask(prompt, context)
    """

    def stream(
        self,
        user_message: str,
        chat_history: list[dict] | None = None,
        context: dict | None = None,
    ) -> Generator[tuple[str, str], None, None]:
        """
        Yields (text_chunk, decision_maker) tuples for streaming UI.
        decision_maker is set once at the end of each stage.
        """
        history = chat_history or []
        ctx     = context or {}

        enriched = self._enrich(user_message, ctx)

        # ── Stage 1: DeepSeek first pass ─────────────────────────────────────
        ds_prompt = self._build_ds_prompt(enriched, history)
        ds_answer = _deepseek_chat(ds_prompt)

        has_deepseek = bool(ds_answer)
        has_claude   = bool(os.getenv("ANTHROPIC_API_KEY", ""))

        if not has_deepseek and not has_claude:
            yield ("⚠️ Both LLMs unavailable — check DEEPSEEK_API_KEY and ANTHROPIC_API_KEY in `.env`.", "offline")
            return

        if not has_deepseek:
            # Fallback: Claude solo
            messages = self._build_claude_messages(enriched, history, ds_answer=None)
            full = ""
            for chunk in _claude_stream(messages, _CLAUDE_REVIEW_SYSTEM.replace("Stage-1 analysis from DeepSeek", "question")):
                full += chunk
                yield (chunk, "claude_solo")
            return

        if not has_claude:
            # DeepSeek only
            yield (ds_answer or "", "deepseek")
            return

        # ── Stage 2: Claude validates/overrides ───────────────────────────────
        messages = self._build_claude_messages(enriched, history, ds_answer)
        full = ""
        decision_maker = "claude_validated"
        for chunk in _claude_stream(messages, _CLAUDE_REVIEW_SYSTEM):
            full += chunk
            # Detect override after enough text has arrived
            if "[OVERRIDE]" in full:
                decision_maker = "claude_override"
            yield (chunk, decision_maker)

    def ask(
        self,
        prompt: str,
        context: dict | None = None,
        max_tokens: int = 700,
    ) -> tuple[str, str]:
        """
        Synchronous dual-LLM call. Returns (response_text, decision_maker).
        """
        ctx     = context or {}
        enriched = self._enrich(prompt, ctx)

        ds_answer = _deepseek_chat(enriched, max_tokens=max_tokens)

        if not ds_answer and not os.getenv("ANTHROPIC_API_KEY", ""):
            return ("Both LLMs unavailable.", "offline")

        if not ds_answer:
            result = _claude_sync([{"role": "user", "content": enriched}],
                                  _CLAUDE_REVIEW_SYSTEM, max_tokens)
            return (result or "No response.", "claude_solo")

        messages = self._build_claude_messages(enriched, [], ds_answer)
        claude_result = _claude_sync(messages, _CLAUDE_REVIEW_SYSTEM, max_tokens)

        if not claude_result:
            return (ds_answer, "deepseek")

        decision_maker = "claude_override" if "[OVERRIDE]" in claude_result else "claude_validated"
        return (claude_result, decision_maker)

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _enrich(user_message: str, ctx: dict) -> str:
        parts = [user_message]
        if ctx.get("symbol"):
            parts.append(f"\n[Active symbol: {ctx['symbol']}]")
        if ctx.get("price"):
            parts.append(f"[Last price: ₹{ctx['price']:,.2f}]")
        if ctx.get("indicators"):
            ind = ctx["indicators"]
            parts.append(
                f"[Indicators — RSI: {ind.get('rsi_14', '—')}, "
                f"Z-score: {ind.get('zscore_20', '—')}, "
                f"Momentum: {ind.get('momentum_5d_pct', '—')}%]"
            )
        return "".join(parts)

    @staticmethod
    def _build_ds_prompt(user_message: str, history: list[dict]) -> str:
        recent = "\n".join(
            f"{m['role'].upper()}: {m['content'][:200]}"
            for m in history[-6:]
        )
        if recent:
            return f"Conversation context:\n{recent}\n\nLatest question: {user_message}"
        return user_message

    @staticmethod
    def _build_claude_messages(
        user_message: str,
        history: list[dict],
        ds_answer: str | None,
    ) -> list[dict]:
        messages: list[dict] = []
        for m in history[-8:]:
            messages.append({"role": m["role"], "content": m["content"][:400]})

        if ds_answer:
            content = (
                f"**User question:** {user_message}\n\n"
                f"**DeepSeek Stage-1 analysis:**\n{ds_answer}\n\n"
                "Review the above and produce the final answer."
            )
        else:
            content = user_message

        messages.append({"role": "user", "content": content})
        return messages


_DECISION_MAKER_BADGE = {
    "claude_validated": ("✅ DeepSeek → Claude", "#00d4ff"),
    "claude_override":  ("⚡ Claude Override",   "#ffb800"),
    "deepseek":         ("🔵 DeepSeek",          "#8892a4"),
    "claude_solo":      ("🟣 Claude",            "#a050ff"),
    "offline":          ("🔴 Offline",           "#ff4466"),
}


def decision_badge_html(decision_maker: str) -> str:
    label, color = _DECISION_MAKER_BADGE.get(decision_maker, ("?", "#8892a4"))
    return (
        f"<span style='font-size:.65rem;color:{color};font-weight:600;"
        f"font-family:JetBrains Mono,monospace;background:rgba(255,255,255,.05);"
        f"padding:.15rem .4rem;border-radius:4px;border:1px solid {color}44'>{label}</span>"
    )
