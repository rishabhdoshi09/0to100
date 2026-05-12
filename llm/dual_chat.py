"""
DualChatEngine — DeepSeek V3 (fast) + R1 (deep reasoning) for the Co-Pilot.

Pipeline:
  1. DeepSeek V3 (deepseek-chat) — fast, data-driven first-pass analysis.
     Streams tokens for instant UI feedback.
  2. DeepSeek R1 (deepseek-reasoner) — optional deep validation.
     Runs when use_r1=True; adds chain-of-thought reasoning.

Decision makers returned:
  "deepseek_v3"           — V3 only (default, streaming)
  "deepseek_r1_validated" — R1 agreed with V3
  "deepseek_r1_override"  — R1 produced a different answer
  "offline"               — No API key available
"""
from __future__ import annotations

import os
from typing import Generator, Iterator

from openai import OpenAI

_DS_BASE  = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
_V3_MODEL = "deepseek-chat"
_R1_MODEL = "deepseek-reasoner"

_V3_SYSTEM = """You are a senior quant analyst specialising in Indian equity markets (NSE/BSE).
Give thorough, data-driven analysis. Use bullet points where helpful.
Be concrete — cite indicator values, price levels, and sector context when given.
Keep responses focused and actionable. Market context: NSE/BSE, INR, IST timezone."""

_R1_REVIEW_SYSTEM = """You are a senior risk analyst reviewing a co-pilot's response for a trading terminal.

Your role:
1. Read the original question and the V3 analysis.
2. If correct and thorough → add [VALIDATED] and refine with any important additions.
3. If errors or missed risks → add [OVERRIDE] and provide a corrected analysis.

Be brief, specific, actionable. Mark your label at the start."""


def _api_key() -> str:
    return os.getenv("DEEPSEEK_API_KEY", "")


def _v3_stream(messages: list[dict]) -> Iterator[str]:
    """Stream DeepSeek V3 tokens."""
    key = _api_key()
    if not key:
        return
    try:
        client = OpenAI(api_key=key, base_url=_DS_BASE)
        with client.chat.completions.create(
            model=_V3_MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=900,
            stream=True,
        ) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    yield delta
    except Exception:
        return


def _v3_sync(messages: list[dict], max_tokens: int = 700) -> str | None:
    """Synchronous V3 call."""
    key = _api_key()
    if not key:
        return None
    try:
        client = OpenAI(api_key=key, base_url=_DS_BASE)
        resp = client.chat.completions.create(
            model=_V3_MODEL,
            messages=messages,
            temperature=0.5,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or None
    except Exception:
        return None


def _r1_sync(messages: list[dict], max_tokens: int = 2000) -> str | None:
    """Synchronous R1 call (no temperature, no streaming — R1 thinks first)."""
    key = _api_key()
    if not key:
        return None
    try:
        client = OpenAI(api_key=key, base_url=_DS_BASE)
        resp = client.chat.completions.create(
            model=_R1_MODEL,
            messages=messages,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or None
    except Exception:
        return None


class DualChatEngine:
    """
    Co-Pilot conversational engine.

    Streaming (for real-time UI):
        for chunk, dm in engine.stream(user_msg, history, context):
            ui.write(chunk)

    Sync (for one-shot analysis):
        text, dm = engine.ask(prompt, context, use_r1=False)

    Set use_r1=True for deep analysis — R1 adds chain-of-thought but is slower.
    """

    def stream(
        self,
        user_message: str,
        chat_history: list[dict] | None = None,
        context: dict | None = None,
    ) -> Generator[tuple[str, str], None, None]:
        """Yields (text_chunk, decision_maker) — streams V3 tokens directly."""
        if not _api_key():
            yield ("⚠️ DEEPSEEK_API_KEY not set. Add it to .env to use the Co-Pilot.", "offline")
            return

        enriched = self._enrich(user_message, context or {})
        messages  = self._build_messages(enriched, chat_history or [])

        full = ""
        for chunk in _v3_stream(messages):
            full += chunk
            yield (chunk, "deepseek_v3")

    def ask(
        self,
        prompt: str,
        context: dict | None = None,
        max_tokens: int = 700,
        use_r1: bool = False,
    ) -> tuple[str, str]:
        """
        Synchronous call. Returns (response_text, decision_maker).
        use_r1=True enables deep R1 validation (slower, better for complex analysis).
        """
        enriched = self._enrich(prompt, context or {})
        messages  = self._build_messages(enriched, [])

        v3_answer = _v3_sync(messages, max_tokens=max_tokens)

        if not v3_answer:
            return ("DeepSeek unavailable — check DEEPSEEK_API_KEY in .env.", "offline")

        if not use_r1:
            return (v3_answer, "deepseek_v3")

        # R1 validation
        r1_messages = [
            {"role": "system", "content": _R1_REVIEW_SYSTEM},
            {"role": "user",   "content":
                f"Original question: {prompt}\n\n"
                f"V3 analysis:\n{v3_answer}\n\n"
                "Validate or override. Be brief."
            },
        ]
        r1_answer = _r1_sync(r1_messages, max_tokens=max_tokens)

        if not r1_answer:
            return (v3_answer, "deepseek_v3")

        dm = "deepseek_r1_override" if "[OVERRIDE]" in r1_answer else "deepseek_r1_validated"
        return (r1_answer, dm)

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
    def _build_messages(user_message: str, history: list[dict]) -> list[dict]:
        msgs: list[dict] = [{"role": "system", "content": _V3_SYSTEM}]
        for m in history[-8:]:
            role = m.get("role", "user")
            if role in ("user", "assistant"):
                msgs.append({"role": role, "content": str(m.get("content", ""))[:400]})
        msgs.append({"role": "user", "content": user_message})
        return msgs


# ── Badge HTML ─────────────────────────────────────────────────────────────────

_DECISION_MAKER_BADGE: dict[str, tuple[str, str]] = {
    "deepseek_v3":           ("⚡ DeepSeek V3",        "#00d4ff"),
    "deepseek_r1_validated": ("✅ V3 → R1 Validated",  "#00d4a0"),
    "deepseek_r1_override":  ("🔴 R1 Override",        "#ffb800"),
    "offline":               ("🔴 Offline",            "#ff4466"),
}


def decision_badge_html(decision_maker: str) -> str:
    label, color = _DECISION_MAKER_BADGE.get(decision_maker, ("🔵 DeepSeek", "#8892a4"))
    return (
        f"<span style='font-size:.65rem;color:{color};font-weight:600;"
        f"font-family:JetBrains Mono,monospace;background:rgba(255,255,255,.05);"
        f"padding:.15rem .4rem;border-radius:4px;border:1px solid {color}44'>{label}</span>"
    )
