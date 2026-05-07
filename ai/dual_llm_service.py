"""
DualLLMService — single access point for all dual-LLM interactions in DevBloom.

Every AI surface (Co-Pilot, sidebar, Charts, Decision Terminal, bias detector,
order confirmation) routes through this class. It wraps both:

  · DualChatEngine  — free-form conversational queries (streaming + sync)
  · DualLLMEngine   — structured trading signals (returns TradingSignal)

Benefits:
  - One place to tune models, timeouts, fallback behaviour
  - Badges and disagreement details generated consistently everywhere
  - Chat history shared across UI surfaces via st.session_state["devbloom_chat"]
"""
from __future__ import annotations

import os
from typing import Generator

from llm.dual_chat import DualChatEngine, decision_badge_html, _DECISION_MAKER_BADGE  # noqa: F401


_SHARED_HISTORY_KEY = "devbloom_chat"
_MAX_HISTORY = 40  # messages kept in session (user + assistant pairs)


class DualLLMService:
    """
    Singleton-friendly wrapper used by all DevBloom UI modules.

    Streaming chat:
        for chunk, dm, detail in service.stream(msg, context):
            ...

    Sync chat (bias detector, one-off analysis):
        text, dm, detail = service.ask(prompt, context)

    Structured signal (Decision Terminal, Charts expander):
        signal = service.signal(context_prompt, symbol)
        signal.action / signal.confidence / signal.reasoning / signal.llm_decision_maker

    Badge HTML (consistent across all call sites):
        service.badge(decision_maker)                   # colour chip only
        service.badge(decision_maker, detail=detail)    # chip + tooltip with disagreement detail
    """

    def __init__(self):
        self._chat = DualChatEngine()
        self._signal_engine = None  # lazy-loaded to avoid import cost at startup

    # ── Chat history (shared across sidebar + main tab) ───────────────────────

    @staticmethod
    def get_history() -> list[dict]:
        import streamlit as st
        if _SHARED_HISTORY_KEY not in st.session_state:
            st.session_state[_SHARED_HISTORY_KEY] = []
        return st.session_state[_SHARED_HISTORY_KEY]

    @staticmethod
    def append_user(content: str):
        import streamlit as st
        history = DualLLMService.get_history()
        history.append({"role": "user", "content": content})
        st.session_state[_SHARED_HISTORY_KEY] = history[-_MAX_HISTORY * 2:]

    @staticmethod
    def append_assistant(content: str, decision_maker: str, detail: str = ""):
        import streamlit as st
        history = DualLLMService.get_history()
        history.append({
            "role": "assistant",
            "content": content,
            "decision_maker": decision_maker,
            "detail": detail,
        })
        st.session_state[_SHARED_HISTORY_KEY] = history[-_MAX_HISTORY * 2:]

    @staticmethod
    def clear_history():
        import streamlit as st
        st.session_state[_SHARED_HISTORY_KEY] = []

    # ── Streaming chat ────────────────────────────────────────────────────────

    def stream(
        self,
        user_message: str,
        context: dict | None = None,
    ) -> Generator[tuple[str, str, str], None, None]:
        """
        Yields (text_chunk, decision_maker, detail) triples.
        `detail` carries the DeepSeek draft for override tooltips — populated
        once Claude has labelled its response [OVERRIDE].
        """
        history = self.get_history()
        prior = [m for m in history if m["role"] in ("user", "assistant")]

        ds_draft = ""
        full = ""
        dm = "claude_validated"
        override_detail = ""

        # Pull DeepSeek draft out of the engine internals so we can surface it
        # in the override tooltip. We monkey-patch nothing — instead we pass
        # a sentinel context key that the engine already ignores gracefully.
        from llm.dual_chat import _deepseek_chat, _claude_stream, _CLAUDE_REVIEW_SYSTEM
        from llm.dual_chat import DualChatEngine as _Engine

        enriched = _Engine._enrich(user_message, context or {})
        ds_prompt = _Engine._build_ds_prompt(enriched, prior[:-1] if prior else [])
        ds_draft = _deepseek_chat(ds_prompt) or ""

        has_claude = bool(os.getenv("ANTHROPIC_API_KEY", ""))

        if not ds_draft and not has_claude:
            yield ("⚠️ Both LLMs unavailable — check DEEPSEEK_API_KEY and ANTHROPIC_API_KEY in `.env`.", "offline", "")
            return

        if not ds_draft:
            messages = _Engine._build_claude_messages(enriched, prior, ds_answer=None)
            for chunk in _claude_stream(messages, _CLAUDE_REVIEW_SYSTEM.replace("Stage-1 analysis from DeepSeek", "question")):
                full += chunk
                yield (chunk, "claude_solo", "")
            return

        if not has_claude:
            yield (ds_draft, "deepseek", "")
            return

        messages = _Engine._build_claude_messages(enriched, prior, ds_draft)
        for chunk in _claude_stream(messages, _CLAUDE_REVIEW_SYSTEM):
            full += chunk
            if "[OVERRIDE]" in full and not override_detail:
                dm = "claude_override"
                override_detail = ds_draft[:400]  # what Claude disagreed with
            yield (chunk, dm, override_detail)

    # ── Sync chat ─────────────────────────────────────────────────────────────

    def ask(
        self,
        prompt: str,
        context: dict | None = None,
        max_tokens: int = 700,
    ) -> tuple[str, str, str]:
        """Returns (text, decision_maker, detail)."""
        text, dm = self._chat.ask(prompt, context, max_tokens)
        detail = ""
        # For override, re-run DeepSeek to surface the disagreement detail
        # (bias detector, one-shot analysis — latency not critical here)
        if dm == "claude_override":
            from llm.dual_chat import _deepseek_chat
            ds = _deepseek_chat(prompt, max_tokens=max_tokens // 2)
            detail = (ds or "")[:400]
        return text, dm, detail

    # ── Structured signal ─────────────────────────────────────────────────────

    def signal(self, context_prompt: str, symbol: str):
        """Returns a TradingSignal (see llm/signal_validator.py)."""
        if self._signal_engine is None:
            from llm.dual_engine import DualLLMEngine
            self._signal_engine = DualLLMEngine()
        return self._signal_engine.get_signal(context_prompt, symbol)

    # ── Badge HTML ────────────────────────────────────────────────────────────

    @staticmethod
    def badge(decision_maker: str, detail: str = "") -> str:
        """
        Returns an HTML badge span.
        If `detail` is provided and the decision is claude_override, the badge
        gains a title tooltip showing the DeepSeek draft Claude disagreed with.
        """
        label, color = _DECISION_MAKER_BADGE.get(decision_maker, ("?", "#8892a4"))
        title_attr = ""
        if detail and decision_maker == "claude_override":
            # Escape quotes for HTML attribute
            escaped = detail.replace('"', "&quot;").replace("'", "&#39;")
            title_attr = f' title="DeepSeek said: {escaped}"'
        return (
            f"<span style='font-size:.65rem;color:{color};font-weight:600;"
            f"font-family:JetBrains Mono,monospace;background:rgba(255,255,255,.05);"
            f"padding:.15rem .4rem;border-radius:4px;border:1px solid {color}44;"
            f"cursor:default'{title_attr}>{label}</span>"
        )

    # ── Order confirmation check ──────────────────────────────────────────────

    def order_confirmation(
        self,
        symbol: str,
        action: str,
        entry: float,
        stop: float,
        target: float,
        qty: int,
        indicators: dict | None = None,
    ) -> tuple[str, str, str]:
        """
        Quick dual-LLM sanity check before placing a paper order.
        Returns (verdict, decision_maker, detail).
        verdict is one of: "GO", "CAUTION", "ABORT"
        """
        ind = indicators or {}
        rr = abs(target - entry) / abs(entry - stop) if abs(entry - stop) > 0 else 0
        prompt = (
            f"Trade confirmation check for {symbol}:\n"
            f"Action: {action} | Entry: ₹{entry:.2f} | Stop: ₹{stop:.2f} | "
            f"Target: ₹{target:.2f} | Qty: {qty} | R:R: {rr:.2f}\n"
            f"RSI: {ind.get('rsi_14', 'N/A')} | Momentum: {ind.get('momentum_5d_pct', 'N/A')}%\n\n"
            "Respond with exactly one of: GO / CAUTION / ABORT\n"
            "Then one sentence explaining why. Be direct and brief."
        )
        text, dm, detail = self.ask(prompt, max_tokens=150)
        verdict = "GO"
        upper = text.upper()
        if "ABORT" in upper:
            verdict = "ABORT"
        elif "CAUTION" in upper:
            verdict = "CAUTION"
        return verdict, dm, detail


# Module-level singleton so all UI modules share one instance
_service: DualLLMService | None = None


def get_service() -> DualLLMService:
    global _service
    if _service is None:
        _service = DualLLMService()
    return _service
