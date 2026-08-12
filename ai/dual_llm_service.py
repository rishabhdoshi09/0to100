"""
DualLLMService — single access point for all DeepSeek interactions in DevBloom.

Every AI surface (Co-Pilot, Charts, Decision Terminal, order confirmation)
routes through this class. It wraps:

  · DualChatEngine  — free-form chat (V3 streaming, optional R1 validation)
  · DualLLMEngine   — structured trading signals (V3 fast pass, R1 validation)

Pure DeepSeek — no Claude dependency.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Generator

from llm.dual_chat import DualChatEngine, _DECISION_MAKER_BADGE  # noqa: F401

_log = logging.getLogger("devbloom.svc")

_SHARED_HISTORY_KEY = "devbloom_chat"
_MAX_HISTORY = 40  # messages kept in session (user + assistant pairs)
_LATENCY_WARN_S = 10.0  # log a warning if any stage exceeds this


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
        service.badge(decision_maker)                        # colour chip only
        service.badge(decision_maker, detail=detail)         # chip + tooltip
        service.badge(decision_maker, detail, ts=True)       # chip + tooltip + timestamp
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
            "ts": datetime.now().strftime("%H:%M"),
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
        Streams DeepSeek V3 tokens directly for instant UI feedback.
        detail is always "" (no override tooltip needed in V3-only mode).
        """
        history = self.get_history()
        prior = [m for m in history if m["role"] in ("user", "assistant")]

        t0 = time.perf_counter()
        for chunk, dm in self._chat.stream(user_message, prior, context):
            yield (chunk, dm, "")
        _log.info("stream_complete elapsed=%.2fs", time.perf_counter() - t0)

    # ── Sync chat ─────────────────────────────────────────────────────────────

    def ask(
        self,
        prompt: str,
        context: dict | None = None,
        max_tokens: int = 700,
        use_r1: bool = False,
    ) -> tuple[str, str, str]:
        """Returns (text, decision_maker, detail). detail always ""."""
        t0 = time.perf_counter()
        text, dm = self._chat.ask(prompt, context, max_tokens, use_r1=use_r1)
        _log.info("dual_ask dm=%s elapsed=%.2fs", dm, time.perf_counter() - t0)
        return text, dm, ""

    # ── Structured signal ─────────────────────────────────────────────────────

    def signal(self, context_prompt: str, symbol: str):
        """Returns a TradingSignal (see llm/signal_validator.py)."""
        if self._signal_engine is None:
            from llm.dual_engine import DualLLMEngine
            self._signal_engine = DualLLMEngine()
        t0 = time.perf_counter()
        sig = self._signal_engine.get_signal(context_prompt, symbol)
        _log.info("signal symbol=%s dm=%s elapsed=%.2fs", symbol, sig.llm_decision_maker, time.perf_counter() - t0)
        return sig

    # ── Badge HTML ────────────────────────────────────────────────────────────

    @staticmethod
    def badge(decision_maker: str, detail: str = "", ts: str = "") -> str:
        """
        Returns an HTML badge span.
        - If `detail` is set and decision is claude_override, adds hover tooltip
          showing what DeepSeek originally said.
        - If `ts` is set (HH:MM string), appends a timestamp after the label.
        """
        label, color = _DECISION_MAKER_BADGE.get(decision_maker, ("🔵 DeepSeek V3", "#8892a4"))
        title_attr = ""
        ts_html = (
            f"<span style='font-size:.55rem;color:{color}88;margin-left:.3rem;font-family:JetBrains Mono,monospace'>{ts}</span>"
            if ts else ""
        )
        return (
            f"<span style='font-size:.65rem;color:{color};font-weight:600;"
            f"font-family:JetBrains Mono,monospace;background:rgba(255,255,255,.05);"
            f"padding:.15rem .4rem;border-radius:4px;border:1px solid {color}44;"
            f"cursor:default'{title_attr}>{label}</span>{ts_html}"
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
