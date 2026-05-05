"""
Dual-LLM decision engine.

Pipeline:
  1. DeepSeek (R1/V3) generates the first-pass trading signal — fast, data-driven.
  2. Claude (Sonnet) acts as senior analyst: it sees the same context PLUS
     DeepSeek's reasoning and either agrees or overrides.
  3. The final TradingSignal carries `llm_decision_maker`:
       "deepseek"         — Claude unavailable; DeepSeek signal used as-is.
       "claude_validated" — Claude agreed with DeepSeek; full structured signal kept.
       "claude_override"  — Claude disagreed; action/confidence/reasoning replaced.

If DeepSeek is down, returns a HOLD rejection signal immediately.
If Claude is down, DeepSeek signal is passed through (graceful degradation).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from llm.claude_client import ClaudeClient
from llm.deepseek_client import DeepSeekClient
from llm.signal_validator import SignalValidator, TradingSignal
from logger import get_logger

log = get_logger(__name__)

# Claude overrides when its action differs OR its confidence is this much lower
_CONFIDENCE_OVERRIDE_DELTA = 0.15


class DualLLMEngine:
    """
    Orchestrates DeepSeek → Claude two-stage signal generation.

    Usage
    -----
    engine = DualLLMEngine()
    signal = engine.get_signal(context_prompt, symbol)
    print(signal.action, signal.llm_decision_maker)
    """

    def __init__(self) -> None:
        self._deepseek = DeepSeekClient()
        self._claude = ClaudeClient()
        self._validator = SignalValidator()

    def get_signal(self, context_prompt: str, symbol: str) -> TradingSignal:
        """
        Run the full dual-LLM pipeline and return a validated TradingSignal.
        Never raises — returns a HOLD on any failure.
        """
        # ── Stage 1: DeepSeek ──────────────────────────────────────────────
        ds_raw = self._deepseek.get_signal(context_prompt)
        ds_signal = self._validator.validate(ds_raw, symbol)
        ds_signal.llm_decision_maker = "deepseek"

        if ds_signal.rejected:
            log.warning("deepseek_signal_rejected", symbol=symbol,
                        reason=ds_signal.rejection_reason)

        # ── Stage 2: Claude validation / override ──────────────────────────
        if not self._claude.available:
            log.info("claude_unavailable_using_deepseek", symbol=symbol)
            return ds_signal

        claude_prompt = self._build_claude_prompt(context_prompt, ds_raw, ds_signal)
        claude_raw = self._claude.get_signal(claude_prompt)

        if claude_raw is None:
            log.warning("claude_returned_none_using_deepseek", symbol=symbol)
            return ds_signal

        claude_action = str(claude_raw.get("action", "HOLD")).upper()
        try:
            claude_confidence = float(claude_raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            claude_confidence = 0.0

        log.info(
            "dual_llm_comparison",
            symbol=symbol,
            deepseek_action=ds_signal.action,
            deepseek_confidence=ds_signal.confidence,
            claude_action=claude_action,
            claude_confidence=claude_confidence,
        )

        # Override when Claude's view materially differs
        action_disagrees = claude_action != ds_signal.action
        confidence_drops = claude_confidence < ds_signal.confidence - _CONFIDENCE_OVERRIDE_DELTA

        if action_disagrees or confidence_drops:
            return self._build_override_signal(
                ds_raw, ds_signal, claude_raw, claude_action,
                claude_confidence, symbol,
            )

        # Claude agrees — keep DeepSeek's full structured signal
        ds_signal.llm_decision_maker = "claude_validated"
        log.info("claude_validated_deepseek_signal", symbol=symbol,
                 action=ds_signal.action, confidence=ds_signal.confidence)
        return ds_signal

    # ── Helpers ────────────────────────────────────────────────────────────

    def _build_claude_prompt(
        self,
        original_context: str,
        ds_raw: Optional[Dict[str, Any]],
        ds_signal: TradingSignal,
    ) -> str:
        ds_summary = (
            f"Action: {ds_signal.action}\n"
            f"Confidence: {ds_signal.confidence:.2f}\n"
            f"Risk Level: {ds_signal.risk_level}\n"
            f"Time Horizon: {ds_signal.time_horizon}\n"
            f"Reasoning: {ds_signal.reasoning}"
        )
        return (
            f"{original_context}\n\n"
            "---\n"
            "DEEPSEEK FIRST-PASS SIGNAL (your peer analyst's view):\n"
            f"{ds_summary}\n\n"
            "As senior analyst, review the above signal critically.\n"
            "If you agree, output the same action with similar confidence.\n"
            "If you disagree, override with your own action and lower confidence. "
            "Explain why in the reasoning field.\n"
            "Output ONLY the JSON signal."
        )

    def _build_override_signal(
        self,
        ds_raw: Optional[Dict[str, Any]],
        ds_signal: TradingSignal,
        claude_raw: Dict[str, Any],
        claude_action: str,
        claude_confidence: float,
        symbol: str,
    ) -> TradingSignal:
        claude_reasoning = str(claude_raw.get("reasoning", "")).strip()
        merged_reasoning = (
            f"[Claude Override] {claude_reasoning} "
            f"| [DeepSeek] {ds_signal.reasoning}"
        )[:500]

        # Merge: keep DeepSeek's structured fields, replace action/confidence/reasoning
        merged_raw: Dict[str, Any] = dict(ds_raw) if ds_raw else {}
        merged_raw.update({
            "symbol": symbol,
            "action": claude_action,
            "confidence": claude_confidence,
            "reasoning": merged_reasoning,
            # Keep DS risk_level/horizon unless Claude explicitly provided them
            "risk_level": claude_raw.get("risk_level", ds_signal.risk_level),
            "time_horizon": claude_raw.get("time_horizon", ds_signal.time_horizon),
            "position_size": ds_raw.get("position_size", 0.0) if ds_raw else 0.0,
        })

        override_signal = self._validator.validate(merged_raw, symbol)
        override_signal.llm_decision_maker = "claude_override"
        log.info(
            "claude_overrode_deepseek",
            symbol=symbol,
            ds_action=ds_signal.action,
            claude_action=claude_action,
            new_confidence=claude_confidence,
        )
        return override_signal
