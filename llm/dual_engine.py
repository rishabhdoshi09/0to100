"""
Dual-LLM decision engine — DeepSeek V3 + DeepSeek R1.

Pipeline:
  1. DeepSeek V3 (deepseek-chat) — fast, structured JSON signal.
  2. DeepSeek R1 (deepseek-reasoner) — chain-of-thought validation.
     R1 sees V3's signal and either confirms or overrides with deeper reasoning.

Decision makers:
  "deepseek_v3"         — R1 unavailable or agreed with V3; V3 signal used.
  "deepseek_r1_validated" — R1 confirmed V3's view.
  "deepseek_r1_override"  — R1 disagreed; action/confidence replaced.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

from openai import OpenAI

from llm.deepseek_client import DeepSeekClient
from llm.signal_validator import SignalValidator, TradingSignal
from logger import get_logger

log = get_logger(__name__)

_CONFIDENCE_OVERRIDE_DELTA = 0.15

_R1_VALIDATION_SYSTEM = """You are a senior quant risk manager reviewing a trading signal for Indian equities (NSE).

A junior analyst (DeepSeek V3) has proposed a trade. Your job:
1. Review the market data and the proposed signal critically.
2. If you agree with the signal → output the SAME action with equal or slightly adjusted confidence.
3. If you disagree → override with a different action and explain why.

Output ONLY a strict JSON object (no text before or after):
{
  "symbol": "<NSE symbol>",
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": <float 0.0-1.0>,
  "time_horizon": "intraday" | "swing" | "positional",
  "position_size": <float 0.0-1.0>,
  "reasoning": "<your validation reasoning — 2-3 sentences>",
  "risk_level": "low" | "medium" | "high"
}"""


class DualLLMEngine:
    """
    DeepSeek V3 → R1 two-stage signal generation.
    Falls back to V3-only if R1 is unavailable.
    """

    def __init__(self) -> None:
        self._v3 = DeepSeekClient()
        self._validator = SignalValidator()
        self._r1_client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )

    def get_signal(self, context_prompt: str, symbol: str) -> TradingSignal:
        """Run the full dual pipeline. Never raises — returns HOLD on failure."""

        # ── Stage 1: V3 fast signal ────────────────────────────────────────────
        v3_raw = self._v3.get_signal(context_prompt)
        v3_signal = self._validator.validate(v3_raw, symbol)
        v3_signal.llm_decision_maker = "deepseek_v3"

        if v3_signal.rejected:
            log.warning("v3_signal_rejected", symbol=symbol, reason=v3_signal.rejection_reason)

        # ── Stage 2: R1 validation ─────────────────────────────────────────────
        r1_raw = self._r1_validate(context_prompt, v3_raw, v3_signal, symbol)
        if r1_raw is None:
            return v3_signal

        r1_action = str(r1_raw.get("action", "HOLD")).upper()
        try:
            r1_confidence = float(r1_raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            r1_confidence = 0.0

        log.info("dual_llm_comparison", symbol=symbol,
                 v3_action=v3_signal.action, v3_conf=v3_signal.confidence,
                 r1_action=r1_action, r1_conf=r1_confidence)

        # Override if R1 materially disagrees
        action_differs   = r1_action != v3_signal.action
        confidence_drops = r1_confidence < v3_signal.confidence - _CONFIDENCE_OVERRIDE_DELTA

        if action_differs or confidence_drops:
            return self._build_override(v3_raw, v3_signal, r1_raw, r1_action, r1_confidence, symbol)

        v3_signal.llm_decision_maker = "deepseek_r1_validated"
        log.info("r1_validated_v3_signal", symbol=symbol,
                 action=v3_signal.action, confidence=v3_signal.confidence)
        return v3_signal

    # ── R1 call ────────────────────────────────────────────────────────────────

    def _r1_validate(
        self,
        context: str,
        v3_raw: Optional[Dict],
        v3_signal: TradingSignal,
        symbol: str,
    ) -> Optional[Dict]:
        if not os.getenv("DEEPSEEK_API_KEY"):
            return None
        try:
            v3_summary = (
                f"V3 proposed: {v3_signal.action} | confidence {v3_signal.confidence:.2f} | "
                f"risk {v3_signal.risk_level} | reasoning: {v3_signal.reasoning}"
            )
            prompt = (
                f"{context}\n\n"
                "---\n"
                f"V3 FIRST-PASS SIGNAL:\n{v3_summary}\n\n"
                "Review critically. Confirm or override. Output ONLY the JSON signal."
            )
            resp = self._r1_client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": _R1_VALIDATION_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=4000,
            )
            raw = resp.choices[0].message.content or ""
            if "```" in raw:
                raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if m:
                    return json.loads(m.group())
        except Exception as exc:
            log.warning("r1_validation_failed", symbol=symbol, error=str(exc))
        return None

    def _build_override(
        self,
        v3_raw: Optional[Dict],
        v3_signal: TradingSignal,
        r1_raw: Dict,
        r1_action: str,
        r1_confidence: float,
        symbol: str,
    ) -> TradingSignal:
        r1_reasoning = str(r1_raw.get("reasoning", "")).strip()
        merged_reasoning = (
            f"[R1 Override] {r1_reasoning} | [V3] {v3_signal.reasoning}"
        )[:500]

        merged: Dict[str, Any] = dict(v3_raw) if v3_raw else {}
        merged.update({
            "symbol": symbol,
            "action": r1_action,
            "confidence": r1_confidence,
            "reasoning": merged_reasoning,
            "risk_level":    r1_raw.get("risk_level",    v3_signal.risk_level),
            "time_horizon":  r1_raw.get("time_horizon",  v3_signal.time_horizon),
            "position_size": v3_raw.get("position_size", 0.0) if v3_raw else 0.0,
        })

        sig = self._validator.validate(merged, symbol)
        sig.llm_decision_maker = "deepseek_r1_override"
        log.info("r1_overrode_v3", symbol=symbol,
                 v3_action=v3_signal.action, r1_action=r1_action, new_conf=r1_confidence)
        return sig
