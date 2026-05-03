"""MetaTrader — second-layer risk veto using DeepSeek Reasoner.

Sits between Claude's decision and the executor.  Every non-HOLD signal
is submitted here for deep validation before any order is placed.

Verdict keys:
    verdict      : "ACCEPT" | "REJECT" | "SCALE_DOWN" | "HOLD"
    size_multiplier : float 0-1  (1.0 = full size, 0 = don't trade)
    risk_flag    : "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    reason_codes : list[str]  (short machine-readable tags)
    final_confidence : float 0-1
    reasoning    : str  (one-sentence human summary)
"""
from __future__ import annotations

import json
import re
from typing import Any

from sq_ai.backend.llm_clients import DeepSeekClient

REASONER_MODEL = "deepseek-reasoner"

SYSTEM_PROMPT = """\
You are MetaTrader, a senior quantitative risk officer at a systematic hedge fund.
Your role is to validate trade signals before execution with rigorous risk assessment.

You receive:
- Symbol, proposed action, size, stop, target, confidence
- Technical indicators (regime, RSI, ADX, ATR, momentum, z-score)
- Portfolio context (cash, exposure, open positions)
- Claude's original reasoning

Your job is to apply a second-layer veto based on:
1. Risk/reward ratio (must be ≥ 2.0 for ACCEPT)
2. Position concentration (reject if single name > 15% equity)
3. Regime alignment (never BUY in regime=0)
4. Trend quality via ADX (prefer ADX > 20 for directional trades)
5. Volatility-adjusted sizing (scale down if ATR/price > 3%)
6. Correlation risk with existing positions
7. Overextension (RSI > 75 → scale down or reject BUY)
8. Maximum drawdown guard (reject if daily PnL already < -1.5%)

Respond ONLY with a JSON object. No markdown, no explanation outside JSON.
{
  "verdict": "ACCEPT|REJECT|SCALE_DOWN|HOLD",
  "size_multiplier": <float 0-1>,
  "risk_flag": "LOW|MEDIUM|HIGH|CRITICAL",
  "reason_codes": ["<tag>", ...],
  "final_confidence": <float 0-1>,
  "reasoning": "<one concise sentence>"
}
"""


def _build_prompt(
    sym: str,
    decision: dict[str, Any],
    features: dict[str, Any],
    snap: dict[str, Any],
) -> str:
    action = decision.get("action", "HOLD")
    size_pct = decision.get("size_pct", 0.0)
    stop = decision.get("stop", 0.0)
    target = decision.get("target", 0.0)
    confidence = decision.get("confidence", 0.0)
    reasoning = decision.get("reasoning", "")

    price = features.get("close", 0.0)
    rr = ((target - price) / (price - stop)) if stop and target and price and (price - stop) > 0 else 0.0

    adx_val = features.get("adx", 0.0)
    rsi_val = features.get("rsi", 50.0)
    atr_val = features.get("atr", 0.0)
    regime = features.get("regime", 1)
    momentum = features.get("momentum_5d", 0.0)
    zscore = features.get("zscore_20", 0.0)
    vol_20 = features.get("volatility_20", 0.0)

    equity = snap.get("equity", 0.0)
    cash = snap.get("cash", 0.0)
    exposure_pct = snap.get("exposure_pct", 0.0)
    daily_pnl_pct = snap.get("daily_pnl_pct", 0.0)
    n_positions = len(snap.get("positions", []))
    proposed_exposure = (size_pct / 100.0) * equity

    atr_pct = (atr_val / price * 100) if price else 0.0

    return f"""TRADE VALIDATION REQUEST
========================
Symbol      : {sym}
Action      : {action}
Size        : {size_pct:.1f}% of equity  (≈ ₹{proposed_exposure:,.0f})
Price       : ₹{price:,.2f}
Stop        : ₹{stop:,.2f}
Target      : ₹{target:,.2f}
Risk/Reward : {rr:.2f}
Confidence  : {confidence:.0%}

TECHNICALS
----------
Regime      : {regime}  (0=down 1=side 2=up)
ADX(14)     : {adx_val:.1f}
RSI(14)     : {rsi_val:.1f}
ATR(14)     : ₹{atr_val:.2f}  ({atr_pct:.2f}% of price)
Momentum 5d : {momentum:+.2%}
Z-score 20d : {zscore:+.2f}
Volatility  : {vol_20:.2%} annualised

PORTFOLIO
---------
Equity      : ₹{equity:,.0f}
Cash        : ₹{cash:,.0f}
Exposure    : {exposure_pct:.1f}%
Daily PnL   : {daily_pnl_pct:+.2f}%
Open pos.   : {n_positions}

CLAUDE REASONING
----------------
{reasoning}

Validate this trade. Return JSON only."""


class MetaTrader:
    """Risk-veto agent powered by DeepSeek Reasoner."""

    DEFAULT_VERDICT: dict[str, Any] = {
        "verdict": "ACCEPT",
        "size_multiplier": 1.0,
        "risk_flag": "LOW",
        "reason_codes": ["fallback_accept"],
        "final_confidence": 0.5,
        "reasoning": "MetaTrader unavailable — passthrough",
    }

    def __init__(self, deepseek: DeepSeekClient | None = None) -> None:
        self._client = deepseek or DeepSeekClient(model=REASONER_MODEL)
        if self._client.model != REASONER_MODEL:
            self._client.model = REASONER_MODEL

    @property
    def available(self) -> bool:
        return self._client.available

    def validate(
        self,
        sym: str,
        decision: dict[str, Any],
        features: dict[str, Any],
        snap: dict[str, Any],
    ) -> dict[str, Any]:
        """Return a MetaTrader verdict dict.

        Always returns a valid dict — never raises.
        """
        action = decision.get("action", "HOLD")
        if action == "HOLD":
            return {
                "verdict": "HOLD",
                "size_multiplier": 0.0,
                "risk_flag": "LOW",
                "reason_codes": ["hold_passthrough"],
                "final_confidence": float(decision.get("confidence", 0.0)),
                "reasoning": "Original signal is HOLD — no validation needed",
            }

        if not self.available:
            return self.DEFAULT_VERDICT.copy()

        prompt = _build_prompt(sym, decision, features, snap)
        raw = self._client.generate(
            prompt,
            max_tokens=512,
            temperature=0.0,  # deterministic for risk decisions
            system=SYSTEM_PROMPT,
        )
        return self._parse(raw)

    # ---------------------------------------------------------------- private
    def _parse(self, raw: str | None) -> dict[str, Any]:
        if not raw:
            return self.DEFAULT_VERDICT.copy()
        try:
            # strip markdown code fences if present
            text = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
            obj = json.loads(text)
            verdict = str(obj.get("verdict", "ACCEPT")).upper()
            if verdict not in ("ACCEPT", "REJECT", "SCALE_DOWN", "HOLD"):
                verdict = "ACCEPT"
            return {
                "verdict": verdict,
                "size_multiplier": float(obj.get("size_multiplier", 1.0)),
                "risk_flag": str(obj.get("risk_flag", "LOW")).upper(),
                "reason_codes": list(obj.get("reason_codes", [])),
                "final_confidence": float(obj.get("final_confidence", 0.5)),
                "reasoning": str(obj.get("reasoning", "")),
            }
        except Exception:
            return self.DEFAULT_VERDICT.copy()


__all__ = ["MetaTrader"]
