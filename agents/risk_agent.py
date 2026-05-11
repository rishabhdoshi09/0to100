"""Risk assessment agent – uses DeepSeek R1 (reasoning) for deeper analysis."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from agents.prompts import RISK_AGENT_PROMPT
from agents.tools import get_portfolio_state, get_technical_indicators
from ai.deepseek_dual import DeepSeekDual


class RiskAgent:
    def __init__(self) -> None:
        self.llm = DeepSeekDual()

    def assess(self, symbol: str, proposed_trade: Optional[Dict] = None) -> Dict[str, Any]:
        indicators = get_technical_indicators(symbol)
        portfolio = get_portfolio_state()

        # Keep only the most risk-relevant indicators to reduce token count
        _risk_keys = ("last_close", "atr_14", "atr_pct", "rsi_14", "vol_20d_ann", "volume_ratio")
        risk_indicators = {k: indicators.get(k) for k in _risk_keys}

        context = {
            "symbol": symbol,
            "proposed_trade": proposed_trade or {},
            "risk_indicators": risk_indicators,
            "portfolio": portfolio,
        }
        prompt = (
            f"{RISK_AGENT_PROMPT}\n\n"
            f"Trade context:\n{json.dumps(context, indent=2)}\n\n"
            "Return your assessment as a JSON object."
        )
        result = self.llm.structured_response(prompt, reasoning=True)
        result.setdefault("symbol", symbol)
        return result
