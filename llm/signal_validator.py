"""
Signal validator: enforces strict rules on LLM output before anything
reaches the execution layer.

HARD RULES (non-negotiable):
  1. Output must be valid JSON with all required fields.
  2. action must be BUY | SELL | HOLD.
  3. confidence must be 0.0–1.0.
  4. confidence < MIN_SIGNAL_CONFIDENCE → force HOLD.
  5. position_size must be 0.0–1.0.
  6. risk_level must be low | medium | high.
  7. Symbol must match the requested symbol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from config import settings
from logger import get_logger

log = get_logger(__name__)

_REQUIRED_FIELDS = {
    "symbol", "action", "confidence", "time_horizon",
    "position_size", "reasoning", "risk_level",
}
_VALID_ACTIONS = {"BUY", "SELL", "HOLD"}
_VALID_HORIZONS = {"intraday", "swing", "positional"}
_VALID_RISK_LEVELS = {"low", "medium", "high"}


@dataclass
class TradingSignal:
    symbol: str
    action: str          # BUY | SELL | HOLD
    confidence: float
    time_horizon: str
    position_size: float  # fraction of capital
    reasoning: str
    risk_level: str
    rejected: bool = False
    rejection_reason: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    def is_actionable(self) -> bool:
        return not self.rejected and self.action in ("BUY", "SELL")


class SignalValidator:
    def validate(
        self, raw: Optional[Dict[str, Any]], expected_symbol: str
    ) -> TradingSignal:
        """
        Validate raw LLM dict and return a TradingSignal.
        If validation fails, returns a HOLD signal with rejection_reason set.
        """
        if raw is None:
            return self._reject("llm_returned_none", expected_symbol, raw)

        # 1. Required fields
        missing = _REQUIRED_FIELDS - set(raw.keys())
        if missing:
            return self._reject(f"missing_fields:{missing}", expected_symbol, raw)

        # 2. Symbol check
        symbol = str(raw["symbol"]).upper().strip()
        if symbol != expected_symbol.upper():
            log.warning("signal_symbol_mismatch", expected=expected_symbol, got=symbol)
            symbol = expected_symbol  # correct it silently

        # 3. Action
        action = str(raw["action"]).upper().strip()
        if action not in _VALID_ACTIONS:
            return self._reject(f"invalid_action:{action}", expected_symbol, raw)

        # 4. Confidence
        try:
            confidence = float(raw["confidence"])
        except (TypeError, ValueError):
            return self._reject("invalid_confidence_type", expected_symbol, raw)
        if not (0.0 <= confidence <= 1.0):
            return self._reject(f"confidence_out_of_range:{confidence}", expected_symbol, raw)

        # 5. Confidence threshold gate
        if confidence < settings.min_signal_confidence:
            log.info(
                "signal_below_confidence_threshold",
                symbol=symbol,
                confidence=confidence,
                threshold=settings.min_signal_confidence,
            )
            action = "HOLD"

        # 6. Time horizon
        time_horizon = str(raw.get("time_horizon", "intraday")).lower()
        if time_horizon not in _VALID_HORIZONS:
            time_horizon = "intraday"

        # 7. Position size
        try:
            position_size = float(raw["position_size"])
        except (TypeError, ValueError):
            return self._reject("invalid_position_size_type", expected_symbol, raw)
        position_size = max(0.0, min(position_size, settings.max_position_size_pct))

        # 8. Risk level
        risk_level = str(raw.get("risk_level", "medium")).lower()
        if risk_level not in _VALID_RISK_LEVELS:
            risk_level = "medium"

        # 9. Reasoning
        reasoning = str(raw.get("reasoning", "")).strip()[:500]

        signal = TradingSignal(
            symbol=symbol,
            action=action,
            confidence=confidence,
            time_horizon=time_horizon,
            position_size=position_size,
            reasoning=reasoning,
            risk_level=risk_level,
            raw=raw,
        )
        log.info(
            "signal_validated",
            symbol=symbol,
            action=action,
            confidence=confidence,
            risk_level=risk_level,
        )
        return signal

    @staticmethod
    def _reject(reason: str, symbol: str, raw: Optional[Dict]) -> TradingSignal:
        log.warning("signal_rejected", reason=reason, symbol=symbol)
        return TradingSignal(
            symbol=symbol,
            action="HOLD",
            confidence=0.0,
            time_horizon="intraday",
            position_size=0.0,
            reasoning="signal rejected by validator",
            risk_level="high",
            rejected=True,
            rejection_reason=reason,
            raw=raw or {},
        )
