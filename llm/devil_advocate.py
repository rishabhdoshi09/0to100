"""
Devil's Advocate — DeepSeek R1 challenge layer.

After DeepSeek V3 generates a trading signal, R1 is given ONLY the bear case:
"Give me 5 reasons this trade will FAIL." If R1's counter-arguments are weak
or generic, the signal survives. If R1 raises a specific, data-backed concern,
confidence is penalised.

Returns a ChallengeResult with:
  - survived: bool
  - penalty: float (0.0 = no penalty, 1.0 = full reject)
  - concerns: list of specific concerns raised
  - reasoning: R1's full chain-of-thought summary
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from llm.deepseek_client import DeepSeekDual
from logger import get_logger

log = get_logger(__name__)

_DEVIL_SYSTEM = """You are a rigorous risk analyst and devil's advocate for a trading desk.
A junior analyst has proposed a trade. Your ONLY job is to find reasons it will FAIL.

Be specific — reference the actual data provided. Generic platitudes like
"markets are unpredictable" do not count. You must cite specific numbers,
patterns, or conditions from the context.

Respond in strict JSON:
{
  "concerns": ["<specific concern 1>", "<specific concern 2>", ...],
  "strongest_concern": "<the single most dangerous issue>",
  "override_confidence": <float 0.0-1.0, how confident are YOU that the trade fails>,
  "summary": "<2-sentence synthesis>"
}

override_confidence guide:
  0.0-0.2 = weak challenge (signal should proceed)
  0.3-0.5 = moderate concern (reduce confidence)
  0.6-0.8 = strong concern (significantly reduce confidence)
  0.9-1.0 = trade should be rejected
"""


@dataclass
class ChallengeResult:
    survived: bool
    penalty: float           # multiply original confidence by (1 - penalty)
    concerns: list[str] = field(default_factory=list)
    strongest_concern: str = ""
    override_confidence: float = 0.0
    summary: str = ""
    raw: str = ""


def challenge_signal(
    symbol: str,
    action: str,
    confidence: float,
    context: Dict[str, Any],
    reasoning: str = "",
) -> ChallengeResult:
    """
    Run R1 devil's advocate on a proposed signal.

    Args:
        symbol: NSE ticker
        action: BUY | SELL | HOLD
        confidence: original V3 confidence (0-1)
        context: dict with indicators, news, fundamentals
        reasoning: V3's original reasoning text

    Returns:
        ChallengeResult with penalty to apply to confidence
    """
    client = DeepSeekDual()

    indicators = context.get("indicators", {})
    news_summary = context.get("news_summary", "No news context.")
    fundamentals = context.get("fundamentals", {})

    prompt = f"""A trade has been proposed. Challenge it aggressively.

PROPOSED TRADE:
  Symbol: {symbol}
  Action: {action}
  Confidence: {confidence:.0%}
  Original reasoning: {reasoning or "Not provided"}

MARKET DATA:
  RSI-14: {indicators.get("rsi_14", "N/A")}
  MACD signal: {indicators.get("macd_signal", "N/A")}
  ATR: {indicators.get("atr_14", "N/A")}
  Volume ratio vs avg: {indicators.get("volume_ratio", "N/A")}
  Price vs 20 SMA: {indicators.get("price_vs_sma20_pct", "N/A")}%
  Momentum 5d: {indicators.get("momentum_5d_pct", "N/A")}%

FUNDAMENTAL CONTEXT:
  PE ratio: {fundamentals.get("pe", "N/A")}
  Promoter holding: {fundamentals.get("promoter_pct", "N/A")}%
  Debt/Equity: {fundamentals.get("debt_equity", "N/A")}

NEWS CONTEXT:
{news_summary}

Find specific, data-backed reasons this {action} trade on {symbol} will FAIL.
"""

    try:
        result = client.structured_response(prompt, system=_DEVIL_SYSTEM, reasoning=True)

        override_conf = float(result.get("override_confidence", 0.0))
        concerns = result.get("concerns", [])
        strongest = result.get("strongest_concern", "")
        summary = result.get("summary", "")

        # Penalty mapping: R1's override confidence → confidence reduction
        # 0.0-0.25 → 0% penalty (weak challenge)
        # 0.25-0.5 → 10-25% penalty
        # 0.5-0.75 → 25-50% penalty
        # 0.75+    → reject (survived=False)
        if override_conf < 0.25:
            penalty = 0.0
            survived = True
        elif override_conf < 0.5:
            penalty = override_conf * 0.5       # max ~25%
            survived = True
        elif override_conf < 0.75:
            penalty = override_conf * 0.7       # max ~52%
            survived = True
        else:
            penalty = 1.0
            survived = False

        log.info(
            "devil_advocate_result",
            symbol=symbol,
            action=action,
            override_conf=override_conf,
            penalty=penalty,
            survived=survived,
            strongest=strongest[:80] if strongest else "",
        )

        return ChallengeResult(
            survived=survived,
            penalty=penalty,
            concerns=concerns,
            strongest_concern=strongest,
            override_confidence=override_conf,
            summary=summary,
            raw=str(result),
        )

    except Exception as exc:
        log.warning("devil_advocate_failed", symbol=symbol, error=str(exc))
        # On failure: don't penalise (graceful degradation)
        return ChallengeResult(survived=True, penalty=0.0, summary=f"Challenge skipped: {exc}")
