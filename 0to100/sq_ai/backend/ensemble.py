"""Ensemble veto – when Claude proposes ``size_pct > threshold`` (default 10 %),
re-prompt DeepSeek with the **same prompt** and execute only on agreement.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from sq_ai.backend.llm_clients import ClaudeClient, DeepSeekClient
from sq_ai.portfolio.tracker import PortfolioTracker


CLAUDE_DECISION_SYSTEM = (
    "You are a disciplined Indian equities trading assistant. "
    "Output ONLY valid JSON: "
    '{"action":"BUY|SELL|HOLD","size_pct":<0..10>,'
    '"stop":<float>,"target":<float>,"confidence":<0..1>,'
    '"reasoning":"<=240 chars"}'
)


def build_decision_prompt(symbol: str, features: dict, news: list[dict],
                          portfolio_snapshot: dict) -> str:
    news_lines = "\n".join(
        f"- [{n.get('source', '')}] {n.get('title', '')}" for n in news[:3]
    ) or "- (no news)"
    return (
        f"DECIDE for {symbol}\n"
        f"price={features.get('close', 0):.2f} rsi={features.get('rsi', 50):.1f} "
        f"z={features.get('zscore_20', 0):.2f} atr={features.get('atr', 0):.2f} "
        f"regime={features.get('regime', 1)} ml_p_up={features.get('ml_proba_up', 0.5):.2f}\n"
        f"NEWS:\n{news_lines}\n"
        f"PORTFOLIO: cash={portfolio_snapshot.get('cash', 0):.0f} "
        f"equity={portfolio_snapshot.get('equity', 0):.0f} "
        f"exposure_pct={portfolio_snapshot.get('exposure_pct', 0):.1f} "
        f"daily_pnl_pct={portfolio_snapshot.get('daily_pnl_pct', 0):.2f}\n"
        f"Rules: never BUY if regime=0; HOLD if confidence<0.5; "
        f"stop=2*ATR; target=3*ATR."
    )


def parse_decision(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if "action" not in obj:
        return None
    obj["action"] = str(obj.get("action", "HOLD")).upper()
    obj["size_pct"] = float(obj.get("size_pct", 0))
    obj["confidence"] = float(obj.get("confidence", 0))
    return obj


# ─────────────────────────────────────────────────────────────────────────────
class EnsembleVeto:
    def __init__(self,
                 tracker: PortfolioTracker | None = None,
                 claude: ClaudeClient | None = None,
                 deepseek: DeepSeekClient | None = None,
                 threshold_pct: float = 10.0) -> None:
        self.tracker = tracker or PortfolioTracker()
        self.claude = claude or ClaudeClient()
        self.deepseek = deepseek or DeepSeekClient()
        self.threshold_pct = float(threshold_pct)

    def maybe_veto(self, symbol: str, claude_decision: dict[str, Any],
                   prompt: str) -> dict[str, Any]:
        """Returns a dict with the *final* decision and full audit info."""
        size_pct = float(claude_decision.get("size_pct", 0))
        result = {
            "symbol": symbol,
            "claude_action": claude_decision["action"],
            "claude_confidence": float(claude_decision.get("confidence", 0)),
            "deepseek_action": None,
            "deepseek_confidence": None,
            "final_action": claude_decision["action"],
            "vetoed": False,
            "below_threshold": size_pct <= self.threshold_pct,
        }
        if result["below_threshold"]:
            return result

        ds_text = self.deepseek.generate(
            prompt, max_tokens=300, temperature=0.2,
            system=CLAUDE_DECISION_SYSTEM,
        )
        ds_decision = parse_decision(ds_text)
        if ds_decision:
            result["deepseek_action"] = ds_decision["action"]
            result["deepseek_confidence"] = float(ds_decision.get("confidence", 0))
        else:
            # If DeepSeek is unavailable / unparseable, conservative: HOLD
            result["deepseek_action"] = "UNKNOWN"
            result["final_action"] = "HOLD"
            result["vetoed"] = True

        if ds_decision and ds_decision["action"] != claude_decision["action"]:
            result["final_action"] = "HOLD"
            result["vetoed"] = True

        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        self.tracker.log_disagreement({
            "symbol": symbol,
            "claude_action": result["claude_action"],
            "deepseek_action": result["deepseek_action"],
            "claude_confidence": result["claude_confidence"],
            "deepseek_confidence": result["deepseek_confidence"] or 0.0,
            "prompt_hash": prompt_hash,
            "final_action": result["final_action"],
        })
        return result


__all__ = ["EnsembleVeto", "build_decision_prompt", "parse_decision",
           "CLAUDE_DECISION_SYSTEM"]
