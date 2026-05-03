"""Anthropic Claude client – assembles the 5-min market brief, calls the
API, parses the JSON response.  Has a deterministic ML+regime fallback
when the API is unreachable or returns garbage.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

SYSTEM_PROMPT = """You are a disciplined, risk-averse Indian equities trading
assistant. You output ONLY valid JSON (no prose, no markdown fences) with this
exact schema:

{
  "decisions": [
    {
      "symbol": "<SYMBOL>",
      "action": "BUY" | "SELL" | "HOLD",
      "size_pct": <float 0..10>,
      "stop": <float price>,
      "target": <float price>,
      "confidence": <float 0..1>,
      "reasoning": "<<=240 chars>"
    }
  ],
  "portfolio_note": "<<=200 chars>"
}

Rules:
• Never BUY when regime == 0 (downtrend).
• Confidence < 0.5 ⇒ action MUST be HOLD.
• stop must respect 2*ATR; target ~3*ATR (asymmetric R:R).
• Size_pct must respect provided portfolio cash and 50% gross-exposure cap.
"""


@dataclass
class ClaudeDecision:
    symbol: str
    action: str
    size_pct: float
    stop: float
    target: float
    confidence: float
    reasoning: str

    @classmethod
    def hold(cls, symbol: str, reason: str = "fallback") -> "ClaudeDecision":
        return cls(symbol, "HOLD", 0.0, 0.0, 0.0, 0.0, reason)


class ClaudeClient:
    def __init__(self, api_key: str | None = None,
                 model: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model or os.environ.get(
            "CLAUDE_MODEL", "claude-sonnet-4-5-20250929"
        )
        self._client = None
        if self.api_key and not self.api_key.startswith("sk-ant-REPLACE"):
            try:
                import anthropic  # noqa: WPS433
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except Exception as exc:  # pragma: no cover
                print(f"[ClaudeClient] init failed: {exc}")

    @property
    def available(self) -> bool:
        return self._client is not None

    # ----------------------------------------------------------------- brief
    @staticmethod
    def build_brief(watchlist_features: list[dict[str, Any]],
                    news: list[dict[str, Any]],
                    portfolio: dict[str, Any]) -> str:
        lines: list[str] = []
        lines.append("WATCHLIST (per symbol):")
        for f in watchlist_features:
            lines.append(
                f"- {f['symbol']}: price={f.get('close', 0):.2f} "
                f"rsi={f.get('rsi', 50):.1f} z={f.get('zscore_20', 0):.2f} "
                f"atr={f.get('atr', 0):.2f} regime={f.get('regime', 1)} "
                f"ml_p_up={f.get('ml_proba_up', 0.5):.2f}"
            )
        lines.append("\nTOP NEWS:")
        for n in news[:3]:
            lines.append(f"- [{n.get('source', '')}] {n.get('title', '')}")
        lines.append("\nPORTFOLIO:")
        lines.append(
            f"cash={portfolio.get('cash', 0):.0f} "
            f"equity={portfolio.get('equity', 0):.0f} "
            f"exposure_pct={portfolio.get('exposure_pct', 0):.1f} "
            f"daily_pnl_pct={portfolio.get('daily_pnl_pct', 0):.2f} "
            f"open_positions={len(portfolio.get('positions', []))}"
        )
        return "\n".join(lines)

    # ----------------------------------------------------------------- call
    def decide(self, brief: str, max_tokens: int = 1024) -> dict[str, Any] | None:
        if not self.available:
            return None
        backoff = 1.0
        for attempt in range(3):
            try:
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": brief}],
                )
                txt = resp.content[0].text if resp.content else ""
                return self._parse_json(txt)
            except Exception as exc:  # pragma: no cover
                msg = str(exc).lower()
                if "rate" in msg or "overloaded" in msg or "529" in msg:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                print(f"[ClaudeClient] decide error: {exc}")
                return None
        return None

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        text = text.strip()
        # strip ```json … ``` fences if Claude misbehaves
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

    # ----------------------------------------------------------- fallback
    @staticmethod
    def fallback_decisions(watchlist_features: list[dict[str, Any]],
                           composite_results: dict[str, dict]) -> dict[str, Any]:
        """Pure ML+regime decision when Claude is offline."""
        decisions = []
        for f in watchlist_features:
            sym = f["symbol"]
            comp = composite_results.get(sym, {})
            direction = comp.get("direction", 0)
            conf = comp.get("confidence", 0.0) / 100.0
            regime = int(f.get("regime", 1))
            if direction == 1 and regime != 0 and conf >= 0.5:
                action = "BUY"
            elif direction == -1 and conf >= 0.5:
                action = "SELL"
            else:
                action = "HOLD"
            atr_v = f.get("atr", 0.0)
            price = f.get("close", 0.0)
            decisions.append({
                "symbol": sym,
                "action": action,
                "size_pct": min(5.0, conf * 10) if action == "BUY" else 0.0,
                "stop": price - 2 * atr_v if action == "BUY" else 0.0,
                "target": price + 3 * atr_v if action == "BUY" else 0.0,
                "confidence": conf,
                "reasoning": f"fallback ml+regime: dir={direction} regime={regime}",
            })
        return {"decisions": decisions,
                "portfolio_note": "claude offline – ml+regime fallback"}
