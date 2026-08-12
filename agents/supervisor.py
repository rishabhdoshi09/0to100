"""
Agent Supervisor – orchestrates Technical, Sentiment, and Risk agents,
then issues a final BUY/SELL/HOLD decision using DeepSeek R1.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

from agents.prompts import SUPERVISOR_PROMPT
from agents.risk_agent import RiskAgent
from agents.sentiment_agent import SentimentAgent
from agents.technical_agent import TechnicalAgent
from llm.deepseek_client import DeepSeekDual


class AgentSupervisor:
    def __init__(self) -> None:
        self.technical = TechnicalAgent()
        self.sentiment = SentimentAgent()
        self.risk = RiskAgent()
        self.llm = DeepSeekDual()

    def evaluate_stock(
        self,
        symbol: str,
        proposed_trade: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Run all three specialist agents in parallel, then use R1 to synthesise
        a final trading decision.  Returns a dict with DECISION, CONFIDENCE,
        REASONING, RISK_OVERRIDE, agent_reports, and symbol.
        """
        agent_reports: Dict[str, Any] = {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self.technical.analyze, symbol): "technical",
                executor.submit(self.sentiment.analyze, symbol): "sentiment",
                executor.submit(self.risk.assess, symbol, proposed_trade): "risk",
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    agent_reports[key] = future.result()
                except Exception as exc:
                    agent_reports[key] = {"error": str(exc), "symbol": symbol}

        combined = {
            **agent_reports,
            "user_query": f"Should I BUY, SELL, or HOLD {symbol}?",
        }
        final_prompt = (
            f"{SUPERVISOR_PROMPT}\n\n"
            f"Agent reports for {symbol}:\n"
            f"{json.dumps(combined, indent=2)}\n\n"
            "Return your final decision as a JSON object."
        )
        decision = self.llm.structured_response(final_prompt, reasoning=True)
        decision["agent_reports"] = agent_reports
        decision["symbol"] = symbol
        return decision
