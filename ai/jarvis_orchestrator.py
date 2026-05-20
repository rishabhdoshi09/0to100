"""
JARVIS Orchestrator — routes queries to specialist agents and synthesises results.

Flow:
  1. Classify query → which agents are needed (or direct chat if none)
  2. Run agents in parallel (ThreadPoolExecutor)
  3. Agents publish status to AgentBus in real time
  4. Synthesise all results into a final JARVIS response
  5. Return OrchestratorResult with full activity trace
"""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ai.agent_bus import AgentBus, AgentMessage
from ai.jarvis_agents import (
    AgentResult,
    AnalysisAgent,
    CodeAgent,
    DataAgent,
    ResearchAgent,
    SystemAgent,
    _llm,
    _parse_json,
)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class OrchestratorResult:
    query: str
    answer: str
    agent_results: dict[str, AgentResult] = field(default_factory=dict)
    agent_activity: list[dict] = field(default_factory=list)   # bus log as dicts
    used_agents: list[str] = field(default_factory=list)
    routed: bool = False                                        # True = agents used
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


# ── Routing prompt ────────────────────────────────────────────────────────────

_ROUTE_PROMPT = """You are the JARVIS Orchestrator for QUANTTERM — an institutional trading system.

Available specialist agents:
- DataAgent: Live Kite quotes, historical OHLCV, option chain, open positions, index values
- ResearchAgent: Internet/web search, URL fetching, news scraping, NSE website data
- CodeAgent: Write/deploy Python code, read/write files, git commit/push/merge/branch, run shell commands
- AnalysisAgent: Market regime, setup scan, opportunity score, chart patterns, technical indicators, journal stats
- SystemAgent: Process management, application logs, disk usage, pip install, restart Streamlit

Classify this query and decide which agents (if any) are needed.
Return ONLY valid JSON:
{
  "needs_agents": true/false,
  "reason": "one-line reasoning",
  "agent_tasks": {
    "AgentName": "precise task description for this agent"
  },
  "synthesis_prompt": "how to combine agent outputs into the final answer (empty if needs_agents=false)"
}

Use agents when: live internet data needed, code changes/deployment needed, git operations,
system operations, real-time stock data, news scraping, file management.

Do NOT use agents for: regime interpretation (data already in context), setup explanations,
position sizing math, playbook advice, win rate questions (data already in context),
general trading questions that context can answer.

Query: "{query}"
Context summary: {context_summary}"""


# ── Synthesis prompt ──────────────────────────────────────────────────────────

_SYNTH_PROMPT = """You are JARVIS, the AI co-pilot for QUANTTERM — an institutional trading system.

The user asked: "{query}"

Your specialist agents have completed their work. Here are their results:
{agent_results}

Synthesis instruction: {synthesis_prompt}

Write a clear, direct, institutional-grade response. Use ₹ for prices, % for rates.
Include concrete numbers. No fluff. Max 400 words."""


# ── Direct chat prompt ────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are JARVIS, the AI co-pilot for QUANTTERM — an institutional trading system.

Personality: terse, confident, institutional. Like a senior prop trader.
- Never hedge excessively. Give a clear view.
- Use numbers: ₹ for prices, R-multiples for trades, % for rates.
- No fluff. No disclaimers.
- Think in probability and expectancy.

Start your first session response with a 2-sentence market brief."""


# ── Orchestrator ──────────────────────────────────────────────────────────────

class JarvisOrchestrator:
    """Main JARVIS brain — classifies queries, spawns agents, synthesises."""

    def __init__(self) -> None:
        self.bus = AgentBus()
        self._agents = {
            "DataAgent":     DataAgent(self.bus),
            "ResearchAgent": ResearchAgent(self.bus),
            "CodeAgent":     CodeAgent(self.bus),
            "AnalysisAgent": AnalysisAgent(self.bus),
            "SystemAgent":   SystemAgent(self.bus),
        }

    def _route(self, query: str, context: str) -> dict:
        prompt = _ROUTE_PROMPT.format(
            query=query,
            context_summary=context[:600],
        )
        raw = _llm([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=400)
        plan = _parse_json(raw)
        # Validate agent names
        valid = set(self._agents.keys())
        agent_tasks = {k: v for k, v in plan.get("agent_tasks", {}).items() if k in valid}
        return {
            "needs_agents": bool(plan.get("needs_agents") and agent_tasks),
            "reason": plan.get("reason", ""),
            "agent_tasks": agent_tasks,
            "synthesis_prompt": plan.get("synthesis_prompt", ""),
        }

    def _synthesise(
        self,
        query: str,
        agent_results: dict[str, AgentResult],
        synthesis_prompt: str,
    ) -> str:
        parts = []
        for name, res in agent_results.items():
            status = "✓" if res.success else "✗"
            parts.append(f"[{status} {name}]\n{res.result}")
        results_text = "\n\n".join(parts)

        prompt = _SYNTH_PROMPT.format(
            query=query,
            agent_results=results_text[:3000],
            synthesis_prompt=synthesis_prompt or "Synthesise all agent results into a clear answer.",
        )
        return _llm(
            [{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600,
        )

    def run(
        self,
        query: str,
        context: str,
        history: list[dict] | None = None,
    ) -> OrchestratorResult:
        self.bus.clear()

        # Route
        plan = self._route(query, context)

        if not plan["needs_agents"]:
            # Direct chat — no agents needed
            messages = [{"role": "system", "content": _SYSTEM_PROMPT + "\n\n" + context}]
            for m in (history or [])[-10:]:
                messages.append({"role": m["role"], "content": m["content"]})
            messages.append({"role": "user", "content": query})
            answer = _llm(messages, temperature=0.2, max_tokens=600)
            return OrchestratorResult(
                query=query,
                answer=answer,
                routed=False,
                agent_activity=[],
            )

        # Spawn agents in parallel
        self.bus.publish(AgentMessage(
            from_agent="ORCHESTRATOR",
            to_agent="ALL",
            msg_type="STATUS",
            content=f"Routing to: {', '.join(plan['agent_tasks'].keys())}",
        ))

        agent_results: dict[str, AgentResult] = {}
        with ThreadPoolExecutor(max_workers=len(plan["agent_tasks"])) as ex:
            futures = {
                ex.submit(self._agents[name].run, task, context): name
                for name, task in plan["agent_tasks"].items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    agent_results[name] = future.result()
                except Exception as exc:
                    agent_results[name] = AgentResult(
                        agent_name=name,
                        task=plan["agent_tasks"][name],
                        result=f"Agent crashed: {exc}",
                        success=False,
                    )

        # Synthesise
        answer = self._synthesise(query, agent_results, plan.get("synthesis_prompt", ""))

        activity = [
            {
                "from": m.from_agent,
                "to": m.to_agent,
                "type": m.msg_type,
                "content": m.content,
                "ts": m.timestamp,
            }
            for m in self.bus.get_log()
        ]

        return OrchestratorResult(
            query=query,
            answer=answer,
            agent_results=agent_results,
            agent_activity=activity,
            used_agents=list(agent_results.keys()),
            routed=True,
        )


# ── Module-level singleton ────────────────────────────────────────────────────

_orchestrator: JarvisOrchestrator | None = None


def get_orchestrator() -> JarvisOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = JarvisOrchestrator()
    return _orchestrator
