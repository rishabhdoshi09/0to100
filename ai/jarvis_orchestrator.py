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
- AnalysisAgent: Market regime, setup scan, opportunity score, chart patterns, technical indicators, journal stats, pre-trade checklist (volume + momentum + news + sector + pattern all-in-one for a specific stock)
- SystemAgent: Process management, application logs, disk usage, pip install, restart Streamlit

Classify this query and decide which agents (if any) are needed.
Return ONLY valid JSON:
{{
  "needs_agents": true/false,
  "reason": "one-line reasoning",
  "agent_tasks": {{
    "AgentName": "precise task description for this agent"
  }},
  "synthesis_prompt": "how to combine agent outputs into the final answer (empty if needs_agents=false)"
}}

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

Personality: terse, confident, institutional. Like a senior prop trader who's seen it all.
- Never hedge excessively. Give a clear view.
- Use numbers: ₹ for prices, R-multiples for trades, % for rates.
- No fluff. No disclaimers unless the risk is REAL.
- Think in probability and expectancy.
- You REMEMBER previous sessions — reference them when relevant.

CRITICAL RULE — PUSHBACK PROTOCOL:
If the user asks about entering a trade, buying, or selling a specific stock:
1. Check the ACTIVE TRADING RULES in context — if allow_new_trades=False, REFUSE and explain once clearly.
2. Check the user's PERSONAL EDGE data. If win rate in the current regime is below 40%,
   or expectancy is negative, lead with a direct warning using their actual numbers.
3. You NEVER approve a bad trade just because the user is excited about it.
4. If conditions are good and the trade idea is sound, confirm with specific entry/stop/target.

IMPORTANT — REPEAT QUESTION HANDLING:
- Only treat a question as "repeated" if the EXACT same question appears in the CURRENT conversation
  multiple times in a row immediately before the current message.
- Do NOT count questions from earlier topics in the conversation as repeats of a new question.
- Each new distinct question gets a fresh, complete answer — never say "I've answered this before"
  for a genuinely new question.
- Never escalate tone or lecture the user based on history from a DIFFERENT topic.

MEMORY CONTEXT:
If previous session data is included, reference it naturally: "Last time you were watching HDFCBANK —
it's now at ₹X" or "You mentioned your capital is ₹Y — sizing at 2R risk = ₹Z position."

Start your first response in a session with a 2-sentence market brief, then reference what you
remember from last time if relevant."""


# ── History trimmer ───────────────────────────────────────────────────────────

def _trim_history(history: list[dict], current_query: str, window: int = 6) -> list[dict]:
    """
    Return the most relevant recent history for the current query.

    Strategy: take the last `window` messages, but if the conversation
    recently switched topics (detected by a gap in subject keywords),
    only include messages from after the last topic switch.
    This prevents "I'll buy WIPRO?" spam history from poisoning a fresh question.
    """
    if not history:
        return []

    recent = history[-window * 2:]  # look at a wider window to find topic boundary

    # Find the last "topic boundary" — a user message that is clearly different
    # from the current query (no keyword overlap)
    current_words = set(current_query.lower().split())
    boundary_idx = 0

    for i, msg in enumerate(recent):
        if msg["role"] != "user":
            continue
        msg_words = set(msg["content"].lower().split())
        overlap = current_words & msg_words
        # If overlap is very low and this is a user message that was repeated
        # many times, treat it as a topic boundary
        if len(overlap) <= 1 and len(msg_words) > 2:
            # Count consecutive occurrences of this message
            count = sum(
                1 for m in recent
                if m["role"] == "user" and m["content"].strip().lower() == msg["content"].strip().lower()
            )
            if count >= 2:
                # This is a repeated different-topic message — mark boundary after it
                boundary_idx = i + 1

    # Return messages from boundary onwards, capped at window
    trimmed = recent[boundary_idx:]
    return trimmed[-window:]


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

        # Load persistent memory
        memory_context = ""
        try:
            from ai.jarvis_memory import get_memory
            mem = get_memory()
            memory_context = mem.build_morning_context()
            # Auto-detect and store price alert requests
            self._extract_and_store_facts(query, mem)
        except Exception:
            pass

        full_context = (memory_context + "\n\n" + context).strip() if memory_context else context

        # ── Live regime block (injected fresh on every query) ─────────────────
        live_state = self._build_live_state_block()
        if live_state:
            full_context = live_state + "\n\n" + full_context

        # Route
        plan = self._route(query, full_context)

        if not plan["needs_agents"]:
            # Direct chat — no agents needed
            messages = [{"role": "system", "content": _SYSTEM_PROMPT + "\n\n" + full_context}]
            # Send last 6 messages but only if they are topically related to the current query.
            # This prevents history from a different topic polluting the current answer.
            trimmed_history = _trim_history(history or [], query, window=6)
            for m in trimmed_history:
                messages.append({"role": m["role"], "content": m["content"]})
            messages.append({"role": "user", "content": query})
            answer = _llm(messages, temperature=0.2, max_tokens=700)
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
                ex.submit(self._agents[name].run, task, full_context): name
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


    def _build_live_state_block(self) -> str:
        """
        Build a structured live-market context block injected into every JARVIS query.
        Gives JARVIS regime, VIX, breadth, personal edge — so it never gives
        generic advice when market conditions are extreme.
        """
        lines = []
        try:
            from datetime import datetime
            lines.append(f"── LIVE MARKET STATE ({datetime.now().strftime('%H:%M')}) ──")

            # Regime
            try:
                from core.regime_engine import compute_regime
                r = compute_regime()
                regime = r.market_regime
                score  = r.regime_score
                vix    = r.vix
                breadth = r.breadth_label

                regime_advice = {
                    "TRENDING_BULL": "✅ Bull trend — full size, chase breakouts, let winners run.",
                    "EXPANSION":     "✅ Expansion — good conditions, normal sizing.",
                    "CHOPPY":        "⚠️ Choppy — reduce size 50%, be selective, quick profits.",
                    "DISTRIBUTION":  "🔴 Distribution — no new longs, protect capital.",
                    "TRENDING_BEAR": "🔴 Bear trend — cash is a position, only shorts or wait.",
                    "COMPRESSION":   "⏸ Compression — wait for breakout, no new positions.",
                }.get(regime, "Monitor conditions.")

                lines.append(f"Regime: {regime} (score {score:.0f}/100) | VIX: {vix:.1f} | Breadth: {breadth}")
                lines.append(f"Regime guidance: {regime_advice}")
            except Exception:
                pass

            # Personal edge from journal
            try:
                from analytics.auto_journal import get_personal_edge_by_archetype, get_journal_stats
                stats = get_journal_stats()
                if stats.get("total_trades", 0) >= 3:
                    lines.append(
                        f"Personal edge: {stats['total_trades']} trades | "
                        f"Win rate {stats['win_rate']:.0f}% | Avg {stats['avg_r']:+.2f}R | "
                        f"Best setup: {stats.get('best_archetype','?')} | "
                        f"Best regime: {stats.get('best_regime','?')}"
                    )
            except Exception:
                pass

            # System signal accuracy (self-improving feedback)
            try:
                from core.signal_outcome_tracker import get_accuracy_report, update_outcomes
                update_outcomes()  # refresh any pending outcomes
                acc = get_accuracy_report()
                if acc.get("total_signals", 0) >= 5:
                    lines.append(
                        f"Signal accuracy (self-tracking): {acc['overall_accuracy']:.0f}% "
                        f"({acc['wins']}W/{acc['losses']}L) | "
                        f"Best setup: {acc.get('best_archetype','?')} | "
                        f"System edge: {acc.get('system_edge', 0):+.2f}R"
                    )
            except Exception:
                pass

            # Historical analog — what did the market do last time conditions were like this?
            try:
                from core.regime_analog import find_analogs, format_analog_insight
                regime_state_dict = {
                    "market_regime": getattr(r, "market_regime", ""),
                    "volatility_regime": getattr(r, "volatility_regime", ""),
                    "breadth_label": getattr(r, "breadth_label", ""),
                    "vix": getattr(r, "vix", 0),
                }
                analogs = find_analogs(regime_state_dict, top_n=2)
                if analogs:
                    analog_text = format_analog_insight(analogs)
                    if analog_text:
                        lines.append(f"Historical analog: {analog_text[:300]}")
            except Exception:
                pass

            # Active trading rules
            try:
                from core.intelligence_hub import compute_opportunity_score
                from core.regime_engine import compute_regime
                from core.adaptive_engine import AdaptiveEngine
                opp = compute_opportunity_score(compute_regime(), [], AdaptiveEngine())
                allow = opp.total >= 30
                lines.append(
                    f"Market opportunity score: {opp.total:.0f}/100 ({opp.grade})"
                )
                if not allow:
                    lines.append("ACTIVE TRADING RULES: allow_new_trades=False — conditions too poor. Refuse new entries.")
            except Exception:
                pass

            # ── User's watchlist ──────────────────────────────────────────
            try:
                import sqlite3
                from pathlib import Path
                _wdb = Path("logs/watchlist.db")
                if _wdb.exists():
                    _conn = sqlite3.connect(_wdb, check_same_thread=False)
                    _rows = _conn.execute(
                        "SELECT symbol, notes FROM watchlist ORDER BY added_date DESC LIMIT 15"
                    ).fetchall()
                    _conn.close()
                    if _rows:
                        _wl_syms = [r[0] for r in _rows]
                        lines.append(f"User's watchlist ({len(_wl_syms)} stocks): {', '.join(_wl_syms)}")
                        _notes = [(r[0], r[1]) for r in _rows if r[1]]
                        if _notes:
                            for sym, note in _notes[:3]:
                                lines.append(f"  Watchlist note — {sym}: {note}")
            except Exception:
                pass

            # ── Open positions / holdings ─────────────────────────────────
            try:
                from portfolio.state import PortfolioState
                _ps = PortfolioState(initial_capital=1_000_000)
                _pos = _ps.get_open_positions()
                if _pos:
                    _pos_parts = []
                    for sym, qty in list(_pos.items())[:8]:
                        _pos_parts.append(f"{sym}×{qty:.0f}")
                    lines.append(f"Open positions ({len(_pos)}): {', '.join(_pos_parts)}")
                else:
                    lines.append("Open positions: None (fully in cash)")
            except Exception:
                pass

            # ── Today's top setups — from the whole-market auto-scan store ─
            try:
                from scan.auto_scan import get_results as _asr
                _results, _uni_size, _scan_ts, _ = _asr()
                if _results:
                    _s_lines = []
                    for _s in _results[:6]:
                        _sig = "+".join(_s.get("signals", [])[:2])
                        _s_lines.append(
                            f"{_s['symbol']} [{_s.get('verdict','?')}] {_sig} "
                            f"entry ₹{_s.get('entry',0):.0f} stop ₹{_s.get('stop',0):.0f} "
                            f"target ₹{_s.get('target',0):.0f}")
                    _age_min = int((datetime.now().timestamp() - _scan_ts) / 60) if _scan_ts else -1
                    lines.append(f"Today's top setups (whole market, {_uni_size} stocks, "
                                 f"scanned {_age_min} min ago): {' | '.join(_s_lines)}")
                    # Hot sectors from the same scan
                    _secs: dict = {}
                    for _s in _results:
                        if _s.get("sector"):
                            _secs[_s["sector"]] = _secs.get(_s["sector"], 0) + 1
                    if _secs:
                        _hot = ", ".join(f"{k} ({v})" for k, v in
                                         sorted(_secs.items(), key=lambda kv: -kv[1])[:3])
                        lines.append(f"Hot sectors right now: {_hot}")
                else:
                    lines.append("Today's top setups: scan not completed yet")
            except Exception:
                pass

            # ── Measured signal accuracy (walk-forward backtest) ──────────
            try:
                from scan.signal_backtest import load_report as _lbr
                _rep = _lbr()
                if _rep:
                    _best = sorted(_rep.get("signals", {}).items(),
                                   key=lambda kv: -kv[1].get("expectancy_r", 0))
                    _parts = [f"{k}: {v['win_rate']:.0f}%WR {v['expectancy_r']:+.2f}R"
                              for k, v in _best[:4] if v.get("trades", 0) >= 20]
                    if _parts:
                        lines.append(f"Backtested signal accuracy (trust these numbers): "
                                     f"{' | '.join(_parts)}")
            except Exception:
                pass

            # ── My open trades + portfolio risk (the user's ACTUAL book) ──
            try:
                from risk.position_manager import review_positions
                _mypos = review_positions()
                if _mypos:
                    _p_lines = []
                    for _p in _mypos[:6]:
                        _rp = (f"{_p['r_progress']:+.1f}R"
                               if _p.get("r_progress") is not None else "?R")
                        _p_lines.append(f"{_p['symbol']} ({_p['mode']}) {_rp}: "
                                        f"{_p['advice'][:80]}")
                    lines.append("User's open trades: " + " | ".join(_p_lines))
                from risk.portfolio_risk import portfolio_risk_report
                _prr = portfolio_risk_report()
                if _prr["n_positions"]:
                    lines.append(
                        f"Portfolio risk: {_prr['verdict']} — open risk "
                        f"{_prr['open_risk_pct']:.1f}% of capital, "
                        f"{_prr['n_positions']}/{_prr['max_positions']} positions"
                        + ("; " + "; ".join(_prr["warnings"][:2])
                           if _prr["warnings"] else ""))
            except Exception:
                pass

            # ── Options market read (NIFTY chain) ─────────────────────────
            try:
                from options.analytics import nifty_options_summary
                _opt = nifty_options_summary()
                if _opt:
                    lines.append(f"NIFTY options: {_opt['bias']} — {_opt['note']}; "
                                 f"max pain {_opt['max_pain']:,.0f}")
            except Exception:
                pass

            # ── Data freshness ─────────────────────────────────────────────
            try:
                from datetime import date as _date
                from data.bhavcopy_store import get_ohlcv as _go
                _df = _go("RELIANCE")
                if _df is not None and len(_df):
                    _last = _df.index[-1].date()
                    lines.append("Data freshness: LIVE intraday bar loaded"
                                 if _last == _date.today()
                                 else f"Data freshness: EOD close of {_last} (market data "
                                      f"is {( _date.today() - _last).days} day(s) old)")
            except Exception:
                pass

            # ── Recent news headlines ─────────────────────────────────────
            try:
                from news.fetcher import NewsFetcher
                _articles = NewsFetcher().fetch_all(max_age_hours=6)
                if _articles:
                    _headlines = [a.title for a in _articles[:5] if a.title]
                    lines.append(f"Recent news headlines: {' | '.join(_headlines)}")
            except Exception:
                pass

            # ── Research OS health (JARVIS QUERIES the research layer, never
            #    computes it) — so "which gates got less profitable?", "is our
            #    research healthy?", "what do we believe?" are answered from the
            #    Research Overview, not re-derived. ────────────────────────────
            try:
                lines.append(self._build_research_os_block())
            except Exception:
                pass

        except Exception:
            pass

        return "\n".join([ln for ln in lines if ln]) if len(lines) > 1 else ""

    def _build_research_os_block(self) -> str:
        """A compact snapshot of the Research OS mission-control read, injected so
        JARVIS answers research-health questions by READING this layer rather than
        computing anything itself. Fail-open → ''."""
        try:
            from research.research_overview import overview
        except Exception:
            return ""
        o = overview()
        rh, eh, debt = (o.get("research_health", {}), o.get("edge_health", {}),
                        o.get("research_debt", {}))
        parts = [
            f"Research OS — beliefs: {rh.get('beliefs_active', 0)} active / "
            f"{rh.get('beliefs_watch', 0)} watch / {rh.get('beliefs_retired', 0)} "
            f"retired; promoted this week: {rh.get('promoted_this_week', 0)}.",
            f"Edge health: {eh.get('durable', 0)} durable, {eh.get('decaying', 0)} "
            f"decaying, {eh.get('recovering', 0)} recovering, {eh.get('dead', 0)} dead.",
        ]
        card = o.get("gate_scorecard", [])
        too_cons = [g for g in card if g.get("verdict") == "TOO_CONSERVATIVE"]
        if too_cons:
            parts.append("Gates flagged too conservative: " + ", ".join(
                f"{g['gate']} (trend {g['trend']})" for g in too_cons[:3]) + ".")
        _debt = (debt.get("experiments_awaiting_validation", 0)
                 + debt.get("beliefs_overdue_review", 0)
                 + debt.get("drift_alerts_unresolved", 0))
        if _debt:
            parts.append(
                f"Research debt: {debt.get('experiments_awaiting_validation', 0)} "
                f"experiments awaiting validation, {debt.get('beliefs_overdue_review', 0)} "
                f"beliefs overdue, {debt.get('drift_alerts_unresolved', 0)} drift "
                f"alerts unresolved.")
        return " ".join(parts)

    def _extract_and_store_facts(self, query: str, mem) -> None:
        """Auto-detect price alerts and capital mentions from the user's message."""
        import re
        # Price alert: "alert me when X hits Y" / "set alert RELIANCE above 1400"
        alert_pattern = re.search(
            r"alert[^\w]*(?:me\s+)?(?:when\s+)?([A-Z]{2,12})\s+(?:hits?|above|below|crosses?|reaches?)\s+[₹]?([\d,]+)",
            query, re.IGNORECASE,
        )
        if alert_pattern:
            sym = alert_pattern.group(1).upper()
            price_str = alert_pattern.group(2).replace(",", "")
            try:
                price = float(price_str)
                direction = "above" if re.search(r"above|crosses|hits|reaches", query, re.IGNORECASE) else "below"
                mem.set_price_alert(sym, price, direction)
            except Exception:
                pass

        # Capital mention: "my capital is X lakh" / "I have X lakh"
        cap_pattern = re.search(
            r"(?:capital|portfolio|account)[^\d]*[₹]?([\d,]+)\s*(lakh|lac|cr|crore|k|thousand)?",
            query, re.IGNORECASE,
        )
        if cap_pattern:
            val = cap_pattern.group(1).replace(",", "")
            unit = (cap_pattern.group(2) or "").lower()
            multiplier = {"lakh": 100_000, "lac": 100_000, "cr": 10_000_000,
                          "crore": 10_000_000, "k": 1_000, "thousand": 1_000}.get(unit, 1)
            try:
                mem.remember("capital_size", f"₹{int(val) * multiplier:,}", "preference")
            except Exception:
                pass

        # Risk preference
        if re.search(r"\b(aggressive|full risk|high risk)\b", query, re.IGNORECASE):
            mem.remember("risk_preference", "aggressive", "preference")
        elif re.search(r"\b(conservative|low risk|cautious)\b", query, re.IGNORECASE):
            mem.remember("risk_preference", "conservative", "preference")

    def save_session(self, messages: list[dict]) -> None:
        """Call at end of session to compress conversation into memory."""
        try:
            from ai.jarvis_memory import get_memory
            get_memory().summarize_and_save_session(messages)
        except Exception:
            pass


# ── Module-level singleton ────────────────────────────────────────────────────

_orchestrator: JarvisOrchestrator | None = None


def get_orchestrator() -> JarvisOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = JarvisOrchestrator()
    return _orchestrator
