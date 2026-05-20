"""
JARVIS — Just A Rather Very Intelligent System.

True autonomous multi-agent trading co-pilot.
Spawns specialist agents (Data, Research, Code, Analysis, System),
runs them in parallel, shows live activity, and synthesises the answer.

Capabilities:
  · Internet access (web search, URL fetch, news scraping)
  · Code deployment (write files, git commit/push/merge/branch)
  · System control (logs, restart, process management, pip install)
  · Live market data via Kite Connect
  · Deep market analysis (regime, patterns, indicators)
  · Memory across conversation (session history)
"""
from __future__ import annotations

import os
from datetime import datetime

import streamlit as st


# ── Agent colour palette ──────────────────────────────────────────────────────

_AGENT_COLORS = {
    "DataAgent":      "#00d4ff",
    "ResearchAgent":  "#a78bfa",
    "CodeAgent":      "#34d399",
    "AnalysisAgent":  "#fb923c",
    "SystemAgent":    "#f87171",
    "ORCHESTRATOR":   "#fbbf24",
}
_AGENT_ICONS = {
    "DataAgent":      "📡",
    "ResearchAgent":  "🌐",
    "CodeAgent":      "⚙️",
    "AnalysisAgent":  "📊",
    "SystemAgent":    "🖥️",
    "ORCHESTRATOR":   "🧠",
}
_MSG_TYPE_COLORS = {
    "STATUS":  "#8892a4",
    "LOG":     "#4a5568",
    "RESULT":  "#00d4a0",
    "REQUEST": "#f59e0b",
}


# ── Context builder ───────────────────────────────────────────────────────────

def _get_context() -> str:
    try:
        from core.intelligence_hub import build_jarvis_context, compute_opportunity_score
        from core.regime_engine import compute_regime
        regime = compute_regime()
        setups = st.session_state.get("jarvis_setups", [])
        try:
            from core.adaptive_engine import AdaptiveEngine
            edge = AdaptiveEngine()
        except Exception:
            edge = None
        opp = compute_opportunity_score(regime, setups, edge)
        return build_jarvis_context(regime, setups, opp, edge)
    except Exception as e:
        return f"System context unavailable: {e}"


@st.cache_data(ttl=1800, show_spinner=False)
def _get_setups_for_jarvis(universe_key: str) -> list[dict]:
    try:
        from scan.pipeline import ScanPipeline
        from core.regime_engine import compute_regime
        regime = compute_regime()
        universe = [s for s in universe_key.split(",") if s]
        results = ScanPipeline(max_workers=12, min_quality_score=40).run(
            universe, top_n=10, regime_state=regime, skip_liquidity_filter=True
        )
        return [vars(r) for r in results]
    except Exception:
        return []


# ── Sub-components ────────────────────────────────────────────────────────────

def _render_header() -> None:
    st.markdown(
        "<div style='background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);"
        "border:1px solid #00d4ff33;border-radius:14px;padding:1rem 1.4rem;margin-bottom:1rem'>"
        "<div style='display:flex;align-items:center;justify-content:space-between'>"
        "<div style='display:flex;align-items:center;gap:.8rem'>"
        "<div style='font-size:1.6rem'>🤖</div>"
        "<div>"
        "<div style='color:#00d4ff;font-family:JetBrains Mono,monospace;font-size:.95rem;"
        "font-weight:700;letter-spacing:2px'>J.A.R.V.I.S</div>"
        "<div style='color:#4a5568;font-size:.65rem;letter-spacing:.08em'>"
        "JUST A RATHER VERY INTELLIGENT SYSTEM · AUTONOMOUS MULTI-AGENT CO-PILOT</div>"
        "</div></div>"
        "<div style='text-align:right'>"
        "<div style='color:#00d4a0;font-size:.65rem;font-family:JetBrains Mono,monospace'>"
        "● AGENTS ONLINE</div>"
        "<div style='color:#4a5568;font-size:.6rem'>DataAgent · ResearchAgent · CodeAgent · AnalysisAgent · SystemAgent</div>"
        "</div></div></div>",
        unsafe_allow_html=True,
    )


def _render_opportunity_score() -> None:
    try:
        from core.intelligence_hub import compute_opportunity_score
        from core.regime_engine import compute_regime
        regime = compute_regime()
        setups = st.session_state.get("jarvis_setups", [])
        try:
            from core.adaptive_engine import AdaptiveEngine
            edge = AdaptiveEngine()
        except Exception:
            edge = None
        opp = compute_opportunity_score(regime, setups, edge)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Opportunity", f"{opp.total:.0f}/100", opp.grade)
        c2.metric("Regime", f"{getattr(regime, 'regime_score', 0):.0f}/100")
        c3.metric("Breadth", f"{getattr(regime, 'breadth_strength', 0)}/100")
        c4.metric("Aggression", opp.recommended_aggression.split("(")[0].strip())
        c5.metric("Max Trades", str(opp.max_open_trades))

        color = opp.color
        pct = opp.total
        st.markdown(
            f"<div style='background:#161b22;border-radius:6px;height:6px;margin:-.3rem 0 .8rem'>"
            f"<div style='background:{color};width:{pct}%;height:6px;border-radius:6px;"
            f"transition:width .5s'></div></div>",
            unsafe_allow_html=True,
        )
    except Exception:
        pass


def _render_alerts(regime: object, opp: object) -> None:
    try:
        from core.intelligence_hub import generate_alerts
        prev_regime = st.session_state.get("jarvis_prev_regime", "")
        alerts = generate_alerts(
            regime_state=regime,
            prev_regime=prev_regime,
            top_setups=st.session_state.get("jarvis_setups", []),
            opp_score=opp,
        )
        new_alerts = [a for a in alerts if not a.dismissed]
        if new_alerts:
            st.session_state.setdefault("jarvis_alerts", [])
            existing_titles = {a.get("title") for a in st.session_state["jarvis_alerts"]}
            for a in new_alerts:
                if a.title not in existing_titles:
                    st.session_state["jarvis_alerts"].append(a.to_dict())

        _level_colors = {"INFO": "#00d4a0", "WARNING": "#f59e0b", "CRITICAL": "#ff4b4b"}
        _level_icons  = {"INFO": "ℹ", "WARNING": "⚠", "CRITICAL": "🚨"}

        stored = st.session_state.get("jarvis_alerts", [])
        if stored:
            with st.expander(f"🔔 {len(stored)} Active Alert{'s' if len(stored)!=1 else ''}", expanded=False):
                for alert in reversed(stored[-5:]):
                    lvl = alert.get("level", "INFO")
                    col = _level_colors.get(lvl, "#8892a4")
                    icon = _level_icons.get(lvl, "ℹ")
                    st.markdown(
                        f"<div style='background:{col}11;border-left:3px solid {col};"
                        f"border-radius:0 8px 8px 0;padding:.4rem .8rem;margin:.3rem 0'>"
                        f"<span style='color:{col};font-weight:700;font-size:.8rem'>"
                        f"{icon} {alert.get('title','')}</span><br>"
                        f"<span style='color:#8892a4;font-size:.75rem'>{alert.get('body','')}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                if st.button("Clear alerts", key="jarvis_clear_alerts"):
                    st.session_state["jarvis_alerts"] = []
                    st.rerun()
    except Exception:
        pass


def _render_agent_activity(activity: list[dict], agent_results: dict) -> None:
    if not activity and not agent_results:
        return

    with st.expander(
        f"🔬 Agent Activity — {len(agent_results)} agent{'s' if len(agent_results)!=1 else ''} ran",
        expanded=True,
    ):
        # Per-agent result cards
        if agent_results:
            cols = st.columns(min(len(agent_results), 3))
            for idx, (name, res) in enumerate(agent_results.items()):
                col = cols[idx % len(cols)]
                color = _AGENT_COLORS.get(name, "#8892a4")
                icon = _AGENT_ICONS.get(name, "🤖")
                status_icon = "✓" if res.success else "✗"
                status_color = "#00d4a0" if res.success else "#ff4b4b"
                with col:
                    st.markdown(
                        f"<div style='background:#0d1117;border:1px solid {color}44;"
                        f"border-radius:10px;padding:.6rem .8rem;margin:.2rem 0'>"
                        f"<div style='color:{color};font-size:.7rem;font-weight:700;"
                        f"font-family:JetBrains Mono,monospace'>{icon} {name}</div>"
                        f"<div style='color:{status_color};font-size:.65rem;margin:.1rem 0'>"
                        f"{status_icon} {len(res.steps)} step{'s' if len(res.steps)!=1 else ''}</div>"
                        f"<div style='color:#8892a4;font-size:.7rem;line-height:1.4'>"
                        f"{res.task[:80]}…</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # Steps detail
                if res.steps:
                    with st.expander(f"{name} steps", expanded=False):
                        for i, step in enumerate(res.steps, 1):
                            action = step.get("action", "?")
                            thought = step.get("thought", "")[:100]
                            obs = step.get("observation", "")[:200]
                            st.markdown(
                                f"**Step {i}** `{action}`  \n"
                                f"*{thought}*  \n"
                                f"`{obs}`"
                            )

        # Communication timeline
        if activity:
            st.markdown(
                "<div style='color:#4a5568;font-size:.65rem;font-family:JetBrains Mono,monospace;"
                "letter-spacing:.08em;margin:.6rem 0 .2rem'>AGENT COMMUNICATION TIMELINE</div>",
                unsafe_allow_html=True,
            )
            for msg in activity:
                from_ag = msg.get("from", "?")
                color = _AGENT_COLORS.get(from_ag, "#8892a4")
                type_color = _MSG_TYPE_COLORS.get(msg.get("type", "LOG"), "#4a5568")
                icon = _AGENT_ICONS.get(from_ag, "●")
                st.markdown(
                    f"<div style='display:flex;gap:.5rem;align-items:flex-start;"
                    f"padding:.2rem 0;border-bottom:1px solid #161b22'>"
                    f"<span style='color:#4a5568;font-size:.6rem;font-family:JetBrains Mono,monospace;"
                    f"min-width:4rem'>{msg.get('ts','')}</span>"
                    f"<span style='color:{color};font-size:.65rem;font-weight:700;min-width:7rem'>"
                    f"{icon} {from_ag}</span>"
                    f"<span style='color:{type_color};font-size:.6rem;min-width:4.5rem'>"
                    f"[{msg.get('type','LOG')}]</span>"
                    f"<span style='color:#8892a4;font-size:.7rem'>{msg.get('content','')[:120]}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


def _render_chat_history() -> None:
    for msg in st.session_state.get("jarvis_messages", []):
        role = msg["role"]
        content = msg["content"]
        used = msg.get("used_agents", [])

        if role == "assistant":
            agent_badge = ""
            if used:
                badges = " ".join(
                    f"<span style='background:{_AGENT_COLORS.get(a, \"#4a5568\")}22;"
                    f"border:1px solid {_AGENT_COLORS.get(a, \"#4a5568\")}44;"
                    f"border-radius:3px;padding:.05rem .3rem;font-size:.6rem;color:"
                    f"{_AGENT_COLORS.get(a, \"#4a5568\")}'>{_AGENT_ICONS.get(a,\"\")} {a}</span>"
                    for a in used
                )
                agent_badge = f"<div style='margin-bottom:.3rem'>{badges}</div>"
            st.markdown(
                f"<div style='background:#0d1117;border:1px solid #21262d;border-radius:10px;"
                f"padding:.7rem 1rem;margin:.4rem 0;border-left:3px solid #00d4ff'>"
                f"<span style='color:#00d4ff;font-size:.65rem;font-weight:700;letter-spacing:.1em'>"
                f"JARVIS</span><br>{agent_badge}"
                f"<span style='color:#c8cfe0;font-size:.85rem;line-height:1.6'>"
                f"{content}</span></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='background:#161b22;border:1px solid #21262d;border-radius:10px;"
                f"padding:.7rem 1rem;margin:.4rem 0;border-left:3px solid #4a5568;text-align:right'>"
                f"<span style='color:#4a5568;font-size:.65rem;font-weight:700;letter-spacing:.1em'>"
                f"YOU</span><br>"
                f"<span style='color:#e8eaf0;font-size:.85rem'>{content}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


def _render_system_panel() -> None:
    """Quick-action system control panel."""
    with st.expander("⚡ System Control", expanded=False):
        st.caption("Direct system actions via JARVIS agents. Requires confirmation for destructive ops.")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("📋 Git Status", key="sys_git_status", use_container_width=True):
                st.session_state["jarvis_pending"] = "Show me the current git status and any uncommitted changes"
                st.rerun()
        with c2:
            if st.button("🔄 Git Pull", key="sys_git_pull", use_container_width=True):
                st.session_state["jarvis_pending"] = "Pull the latest changes from the remote git repository"
                st.rerun()
        with c3:
            if st.button("📜 View Logs", key="sys_view_logs", use_container_width=True):
                st.session_state["jarvis_pending"] = "Show me the last 30 lines of the application log"
                st.rerun()
        with c4:
            if st.button("🌐 Market Scan", key="sys_market_scan", use_container_width=True):
                st.session_state["jarvis_pending"] = "Run a full market scan and show me the top 5 setups right now"
                st.rerun()

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            if st.button("💾 Deploy Code", key="sys_deploy", use_container_width=True):
                st.session_state["jarvis_pending"] = "Commit all uncommitted changes and push to the current branch"
                st.rerun()
        with c6:
            if st.button("🔍 Web Search", key="sys_websearch", use_container_width=True):
                st.session_state["jarvis_pending"] = "Search the web for latest Nifty 50 market news and analysis today"
                st.rerun()
        with c7:
            if st.button("🖥️ System Info", key="sys_info", use_container_width=True):
                st.session_state["jarvis_pending"] = "Show me system status: running processes, disk usage, and environment config"
                st.rerun()
        with c8:
            if st.button("📡 Live Prices", key="sys_prices", use_container_width=True):
                st.session_state["jarvis_pending"] = "Show me live prices for Nifty, VIX, Bank Nifty and today's regime"
                st.rerun()


# ── Main render ───────────────────────────────────────────────────────────────

def render_jarvis(universe: list[str]) -> None:
    universe_key = ",".join(sorted(universe))
    if "jarvis_setups" not in st.session_state:
        with st.spinner("JARVIS loading market intelligence…"):
            st.session_state["jarvis_setups"] = _get_setups_for_jarvis(universe_key)

    if "jarvis_messages" not in st.session_state:
        st.session_state["jarvis_messages"] = []

    # ── Header ────────────────────────────────────────────────────────────────
    _render_header()

    # ── Opportunity score ─────────────────────────────────────────────────────
    _render_opportunity_score()

    # ── Alerts ────────────────────────────────────────────────────────────────
    try:
        from core.regime_engine import compute_regime
        from core.intelligence_hub import compute_opportunity_score
        from core.adaptive_engine import AdaptiveEngine
        regime = compute_regime()
        edge = AdaptiveEngine()
        opp = compute_opportunity_score(regime, st.session_state.get("jarvis_setups", []), edge)
        _render_alerts(regime, opp)
    except Exception:
        pass

    # ── System control panel ──────────────────────────────────────────────────
    _render_system_panel()

    st.divider()

    # ── Latest agent activity (from last run) ─────────────────────────────────
    if st.session_state.get("jarvis_last_activity"):
        _render_agent_activity(
            st.session_state["jarvis_last_activity"],
            st.session_state.get("jarvis_last_agent_results", {}),
        )

    # ── Chat history ──────────────────────────────────────────────────────────
    _render_chat_history()

    # ── Suggested prompts ─────────────────────────────────────────────────────
    if not st.session_state["jarvis_messages"]:
        st.markdown(
            "<div style='color:#4a5568;font-size:.7rem;text-transform:uppercase;"
            "letter-spacing:.08em;margin:.5rem 0 .3rem'>Ask anything — agents will do the work</div>",
            unsafe_allow_html=True,
        )
        suggestions = [
            "What's the best setup right now? Run a scan.",
            "Search the web for Nifty 50 outlook this week",
            "Show me sector P/E ratios — look it up online",
            "Commit my changes and push to the current branch",
            "What's my win rate and edge by playbook type?",
            "Check the logs for any errors in the last hour",
        ]
        cols = st.columns(2)
        for i, sug in enumerate(suggestions):
            if cols[i % 2].button(sug, key=f"jarvis_sug_{i}", use_container_width=True):
                st.session_state["jarvis_pending"] = sug
                st.rerun()

    # ── Input bar ─────────────────────────────────────────────────────────────
    col_in, col_send = st.columns([8, 1])
    with col_in:
        user_input = st.text_input(
            "Ask JARVIS",
            key="jarvis_input",
            placeholder="Ask anything — agents have internet access, code execution, and git control…",
            label_visibility="collapsed",
        )
    with col_send:
        send = st.button("Send", key="jarvis_send", use_container_width=True, type="primary")

    if "jarvis_pending" in st.session_state:
        user_input = st.session_state.pop("jarvis_pending")
        send = True

    if (send or user_input) and user_input and user_input.strip():
        query = user_input.strip()
        context = _get_context()
        history = st.session_state["jarvis_messages"]

        st.session_state["jarvis_messages"].append({"role": "user", "content": query})

        with st.spinner("🧠 JARVIS orchestrating…"):
            from ai.jarvis_orchestrator import get_orchestrator
            orchestrator = get_orchestrator()
            result = orchestrator.run(query, context, history)

        # Store activity for display
        agent_results_dicts = {}
        if result.routed:
            for name, res in result.agent_results.items():
                agent_results_dicts[name] = res

        st.session_state["jarvis_last_activity"] = result.agent_activity
        st.session_state["jarvis_last_agent_results"] = agent_results_dicts

        # Append assistant message with agent metadata
        st.session_state["jarvis_messages"].append({
            "role": "assistant",
            "content": result.answer,
            "used_agents": result.used_agents,
        })

        st.rerun()

    # ── Controls ──────────────────────────────────────────────────────────────
    cc1, cc2, cc3, _ = st.columns([2, 2, 2, 4])
    with cc1:
        if st.button("Clear chat", key="jarvis_clear_chat"):
            st.session_state["jarvis_messages"] = []
            st.session_state.pop("jarvis_last_activity", None)
            st.session_state.pop("jarvis_last_agent_results", None)
            st.rerun()
    with cc2:
        if st.button("Refresh context", key="jarvis_refresh"):
            st.session_state.pop("jarvis_setups", None)
            st.cache_data.clear()
            st.rerun()
    with cc3:
        if st.button("Reset agents", key="jarvis_reset_agents"):
            # Force fresh orchestrator
            import ai.jarvis_orchestrator as _orch_mod
            _orch_mod._orchestrator = None
            st.success("Agents restarted")
            st.rerun()
