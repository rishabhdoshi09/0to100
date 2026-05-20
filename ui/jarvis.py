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

def _plain(text: str) -> str:
    """Apply plain English translation if mode is active."""
    try:
        from ui.plain_english import translate
        return translate(text)
    except Exception:
        return text


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
        from core.intelligence_hub import compute_opportunity_score, get_trading_rules
        from core.regime_engine import compute_regime
        regime = compute_regime()
        setups = st.session_state.get("jarvis_setups", [])
        try:
            from core.adaptive_engine import AdaptiveEngine
            edge = AdaptiveEngine()
        except Exception:
            edge = None
        opp = compute_opportunity_score(regime, setups, edge)
        rules = get_trading_rules(opp)

        confidence = getattr(regime, "regime_confidence", 0)
        confidence_label = getattr(regime, "regime_confidence_label", "—")
        conf_color = {"HIGH": "#00d4a0", "MODERATE": "#f59e0b", "LOW": "#f97316", "UNCERTAIN": "#ff4b4b"}.get(confidence_label, "#8892a4")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Opportunity", f"{opp.total:.0f}/100", opp.grade)
        c2.metric("Regime Score", f"{getattr(regime, 'regime_score', 0):.0f}/100")
        c3.metric("Confidence", f"{confidence:.0f}%", confidence_label)
        c4.metric("Max Positions", str(rules.max_positions))
        c5.metric("Size", rules.display_size())

        color = rules.color
        pct = opp.total
        st.markdown(
            f"<div style='background:#161b22;border-radius:6px;height:6px;margin:-.3rem 0 .4rem'>"
            f"<div style='background:{color};width:{pct}%;height:6px;border-radius:6px;"
            f"transition:width .5s'></div></div>",
            unsafe_allow_html=True,
        )

        # Hard rules banner — always visible, color-coded
        border_style = f"2px solid {rules.color}55" if rules.allow_new_trades else f"2px solid {rules.color}"
        st.markdown(
            f"<div style='background:{rules.color}11;border:{border_style};"
            f"border-radius:8px;padding:.4rem .9rem;margin-bottom:.6rem;"
            f"display:flex;justify-content:space-between;align-items:center'>"
            f"<span style='color:{rules.color};font-family:JetBrains Mono,monospace;"
            f"font-size:.72rem;font-weight:700'>"
            f"{'🚫' if not rules.allow_new_trades else '⚡'} {rules.label.upper()}: {rules.one_liner()}</span>"
            f"<span style='color:#4a5568;font-size:.65rem'>{rules.rationale}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Plain English regime explanation (shown when mode is active)
        try:
            from ui.plain_english import regime_card_html, is_plain_english
            if is_plain_english():
                html = regime_card_html(
                    regime.market_regime,
                    getattr(regime, "regime_score", 50),
                    regime.vix,
                    regime.breadth_label,
                )
                if html:
                    st.markdown(html, unsafe_allow_html=True)
        except Exception:
            pass

        # Store rules in session for other components to use
        st.session_state["jarvis_trading_rules"] = {
            "max_positions": rules.max_positions,
            "size_multiplier": rules.size_multiplier,
            "min_tier": rules.min_tier,
            "allow_new_trades": rules.allow_new_trades,
            "label": rules.label,
            "color": rules.color,
        }
    except Exception:
        pass


def _render_intelligence_bar() -> None:
    """FII/DII + block deals + circuit breakers — institutional intelligence feed."""
    rows = []

    # ── FII/DII ───────────────────────────────────────────────────────────────
    try:
        from data.fii_flow import get_fii_dii_flow
        flow = get_fii_dii_flow(days=5)
        today = flow.get("today", {})
        if today:
            fii = today.get("fii_net_cr", 0)
            dii = today.get("dii_net_cr", 0)
            fs  = flow.get("fii_streak", 0)
            ds  = flow.get("dii_streak", 0)
            fii_c = "#00d4a0" if fii >= 0 else "#ff4b4b"
            dii_c = "#00d4a0" if dii >= 0 else "#ff4b4b"

            def streak_tag(s: int) -> str:
                if not s:
                    return ""
                return f"<span style='color:#4a5568;font-size:.6rem'> {abs(s)}d {'buy' if s>0 else 'sell'}</span>"

            rows.append(
                f"<span style='color:#4a5568;font-size:.65rem;font-family:JetBrains Mono,monospace'>FII/DII</span>&nbsp;&nbsp;"
                f"<span style='color:{fii_c};font-weight:700'>{'▲' if fii>=0 else '▼'} ₹{abs(fii):,.0f}Cr</span>{streak_tag(fs)}"
                f"&nbsp;&nbsp;"
                f"<span style='color:{dii_c};font-weight:700'>DII {'▲' if dii>=0 else '▼'} ₹{abs(dii):,.0f}Cr</span>{streak_tag(ds)}"
                f"&nbsp;&nbsp;<span style='color:#8892a4;font-size:.7rem'>{flow.get('insight','')}</span>"
            )
    except Exception:
        pass

    # ── Block deals ───────────────────────────────────────────────────────────
    try:
        from data.block_deals import get_significant_deals, format_deal_insight
        from config import settings
        uni = list(getattr(settings, "universe", []))[:30] or []
        deals = get_significant_deals(uni, min_value_cr=25.0) if uni else []
        if deals:
            insight = format_deal_insight(deals[:2])
            rows.append(
                f"<span style='color:#4a5568;font-size:.65rem;font-family:JetBrains Mono,monospace'>BLOCK</span>&nbsp;&nbsp;"
                f"<span style='color:#a78bfa;font-size:.72rem'>{insight}</span>"
            )
    except Exception:
        pass

    # ── Circuit breakers ──────────────────────────────────────────────────────
    try:
        from data.circuit_breakers import get_circuit_stocks
        circuits = get_circuit_stocks("both")
        if circuits:
            upper = [c for c in circuits if c.circuit_type == "UPPER"]
            lower = [c for c in circuits if c.circuit_type == "LOWER"]
            parts = []
            if upper:
                syms = ", ".join(c.symbol for c in upper[:3])
                parts.append(f"<span style='color:#00d4a0'>↑UC: {syms}</span>")
            if lower:
                syms = ", ".join(c.symbol for c in lower[:3])
                parts.append(f"<span style='color:#ff4b4b'>↓LC: {syms}</span>")
            if parts:
                rows.append(
                    f"<span style='color:#4a5568;font-size:.65rem;font-family:JetBrains Mono,monospace'>CIRCUIT</span>&nbsp;&nbsp;"
                    + "&nbsp;&nbsp;".join(parts)
                )
    except Exception:
        pass

    if not rows:
        return

    content = "&nbsp;&nbsp;<span style='color:#21262d'>|</span>&nbsp;&nbsp;".join(rows)
    st.markdown(
        f"<div style='background:#0d1117;border:1px solid #21262d;border-radius:10px;"
        f"padding:.45rem 1rem;margin-bottom:.5rem;font-size:.75rem;line-height:1.8;overflow-x:auto'>"
        f"{content}</div>",
        unsafe_allow_html=True,
    )


def _render_memory_context() -> None:
    """Show what JARVIS remembers — collapsible."""
    try:
        from ai.jarvis_memory import get_memory
        mem = get_memory()
        sessions = mem.get_recent_sessions(1)
        facts = mem.recall(limit=10)
        alerts = mem.get_active_alerts()

        if not sessions and not facts and not alerts:
            return

        with st.expander("🧠 JARVIS Memory", expanded=False):
            if sessions:
                s = sessions[0]
                ts = s["timestamp"][:10]
                st.markdown(
                    f"<div style='color:#8892a4;font-size:.72rem'>"
                    f"<span style='color:#00d4ff;font-weight:700'>Last session ({ts}):</span> "
                    f"{s['summary']}</div>",
                    unsafe_allow_html=True,
                )
            if facts:
                st.markdown(
                    "<div style='color:#4a5568;font-size:.65rem;margin-top:.4rem;"
                    "font-family:JetBrains Mono,monospace;letter-spacing:.06em'>REMEMBERED FACTS</div>",
                    unsafe_allow_html=True,
                )
                for f in facts[:8]:
                    st.markdown(
                        f"<div style='color:#8892a4;font-size:.72rem'>"
                        f"<span style='color:#a78bfa'>{f['key']}</span>: {f['value']}</div>",
                        unsafe_allow_html=True,
                    )
            if alerts:
                st.markdown(
                    "<div style='color:#f59e0b;font-size:.65rem;margin-top:.4rem;"
                    "font-family:JetBrains Mono,monospace'>PRICE ALERTS ACTIVE</div>",
                    unsafe_allow_html=True,
                )
                for a in alerts:
                    st.markdown(
                        f"<div style='color:#8892a4;font-size:.72rem'>"
                        f"<span style='color:#fbbf24'>{a['symbol']}</span> "
                        f"{a['direction']} ₹{a['price']:,.0f}</div>",
                        unsafe_allow_html=True,
                    )
            col_clear, _ = st.columns([2, 6])
            with col_clear:
                if st.button("Clear memory", key="clear_jarvis_memory"):
                    for f in facts:
                        mem.forget(f["key"])
                    st.rerun()
    except Exception:
        pass


def _render_regime_analog() -> None:
    """Historical regime analog — show similar past periods."""
    try:
        from core.regime_analog import find_analogs
        from core.regime_engine import compute_regime
        regime = compute_regime()
        regime_dict = {
            "market_regime": regime.market_regime,
            "volatility_regime": regime.volatility_regime,
            "breadth_label": regime.breadth_label,
            "vix": regime.vix,
            "nifty_change_5d": regime.nifty_change_5d,
            "sector_returns": regime.sector_returns,
        }
        analogs = find_analogs(regime_dict, top_n=2)
        if not analogs:
            return

        with st.expander("📅 Historical Regime Analogs", expanded=False):
            for analog in analogs:
                color = "#00d4ff" if analog.similarity_score >= 70 else "#f59e0b"
                st.markdown(
                    f"<div style='background:#0d1117;border:1px solid {color}33;"
                    f"border-radius:8px;padding:.6rem .9rem;margin:.3rem 0'>"
                    f"<div style='display:flex;justify-content:space-between'>"
                    f"<span style='color:{color};font-weight:700;font-size:.8rem'>"
                    f"{analog.period}</span>"
                    f"<span style='color:#4a5568;font-size:.65rem'>"
                    f"Similarity: {analog.similarity_score:.0f}%</span>"
                    f"</div>"
                    f"<div style='color:#8892a4;font-size:.72rem;margin:.2rem 0'>"
                    f"{analog.outcome_summary}</div>"
                    f"<div style='display:flex;gap:1rem;margin-top:.3rem'>"
                    f"<div><div style='color:#00d4a0;font-size:.65rem;font-weight:700'>WORKED</div>"
                    + "".join(f"<div style='color:#8892a4;font-size:.7rem'>✓ {w}</div>" for w in analog.what_worked[:2])
                    + f"</div><div><div style='color:#ff4b4b;font-size:.65rem;font-weight:700'>FAILED</div>"
                    + "".join(f"<div style='color:#8892a4;font-size:.7rem'>✗ {f}</div>" for f in analog.what_failed[:2])
                    + f"</div></div></div>",
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
                badge_parts = []
                for a in used:
                    c = _AGENT_COLORS.get(a, "#4a5568")
                    ic = _AGENT_ICONS.get(a, "")
                    badge_parts.append(
                        f"<span style='background:{c}22;border:1px solid {c}44;"
                        f"border-radius:3px;padding:.05rem .3rem;font-size:.6rem;color:{c}'>"
                        f"{ic} {a}</span>"
                    )
                badges = " ".join(badge_parts)
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

    # ── Intelligence Bar (FII/DII + Block Deals + Circuit Breakers) ──────────
    _render_intelligence_bar()

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

    # ── Memory context + Regime analog ───────────────────────────────────────
    col_mem, col_analog = st.columns(2)
    with col_mem:
        _render_memory_context()
    with col_analog:
        _render_regime_analog()

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
            # Save session to memory before clearing
            msgs = st.session_state.get("jarvis_messages", [])
            if len(msgs) >= 4:
                try:
                    from ai.jarvis_orchestrator import get_orchestrator
                    get_orchestrator().save_session(msgs)
                except Exception:
                    pass
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
