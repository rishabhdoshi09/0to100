"""
Streamlit page for the Multi-Agent Trading Advisor.
Imported by app.py and rendered inside the 🤖 Agents top-level tab.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import streamlit as st


def render_agent_dashboard() -> None:
    st.markdown(
        "<h2 style='color:#00d4ff;font-family:JetBrains Mono,monospace;"
        "font-size:1.4rem;margin:0'>🤖 Multi-Agent Trading Advisor</h2>"
        "<p style='color:#8892a4;font-size:.8rem;margin:.2rem 0 1rem'>"
        "Three specialist agents (Technical · Sentiment · Risk) orchestrated by "
        "a Supervisor — all powered by DeepSeek R1 &amp; V3.</p>",
        unsafe_allow_html=True,
    )

    # Lazy-init supervisor so it only loads when this tab is opened
    if "agent_supervisor" not in st.session_state:
        with st.spinner("Initialising agents…"):
            from agents.supervisor import AgentSupervisor
            st.session_state.agent_supervisor = AgentSupervisor()

    # ── Input row ─────────────────────────────────────────────────────────────
    col_sym, col_btn = st.columns([4, 1])
    with col_sym:
        symbol = st.text_input(
            "NSE Stock Symbol",
            value=st.session_state.get("agent_last_symbol", "RELIANCE"),
            key="agent_symbol",
            placeholder="e.g. INFY, HDFCBANK, TCS",
        ).upper().strip()
    with col_btn:
        st.write("")  # vertical alignment
        st.write("")
        run_btn = st.button("🤖 Analyse", key="agent_run", type="primary", use_container_width=True)

    st.divider()

    # ── Run agents ────────────────────────────────────────────────────────────
    if run_btn and symbol:
        with st.spinner(f"Agents analysing **{symbol}** (DeepSeek R1 + V3) — this may take ~30s…"):
            try:
                result = st.session_state.agent_supervisor.evaluate_stock(symbol)
                st.session_state.agent_last_result = result
                st.session_state.agent_last_symbol = symbol
            except Exception as exc:
                st.error(f"Agent error: {exc}")
                st.exception(exc)
                return

    # ── Display last result ───────────────────────────────────────────────────
    if "agent_last_result" in st.session_state:
        _render_result(st.session_state.agent_last_result)


# ── Private helpers ────────────────────────────────────────────────────────────

def _render_result(result: Dict[str, Any]) -> None:
    symbol = result.get("symbol", "")
    decision = str(result.get("DECISION", result.get("decision", "HOLD"))).upper()
    confidence = result.get("CONFIDENCE", result.get("confidence", 0))
    risk_override = str(result.get("RISK_OVERRIDE", result.get("risk_override", "NO"))).upper()
    reasoning = result.get("REASONING", result.get("reasoning", []))

    # ── Decision banner ───────────────────────────────────────────────────────
    _colors = {"BUY": "#00d4a0", "SELL": "#ff4b4b", "HOLD": "#f0a500"}
    color = _colors.get(decision, "#8892a4")
    st.markdown(
        f"<div style='background:{color}22;border:1.5px solid {color};"
        f"border-radius:10px;padding:1rem 1.5rem;margin:.5rem 0'>"
        f"<span style='color:{color};font-size:2.2rem;font-weight:700'>{decision}</span>"
        f"<span style='color:#8892a4;font-size:1rem;margin-left:1.2rem'>"
        f"Confidence: <b style='color:#e0e0e0'>{confidence}%</b></span>"
        f"<span style='color:#8892a4;font-size:.85rem;margin-left:1rem'>· {symbol}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if risk_override == "YES":
        st.error("⚠️ Risk Agent override — trade rejected by risk controls.")

    # ── Price guidance ────────────────────────────────────────────────────────
    g1, g2, g3 = st.columns(3)
    g1.metric("Entry Guidance", result.get("entry_price_guidance", "N/A"))
    g2.metric("Stop Loss", result.get("stop_loss_guidance", "N/A"))
    g3.metric("Target", result.get("target_price_guidance", "N/A"))

    # ── Reasoning ─────────────────────────────────────────────────────────────
    st.markdown("#### Supervisor Reasoning")
    if isinstance(reasoning, list):
        for point in reasoning:
            st.markdown(f"- {point}")
    elif isinstance(reasoning, str) and reasoning:
        st.write(reasoning)
    else:
        st.info("No reasoning returned.")

    # ── Agent sub-reports ─────────────────────────────────────────────────────
    st.markdown("#### Agent Reports")
    reports = result.get("agent_reports", {})

    ta1, ta2, ta3 = st.tabs(["📈 Technical", "📰 Sentiment", "⚠️ Risk"])

    with ta1:
        tech = reports.get("technical", {})
        if "error" in tech:
            st.warning(f"Technical agent error: {tech['error']}")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Direction", tech.get("direction", "N/A"))
            c2.metric("Strength", f"{tech.get('strength', 'N/A')}/10")
            signals = tech.get("key_signals", [])
            c3.metric("Key Signals", len(signals))
            if signals:
                for s in signals:
                    st.markdown(f"- {s}")
            st.caption(tech.get("summary", ""))
            if tech.get("support_level"):
                st.write(f"Support: {tech['support_level']}  |  Resistance: {tech.get('resistance_level', 'N/A')}")

    with ta2:
        sent = reports.get("sentiment", {})
        if "error" in sent:
            st.warning(f"Sentiment agent error: {sent['error']}")
        else:
            s1, s2 = st.columns(2)
            s1.metric("Sentiment", sent.get("overall_sentiment", "N/A"))
            s2.metric("Intensity", f"{sent.get('intensity', 'N/A')}/10")
            events = sent.get("key_events", [])
            if events:
                st.markdown("**Key events:**")
                for e in events:
                    st.markdown(f"- {e}")
            catalyst = sent.get("catalyst_risk", "")
            if catalyst and catalyst.lower() not in ("none", ""):
                st.warning(f"Catalyst risk: {catalyst}")
            st.caption(sent.get("summary", ""))

    with ta3:
        risk = reports.get("risk", {})
        if "error" in risk:
            st.warning(f"Risk agent error: {risk['error']}")
        else:
            r1, r2, r3 = st.columns(3)
            r1.metric("Decision", risk.get("decision", "N/A"))
            r2.metric("Position Size", f"{risk.get('position_size_pct', 'N/A')}%")
            r3.metric("Max Loss", f"{risk.get('max_loss_pct', 'N/A')}%")
            if risk.get("stop_loss_level"):
                st.write(f"Stop Loss Level: **{risk['stop_loss_level']}**")
            factors = risk.get("risk_factors", [])
            if factors:
                st.markdown("**Risk factors:**")
                for f in factors:
                    st.markdown(f"- {f}")
            st.caption(risk.get("summary", ""))

    # ── Raw JSON ──────────────────────────────────────────────────────────────
    with st.expander("Raw JSON response"):
        display = {k: v for k, v in result.items() if k != "agent_reports"}
        st.json(display)
