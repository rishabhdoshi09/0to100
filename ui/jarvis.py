"""
JARVIS — Just A Rather Very Intelligent System.
A conversational AI co-pilot that knows everything about the current market,
your setups, your edge, and your portfolio.

Unlike a chatbot, JARVIS has full system context injected into every message.
It knows the regime, the top setups, your win rate, open positions.
It doesn't just answer — it anticipates.
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import streamlit as st


_SYSTEM_PROMPT = """You are JARVIS, the AI co-pilot for QUANTTERM — an institutional-grade trading intelligence system.

You have real-time access to:
- Current market regime, Nifty levels, VIX, sector rotation
- Live scan results: top setups ranked by quality and expected value
- The user's personal trade edge (win rates by playbook and regime)
- Portfolio position and risk state
- Opportunity Score (0-100) synthesising all market dimensions

Your personality:
- Terse, confident, institutional. Like a senior prop trader.
- Never hedge excessively. Give a clear view.
- If the market is bad, say so directly. If it's good, say so.
- Use numbers. ₹ for prices. R-multiples for trade outcomes. % for rates.
- No fluff. No "great question!" No disclaimers about market risk (user knows).
- Think in probability and expectancy, not certainty.

Capabilities:
- Explain any setup, playbook, or regime in plain language
- Calculate position size given account size and risk
- Compare current conditions to historical analogs
- Identify what's wrong with a trade idea
- Give a structured entry/stop/target for any setup in the queue
- Analyse the user's journal data for patterns and biases

Always start your first response in a session with a 2-sentence market brief.
After that, respond directly to what's asked."""


def _get_context() -> str:
    """Pull live system context."""
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
    """Pre-run pipeline so JARVIS has setup context."""
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


def _call_deepseek(messages: list[dict]) -> str:
    key = os.getenv("DEEPSEEK_API_KEY", "")
    if not key:
        return "DEEPSEEK_API_KEY not configured. Add it to your .env file."
    import requests
    try:
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": "deepseek-chat",
                "messages": messages,
                "temperature": 0.2,
                "max_tokens": 600,
            },
            timeout=30,
        )
        data = resp.json()
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        return f"Error: {data.get('error', {}).get('message', 'Unknown error')}"
    except Exception as e:
        return f"Connection error: {e}"


def render_jarvis(universe: list[str]) -> None:
    # Pre-load setup context
    universe_key = ",".join(sorted(universe))
    if "jarvis_setups" not in st.session_state:
        with st.spinner("JARVIS loading market intelligence…"):
            st.session_state["jarvis_setups"] = _get_setups_for_jarvis(universe_key)

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        "<div style='background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);"
        "border:1px solid #00d4ff33;border-radius:14px;padding:1rem 1.4rem;margin-bottom:1rem'>"
        "<div style='display:flex;align-items:center;gap:.8rem'>"
        "<div style='font-size:1.6rem'>🤖</div>"
        "<div>"
        "<div style='color:#00d4ff;font-family:JetBrains Mono,monospace;font-size:.95rem;"
        "font-weight:700;letter-spacing:2px'>J.A.R.V.I.S</div>"
        "<div style='color:#4a5568;font-size:.65rem;letter-spacing:.08em'>"
        "JUST A RATHER VERY INTELLIGENT SYSTEM · QUANTTERM AI CO-PILOT</div>"
        "</div></div></div>",
        unsafe_allow_html=True,
    )

    # ── Opportunity Score sidebar ─────────────────────────────────────────────
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

        # Score bar
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

    # ── Alerts ────────────────────────────────────────────────────────────────
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
            # Deduplicate by title
            existing_titles = {a.get("title") for a in st.session_state["jarvis_alerts"]}
            for a in new_alerts:
                if a.title not in existing_titles:
                    st.session_state["jarvis_alerts"].append(a.to_dict())

        _level_colors = {"INFO": "#00d4a0", "WARNING": "#f59e0b", "CRITICAL": "#ff4b4b"}
        _level_icons  = {"INFO": "ℹ", "WARNING": "⚠", "CRITICAL": "🚨"}

        stored = st.session_state.get("jarvis_alerts", [])
        if stored:
            with st.expander(f"🔔 {len(stored)} Active Alert{'s' if len(stored)!=1 else ''}", expanded=False):
                for i, alert in enumerate(reversed(stored[-5:])):
                    lvl = alert.get("level", "INFO")
                    col = _level_colors.get(lvl, "#8892a4")
                    icon = _level_icons.get(lvl, "ℹ")
                    st.markdown(
                        f"<div style='background:{col}11;border-left:3px solid {col};"
                        f"border-radius:0 8px 8px 0;padding:.4rem .8rem;margin:.3rem 0'>"
                        f"<span style='color:{col};font-weight:700;font-size:.8rem'>"
                        f"{icon} {alert.get('title','')}</span><br>"
                        f"<span style='color:#8892a4;font-size:.75rem'>{alert.get('body','')}</span>"
                        f"<span style='color:#4a5568;font-size:.65rem;float:right'>{alert.get('timestamp','')}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                if st.button("Clear alerts", key="jarvis_clear_alerts"):
                    st.session_state["jarvis_alerts"] = []
                    st.rerun()
    except Exception:
        pass

    st.divider()

    # ── Chat interface ────────────────────────────────────────────────────────
    if "jarvis_messages" not in st.session_state:
        st.session_state["jarvis_messages"] = []

    # Render chat history
    for msg in st.session_state["jarvis_messages"]:
        role = msg["role"]
        content = msg["content"]
        if role == "assistant":
            st.markdown(
                f"<div style='background:#0d1117;border:1px solid #21262d;border-radius:10px;"
                f"padding:.7rem 1rem;margin:.4rem 0;border-left:3px solid #00d4ff'>"
                f"<span style='color:#00d4ff;font-size:.65rem;font-weight:700;letter-spacing:.1em'>"
                f"JARVIS</span><br>"
                f"<span style='color:#c8cfe0;font-size:.85rem;line-height:1.6'>{content}</span>"
                f"</div>",
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

    # Suggested prompts (only when chat is empty)
    if not st.session_state["jarvis_messages"]:
        st.markdown(
            "<div style='color:#4a5568;font-size:.7rem;text-transform:uppercase;"
            "letter-spacing:.08em;margin:.5rem 0 .3rem'>Quick questions</div>",
            unsafe_allow_html=True,
        )
        suggestions = [
            "What's the best trade setup today?",
            "Is this a good day to trade? Give me the opportunity score breakdown.",
            "Which playbooks should I focus on right now?",
            "Explain the current regime and what it means for my trades.",
        ]
        cols = st.columns(2)
        for i, sug in enumerate(suggestions):
            if cols[i % 2].button(sug, key=f"jarvis_sug_{i}", use_container_width=True):
                st.session_state["jarvis_pending"] = sug
                st.rerun()

    # Input bar
    col_in, col_send = st.columns([8, 1])
    with col_in:
        user_input = st.text_input(
            "Ask JARVIS",
            key="jarvis_input",
            placeholder="Ask anything about the market, your setups, or trading strategy…",
            label_visibility="collapsed",
        )
    with col_send:
        send = st.button("Send", key="jarvis_send", use_container_width=True, type="primary")

    # Handle pending suggestion click
    if "jarvis_pending" in st.session_state:
        user_input = st.session_state.pop("jarvis_pending")
        send = True

    if (send or user_input) and user_input and user_input.strip():
        # Build message list with system context
        context = _get_context()
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT + "\n\n" + context},
        ]
        # Add conversation history (last 6 turns to stay within token budget)
        for m in st.session_state["jarvis_messages"][-12:]:
            messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": user_input.strip()})

        # Store user message
        st.session_state["jarvis_messages"].append({"role": "user", "content": user_input.strip()})

        with st.spinner("JARVIS thinking…"):
            response = _call_deepseek(messages)

        st.session_state["jarvis_messages"].append({"role": "assistant", "content": response})
        st.rerun()

    # Controls
    cc1, cc2, cc3 = st.columns([2, 2, 6])
    with cc1:
        if st.button("Clear chat", key="jarvis_clear_chat"):
            st.session_state["jarvis_messages"] = []
            st.rerun()
    with cc2:
        if st.button("Refresh context", key="jarvis_refresh"):
            st.session_state.pop("jarvis_setups", None)
            st.cache_data.clear()
            st.rerun()
