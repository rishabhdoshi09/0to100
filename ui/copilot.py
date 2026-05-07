"""
DevBloom AI Co-Pilot "Dev" — dual-LLM streaming chat (DeepSeek → Claude).

Every query goes through the DualChatEngine:
  Stage 1 · DeepSeek  — fast first-pass financial analysis
  Stage 2 · Claude    — validates, refines, or overrides

The UI shows a decision badge so the user always knows which model
had the final say on each response.
"""
from __future__ import annotations

from datetime import datetime

import streamlit as st

from llm.dual_chat import DualChatEngine, decision_badge_html

_engine = DualChatEngine()

MAX_HISTORY = 20

_COPILOT_INTRO = """You are Dev, the AI co-pilot in DevBloom Terminal — a professional trading cockpit for NSE equities.

You have deep expertise in:
• Indian equity markets (NSE/BSE), Nifty sectors, FII/DII flows, RBI policy
• Technical analysis: price action, volume, indicators, multi-timeframe reads
• Fundamental analysis: P&L quality, balance sheet health, DCF, earnings
• Quant: backtesting, factor models, regime detection, conviction scoring

Style:
• Concise and precise — use bullets for multi-part answers
• Use trader language, not textbook prose
• For /why: macro → sector → stock → key chart level
• For /model: quick DCF skeleton with stated assumptions
• For /idea: entry / stop / target / time horizon / one-line thesis
• Always state uncertainty clearly

Slash commands:
/why [symbol/index]   explain recent move
/model [symbol]       quick DCF or valuation frame
/idea [symbol]        generate a trade idea
/compare [A] [B]      compare two stocks
/explain [indicator]  plain-language explanation
/scan [criteria]      suggest how to screen for a setup"""


def _morning_prompt() -> str:
    return (
        "/idea Generate the top 5 NSE swing trade setups for today. "
        "For each give: symbol, entry zone, stop, target, R:R, time horizon, 1-line thesis. "
        "Focus on momentum + volume breakouts. "
        "Current date: " + datetime.now().strftime("%Y-%m-%d %A")
    )


def render_copilot_sidebar(context: dict | None = None):
    """Dev co-pilot as a persistent sidebar panel."""
    ctx = context or {}

    if "devbloom_chat" not in st.session_state:
        st.session_state["devbloom_chat"] = []

    st.sidebar.markdown(
        "<div style='padding:.5rem 0 .25rem'>"
        "<span style='font-size:.6rem;color:#8892a4;text-transform:uppercase;letter-spacing:.06em'>AI Co-Pilot</span><br>"
        "<span style='font-size:1.1rem;color:#00d4ff;font-weight:700'>⚡ Dev</span>"
        "<span style='font-size:.6rem;color:#8892a4;margin-left:.5rem'>DeepSeek → Claude</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    history = st.session_state["devbloom_chat"]

    for msg in history[-10:]:
        if msg["role"] == "user":
            st.sidebar.markdown(
                f"<div style='background:rgba(0,212,255,.08);border-left:2px solid #00d4ff;"
                f"border-radius:0 8px 8px 0;padding:.4rem .7rem;margin:.2rem 0;"
                f"font-size:.78rem;color:#e8eaf0'>{msg['content']}</div>",
                unsafe_allow_html=True,
            )
        else:
            badge = decision_badge_html(msg.get("decision_maker", "claude_validated"))
            st.sidebar.markdown(
                f"<div style='background:rgba(255,255,255,.04);border-left:2px solid #8892a4;"
                f"border-radius:0 8px 8px 0;padding:.4rem .7rem;margin:.2rem 0;"
                f"font-size:.78rem;color:#c8cfe0'>"
                f"{badge}<br>{msg['content']}"
                f"</div>",
                unsafe_allow_html=True,
            )

    user_input = st.sidebar.text_input(
        "Ask Dev…",
        placeholder="/why RELIANCE  /idea INFY  or anything",
        key="copilot_sidebar_input",
        label_visibility="collapsed",
    )
    send = st.sidebar.button("Send ↵", key="copilot_sidebar_send", use_container_width=True)

    if send and user_input.strip():
        history.append({"role": "user", "content": user_input.strip()})
        with st.sidebar:
            placeholder = st.empty()
            full, dm = "", "claude_validated"
            for chunk, dm in _engine.stream(user_input.strip(), history[:-1], ctx):
                full += chunk
                placeholder.markdown(
                    f"<div style='font-size:.78rem;color:#e8eaf0'>{full}▍</div>",
                    unsafe_allow_html=True,
                )
            placeholder.markdown(
                f"<div style='font-size:.78rem;color:#e8eaf0'>{full}</div>",
                unsafe_allow_html=True,
            )
        history.append({"role": "assistant", "content": full, "decision_maker": dm})
        st.session_state["devbloom_chat"] = history[-MAX_HISTORY * 2:]
        st.rerun()

    if st.sidebar.button("Clear", key="copilot_sidebar_clear"):
        st.session_state["devbloom_chat"] = []
        st.rerun()


def render_copilot_inline(context: dict | None = None):
    """Full inline Co-Pilot tab — dual-LLM streaming chat."""
    ctx = context or {}

    st.markdown("### ⚡ Dev — AI Co-Pilot")
    st.caption(
        "Powered by **DeepSeek → Claude** dual-LLM pipeline. "
        "DeepSeek generates the first-pass analysis; Claude validates or overrides. "
        "Use `/why`, `/model`, `/idea`, `/compare` slash commands."
    )

    if "devbloom_chat_inline" not in st.session_state:
        st.session_state["devbloom_chat_inline"] = []

    history = st.session_state["devbloom_chat_inline"]

    # Morning ideas shortcut
    if st.button("☀️ Generate Morning Trade Ideas  (DeepSeek → Claude)", key="morning_ideas"):
        _fire_and_append(history, _morning_prompt(), ctx)
        st.rerun()

    st.divider()

    # Chat history
    for msg in history[-20:]:
        if msg["role"] == "user":
            st.markdown(
                f"<div style='margin:.3rem 0'>"
                f"<span style='color:#00d4ff;font-size:.7rem;font-weight:600;text-transform:uppercase'>You</span><br>"
                f"<span style='color:#e8eaf0;font-size:.88rem;line-height:1.7'>{msg['content']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            badge = decision_badge_html(msg.get("decision_maker", "claude_validated"))
            st.markdown(
                f"<div style='margin:.3rem 0'>"
                f"<span style='color:#ffb800;font-size:.7rem;font-weight:600;text-transform:uppercase'>⚡ Dev</span> "
                f"{badge}<br>"
                f"<span style='color:#e8eaf0;font-size:.88rem;line-height:1.7'>{msg['content']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Input
    col1, col2 = st.columns([5, 1])
    user_input = col1.text_input(
        "Message", placeholder="Type /why RELIANCE or ask anything…",
        label_visibility="collapsed", key="copilot_inline_input",
    )
    send = col2.button("Send", key="copilot_inline_send", use_container_width=True)

    if send and user_input.strip():
        history.append({"role": "user", "content": user_input.strip()})

        # Show streaming output
        st.markdown(
            f"<span style='color:#00d4ff;font-size:.7rem;font-weight:600'>⚡ Dev (thinking…)</span>",
            unsafe_allow_html=True,
        )
        placeholder = st.empty()
        full, dm = "", "claude_validated"
        ds_shown = False

        for chunk, dm in _engine.stream(user_input.strip(), history[:-1], ctx):
            full += chunk
            # Show a "DeepSeek ✓" indicator once Claude takes over
            if not ds_shown and "[VALIDATED]" in full or "[OVERRIDE]" in full:
                ds_shown = True
            placeholder.markdown(
                f"<div style='color:#e8eaf0;font-size:.88rem;line-height:1.7'>{full}▍</div>",
                unsafe_allow_html=True,
            )
        placeholder.markdown(
            f"<div style='color:#e8eaf0;font-size:.88rem;line-height:1.7'>{full}</div>",
            unsafe_allow_html=True,
        )

        history.append({"role": "assistant", "content": full, "decision_maker": dm})
        st.session_state["devbloom_chat_inline"] = history[-MAX_HISTORY * 2:]
        st.rerun()

    if st.button("🗑 Clear chat", key="copilot_inline_clear"):
        st.session_state["devbloom_chat_inline"] = []
        st.rerun()


def _fire_and_append(history: list[dict], prompt: str, ctx: dict):
    """Run a dual-LLM query and append result to history (used for buttons)."""
    history.append({"role": "user", "content": prompt})
    full, dm = _engine.ask(prompt, ctx, max_tokens=1200)
    history.append({"role": "assistant", "content": full, "decision_maker": dm})
    st.session_state["devbloom_chat_inline"] = history[-MAX_HISTORY * 2:]
