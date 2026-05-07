"""DevBloom AI Co-Pilot "Dev" — streaming chat sidebar with slash-command routing."""
from __future__ import annotations

import json
import os
from datetime import datetime

import streamlit as st

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


MODEL = "claude-haiku-4-5-20251001"
MAX_HISTORY = 20   # rolling context window for chat

SYSTEM_PROMPT = """You are Dev, the AI co-pilot embedded in DevBloom Terminal — a professional trading cockpit for NSE equity markets.

You have deep expertise in:
• Indian equity markets (NSE/BSE), Nifty sectors, FII/DII flows
• Technical analysis: price action, volume, indicators, multi-timeframe reads
• Fundamental analysis: P&L, balance sheet, DCF, earnings quality
• Macro: RBI policy, USD/INR, global risk-off/on dynamics
• Quant: backtesting, factor models, regime detection, signal validation

Style guidelines:
• Be concise and precise — use bullet points for multi-part answers
• Use market practitioner language (not textbook)
• When uncertain, say so and suggest where to look
• For /why commands: macro events → sector impact → stock impact → key chart level
• For /model commands: quick DCF setup with assumptions stated clearly
• For /idea commands: give entry, stop, target, time horizon, and one-line thesis

Slash commands you understand:
/why [symbol or index]  — explain recent move with data
/model [symbol]         — sketch a quick DCF or valuation framework
/idea [symbol]          — generate a trade idea
/scan [criteria]        — suggest how to scan for a setup
/explain [indicator]    — explain an indicator in plain language
/compare [sym1] [sym2]  — compare two stocks
"""


def _get_client():
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or not HAS_ANTHROPIC:
        return None
    return anthropic.Anthropic(api_key=api_key)


def _enrich_prompt(user_msg: str, context: dict) -> str:
    """Append live portfolio/market context to the user message."""
    parts = [user_msg]
    if context.get("symbol"):
        parts.append(f"\n[Active symbol: {context['symbol']}]")
    if context.get("price"):
        parts.append(f"[Last price: ₹{context['price']:,.2f}]")
    if context.get("indicators"):
        ind = context["indicators"]
        parts.append(
            f"[Indicators: RSI={ind.get('rsi_14',50):.1f}, "
            f"Z-score={ind.get('zscore_20',0):.2f}, "
            f"Momentum={ind.get('momentum_5d_pct',0):.2f}%]"
        )
    return "".join(parts)


def _stream_response(client, messages: list[dict]) -> str:
    full = ""
    placeholder = st.empty()
    with client.messages.stream(
        model=MODEL,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        for text in stream.text_stream:
            full += text
            placeholder.markdown(
                f"<div style='color:#e8eaf0;font-size:.88rem;line-height:1.6'>{full}▍</div>",
                unsafe_allow_html=True,
            )
    placeholder.markdown(
        f"<div style='color:#e8eaf0;font-size:.88rem;line-height:1.6'>{full}</div>",
        unsafe_allow_html=True,
    )
    return full


def _fallback_response(user_msg: str) -> str:
    return (
        "ANTHROPIC_API_KEY not set — Dev is offline.\n\n"
        "Set it in your `.env` file:\n```\nANTHROPIC_API_KEY=sk-ant-...\n```"
    )


def render_copilot_sidebar(context: dict | None = None):
    """Render the Dev co-pilot chat panel in the sidebar."""
    ctx = context or {}

    if "devbloom_chat" not in st.session_state:
        st.session_state["devbloom_chat"] = []

    st.sidebar.markdown(
        "<div style='padding:.5rem 0;'>"
        "<span style='font-size:.65rem;color:#8892a4;text-transform:uppercase;letter-spacing:.06em'>AI Co-Pilot</span><br>"
        "<span style='font-size:1.1rem;color:#00d4ff;font-weight:700'>⚡ Dev</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    chat_history = st.session_state["devbloom_chat"]

    # Render chat history
    chat_box = st.sidebar.container()
    with chat_box:
        for msg in chat_history[-12:]:
            if msg["role"] == "user":
                st.markdown(
                    f"<div style='background:rgba(0,212,255,.08);border-left:2px solid #00d4ff;"
                    f"border-radius:0 8px 8px 0;padding:.4rem .7rem;margin:.2rem 0;"
                    f"font-size:.8rem;color:#e8eaf0'>{msg['content']}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='background:rgba(255,255,255,.04);border-left:2px solid #8892a4;"
                    f"border-radius:0 8px 8px 0;padding:.4rem .7rem;margin:.2rem 0;"
                    f"font-size:.8rem;color:#c8cfe0'>{msg['content']}</div>",
                    unsafe_allow_html=True,
                )

    user_input = st.sidebar.text_input(
        "Ask Dev…",
        placeholder="/why RELIANCE  or  /idea INFY  or  any question",
        key="copilot_input",
        label_visibility="collapsed",
    )
    send = st.sidebar.button("Send ↵", key="copilot_send", use_container_width=True)

    if (send or user_input.endswith("\n")) and user_input.strip():
        enriched = _enrich_prompt(user_input.strip(), ctx)
        chat_history.append({"role": "user", "content": user_input.strip()})

        client = _get_client()
        with st.sidebar:
            with st.spinner("Dev is thinking…"):
                if client:
                    api_msgs = [{"role": m["role"], "content": m["content"]} for m in chat_history[-MAX_HISTORY:]]
                    # Use last msg as the enriched one
                    api_msgs[-1]["content"] = enriched
                    try:
                        reply = _stream_response(client, api_msgs)
                    except Exception as e:
                        reply = f"Error: {e}"
                else:
                    reply = _fallback_response(user_input)

        chat_history.append({"role": "assistant", "content": reply})
        st.session_state["devbloom_chat"] = chat_history[-MAX_HISTORY * 2:]
        st.rerun()

    if st.sidebar.button("Clear chat", key="copilot_clear"):
        st.session_state["devbloom_chat"] = []
        st.rerun()


def render_copilot_inline(context: dict | None = None):
    """Render Dev co-pilot as a full inline chat panel (for the Co-Pilot tab)."""
    ctx = context or {}
    client = _get_client()

    st.markdown("### ⚡ Dev — Your AI Co-Pilot")
    st.caption("Ask anything about markets, charts, or your portfolio. Use /why, /model, /idea, /explain slash commands.")

    if "devbloom_chat_inline" not in st.session_state:
        st.session_state["devbloom_chat_inline"] = []

    history = st.session_state["devbloom_chat_inline"]

    # Morning ideas button
    if st.button("☀️ Generate Morning Trade Ideas", key="morning_ideas"):
        prompt = (
            "/idea Scan NSE 500 for the top 5 swing trade setups today. "
            "For each: entry, stop, target, R:R, 1-line thesis. "
            "Focus on momentum + volume breakouts. Current date: " + datetime.now().strftime("%Y-%m-%d")
        )
        history.append({"role": "user", "content": prompt})
        if client:
            try:
                api_msgs = [{"role": m["role"], "content": m["content"]} for m in history[-MAX_HISTORY:]]
                reply = ""
                with client.messages.stream(model=MODEL, max_tokens=1200, system=SYSTEM_PROMPT, messages=api_msgs) as s:
                    placeholder = st.empty()
                    for text in s.text_stream:
                        reply += text
                        placeholder.markdown(reply + "▍")
                    placeholder.markdown(reply)
            except Exception as e:
                reply = f"Error: {e}"
        else:
            reply = _fallback_response(prompt)
        history.append({"role": "assistant", "content": reply})
        st.session_state["devbloom_chat_inline"] = history[-MAX_HISTORY * 2:]

    st.divider()

    # Chat history
    for msg in history[-16:]:
        role_label = "You" if msg["role"] == "user" else "⚡ Dev"
        role_color = "#00d4ff" if msg["role"] == "user" else "#ffb800"
        st.markdown(
            f"<div style='margin:.4rem 0'>"
            f"<span style='color:{role_color};font-size:.72rem;font-weight:600;text-transform:uppercase'>{role_label}</span><br>"
            f"<span style='color:#e8eaf0;font-size:.88rem;line-height:1.7'>{msg['content']}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Input
    col1, col2 = st.columns([5, 1])
    user_input = col1.text_input("Message", placeholder="Type /why RELIANCE or ask anything…",
                                 label_visibility="collapsed", key="copilot_inline_input")
    send       = col2.button("Send", key="copilot_inline_send", use_container_width=True)

    if send and user_input.strip():
        enriched = _enrich_prompt(user_input.strip(), ctx)
        history.append({"role": "user", "content": user_input.strip()})
        if client:
            try:
                api_msgs = [{"role": m["role"], "content": m["content"]} for m in history[-MAX_HISTORY:]]
                api_msgs[-1]["content"] = enriched
                with st.spinner("Dev is thinking…"):
                    reply = ""
                    with client.messages.stream(model=MODEL, max_tokens=1000, system=SYSTEM_PROMPT, messages=api_msgs) as s:
                        placeholder = st.empty()
                        for text in s.text_stream:
                            reply += text
                            placeholder.markdown(reply + "▍")
                        placeholder.markdown(reply)
            except Exception as e:
                reply = f"Error: {e}"
        else:
            reply = _fallback_response(user_input)
        history.append({"role": "assistant", "content": reply})
        st.session_state["devbloom_chat_inline"] = history[-MAX_HISTORY * 2:]
        st.rerun()

    if st.button("🗑 Clear", key="copilot_inline_clear"):
        st.session_state["devbloom_chat_inline"] = []
        st.rerun()
