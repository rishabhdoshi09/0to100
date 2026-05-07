"""
DevBloom AI Co-Pilot "Dev" — dual-LLM streaming chat (DeepSeek → Claude).

Both the sidebar mini-copilot and the full inline tab share a single
st.session_state key ("devbloom_chat") via DualLLMService, so a question
asked in the sidebar is visible when the user opens the Co-Pilot tab and
vice versa.

Pipeline:
  Stage 1 · DeepSeek  — fast first-pass analysis (shown as "draft" until
                         Claude completes)
  Stage 2 · Claude    — validates, refines, or overrides; labels response
                         [VALIDATED] / [OVERRIDE]

The UI shows a decision badge on every assistant message. When Claude
overrides DeepSeek, the badge gains a tooltip showing what DeepSeek had said.
"""
from __future__ import annotations

from datetime import datetime

import streamlit as st

from ai.dual_llm_service import get_service

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


def _render_assistant_msg(content: str, decision_maker: str, detail: str = "", ts: str = "", sidebar: bool = False):
    svc = get_service()
    badge = svc.badge(decision_maker, detail, ts=ts)
    if sidebar:
        st.sidebar.markdown(
            f"<div style='background:rgba(255,255,255,.04);border-left:2px solid #8892a4;"
            f"border-radius:0 8px 8px 0;padding:.4rem .7rem;margin:.2rem 0;"
            f"font-size:.78rem;color:#c8cfe0'>{badge}<br>{content}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='margin:.3rem 0'>"
            f"<span style='color:#ffb800;font-size:.7rem;font-weight:600;text-transform:uppercase'>⚡ Dev</span> "
            f"{badge}<br>"
            f"<span style='color:#e8eaf0;font-size:.88rem;line-height:1.7'>{content}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_copilot_sidebar(context: dict | None = None):
    """Dev co-pilot as a persistent sidebar panel — shares history with the main tab."""
    ctx = context or {}
    svc = get_service()
    history = svc.get_history()

    st.sidebar.markdown(
        "<div style='padding:.5rem 0 .25rem'>"
        "<span style='font-size:.6rem;color:#8892a4;text-transform:uppercase;letter-spacing:.06em'>AI Co-Pilot</span><br>"
        "<span style='font-size:1.1rem;color:#00d4ff;font-weight:700'>⚡ Dev</span>"
        "<span style='font-size:.6rem;color:#8892a4;margin-left:.5rem'>DeepSeek → Claude</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    for msg in history[-10:]:
        if msg["role"] == "user":
            st.sidebar.markdown(
                f"<div style='background:rgba(0,212,255,.08);border-left:2px solid #00d4ff;"
                f"border-radius:0 8px 8px 0;padding:.4rem .7rem;margin:.2rem 0;"
                f"font-size:.78rem;color:#e8eaf0'>{msg['content']}</div>",
                unsafe_allow_html=True,
            )
        else:
            _render_assistant_msg(
                msg["content"],
                msg.get("decision_maker", "claude_validated"),
                msg.get("detail", ""),
                ts=msg.get("ts", ""),
                sidebar=True,
            )

    user_input = st.sidebar.text_input(
        "Ask Dev…",
        placeholder="/why RELIANCE  /idea INFY  or anything",
        key="copilot_sidebar_input",
        label_visibility="collapsed",
    )
    send = st.sidebar.button("Send ↵", key="copilot_sidebar_send", use_container_width=True)

    if send and user_input.strip():
        svc.append_user(user_input.strip())
        with st.sidebar:
            placeholder = st.empty()
            full, dm, detail = "", "claude_validated", ""
            for chunk, dm, detail in svc.stream(user_input.strip(), ctx):
                full += chunk
                placeholder.markdown(
                    f"<div style='font-size:.78rem;color:#8892a4;font-style:italic'>{full}▍</div>",
                    unsafe_allow_html=True,
                )
            placeholder.empty()
        svc.append_assistant(full, dm, detail)
        st.rerun()

    if st.sidebar.button("Clear", key="copilot_sidebar_clear"):
        svc.clear_history()
        st.rerun()


def render_copilot_inline(context: dict | None = None):
    """Full inline Co-Pilot tab — dual-LLM streaming chat, shares history with sidebar."""
    ctx = context or {}
    svc = get_service()
    history = svc.get_history()

    st.markdown("### ⚡ Dev — AI Co-Pilot")
    st.caption(
        "Powered by **DeepSeek → Claude** dual-LLM pipeline. "
        "DeepSeek generates the first-pass draft; Claude validates or overrides. "
        "Hover a **⚡ Claude Override** badge to see what DeepSeek originally said. "
        "Use `/why`, `/model`, `/idea`, `/compare` slash commands."
    )

    if st.button("☀️ Generate Morning Trade Ideas  (DeepSeek → Claude)", key="morning_ideas"):
        _fire_and_append(svc, _morning_prompt(), ctx)
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
            _render_assistant_msg(
                msg["content"],
                msg.get("decision_maker", "claude_validated"),
                msg.get("detail", ""),
                ts=msg.get("ts", ""),
                sidebar=False,
            )

    col1, col2 = st.columns([5, 1])
    user_input = col1.text_input(
        "Message", placeholder="Type /why RELIANCE or ask anything…",
        label_visibility="collapsed", key="copilot_inline_input",
    )
    send = col2.button("Send", key="copilot_inline_send", use_container_width=True)

    if send and user_input.strip():
        svc.append_user(user_input.strip())

        # Two-phase streaming display:
        # Phase 1 (DeepSeek drafting) — muted italic ghost text
        # Phase 2 (Claude label arrives) — full-brightness final text
        st.markdown(
            "<span style='color:#8892a4;font-size:.7rem;font-weight:600'>⚡ Dev (DeepSeek drafting…)</span>",
            unsafe_allow_html=True,
        )
        placeholder = st.empty()
        full, dm, detail = "", "claude_validated", ""
        claude_started = False

        for chunk, dm, detail in svc.stream(user_input.strip(), ctx):
            full += chunk
            if not claude_started and ("[VALIDATED]" in full or "[OVERRIDE]" in full):
                claude_started = True
            if claude_started:
                placeholder.markdown(
                    f"<div style='color:#e8eaf0;font-size:.88rem;line-height:1.7'>{full}▍</div>",
                    unsafe_allow_html=True,
                )
            else:
                placeholder.markdown(
                    f"<div style='color:#8892a4;font-size:.85rem;line-height:1.7;font-style:italic'>{full}▍</div>",
                    unsafe_allow_html=True,
                )

        placeholder.markdown(
            f"<div style='color:#e8eaf0;font-size:.88rem;line-height:1.7'>{full}</div>",
            unsafe_allow_html=True,
        )
        svc.append_assistant(full, dm, detail)
        st.rerun()

    if st.button("🗑 Clear chat", key="copilot_inline_clear"):
        svc.clear_history()
        st.rerun()


def _fire_and_append(svc, prompt: str, ctx: dict):
    """Run a sync dual-LLM query and append to shared history (used by buttons)."""
    svc.append_user(prompt)
    text, dm, detail = svc.ask(prompt, ctx, max_tokens=1200)
    svc.append_assistant(text, dm, detail)
