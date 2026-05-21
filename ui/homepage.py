"""
Homepage — Morning Briefing Dashboard.
Four sections: Market Briefing bar, Summary cards, Today's Playbooks, Quick Nav grid.
"""
from __future__ import annotations

import json
import pathlib
import datetime

import streamlit as st

from ui.context import go_to, nav_button


# ── Helpers ───────────────────────────────────────────────────────────────────

def _market_status() -> str:
    """Return 'OPEN' or 'CLOSED' based on current IST time (weekday 09:15–15:30)."""
    now_utc = datetime.datetime.utcnow()
    # IST = UTC+5:30
    now_ist = now_utc + datetime.timedelta(hours=5, minutes=30)
    if now_ist.weekday() >= 5:
        return "CLOSED"
    market_open  = now_ist.replace(hour=9,  minute=15, second=0, microsecond=0)
    market_close = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
    if market_open <= now_ist <= market_close:
        return "OPEN"
    return "CLOSED"


# ── Section 1: Market Briefing Bar ───────────────────────────────────────────

def _render_briefing_bar() -> None:
    try:
        from ui.regime_bar import get_regime
        r = get_regime()
    except Exception:
        r = {
            "regime": "UNKNOWN",
            "vix": 0.0,
            "vix_state": "NORMAL",
            "nifty_change_pct": 0.0,
            "breadth": "NEUTRAL",
            "sector_leader": "N/A",
        }

    regime_raw = r.get("regime", "UNKNOWN")
    regime_label = regime_raw.replace("_", " ").title()
    vix = float(r.get("vix", 0.0) or 0.0)
    vix_state = r.get("vix_state", "NORMAL")
    status = _market_status()

    _REGIME_COLORS = {
        "BULL_TREND":   ("#00d4a0", "rgba(0,212,160,.12)"),
        "EXPANSION":    ("#00d4ff", "rgba(0,212,255,.10)"),
        "CHOPPY":       ("#f59e0b", "rgba(245,158,11,.12)"),
        "DISTRIBUTION": ("#fb923c", "rgba(251,146,60,.12)"),
        "BEAR":         ("#ff4b4b", "rgba(255,75,75,.12)"),
        "UNKNOWN":      ("#8892a4", "rgba(136,146,164,.08)"),
    }
    fg, bg = _REGIME_COLORS.get(regime_raw, ("#8892a4", "rgba(136,146,164,.08)"))

    vix_color = {"LOW": "#00d4a0", "NORMAL": "#f59e0b", "HIGH": "#ff4b4b"}.get(vix_state, "#8892a4")
    status_color = "#00d4a0" if status == "OPEN" else "#ff4b4b"
    status_dot   = "●" if status == "OPEN" else "○"

    st.markdown(
        f"<div style='background:{bg};border:1px solid {fg}33;border-left:4px solid {fg};"
        f"border-radius:10px;padding:.7rem 1.2rem;margin-bottom:1.2rem;"
        f"display:flex;align-items:center;gap:2rem;flex-wrap:wrap'>"
        f"<span style='color:{fg};font-size:.72rem;text-transform:uppercase;"
        f"letter-spacing:.12em;font-weight:700'>Regime</span>"
        f"<span style='color:#e8eaf0;font-size:.95rem;font-weight:700'>{regime_label}</span>"
        f"<span style='color:#4a5568;font-size:.8rem'>|</span>"
        f"<span style='color:#8892a4;font-size:.72rem;text-transform:uppercase;"
        f"letter-spacing:.08em'>India VIX</span>"
        f"<span style='color:{vix_color};font-size:.95rem;font-weight:700'>"
        f"{vix:.1f} <span style='font-size:.65rem;opacity:.7'>({vix_state})</span></span>"
        f"<span style='color:#4a5568;font-size:.8rem'>|</span>"
        f"<span style='color:{status_color};font-size:.8rem;font-weight:700'>"
        f"{status_dot} NSE {status}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Section 2: Three Summary Cards ───────────────────────────────────────────

def _card_top_opportunity() -> None:
    st.markdown(
        "<div style='font-size:.68rem;color:#8892a4;text-transform:uppercase;"
        "letter-spacing:.1em;margin-bottom:.5rem'>Top Opportunity</div>",
        unsafe_allow_html=True,
    )
    try:
        cached = st.session_state.get("iq2_pipeline_cache", [])
        if cached:
            top = cached[0]
            sym     = getattr(top, "symbol",     top.get("symbol",     "—") if isinstance(top, dict) else "—")
            pattern = getattr(top, "pattern",    top.get("pattern",    "Setup") if isinstance(top, dict) else "Setup")
            conf    = getattr(top, "confidence", top.get("confidence", 0.0)    if isinstance(top, dict) else 0.0)
            pattern_label = str(pattern).replace("_", " ").title()
            conf_pct = float(conf) * 100 if float(conf) <= 1.0 else float(conf)

            st.markdown(
                f"<div style='background:rgba(0,212,160,.06);border:1px solid rgba(0,212,160,.2);"
                f"border-radius:8px;padding:.7rem .9rem;margin-bottom:.6rem'>"
                f"<div style='color:#e8eaf0;font-size:1.1rem;font-weight:700;"
                f"font-family:JetBrains Mono,monospace'>{sym}</div>"
                f"<div style='color:#8892a4;font-size:.78rem;margin:.2rem 0'>{pattern_label}</div>"
                f"<div style='color:#00d4a0;font-size:.82rem;font-weight:600'>"
                f"Confidence: {conf_pct:.0f}%</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            nav_button("View →", "Institutional", symbol=sym, key="home_top_opp_view")
        else:
            st.info("No scan yet — go to Institutional tab to discover setups")
            nav_button("🏛️ Find Opportunities", "Institutional", key="home_to_inst")
    except Exception:
        st.info("Scan data unavailable")


def _card_portfolio_pulse() -> None:
    st.markdown(
        "<div style='font-size:.68rem;color:#8892a4;text-transform:uppercase;"
        "letter-spacing:.1em;margin-bottom:.5rem'>Portfolio Pulse</div>",
        unsafe_allow_html=True,
    )
    try:
        from ui.real_holdings import get_holdings_summary  # type: ignore[import]
        summary = get_holdings_summary()
        invested   = summary.get("total_invested", 0)
        daily_pnl  = summary.get("daily_pnl", 0)
        n_positions = summary.get("open_positions", 0)
        col_a, col_b = st.columns(2)
        col_a.metric("Invested", f"₹{invested:,.0f}")
        col_b.metric("Today's P&L", f"₹{daily_pnl:+,.0f}")
        st.caption(f"{n_positions} open position(s)")
    except Exception:
        p = pathlib.Path("logs/portfolio_state.json")
        if p.exists():
            try:
                data = json.loads(p.read_text())
                st.metric("Today's P&L", f"₹{data.get('daily_pnl', 0):+,.0f}")
            except Exception:
                st.info("No active positions")
        else:
            st.info("No active positions")
    nav_button("💼 View Holdings", "Holdings", key="home_to_holdings")


def _card_jarvis_quick_ask() -> None:
    st.markdown(
        "<div style='font-size:.68rem;color:#8892a4;text-transform:uppercase;"
        "letter-spacing:.1em;margin-bottom:.5rem'>JARVIS Quick Ask</div>",
        unsafe_allow_html=True,
    )
    q = st.text_input(
        "Ask JARVIS anything...",
        key="home_jarvis_q",
        placeholder="What's the best trade today?",
        label_visibility="collapsed",
    )
    if st.button("Ask →", key="home_ask_jarvis") and q:
        go_to("JARVIS", jarvis_query=q)


def _render_summary_cards() -> None:
    card1, card2, card3 = st.columns(3)
    with card1:
        with st.container(border=True):
            _card_top_opportunity()
    with card2:
        with st.container(border=True):
            _card_portfolio_pulse()
    with card3:
        with st.container(border=True):
            _card_jarvis_quick_ask()


# ── Section 3: Today's Playbooks ─────────────────────────────────────────────

def _render_playbooks() -> None:
    st.markdown(
        "<h3 style='color:#e8eaf0;font-size:1rem;margin:.8rem 0 .4rem'>"
        "Today's Playbooks</h3>",
        unsafe_allow_html=True,
    )
    try:
        from playbooks import get_playbooks_for_regime  # type: ignore[import]
        from ui.regime_bar import get_regime
        r = get_regime()
        regime     = r.get("regime", "UNKNOWN")
        vix_state  = r.get("vix_state", "NORMAL")
        breadth    = r.get("breadth", "NEUTRAL")
        playbooks  = get_playbooks_for_regime(regime, vix_state, breadth)[:3]

        if not playbooks:
            st.info("No regime-aligned playbooks found for current market conditions.")
            return

        pb_cols = st.columns(len(playbooks))
        for col, pb in zip(pb_cols, playbooks):
            with col:
                wr_pct = pb.baseline_win_rate * 100
                ev     = pb.baseline_expectancy
                ev_color = "#00d4a0" if ev > 0 else "#ff4b4b"
                st.markdown(
                    f"<div style='background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);"
                    f"border-radius:10px;padding:.8rem 1rem'>"
                    f"<div style='font-size:.95rem;font-weight:700;color:#e8eaf0;margin-bottom:.3rem'>"
                    f"{pb.emoji} {pb.name}</div>"
                    f"<div style='display:flex;gap:1rem;margin-bottom:.4rem'>"
                    f"<span style='font-size:.75rem;color:#8892a4'>Win Rate "
                    f"<span style='color:#00d4a0;font-weight:600'>{wr_pct:.0f}%</span></span>"
                    f"<span style='font-size:.75rem;color:#8892a4'>EV "
                    f"<span style='color:{ev_color};font-weight:600'>{ev:+.2f}</span></span>"
                    f"</div>"
                    f"<div style='font-size:.73rem;color:#8892a4;line-height:1.5;margin-bottom:.5rem'>"
                    f"{pb.notes[:120]}{'…' if len(pb.notes) > 120 else ''}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                nav_button("→ Find setups", "Institutional", key=f"home_pb_{pb.id}")
    except Exception:
        st.info("Playbook data unavailable for current regime.")


# ── Section 4: Quick Navigation Grid ─────────────────────────────────────────

_NAV_GRID = [
    ("🏛️ Institutional", "Institutional", "Find & trade setups"),
    ("🤖 JARVIS",        "JARVIS",        "Ask the AI anything"),
    ("⚡ Terminal",       "Terminal",      "Live charts & execution"),
    ("🔬 Research",       "Research",      "Deep dive a stock"),
    ("👁 My Watchlist",   "Watchlist",     "Your stock radar"),
    ("💼 My Holdings",    "Holdings",      "Your portfolio"),
    ("🧬 AlgoLab",        "AlgoLab",       "Test strategies"),
    ("🔍 Screener",       "Screener",      "Find stocks by fundamentals"),
]


def _render_quick_nav() -> None:
    st.markdown(
        "<h3 style='color:#e8eaf0;font-size:1rem;margin:.8rem 0 .4rem'>"
        "Quick Navigation</h3>",
        unsafe_allow_html=True,
    )
    rows = [_NAV_GRID[:4], _NAV_GRID[4:]]
    for row in rows:
        cols = st.columns(4)
        for col, (label, page, desc) in zip(cols, row):
            with col:
                st.markdown(
                    f"<div style='font-size:.68rem;color:#8892a4;margin-bottom:.2rem'>{desc}</div>",
                    unsafe_allow_html=True,
                )
                nav_button(label, page, key=f"home_nav_{page}", use_container_width=True)


# ── Main entry point ──────────────────────────────────────────────────────────

def render_homepage(universe: list[str]) -> None:
    # Section 1: Market Briefing Bar
    _render_briefing_bar()

    # Section 2: Three Summary Cards
    _render_summary_cards()

    st.divider()

    # Section 3: Today's Playbooks
    _render_playbooks()

    st.divider()

    # Section 4: Quick Navigation Grid
    _render_quick_nav()
