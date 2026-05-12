"""
Market Scanner UI — Momentum & Breakout panels.
Rendered on the homepage.
"""
from __future__ import annotations

import streamlit as st

_BREAKOUT_ICONS = {
    "52W_HIGH":        "🔴",
    "GOLDEN_CROSS":    "🟢",
    "VOL_SQUEEZE":     "🔵",
    "RESISTANCE_BREAK":"🟡",
    "CUP_HANDLE":      "⭐",
}
_SIGNAL_COLOR = {"BUY": "#00d4a0", "WATCH": "#f59e0b", "NEUTRAL": "#8892a4"}


@st.cache_data(ttl=300, show_spinner=False)
def _cached_momentum(symbols_key: str, top_n: int):
    from screener.momentum_scanner import MomentumScanner
    symbols = [s.strip() for s in symbols_key.split(",") if s.strip()]
    return MomentumScanner().scan_momentum(symbols, top_n=top_n)


@st.cache_data(ttl=300, show_spinner=False)
def _cached_breakouts(symbols_key: str, top_n: int):
    from screener.momentum_scanner import MomentumScanner
    symbols = [s.strip() for s in symbols_key.split(",") if s.strip()]
    return MomentumScanner().scan_breakouts(symbols, top_n=top_n)


def render_scanner(universe: list[str]) -> None:
    symbols_key = ",".join(universe)

    st.markdown(
        "<h2 style='color:#00d4ff;font-family:JetBrains Mono,monospace;"
        "font-size:1.3rem;letter-spacing:2px;margin-bottom:4px'>"
        "📡 MARKET SCANNER</h2>"
        "<p style='color:#4a5568;font-size:.72rem;margin-bottom:1rem'>"
        f"Scanning {len(universe)} stocks · auto-refresh every 5 min · no LLM cost</p>",
        unsafe_allow_html=True,
    )

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        scan_now = st.button("⚡ Scan Now", key="scanner_refresh", type="primary")
    with col_info:
        st.caption("Results cached 5 min · Click any row to open in Terminal")

    if scan_now:
        st.cache_data.clear()

    col_mom, col_brk = st.columns(2)

    # ── Momentum ──────────────────────────────────────────────────────────────
    with col_mom:
        st.markdown(
            "<div style='background:rgba(0,212,160,.06);border:1px solid rgba(0,212,160,.25);"
            "border-radius:10px;padding:14px 16px;margin-bottom:4px'>"
            "<span style='color:#00d4a0;font-size:.75rem;font-weight:700;"
            "letter-spacing:2px'>🚀 MOMENTUM STOCKS</span></div>",
            unsafe_allow_html=True,
        )
        with st.spinner("Scanning momentum..."):
            stocks = _cached_momentum(symbols_key, top_n=15)

        if not stocks:
            st.info("No momentum signals. Ensure historical data is available.")
        else:
            _render_momentum_table(stocks)

    # ── Breakout ──────────────────────────────────────────────────────────────
    with col_brk:
        st.markdown(
            "<div style='background:rgba(251,146,60,.06);border:1px solid rgba(251,146,60,.25);"
            "border-radius:10px;padding:14px 16px;margin-bottom:4px'>"
            "<span style='color:#f97316;font-size:.75rem;font-weight:700;"
            "letter-spacing:2px'>💥 BREAKOUT STOCKS</span></div>",
            unsafe_allow_html=True,
        )
        with st.spinner("Scanning breakouts..."):
            breakouts = _cached_breakouts(symbols_key, top_n=15)

        if not breakouts:
            st.info("No breakout setups detected right now.")
        else:
            _render_breakout_table(breakouts)


def _render_momentum_table(stocks) -> None:
    header = (
        "<div style='display:grid;grid-template-columns:80px 70px 60px 50px 60px 55px;"
        "gap:4px;padding:4px 8px;font-size:.6rem;color:#4a5568;font-weight:700;"
        "border-bottom:1px solid #1e2d4a;margin-bottom:4px'>"
        "<span>STOCK</span><span>PRICE</span><span>CHG%</span>"
        "<span>RSI</span><span>VOL✕</span><span>SIGNAL</span></div>"
    )
    st.markdown(header, unsafe_allow_html=True)

    for s in stocks:
        chg_color = "#00d4a0" if s.change_pct >= 0 else "#ff4b4b"
        sig_color = _SIGNAL_COLOR.get(s.signal, "#8892a4")
        bar_w = int(s.momentum_score * 0.8)  # max ~80px

        row = (
            f"<div style='display:grid;grid-template-columns:80px 70px 60px 50px 60px 55px;"
            f"gap:4px;padding:5px 8px;font-size:.7rem;font-family:JetBrains Mono,monospace;"
            f"border-bottom:1px solid rgba(255,255,255,.03);align-items:center'>"
            f"<span style='color:#e8eaf0;font-weight:600'>{s.symbol}</span>"
            f"<span style='color:#e8eaf0'>₹{s.price:,.0f}</span>"
            f"<span style='color:{chg_color}'>{s.change_pct:+.1f}%</span>"
            f"<span style='color:#8892a4'>{s.rsi:.0f}</span>"
            f"<span style='color:#8892a4'>{s.volume_ratio:.1f}x</span>"
            f"<span style='color:{sig_color};font-weight:700;font-size:.62rem'>{s.signal}</span>"
            f"</div>"
            f"<div style='height:3px;width:{bar_w}%;background:{sig_color}22;"
            f"border-radius:2px;margin:-2px 8px 2px'>"
            f"<div style='height:3px;width:{int(s.momentum_score)}%;background:{sig_color};"
            f"border-radius:2px'></div></div>"
        )
        st.markdown(row, unsafe_allow_html=True)


def _render_breakout_table(stocks) -> None:
    header = (
        "<div style='display:grid;grid-template-columns:80px 70px 120px 60px 65px;"
        "gap:4px;padding:4px 8px;font-size:.6rem;color:#4a5568;font-weight:700;"
        "border-bottom:1px solid #1e2d4a;margin-bottom:4px'>"
        "<span>STOCK</span><span>PRICE</span><span>BREAKOUT TYPE</span>"
        "<span>ATR</span><span>CONF%</span></div>"
    )
    st.markdown(header, unsafe_allow_html=True)

    for s in stocks:
        icon = _BREAKOUT_ICONS.get(s.breakout_type, "⚡")
        conf_color = "#00d4a0" if s.confidence >= 70 else ("#f59e0b" if s.confidence >= 55 else "#8892a4")
        label = s.breakout_type.replace("_", " ")

        row = (
            f"<div style='display:grid;grid-template-columns:80px 70px 120px 60px 65px;"
            f"gap:4px;padding:5px 8px;font-size:.7rem;font-family:JetBrains Mono,monospace;"
            f"border-bottom:1px solid rgba(255,255,255,.03);align-items:center'>"
            f"<span style='color:#e8eaf0;font-weight:600'>{s.symbol}</span>"
            f"<span style='color:#e8eaf0'>₹{s.price:,.0f}</span>"
            f"<span style='color:#f97316;font-size:.65rem'>{icon} {label}</span>"
            f"<span style='color:#8892a4'>{s.atr:.1f}</span>"
            f"<span style='color:{conf_color};font-weight:700'>{s.confidence:.0f}%</span>"
            f"</div>"
        )
        st.markdown(row, unsafe_allow_html=True)
