"""
Market Scanner UI — Momentum & Breakout panels.

Two modes:
  · Quick Scan  — your configured universe (instant, ~10-30 stocks)
  · Full NSE    — all ~1900 NSE EQ symbols scanned on-click (takes 2-4 min)

Rendered on the Dashboard homepage.
"""
from __future__ import annotations

import streamlit as st

_BREAKOUT_ICONS = {
    "52W_HIGH":         "🔴",
    "GOLDEN_CROSS":     "🟢",
    "VOL_SQUEEZE":      "🔵",
    "RESISTANCE_BREAK": "🟡",
    "CUP_HANDLE":       "⭐",
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


@st.cache_data(ttl=86_400, show_spinner=False)
def _get_full_universe() -> list[str]:
    """Load full NSE EQ universe (~1900 symbols). Cached 24h."""
    try:
        from screener.universe import StockUniverseFetcher
        return StockUniverseFetcher().get_all_symbols()
    except Exception:
        return []


def render_scanner(universe: list[str]) -> None:
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        "<h2 style='color:#00d4ff;font-family:JetBrains Mono,monospace;"
        "font-size:1.3rem;letter-spacing:2px;margin-bottom:2px'>"
        "📡 MARKET SCANNER</h2>",
        unsafe_allow_html=True,
    )

    # ── Mode selector + buttons ───────────────────────────────────────────────
    _c1, _c2, _c3, _c4 = st.columns([2, 1, 1, 3])
    with _c1:
        scan_mode = st.radio(
            "Scan mode",
            ["My Universe", "Full NSE (~1900 stocks)"],
            horizontal=True,
            key="scanner_mode",
            label_visibility="collapsed",
        )
    with _c2:
        scan_quick = st.button(
            "⚡ Quick Scan", key="scanner_quick",
            type="primary", use_container_width=True,
        )
    with _c3:
        scan_full = st.button(
            "🌐 Full NSE Scan", key="scanner_full",
            use_container_width=True,
        )
    with _c4:
        if scan_mode == "Full NSE (~1900 stocks)":
            st.caption("⏱ Full scan: ~3-4 min · parallel 16 workers · results cached 5 min")
        else:
            st.caption(f"Scanning {len(universe)} stocks from your universe · cached 5 min")

    # ── Resolve symbol list ────────────────────────────────────────────────────
    if scan_full or scan_mode == "Full NSE (~1900 stocks)":
        with st.spinner("Loading NSE symbol list…"):
            all_nse = _get_full_universe()
        if not all_nse:
            st.warning("Could not load NSE universe. Using your configured universe.")
            symbols = universe
        else:
            symbols = all_nse
            st.caption(
                f"<span style='color:#00d4ff;font-size:.72rem'>✅ {len(symbols)} NSE symbols loaded</span>",
                unsafe_allow_html=True,
            )
    else:
        symbols = universe

    symbols_key = ",".join(symbols)

    # Clear cache on manual scan trigger
    if scan_quick or scan_full:
        st.cache_data.clear()

    # ── Show progress for full scan ───────────────────────────────────────────
    if len(symbols) > 100:
        st.info(
            f"🔄 Scanning **{len(symbols)} stocks** in parallel (16 workers). "
            "This may take 3-4 minutes. Results appear when complete.",
            icon="⏱",
        )

    # ── Two-column results ────────────────────────────────────────────────────
    col_mom, col_brk = st.columns(2)

    with col_mom:
        st.markdown(
            "<div style='background:rgba(0,212,160,.06);border:1px solid rgba(0,212,160,.25);"
            "border-radius:10px;padding:12px 16px;margin-bottom:8px'>"
            "<span style='color:#00d4a0;font-size:.75rem;font-weight:700;"
            "letter-spacing:2px'>🚀 TOP MOMENTUM</span>"
            "<span style='color:#4a5568;font-size:.65rem;margin-left:.5rem'>top 20 by composite score</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        with st.spinner(f"Scanning momentum across {len(symbols)} stocks…"):
            momentum_stocks = _cached_momentum(symbols_key, top_n=20)

        if not momentum_stocks:
            st.markdown(
                "<div style='text-align:center;padding:2rem;color:#4a5568;font-size:.82rem'>"
                "No momentum signals found.<br><span style='font-size:.7rem'>"
                "Try Full NSE Scan for more results.</span></div>",
                unsafe_allow_html=True,
            )
        else:
            _render_momentum_table(momentum_stocks)

    with col_brk:
        st.markdown(
            "<div style='background:rgba(251,146,60,.06);border:1px solid rgba(251,146,60,.25);"
            "border-radius:10px;padding:12px 16px;margin-bottom:8px'>"
            "<span style='color:#f97316;font-size:.75rem;font-weight:700;"
            "letter-spacing:2px'>💥 BREAKOUTS</span>"
            "<span style='color:#4a5568;font-size:.65rem;margin-left:.5rem'>top 20 by confidence</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        with st.spinner(f"Scanning breakouts across {len(symbols)} stocks…"):
            breakout_stocks = _cached_breakouts(symbols_key, top_n=20)

        if not breakout_stocks:
            st.markdown(
                "<div style='text-align:center;padding:2rem;color:#4a5568;font-size:.82rem'>"
                "No breakout setups detected.<br><span style='font-size:.7rem'>"
                "Try Full NSE Scan for more results.</span></div>",
                unsafe_allow_html=True,
            )
        else:
            _render_breakout_table(breakout_stocks)


def _render_momentum_table(stocks) -> None:
    st.markdown(
        "<div style='display:grid;grid-template-columns:90px 75px 60px 50px 55px 55px;"
        "gap:4px;padding:4px 8px;font-size:.6rem;color:#4a5568;font-weight:700;"
        "text-transform:uppercase;letter-spacing:.05em;"
        "border-bottom:1px solid rgba(255,255,255,.06);margin-bottom:4px'>"
        "<span>Stock</span><span>Price</span><span>Chg%</span>"
        "<span>RSI</span><span>Vol✕</span><span>Signal</span></div>",
        unsafe_allow_html=True,
    )
    for s in stocks:
        chg_color = "#00d4a0" if s.change_pct >= 0 else "#ff4b4b"
        sig_color = _SIGNAL_COLOR.get(s.signal, "#8892a4")
        bar_pct   = min(int(s.momentum_score), 100)
        st.markdown(
            f"<div style='display:grid;grid-template-columns:90px 75px 60px 50px 55px 55px;"
            f"gap:4px;padding:5px 8px;font-size:.72rem;font-family:JetBrains Mono,monospace;"
            f"border-bottom:1px solid rgba(255,255,255,.03);align-items:center'>"
            f"<span style='color:#e8eaf0;font-weight:700'>{s.symbol}</span>"
            f"<span style='color:#c9d1e0'>₹{s.price:,.0f}</span>"
            f"<span style='color:{chg_color}'>{s.change_pct:+.1f}%</span>"
            f"<span style='color:#8892a4'>{s.rsi:.0f}</span>"
            f"<span style='color:#8892a4'>{s.volume_ratio:.1f}x</span>"
            f"<span style='color:{sig_color};font-weight:700;font-size:.65rem'>{s.signal}</span>"
            f"</div>"
            f"<div style='height:2px;margin:0 8px 3px;background:rgba(255,255,255,.04);border-radius:1px'>"
            f"<div style='height:2px;width:{bar_pct}%;background:{sig_color}88;border-radius:1px'></div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _render_breakout_table(stocks) -> None:
    st.markdown(
        "<div style='display:grid;grid-template-columns:90px 75px 130px 55px 60px;"
        "gap:4px;padding:4px 8px;font-size:.6rem;color:#4a5568;font-weight:700;"
        "text-transform:uppercase;letter-spacing:.05em;"
        "border-bottom:1px solid rgba(255,255,255,.06);margin-bottom:4px'>"
        "<span>Stock</span><span>Price</span><span>Pattern</span>"
        "<span>ATR</span><span>Conf%</span></div>",
        unsafe_allow_html=True,
    )
    for s in stocks:
        icon       = _BREAKOUT_ICONS.get(s.breakout_type, "⚡")
        label      = s.breakout_type.replace("_", " ")
        conf_color = (
            "#00d4a0" if s.confidence >= 70
            else ("#f59e0b" if s.confidence >= 55 else "#8892a4")
        )
        st.markdown(
            f"<div style='display:grid;grid-template-columns:90px 75px 130px 55px 60px;"
            f"gap:4px;padding:5px 8px;font-size:.72rem;font-family:JetBrains Mono,monospace;"
            f"border-bottom:1px solid rgba(255,255,255,.03);align-items:center'>"
            f"<span style='color:#e8eaf0;font-weight:700'>{s.symbol}</span>"
            f"<span style='color:#c9d1e0'>₹{s.price:,.0f}</span>"
            f"<span style='color:#f97316;font-size:.67rem'>{icon} {label}</span>"
            f"<span style='color:#8892a4'>{s.atr:.1f}</span>"
            f"<span style='color:{conf_color};font-weight:700'>{s.confidence:.0f}%</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
