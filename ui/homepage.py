"""
Homepage — Market Scanner + User Watchlist.
Default landing page when app loads.
"""
from __future__ import annotations

import streamlit as st

from ui.scanner import render_scanner


_SIGNAL_COLOR = {"BUY": "#00d4a0", "SELL": "#ff4b4b", "HOLD": "#8892a4", "WATCH": "#f59e0b"}
_SIGNAL_BG    = {"BUY": "rgba(0,212,160,.08)", "SELL": "rgba(255,75,75,.08)",
                 "HOLD": "rgba(136,146,164,.06)", "WATCH": "rgba(245,158,11,.08)"}


def render_homepage(universe: list[str]) -> None:
    """
    Full homepage render:
      1. Market Scanner (momentum + breakout)
      2. User watchlist (user-managed, stored in session state)
    """
    # ── Scanner ───────────────────────────────────────────────────────────────
    render_scanner(universe)

    st.divider()

    # ── Watchlist ──────────────────────────────────────────────────────────────
    st.markdown(
        "<h2 style='color:#00d4ff;font-family:JetBrains Mono,monospace;"
        "font-size:1.3rem;letter-spacing:2px;margin-bottom:4px'>"
        "📋 MY WATCHLIST</h2>"
        "<p style='color:#4a5568;font-size:.72rem;margin-bottom:1rem'>"
        "Add any NSE symbol · signals computed on-demand</p>",
        unsafe_allow_html=True,
    )

    _render_watchlist_controls()
    _render_watchlist_cards()


# ── Watchlist controls ────────────────────────────────────────────────────────

def _render_watchlist_controls() -> None:
    if "watchlist" not in st.session_state:
        st.session_state["watchlist"] = []

    col_input, col_add = st.columns([4, 1])
    with col_input:
        new_sym = st.text_input(
            "Add stock",
            key="wl_add_input",
            placeholder="Type NSE symbol e.g. RELIANCE",
            label_visibility="collapsed",
        ).strip().upper()
    with col_add:
        if st.button("＋ Add", key="wl_add_btn", type="primary"):
            if new_sym and new_sym not in st.session_state["watchlist"]:
                st.session_state["watchlist"].append(new_sym)
                st.rerun()


def _render_watchlist_cards() -> None:
    wl: list[str] = st.session_state.get("watchlist", [])

    if not wl:
        st.markdown(
            "<div style='text-align:center;padding:40px;color:#4a5568;"
            "font-family:JetBrains Mono,monospace;font-size:.8rem'>"
            "🔍 Your watchlist is empty.<br>"
            "<span style='font-size:.7rem'>Search and add stocks above</span></div>",
            unsafe_allow_html=True,
        )
        return

    # Fetch live prices for all watchlist symbols
    prices = _get_prices_batch(wl)

    cols = st.columns(min(len(wl), 4))
    to_remove = []

    for idx, sym in enumerate(wl):
        col = cols[idx % 4]
        with col:
            price_data = prices.get(sym, {})
            price     = price_data.get("price", 0.0)
            change    = price_data.get("change_pct", 0.0)
            signal    = price_data.get("signal", "HOLD")

            sig_color = _SIGNAL_COLOR.get(signal, "#8892a4")
            sig_bg    = _SIGNAL_BG.get(signal, "rgba(136,146,164,.06)")
            chg_color = "#00d4a0" if change >= 0 else "#ff4b4b"
            chg_arrow = "▲" if change >= 0 else "▼"

            st.markdown(
                f"<div style='background:{sig_bg};border:1px solid {sig_color}33;"
                f"border-left:3px solid {sig_color};border-radius:10px;"
                f"padding:14px 16px;margin-bottom:8px;position:relative'>"
                f"<div style='display:flex;justify-content:space-between;align-items:start'>"
                f"<span style='color:#e8eaf0;font-size:.85rem;font-weight:700;"
                f"font-family:JetBrains Mono,monospace'>{sym}</span>"
                f"<span style='color:{sig_color};font-size:.62rem;font-weight:700;"
                f"background:rgba(255,255,255,.05);padding:2px 6px;border-radius:4px;"
                f"border:1px solid {sig_color}44'>{signal}</span></div>"
                f"<div style='margin-top:8px'>"
                f"<span style='color:#e8eaf0;font-size:1.1rem;font-weight:700;"
                f"font-family:JetBrains Mono,monospace'>₹{price:,.2f}</span>"
                f"<span style='color:{chg_color};font-size:.72rem;margin-left:8px'>"
                f"{chg_arrow} {abs(change):.1f}%</span></div></div>",
                unsafe_allow_html=True,
            )

            col_run, col_rm = st.columns([3, 1])
            with col_run:
                if st.button("Run Analysis", key=f"wl_run_{sym}", use_container_width=True):
                    st.session_state["terminal_symbol"] = sym
                    st.session_state["active_tab"] = "terminal"
                    st.rerun()
            with col_rm:
                if st.button("✕", key=f"wl_rm_{sym}", use_container_width=True):
                    to_remove.append(sym)

    for sym in to_remove:
        st.session_state["watchlist"].remove(sym)
    if to_remove:
        st.rerun()


# ── Price fetcher ─────────────────────────────────────────────────────────────

def _get_prices_batch(symbols: list[str]) -> dict:
    """Fetch live LTP for all symbols. Falls back to yfinance if Kite not available."""
    result: dict[str, dict] = {sym: {"price": 0.0, "change_pct": 0.0, "signal": "HOLD"} for sym in symbols}

    try:
        from data.kite_client import KiteClient
        kite = KiteClient()
        if kite.is_connected():
            from data.instruments import InstrumentManager
            mgr = InstrumentManager(kite)
            nse_syms = [f"NSE:{s}" for s in symbols]
            quotes = kite.kite.quote(nse_syms)
            for sym in symbols:
                key = f"NSE:{sym}"
                if key in quotes:
                    q = quotes[key]
                    price = q.get("last_price", 0.0)
                    close = q.get("ohlc", {}).get("close", price)
                    change = ((price / close) - 1) * 100 if close else 0.0
                    result[sym]["price"] = price
                    result[sym]["change_pct"] = round(change, 2)
            return result
    except Exception:
        pass

    # yfinance fallback
    try:
        import yfinance as yf
        for sym in symbols:
            try:
                ticker = yf.Ticker(f"{sym}.NS")
                info = ticker.fast_info
                price = float(getattr(info, "last_price", 0) or 0)
                prev  = float(getattr(info, "previous_close", price) or price)
                change = ((price / prev) - 1) * 100 if prev else 0.0
                result[sym]["price"] = round(price, 2)
                result[sym]["change_pct"] = round(change, 2)
            except Exception:
                pass
    except Exception:
        pass

    return result
