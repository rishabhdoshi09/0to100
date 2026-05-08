"""Nested watchlist with sparklines — uses Kite if token is set, yfinance fallback."""
from __future__ import annotations

import streamlit as st

from data.market_data import get_provider

DEFAULT_WATCHLIST = {
    "Momentum":  ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"],
    "Reversal":  ["BHARTIARTL", "WIPRO", "AXISBANK", "SBIN"],
    "Breakout":  ["LT", "BAJFINANCE", "MARUTI", "TITAN"],
}


def _mini_sparkline_svg(closes: list[float], width=60, height=22) -> str:
    if len(closes) < 2:
        return ""
    mn, mx = min(closes), max(closes)
    rng = mx - mn or 1
    xs = [int(i / (len(closes) - 1) * width) for i in range(len(closes))]
    ys = [int((1 - (c - mn) / rng) * height) for c in closes]
    pts = " ".join(f"{x},{y}" for x, y in zip(xs, ys))
    color = "#00ff88" if closes[-1] >= closes[0] else "#ff4466"
    return (
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5" stroke-linecap="round"/>'
        f"</svg>"
    )


@st.cache_data(ttl=60)
def _fetch_group(symbols: tuple[str, ...]) -> dict[str, dict]:
    """Fetch quotes for a group of symbols (cached 60 s)."""
    mdp = get_provider()
    quotes = mdp.quotes(list(symbols))
    # Add sparkline closes per symbol
    for sym in symbols:
        quotes[sym]["closes"] = mdp.history_closes(sym, days=15)
    return quotes


def render_watchlist(watchlist: dict[str, list[str]] | None = None):
    wl = watchlist or DEFAULT_WATCHLIST

    for group, symbols in wl.items():
        with st.expander(f"📂 {group}", expanded=True):
            header_cols = st.columns([2, 1.5, 1.5, 1.5, 1])
            for col, label in zip(header_cols, ["Symbol", "Price", "Chg%", "Vol Spike", "15d"]):
                col.markdown(
                    f"<span style='color:#8892a4;font-size:.7rem;text-transform:uppercase'>{label}</span>",
                    unsafe_allow_html=True,
                )

            rows = _fetch_group(tuple(symbols))

            for sym in symbols:
                row = rows.get(sym, {})
                price     = row.get("price", 0)
                chg       = row.get("chg_pct", 0)
                vol       = row.get("volume", 0)
                avg_v     = row.get("avg_volume", 1) or 1
                closes    = row.get("closes", [])
                vol_spike = vol / avg_v if avg_v and vol else 0.0

                chg_color   = "#00ff88" if chg >= 0 else "#ff4466"
                spike_color = "#ffb800" if vol_spike > 2 else "#8892a4"
                svg         = _mini_sparkline_svg(closes)

                cols = st.columns([2, 1.5, 1.5, 1.5, 1])
                cols[0].markdown(f"**{sym}**")
                cols[1].markdown(
                    f"<span style='font-family:JetBrains Mono,monospace;font-size:.9rem'>"
                    f"₹{price:,.1f}</span>",
                    unsafe_allow_html=True,
                )
                cols[2].markdown(
                    f"<span style='color:{chg_color};font-family:JetBrains Mono,monospace;font-size:.9rem'>"
                    f"{'+'if chg>=0 else ''}{chg:.2f}%</span>",
                    unsafe_allow_html=True,
                )
                cols[3].markdown(
                    f"<span style='color:{spike_color};font-family:JetBrains Mono,monospace;font-size:.88rem'>"
                    f"{vol_spike:.1f}x</span>",
                    unsafe_allow_html=True,
                )
                cols[4].markdown(svg, unsafe_allow_html=True)
