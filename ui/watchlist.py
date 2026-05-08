"""Nested watchlist with sparklines and signal badges."""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

DEFAULT_WATCHLIST = {
    "Momentum":  ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"],
    "Reversal":  ["BHARTIARTL", "WIPRO", "AXISBANK", "SBIN"],
    "Breakout":  ["LT", "BAJFINANCE", "TATAMOTORS", "MARUTI"],
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


@st.cache_data(ttl=300)
def _fetch_row(symbol: str) -> dict:
    try:
        t    = yf.Ticker(symbol + ".NS")
        info = t.fast_info
        hist = t.history(period="20d")
        price = info.get("last_price") or info.get("regularMarketPrice", 0)
        prev  = info.get("previous_close") or info.get("regularMarketPreviousClose", price)
        chg   = (price - prev) / prev * 100 if prev else 0
        vol   = info.get("last_volume") or 0
        if not hist.empty and isinstance(hist.columns, __import__("pandas").MultiIndex):
            hist.columns = [c[0] for c in hist.columns]
        avg_v = hist["Volume"].mean() if len(hist) > 5 else 1
        vol_spike = vol / avg_v if avg_v else 1.0
        closes = hist["Close"].tolist()[-15:]
        return {
            "price":   price,
            "chg":     chg,
            "vol_spike": vol_spike,
            "closes":  closes,
        }
    except Exception:
        return {"price": 0, "chg": 0, "vol_spike": 0, "closes": []}


def render_watchlist(watchlist: dict[str, list[str]] | None = None):
    wl = watchlist or DEFAULT_WATCHLIST
    for group, symbols in wl.items():
        with st.expander(f"📂 {group}", expanded=True):
            header_cols = st.columns([2, 1.5, 1.5, 1.5, 1])
            header_cols[0].markdown("<span style='color:#8892a4;font-size:.7rem;text-transform:uppercase'>Symbol</span>", unsafe_allow_html=True)
            header_cols[1].markdown("<span style='color:#8892a4;font-size:.7rem;text-transform:uppercase'>Price</span>", unsafe_allow_html=True)
            header_cols[2].markdown("<span style='color:#8892a4;font-size:.7rem;text-transform:uppercase'>Chg%</span>", unsafe_allow_html=True)
            header_cols[3].markdown("<span style='color:#8892a4;font-size:.7rem;text-transform:uppercase'>Vol Spike</span>", unsafe_allow_html=True)
            header_cols[4].markdown("<span style='color:#8892a4;font-size:.7rem;text-transform:uppercase'>15d</span>", unsafe_allow_html=True)

            for sym in symbols:
                row = _fetch_row(sym)
                chg_color = "#00ff88" if row["chg"] >= 0 else "#ff4466"
                spike_color = "#ffb800" if row["vol_spike"] > 2 else "#8892a4"
                svg = _mini_sparkline_svg(row["closes"])

                cols = st.columns([2, 1.5, 1.5, 1.5, 1])
                cols[0].markdown(f"**{sym}**")
                cols[1].markdown(
                    f"<span style='font-family:JetBrains Mono,monospace;font-size:.9rem'>₹{row['price']:,.1f}</span>",
                    unsafe_allow_html=True,
                )
                cols[2].markdown(
                    f"<span style='color:{chg_color};font-family:JetBrains Mono,monospace;font-size:.9rem'>"
                    f"{'+'if row['chg']>=0 else ''}{row['chg']:.2f}%</span>",
                    unsafe_allow_html=True,
                )
                cols[3].markdown(
                    f"<span style='color:{spike_color};font-family:JetBrains Mono,monospace;font-size:.88rem'>"
                    f"{row['vol_spike']:.1f}x</span>",
                    unsafe_allow_html=True,
                )
                cols[4].markdown(svg, unsafe_allow_html=True)
