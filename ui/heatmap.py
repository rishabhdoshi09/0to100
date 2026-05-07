"""Global market heatmap — Plotly treemap of indices, sectors, and top NSE stocks."""
from __future__ import annotations

import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# NSE sector proxies via yfinance
SECTOR_TICKERS = {
    "Nifty 50":       "^NSEI",
    "Bank Nifty":     "^NSEBANK",
    "IT":             "^CNXIT",
    "Auto":           "^CNXAUTO",
    "FMCG":           "^CNXFMCG",
    "Pharma":         "^CNXPHARMA",
    "Metal":          "^CNXMETAL",
    "Energy":         "^CNXENERGY",
    "Realty":         "^CNXREALTY",
    "Media":          "^CNXMEDIA",
}

SPOT_TICKERS = {
    "VIX":   "^INDIAVIX",
    "DXY":   "DX-Y.NYB",
    "Gold":  "GC=F",
    "Crude": "CL=F",
    "BTC":   "BTC-USD",
}


@st.cache_data(ttl=120)
def _fetch_changes(tickers: dict[str, str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, sym in tickers.items():
        try:
            d = yf.Ticker(sym).fast_info
            prev  = d.get("previous_close") or d.get("regularMarketPreviousClose", 0)
            price = d.get("last_price") or d.get("regularMarketPrice", prev)
            if prev and prev != 0:
                out[name] = round((price - prev) / prev * 100, 2)
        except Exception:
            out[name] = 0.0
    return out


def render_heatmap():
    """Render an interactive sector / macro heatmap treemap."""
    changes = _fetch_changes(SECTOR_TICKERS)

    if not changes:
        st.info("Market data unavailable — check network / API limits.")
        return

    labels = list(changes.keys())
    values = [abs(v) + 0.5 for v in changes.values()]   # size ≥ 0.5
    pcts   = list(changes.values())
    colors = ["#00ff88" if p >= 0 else "#ff4466" for p in pcts]
    texts  = [f"{l}<br>{'+' if p>=0 else ''}{p:.2f}%" for l, p in zip(labels, pcts)]

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=[""] * len(labels),
        values=values,
        text=texts,
        textinfo="text",
        hovertemplate="%{text}<extra></extra>",
        marker=dict(
            colors=colors,
            line=dict(width=2, color="#0a0e1a"),
        ),
        textfont=dict(size=13, color="#ffffff", family="JetBrains Mono"),
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=260,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_macro_strip():
    """One-row strip of macro spot prices (VIX, DXY, Gold, Crude, BTC)."""
    changes = _fetch_changes(SPOT_TICKERS)
    cols = st.columns(len(SPOT_TICKERS))
    for col, (name, pct) in zip(cols, changes.items()):
        arrow = "▲" if pct >= 0 else "▼"
        color = "#00ff88" if pct >= 0 else "#ff4466"
        col.markdown(
            f"<div style='text-align:center;'>"
            f"<div style='font-size:.68rem;color:#8892a4;text-transform:uppercase;letter-spacing:.05em'>{name}</div>"
            f"<div style='font-size:1rem;font-family:JetBrains Mono,monospace;color:{color};font-weight:600'>"
            f"{arrow} {'+' if pct>=0 else ''}{pct:.2f}%</div></div>",
            unsafe_allow_html=True,
        )
