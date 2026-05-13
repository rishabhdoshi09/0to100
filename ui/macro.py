"""Macro dashboard — bond yields, VIX, DXY, commodities, news sentiment index."""
from __future__ import annotations

import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

YIELD_CURVE_TICKERS = {
    "3M":  "^IRX",
    "2Y":  "^TYX",   # approximation available via yf
    "5Y":  "^FVX",
    "10Y": "^TNX",
    "30Y": "^TYX",
}

MACRO_SPOTS = {
    "DXY":        ("DX-Y.NYB",   "dollar"),
    "Gold $/oz":  ("GC=F",       "commodity"),
    "Crude WTI":  ("CL=F",       "commodity"),
    "India VIX":  ("^INDIAVIX",  "risk"),
    "Nifty 50":   ("^NSEI",      "equity"),
    "USD/INR":    ("INR=X",      "fx"),
}


@st.cache_data(ttl=180)
def _spot(ticker: str) -> tuple[float, float]:
    """Return (price, pct_change) for a yfinance ticker."""
    try:
        h = yf.Ticker(ticker).history(period="5d")
        if h is None or len(h) < 2:
            return 0.0, 0.0
        if isinstance(h.columns, pd.MultiIndex):
            h.columns = [c[0] for c in h.columns]
        price = float(h["Close"].iloc[-1])
        prev  = float(h["Close"].iloc[-2])
        chg   = (price - prev) / prev * 100 if prev else 0.0
        return round(price, 2), round(chg, 2)
    except Exception:
        return 0.0, 0.0


@st.cache_data(ttl=300)
def _yield_curve_data() -> pd.Series:
    maturities = {"3M": "^IRX", "5Y": "^FVX", "10Y": "^TNX", "30Y": "^TYX"}
    data = {}
    for label, sym in maturities.items():
        p, _ = _spot(sym)
        if p:
            data[label] = p
    return pd.Series(data)


def render_macro_dashboard():
    st.markdown("#### Macro Pulse")

    # ── Spot row ──────────────────────────────────────────────────────────────
    cols = st.columns(len(MACRO_SPOTS))
    for col, (name, (sym, _)) in zip(cols, MACRO_SPOTS.items()):
        price, chg = _spot(sym)
        arrow = "▲" if chg >= 0 else "▼"
        color = "#00ff88" if chg >= 0 else "#ff4466"
        col.markdown(
            f"<div class='devbloom-card' style='padding:.75rem;text-align:center'>"
            f"<div style='font-size:.65rem;color:#8892a4;text-transform:uppercase;letter-spacing:.05em'>{name}</div>"
            f"<div style='font-size:1.05rem;font-family:JetBrains Mono,monospace;color:#e8eaf0;font-weight:600'>{price:,.2f}</div>"
            f"<div style='font-size:.8rem;color:{color}'>{arrow} {abs(chg):.2f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Yield curve ───────────────────────────────────────────────────────────
    yc = _yield_curve_data()
    if len(yc) >= 3:
        with st.expander("📈 US Yield Curve", expanded=False):
            fig = go.Figure(go.Scatter(
                x=list(yc.index),
                y=list(yc.values),
                mode="lines+markers",
                line=dict(color="#00d4ff", width=2),
                marker=dict(size=6, color="#00d4ff"),
                fill="tozeroy",
                fillcolor="rgba(0,212,255,0.06)",
                hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=8, b=0),
                height=160,
                xaxis=dict(color="#8892a4", tickfont=dict(size=10)),
                yaxis=dict(color="#8892a4", tickfont=dict(size=10), ticksuffix="%"),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="macro_chart")
            # 2s10s spread
            if "2Y" in yc.index and "10Y" in yc.index:
                spread = yc["10Y"] - yc.get("2Y", yc["3M"])
                color  = "#00ff88" if spread >= 0 else "#ff4466"
                st.markdown(
                    f"<span style='color:#8892a4;font-size:.75rem'>2s10s spread: </span>"
                    f"<span style='color:{color};font-family:JetBrains Mono,monospace;font-size:.85rem'>"
                    f"{'+' if spread>=0 else ''}{spread:.2f}bps</span>",
                    unsafe_allow_html=True,
                )
