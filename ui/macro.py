"""Macro dashboard — bond yields, VIX, DXY, commodities, spreads."""
from __future__ import annotations

import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Correct US Treasury yfinance tickers
# ^IRX = 13-week (3M), ^FVX = 5-year, ^TNX = 10-year, ^TYX = 30-year
# No reliable 2Y ticker on yfinance; omit rather than duplicate TYX.
_YIELD_MATURITIES = {
    "3M":  "^IRX",
    "5Y":  "^FVX",
    "10Y": "^TNX",
    "30Y": "^TYX",
}

MACRO_SPOTS = {
    "DXY":        ("DX-Y.NYB",  "dollar"),
    "Gold $/oz":  ("GC=F",      "commodity"),
    "Crude WTI":  ("CL=F",      "commodity"),
    "India VIX":  ("^INDIAVIX", "risk"),
    "Nifty 50":   ("^NSEI",     "equity"),
    "USD/INR":    ("INR=X",     "fx"),
}

# Additional India-centric macro tickers shown in the second row
INDIA_SPOTS = {
    "Nifty Bank":  ("^NSEBANK",  "equity"),
    "Nifty IT":    ("^CNXIT",    "equity"),
    "Silver ₹":    ("SI=F",      "commodity"),
    "Brent Crude": ("BZ=F",      "commodity"),
    "EUR/USD":     ("EURUSD=X",  "fx"),
    "Nifty Next50":("^NSMIDCP",  "equity"),
}


@st.cache_data(ttl=180, persist="disk")
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


@st.cache_data(ttl=300, persist="disk")
def _yield_curve_data() -> pd.Series:
    data = {}
    for label, sym in _YIELD_MATURITIES.items():
        p, _ = _spot(sym)
        if p:
            data[label] = p
    return pd.Series(data)


def _spot_card(col, name: str, sym: str) -> None:
    price, chg = _spot(sym)
    arrow = "▲" if chg >= 0 else "▼"
    color = "#00ff88" if chg >= 0 else "#ff4466"
    col.markdown(
        f"<div class='devbloom-card' style='padding:.65rem .75rem;text-align:center'>"
        f"<div style='font-size:.6rem;color:#8892a4;text-transform:uppercase;"
        f"letter-spacing:.05em;margin-bottom:.15rem'>{name}</div>"
        f"<div style='font-size:1rem;font-family:JetBrains Mono,monospace;"
        f"color:#e8eaf0;font-weight:600'>{price:,.2f}</div>"
        f"<div style='font-size:.75rem;color:{color}'>{arrow} {abs(chg):.2f}%</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_macro_dashboard() -> None:
    st.markdown("#### Macro Pulse")

    # ── Global spots row ──────────────────────────────────────────────────────
    cols = st.columns(len(MACRO_SPOTS))
    for col, (name, (sym, _)) in zip(cols, MACRO_SPOTS.items()):
        _spot_card(col, name, sym)

    # ── India / extended spots (collapsed by default) ─────────────────────────
    with st.expander("🇮🇳 Extended Macro", expanded=False):
        icols = st.columns(len(INDIA_SPOTS))
        for col, (name, (sym, _)) in zip(icols, INDIA_SPOTS.items()):
            _spot_card(col, name, sym)

    # ── US Yield curve ────────────────────────────────────────────────────────
    yc = _yield_curve_data()
    if len(yc) >= 2:
        with st.expander("📈 US Yield Curve", expanded=False):
            # Sorted by maturity order
            order = ["3M", "5Y", "10Y", "30Y"]
            yc_sorted = yc.reindex([m for m in order if m in yc.index])

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(yc_sorted.index),
                y=list(yc_sorted.values),
                mode="lines+markers",
                line=dict(color="#00d4ff", width=2),
                marker=dict(size=7, color="#00d4ff",
                            line=dict(color="#0a0f1e", width=1.5)),
                fill="tozeroy",
                fillcolor="rgba(0,212,255,0.06)",
                hovertemplate="%{x}: %{y:.3f}%<extra></extra>",
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=8, b=0),
                height=170,
                xaxis=dict(color="#8892a4", tickfont=dict(size=10)),
                yaxis=dict(color="#8892a4", tickfont=dict(size=10), ticksuffix="%"),
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})

            # Spread metrics
            s_col1, s_col2, s_col3 = st.columns(3)
            if "3M" in yc_sorted and "10Y" in yc_sorted:
                spread_310 = yc_sorted["10Y"] - yc_sorted["3M"]
                c310 = "#00ff88" if spread_310 >= 0 else "#ff4466"
                s_col1.markdown(
                    f"<span style='color:#8892a4;font-size:.72rem'>3M–10Y: </span>"
                    f"<span style='color:{c310};font-family:JetBrains Mono,monospace;"
                    f"font-size:.82rem;font-weight:600'>"
                    f"{'+' if spread_310 >= 0 else ''}{spread_310:.2f}%</span>",
                    unsafe_allow_html=True,
                )
            if "5Y" in yc_sorted and "30Y" in yc_sorted:
                spread_530 = yc_sorted["30Y"] - yc_sorted["5Y"]
                c530 = "#00ff88" if spread_530 >= 0 else "#ff4466"
                s_col2.markdown(
                    f"<span style='color:#8892a4;font-size:.72rem'>5Y–30Y: </span>"
                    f"<span style='color:{c530};font-family:JetBrains Mono,monospace;"
                    f"font-size:.82rem;font-weight:600'>"
                    f"{'+' if spread_530 >= 0 else ''}{spread_530:.2f}%</span>",
                    unsafe_allow_html=True,
                )
            if "10Y" in yc_sorted and "30Y" in yc_sorted:
                invert = yc_sorted["10Y"] > yc_sorted["30Y"]
                s_col3.markdown(
                    f"<span style='color:#8892a4;font-size:.72rem'>Curve: </span>"
                    f"<span style='color:{'#ff4466' if invert else '#00ff88'};"
                    f"font-size:.82rem;font-weight:600'>"
                    f"{'⚠️ Inverted' if invert else '✅ Normal'}</span>",
                    unsafe_allow_html=True,
                )
