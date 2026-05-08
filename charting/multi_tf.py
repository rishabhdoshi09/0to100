"""Multi-timeframe chart grid — up to 8 linked Plotly charts for one symbol."""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

TIMEFRAMES = {
    "1d":  ("1d",  "1d",  90),
    "5d":  ("5d",  "1d",  30),
    "1mo": ("1mo", "1wk", 52),
    "3mo": ("3mo", "1d",  90),
    "6mo": ("6mo", "1d",  180),
    "1y":  ("1y",  "1wk", 52),
    "2y":  ("2y",  "1mo", 24),
    "5y":  ("5y",  "3mo", 20),
}

TF_LABELS = {
    "1d":  "Intraday",
    "5d":  "5-Day",
    "1mo": "1-Month",
    "3mo": "3-Month",
    "6mo": "6-Month",
    "1y":  "1-Year",
    "2y":  "2-Year",
    "5y":  "5-Year",
}


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_ohlcv(symbol: str, period: str, interval: str) -> pd.DataFrame:
    try:
        ticker = symbol if symbol.endswith(".NS") else symbol + ".NS"
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        # yfinance ≥0.2.40 returns MultiIndex columns like ('Close', 'RELIANCE.NS')
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        return df.dropna()
    except Exception:
        return pd.DataFrame()


def _mini_chart(df: pd.DataFrame, label: str) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data", x=0.5, y=0.5, xref="paper", yref="paper",
                           showarrow=False, font=dict(color="#8892a4", size=11))
    else:
        color = "#00d4ff" if df["close"].iloc[-1] >= df["close"].iloc[0] else "#ff4466"
        fig = go.Figure(go.Scatter(
            x=df.index, y=df["close"],
            mode="lines", line=dict(color=color, width=1.5),
            fill="tozeroy", fillcolor=color.replace("ff", "1a").replace("66", "0d") + "18",
            hovertemplate="%{x|%Y-%m-%d}: ₹%{y:,.2f}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(text=label, font=dict(size=10, color="#8892a4"), x=0.02, y=0.95),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,12,28,0.6)",
        margin=dict(l=8, r=8, t=24, b=8),
        height=150,
        xaxis=dict(showgrid=False, showticklabels=False, color="#8892a4"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)",
                   tickfont=dict(size=8, color="#8892a4"), tickprefix="₹"),
        showlegend=False,
    )
    return fig


def render_multi_tf_grid(symbol: str, selected_tfs: list[str] | None = None):
    """Render a 2×4 grid of mini charts across timeframes."""
    tfs = selected_tfs or list(TIMEFRAMES.keys())[:8]

    cols_per_row = 4
    rows = [tfs[i:i+cols_per_row] for i in range(0, len(tfs), cols_per_row)]

    for row_tfs in rows:
        cols = st.columns(len(row_tfs))
        for col, tf in zip(cols, row_tfs):
            period, interval, _ = TIMEFRAMES[tf]
            df = _fetch_ohlcv(symbol, period, interval)
            fig = _mini_chart(df, TF_LABELS[tf])
            col.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
