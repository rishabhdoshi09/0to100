"""Multi-timeframe chart grid — up to 8 linked Plotly charts for one symbol."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

_UP_COLOR   = "#00d4ff"
_DOWN_COLOR = "#ff4466"
_UP_FILL    = "rgba(0,212,255,0.07)"
_DOWN_FILL  = "rgba(255,68,102,0.07)"
_EMA20_COL  = "rgba(255,200,50,0.80)"
_EMA50_COL  = "rgba(180,80,220,0.70)"
_VOL_UP     = "rgba(0,212,255,0.35)"
_VOL_DOWN   = "rgba(255,68,102,0.35)"


@st.cache_data(ttl=300, show_spinner=False, persist="disk")
def _fetch_ohlcv(symbol: str, period: str, interval: str) -> pd.DataFrame:
    try:
        ticker = symbol if symbol.endswith(".NS") else symbol + ".NS"
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        return df.dropna()
    except Exception:
        return pd.DataFrame()


def _pct_change(df: pd.DataFrame) -> float:
    if df.empty or len(df) < 2:
        return 0.0
    first = float(df["close"].iloc[0])
    last  = float(df["close"].iloc[-1])
    return (last - first) / first * 100 if first else 0.0


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


# ─────────────────────────────────────────────────────────────────────────────
# LINE chart (compact, fast)
# ─────────────────────────────────────────────────────────────────────────────

def _line_chart(df: pd.DataFrame, label: str, show_ema: bool) -> go.Figure:
    pct = _pct_change(df)
    is_up = pct >= 0
    color = _UP_COLOR if is_up else _DOWN_COLOR
    fill  = _UP_FILL  if is_up else _DOWN_FILL
    sign  = "+" if pct >= 0 else ""
    title = f"{label}  <span style='color:{color}'>{sign}{pct:.1f}%</span>"

    fig = go.Figure()
    if not df.empty:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["close"],
            mode="lines", line=dict(color=color, width=1.5),
            fill="tozeroy", fillcolor=fill,
            hovertemplate="%{x|%Y-%m-%d}: ₹%{y:,.2f}<extra></extra>",
            name="Close",
        ))
        if show_ema and len(df) >= 20:
            fig.add_trace(go.Scatter(
                x=df.index, y=_ema(df["close"], 20),
                mode="lines", line=dict(color=_EMA20_COL, width=1, dash="dot"),
                hovertemplate="EMA20: ₹%{y:,.2f}<extra></extra>",
                name="EMA20",
            ))
        if show_ema and len(df) >= 50:
            fig.add_trace(go.Scatter(
                x=df.index, y=_ema(df["close"], 50),
                mode="lines", line=dict(color=_EMA50_COL, width=1, dash="dot"),
                hovertemplate="EMA50: ₹%{y:,.2f}<extra></extra>",
                name="EMA50",
            ))
    else:
        fig.add_annotation(text="No data", x=0.5, y=0.5,
                           xref="paper", yref="paper",
                           showarrow=False, font=dict(color="#8892a4", size=11))

    fig.update_layout(
        title=dict(text=title, font=dict(size=10, color="#8892a4"), x=0.02, y=0.96),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,12,28,0.6)",
        margin=dict(l=8, r=8, t=26, b=8),
        height=160,
        xaxis=dict(showgrid=False, showticklabels=False, color="#8892a4"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)",
                   tickfont=dict(size=8, color="#8892a4"), tickprefix="₹"),
        showlegend=False,
        hovermode="x unified",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CANDLESTICK chart with volume subplot
# ─────────────────────────────────────────────────────────────────────────────

def _candle_chart(df: pd.DataFrame, label: str, show_ema: bool) -> go.Figure:
    pct = _pct_change(df)
    is_up = pct >= 0
    color = _UP_COLOR if is_up else _DOWN_COLOR
    sign  = "+" if pct >= 0 else ""
    title = f"{label}  <span style='color:{color}'>{sign}{pct:.1f}%</span>"

    has_vol = not df.empty and "volume" in df.columns and df["volume"].sum() > 0

    row_heights = [0.72, 0.28] if has_vol else [1.0]
    rows = 2 if has_vol else 1
    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights,
    )

    if not df.empty:
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["open"], high=df["high"],
            low=df["low"],  close=df["close"],
            increasing_line_color=_UP_COLOR,
            decreasing_line_color=_DOWN_COLOR,
            increasing_fillcolor="rgba(0,212,255,0.4)",
            decreasing_fillcolor="rgba(255,68,102,0.4)",
            line=dict(width=0.8),
            whiskerwidth=0.8,
            hovertext=[
                f"O:{o:.2f} H:{h:.2f} L:{l:.2f} C:{c:.2f}"
                for o, h, l, c in zip(df["open"], df["high"], df["low"], df["close"])
            ],
            name="OHLC",
        ), row=1, col=1)

        if show_ema and len(df) >= 20:
            fig.add_trace(go.Scatter(
                x=df.index, y=_ema(df["close"], 20),
                mode="lines", line=dict(color=_EMA20_COL, width=1),
                hovertemplate="EMA20: ₹%{y:,.2f}<extra></extra>",
                name="EMA20",
            ), row=1, col=1)
        if show_ema and len(df) >= 50:
            fig.add_trace(go.Scatter(
                x=df.index, y=_ema(df["close"], 50),
                mode="lines", line=dict(color=_EMA50_COL, width=1),
                hovertemplate="EMA50: ₹%{y:,.2f}<extra></extra>",
                name="EMA50",
            ), row=1, col=1)

        if has_vol:
            vol_colors = [
                _VOL_UP if c >= o else _VOL_DOWN
                for c, o in zip(df["close"], df["open"])
            ]
            fig.add_trace(go.Bar(
                x=df.index, y=df["volume"],
                marker_color=vol_colors,
                hovertemplate="%{x|%Y-%m-%d}: %{y:,.0f}<extra>Vol</extra>",
                name="Volume",
            ), row=2, col=1)
    else:
        fig.add_annotation(text="No data", x=0.5, y=0.5,
                           xref="paper", yref="paper",
                           showarrow=False, font=dict(color="#8892a4", size=11))

    fig.update_layout(
        title=dict(text=title, font=dict(size=10, color="#8892a4"), x=0.02, y=0.99),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,12,28,0.6)",
        margin=dict(l=8, r=8, t=26, b=8),
        height=220,
        showlegend=False,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
    )
    _ax_style = dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)",
                     color="#8892a4", showticklabels=False)
    fig.update_xaxes(**_ax_style)
    fig.update_yaxes(
        tickfont=dict(size=8, color="#8892a4"),
        tickprefix="₹", showgrid=True,
        gridcolor="rgba(255,255,255,0.04)",
        row=1, col=1,
    )
    if has_vol:
        fig.update_yaxes(
            tickfont=dict(size=7, color="#8892a4"),
            showgrid=False, tickformat=".2s",
            row=2, col=1,
        )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Public renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_multi_tf_grid(symbol: str, selected_tfs: list[str] | None = None):
    """Render a 2×4 grid of mini charts across timeframes."""
    tfs = selected_tfs or list(TIMEFRAMES.keys())[:8]

    # ── Controls row ────────────────────────────────────────────────────────
    ctrl_l, ctrl_m, ctrl_r = st.columns([2, 2, 2])
    with ctrl_l:
        chart_type = st.radio(
            "Chart type", ["Line", "Candlestick"],
            horizontal=True, key="mtf_chart_type",
            label_visibility="collapsed",
        )
    with ctrl_m:
        show_ema = st.toggle("EMA 20/50 overlay", value=True, key="mtf_show_ema")
    with ctrl_r:
        cols_per_row = st.radio(
            "Columns", [2, 3, 4], index=2, horizontal=True,
            key="mtf_cols", label_visibility="collapsed",
        )

    st.caption(
        f"{'📊 Candlestick' if chart_type == 'Candlestick' else '📈 Line'} view · "
        f"{'EMA 20/50 on' if show_ema else 'EMA off'} · "
        f"{len(tfs)} timeframes"
    )

    rows = [tfs[i:i + cols_per_row] for i in range(0, len(tfs), cols_per_row)]
    for row_tfs in rows:
        cols = st.columns(len(row_tfs))
        for col, tf in zip(cols, row_tfs):
            period, interval, _ = TIMEFRAMES[tf]
            df = _fetch_ohlcv(symbol, period, interval)
            if chart_type == "Candlestick":
                fig = _candle_chart(df, TF_LABELS[tf], show_ema)
            else:
                fig = _line_chart(df, TF_LABELS[tf], show_ema)
            col.plotly_chart(fig, use_container_width=True,
                             config={"displayModeBar": False})
