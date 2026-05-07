"""
SmartChart — professional interactive Plotly chart.

Layout (4 rows, shared x-axis):
  Row 1 (50%): Candlestick + Volume bars + VWAP + Bollinger Bands +
               Donchian Channels + Volume Profile sidebar (shapes)
  Row 2 (17%): RSI (14) with 70/30 bands
  Row 3 (20%): MACD (12,26,9) — histogram + lines
  Row 4 (13%): ATR (14) — volatility ribbon

Volume Profile is rendered as Plotly shapes with xref="paper" so the
horizontal bars sit in the right-margin area and share the price y-axis.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from charting.explanations import explain as _explain
from charting.volume_profile import VolumeProfile
from logger import get_logger

log = get_logger(__name__)

# ── Indicator helpers ──────────────────────────────────────────────────────────

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def _compute_vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    return (typical * df["volume"]).cumsum() / df["volume"].cumsum()

def _compute_bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0):
    mid = _sma(close, window)
    std = close.rolling(window).std()
    return mid - num_std * std, mid, mid + num_std * std

def _compute_donchian(high: pd.Series, low: pd.Series, window: int = 20):
    return high.rolling(window).max(), low.rolling(window).min()

def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd = _ema(close, fast) - _ema(close, slow)
    sig = _ema(macd, signal)
    return macd, sig, macd - sig

def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ── SmartChart ─────────────────────────────────────────────────────────────────

class SmartChart:
    """
    Build a professional, interactive Plotly chart for a symbol.

    Usage
    -----
    chart = SmartChart()
    fig   = chart.build(df, symbol="RELIANCE")
    fig.write_html("chart.html")
    """

    # VP layout constants (paper coordinates, 0–1)
    _VP_X0 = 0.865   # left edge of VP sidebar in paper space
    _VP_MAX_W = 0.12  # maximum bar width in paper space

    @staticmethod
    def explain(indicator: str) -> str:
        """Return plain-language explanation for any indicator."""
        return _explain(indicator)

    def build(
        self,
        df: pd.DataFrame,
        symbol: str,
        show_vp: bool = True,
        vp_bins: int = 40,
    ) -> go.Figure:
        """
        Build the complete professional chart.

        Parameters
        ----------
        df      : OHLCV DataFrame (columns: open, high, low, close, volume)
                  with a DatetimeIndex (or integer index).
        symbol  : Used in title and annotations.
        show_vp : Overlay Volume Profile sidebar.
        vp_bins : Number of price bins for VPVR.

        Returns
        -------
        go.Figure – ready to display or save.
        """
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # ── indicators ────────────────────────────────────────────────────
        vwap = _compute_vwap(df)
        bb_lo, bb_mid, bb_hi = _compute_bollinger(df["close"])
        dc_hi, dc_lo = _compute_donchian(df["high"], df["low"])
        rsi = _compute_rsi(df["close"])
        macd, macd_sig, macd_hist = _compute_macd(df["close"])
        atr = _compute_atr(df["high"], df["low"], df["close"])

        is_green = df["close"] >= df["open"]
        vol_colors = ["#26a69a" if g else "#ef5350" for g in is_green]

        # ── figure skeleton ───────────────────────────────────────────────
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            row_heights=[0.50, 0.17, 0.20, 0.13],
            vertical_spacing=0.025,
            subplot_titles=[symbol, "RSI (14)", "MACD (12,26,9)", "ATR (14)"],
        )

        # ── row 1: candlesticks ───────────────────────────────────────────
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["open"], high=df["high"],
            low=df["low"],  close=df["close"],
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",
            decreasing_fillcolor="#ef5350",
        ), row=1, col=1)

        # Volume bars on a secondary y so they sit at the bottom of row 1
        fig.add_trace(go.Bar(
            x=df.index, y=df["volume"],
            name="Volume",
            marker_color=vol_colors,
            opacity=0.30,
            yaxis="y5",
            showlegend=False,
            hovertemplate="Vol: %{y:,.0f}<extra></extra>",
        ), row=1, col=1)

        # VWAP
        fig.add_trace(go.Scatter(
            x=df.index, y=vwap,
            name="VWAP",
            line=dict(color="#ff9800", width=1.8, dash="dash"),
            hovertemplate="VWAP: ₹%{y:.2f}<extra></extra>",
        ), row=1, col=1)

        # Bollinger Bands — upper + lower + shaded fill
        fig.add_trace(go.Scatter(
            x=df.index, y=bb_hi,
            name="BB Upper",
            line=dict(color="rgba(100,181,246,0.55)", width=1),
            showlegend=False,
            hovertemplate="BB Upper: ₹%{y:.2f}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=bb_lo,
            name="Bollinger Bands",
            fill="tonexty",
            fillcolor="rgba(100,181,246,0.07)",
            line=dict(color="rgba(100,181,246,0.55)", width=1),
            hovertemplate="BB Lower: ₹%{y:.2f}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=bb_mid,
            name="BB Mid (SMA20)",
            line=dict(color="rgba(100,181,246,0.35)", width=0.8, dash="dot"),
            showlegend=False,
        ), row=1, col=1)

        # Donchian Channels — upper + lower + thin shaded fill
        fig.add_trace(go.Scatter(
            x=df.index, y=dc_hi,
            name="Donchian Hi",
            line=dict(color="rgba(171,71,188,0.6)", width=1, dash="dot"),
            showlegend=False,
            hovertemplate="DC Upper: ₹%{y:.2f}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=dc_lo,
            name="Donchian Channels",
            fill="tonexty",
            fillcolor="rgba(171,71,188,0.04)",
            line=dict(color="rgba(171,71,188,0.6)", width=1, dash="dot"),
            hovertemplate="DC Lower: ₹%{y:.2f}<extra></extra>",
        ), row=1, col=1)

        # ── row 2: RSI ────────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=df.index, y=rsi,
            name="RSI (14)",
            line=dict(color="#e91e63", width=1.5),
            hovertemplate="RSI: %{y:.1f}<extra></extra>",
        ), row=2, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.06)",
                      line_width=0, row=2, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(38,166,154,0.06)",
                      line_width=0, row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(239,83,80,0.5)",
                      line_width=1, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(38,166,154,0.5)",
                      line_width=1, row=2, col=1)

        # ── row 3: MACD ───────────────────────────────────────────────────
        hist_colors = ["#26a69a" if v >= 0 else "#ef5350"
                       for v in macd_hist.fillna(0)]
        fig.add_trace(go.Bar(
            x=df.index, y=macd_hist,
            name="MACD Histogram",
            marker_color=hist_colors,
            showlegend=False,
            hovertemplate="Hist: %{y:.4f}<extra></extra>",
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=macd,
            name="MACD",
            line=dict(color="#2196f3", width=1.5),
            hovertemplate="MACD: %{y:.4f}<extra></extra>",
        ), row=3, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=macd_sig,
            name="Signal",
            line=dict(color="#ff9800", width=1.5),
            hovertemplate="Signal: %{y:.4f}<extra></extra>",
        ), row=3, col=1)

        # ── row 4: ATR ────────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=df.index, y=atr,
            name="ATR (14)",
            line=dict(color="#9c27b0", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(156,39,176,0.08)",
            hovertemplate="ATR: ₹%{y:.2f}<extra></extra>",
        ), row=4, col=1)

        # ── Volume Profile sidebar ────────────────────────────────────────
        if show_vp:
            # Constrain main xaxes to [0, VP_X0 − 0.03] so VP shapes
            # appear just to the right of the plot area without overlapping.
            x_domain_end = self._VP_X0 - 0.015
            for ax in ["xaxis", "xaxis2", "xaxis3", "xaxis4"]:
                fig.layout[ax].domain = [0.0, x_domain_end]

            self._add_volume_profile_shapes(fig, df, vp_bins)

        # ── global layout ─────────────────────────────────────────────────
        r_margin = 160 if show_vp else 60
        fig.update_layout(
            title=dict(
                text=f"{symbol} — Professional Chart",
                font=dict(size=17),
            ),
            template="plotly_dark",
            height=920,
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.01,
                xanchor="right", x=x_domain_end if show_vp else 1.0,
                font=dict(size=10),
            ),
            margin=dict(l=60, r=r_margin, t=80, b=40),
            # Secondary y for volume bars: scaled so bars occupy bottom 20%
            yaxis5=dict(
                overlaying="y",
                side="right",
                showticklabels=False,
                showgrid=False,
                range=[0, float(df["volume"].max()) * 5],
            ),
        )
        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
        fig.update_yaxes(title_text="MACD",  row=3, col=1)
        fig.update_yaxes(title_text="ATR (₹)", row=4, col=1)

        log.info("smartchart_built", symbol=symbol, candles=len(df),
                 show_vp=show_vp)
        return fig

    # ── Volume Profile helper ──────────────────────────────────────────────────

    def _add_volume_profile_shapes(
        self, fig: go.Figure, df: pd.DataFrame, bins: int
    ) -> None:
        """
        Render the VPVR as Plotly shapes using mixed xref/yref:
          xref="paper"  → x position is a fraction of figure width
          yref="y"      → y position uses the price data axis
        This makes the VP bars align perfectly with price levels.
        """
        result = VolumeProfile().compute(df, bins=bins)
        if result is None:
            return

        prices = result.prices
        volumes = result.volumes
        poc = result.poc
        vah = result.vah
        val = result.val
        bh = result.bin_height / 2          # half-height of one bin
        max_v = max(volumes) if volumes else 1

        for price, vol in zip(prices, volumes):
            width = (vol / max_v) * self._VP_MAX_W
            is_poc = abs(price - poc) < bh
            in_va  = val <= price <= vah

            if is_poc:
                color = "rgba(255,215,0,0.75)"
            elif in_va:
                color = "rgba(38,166,154,0.48)"
            else:
                color = "rgba(100,149,237,0.28)"

            fig.add_shape(
                type="rect",
                xref="paper", yref="y",
                x0=self._VP_X0,
                x1=self._VP_X0 + width,
                y0=price - bh,
                y1=price + bh,
                fillcolor=color,
                line=dict(width=0),
                layer="above",
            )

        # POC dashed horizontal line across the full chart
        fig.add_shape(
            type="line",
            xref="paper", yref="y",
            x0=0.0, x1=1.0,
            y0=poc, y1=poc,
            line=dict(color="rgba(255,215,0,0.55)", width=1, dash="dot"),
        )

        # VAH / VAL reference lines (right side only)
        for level, label in [(vah, "VAH"), (val, "VAL")]:
            fig.add_shape(
                type="line",
                xref="paper", yref="y",
                x0=self._VP_X0 - 0.01, x1=1.0,
                y0=level, y1=level,
                line=dict(color="rgba(38,166,154,0.45)", width=1, dash="dot"),
            )

        # Labels
        fig.add_annotation(
            xref="paper", yref="y",
            x=0.995, y=poc,
            text=f"POC ₹{poc:.1f}",
            showarrow=False,
            font=dict(size=8, color="rgba(255,215,0,0.9)"),
            xanchor="right", yanchor="middle",
            bgcolor="rgba(0,0,0,0.4)",
        )
        fig.add_annotation(
            xref="paper", yref="y",
            x=0.995, y=vah,
            text=f"VAH ₹{vah:.1f}",
            showarrow=False,
            font=dict(size=7, color="rgba(38,166,154,0.8)"),
            xanchor="right", yanchor="bottom",
        )
        fig.add_annotation(
            xref="paper", yref="y",
            x=0.995, y=val,
            text=f"VAL ₹{val:.1f}",
            showarrow=False,
            font=dict(size=7, color="rgba(38,166,154,0.8)"),
            xanchor="right", yanchor="top",
        )

        log.debug("vp_shapes_added", poc=round(poc, 2), vah=round(vah, 2),
                  val=round(val, 2), bins=len(prices))
