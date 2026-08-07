"""
Footprint Chart (Order Flow) from OHLCV data.

Real footprint charts require tick-level bid/ask data from a live
WebSocket feed.  When that is unavailable, we estimate bid and ask
volumes from price action:

  bull candle (close > open) → more volume hit the ask (buyers led)
  bear candle (close < open) → more volume hit the bid (sellers led)

The split fraction = (close − low) / (high − low), clamped to [0.1, 0.9].
Imbalance = one side is ≥ 3× the other.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from charting.explanations import explain as _explain
from logger import get_logger

log = get_logger(__name__)

_IMBALANCE_RATIO = 3.0


class FootprintAnalyzer:
    """
    Simulate footprint data from 1-minute OHLCV and build a Plotly figure
    with candlesticks, imbalance markers, bid/ask hover, and cumulative delta.

    Usage
    -----
    fa      = FootprintAnalyzer()
    analysis = fa.analyze(df)
    fig      = fa.build_figure(df, symbol="RELIANCE")
    """

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return df extended with columns:
          ask_vol, bid_vol, delta, cum_delta, imbalance_up, imbalance_dn
        """
        df = df.copy()
        ranges = (df["high"] - df["low"])
        ranges = ranges.where(ranges > 0, df["close"] * 0.001)
        # Fraction of volume that was aggressive buying
        bull_frac = ((df["close"] - df["low"]) / ranges).clip(0.1, 0.9)
        df["ask_vol"] = (df["volume"] * bull_frac).round(0)
        df["bid_vol"] = (df["volume"] - df["ask_vol"]).round(0)
        df["delta"] = df["ask_vol"] - df["bid_vol"]
        df["cum_delta"] = df["delta"].cumsum()
        ratio_up = df["ask_vol"] / df["bid_vol"].replace(0, 1)
        ratio_dn = df["bid_vol"] / df["ask_vol"].replace(0, 1)
        df["imbalance_up"] = ratio_up >= _IMBALANCE_RATIO   # buyers ≥ 3× sellers
        df["imbalance_dn"] = ratio_dn >= _IMBALANCE_RATIO   # sellers ≥ 3× buyers
        log.debug("footprint_analyzed", bars=len(df),
                  imb_up=int(df["imbalance_up"].sum()),
                  imb_dn=int(df["imbalance_dn"].sum()))
        return df

    def build_figure(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        simulated_note: bool = True,
    ) -> go.Figure:
        """
        Build a 2-row Plotly figure:
          Row 1 – Candlesticks + imbalance star markers + order-flow hover
          Row 2 – Cumulative Delta (bar chart)
        """
        analysis = self.analyze(df)

        suffix = " (Simulated from OHLCV)" if simulated_note else ""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.68, 0.32],
            vertical_spacing=0.03,
            subplot_titles=[
                f"{symbol} Footprint Chart{suffix}",
                "Cumulative Delta",
            ],
        )

        # ── Row 1: Candlesticks ───────────────────────────────────────────
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

        # Invisible scatter so order-flow data appears on hover
        fig.add_trace(go.Scatter(
            x=analysis.index,
            y=(df["high"] + df["low"]) / 2,
            mode="markers",
            marker=dict(size=1, opacity=0),
            text=[
                (f"<b>Ask (buy)</b>: {int(r.ask_vol):,}<br>"
                 f"<b>Bid (sell)</b>: {int(r.bid_vol):,}<br>"
                 f"<b>Delta</b>: {int(r.delta):+,}<br>"
                 f"{'⭐ BUY IMBALANCE' if r.imbalance_up else '⭐ SELL IMBALANCE' if r.imbalance_dn else ''}")
                for r in analysis.itertuples()
            ],
            hoverinfo="text+x",
            name="Order Flow",
            showlegend=False,
        ), row=1, col=1)

        # Ask imbalance stars (green, above candle)
        up_imb = analysis[analysis["imbalance_up"]]
        if not up_imb.empty:
            fig.add_trace(go.Scatter(
                x=up_imb.index,
                y=df.loc[up_imb.index, "high"] * 1.003,
                mode="markers",
                marker=dict(symbol="star", size=11, color="#26a69a",
                            line=dict(color="#ffffff", width=0.5)),
                name="Ask Imbalance ⭐",
                hovertemplate="Ask Imbalance<br>Ask: %{customdata[0]:,} / Bid: %{customdata[1]:,}<extra></extra>",
                customdata=up_imb[["ask_vol", "bid_vol"]].values,
            ), row=1, col=1)

        # Bid imbalance stars (red, below candle)
        dn_imb = analysis[analysis["imbalance_dn"]]
        if not dn_imb.empty:
            fig.add_trace(go.Scatter(
                x=dn_imb.index,
                y=df.loc[dn_imb.index, "low"] * 0.997,
                mode="markers",
                marker=dict(symbol="star", size=11, color="#ef5350",
                            line=dict(color="#ffffff", width=0.5)),
                name="Bid Imbalance ⭐",
                hovertemplate="Bid Imbalance<br>Bid: %{customdata[0]:,} / Ask: %{customdata[1]:,}<extra></extra>",
                customdata=dn_imb[["bid_vol", "ask_vol"]].values,
            ), row=1, col=1)

        # ── Row 2: Cumulative Delta ───────────────────────────────────────
        delta_colors = [
            "#26a69a" if v >= 0 else "#ef5350"
            for v in analysis["cum_delta"]
        ]
        fig.add_trace(go.Bar(
            x=analysis.index,
            y=analysis["cum_delta"],
            name="Cumulative Delta",
            marker_color=delta_colors,
            hovertemplate="CumΔ: %{y:+,.0f}<extra></extra>",
        ), row=2, col=1)
        fig.add_hline(y=0, line_color="rgba(255,255,255,0.25)",
                      line_width=1, row=2, col=1)

        fig.update_layout(
            title=dict(
                text=f"{symbol} — Order Flow & Footprint{suffix}",
                font=dict(size=16),
            ),
            template="plotly_dark",
            height=680,
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
            margin=dict(l=60, r=40, t=80, b=40),
        )
        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
        fig.update_yaxes(title_text="Cum. Delta", row=2, col=1)

        log.info("footprint_figure_built", symbol=symbol, bars=len(df))
        return fig

    @staticmethod
    def explain() -> str:
        return _explain("footprint")
