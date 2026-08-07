"""
Liquidity Heatmap — simulated order book depth chart.

A live implementation would stream Zerodha's 20-level market depth via
KiteConnect's WebSocket.  Until then we generate a realistic-looking book
using exponential decay from the current price plus random iceberg spikes.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict

from charting.explanations import explain as _explain
from logger import get_logger

log = get_logger(__name__)

_SIMULATED_NOTE = "(Simulated — live data requires Zerodha WebSocket)"


class LiquidityHeatmap:
    """
    Build an interactive order-book depth chart.

    Usage
    -----
    lh  = LiquidityHeatmap()
    book = lh.simulate_book(current_price=2500.0)
    fig  = lh.build_figure(book, symbol="RELIANCE")
    """

    def simulate_book(
        self,
        current_price: float,
        levels: int = 20,
        trend_bias: float = 0.0,
        seed: int | None = None,
    ) -> Dict:
        """
        Simulate `levels` bid and ask price levels around `current_price`.

        trend_bias > 0 → heavier ask side (bearish pressure)
        trend_bias < 0 → heavier bid side (bullish pressure)

        Returns a dict suitable for build_figure().
        """
        rng = np.random.default_rng(seed)
        tick = max(current_price * 0.0005, 0.05)

        bid_prices, bid_sizes = [], []
        ask_prices, ask_sizes = [], []

        for i in range(levels):
            # Exponential decay — more orders near best bid/ask
            base_b = rng.exponential(scale=1000) * np.exp(-i * 0.12)
            base_a = rng.exponential(scale=1000) * np.exp(-i * 0.12)
            # Occasional iceberg / institutional block
            if rng.random() < 0.12:
                base_b *= rng.uniform(6, 18)
            if rng.random() < 0.12:
                base_a *= rng.uniform(6, 18)
            bias_b = 1.0 - max(0.0, trend_bias) * 0.4
            bias_a = 1.0 + max(0.0, trend_bias) * 0.4
            bid_prices.append(round(current_price - tick * (i + 1), 2))
            bid_sizes.append(max(1, int(base_b * bias_b)))
            ask_prices.append(round(current_price + tick * (i + 1), 2))
            ask_sizes.append(max(1, int(base_a * bias_a)))

        log.debug("liquidity_book_simulated", price=current_price, levels=levels)
        return dict(
            bid_prices=bid_prices,
            bid_sizes=bid_sizes,
            ask_prices=ask_prices,
            ask_sizes=ask_sizes,
            current_price=current_price,
        )

    def build_figure(
        self,
        book: Dict,
        symbol: str = "",
        simulated: bool = True,
    ) -> go.Figure:
        """
        Build a butterfly-style (back-to-back) horizontal bar chart:
          left  side = bids (buy orders, green)
          right side = asks (sell orders, red)
        """
        bid_prices = book["bid_prices"]
        bid_sizes  = book["bid_sizes"]
        ask_prices = book["ask_prices"]
        ask_sizes  = book["ask_sizes"]
        current    = book["current_price"]

        all_prices = sorted(set(bid_prices + ask_prices), reverse=True)
        # Map each price to its bid/ask size (0 if absent)
        bid_map = dict(zip(bid_prices, bid_sizes))
        ask_map = dict(zip(ask_prices, ask_sizes))

        y_labels  = [f"₹{p:,.2f}" for p in all_prices]
        bid_vals  = [-bid_map.get(p, 0) for p in all_prices]   # negative = left
        ask_vals  = [ ask_map.get(p, 0) for p in all_prices]

        max_size = max(max(abs(v) for v in bid_vals), max(ask_vals), 1)

        def _alpha(size: int) -> float:
            return min(0.9, abs(size) / max_size + 0.25)

        fig = go.Figure()

        # Bid bars (green, pointing left)
        fig.add_trace(go.Bar(
            y=y_labels,
            x=bid_vals,
            orientation="h",
            name="Bids (Buy Orders)",
            marker=dict(
                color=[f"rgba(38,166,154,{_alpha(s):.2f})" for s in bid_vals],
                line=dict(width=0),
            ),
            customdata=[abs(v) for v in bid_vals],
            hovertemplate="Price: %{y}<br>Buy qty: %{customdata:,.0f}<extra></extra>",
        ))

        # Ask bars (red, pointing right)
        fig.add_trace(go.Bar(
            y=y_labels,
            x=ask_vals,
            orientation="h",
            name="Asks (Sell Orders)",
            marker=dict(
                color=[f"rgba(239,83,80,{_alpha(s):.2f})" for s in ask_vals],
                line=dict(width=0),
            ),
            hovertemplate="Price: %{y}<br>Sell qty: %{x:,.0f}<extra></extra>",
        ))

        # Current price marker
        closest_label = min(y_labels,
                            key=lambda lbl: abs(float(lbl.replace("₹", "").replace(",", "")) - current))
        fig.add_annotation(
            y=closest_label,
            x=0,
            text=f"  LTP ₹{current:,.2f}",
            showarrow=False,
            font=dict(color="#ffd700", size=11, family="monospace"),
            xanchor="left",
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="#ffd700",
            borderwidth=1,
        )

        note = f" {_SIMULATED_NOTE}" if simulated else ""
        fig.update_layout(
            title=dict(
                text=f"{symbol} Order Book Depth{note}",
                font=dict(size=15),
            ),
            template="plotly_dark",
            height=560,
            barmode="overlay",
            xaxis=dict(
                title="Order Size (qty)",
                tickformat=",",
                zeroline=True,
                zerolinecolor="rgba(255,255,255,0.4)",
                zerolinewidth=2,
            ),
            yaxis=dict(title="Price Level", autorange=True),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
            margin=dict(l=80, r=40, t=80, b=40),
        )

        log.info("liquidity_heatmap_built", symbol=symbol, levels=len(bid_prices))
        return fig

    @staticmethod
    def explain() -> str:
        return _explain("liquidity_heatmap")
