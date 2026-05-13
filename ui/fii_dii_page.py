"""
FII/DII Activity page — "Follow the Smart Money"
Renders cash-market flows, derivatives positioning, bulk and block deal tables.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.fii_dii import (
    get_block_deals,
    get_bulk_deals,
    get_fii_derivative_stats,
    get_fii_dii_activity,
)

# ── Shared dark-theme layout defaults ─────────────────────────────────────────
_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(8,12,28,0.6)",
    font=dict(color="#c9d1d9", family="JetBrains Mono, monospace"),
    xaxis=dict(showgrid=False, zeroline=False, color="#8892a4"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False, color="#8892a4"),
    margin=dict(t=48, b=32, l=0, r=0),
    legend=dict(orientation="h", y=1.08, x=0),
)

_FII_BLUE = "#3b82f6"
_DII_ORANGE = "#f97316"
_GREEN = "#22c55e"
_RED = "#ef4444"


# ── Helper: smart-money stance ─────────────────────────────────────────────────
def _smart_money_stance(df: pd.DataFrame) -> str:
    """Return stance string based on FII net flow over last 5 rows."""
    if df.empty or "fii_net" not in df.columns:
        return "⚪ Insufficient data"
    recent = df.head(5)["fii_net"]
    consecutive_buy = 0
    consecutive_sell = 0
    for v in recent:
        if v > 0:
            consecutive_buy += 1
            consecutive_sell = 0
        elif v < 0:
            consecutive_sell += 1
            consecutive_buy = 0
        else:
            break
    if consecutive_buy >= 3:
        return "🟢 Accumulating"
    if consecutive_sell >= 3:
        return "🔴 Distributing"
    if recent.sum() > 0:
        return "🟡 Mildly Bullish"
    return "🟠 Mildly Bearish"


# ── Helper: colour-coded deals dataframe ──────────────────────────────────────
def _render_deals_table(df: pd.DataFrame, key: str) -> None:
    if df.empty:
        st.info("No data available for the selected period.")
        return

    display = df.copy()
    display["date"] = display["date"].dt.strftime("%d %b %Y")
    display["quantity"] = display["quantity"].apply(lambda x: f"{x:,}")
    display["price"] = display["price"].apply(lambda x: f"₹{x:,.2f}")
    display.columns = ["Date", "Symbol", "Client", "B/S", "Quantity", "Price"]

    def _row_style(row):
        colour = "rgba(34,197,94,0.12)" if row["B/S"] in ("BUY", "B") else "rgba(239,68,68,0.12)"
        return [f"background-color:{colour}"] * len(row)

    st.dataframe(
        display.style.apply(_row_style, axis=1),
        hide_index=True,
        use_container_width=True,
        key=key,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RENDER
# ══════════════════════════════════════════════════════════════════════════════
def render_fii_dii_page() -> None:
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        "<h2 style='color:#00d4ff;font-family:JetBrains Mono,monospace;"
        "font-size:1.4rem;letter-spacing:2px;margin:0'>🌊 FII/DII Activity — Follow the Smart Money</h2>"
        "<p style='color:#8892a4;font-size:.8rem;margin:.3rem 0 1.2rem'>"
        "Cash-market flows · Derivatives positioning · Bulk &amp; Block deals · NSE public data · 1-hr cache</p>",
        unsafe_allow_html=True,
    )

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl_c1, ctrl_c2, ctrl_c3 = st.columns([2, 2, 3])
    with ctrl_c1:
        days = st.slider("History (days)", min_value=5, max_value=90, value=30, step=5, key="fii_days")
    with ctrl_c2:
        deal_days = st.slider("Deal window (days)", min_value=1, max_value=30, value=10, step=1, key="fii_deal_days")
    with ctrl_c3:
        if st.button("🔄 Refresh data", key="fii_refresh"):
            get_fii_dii_activity.clear()
            get_bulk_deals.clear()
            get_block_deals.clear()
            get_fii_derivative_stats.clear()
            st.rerun()

    st.divider()

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Fetching FII/DII data from NSE…"):
        df_flow = get_fii_dii_activity(days=days)
        df_bulk = get_bulk_deals(days=deal_days)
        df_block = get_block_deals(days=deal_days)
        deriv = get_fii_derivative_stats()

    # ── FII vs DII Net Flow chart ─────────────────────────────────────────────
    if not df_flow.empty:
        # Sort ascending for left→right chronology
        df_chart = df_flow.sort_values("date")

        fii_colors = [_GREEN if v >= 0 else _RED for v in df_chart["fii_net"]]
        dii_colors = [_GREEN if v >= 0 else _RED for v in df_chart["dii_net"]]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=df_chart["date"],
                y=df_chart["fii_net"],
                name="FII Net",
                marker_color=fii_colors,
                marker_line_color=_FII_BLUE,
                marker_line_width=1.2,
                opacity=0.85,
            )
        )
        fig.add_trace(
            go.Bar(
                x=df_chart["date"],
                y=df_chart["dii_net"],
                name="DII Net",
                marker_color=dii_colors,
                marker_line_color=_DII_ORANGE,
                marker_line_width=1.2,
                opacity=0.70,
            )
        )
        # Zero reference line
        fig.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)

        layout = dict(**_DARK_LAYOUT)
        layout.update(
            title=dict(text="FII vs DII Net Flow (₹ Cr)", font=dict(size=14, color="#00d4ff")),
            barmode="group",
            height=380,
            yaxis_title="₹ Crore",
        )
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Could not load FII/DII flow data from NSE. NSE sometimes blocks automated requests — try refreshing later.")

    # ── Summary metrics row ───────────────────────────────────────────────────
    st.markdown("#### 📊 Flow Summary")
    m1, m2, m3, m4 = st.columns(4)

    if not df_flow.empty:
        last5 = df_flow.head(5)
        fii_5d = last5["fii_net"].sum()
        dii_5d = last5["dii_net"].sum()
        stance = _smart_money_stance(df_flow)
        last_date = df_flow["date"].max()
        update_label = last_date.strftime("%d %b %Y") if not pd.isna(last_date) else "N/A"

        with m1:
            st.metric(
                "FII 5-Day Net",
                f"₹{fii_5d:,.0f} Cr",
                delta="Buying" if fii_5d > 0 else "Selling",
                delta_color="normal" if fii_5d > 0 else "inverse",
            )
        with m2:
            st.metric(
                "DII 5-Day Net",
                f"₹{dii_5d:,.0f} Cr",
                delta="Buying" if dii_5d > 0 else "Selling",
                delta_color="normal" if dii_5d > 0 else "inverse",
            )
        with m3:
            st.markdown(
                f"<div style='border:1px solid rgba(255,255,255,.08);border-radius:8px;"
                f"padding:.8rem 1rem;background:rgba(8,12,28,.6)'>"
                f"<p style='color:#8892a4;font-size:.75rem;margin:0 0 .2rem'>Smart Money Stance</p>"
                f"<p style='font-size:1.05rem;font-weight:700;margin:0'>{stance}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f"<div style='border:1px solid rgba(255,255,255,.08);border-radius:8px;"
                f"padding:.8rem 1rem;background:rgba(8,12,28,.6)'>"
                f"<p style='color:#8892a4;font-size:.75rem;margin:0 0 .2rem'>Last Data Point</p>"
                f"<p style='font-size:1.05rem;font-weight:700;margin:0;color:#00d4ff'>{update_label}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        for col in [m1, m2, m3, m4]:
            with col:
                st.metric("—", "N/A")

    st.divider()

    # ── Derivatives Positioning ───────────────────────────────────────────────
    if any(v is not None for v in deriv.values()):
        st.markdown("#### 📐 FII Derivatives Positioning")
        d1, d2, d3, d4 = st.columns(4)

        def _deriv_metric(col, label: str, key: str) -> None:
            val = deriv.get(key)
            with col:
                if val is not None:
                    col.metric(
                        label,
                        f"₹{val:,.0f} Cr",
                        delta="Long bias" if val > 0 else "Short bias",
                        delta_color="normal" if val > 0 else "inverse",
                    )
                else:
                    col.metric(label, "N/A")

        _deriv_metric(d1, "Index Futures", "index_futures_net")
        _deriv_metric(d2, "Index Options", "index_options_net")
        _deriv_metric(d3, "Stock Futures", "stock_futures_net")
        _deriv_metric(d4, "Stock Options", "stock_options_net")

        total = deriv.get("total_net")
        if total is not None:
            stance_color = _GREEN if total > 0 else _RED
            st.markdown(
                f"<div style='margin-top:.5rem;padding:.6rem 1rem;border-radius:6px;"
                f"background:rgba(8,12,28,.6);border-left:3px solid {stance_color}'>"
                f"<span style='color:#8892a4;font-size:.8rem'>Total FII Derivatives Net: </span>"
                f"<strong style='color:{stance_color};font-size:1rem'>₹{total:,.0f} Cr</strong>"
                f"{'  🐂 Net Long' if total > 0 else '  🐻 Net Short'}</div>",
                unsafe_allow_html=True,
            )
        st.divider()

    # ── Bulk Deals ────────────────────────────────────────────────────────────
    st.markdown("#### 📦 Bulk Deals")
    st.caption(f"Trades ≥ 0.5% of company equity in a single transaction · last {deal_days} days")

    bulk_filter = st.text_input(
        "Filter by symbol", placeholder="e.g. RELIANCE", key="bulk_sym_filter"
    ).upper().strip()

    df_bulk_display = df_bulk.copy()
    if bulk_filter and not df_bulk_display.empty:
        df_bulk_display = df_bulk_display[
            df_bulk_display["symbol"].str.contains(bulk_filter, na=False)
        ]

    _render_deals_table(df_bulk_display, key="bulk_deals_table")

    st.divider()

    # ── Block Deals ───────────────────────────────────────────────────────────
    st.markdown("#### 🧱 Block Deals")
    st.caption(f"Large pre-negotiated trades executed in a separate window · last {deal_days} days")

    block_filter = st.text_input(
        "Filter by symbol", placeholder="e.g. INFY", key="block_sym_filter"
    ).upper().strip()

    df_block_display = df_block.copy()
    if block_filter and not df_block_display.empty:
        df_block_display = df_block_display[
            df_block_display["symbol"].str.contains(block_filter, na=False)
        ]

    _render_deals_table(df_block_display, key="block_deals_table")

    # ── Footer note ───────────────────────────────────────────────────────────
    st.markdown(
        "<p style='color:#4a5568;font-size:.72rem;margin-top:1rem'>"
        "Data sourced from NSE public endpoints (nseindia.com/api). "
        "FII/DII figures are provisional and subject to revision. "
        "Cache TTL: 1 hour. Not investment advice."
        "</p>",
        unsafe_allow_html=True,
    )
