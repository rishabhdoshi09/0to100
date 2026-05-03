"""Bloomberg-grade dashboard — live cockpit with market breadth + signals."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sq_ai.ui._api import get, post


_NIFTY50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "ITC.NS", "SBIN.NS", "LT.NS", "AXISBANK.NS", "KOTAKBANK.NS",
    "HINDUNILVR.NS", "BHARTIARTL.NS", "MARUTI.NS", "TITAN.NS",
    "BAJFINANCE.NS", "WIPRO.NS", "ONGC.NS", "ADANIENT.NS", "NTPC.NS",
    "POWERGRID.NS",
]

_VERDICT_COLOR = {
    "STRONG BUY": "#00c853", "BUY": "#00c853",
    "HOLD": "#ffd740",
    "SELL": "#d50000", "STRONG SELL": "#d50000",
}


def _equity_chart(eq_data: list[dict]) -> go.Figure | None:
    if not eq_data:
        return None
    df = pd.DataFrame(eq_data)
    df["date"] = pd.to_datetime(df["date"])
    initial = df["equity"].iloc[0]
    df["return_pct"] = (df["equity"] / initial - 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["equity"], name="Equity",
        line={"color": "#00c853", "width": 2},
        fill="tozeroy", fillcolor="rgba(0,200,83,0.08)",
        hovertemplate="₹%{y:,.0f}<extra>Equity</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["cash"], name="Cash",
        line={"color": "#40c4ff", "width": 1.5, "dash": "dot"},
        hovertemplate="₹%{y:,.0f}<extra>Cash</extra>",
    ))
    fig.update_layout(
        height=280, margin={"l": 0, "r": 0, "t": 16, "b": 0},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
        yaxis={"tickprefix": "₹", "tickformat": ",.0f", "side": "right"},
        xaxis={"showgrid": False},
        hovermode="x unified",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _sector_heatmap(sectors: list[dict]) -> go.Figure | None:
    if not sectors:
        return None
    df = pd.DataFrame(sectors)
    if "change_pct" not in df.columns:
        return None
    df["change_pct"] = pd.to_numeric(df["change_pct"], errors="coerce").fillna(0)
    fig = go.Figure(go.Treemap(
        labels=df.get("sector", df.iloc[:, 0]),
        parents=[""] * len(df),
        values=df.get("market_cap", pd.Series([1] * len(df))),
        customdata=df["change_pct"],
        texttemplate="%{label}<br>%{customdata:+.2f}%",
        marker_colors=df["change_pct"],
        marker_colorscale=[[0, "#d50000"], [0.5, "#333"], [1, "#00c853"]],
        marker_cmid=0,
    ))
    fig.update_layout(
        height=260, margin=dict(t=0, b=0, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render() -> None:
    st.markdown(
        "<style>div[data-testid='metric-container']{background:#1a1a1a;"
        "border:1px solid #333;border-radius:8px;padding:10px}</style>",
        unsafe_allow_html=True,
    )

    # ── Portfolio KPIs ────────────────────────────────────────────────────────
    snap = get("/api/portfolio") or {}
    c1, c2, c3, c4, c5 = st.columns(5)
    eq = snap.get("equity", 0)
    cash = snap.get("cash", 0)
    gross = snap.get("gross_exposure", 0)
    exp_pct = snap.get("exposure_pct", 0)
    dpnl = snap.get("daily_pnl_pct", 0)
    unreal = snap.get("unrealized_pnl", 0)

    c1.metric("Portfolio equity", f"₹{eq:,.0f}")
    c2.metric("Cash available", f"₹{cash:,.0f}")
    c3.metric("Gross exposure", f"₹{gross:,.0f}", f"{exp_pct:.1f}%")
    c4.metric("Day P&L", f"{dpnl:+.2f}%",
              delta_color="normal" if dpnl >= 0 else "inverse")
    c5.metric("Unrealised P&L", f"₹{unreal:+,.0f}",
              delta_color="normal" if unreal >= 0 else "inverse")

    st.divider()

    # ── Market snapshot ────────────────────────────────────────────────────────
    refresh = st.button("⟳ Refresh snapshot")
    if refresh:
        st.cache_data.clear()

    @st.cache_data(ttl=120)
    def _mkt() -> dict:
        from sq_ai.backend.report_scheduler import market_snapshot, top_movers
        return {
            "snap": market_snapshot(),
            "movers": top_movers(_NIFTY50, top_n=5),
        }

    mkt = _mkt()

    col_idx, col_movers = st.columns([3, 4])
    with col_idx:
        st.subheader("Indices")
        idx = mkt["snap"].get("indices") or []
        if idx:
            df_idx = pd.DataFrame(idx)
            st.dataframe(df_idx, use_container_width=True, hide_index=True)
        else:
            st.caption("market data loading…")

    with col_movers:
        col_g, col_l = st.columns(2)
        gainers = mkt["movers"].get("gainers") or []
        losers = mkt["movers"].get("losers") or []
        with col_g:
            st.subheader("🟢 Top gainers")
            if gainers:
                st.dataframe(pd.DataFrame(gainers),
                             use_container_width=True, hide_index=True)
        with col_l:
            st.subheader("🔴 Top losers")
            if losers:
                st.dataframe(pd.DataFrame(losers),
                             use_container_width=True, hide_index=True)

    # ── Sector heatmap ─────────────────────────────────────────────────────────
    sectors = mkt["snap"].get("sectors") or []
    if sectors:
        st.subheader("Sector heatmap")
        fig_ht = _sector_heatmap(sectors)
        if fig_ht:
            st.plotly_chart(fig_ht, use_container_width=True)
        else:
            st.dataframe(pd.DataFrame(sectors),
                         use_container_width=True, hide_index=True)

    st.divider()

    # ── Equity curve ──────────────────────────────────────────────────────────
    eq_data = get("/api/equity") or []
    st.subheader("Equity curve")
    if not eq_data:
        st.caption("no equity history yet — records after each 5-min cycle")
    else:
        fig_eq = _equity_chart(eq_data)
        if fig_eq:
            st.plotly_chart(fig_eq, use_container_width=True)
        df_eq = pd.DataFrame(eq_data)
        initial = df_eq["equity"].iloc[0]
        latest = df_eq["equity"].iloc[-1]
        peak = df_eq["equity"].max()
        trough = df_eq.loc[df_eq["equity"].idxmax():, "equity"].min()
        dd = (trough / peak - 1) * 100 if peak else 0
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Total return", f"{(latest/initial-1)*100:+.2f}%")
        mc2.metric("Peak equity", f"₹{peak:,.0f}")
        mc3.metric("Max drawdown", f"{dd:.2f}%")

    st.divider()

    # ── Recent AI signals ─────────────────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.subheader("Recent AI signals")
        sigs = get("/api/signals/latest", params={"limit": 8}) or []
        if sigs:
            df_sig = pd.DataFrame(sigs)
            def _color_action(val: str) -> str:
                if "BUY" in str(val):
                    return "color: #00c853; font-weight: bold"
                if "SELL" in str(val):
                    return "color: #d50000; font-weight: bold"
                return "color: #ffd740"

            if "action" in df_sig.columns:
                styled = df_sig.style.applymap(_color_action, subset=["action"])
                st.dataframe(styled, use_container_width=True, hide_index=True)
            else:
                st.dataframe(df_sig, use_container_width=True, hide_index=True)
        else:
            st.caption("no signals yet — first cycle runs at 09:15 IST")

    with right:
        st.subheader("Open positions")
        pos = get("/api/positions") or []
        if pos:
            df_pos = pd.DataFrame(pos)
            st.dataframe(df_pos, use_container_width=True, hide_index=True)
        else:
            st.caption("no open positions")

    st.divider()

    # ── Manual controls ───────────────────────────────────────────────────────
    mc1, mc2 = st.columns(2)
    with mc1:
        if st.button("▶ Run decision cycle now", type="primary"):
            with st.spinner("running…"):
                result = post("/api/cycle/run")
                st.json(result)
    with mc2:
        if st.button("🔎 Run screener now"):
            with st.spinner("running screener…"):
                st.json(post("/api/screener/run"))
