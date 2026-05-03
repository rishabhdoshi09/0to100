"""Bloomberg-grade dashboard: indices, top movers, sector heatmap, portfolio."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from sq_ai.ui._api import get, post


def render() -> None:
    st.title("Bloomberg-grade dashboard")

    col_a, col_b, col_c, col_d = st.columns(4)
    snap = get("/api/portfolio") or {}
    col_a.metric("Equity", f"₹{snap.get('equity', 0):,.0f}")
    col_b.metric("Cash", f"₹{snap.get('cash', 0):,.0f}")
    col_c.metric("Exposure", f"{snap.get('exposure_pct', 0):.1f}%")
    col_d.metric("Day P&L", f"{snap.get('daily_pnl_pct', 0):+.2f}%")

    st.divider()

    # --- Market snapshot via daily report endpoint helper ----------------
    if st.button("⟳ Refresh market snapshot"):
        st.cache_data.clear()

    @st.cache_data(ttl=120)
    def _snapshot() -> dict:
        from sq_ai.backend.report_scheduler import market_snapshot, top_movers
        return {
            "snap": market_snapshot(),
            "movers": top_movers([
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
                "ITC.NS", "SBIN.NS", "LT.NS", "AXISBANK.NS", "KOTAKBANK.NS",
                "HINDUNILVR.NS", "BHARTIARTL.NS", "MARUTI.NS", "TITAN.NS",
                "BAJFINANCE.NS",
            ], top_n=5),
        }

    data = _snapshot()
    st.subheader("Indices")
    if data["snap"]["indices"]:
        st.dataframe(pd.DataFrame(data["snap"]["indices"]),
                     use_container_width=True, hide_index=True)
    else:
        st.info("Index data unavailable.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top gainers")
        if data["movers"]["gainers"]:
            st.dataframe(pd.DataFrame(data["movers"]["gainers"]),
                         use_container_width=True, hide_index=True)
    with c2:
        st.subheader("Top losers")
        if data["movers"]["losers"]:
            st.dataframe(pd.DataFrame(data["movers"]["losers"]),
                         use_container_width=True, hide_index=True)

    st.subheader("Sector heatmap")
    if data["snap"]["sectors"]:
        st.dataframe(pd.DataFrame(data["snap"]["sectors"]),
                     use_container_width=True, hide_index=True)

    # --- Recent signals --------------------------------------------------
    st.divider()
    st.subheader("Recent Claude signals")
    sigs = get("/api/signals/latest", params={"limit": 5}) or []
    if sigs:
        st.dataframe(pd.DataFrame(sigs), use_container_width=True, hide_index=True)
    else:
        st.caption("no signals yet — wait for the first 5-min cycle")

    st.subheader("Open positions")
    pos = get("/api/positions") or []
    if pos:
        st.dataframe(pd.DataFrame(pos), use_container_width=True, hide_index=True)
    else:
        st.caption("no open positions")

    if st.button("Run cycle now"):
        with st.spinner("running…"):
            st.json(post("/api/cycle/run"))
