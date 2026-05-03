"""Portfolio + manual trade Streamlit page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from sq_ai.ui._api import delete as api_delete, get, post


def render() -> None:
    st.title("Portfolio & trade")

    snap = get("/api/portfolio") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Equity", f"₹{snap.get('equity', 0):,.0f}")
    c2.metric("Cash", f"₹{snap.get('cash', 0):,.0f}")
    c3.metric("Exposure", f"{snap.get('exposure_pct', 0):.1f}%")
    c4.metric("Day P&L", f"{snap.get('daily_pnl_pct', 0):+.2f}%")

    st.subheader("Open positions")
    pos = get("/api/positions") or []
    if pos:
        st.dataframe(pd.DataFrame(pos), use_container_width=True, hide_index=True)
    else:
        st.caption("no open positions")

    st.subheader("Trade journal (last 100 closed)")
    closed = get("/api/trades", params={"limit": 100}) or []
    if closed:
        df = pd.DataFrame(closed)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Export CSV",
                           data=df.to_csv(index=False),
                           file_name="trades.csv", mime="text/csv")

    st.subheader("Watchlist")
    wl = get("/api/watchlist") or []
    if wl:
        st.dataframe(pd.DataFrame(wl), use_container_width=True, hide_index=True)
    cols = st.columns([2, 3, 1])
    add_sym = cols[0].text_input("Symbol")
    add_note = cols[1].text_input("Note")
    if cols[2].button("Add"):
        post("/api/watchlist", json={"symbol": add_sym, "note": add_note})
        st.rerun()
    if wl:
        rm = st.selectbox("Remove", [w["symbol"] for w in wl])
        if st.button("✗ Remove"):
            api_delete(f"/api/watchlist/{rm}")
            st.rerun()

    st.subheader("Manual order")
    with st.form("order"):
        cols = st.columns(5)
        sym = cols[0].text_input("Symbol", value="RELIANCE.NS")
        action = cols[1].selectbox("Action", ["BUY", "SELL"])
        qty = cols[2].number_input("Qty", min_value=1, value=1)
        price = cols[3].number_input("Price", min_value=0.0, value=0.0)
        stop = cols[4].number_input("Stop", min_value=0.0, value=0.0)
        target = st.number_input("Target", min_value=0.0, value=0.0)
        if st.form_submit_button("Place"):
            res = post("/api/trade", json={
                "symbol": sym, "action": action, "qty": qty,
                "price": price, "stop": stop, "target": target,
            })
            st.json(res)
            st.rerun()
