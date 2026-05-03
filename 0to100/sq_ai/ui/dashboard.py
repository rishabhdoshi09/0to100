"""Optional Streamlit dashboard – charts + tables.

Run with::

    streamlit run sq_ai/ui/dashboard.py
"""
from __future__ import annotations

import os

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HOST = os.environ.get("SQ_API_HOST", "127.0.0.1")
PORT = os.environ.get("SQ_API_PORT", "8000")
BASE = f"http://{HOST}:{PORT}"


@st.cache_data(ttl=2)
def _get(path: str) -> dict | list:
    try:
        r = httpx.get(f"{BASE}{path}", timeout=2.0)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def main() -> None:
    st.set_page_config(page_title="sq_ai cockpit", layout="wide")
    st.title("sq_ai – cockpit dashboard")

    snap = _get("/api/portfolio") or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Equity", f"₹{snap.get('equity', 0):,.0f}")
    c2.metric("Cash", f"₹{snap.get('cash', 0):,.0f}")
    c3.metric("Exposure", f"{snap.get('exposure_pct', 0):.1f}%")
    c4.metric("Day P&L", f"{snap.get('daily_pnl_pct', 0):+.2f}%")

    st.subheader("Open positions")
    positions = _get("/api/positions") or []
    if positions:
        st.dataframe(pd.DataFrame(positions), use_container_width=True)
    else:
        st.info("no open positions")

    st.subheader("Latest signals")
    sigs = _get("/api/signals/latest?limit=30") or []
    if sigs:
        st.dataframe(pd.DataFrame(sigs), use_container_width=True)

    st.subheader("Last cycle")
    cyc = _get("/api/cycle/last") or {}
    st.json(cyc, expanded=False)

    st.subheader("Screener (latest)")
    sc = _get("/api/screener") or []
    if sc:
        st.dataframe(pd.DataFrame(sc), use_container_width=True)


if __name__ == "__main__":
    main()
