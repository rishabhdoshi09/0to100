"""QuantTerm retail entry point.

The retail experience is the default. Streamlit's multipage navigation exposes
one-click access to Momentum, Paper Trading, Portfolio, Market, Backtest,
Reports, Alerts, Data & Broker, Help and Advanced.
"""
from __future__ import annotations

import streamlit as st

from ui.retail_home import render_retail_home

st.set_page_config(
    page_title="QuantTerm",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.markdown("## ⚡ QuantTerm")
    st.caption("Evidence-gated retail research and automatic paper trading")
    st.info("Home is the default. Engineering and research internals are under Advanced.")

render_retail_home()
