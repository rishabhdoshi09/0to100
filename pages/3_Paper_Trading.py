"""Retail PAPER_AUTO control entry point."""
from __future__ import annotations

import streamlit as st

from ui.paper_trading_page import render_paper_trading_page

st.set_page_config(page_title="Paper Trading | QuantTerm", page_icon="🧪", layout="wide")
render_paper_trading_page()
