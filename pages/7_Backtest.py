"""Retail Backtest entry point."""
from __future__ import annotations

import streamlit as st

from ui.retail_backtest import render_retail_backtest

st.set_page_config(page_title="Backtest | QuantTerm", page_icon="🧪", layout="wide")
render_retail_backtest()
