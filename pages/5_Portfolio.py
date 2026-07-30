"""Retail Portfolio entry point."""
from __future__ import annotations

import streamlit as st

from ui.retail_portfolio import render_retail_portfolio

st.set_page_config(page_title="Portfolio | QuantTerm", page_icon="💼", layout="wide")
render_retail_portfolio()
