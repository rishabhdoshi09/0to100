"""Retail Market entry point."""
from __future__ import annotations

import streamlit as st

from ui.retail_market import render_retail_market

st.set_page_config(page_title="Market | QuantTerm", page_icon="📈", layout="wide")
render_retail_market()
