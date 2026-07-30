"""Retail Reports entry point."""
from __future__ import annotations

import streamlit as st

from ui.retail_reports import render_retail_reports

st.set_page_config(page_title="Reports | QuantTerm", page_icon="📊", layout="wide")
render_retail_reports()
