"""Retail help entry point."""
from __future__ import annotations

import streamlit as st

from ui.retail_help import render_retail_help

st.set_page_config(page_title="Help | QuantTerm", page_icon="❓", layout="wide")
render_retail_help()
