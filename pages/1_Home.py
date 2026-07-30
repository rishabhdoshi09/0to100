"""Retail Home entry point."""
from __future__ import annotations

import streamlit as st

from ui.retail_home import render_retail_home

st.set_page_config(page_title="Home | QuantTerm", page_icon="⚡", layout="wide")
render_retail_home()
