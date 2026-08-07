"""Direct retail entry-point for complete F&O momentum discovery."""
from __future__ import annotations

import streamlit as st

from ui.fno_momentum_page import render_fno_momentum_page

st.set_page_config(page_title="F&O Momentum | QuantTerm", page_icon="⚡", layout="wide")
render_fno_momentum_page()
