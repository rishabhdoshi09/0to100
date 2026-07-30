"""Retail data and broker readiness entry point."""
from __future__ import annotations

import streamlit as st

from ui.data_broker_page import render_data_broker_page

st.set_page_config(page_title="Data & Broker | QuantTerm", page_icon="🔌", layout="wide")
render_data_broker_page()
