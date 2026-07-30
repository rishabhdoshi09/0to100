"""Retail Alerts entry point."""
from __future__ import annotations

import streamlit as st

from ui.retail_alerts import render_retail_alerts

st.set_page_config(page_title="Alerts | QuantTerm", page_icon="🔔", layout="wide")
render_retail_alerts()
