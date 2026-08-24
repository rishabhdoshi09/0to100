"""QuantTerm Reco-style Streamlit entrypoint.

Six work pages. The previous 16-item consumer sidebar is collapsed. The
institutional dark theme is injected here so every page looks like a desk,
not a hobby app. FEATURE-002 remains a fail-open observer only.
"""
from __future__ import annotations

import streamlit as st

from ui.desk_pages import render_desk, render_desk_backtest, render_setups
from ui.retail_pages_v2 import render_home, render_paper_trading, render_portfolio
from ui.theme import DEVBLOOM_CSS

st.set_page_config(page_title="QuantTerm", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")
st.markdown(DEVBLOOM_CSS, unsafe_allow_html=True)

try:
    import scan.market_scan_service as _mss
    from research.feature002.observe import try_observe_production_scan
    if getattr(_mss, "_feature002_hook", None) is None:
        _mss._feature002_hook = try_observe_production_scan
except Exception:
    pass

pages = [
    st.Page(render_home, title="Today", icon="⚡", default=True),
    st.Page(render_setups, title="Setups", icon="📈"),
    st.Page(render_paper_trading, title="Paper Desk", icon="📋"),
    st.Page(render_desk_backtest, title="Backtest", icon="🧪"),
    st.Page(render_portfolio, title="Portfolio", icon="💼"),
    st.Page(render_desk, title="Desk", icon="🖥️"),
]

navigation = st.navigation(pages, position="sidebar")
navigation.run()
