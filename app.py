"""QuantTerm professional retail-first Streamlit entrypoint.

The default product is a compact command center and unified scanner.  Existing
research and operations pages remain available behind a simpler navigation
hierarchy, while legacy_app.py preserves the old institutional terminal.
"""
from __future__ import annotations

import streamlit as st

from ui.pro_theme import apply_pro_theme, render_sidebar_brand
from ui.retail_pages_v2 import (
    render_advanced,
    render_alerts,
    render_autonomy,
    render_backtest,
    render_command_center,
    render_data_zerodha,
    render_help,
    render_learned,
    render_market,
    render_news,
    render_paper_trading,
    render_portfolio,
    render_reports,
    render_scanner_workspace,
    render_settings,
)

st.set_page_config(
    page_title="QuantTerm Professional",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_pro_theme()
render_sidebar_brand()

pages = {
    "Workspace": [
        st.Page(render_command_center, title="Command Center", icon="⚡", default=True),
        st.Page(render_scanner_workspace, title="Scanner", icon="🎯"),
        st.Page(render_portfolio, title="Portfolio", icon="💼"),
        st.Page(render_market, title="Market", icon="🌐"),
    ],
    "Research": [
        st.Page(render_learned, title="What We’ve Learned", icon="📚"),
        st.Page(render_backtest, title="Backtest", icon="🧪"),
        st.Page(render_reports, title="Reports", icon="📊"),
        st.Page(render_advanced, title="Research Laboratory", icon="🔬"),
    ],
    "Operations": [
        st.Page(render_paper_trading, title="Paper Trading", icon="🤖"),
        st.Page(render_autonomy, title="Automation", icon="🛰️"),
        st.Page(render_alerts, title="Alerts", icon="🔔"),
        st.Page(render_data_zerodha, title="Data & Zerodha", icon="🔌"),
    ],
    "More": [
        st.Page(render_news, title="Market News", icon="📰"),
        st.Page(render_settings, title="Settings", icon="⚙️"),
        st.Page(render_help, title="Help", icon="❓"),
    ],
}

navigation = st.navigation(pages, position="sidebar")
navigation.run()
