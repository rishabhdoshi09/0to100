"""QuantTerm retail-first Streamlit entrypoint.

The previous institutional terminal is preserved as legacy_app.py. This default
experience is intentionally small, plain-language and action-oriented.
"""
from __future__ import annotations

import streamlit as st

from ui.retail_pages_v2 import (
    render_advanced,
    render_alerts,
    render_autonomy,
    render_backtest,
    render_conviction,
    render_data_zerodha,
    render_help,
    render_home,
    render_learned,
    render_long_term,
    render_market,
    render_momentum,
    render_news,
    render_paper_trading,
    render_portfolio,
    render_reports,
    render_settings,
)

st.set_page_config(page_title="QuantTerm", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

try:
    import scan.market_scan_service as _mss
    from research.feature002.observe import try_observe_production_scan
    if getattr(_mss, "_feature002_hook", None) is None:
        _mss._feature002_hook = try_observe_production_scan
except Exception:
    pass

pages = {
    "Everyday": [
        st.Page(render_home, title="Home", icon="🏠", default=True),
        st.Page(render_momentum, title="Momentum Stocks", icon="📈"),
        st.Page(render_conviction, title="Conviction Buying", icon="🎯"),
        st.Page(render_long_term, title="Long-Term Picks", icon="💎"),
        st.Page(render_news, title="Market News", icon="📰"),
        st.Page(render_paper_trading, title="Automatic Paper Trading", icon="🤖"),
        st.Page(render_autonomy, title="Autonomy", icon="🛰️"),
        st.Page(render_portfolio, title="Portfolio", icon="💼"),
        st.Page(render_market, title="Market", icon="🌐"),
    ],
    "Learn and Test": [
        st.Page(render_backtest, title="Backtest", icon="🧪"),
        st.Page(render_learned, title="What We’ve Learned", icon="📚"),
        st.Page(render_reports, title="Reports", icon="📊"),
    ],
    "System": [
        st.Page(render_data_zerodha, title="Data and Zerodha", icon="🔌"),
        st.Page(render_alerts, title="Alerts", icon="🔔"),
        st.Page(render_settings, title="Settings", icon="⚙️"),
        st.Page(render_help, title="Help", icon="❓"),
    ],
    "Advanced": [
        st.Page(render_advanced, title="Research Laboratory", icon="🔬"),
    ],
}

navigation = st.navigation(pages, position="sidebar")
navigation.run()
