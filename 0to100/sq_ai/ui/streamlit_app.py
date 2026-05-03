"""sq_ai – unified Streamlit app.

Pages: Dashboard | Screener | Stock Research | Portfolio | Reports | Settings.
All pages call FastAPI on ``$SQ_API_HOST:$SQ_API_PORT``.
"""
from __future__ import annotations

import streamlit as st

from sq_ai.ui import (
    dashboard_page, instruments_page, portfolio_page, reports_page,
    screener_page, settings_page, stock_research_page,
)


PAGES = {
    "📊 Dashboard":        dashboard_page,
    "📋 Instruments":      instruments_page,
    "🔎 Screener":         screener_page,
    "🔬 Stock research":   stock_research_page,
    "💼 Portfolio":        portfolio_page,
    "🗞 Reports":          reports_page,
    "⚙ Settings":          settings_page,
}


def main() -> None:
    st.set_page_config(page_title="sq_ai cockpit", layout="wide",
                       initial_sidebar_state="expanded")
    st.sidebar.title("sq_ai cockpit")
    st.sidebar.caption("Bloomberg + Screener.in + Moneycontrol – on a MacBook Air")
    choice = st.sidebar.radio("Navigation", list(PAGES.keys()))
    st.sidebar.divider()
    st.sidebar.caption(
        "FastAPI on :8000 ▸ Streamlit on :8501\n"
        "Configure keys in `.env` and risk in `config.yaml`."
    )
    PAGES[choice].render()


if __name__ == "__main__":
    main()
