"""Collapsed Reco-style pages: Setups, Backtest, and the rest of the desk."""
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
    render_learned,
    render_long_term,
    render_market,
    render_momentum,
    render_news,
    render_reports,
    render_settings,
)


def render_sepa_setups() -> None:
    from ui.desk_board import render_sepa_best_setups
    from product.scan_store import load_scan

    st.caption(
        "Best Setups ranks the last scan on Minervini's 7-rule Stage-2 template "
        "(near 52-week highs, rising 200-DMA). This does not change paper or live orders."
    )
    render_sepa_best_setups(
        scan_payload=load_scan(),
        limit=24,
        score_cap=80,
        max_seconds=20.0,
        heading="Best Setups · SEPA qualified",
    )


def render_setups() -> None:
    st.markdown("<div class='qt-eyebrow'>Trade ideas</div>", unsafe_allow_html=True)
    st.title("Setups")
    st.caption(
        "Four lists, four jobs. Best Setups is SEPA-qualified Stage-2. Momentum is today's tape. "
        "Conviction needs extra confirmation. Long-term is weeks-to-months — not a day trade."
    )
    sepa, momentum, conviction, long_term = st.tabs(
        ["Best Setups", "Momentum", "Conviction", "Long-term"]
    )
    with sepa:
        render_sepa_setups()
    with momentum:
        render_momentum()
    with conviction:
        render_conviction()
    with long_term:
        render_long_term()


def render_desk_backtest() -> None:
    test, learned = st.tabs(["Run a backtest", "What we learned"])
    with test:
        render_backtest()
    with learned:
        st.caption("Research memory only. It does not change today's BUY list or paper autopilot.")
        render_learned()


def render_desk() -> None:
    st.markdown("<div class='qt-eyebrow'>Operations</div>", unsafe_allow_html=True)
    st.title("Desk")
    st.caption("Market, data, alerts, and lab tools. Daily trading lives on Today, Setups, Paper, and Backtest.")
    market, news, data, alerts, autonomy, lab, reports, settings, help_tab = st.tabs(
        ["Market", "News", "Data", "Alerts", "Autonomy", "Lab", "Reports", "Settings", "Help"]
    )
    with market:
        render_market()
    with news:
        render_news()
    with data:
        render_data_zerodha()
    with alerts:
        render_alerts()
    with autonomy:
        render_autonomy()
    with lab:
        render_advanced()
    with reports:
        render_reports()
    with settings:
        render_settings()
    with help_tab:
        render_help()
