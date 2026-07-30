"""Retail market context page reusing existing market modules."""
from __future__ import annotations

import streamlit as st


def render_retail_market() -> None:
    st.title("Market")
    st.caption(
        "Current market context and positioning. These views provide context; they do not bypass "
        "the strategy, evidence or risk gates."
    )

    pulse, options, institutions = st.tabs(["Market condition", "Options flow", "Institutional flows"])
    with pulse:
        try:
            from ui.street_pulse_page import render_street_pulse

            render_street_pulse()
        except Exception as exc:
            st.warning(f"Market condition is unavailable: {exc}")
    with options:
        try:
            from ui.options_flow_scanner import render_options_flow_scanner

            render_options_flow_scanner()
        except Exception as exc:
            st.warning(f"Options-flow view is unavailable: {exc}")
    with institutions:
        try:
            from ui.fii_dii_page import render_fii_dii_page

            render_fii_dii_page()
        except Exception as exc:
            st.warning(f"Institutional-flow view is unavailable: {exc}")
