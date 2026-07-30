"""Retail alerts wrapper over existing actionable alert modules."""
from __future__ import annotations

import streamlit as st


def render_retail_alerts() -> None:
    st.title("Alerts")
    st.caption(
        "Actionable user alerts are shown first. Internal daemon and research diagnostics remain in Advanced."
    )
    telegram, inbox = st.tabs(["Alert settings", "Alert inbox"])
    with telegram:
        try:
            from ui.alerts_page import render_alerts_page

            render_alerts_page()
        except Exception as exc:
            st.warning(f"Alert settings are unavailable: {exc}")
    with inbox:
        try:
            from ui.alert_inbox import render_alert_inbox

            render_alert_inbox()
        except Exception as exc:
            st.warning(f"Alert inbox is unavailable: {exc}")
