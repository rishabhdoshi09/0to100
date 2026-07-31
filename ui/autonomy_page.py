"""Retail control-room page: what the autonomous organisation is doing (read-only)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from product.autonomy_status import read_autonomy_status


def render_autonomy() -> None:
    st.title("Autonomy")
    st.caption("QuantTerm runs on its own after login. This page only reports what it is doing — it "
               "does not start or control the engine, and it never places a real order.")
    status = read_autonomy_status()

    if not status["running"]:
        st.info("The autonomous supervisor is not running. Start it with `python main.py autonomy` "
                "(it runs independently of this browser).")
    dot = {"OBSERVING": "🟢", "PAPER_ACTIVE": "🟢", "DATA_READY": "🟢", "RESEARCHING": "🔵",
           "DEGRADED": "🟠", "DATA_BLOCKED": "🟠", "AUTH_REQUIRED": "⚪", "HALTED": "🔴"}.get(
        status["state"], "⚪")
    st.subheader(f"{dot} {status['plain_state']}")
    if status["explanation"]:
        st.write(status["explanation"])

    c1, c2, c3 = st.columns(3)
    c1.metric("New paper entries", status["new_paper_entries"].title())
    c2.metric("Existing positions", status["existing_exits"].title())
    c3.metric("Research", status["research"].title())
    for note in status["capability_notes"]:
        st.warning(note)
    if status["heartbeat_ist"]:
        st.caption(f"Last heartbeat: {status['heartbeat_ist']} · Active data: {status['snapshot_id'] or '—'}")

    if status["jobs"]:
        st.markdown("**Scheduled work**")
        st.dataframe(pd.DataFrame([{"Status": k, "Jobs": v} for k, v in status["jobs"].items()]),
                     hide_index=True, width="stretch")

    if status["recent_dialogue"]:
        st.markdown("**What the system is saying to itself** (typed records, not chat)")
        st.dataframe(pd.DataFrame(status["recent_dialogue"]), hide_index=True, width="stretch")

    if status["recent_transitions"]:
        with st.expander("Recent state changes"):
            st.dataframe(pd.DataFrame([
                {"When (IST)": t.get("at_ist", "")[:19], "From": t.get("from_state"),
                 "To": t.get("to_state"), "Why": t.get("explanation")}
                for t in status["recent_transitions"]]), hide_index=True, width="stretch")
