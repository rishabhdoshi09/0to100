"""Reports Streamlit page – list / generate / download."""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from sq_ai.ui._api import get, post


def render() -> None:
    st.title("Daily market reports")

    if st.button("⚙ Generate report now"):
        with st.spinner("running…"):
            r = post("/api/reports/generate")
        if "error" in r:
            st.error(r)
        else:
            st.success(f"generated {r['filename']}")
            st.write(r.get("narrative_preview", ""))

    rows = get("/api/reports/list") or []
    if rows:
        st.subheader("Report archive")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        host = os.environ.get("SQ_API_HOST", "127.0.0.1")
        port = os.environ.get("SQ_API_PORT", "8000")
        for r in rows[:10]:
            st.markdown(
                f"- [{r['filename']}]"
                f"(http://{host}:{port}/api/reports/download/{r['filename']}) "
                f"– {r.get('generated_at', '')}"
            )
    else:
        st.caption("no reports yet — APScheduler runs at 17:30 IST or click the button.")
