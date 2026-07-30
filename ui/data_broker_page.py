"""Retail data and broker readiness journey."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from product.gather import gather_product_inputs
from product.projection import ProductStatus, build_home_projection


def _row(name: str, status: ProductStatus, explanation: str, evidence: str = "") -> dict:
    return {
        "Component": name,
        "Status": status.value,
        "What it means": explanation,
        "Evidence": evidence or "—",
    }


def render_data_broker_page() -> None:
    st.title("Data & Broker")
    st.caption(
        "Research data, live screening data and broker connectivity are separate. A problem in one "
        "area is not silently converted into a failure—or a success—in another."
    )

    inputs = gather_product_inputs()
    home = build_home_projection(inputs)

    rows = [
        _row(
            "Historical market data",
            home.research.status,
            home.research.summary,
            (
                f"Snapshot {inputs.snapshot_id} · last trading date "
                f"{inputs.snapshot_last_trading_date or 'unknown'}"
                if inputs.snapshot_id else "No active snapshot"
            ),
        ),
        _row(
            "Live market data",
            home.live.status,
            home.live.summary,
            inputs.live_data_timestamp.isoformat() if inputs.live_data_timestamp else "Not verified",
        ),
        _row(
            "Zerodha data connection",
            home.broker.status,
            home.broker.summary,
            "Data-only facade; no live-order permission is inferred",
        ),
        _row(
            "F&O instrument master",
            (
                ProductStatus.READY
                if inputs.instrument_master_source not in (None, "unavailable")
                else ProductStatus.MISSING
            ),
            (
                "Current/cached instrument data is available."
                if inputs.instrument_master_source not in (None, "unavailable")
                else "Instrument data is unavailable; QuantTerm will not show a made-up shortlist."
            ),
            f"{inputs.instrument_master_source or 'unknown'} · {inputs.instrument_master_count or 0:,} rows",
        ),
        _row(
            "PAPER_AUTO eligibility",
            home.paper.status,
            home.paper.summary,
            f"Mode {inputs.paper_mode or 'unknown'} · reconciled {inputs.runtime_reconciled}",
        ),
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader("What each data type is for")
    a, b, c = st.columns(3)
    with a:
        st.markdown("**Historical research data**")
        st.write(
            "Immutable snapshots used for reproducible backtests and evidence. They must be verified; "
            "live quotes are not a substitute."
        )
    with b:
        st.markdown("**Live screening data**")
        st.write(
            "Current prices and instrument lists used to screen today. They may be unavailable while "
            "historical research remains fully usable."
        )
    with c:
        st.markdown("**Broker connectivity**")
        st.write(
            "The existing Zerodha session can provide market data. A connected session does not imply "
            "that QuantTerm has permission to place live orders."
        )

    st.subheader("Available actions")
    st.info(
        "Use the existing Historical Data Setup and Zerodha login flows already present in Advanced. "
        "This retail page intentionally does not create fake Connect, Repair or Refresh actions."
    )

    with st.expander("Snapshot evidence", expanded=False):
        st.json(
            {
                "snapshot_id": inputs.snapshot_id,
                "verified": inputs.snapshot_verified,
                "last_trading_date": str(inputs.snapshot_last_trading_date or ""),
                "instrument_count": inputs.snapshot_instrument_count,
                "has_benchmark": inputs.snapshot_has_benchmark,
                "has_universe_history": inputs.snapshot_has_universe_history,
                "has_corporate_actions": inputs.snapshot_has_corporate_actions,
            }
        )
