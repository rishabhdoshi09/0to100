"""Retail control surface for the canonical PAPER_AUTO runtime."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from product.gather import gather_product_inputs
from product.projection import ProductStatus, build_paper_trading_projection


def _status_box(status: ProductStatus, text: str) -> None:
    if status == ProductStatus.READY:
        st.success(text)
    elif status in {ProductStatus.MISSING, ProductStatus.STALE}:
        st.error(text)
    elif status in {ProductStatus.ATTENTION, ProductStatus.UNKNOWN}:
        st.warning(text)
    else:
        st.info(text)


def render_paper_trading_page() -> None:
    st.title("Automatic Paper Trading")
    st.caption(
        "Controls the existing PAPER_AUTO runtime. No second autonomy flag, paper book or "
        "execution path is created here. Live trading remains outside this page."
    )

    inputs = gather_product_inputs()
    view = build_paper_trading_projection(inputs)

    _status_box(view.status, f"**{view.mode_label}**\n\n{view.entries_reason}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Paper capital", f"₹{view.capital:,.0f}" if view.capital is not None else "Unknown")
    c2.metric("Paper equity", f"₹{view.equity:,.0f}" if view.equity is not None else "Unknown")
    c3.metric("Open risk", f"₹{view.open_risk:,.0f}" if view.open_risk is not None else "Unknown")
    c4.metric("Open positions", len(view.open_positions))

    st.subheader("Controls")
    left, right = st.columns(2)
    with left:
        engage = st.button(
            "Engage automatic paper trading",
            type="primary",
            disabled=not view.can_engage,
            width="stretch",
            help="Enables the canonical PAPER_AUTO flag. It does not unlock live trading.",
        )
        if engage:
            from research.auto_research import get_brain

            get_brain().enable_paper_auto()
            st.success("Automatic paper trading enabled in the canonical runtime.")
            st.rerun()
    with right:
        disengage = st.button(
            "Stop automatic paper trading",
            disabled=not view.can_disengage,
            width="stretch",
            help=(
                "Disables the canonical PAPER_AUTO cycle. Existing simulated positions remain "
                "in the paper book; review them before leaving the runtime stopped."
            ),
        )
        if disengage:
            from research.auto_research import get_brain

            get_brain().disable_paper_auto()
            st.warning("Automatic paper trading disabled in the canonical runtime.")
            st.rerun()

    st.caption(
        "These buttons call `AutoResearchBrain.enable_paper_auto()` and "
        "`disable_paper_auto()` directly. They do not create a UI-only setting."
    )

    st.subheader("Cycle status")
    s1, s2, s3 = st.columns(3)
    s1.metric("New entries", "Allowed" if view.entries_allowed else "Paused")
    s2.metric("Worker", "Running" if view.cycle_running is True else (
        "Stopped" if view.cycle_running is False else "Unknown"
    ))
    s3.metric("Last completed cycle", view.last_completed_cycle)
    if view.last_error:
        st.error(f"Last cycle error: {view.last_error}")

    st.subheader("Open simulated positions")
    if not view.open_positions:
        st.info("No simulated positions are currently open.")
    else:
        frame = pd.DataFrame(view.open_positions)
        preferred = [
            "symbol", "strategy_id", "entry_date", "entry_price", "stop_price",
            "target_price", "qty", "risk_amount", "bars_held", "max_holding_days",
        ]
        columns = [col for col in preferred if col in frame.columns]
        st.dataframe(frame[columns] if columns else frame, hide_index=True, width="stretch")

    with st.expander("Recent risk refusals", expanded=False):
        if not inputs.paper_refusals:
            st.caption("No recorded paper-entry refusals.")
        else:
            rows = []
            for refusal in inputs.paper_refusals[-50:]:
                symbol = refusal[0] if len(refusal) > 0 else ""
                reason = refusal[1] if len(refusal) > 1 else ""
                rows.append({"Stock": symbol, "Reason": reason})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.info(
        "This page shows concise decisions and audited state only. Internal chain-of-thought, "
        "daemon diagnostics and raw research traces belong in Advanced."
    )
