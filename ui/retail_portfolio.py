"""Retail portfolio view over existing simulated books."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from product.gather import gather_product_inputs


def render_retail_portfolio() -> None:
    st.title("Portfolio")
    st.caption(
        "Shows the canonical automatic paper book first. The older manual paper ledger is shown "
        "separately so the two sources are never blended into one misleading total."
    )

    inputs = gather_product_inputs()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Automatic-paper equity", f"₹{inputs.paper_equity:,.0f}" if inputs.paper_equity is not None else "Unknown")
    c2.metric("Open positions", len(inputs.paper_open_positions))
    c3.metric("Open risk", f"₹{inputs.paper_open_risk:,.0f}" if inputs.paper_open_risk is not None else "Unknown")
    c4.metric("Closed trades", inputs.paper_closed_trades if inputs.paper_closed_trades is not None else "Unknown")

    st.subheader("Automatic paper positions")
    if inputs.paper_open_positions:
        frame = pd.DataFrame(inputs.paper_open_positions)
        preferred = [
            "symbol", "strategy_id", "entry_date", "entry_price", "stop_price",
            "target_price", "qty", "risk_amount", "bars_held", "max_holding_days",
        ]
        columns = [name for name in preferred if name in frame.columns]
        st.dataframe(frame[columns] if columns else frame, hide_index=True, width="stretch")
    else:
        st.info("No automatic-paper positions are open.")

    st.subheader("Manual paper ledger")
    try:
        from paper_trading import get_closed_positions, get_open_positions, get_trading_summary, init_db

        init_db()
        summary = get_trading_summary() or {}
        p1, p2, p3 = st.columns(3)
        p1.metric("Manual-paper P&L", f"₹{float(summary.get('total_pnl', 0) or 0):,.0f}")
        p2.metric("Manual win rate", f"{float(summary.get('win_rate', 0) or 0):.1f}%")
        p3.metric("Manual trades", int(summary.get("total_trades", 0) or 0))

        opened = get_open_positions()
        if opened is None or opened.empty:
            st.caption("No manual paper positions are open.")
        else:
            st.dataframe(opened, hide_index=True, width="stretch")

        with st.expander("Closed manual paper trades", expanded=False):
            closed = get_closed_positions()
            if closed is None or closed.empty:
                st.caption("No closed manual paper trades.")
            else:
                st.dataframe(closed, hide_index=True, width="stretch")
    except Exception as exc:
        st.warning(f"The older manual paper ledger could not be read: {exc}")

    st.info(
        "Automatic-paper and manual-paper values are intentionally separated. They use different "
        "canonical ledgers and are not added together by the UI."
    )
