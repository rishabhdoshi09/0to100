"""Retail reports over canonical paper, evidence and data state."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from product.gather import gather_product_inputs


def render_retail_reports() -> None:
    st.title("Reports")
    st.caption("Paper performance, evidence state, decision history and data quality—without raw operator clutter.")

    paper_tab, evidence_tab, decisions_tab, data_tab = st.tabs(
        ["Paper performance", "Strategy evidence", "Decision history", "Data quality"]
    )

    try:
        from research.auto_research import get_brain

        brain = get_brain()
    except Exception as exc:
        st.error(f"Canonical runtime could not be read: {exc}")
        return

    with paper_tab:
        book = getattr(brain, "intel_book", None)
        if book is None:
            st.info("The automatic paper book is unavailable.")
        else:
            stats = book.stats()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Paper equity", f"₹{float(stats.get('equity', 0) or 0):,.0f}")
            c2.metric("Closed trades", int(stats.get("n_trades", 0) or 0))
            c3.metric("Win rate", f"{float(stats.get('win_rate', 0) or 0) * 100:.1f}%")
            c4.metric("Maximum drawdown", f"{float(stats.get('max_drawdown_pct', 0) or 0) * 100:.2f}%")
            d1, d2, d3 = st.columns(3)
            d1.metric("Expectancy", f"{float(stats.get('expectancy_R', 0) or 0):+.3f}R")
            pf = stats.get("profit_factor")
            d2.metric("Profit factor", f"{float(pf):.2f}" if pf is not None else "Unknown")
            d3.metric("Net P&L", f"₹{float(stats.get('net_pnl', 0) or 0):,.0f}")

            closed = [trade.as_dict() for trade in getattr(book, "closed", ())]
            if closed:
                st.dataframe(pd.DataFrame(closed), hide_index=True, width="stretch")
            else:
                st.info("No automatic-paper trades have closed yet.")

    with evidence_tab:
        store = getattr(brain, "event_store", None)
        if store is None:
            st.info("The canonical evidence event store is unavailable.")
        else:
            cards = list(store.latest_cards().values())
            if not cards:
                st.info("No current strategy evidence cards are available.")
            else:
                rows = []
                for card in cards:
                    raw = card.as_dict() if hasattr(card, "as_dict") else vars(card)
                    rows.append(
                        {
                            "Strategy": raw.get("strategy_id"),
                            "Version": raw.get("strategy_version"),
                            "Evidence": raw.get("evidence_state"),
                            "Sample": raw.get("n_trades") or raw.get("sample_size"),
                            "Expectancy": raw.get("expectancy_R") or raw.get("expectancy_r"),
                            "Reason": raw.get("reason") or raw.get("summary"),
                        }
                    )
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("Scientific detail remains available in Advanced → Brain Observatory / Research OS.")

    with decisions_tab:
        store = getattr(brain, "event_store", None)
        events = store.all() if store is not None else []
        if not events:
            st.info("No canonical intelligence decisions have been recorded.")
        else:
            rows = []
            for event in events[-200:]:
                raw = event.as_dict() if hasattr(event, "as_dict") else vars(event)
                rows.append(
                    {
                        "Type": type(event).__name__,
                        "Cycle": raw.get("cycle_id"),
                        "Strategy": raw.get("strategy_id"),
                        "Stock": raw.get("symbol"),
                        "Decision": raw.get("action") or raw.get("decision") or raw.get("event_type"),
                        "Reason": raw.get("reason") or raw.get("summary"),
                    }
                )
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    with data_tab:
        inputs = gather_product_inputs()
        rows = [
            {"Check": "Active snapshot", "Value": inputs.snapshot_id or "Missing"},
            {"Check": "Snapshot verified", "Value": inputs.snapshot_verified},
            {"Check": "Last trading date", "Value": inputs.snapshot_last_trading_date or "Unknown"},
            {"Check": "Snapshot instruments", "Value": inputs.snapshot_instrument_count if inputs.snapshot_instrument_count is not None else "Unknown"},
            {"Check": "Benchmark available", "Value": inputs.snapshot_has_benchmark},
            {"Check": "Universe history available", "Value": inputs.snapshot_has_universe_history},
            {"Check": "Corporate actions covered", "Value": inputs.snapshot_has_corporate_actions},
            {"Check": "F&O instrument source", "Value": inputs.instrument_master_source or "Unknown"},
            {"Check": "F&O instrument rows", "Value": inputs.instrument_master_count if inputs.instrument_master_count is not None else "Unknown"},
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
