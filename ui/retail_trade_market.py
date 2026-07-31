"""Automatic paper-trading and plain-language market pages."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from product.market_view import current_market_view
from product.no_trade import build_no_trade_explanation
from product.scan_store import load_scan
from ui.retail_home_momentum import _money

def render_paper_trading() -> None:
    from research.auto_research.scheduler import get_brain
    brain = get_brain(); book = brain.intel_book
    enabled = brain.is_paper_auto_enabled(); running = brain.state.running

    st.title("Automatic Paper Trading")
    st.info("QuantTerm takes and manages paper trades automatically. You do not approve every trade.")
    status = "RUNNING" if enabled and running else ("READY FOR NEXT SESSION" if enabled else "OFF")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", status); c2.metric("Paper capital", _money(book.capital))
    c3.metric("Paper equity", _money(book.equity())); c4.metric("Open positions", len(book.open))

    a, b = st.columns(2)
    if enabled:
        if a.button("Pause new paper trades", width="stretch"):
            brain.disable_paper_auto(); st.rerun()
    elif a.button("Enable automatic paper trading", type="primary", width="stretch"):
        brain.enable_paper_auto(); brain.start(); st.rerun()
    if b.button("Start / verify worker", width="stretch"):
        brain.start(); st.rerun()

    st.caption(f"Risk per trade: {book.risk_per_trade_pct:.1%} · Maximum positions: {book.max_positions} · Open risk: {_money(book.open_risk())}")
    if brain.state.last_error:
        st.error(brain.state.last_error)

    st.subheader("Open paper positions")
    if book.open:
        st.dataframe(pd.DataFrame([p.as_dict() for p in book.open.values()]), hide_index=True, width="stretch")
    else:
        st.info("No open paper positions.")

    st.subheader("Why no new trade?")
    explanation = build_no_trade_explanation(
        load_scan(), book.refusals[-200:], brain.state.last_intel_cycle or {}, len(book.open)
    )
    st.write(explanation.headline)
    st.dataframe(pd.DataFrame([
        {"Stage": stage.label, "Count": stage.count if stage.count is not None else "Not exposed", "Meaning": stage.detail}
        for stage in explanation.stages
    ]), hide_index=True, width="stretch")
    if explanation.top_reasons:
        st.caption("Most common final safety refusals")
        st.dataframe(pd.DataFrame(explanation.top_reasons, columns=["Reason", "Count"]), hide_index=True)

    with st.expander("Recent closed paper trades"):
        if book.closed:
            st.dataframe(pd.DataFrame([t.as_dict() for t in book.closed[-50:]]), hide_index=True, width="stretch")
        else:
            st.caption("No paper trades have closed yet.")


def render_market() -> None:
    st.title("Market")
    st.caption("A plain-language market condition built from QuantTerm's existing regime engine.")
    try:
        with st.spinner("Reading market condition…"):
            view = current_market_view()
    except Exception as exc:                       # unavailable feed → understandable state, not a hang/crash
        st.warning("Market condition is temporarily unavailable — the market-data feed did not respond. "
                   "This does not affect your saved data or paper positions.")
        st.caption(f"Details: {exc}")
        return
    st.subheader(f"Market is {view.health.lower()}")
    st.write(view.summary)
    st.info(view.trade_stance)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market health", view.health)
    c2.metric("Breadth", view.breadth)
    c3.metric("Nifty today", f"{view.nifty_change_1d:+.2f}%")
    c4.metric("India VIX", f"{view.vix:.1f}" if view.vix else "Unavailable")
    a, b = st.columns(2)
    a.write("**Leading sectors:** " + (", ".join(view.leaders) if view.leaders else "No clear leader"))
    b.write("**Lagging sectors:** " + (", ".join(view.laggards) if view.laggards else "No clear laggard"))
    with st.expander("See scientific market details"):
        st.json(view.technical_details)
    with st.expander("Open the old institutional market view"):
        try:
            from ui.market_narrative import render_market_narrative
            render_market_narrative()
        except Exception as exc:
            st.caption(f"Institutional view unavailable: {exc}")


