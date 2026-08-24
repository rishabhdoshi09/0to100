"""Automatic paper-trading and plain-language market pages."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from product.market_view import current_market_view
from product.no_trade import build_no_trade_explanation
from product.scan_store import load_scan
from ui.retail_home_momentum import _money

def render_paper_trading() -> None:
    from product.paper_status import read_paper_status
    from product.autonomy_status import read_autonomy_status
    from research.autonomy.controls import (
        request_control, ENABLE_PAPER_AUTO, PAUSE_NEW_PAPER_ENTRIES,
        RESUME_NEW_PAPER_ENTRIES, RUN_CYCLE_NOW,
    )

    paper = read_paper_status()
    autonomy = read_autonomy_status()
    owner = dict(autonomy.get("owner_state", {}))
    paused = bool(owner.get("new_entries_paused", False))

    st.markdown("<div class='qt-eyebrow'>Simulated book  ·  no broker orders</div>", unsafe_allow_html=True)
    st.title("Paper Desk")
    st.info("The autonomy service takes and manages simulated trades, then learns from closed ones every day so the next paper cycle skips repeat losers. This page only reads state and queues owner controls. Live orders stay locked. After a closed loss, Backtest is still how you inspect the style — do not increase size because a name feels right.")
    status = ("PAUSED" if paused else ("RUNNING" if paper.enabled and paper.supervisor_running
              else ("READY FOR SUPERVISOR" if paper.enabled else "OFF")))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", status); c2.metric("Paper capital", _money(paper.capital))
    c3.metric("Paper equity", _money(paper.equity)); c4.metric("Open positions", len(paper.open_positions))

    a, b = st.columns(2)
    if not paper.enabled:
        if a.button("Enable automatic paper trading", type="primary", width="stretch"):
            request_control(ENABLE_PAPER_AUTO, reason="owner enabled PAPER_AUTO")
            st.success("Enable request queued."); st.rerun()
    elif paused:
        if a.button("Resume new paper trades", type="primary", width="stretch"):
            request_control(RESUME_NEW_PAPER_ENTRIES, reason="owner resumed new paper entries")
            st.success("Resume request queued."); st.rerun()
    else:
        if a.button("Pause new paper trades", width="stretch"):
            request_control(PAUSE_NEW_PAPER_ENTRIES, reason="owner paused new paper entries")
            st.success("Pause request queued. Existing positions remain manageable."); st.rerun()
    if b.button("Request one paper cycle", width="stretch"):
        request_control(RUN_CYCLE_NOW, reason="owner requested immediate paper cycle")
        st.success("Cycle queued for the autonomy supervisor.")

    st.caption(f"Risk per trade: {paper.risk_per_trade_pct:.1%} · Maximum positions: {paper.max_positions} · Open risk: {_money(paper.open_risk)}")
    if not paper.supervisor_running:
        st.warning("Autonomy supervisor is not reporting a heartbeat. Start `python main.py autonomy` or the installed service.")
    for note in autonomy.get("capability_notes", []):
        st.warning(note)

    st.subheader("Open paper positions")
    if paper.open_positions:
        st.dataframe(pd.DataFrame(list(paper.open_positions)), hide_index=True, width="stretch")
    else:
        st.info("No open paper positions.")

    from ui.desk_board import render_bot_learning, render_paper_loss_followup
    render_bot_learning()
    render_paper_loss_followup(paper.closed_trades)

    st.subheader("Why no new trade?")
    explanation = build_no_trade_explanation(
        load_scan(), list(paper.refusals)[-200:], paper.last_cycle or {}, len(paper.open_positions)
    )
    st.write(explanation.headline)
    st.dataframe(pd.DataFrame([
        {"Stage": stage.label, "Count": stage.count if stage.count is not None else "Not exposed", "Meaning": stage.detail}
        for stage in explanation.stages
    ]), hide_index=True, width="stretch")
    if explanation.top_reasons:
        st.caption("Most common final safety refusals")
        st.dataframe(pd.DataFrame(explanation.top_reasons, columns=["Reason", "Count"]), hide_index=True)

    st.subheader("Recent closed paper trades")
    if paper.closed_trades:
        st.dataframe(pd.DataFrame(list(paper.closed_trades)[-50:]), hide_index=True, width="stretch")
    else:
        st.caption("No paper trades have closed yet. A quiet book is not a failure.")


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


