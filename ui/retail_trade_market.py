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
    from product.forward_evidence_status import read_forward_evidence_dashboard
    from research.autonomy.controls import (
        request_control, ENABLE_PAPER_AUTO, PAUSE_NEW_PAPER_ENTRIES,
        RESUME_NEW_PAPER_ENTRIES, RUN_CYCLE_NOW,
    )

    paper = read_paper_status()
    autonomy = read_autonomy_status()
    owner = dict(autonomy.get("owner_state", {}))
    paused = bool(owner.get("new_entries_paused", False))
    try:
        fwd = read_forward_evidence_dashboard()
    except Exception:
        fwd = {"guide": [], "policies": [], "system": {}}

    st.title("Automatic Paper Trading")
    st.info(
        "Paper trading is for learning. QuantTerm can take simulated trades automatically. "
        "Strategies are being observed, not trusted yet. No real money is being used."
    )
    status = ("PAUSED" if paused else ("RUNNING" if paper.enabled and paper.supervisor_running
              else ("READY FOR SUPERVISOR" if paper.enabled else "OFF")))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", status)
    c2.metric("Paper capital", _money(paper.capital))
    c3.metric("Paper equity", _money(paper.equity))
    c4.metric("Open positions", len(paper.open_positions))

    sys = dict(fwd.get("system") or {})
    st.success(
        f"Forward evidence: {'ARMED' if sys.get('paper_mode_armed') else 'NOT ARMED'} · "
        f"Live trading: NOT AUTHORIZED · "
        f"Paper outcomes recorded: {sys.get('paper_outcomes_recorded', 0)}"
    )

    a, b = st.columns(2)
    if not paper.enabled:
        if a.button("Enable automatic paper trading", type="primary", width="stretch"):
            request_control(ENABLE_PAPER_AUTO, reason="owner enabled PAPER_AUTO")
            st.success("Enable request queued.")
            st.rerun()
    elif paused:
        if a.button("Resume new paper trades", type="primary", width="stretch"):
            request_control(RESUME_NEW_PAPER_ENTRIES, reason="owner resumed new paper entries")
            st.success("Resume request queued.")
            st.rerun()
    else:
        if a.button("Pause new paper trades", width="stretch"):
            request_control(PAUSE_NEW_PAPER_ENTRIES, reason="owner paused paper entries")
            st.success("Pause request queued. Existing positions remain manageable.")
            st.rerun()
    if b.button("Request one paper cycle", width="stretch"):
        request_control(RUN_CYCLE_NOW, reason="owner requested immediate paper cycle")
        st.success("Cycle queued for the autonomy supervisor.")

    st.caption(
        f"Risk per trade: {paper.risk_per_trade_pct:.1%} · "
        f"Maximum positions: {paper.max_positions} · Open risk: {_money(paper.open_risk)}"
    )
    if not paper.supervisor_running:
        st.warning(
            "Autonomy supervisor is not reporting a heartbeat. "
            "Start `python main.py autonomy` or the installed service."
        )
    for note in autonomy.get("capability_notes", []):
        st.warning(note)

    st.subheader("What are we learning?")
    for line in (fwd.get("guide") or [])[:8]:
        st.write(f"- {line}")

    st.subheader("Policies under paper observation")
    cards = list(fwd.get("policies") or [])
    if cards:
        rows = []
        for c in cards:
            rep = c.get("report") or {}
            pf = rep.get("paper_forward") or {}
            rows.append({
                "Policy": c.get("policy"),
                "Scientific status": c.get("scientific_status"),
                "Paper observation": c.get("paper_observation"),
                "Real money": c.get("live_status"),
                "Paper trades": pf.get("n", 0),
                "Paper expectancy (R)": pf.get("expectancy_r"),
                "Paper P&L": pf.get("net_pnl"),
                "Reliability": rep.get("maturity_plain"),
                "Learning": c.get("learning"),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.caption("No paper-observation policies are active yet.")

    st.subheader("Paper vs real money")
    st.caption("NO LIVE EVIDENCE YET — real broker trading is not authorized in this build.")
    if cards:
        cmp_rows = []
        for c in cards:
            pv = c.get("paper_vs_live") or {}
            cmp_rows.append({
                "Policy": c.get("policy"),
                "PAPER trades": pv.get("paper_n", 0),
                "REAL trades": "NO LIVE EVIDENCE YET",
                "PAPER expectancy": pv.get("paper_expectancy_r"),
                "REAL expectancy": "—",
                "Meaning": pv.get("plain_language"),
            })
        st.dataframe(pd.DataFrame(cmp_rows), hide_index=True, width="stretch")

    st.subheader("Open paper positions")
    if paper.open_positions:
        st.dataframe(pd.DataFrame(list(paper.open_positions)), hide_index=True, width="stretch")
    else:
        st.info("No open paper positions.")

    st.subheader("Why no new trade?")
    explanation = build_no_trade_explanation(
        load_scan(), list(paper.refusals)[-200:], paper.last_cycle or {}, len(paper.open_positions)
    )
    st.write(explanation.headline)
    st.dataframe(pd.DataFrame([
        {"Stage": stage.label,
         "Count": stage.count if stage.count is not None else "Not exposed",
         "Meaning": stage.detail}
        for stage in explanation.stages
    ]), hide_index=True, width="stretch")
    if explanation.top_reasons:
        st.caption("Most common final safety refusals")
        st.dataframe(pd.DataFrame(explanation.top_reasons, columns=["Reason", "Count"]),
                     hide_index=True)

    with st.expander("Recent closed paper trades"):
        if paper.closed_trades:
            st.dataframe(pd.DataFrame(list(paper.closed_trades)[-50:]),
                         hide_index=True, width="stretch")
        else:
            st.caption("No paper trades have closed yet.")

    with st.expander("Technical details (forward evidence)"):
        st.json({
            "forward_evidence_location": sys.get("forward_evidence_location"),
            "current_paper_policies": sys.get("current_paper_policies"),
            "denied_policies": sys.get("denied_policies"),
            "live_trading_enabled": False,
            "broker_mutations_enabled": False,
        })


def render_market() -> None:
    st.title("Market")
    st.caption("A plain-language market condition built from QuantTerm's existing regime engine.")
    try:
        with st.spinner("Reading market condition…"):
            view = current_market_view()
    except Exception as exc:
        st.warning(
            "Market condition is temporarily unavailable — the market-data feed did not respond. "
            "This does not affect your saved data or paper positions."
        )
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
