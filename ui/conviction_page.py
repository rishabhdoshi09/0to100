"""Retail conviction-buying page over the canonical saved market scan."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from product.conviction import build_conviction_shortlist
from product.scan_store import load_scan, scan_age_hours
from product.market_view import current_market_view
from research.autonomy.controls import request_control, RUN_SCAN_NOW


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Stock": r.get("symbol"),
        "Company": r.get("company"),
        "Conviction": r.get("classification", "").replace("_", " ").title(),
        "Score": r.get("conviction_score"),
        "Scanner": r.get("scanner_score"),
        "Sector": r.get("sector"),
        "Price": r.get("price"),
        "Entry": r.get("entry"),
        "Stop": r.get("stop"),
        "Target": r.get("target"),
        "Volume": f"{float(r.get('volume_ratio', 0) or 0):.1f}×",
        "RSI": r.get("rsi"),
        "Why": " · ".join(r.get("reasons", [])[:3]),
        "Risks": " · ".join(r.get("risks", [])[:3]),
    } for r in rows])


def render_conviction() -> None:
    st.title("Conviction Buying")
    st.caption("Multiple confirmations, not certainty: market regime + sector leadership + "
               "scanner quality + volume + entry discipline. LIVE orders remain locked.")

    payload = load_scan()
    try:
        market = current_market_view()
    except Exception:
        class _Fallback:
            health = "Mixed"; leaders = (); laggards = ()
            summary = "Market context temporarily unavailable."
            trade_stance = "Use only the clearest setups and paper-evaluate first."
        market = _Fallback()

    top, action = st.columns([3, 1])
    with top:
        age = scan_age_hours(payload)
        st.write(f"**Market:** {market.health} — {market.summary}")
        st.caption(market.trade_stance +
                   ((f" · Saved scan {age:.1f} hours old") if age is not None else ""))
    with action:
        if st.button("Run fresh market scan", type="primary", width="stretch"):
            request_control(RUN_SCAN_NOW, reason="owner requested fresh conviction scan")
            st.success("Scan queued for the autonomy supervisor.")

    if not payload:
        st.info("No saved market scan yet. Run the scan and refresh this page after the supervisor completes it.")
        return

    rows = build_conviction_shortlist(payload, market)
    high = [r for r in rows if r["classification"] == "HIGH_CONVICTION"]
    awaiting = [r for r in rows if r["classification"] == "AWAIT_CONFIRMATION"]
    pullback = [r for r in rows if r["classification"] == "WAIT_FOR_PULLBACK"]
    watch = [r for r in rows if r["classification"] == "WATCH"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("High conviction", len(high))
    c2.metric("Awaiting confirmation", len(awaiting))
    c3.metric("Wait for pullback", len(pullback))
    c4.metric("Other watch", len(watch))

    tabs = st.tabs(["High Conviction", "Await Confirmation", "Wait for Pullback", "All Evidence"])
    for tab, group, empty in [
        (tabs[0], high, "No stock currently has all conviction layers aligned."),
        (tabs[1], awaiting, "No candidate is waiting on confirmation."),
        (tabs[2], pullback, "No extended candidate currently needs a pullback."),
        (tabs[3], rows, "No candidates in the saved scan."),
    ]:
        with tab:
            if group:
                st.dataframe(_frame(group), hide_index=True, width="stretch")
            else:
                st.info(empty)

    st.warning("A High Conviction label is a research shortlist, not a guaranteed return. "
               "Respect the displayed stop, avoid entries far above the plan, and validate in PAPER first.")
