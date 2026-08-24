"""Retail-completion pages: fast saved scans, beginner backtest and plain market view."""
from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from product import build_product_state, gather_product_inputs
from product.market_view import current_market_view
from product.no_trade import build_no_trade_explanation
from product.retail_backtest import BacktestRequest, run_beginner_backtest
from product.scan_store import build_scan_payload, load_scan, save_scan, scan_age_hours, watchlist_rows
from ui.retail_pages import (
    render_advanced,
    render_alerts,
    render_help,
    render_learned,
    render_portfolio,
    render_reports,
    render_settings,
)


def _money(value: float) -> str:
    return f"₹{float(value):,.0f}"


def _run_activation(*, automatic: bool = False):
    """Queue a supervisor-owned data refresh; never start a worker from Streamlit."""
    from research.autonomy.controls import request_control, REFRESH_DATA_NOW
    control = request_control(REFRESH_DATA_NOW,
                              reason="automatic retail readiness request" if automatic else
                                     "owner requested data refresh from retail UI")
    st.session_state["last_activation_control"] = control.control_id
    st.success("Market-data refresh queued for the autonomy supervisor.")
    return control


def _maybe_auto_activate(inputs) -> None:
    """Read-only by design: page rendering never queues or starts background work."""
    return None


def _watchlist_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Stock": row.get("symbol"),
            "Company": row.get("company"),
            "Plan": row.get("status"),
            "Price": row.get("price"),
            "Entry": row.get("entry"),
            "Stop": row.get("stop"),
            "Target": row.get("target"),
            "F&O": "Available" if row.get("fno_available") else "Cash only",
            "Why": row.get("why"),
        }
        for row in rows
    ])


def render_home() -> None:
    from ui.desk_board import (
        render_how_the_desk_works,
        render_market_strip,
        render_paper_loss_followup,
        render_sepa_best_setups,
        render_today_board,
    )

    inputs = gather_product_inputs()
    _maybe_auto_activate(inputs)
    state = build_product_state(inputs)

    st.markdown("<div class='qt-eyebrow'>QuantTerm  ·  NSE desk</div>", unsafe_allow_html=True)
    st.title("Today")
    st.subheader(state.headline)
    st.caption(state.readiness)
    st.write(state.activity)
    if state.attention:
        st.warning(state.attention)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Zerodha", "Connected" if inputs.kite_connected else "Login needed")
    c2.metric("Market data", "Ready" if inputs.data_ready else "Not ready", inputs.latest_market_date or None)
    c3.metric("Paper trading", "ON" if inputs.paper_auto_enabled else "OFF")
    c4.metric("Open paper positions", inputs.open_positions)

    if state.primary_key == "connect":
        try:
            from data.kite_client import KiteClient
            st.link_button("Connect Zerodha", KiteClient().login_url(), type="primary")
        except Exception as exc:
            st.error(f"Zerodha login could not be opened: {exc}")
    elif state.primary_key == "update_data":
        if st.button("Retry Market Data Update", type="primary", width="stretch"):
            st.session_state["retail_auto_activation_attempted"] = True
            _run_activation()
    elif state.primary_key == "paper":
        if st.button("Enable Automatic Paper Trading", type="primary", width="stretch"):
            from research.autonomy.controls import request_control, ENABLE_PAPER_AUTO
            request_control(ENABLE_PAPER_AUTO, reason="owner enabled PAPER_AUTO from Home")
            st.success("PAPER_AUTO enable request queued."); st.rerun()
    elif state.primary_key == "start_worker":
        st.warning("The autonomy service is not running. Start `python main.py autonomy` or the installed service.")
    elif state.primary_key == "backtest":
        st.success("Research mode is ready. Open Backtest from the sidebar — that is how paper losses become better next trades.")
    else:
        st.success(state.primary_action)

    render_how_the_desk_works()
    render_market_strip()
    payload = load_scan()
    render_sepa_best_setups(scan_payload=payload, limit=8, score_cap=24, max_seconds=8.0)
    render_today_board(scan_payload=payload, limit=6)
    rows = watchlist_rows(payload, limit=8)
    if rows:
        with st.expander("Watchlist table"):
            st.dataframe(_watchlist_frame(rows), hide_index=True, width="stretch")

    try:
        from product.paper_status import read_paper_status
        render_paper_loss_followup(read_paper_status().closed_trades)
    except Exception:
        pass

    if not inputs.market_open:
        st.divider()
        st.subheader("After the close")
        st.write("1. Review Paper Desk losses.  2. Open Backtest on those names.  3. Keep tomorrow's watchlist — do not chase.")


def _run_and_save_momentum() -> dict:
    """Foreground UI adapter over the canonical Streamlit-free scan service."""
    from scan.market_scan_service import run_whole_market_scan

    progress = st.progress(0.0, text="Preparing whole-market scan…")

    def update(done, total):
        progress.progress(min(1.0, done / max(total, 1)),
                          text=f"Loading {done:,} of {total:,} stocks…")

    report = run_whole_market_scan(progress_callback=update, save=True)
    progress.empty()
    if not report.ok:
        raise RuntimeError(report.error_message or report.error_code or "whole-market scan failed")
    return report.payload


def _records_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Stock": row.get("symbol"),
            "Company": row.get("company"),
            "Status": row.get("status"),
            "Price": row.get("price"),
            "Momentum 5d %": row.get("momentum_5d"),
            "Score": row.get("score"),
            "Volume vs normal": f"{float(row.get('volume_ratio', 0) or 0):.2f}x",
            "F&O": "Available" if row.get("fno_available") else "Cash only",
            "Why": row.get("why"),
        }
        for row in rows
    ])


def render_momentum() -> None:
    st.title("Momentum Stocks")
    st.caption("The broad approved NSE cash universe is scanned first. F&O availability is added afterwards.")

    payload = load_scan()
    left, right = st.columns([3, 1])
    if payload:
        age = scan_age_hours(payload)
        left.caption(f"Last saved scan: {payload.get('scanned_at', '')[:19]}" +
                     (f" · {age:.1f} hours old" if age is not None else ""))
    else:
        left.caption("No saved scan exists yet.")
    if right.button("Run fresh scan", type="primary", width="stretch"):
        from research.autonomy.controls import request_control, RUN_SCAN_NOW
        request_control(RUN_SCAN_NOW, reason="owner requested fresh whole-market scan")
        st.success("Whole-market scan queued for the autonomy supervisor.")

    if not payload:
        st.info("Run the first scan. QuantTerm will save the results for Home and tomorrow's watchlist.")
        return

    records = list(payload.get("records", []))
    summary = dict(payload.get("summary", {}))
    momentum = [r for r in records if "MOMENTUM" in r.get("signals", [])]
    fno_momentum = [r for r in momentum if r.get("fno_available")]
    cash_momentum = [r for r in momentum if not r.get("fno_available")]
    near = [r for r in records if "PRE_BREAKOUT" in r.get("signals", []) and "MOMENTUM" not in r.get("signals", [])]
    extended = [r for r in records if r.get("chase_risk")]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks scanned", payload.get("universe_size", 0))
    c2.metric("Momentum candidates", summary.get("momentum", 0))
    c3.metric("F&O momentum", summary.get("fno_momentum", 0))
    c4.metric("Entry ready", summary.get("ready_to_trade", 0))

    tabs = st.tabs(["All Momentum", "F&O Momentum", "Cash Only", "Near Breakout", "Avoid for Now", "Tomorrow's Watchlist"])
    groups = [momentum, fno_momentum, cash_momentum, near, extended]
    for tab, rows in zip(tabs[:5], groups):
        with tab:
            if rows:
                st.dataframe(_records_frame(rows), hide_index=True, width="stretch")
            else:
                st.info("No stocks currently match this view. Weak candidates are not forced into the list.")
    with tabs[5]:
        watch = watchlist_rows(payload, limit=40)
        st.dataframe(_watchlist_frame(watch), hide_index=True, width="stretch") if watch else st.info("No watchlist candidates.")

    if st.button("Run complete current F&O-underlying audit"):
        st.session_state["show_full_fno_funnel"] = True
    if st.session_state.get("show_full_fno_funnel"):
        with st.expander("Complete F&O evaluation funnel", expanded=True):
            from ui.fno_momentum_page import render_fno_momentum_page
            render_fno_momentum_page()


