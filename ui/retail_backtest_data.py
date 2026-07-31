"""Beginner backtest and automatic Zerodha-data pages."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from product import gather_product_inputs
from product.retail_backtest import BacktestRequest, run_beginner_backtest
from ui.retail_home_momentum import _money, _maybe_auto_activate, _run_activation

def _show_backtest_result(result: dict) -> None:
    st.subheader("What happened to the paper money?")
    a, b, c, d = st.columns(4)
    a.metric("Starting amount", _money(result["starting_amount"]))
    b.metric("Ending amount", _money(result["ending_amount"]), f"{result['return_pct']:+.1f}%")
    c.metric("Completed trades", result["round_trips"], f"{result['wins']} wins · {result['losses']} losses")
    d.metric("Largest fall", f"{result['largest_fall_pct']:.1f}%")
    st.write(f"**Trading costs and slippage modelled:** {_money(result['trading_costs'])}")
    st.write(f"**Nifty comparison:** {result['comparison']}")
    st.warning(f"**Trustworthiness:** {result['trustworthiness']}")
    st.info(result["conclusion"])

    curve = pd.DataFrame(result.get("equity_curve", []))
    if not curve.empty and "equity" in curve.columns:
        time_col = "timestamp" if "timestamp" in curve.columns else ("date" if "date" in curve.columns else None)
        if time_col:
            curve[time_col] = pd.to_datetime(curve[time_col])
            st.line_chart(curve.set_index(time_col)["equity"])
    with st.expander("See scientific details"):
        st.json(result.get("scientific_details", {}))
        if result.get("walk_forward"):
            st.markdown("**Walk-forward reliability check**")
            st.json(result["walk_forward"])
    with st.expander("See trade log"):
        trades = result.get("trades", [])
        st.dataframe(pd.DataFrame(trades), hide_index=True, width="stretch") if trades else st.caption("No completed trades.")


def render_backtest() -> None:
    st.title("Backtest")
    st.caption("Four simple choices. The existing event-driven engine handles next-bar fills, slippage, costs, portfolio state and risk checks.")
    with st.form("beginner_backtest_form"):
        c1, c2 = st.columns(2)
        strategy = c1.selectbox("Strategy", ["Core technical strategy", "Core technical + walk-forward reliability check"])
        universe = c2.selectbox("Universe", ["Selected stock", "Nifty 50", "Current F&O stocks"])
        c3, c4, c5 = st.columns(3)
        symbol = c3.text_input("Stock", value="RELIANCE", disabled=universe != "Selected stock")
        period = c4.selectbox("Period", ["1 year", "2 years", "3 years", "5 years"], index=2)
        capital = c5.number_input("Starting paper capital", min_value=10_000.0, value=100_000.0, step=10_000.0)
        run = st.form_submit_button("Run Backtest", type="primary", width="stretch")
    if run:
        days = {"1 year": 252, "2 years": 504, "3 years": 756, "5 years": 1260}[period]
        request = BacktestRequest(strategy=strategy, universe=universe, symbol=symbol, days=days, capital=float(capital))
        progress = st.progress(0.0, text="Loading historical data…")
        def update(done, total):
            progress.progress(min(1.0, done / max(total, 1)), text=f"Loading {done} of {total} stocks…")
        with st.spinner("Running the real QuantTerm backtest engine…"):
            result = run_beginner_backtest(request, progress=update).as_dict()
        progress.empty(); st.session_state["retail_backtest_result"] = result
    result = st.session_state.get("retail_backtest_result")
    if result:
        _show_backtest_result(result)
    with st.expander("Advanced strategy laboratory"):
        st.warning("This section is for code-level strategy experiments.")
        try:
            from ui.algolab import render_algolab
            render_algolab()
        except Exception as exc:
            st.error(str(exc))


def render_data_zerodha() -> None:
    st.title("Data and Zerodha")
    inputs = gather_product_inputs(); _maybe_auto_activate(inputs)
    c1, c2, c3 = st.columns(3)
    c1.metric("Zerodha", "Connected" if inputs.kite_connected else "Login needed")
    c2.metric("Saved market data", "Ready" if inputs.data_ready else "Not ready", inputs.latest_market_date or None)
    c3.metric("Stocks in active data", inputs.instrument_count)
    if not inputs.kite_connected:
        try:
            from data.kite_client import KiteClient
            st.link_button("Open Zerodha Login", KiteClient().login_url(), type="primary")
        except Exception as exc:
            st.error(str(exc))
    if st.button("Retry download / update", type="primary", width="stretch"):
        st.session_state["retail_auto_activation_attempted"] = True
        _run_activation()
    st.caption("The autonomy supervisor owns data refresh. This button queues an immediate owner-requested refresh.")
