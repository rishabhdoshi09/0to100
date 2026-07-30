"""Retail workflow over QuantTerm's real event-driven Backtester."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from product.backtest_view import summarize_backtest


def _symbols() -> list[str]:
    try:
        from data.nse_universe import get_nse_universe

        return list(get_nse_universe())
    except Exception:
        return []


def _load_bhavcopy(symbols: list[str], start: date, end: date) -> tuple[dict, dict]:
    from data.bhavcopy_store import get_ohlcv, is_ready

    if not is_ready():
        return {}, {symbol: "Historical bhavcopy store is not ready" for symbol in symbols}
    data = {}
    excluded = {}
    for symbol in symbols:
        frame = get_ohlcv(symbol)
        if frame is None:
            excluded[symbol] = "No historical series in the canonical bhavcopy store"
            continue
        mask = (frame.index.date >= start) & (frame.index.date <= end)
        sliced = frame.loc[mask, ["open", "high", "low", "close", "volume"]].dropna()
        if len(sliced) < 60:
            excluded[symbol] = f"Only {len(sliced)} sessions; at least 60 are required"
            continue
        data[symbol] = sliced
    return data, excluded


def render_retail_backtest() -> None:
    st.title("Backtest")
    st.caption(
        "Runs the repository's real event-driven engine: signals use information available at each "
        "bar and orders fill on the next bar. No illustrative or fabricated result is shown."
    )

    with st.form("retail_backtest_form"):
        strategy = st.selectbox(
            "Strategy",
            ["Canonical conviction swing"],
            help="This maps directly to backtest.backtester.Backtester; no UI-only calculator exists.",
        )
        universe = st.multiselect(
            "Stocks",
            _symbols(),
            default=[symbol for symbol in ("RELIANCE", "TCS", "HDFCBANK") if symbol in _symbols()],
            max_selections=20,
        )
        d1, d2 = st.columns(2)
        start = d1.date_input("Start date", value=date.today() - timedelta(days=730))
        end = d2.date_input("End date", value=date.today())
        a, b, c = st.columns(3)
        capital = a.number_input("Starting capital (₹)", min_value=10_000.0, value=1_000_000.0, step=50_000.0)
        slippage_pct = b.number_input("Slippage per side (%)", min_value=0.0, max_value=2.0, value=0.05, step=0.01)
        cost_pct = c.number_input("Transaction cost per side (%)", min_value=0.0, max_value=2.0, value=0.10, step=0.01)
        st.text_input("Benchmark", value="Nifty 50 — shown as unavailable unless canonical benchmark data is wired", disabled=True)
        run = st.form_submit_button("Run real backtest", type="primary", width="stretch")

    if not run:
        st.info("Choose the inputs and run the test. Missing historical data will block the run.")
        return
    if not universe:
        st.error("Select at least one stock.")
        return
    if start >= end:
        st.error("The start date must be before the end date.")
        return

    with st.spinner("Reading canonical bhavcopy history and running the event-driven engine…"):
        data, excluded = _load_bhavcopy(universe, start, end)
        if not data:
            st.error(
                "The backtest is blocked because no selected stock has sufficient canonical history. "
                "Open Data & Broker, load/validate historical data, and retry."
            )
            if excluded:
                st.dataframe(
                    pd.DataFrame([{"Stock": key, "Reason": value} for key, value in excluded.items()]),
                    hide_index=True,
                    width="stretch",
                )
            return

        from backtest.backtester import Backtester

        result = Backtester(
            historical_data=data,
            initial_capital=float(capital),
            slippage=float(slippage_pct) / 100.0,
            transaction_cost=float(cost_pct) / 100.0,
            use_llm=False,
        ).run()
        summary = summarize_backtest(result)

    st.subheader("Result")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Completed trades", summary.closed_trades)
    m2.metric("Total return", f"{summary.total_return_pct:+.2f}%")
    m3.metric("Maximum drawdown", f"{summary.max_drawdown_pct:.2f}%")
    m4.metric("Win rate", f"{summary.win_rate_pct:.1f}%")
    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Final equity", f"₹{summary.final_equity:,.0f}")
    n2.metric(
        "Average P&L / closed trade",
        f"₹{summary.expectancy_inr:,.0f}" if summary.expectancy_inr is not None else "Unknown",
    )
    n3.metric(
        "Profit factor",
        "∞" if summary.profit_factor == float("inf") else (
            f"{summary.profit_factor:.2f}" if summary.profit_factor is not None else "Unknown"
        ),
    )
    n4.metric("Evidence verdict", summary.verdict)
    st.warning(summary.warning)

    curve = pd.DataFrame(result.get("equity_curve") or ())
    if not curve.empty and {"timestamp", "equity"}.issubset(curve.columns):
        curve["timestamp"] = pd.to_datetime(curve["timestamp"], errors="coerce")
        st.line_chart(curve.dropna(subset=["timestamp"]).set_index("timestamp")["equity"])

    if excluded:
        with st.expander(f"Stocks excluded before the run ({len(excluded)})"):
            st.dataframe(
                pd.DataFrame([{"Stock": key, "Reason": value} for key, value in excluded.items()]),
                hide_index=True,
                width="stretch",
            )

    with st.expander("Scientific details", expanded=False):
        st.write(
            "This operational engine includes chronological bars, next-bar fills, slippage, costs and "
            "risk checks. It does not produce Reality Check, FDR, DSR/PSR or confidence-interval "
            "evidence, so the retail verdict remains INCONCLUSIVE rather than being promoted to PASS."
        )
        st.json(
            {
                "engine": "backtest.backtester.Backtester.run",
                "strategy": strategy,
                "symbols_requested": universe,
                "symbols_tested": list(data),
                "date_range": [start.isoformat(), end.isoformat()],
                "slippage_pct_per_side": slippage_pct,
                "transaction_cost_pct_per_side": cost_pct,
                "raw_trade_records": summary.trades,
            }
        )
