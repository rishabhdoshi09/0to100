"""Execution & Risk Cockpit — order pad, position monitor, backtest bridge."""
from __future__ import annotations

import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from paper_trading import (
    close_position,
    get_closed_positions,
    get_equity_curve,
    get_open_positions,
    get_trading_summary,
    init_db,
    open_position,
)
from sq_ai.signals.trade_setup import compute_trade_setup


def _log_ai_verdict(symbol: str, action: str, price: float, verdict: str):
    """Write the AI pre-trade verdict alongside the journal entry."""
    try:
        from ui.journal import log_trade_to_journal
        log_trade_to_journal(symbol, action, price, 0, ai_verdict=verdict)
    except Exception:
        pass


def render_order_pad(symbol: str = "", price: float = 0.0, indicators: dict | None = None):
    """Advanced order pad with bracket orders and live risk preview."""
    ind = indicators or {}

    st.markdown("#### Order Pad")
    init_db()

    with st.form("order_pad_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        sym    = col1.text_input("Symbol", value=symbol.upper() or "", placeholder="RELIANCE")
        otype  = col2.selectbox("Order Type", ["MARKET", "LIMIT", "SL", "SL-M"])
        action = col3.selectbox("Action", ["BUY", "SELL"])

        col4, col5, col6 = st.columns(3)
        qty    = col4.number_input("Quantity", min_value=1, value=1, step=1)
        entry  = col5.number_input("Entry Price ₹", min_value=0.0, value=float(price), step=0.05, format="%.2f")
        col6.empty()

        # Auto-fill bracket from ATR trade setup
        if price and ind:
            setup = compute_trade_setup(price, ind)
            if setup:
                default_stop   = setup.get("stop_loss", price * 0.98)
                default_target = setup.get("target", price * 1.04)
            else:
                default_stop, default_target = price * 0.98, price * 1.04
        else:
            default_stop, default_target = 0.0, 0.0

        col7, col8 = st.columns(2)
        stop_loss  = col7.number_input("Stop Loss ₹", min_value=0.0, value=round(default_stop, 2), step=0.05, format="%.2f")
        target     = col8.number_input("Target ₹",   min_value=0.0, value=round(default_target, 2), step=0.05, format="%.2f")

        # Live risk preview
        if entry and stop_loss and target and qty:
            risk    = abs(entry - stop_loss) * qty
            reward  = abs(target - entry)   * qty
            rr      = reward / risk if risk else 0
            rr_color = "#00ff88" if rr >= 2 else ("#ffb800" if rr >= 1 else "#ff4466")
            st.markdown(
                f"<div class='devbloom-card' style='padding:.6rem 1rem'>"
                f"<span style='color:#8892a4;font-size:.72rem'>Risk Preview: </span>"
                f"<span style='color:#ff4466;font-family:JetBrains Mono,monospace'>₹{risk:,.0f} risk</span> · "
                f"<span style='color:#00ff88;font-family:JetBrains Mono,monospace'>₹{reward:,.0f} reward</span> · "
                f"<span style='color:{rr_color};font-weight:600'>R:R {rr:.2f}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        col_ai, col_place = st.columns([1, 2])
        ai_check = col_ai.form_submit_button("🤖 AI Check", use_container_width=True)

        # ABORT blocks Place Order — user must clear to override
        ai_data   = st.session_state.get("_order_ai_verdict")
        is_aborted = ai_data and ai_data[0] == "ABORT"
        submitted  = col_place.form_submit_button(
            "⚡ Place Paper Order",
            use_container_width=True,
            disabled=bool(is_aborted),
        )

        if ai_check and sym:
            with st.spinner("DeepSeek → Claude trade check…"):
                from ai.dual_llm_service import get_service
                svc = get_service()
                verdict, dm, detail = svc.order_confirmation(
                    sym, action, entry or price, stop_loss, target, qty, ind
                )
            from datetime import datetime as _dt
            badge = svc.badge(dm, detail, ts=_dt.now().strftime("%H:%M"))
            st.session_state["_order_ai_verdict"] = (verdict, badge, dm)

        # Show stored AI verdict
        ai_data = st.session_state.get("_order_ai_verdict")
        if ai_data:
            v_verdict, v_badge, _v_dm = ai_data
            v_color = {"GO": "#00ff88", "CAUTION": "#ffb800", "ABORT": "#ff4466"}.get(v_verdict, "#8892a4")
            abort_note = " — Place Order disabled. Clear to override." if v_verdict == "ABORT" else ""
            st.markdown(
                f"<div class='devbloom-card' style='padding:.5rem 1rem'>"
                f"{v_badge} "
                f"<span style='color:{v_color};font-weight:700;font-family:JetBrains Mono,monospace'>"
                f"{v_verdict}</span>"
                f"<span style='color:#8892a4;font-size:.7rem'>{abort_note}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        if submitted:
            if not sym:
                st.error("Enter a symbol.")
            else:
                ai_verdict_str = (ai_data[0] if ai_data else "")
                if action == "BUY":
                    open_position(sym, entry or price, qty, "BUY", datetime.now().strftime("%Y-%m-%d"))
                    _log_ai_verdict(sym, action, entry or price, ai_verdict_str)
                    st.success(f"✅ BUY {qty}×{sym} @ ₹{entry or price:,.2f} placed (paper)")
                    st.session_state.pop("_order_ai_verdict", None)
                else:
                    open_pos = get_open_positions()
                    if open_pos is not None and not open_pos.empty:
                        match = open_pos[open_pos["symbol"] == sym]
                        if not match.empty:
                            close_position(match.iloc[0]["id"], entry or price, datetime.now().strftime("%Y-%m-%d"))
                            _log_ai_verdict(sym, action, entry or price, ai_verdict_str)
                            st.success(f"✅ SELL {sym} @ ₹{entry or price:,.2f} — position closed")
                            st.session_state.pop("_order_ai_verdict", None)
                        else:
                            st.warning(f"No open position in {sym}.")
                    else:
                        st.warning("No open positions.")


def render_position_monitor():
    """Live position table with per-tick P&L and ATR trailing stop suggestions."""
    init_db()
    st.markdown("#### Open Positions")
    open_pos = get_open_positions()

    if open_pos is None or open_pos.empty:
        st.markdown(
            "<div style='color:#8892a4;font-size:.85rem;text-align:center;padding:1rem'>"
            "No open positions. Use the Order Pad to enter a trade.</div>",
            unsafe_allow_html=True,
        )
        return

    # Render each position as a card
    for _, row in open_pos.iterrows():
        entry_p  = row.get("entry_price", 0)
        qty      = row.get("quantity", 0)
        sym      = row.get("symbol", "")
        side     = row.get("side", "BUY")
        date_in  = row.get("date", "")

        # Try to fetch current price
        try:
            import yfinance as yf
            price = yf.Ticker(sym + ".NS").fast_info.get("last_price", entry_p) or entry_p
        except Exception:
            price = entry_p

        pnl   = (price - entry_p) * qty if side == "BUY" else (entry_p - price) * qty
        pnl_p = pnl / (entry_p * qty) * 100 if entry_p and qty else 0
        color = "#00ff88" if pnl >= 0 else "#ff4466"

        st.markdown(
            f"<div class='devbloom-card'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"  <div>"
            f"    <span style='font-size:1rem;font-weight:700;color:#e8eaf0'>{sym}</span>"
            f"    <span style='font-size:.7rem;color:#8892a4;margin-left:.5rem'>{'▲' if side=='BUY' else '▼'} {side} · {qty} shares</span>"
            f"  </div>"
            f"  <div style='text-align:right'>"
            f"    <div style='font-family:JetBrains Mono,monospace;font-size:.95rem;color:#e8eaf0'>₹{price:,.2f}</div>"
            f"    <div style='font-family:JetBrains Mono,monospace;font-size:.85rem;color:{color}'>"
            f"      {'+'if pnl>=0 else ''}₹{pnl:,.0f} ({'+' if pnl_p>=0 else ''}{pnl_p:.2f}%)"
            f"    </div>"
            f"  </div>"
            f"</div>"
            f"<div style='font-size:.68rem;color:#8892a4;margin-top:.3rem'>"
            f"  Entry ₹{entry_p:,.2f} · {date_in}"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_equity_curve():
    """Equity curve chart from paper trading database."""
    init_db()
    eq_df = get_equity_curve()

    if eq_df is None or eq_df.empty:
        st.info("No equity history yet. Place trades to build a curve.")
        return

    fig = go.Figure(go.Scatter(
        x=eq_df["date"] if "date" in eq_df.columns else eq_df.index,
        y=eq_df["equity"] if "equity" in eq_df.columns else eq_df.iloc[:, -1],
        mode="lines",
        line=dict(color="#00d4ff", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.06)",
        hovertemplate="%{x}: ₹%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,12,28,0.6)",
        margin=dict(l=0, r=0, t=8, b=0),
        height=220,
        xaxis=dict(color="#8892a4", showgrid=False),
        yaxis=dict(color="#8892a4", tickprefix="₹",
                   gridcolor="rgba(255,255,255,0.04)"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_backtest_bridge(symbol: str, fetcher=None, ie=None):
    """Quick backtest bridge — runs 2-year backtest and shows tearsheet."""
    st.markdown("#### Backtest Bridge (Ctrl+Shift+B)")
    st.caption("Run a 2-year conviction-based backtest on the selected symbol before going live.")

    col1, col2, col3 = st.columns(3)
    capital   = col1.number_input("Capital ₹", value=1_000_000, step=50_000)
    slippage  = col2.slider("Slippage %", 0.0, 1.0, 0.05, 0.01)
    use_llm   = col3.checkbox("Include LLM signals", value=False)

    if st.button("▶ Run Backtest", use_container_width=True, key="run_backtest_bridge"):
        if not symbol:
            st.error("No symbol selected.")
            return
        with st.spinner(f"Backtesting {symbol} on 2 years of daily data…"):
            try:
                from datetime import timedelta
                from backtest.backtester import Backtester

                end   = datetime.now()
                start = end - timedelta(days=730)

                if fetcher is None or ie is None:
                    st.warning("Backtester requires live data fetcher — not available in demo mode.")
                    return

                df = fetcher.fetch(symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), interval="day")
                if df is None or len(df) < 50:
                    st.error(f"Insufficient data for {symbol}.")
                    return

                bt = Backtester(
                    historical_data={symbol: df},
                    initial_capital=capital,
                    slippage=slippage / 100,
                    use_llm=use_llm,
                )
                results = bt.run()

                from analytics.reporter import PerformanceReporter
                rpt = PerformanceReporter(results)
                stats = rpt.summary_stats()

                # Show tearsheet
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Return", f"{stats.get('total_return_pct', 0):.1f}%")
                m2.metric("Sharpe",       f"{stats.get('sharpe_ratio', 0):.2f}")
                m3.metric("Max DD",       f"{stats.get('max_drawdown_pct', 0):.1f}%")
                m4.metric("Win Rate",     f"{stats.get('win_rate_pct', 0):.1f}%")

                eq = results.get("equity_curve", [])
                if eq:
                    eq_df = pd.DataFrame(eq)
                    fig = go.Figure(go.Scatter(
                        x=eq_df["date"], y=eq_df["equity"],
                        mode="lines", line=dict(color="#00d4ff", width=2),
                        fill="tozeroy", fillcolor="rgba(0,212,255,0.06)",
                    ))
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(8,12,28,0.6)",
                        margin=dict(l=0, r=0, t=0, b=0), height=200,
                        xaxis=dict(color="#8892a4", showgrid=False),
                        yaxis=dict(color="#8892a4", tickprefix="₹",
                                   gridcolor="rgba(255,255,255,0.04)"),
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            except Exception as e:
                st.error(f"Backtest failed: {e}")
