"""Common-user Streamlit pages for QuantTerm.

Every page reads or invokes the existing backend. No portfolio, evidence, risk,
execution, scanner or backtest logic is reimplemented here.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import date

import pandas as pd
import streamlit as st

from product import build_product_state, gather_product_inputs


def _money(value: float) -> str:
    return f"₹{value:,.0f}"


def _run_activation() -> None:
    from research.autonomy.controls import request_control, REFRESH_DATA_NOW
    request_control(REFRESH_DATA_NOW, reason="owner requested data refresh from legacy retail page")
    st.success("Market-data refresh queued for the autonomy supervisor.")


def render_home() -> None:
    inputs = gather_product_inputs()
    state = build_product_state(inputs)

    st.title("QuantTerm")
    st.subheader(state.headline)
    st.caption(state.readiness)
    st.write(state.activity)
    if state.attention:
        st.warning(state.attention)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Zerodha", "Connected" if inputs.kite_connected else "Login needed")
    c2.metric("Market data", "Ready" if inputs.data_ready else "Not ready",
              inputs.latest_market_date or None)
    c3.metric("Automatic paper trading", "ON" if inputs.paper_auto_enabled else "OFF")
    c4.metric("Open paper positions", inputs.open_positions)

    if state.primary_key == "connect":
        try:
            from data.kite_client import KiteClient
            st.link_button("Connect Zerodha", KiteClient().login_url(), type="primary")
        except Exception as exc:
            st.error(f"Zerodha login could not be opened: {exc}")
    elif state.primary_key == "update_data":
        if st.button("Update Market Data", type="primary", width="stretch"):
            _run_activation()
    elif state.primary_key == "paper":
        if st.button("Enable Automatic Paper Trading", type="primary", width="stretch"):
            from research.autonomy.controls import (
        request_control, ENABLE_PAPER_AUTO, PAUSE_NEW_PAPER_ENTRIES, RUN_CYCLE_NOW,
    )
            request_control(ENABLE_PAPER_AUTO, reason="owner enabled PAPER_AUTO")
            st.success("PAPER_AUTO enable request queued."); st.rerun()
    elif state.primary_key == "start_worker":
        if st.button("Start Paper-Trading Worker", type="primary", width="stretch"):
            st.warning("Start the dedicated autonomy service with `python main.py autonomy`.")
    elif state.primary_key == "backtest":
        st.info("Open **Backtest** from the sidebar. Market-closed time is research time.")
    else:
        st.success(state.primary_action)

    st.divider()
    st.subheader("Get QuantTerm ready")
    for number, step in enumerate(state.setup_steps, 1):
        a, b = st.columns([1, 5])
        a.markdown(f"### {number}")
        b.markdown(f"**{step.label} — {step.status}**")
        b.caption(step.detail)

    if not inputs.market_open:
        st.divider()
        st.subheader("Useful things to do while the market is closed")
        cols = st.columns(3)
        for idx, label in enumerate(state.useful_actions):
            cols[idx % 3].markdown(f"- {label}")


def _broad_momentum_scan():
    from data.nse_universe import get_nse_universe_with_names
    from scan.bulk_fetcher import prefetch
    from scan.unified_scanner import UnifiedScanner

    universe = get_nse_universe_with_names()
    symbols = sorted(universe)
    progress = st.progress(0, text=f"Loading market history for {len(symbols):,} stocks…")

    def update(done, total):
        progress.progress(min(1.0, done / max(total, 1)), text=f"Loading {done:,} of {total:,} stocks…")

    prefetch(symbols, progress=update)
    results = UnifiedScanner().scan(symbols)
    progress.empty()
    return universe, results


def _signal_frame(rows, names, fno_symbols):
    return pd.DataFrame([
        {
            "Stock": row.symbol,
            "Company": names.get(row.symbol, row.symbol),
            "Status": "Strong candidate" if row.verdict == "BUY" and not row.chase_risk else
                      ("Too extended" if row.chase_risk else "Watch for entry"),
            "Price": row.price,
            "Momentum 5d %": row.momentum_5d,
            "Score": row.score,
            "Volume vs normal": f"{row.volume_ratio:.2f}x",
            "F&O": "Available" if row.symbol in fno_symbols else "Cash only",
            "Why": row.reasons[0] if row.reasons else "",
        }
        for row in rows
    ])


@st.cache_data(ttl=900, show_spinner=False)
def _cached_broad_momentum():
    return _broad_momentum_scan()


def render_momentum() -> None:
    st.title("Momentum Stocks")
    st.caption("QuantTerm scans the broad approved NSE cash universe first. F&O is an overlay, not a small starting shortlist.")
    if st.button("Scan Momentum Stocks", type="primary"):
        _cached_broad_momentum.clear()
    with st.spinner("Scanning the broad NSE universe…"):
        names, results = _cached_broad_momentum()

    try:
        from data.fno_universe import current_fno_universe
        fno = current_fno_universe()
        fno_symbols = set(fno.symbols)
    except Exception:
        fno_symbols = set()

    momentum = [r for r in results if "MOMENTUM" in r.signals]
    near = [r for r in results if "PRE_BREAKOUT" in r.signals and "MOMENTUM" not in r.signals]
    extended = [r for r in results if r.chase_risk]
    fno_momentum = [r for r in momentum if r.symbol in fno_symbols]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cash universe", len(names))
    c2.metric("Momentum candidates", len(momentum))
    c3.metric("F&O momentum", len(fno_momentum))
    c4.metric("Near breakout", len(near))

    tabs = st.tabs(["All Momentum", "F&O Momentum", "Cash Only", "Near Breakout", "Avoid for Now"])
    groups = [
        momentum,
        fno_momentum,
        [r for r in momentum if r.symbol not in fno_symbols],
        near,
        extended,
    ]
    for tab, rows in zip(tabs, groups):
        with tab:
            if rows:
                st.dataframe(_signal_frame(rows, names, fno_symbols), hide_index=True, width="stretch")
            else:
                st.info("No stocks currently match this view. QuantTerm does not force weak candidates.")

    with st.expander("See the complete current F&O evaluation funnel"):
        from ui.fno_momentum_page import render_fno_momentum_page
        render_fno_momentum_page()


def render_paper_trading() -> None:
    from research.autonomy.controls import (
        request_control, ENABLE_PAPER_AUTO, PAUSE_NEW_PAPER_ENTRIES, RUN_CYCLE_NOW,
    )
    from research.auto_research.scheduler import get_brain
    brain = get_brain()
    book = brain.intel_book
    enabled = brain.is_paper_auto_enabled()
    running = brain.state.running

    st.title("Automatic Paper Trading")
    st.info("QuantTerm takes and manages paper trades automatically. You do not approve every trade. You can pause or override it at any time.")

    status = "RUNNING" if enabled and running else ("READY FOR NEXT SESSION" if enabled else "OFF")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", status)
    c2.metric("Paper capital", _money(book.capital))
    c3.metric("Paper equity", _money(book.equity()))
    c4.metric("Open positions", len(book.open))

    a, b = st.columns(2)
    if enabled:
        if a.button("Pause new paper trades", type="secondary", width="stretch"):
            request_control(PAUSE_NEW_PAPER_ENTRIES, reason="owner paused paper entries")
            st.success("Pause request queued. Existing positions remain manageable.")
    else:
        if a.button("Enable automatic paper trading", type="primary", width="stretch"):
            request_control(ENABLE_PAPER_AUTO, reason="owner enabled PAPER_AUTO"); st.rerun()
    if b.button("Request one paper cycle", width="stretch"):
        request_control(RUN_CYCLE_NOW, reason="owner requested immediate paper cycle")
        st.success("Cycle queued for the autonomy supervisor.")

    st.caption(f"Risk per trade: {book.risk_per_trade_pct:.1%} · Maximum positions: {book.max_positions} · Current open risk: {_money(book.open_risk())}")
    if brain.state.last_error:
        st.error(brain.state.last_error)

    st.subheader("Open paper positions")
    if not book.open:
        st.info("No open paper positions. This may simply mean no safe trade qualified.")
    else:
        st.dataframe(pd.DataFrame([p.as_dict() for p in book.open.values()]), hide_index=True, width="stretch")

    st.subheader("Recent paper trades")
    if book.closed:
        st.dataframe(pd.DataFrame([t.as_dict() for t in book.closed[-50:]]), hide_index=True, width="stretch")
    else:
        st.caption("No paper trades have closed yet.")

    with st.expander("Why no trade was taken"):
        if book.refusals:
            st.dataframe(pd.DataFrame(book.refusals[-50:], columns=["Stock", "Reason"]), hide_index=True)
        else:
            st.write("No refusal is recorded. The latest cycle may have found no strategy with sufficient evidence.")


def render_portfolio() -> None:
    st.title("Portfolio")
    try:
        from ui.holdings_page import render_holdings_page
        render_holdings_page()
        return
    except Exception:
        pass
    try:
        from paper_trading import get_open_positions, get_closed_positions
        opened = get_open_positions(); closed = get_closed_positions()
        st.subheader("Open positions")
        st.dataframe(pd.DataFrame(opened), hide_index=True, width="stretch") if opened else st.info("No open positions.")
        with st.expander("Closed positions"):
            st.dataframe(pd.DataFrame(closed), hide_index=True, width="stretch") if closed else st.caption("No closed positions.")
    except Exception as exc:
        st.error(f"Portfolio could not be loaded: {exc}")


def render_market() -> None:
    st.title("Market")
    try:
        from ui.market_narrative import render_market_narrative
        render_market_narrative()
    except Exception:
        try:
            from ui.institutional_terminal import render_institutional_terminal
            render_institutional_terminal()
        except Exception as exc:
            st.error(f"Market view could not be loaded: {exc}")


def render_backtest() -> None:
    st.title("Backtest")
    st.caption("Test a real QuantTerm strategy on past data. Safe defaults are shown first; scientific details remain available inside the workbench.")
    st.info("Start with a strategy, universe, period and paper capital. Negative results are kept visible.")
    try:
        from ui.algolab import render_algolab
        render_algolab()
    except Exception as exc:
        st.error(f"Backtest workbench could not be loaded: {exc}")


def render_learned() -> None:
    st.title("What We’ve Learned")
    try:
        from ui.memory_vault import render_memory_vault
        render_memory_vault()
    except Exception:
        try:
            from ui.auto_research_page import render_auto_research_page
            render_auto_research_page()
        except Exception as exc:
            st.error(f"Learning history could not be loaded: {exc}")


def render_reports() -> None:
    st.title("Reports")
    try:
        from ui.journal import render_journal
        render_journal()
    except Exception as exc:
        st.error(f"Reports could not be loaded: {exc}")


def render_data_zerodha() -> None:
    st.title("Data and Zerodha")
    inputs = gather_product_inputs()
    c1, c2, c3 = st.columns(3)
    c1.metric("Zerodha", "Connected" if inputs.kite_connected else "Login needed")
    c2.metric("Saved market data", "Ready" if inputs.data_ready else "Not ready")
    c3.metric("Stocks in active data", inputs.instrument_count)
    if not inputs.kite_connected:
        try:
            from data.kite_client import KiteClient
            st.link_button("Open Zerodha Login", KiteClient().login_url(), type="primary")
        except Exception as exc:
            st.error(str(exc))
    if st.button("Download / update market data", type="primary", width="stretch"):
        _run_activation()
    st.caption("Routine operation uses Zerodha automatically. Manual bhavcopy import remains an offline fallback.")


def render_alerts() -> None:
    st.title("Alerts")
    try:
        from ui.alerts_page import render_alerts_page
        render_alerts_page()
    except Exception as exc:
        st.error(f"Alerts could not be loaded: {exc}")


def render_settings() -> None:
    st.title("Settings")
    try:
        from config import settings
        st.write("**Paper capital:**", _money(settings.trading_capital))
        st.write("**Risk per trade:**", f"{settings.risk_per_trade_pct:.1%}")
        st.write("**Maximum open positions:**", settings.max_open_positions)
        st.write("**Zerodha:**", "Connected" if settings.kite_access_token else "Login needed")
        st.caption("Sensitive credentials are never displayed.")
    except Exception as exc:
        st.error(str(exc))


def render_help() -> None:
    st.title("Help")
    st.markdown("""
### The desk, in order
1. Connect Zerodha once a day. Market data is owned by the autonomy service.
2. **Today** — SEPA-qualified Best Setups first, then the scanner watchlist.
3. **Setups** — Best Setups (SEPA), Momentum, Conviction, Long-term.
4. **Paper Desk** — enable once. The bot takes, manages and closes simulated trades, then learns from them daily. You do not place broker orders here.
5. **Backtest** — after a paper loss, inspect that stock on past data. A backtest does not change today's BUY list, ranking, or paper autopilot.
6. **Portfolio** — holdings and P&L. **Desk** holds Market, News, Data, Alerts, Settings, and the lab.

### Path to real money
Paper auto → daily paper memory (skip repeat losers, prefer proven names) → Brain 1 evidence once the sample is large enough → live still locked until the owner approves a capital envelope. The bot cannot open that door.

### After a paper loss
The bot records it. Two consecutive losses on the same name pause new paper entries on that name for five days. Also open Backtest with that stock. If the style lost historically after costs, do not keep repeating it in size. If it historically paid, keep risk small — one loss is one outcome.

### Important
A day with no trade is not a failure. It means no setup passed the evidence and safety checks. An empty scan list means the scan has not run yet — that is different from a no-trade day.
""")


def render_advanced() -> None:
    st.title("Advanced — Research Laboratory")
    st.warning("This area is for engineering and scientific inspection. The common-user product does not require it.")
    st.code("streamlit run legacy_app.py")
    st.caption("The complete previous terminal is preserved unchanged in legacy_app.py.")
