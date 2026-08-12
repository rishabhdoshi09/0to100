"""
AlgoLab Live Runner.

Takes a strategy `code` string (Python source defining `generate_signals(df)`),
evaluates it on the latest market data, applies risk gates, and — on user
confirmation — places either a paper-trading entry or a live Kite order.

Design constraints:
  * Manual run only (a "Evaluate Latest Signal" button).
  * Kite live execution requires explicit user confirmation (no auto-trigger).
  * All risk gates must pass before the Confirm button is enabled.
  * No business logic outside this module — UI orchestration only.
"""
from __future__ import annotations

import logging
import traceback
from datetime import datetime, time as dtime
from typing import Optional

import pandas as pd
import streamlit as st
import yfinance as yf

from paper_trading import init_db, open_position

logger = logging.getLogger("devbloom.live_runner")

# ── Default risk controls ────────────────────────────────────────────────────
DEFAULT_RISK = {
    "max_rs_per_trade":     10_000,    # max ₹ exposure per single trade
    "max_open_positions":   5,         # max concurrent open positions
    "daily_loss_limit_rs":  5_000,     # if today's realised loss ≥ this, block new orders
    "trade_window_open":    dtime(9, 15),
    "trade_window_close":   dtime(15, 20),
    "atr_stop_mult":        2.0,
    "atr_target_mult":      4.0,
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def _evaluate_strategy(code: str, df: pd.DataFrame) -> Optional[int]:
    """Compile + execute `generate_signals(df)`; return the last signal (-1/0/+1)."""
    import builtins
    namespace = {
        "pd": pd,
        # Expose full builtins so user strategies can `import numpy`, use
        # `isinstance`, `dict`, etc. The sandbox is best-effort — users run
        # their OWN code here, so don't pretend it's a security boundary.
        "__builtins__": builtins.__dict__,
    }
    exec(compile(code, "<algolab-live>", "exec"), namespace)  # noqa: S102
    if "generate_signals" not in namespace:
        raise RuntimeError("Strategy must define `generate_signals(df)`.")
    signals = namespace["generate_signals"](df.copy())
    if not isinstance(signals, pd.Series) or signals.empty:
        raise RuntimeError("`generate_signals` must return a non-empty pandas Series.")
    return int(signals.iloc[-1])


def _fetch_recent(symbol: str, days: int = 200) -> pd.DataFrame:
    """yfinance fetch with column normalisation."""
    ticker = symbol if symbol.endswith(".NS") else symbol + ".NS"
    raw = yf.download(ticker, period=f"{days}d", interval="1d",
                      progress=False, auto_adjust=True)
    if raw.empty:
        return raw
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    return raw.dropna()


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def _check_risk_gates(action: str, ltp: float, qty: int,
                      risk: dict) -> list[tuple[str, bool, str]]:
    """
    Return [(gate_name, passed, detail)]. Used to block order if any False.
    """
    gates: list[tuple[str, bool, str]] = []

    # 1. Trading window
    now_t = datetime.now().time()
    in_window = risk["trade_window_open"] <= now_t <= risk["trade_window_close"]
    is_weekday = datetime.now().weekday() < 5
    gates.append((
        "Trading window (9:15-15:20 IST, Mon-Fri)",
        bool(in_window and is_weekday),
        f"Now: {now_t.strftime('%H:%M:%S')} · {'Weekday' if is_weekday else 'Weekend'}",
    ))

    # 2. Trade size cap
    notional = ltp * qty
    gates.append((
        "Max ₹ per trade",
        notional <= risk["max_rs_per_trade"],
        f"₹{notional:,.0f} / ₹{risk['max_rs_per_trade']:,.0f}",
    ))

    # 3. Open positions cap (paper DB used as proxy; user can extend with live broker)
    try:
        from paper_trading import get_open_positions
        open_n = len(get_open_positions())
    except Exception:
        open_n = 0
    gates.append((
        "Max open positions",
        open_n < risk["max_open_positions"],
        f"{open_n} open / {risk['max_open_positions']} max",
    ))

    # 4. Daily loss circuit
    try:
        from paper_trading import get_closed_positions
        closed = get_closed_positions()
        today = datetime.now().date().isoformat()
        if not closed.empty and "exit_date" in closed.columns:
            today_pnl = closed[closed["exit_date"].astype(str).str.startswith(today)]["pnl"].sum()
        else:
            today_pnl = 0.0
    except Exception:
        today_pnl = 0.0
    gates.append((
        "Daily loss circuit-breaker",
        today_pnl > -risk["daily_loss_limit_rs"],
        f"Today realised P&L: ₹{today_pnl:,.0f} (limit: -₹{risk['daily_loss_limit_rs']:,})",
    ))

    # 5. Action sanity
    gates.append((
        "Strategy returned a tradable signal",
        action in ("BUY", "SELL"),
        f"Action = {action}",
    ))

    return gates


def _place_paper_order(symbol: str, action: str, ltp: float, qty: int) -> str:
    init_db()
    open_position(symbol, ltp, qty, action,
                  datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return f"PAPER-{datetime.now().strftime('%H%M%S')}"


def _place_kite_order(symbol: str, action: str, qty: int) -> str:
    """Place a real Kite MARKET order. Raises on failure."""
    from data.kite_client import KiteClient
    kite = KiteClient()
    return kite.place_order(
        symbol=symbol,
        transaction_type=action,
        quantity=qty,
        order_type="MARKET",
        tag="algolab_live",
    )


# ── Main UI ──────────────────────────────────────────────────────────────────
def render_live_runner(default_symbol: str = "RELIANCE"):
    st.markdown("##### 🚀 Live Runner")
    st.caption(
        "Evaluate your strategy on the latest market data and place orders. "
        "Risk gates run before every order; live execution always requires explicit confirmation."
    )

    code = st.session_state.get("algolab_code")
    if not code:
        st.info("Write or load a strategy in the editor above first.")
        return

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    sym = c1.text_input("Symbol (NSE)", value=default_symbol,
                        key="lr_sym").upper().strip()
    capital = c2.number_input("Capital ₹", value=10_000, step=1_000,
                              min_value=1_000, key="lr_cap")
    mode = c3.radio("Mode", ["Paper", "Kite Live"], horizontal=False, key="lr_mode")
    eval_btn = c4.button("🔎 Evaluate", key="lr_eval_btn", width="stretch")

    if eval_btn:
        with st.spinner(f"Fetching latest data for {sym}…"):
            try:
                df = _fetch_recent(sym, days=200)
                if df.empty or len(df) < 30:
                    st.error("Not enough recent data for this symbol.")
                    return
                signal = _evaluate_strategy(code, df)
                ltp    = float(df["close"].iloc[-1])
                atr    = _atr(df) if len(df) >= 14 else ltp * 0.015
                st.session_state["lr_eval"] = {
                    "symbol":  sym,
                    "signal":  signal,
                    "ltp":     ltp,
                    "atr":     atr,
                    "ts":      datetime.now().isoformat(timespec="seconds"),
                    "capital": int(capital),
                    "mode":    mode,
                }
                logger.info("live_runner_evaluated sym=%s signal=%s ltp=%.2f atr=%.2f",
                            sym, signal, ltp, atr)
            except Exception as exc:
                logger.exception("live_runner_eval_failed sym=%s", sym)
                st.error(f"Evaluation failed: {exc}")
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())
                return

    eval_state = st.session_state.get("lr_eval")
    if not eval_state:
        return

    st.divider()
    sig = eval_state["signal"]
    ltp = eval_state["ltp"]
    atr = eval_state["atr"]
    cap = eval_state["capital"]
    mode_now = eval_state["mode"]
    action = {1: "BUY", -1: "SELL"}.get(sig, "HOLD")

    badge = {"BUY": "#00ff88", "SELL": "#ff4466", "HOLD": "#ffb800"}[action]
    st.markdown(
        f"**Latest signal for {eval_state['symbol']}** "
        f"<span style='color:{badge};font-weight:700;font-family:monospace'>{action}</span> "
        f"· LTP ₹{ltp:,.2f} · ATR ₹{atr:.2f} · evaluated {eval_state['ts']}",
        unsafe_allow_html=True,
    )

    if action == "HOLD":
        st.info("No tradable signal — strategy returned HOLD. Nothing to do.")
        return

    # Order math
    qty = max(1, int(cap // ltp))
    sl  = round(ltp - DEFAULT_RISK["atr_stop_mult"]   * atr, 2) if action == "BUY" \
          else round(ltp + DEFAULT_RISK["atr_stop_mult"]   * atr, 2)
    tgt = round(ltp + DEFAULT_RISK["atr_target_mult"] * atr, 2) if action == "BUY" \
          else round(ltp - DEFAULT_RISK["atr_target_mult"] * atr, 2)

    o1, o2, o3, o4 = st.columns(4)
    o1.metric("Quantity", f"{qty}")
    o2.metric("Entry (LTP)", f"₹{ltp:,.2f}")
    o3.metric("Stop-Loss", f"₹{sl:,.2f}")
    o4.metric("Target", f"₹{tgt:,.2f}")

    # Risk gates
    gates = _check_risk_gates(action, ltp, qty, DEFAULT_RISK)
    all_pass = all(p for _, p, _ in gates)
    st.markdown("**Risk gates**")
    for name, passed, detail in gates:
        icon = "✅" if passed else "❌"
        color = "#00ff88" if passed else "#ff4466"
        st.markdown(
            f"<div style='font-size:.85rem;color:{color}'>{icon} <b>{name}</b> "
            f"<span style='color:#8892a4'>· {detail}</span></div>",
            unsafe_allow_html=True,
        )

    # Confirm button
    confirm_disabled = not all_pass
    if confirm_disabled:
        st.warning("One or more risk gates failed — order is blocked.")

    confirm_label = (f"✅ Confirm & Place {mode_now.upper()} {action} Order  "
                     f"({qty} × ₹{ltp:,.2f} = ₹{qty*ltp:,.0f})")
    if st.button(confirm_label, key="lr_confirm",
                 disabled=confirm_disabled, type="primary", width="stretch"):
        try:
            if mode_now == "Paper":
                oid = _place_paper_order(eval_state["symbol"], action, ltp, qty)
                st.success(f"Paper order recorded · id={oid}")
                logger.info("paper_order_placed sym=%s action=%s qty=%d ltp=%.2f id=%s",
                            eval_state["symbol"], action, qty, ltp, oid)
            else:  # Kite Live
                oid = _place_kite_order(eval_state["symbol"], action, qty)
                st.success(f"Live order placed on Kite · order_id={oid}")
                logger.info("live_order_placed sym=%s action=%s qty=%d ltp=%.2f id=%s",
                            eval_state["symbol"], action, qty, ltp, oid)
            # Clear the evaluation so the user must re-evaluate before next order
            st.session_state.pop("lr_eval", None)
        except Exception as exc:
            logger.exception("order_placement_failed mode=%s", mode_now)
            st.error(f"Order failed: {exc}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())
