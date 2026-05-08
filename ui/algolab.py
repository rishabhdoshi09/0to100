"""AlgoLab (Code Cave) — strategy editor with live backtest runner."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DB_PATH = Path("algolab_strategies.db")

STARTER_STRATEGY = '''"""
DevBloom AlgoLab — Strategy Template
-------------------------------------
Your function must be named `generate_signals`.
It receives a DataFrame (OHLCV, daily) and returns a Series of signals:
  +1  = BUY
  -1  = SELL
   0  = HOLD

Available helpers:
  df["close"], df["open"], df["high"], df["low"], df["volume"]
  df.rolling(n).mean(), df.ewm(span=n).mean()
"""

import pandas as pd

def generate_signals(df: pd.DataFrame) -> pd.Series:
    # --- Example: EMA crossover ---
    fast = df["close"].ewm(span=10).mean()
    slow = df["close"].ewm(span=30).mean()

    signals = pd.Series(0, index=df.index)
    signals[fast > slow] =  1   # BUY  when fast crosses above slow
    signals[fast < slow] = -1   # SELL when fast crosses below slow
    return signals
'''

MONACO_HTML = ""  # noqa — kept as no-op to preserve module imports; Monaco removed (deprecated components.html)


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS strategies (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            name    TEXT UNIQUE,
            code    TEXT,
            created TEXT,
            updated TEXT
        )
    """)
    conn.commit()
    conn.close()


def _list_strategies() -> list[dict]:
    _init_db()
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT id, name, updated FROM strategies ORDER BY updated DESC").fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "updated": r[2]} for r in rows]


def _save_strategy(name: str, code: str):
    _init_db()
    now = datetime.now().isoformat()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO strategies (name, code, created, updated)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET code=excluded.code, updated=excluded.updated
    """, (name, code, now, now))
    conn.commit()
    conn.close()


def _load_strategy(name: str) -> str:
    _init_db()
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT code FROM strategies WHERE name=?", (name,)).fetchone()
    conn.close()
    return row[0] if row else STARTER_STRATEGY


def _run_backtest(code: str, df: pd.DataFrame, capital: float = 1_000_000) -> dict:
    """Execute user strategy in a sandboxed namespace and simulate trades."""
    import builtins
    namespace = {"pd": pd, "__builtins__": builtins.__dict__}
    try:
        exec(compile(code, "<algolab>", "exec"), namespace)  # noqa: S102
    except Exception as e:
        return {"error": f"Compilation error: {e}"}

    if "generate_signals" not in namespace:
        return {"error": "Function `generate_signals` not found in strategy."}

    try:
        signals = namespace["generate_signals"](df.copy())
    except Exception as e:
        return {"error": f"Runtime error in generate_signals: {traceback.format_exc()}"}

    if not isinstance(signals, pd.Series):
        return {"error": "`generate_signals` must return a pandas Series."}

    # Simple vectorised simulation
    signals = signals.reindex(df.index).fillna(0)
    cash, position, equity = capital, 0, capital
    trades, equity_curve = [], []
    prev_sig = 0

    for i, (dt, row) in enumerate(df.iterrows()):
        price = row["close"]
        sig   = signals.iloc[i]

        if sig == 1 and prev_sig != 1 and cash >= price:
            qty = int(cash * 0.95 / price)
            if qty > 0:
                position += qty
                cash -= qty * price
                trades.append({"date": str(dt.date()), "action": "BUY", "price": price, "qty": qty})

        elif sig == -1 and prev_sig != -1 and position > 0:
            cash += position * price
            trades.append({"date": str(dt.date()), "action": "SELL", "price": price, "qty": position})
            position = 0

        equity = cash + position * price
        equity_curve.append({"date": str(dt.date()), "equity": equity})
        prev_sig = sig

    if not equity_curve:
        return {"error": "No equity data generated."}

    eq_arr  = [e["equity"] for e in equity_curve]
    returns = pd.Series(eq_arr).pct_change().dropna()
    sharpe  = (returns.mean() / returns.std() * (252 ** 0.5)) if returns.std() > 0 else 0
    peak    = pd.Series(eq_arr).cummax()
    dd      = (pd.Series(eq_arr) - peak) / peak
    max_dd  = dd.min() * 100

    wins    = sum(1 for j in range(1, len(trades), 2)
                  if j < len(trades) and trades[j]["price"] > trades[j-1]["price"])
    total_t = len(trades) // 2
    win_rate = wins / total_t * 100 if total_t else 0

    return {
        "equity_curve": equity_curve,
        "trades":       trades,
        "total_return": (eq_arr[-1] - capital) / capital * 100,
        "sharpe":       round(sharpe, 2),
        "max_dd":       round(max_dd, 2),
        "win_rate":     round(win_rate, 1),
        "n_trades":     total_t,
    }


def render_algolab(fetcher=None):
    """Render the AlgoLab tab."""
    st.markdown("### 🧬 AlgoLab — Code Cave")
    st.caption("Write, test, and save Python trading strategies. Results are simulated on historical data.")

    # Ensure the strategy code is always seeded — Live Runner reads from session state.
    if "algolab_code" not in st.session_state:
        st.session_state["algolab_code"] = STARTER_STRATEGY

    # ── Strategy manager ──────────────────────────────────────────────────────
    col_l, col_r = st.columns([3, 1])
    saved = _list_strategies()
    saved_names = [s["name"] for s in saved]

    with col_r:
        st.markdown("**Saved Strategies**")
        if saved_names:
            load_name = st.selectbox("Load", [""] + saved_names, key="algolab_load",
                                     label_visibility="collapsed")
            if load_name and st.button("Load ↑", key="algolab_load_btn"):
                st.session_state["algolab_code"] = _load_strategy(load_name)
                st.rerun()
        else:
            st.caption("No saved strategies yet.")

        save_name = st.text_input("Save as…", placeholder="my_ema_cross", key="algolab_save_name")
        if st.button("💾 Save", key="algolab_save_btn"):
            code_to_save = st.session_state.get("algolab_code", STARTER_STRATEGY)
            if save_name.strip():
                _save_strategy(save_name.strip(), code_to_save)
                st.success(f"Saved '{save_name}'")
            else:
                st.error("Enter a strategy name.")

    # ── Code editor (textarea — Monaco removed; deprecated components.html) ─
    with col_l:
        current_code = st.session_state.get("algolab_code", STARTER_STRATEGY)
        st.markdown("**Strategy Editor** (Python)")
        new_code = st.text_area(
            "Strategy code",
            value=current_code,
            height=380,
            key="algolab_textarea",
            label_visibility="collapsed",
        )
        if new_code != current_code:
            st.session_state["algolab_code"] = new_code

    # ── Backtest config ───────────────────────────────────────────────────────
    st.divider()
    st.markdown("**Run Backtest**")
    bt_col1, bt_col2, bt_col3, bt_col4 = st.columns(4)
    bt_sym  = bt_col1.text_input("Symbol", value="RELIANCE", key="algolab_sym")
    bt_days = bt_col2.number_input("Days of history", value=500, min_value=100, step=50, key="algolab_days")
    bt_cap  = bt_col3.number_input("Capital ₹", value=1_000_000, step=50_000, key="algolab_cap")
    bt_col4.empty()

    if st.button("▶ Run", key="algolab_run", width="stretch"):
        code = st.session_state.get("algolab_code", STARTER_STRATEGY)
        with st.spinner(f"Running strategy on {bt_sym} ({bt_days}d)…"):
            try:
                import yfinance as yf
                df = yf.download(bt_sym + ".NS", period=f"{bt_days}d", interval="1d",
                                 progress=False, auto_adjust=True)
                if isinstance(df.columns, __import__("pandas").MultiIndex):
                    df.columns = [c[0].lower() for c in df.columns]
                else:
                    df.columns = [c.lower() for c in df.columns]
                df = df.dropna()
                if len(df) < 20:
                    st.error("Not enough data to backtest.")
                else:
                    result = _run_backtest(code, df, capital=float(bt_cap))
                    st.session_state["algolab_result"] = result
            except Exception as e:
                st.error(f"Data fetch failed: {e}")

    # ── Results ───────────────────────────────────────────────────────────────
    result = st.session_state.get("algolab_result")
    if result:
        if "error" in result:
            st.error(result["error"])
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Return", f"{result['total_return']:.1f}%")
            m2.metric("Sharpe",       f"{result['sharpe']:.2f}")
            m3.metric("Max DD",       f"{result['max_dd']:.1f}%")
            m4.metric("Win Rate",     f"{result['win_rate']:.1f}%  ({result['n_trades']} trades)")

            eq = pd.DataFrame(result["equity_curve"])
            if not eq.empty:
                fig = go.Figure(go.Scatter(
                    x=eq["date"], y=eq["equity"], mode="lines",
                    line=dict(color="#00d4ff", width=2),
                    fill="tozeroy", fillcolor="rgba(0,212,255,0.06)",
                    hovertemplate="%{x}: ₹%{y:,.0f}<extra></extra>",
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(8,12,28,0.6)",
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=220,
                    xaxis=dict(color="#8892a4", showgrid=False),
                    yaxis=dict(color="#8892a4", tickprefix="₹",
                               gridcolor="rgba(255,255,255,0.04)"),
                )
                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

            with st.expander("Trade Log"):
                if result["trades"]:
                    st.dataframe(pd.DataFrame(result["trades"]), width="stretch")

    # ── Live Runner ──────────────────────────────────────────────────────────
    st.divider()
    from ui.live_runner import render_live_runner
    render_live_runner(default_symbol=st.session_state.get("algolab_sym", "RELIANCE"))
