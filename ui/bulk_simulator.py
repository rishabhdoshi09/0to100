"""
Bulk Simulator — runs an AlgoLab strategy across many symbols in parallel.

Two public renderers:
  * render_bulk_backtest()       → universe-wide historical backtest
  * render_bulk_live_signals()   → universe-wide latest-signal scan

Both share:
  - 4 universes: Nifty 50, Nifty 100, Nifty 500, Custom watchlist
  - 24-hour disk-free TTL cache on yfinance OHLCV fetches
  - ThreadPoolExecutor (2–16 workers, default 8)
  - 30 s per-symbol timeout — failures are skipped with an error count
  - Progress bar: "Processing: N/M completed"
  - Sortable table (default) + Heatmap toggle
"""
from __future__ import annotations

import builtins
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutTimeout
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

logger = logging.getLogger("devbloom.bulk_sim")

# ── Universe definitions ─────────────────────────────────────────────────────
NIFTY_50 = [
    "RELIANCE","TCS","HDFCBANK","ICICIBANK","INFY","HINDUNILVR","ITC","SBIN","BHARTIARTL",
    "KOTAKBANK","LT","HCLTECH","BAJFINANCE","ASIANPAINT","AXISBANK","MARUTI","SUNPHARMA",
    "TITAN","ULTRACEMCO","NESTLEIND","WIPRO","ONGC","NTPC","POWERGRID","M&M","TATAMOTORS",
    "TATASTEEL","JSWSTEEL","ADANIPORTS","ADANIENT","COALINDIA","BAJAJFINSV","TECHM",
    "GRASIM","HINDALCO","DRREDDY","CIPLA","DIVISLAB","BRITANNIA","HEROMOTOCO","BAJAJ-AUTO",
    "EICHERMOT","INDUSINDBK","BPCL","SHRIRAMFIN","TATACONSUM","SBILIFE","HDFCLIFE","LTIM",
    "APOLLOHOSP",
]

# Nifty Next-50 to expand to Nifty 100
NIFTY_NEXT_50 = [
    "ABB","ADANIGREEN","ADANIPOWER","AMBUJACEM","BANKBARODA","BERGEPAINT","BOSCHLTD",
    "CHOLAFIN","COLPAL","DABUR","DLF","DMART","GAIL","GODREJCP","HAVELLS","ICICIPRULI",
    "ICICIGI","INDIGO","IOC","IRCTC","JINDALSTEL","LICI","MARICO","NAUKRI","PIDILITIND",
    "PNB","SBICARD","SIEMENS","SRF","TATAPOWER","TVSMOTOR","UBL","VEDL","ZOMATO",
    "ZYDUSLIFE","ATGL","BEL","CANBK","CGPOWER","CUMMINSIND","HAL","INDUSTOWER","JIOFIN",
    "MOTHERSON","PFC","RECLTD","TIINDIA","TRENT","TORNTPHARM","UNITDSPR",
]

NIFTY_100 = NIFTY_50 + NIFTY_NEXT_50


def _get_nifty_500() -> list[str]:
    """First 500 NSE equity symbols from the instrument cache."""
    try:
        from app import get_all_equity_symbols  # type: ignore
        return list(get_all_equity_symbols().keys())[:500]
    except Exception:
        # Fallback if running standalone — Nifty 100 doubled
        return list(dict.fromkeys(NIFTY_100 * 5))[:500]


UNIVERSES = {
    "Nifty 50":         lambda: NIFTY_50,
    "Nifty 100":        lambda: NIFTY_100,
    "Nifty 500":        _get_nifty_500,
    "Custom watchlist": lambda: [],   # user-provided
}


# ── OHLCV fetch (24 h cache) ─────────────────────────────────────────────────
@st.cache_data(ttl=86_400, show_spinner=False)
def _fetch_ohlcv_cached(symbol: str, days: int) -> pd.DataFrame:
    """yfinance OHLCV fetch — cached for 24 h to avoid hammering on rerun."""
    ticker = symbol if symbol.endswith(".NS") else symbol + ".NS"
    try:
        raw = yf.download(ticker, period=f"{days}d", interval="1d",
                          progress=False, auto_adjust=True, timeout=10)
        if raw.empty:
            return raw
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0].lower() for c in raw.columns]
        else:
            raw.columns = [c.lower() for c in raw.columns]
        return raw.dropna()
    except Exception:
        return pd.DataFrame()


# ── Strategy execution ───────────────────────────────────────────────────────
def _exec_strategy(code: str, df: pd.DataFrame) -> pd.Series:
    """Compile + run `generate_signals(df)`; return the full signal Series."""
    namespace = {"pd": pd, "np": np, "__builtins__": builtins.__dict__}
    exec(compile(code, "<bulk-sim>", "exec"), namespace)  # noqa: S102
    if "generate_signals" not in namespace:
        raise RuntimeError("Strategy missing `generate_signals(df)`")
    sig = namespace["generate_signals"](df.copy())
    if not isinstance(sig, pd.Series):
        raise RuntimeError("`generate_signals` must return a pandas Series")
    return sig.reindex(df.index).fillna(0)


def _backtest_one(symbol: str, code: str, days: int, capital: float) -> dict:
    """Vectorised single-symbol backtest. Returns metrics + last_signal."""
    df = _fetch_ohlcv_cached(symbol, days)
    if df.empty or len(df) < 30:
        return {"symbol": symbol, "error": "no_data"}
    try:
        signals = _exec_strategy(code, df)
    except Exception as exc:
        return {"symbol": symbol, "error": f"strategy: {exc}"}

    cash, position, prev_sig = capital, 0, 0
    eq_curve, trades = [], []
    for i, (_, row) in enumerate(df.iterrows()):
        price = float(row["close"])
        sig   = int(signals.iloc[i])
        if sig == 1 and prev_sig != 1 and cash >= price:
            qty = int(cash * 0.95 / price)
            if qty:
                position += qty
                cash -= qty * price
                trades.append({"side": "BUY", "price": price})
        elif sig == -1 and prev_sig != -1 and position > 0:
            cash += position * price
            trades.append({"side": "SELL", "price": price})
            position = 0
        eq_curve.append(cash + position * price)
        prev_sig = sig
    if not eq_curve:
        return {"symbol": symbol, "error": "empty_equity"}

    eq_arr  = np.asarray(eq_curve, dtype=float)
    rets    = np.diff(eq_arr) / eq_arr[:-1]
    sharpe  = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    peak    = np.maximum.accumulate(eq_arr)
    max_dd  = float((eq_arr - peak).min() / peak[(eq_arr - peak).argmin()] * 100) \
              if peak[(eq_arr - peak).argmin()] else 0.0
    wins = sum(1 for j in range(1, len(trades), 2)
               if j < len(trades) and trades[j]["price"] > trades[j-1]["price"])
    n_trades = len(trades) // 2
    win_rate = (wins / n_trades * 100) if n_trades else 0.0
    last_signal = {1: "BUY", -1: "SELL"}.get(int(signals.iloc[-1]), "HOLD")

    return {
        "symbol":       symbol,
        "last_signal":  last_signal,
        "total_return": (eq_arr[-1] - capital) / capital * 100,
        "sharpe":       sharpe,
        "max_dd":       max_dd,
        "win_rate":     win_rate,
        "n_trades":     n_trades,
        "ltp":          float(df["close"].iloc[-1]),
    }


def _signal_one(symbol: str, code: str, days: int = 200) -> dict:
    """Latest-signal evaluation (no equity simulation)."""
    df = _fetch_ohlcv_cached(symbol, days)
    if df.empty or len(df) < 30:
        return {"symbol": symbol, "error": "no_data"}
    try:
        sig_series = _exec_strategy(code, df)
    except Exception as exc:
        return {"symbol": symbol, "error": f"strategy: {exc}"}

    last = int(sig_series.iloc[-1])
    label = {1: "BUY", -1: "SELL"}.get(last, "HOLD")
    ltp   = float(df["close"].iloc[-1])
    chg   = (df["close"].iloc[-1] / df["close"].iloc[-2] - 1) * 100 \
            if len(df) >= 2 else 0.0
    return {
        "symbol":  symbol,
        "signal":  label,
        "ltp":     ltp,
        "chg_pct": float(chg),
    }


# ── Parallel orchestration with progress + timeout ───────────────────────────
def _run_parallel(fn, symbols: list[str], workers: int, timeout: int,
                  status_caption: str) -> list[dict]:
    results: list[dict] = []
    progress = st.progress(0.0, text=f"{status_caption} 0/{len(symbols)}")
    completed = 0
    errors    = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fn, sym): sym for sym in symbols}
        for fut in as_completed(futures):
            sym = futures[fut]
            try:
                r = fut.result(timeout=timeout)
            except FutTimeout:
                r = {"symbol": sym, "error": f"timeout >{timeout}s"}
                errors += 1
            except Exception as exc:
                r = {"symbol": sym, "error": str(exc)[:80]}
                errors += 1
            if "error" in r:
                errors += 1
            results.append(r)
            completed += 1
            progress.progress(completed / len(symbols),
                              text=f"{status_caption} {completed}/{len(symbols)} · "
                                   f"errors {errors}")
    progress.empty()
    logger.info("bulk_run done=%d errors=%d elapsed=%.1fs",
                completed, errors, time.time() - t0)
    return results


# ── UI helpers ───────────────────────────────────────────────────────────────
def _universe_picker(key_prefix: str) -> list[str]:
    """Render the universe selector. Returns the resolved symbol list."""
    c1, c2 = st.columns([2, 5])
    pick = c1.selectbox(
        "Universe", list(UNIVERSES.keys()),
        key=f"{key_prefix}_uni",
    )
    if pick == "Custom watchlist":
        text = c2.text_input(
            "Symbols (comma-separated)",
            value="RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK",
            key=f"{key_prefix}_custom",
        )
        symbols = [s.strip().upper() for s in text.split(",") if s.strip()]
    else:
        symbols = UNIVERSES[pick]()
        c2.caption(f"{len(symbols)} symbols in {pick}")
    return symbols


def _workers_slider(key_prefix: str) -> tuple[int, int]:
    c1, c2 = st.columns(2)
    workers = c1.slider("Parallel workers", 2, 16, 8, 1, key=f"{key_prefix}_w")
    timeout = c2.slider("Timeout per stock (s)", 5, 60, 30, 5, key=f"{key_prefix}_t")
    return workers, timeout


def _heatmap_color(val: float, lo: float, hi: float, invert: bool = False) -> str:
    """Map a number to a green-red gradient. lo=red end, hi=green end."""
    if pd.isna(val):
        return "background-color: rgba(255,255,255,0.02)"
    span = max(hi - lo, 1e-9)
    t = max(0.0, min(1.0, (val - lo) / span))
    if invert:
        t = 1 - t
    r = int(255 * (1 - t))
    g = int(180 * t)
    return f"background-color: rgba({r},{g+50},80,0.45); color: #ffffff"


def _signal_color(label: str) -> str:
    return {
        "BUY":  "background-color: rgba(0,255,136,0.25); color: #00ff88; font-weight:700",
        "SELL": "background-color: rgba(255,68,102,0.25); color: #ff4466; font-weight:700",
        "HOLD": "background-color: rgba(255,184,0,0.18); color: #ffb800; font-weight:700",
    }.get(label, "")


# ── Public renderers ─────────────────────────────────────────────────────────
def render_bulk_backtest():
    """Bulk historical backtest panel (mounted inside AlgoLab)."""
    st.markdown("##### 🔁 Bulk Backtest")
    st.caption("Run the current strategy across many symbols and rank by performance. "
               "OHLCV data cached for 24 h.")

    code = st.session_state.get("algolab_code")
    if not code:
        st.info("Write or load a strategy in the editor above first.")
        return

    symbols = _universe_picker("bbt")
    c1, c2 = st.columns(2)
    days    = c1.number_input("Days of history", value=500, min_value=100,
                              step=50, key="bbt_days")
    capital = c2.number_input("Capital ₹ per stock", value=100_000, step=10_000,
                              min_value=10_000, key="bbt_cap")
    workers, timeout = _workers_slider("bbt")

    if st.button(f"▶ Run on {len(symbols)} symbols", key="bbt_run",
                 type="primary", use_container_width=True, disabled=not symbols):
        results = _run_parallel(
            lambda s: _backtest_one(s, code, int(days), float(capital)),
            symbols, workers, timeout,
            status_caption="Backtesting",
        )
        st.session_state["bbt_results"] = results

    results = st.session_state.get("bbt_results")
    if not results:
        return

    ok    = [r for r in results if "error" not in r]
    bad   = [r for r in results if "error" in r]
    st.success(f"Done — {len(ok)} succeeded, {len(bad)} skipped/errored.")

    if ok:
        df = pd.DataFrame(ok)[
            ["symbol","last_signal","total_return","sharpe","max_dd","win_rate","n_trades","ltp"]
        ].rename(columns={
            "symbol":"Symbol","last_signal":"Signal","total_return":"Return %",
            "sharpe":"Sharpe","max_dd":"Max DD %","win_rate":"Win %","n_trades":"Trades","ltp":"LTP ₹",
        }).round(2).sort_values("Sharpe", ascending=False).reset_index(drop=True)

        view = st.radio("View", ["Table", "Heatmap"], horizontal=True, key="bbt_view")
        if view == "Table":
            st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            r_lo, r_hi = df["Return %"].min(), df["Return %"].max()
            s_lo, s_hi = df["Sharpe"].min(),    df["Sharpe"].max()
            d_lo, d_hi = df["Max DD %"].min(),  df["Max DD %"].max()
            w_lo, w_hi = df["Win %"].min(),     df["Win %"].max()
            styler = df.style.apply(
                lambda col: [_heatmap_color(v, r_lo, r_hi) for v in col]
                            if col.name == "Return %"
                            else [_heatmap_color(v, s_lo, s_hi) for v in col]
                            if col.name == "Sharpe"
                            else [_heatmap_color(v, d_lo, d_hi, invert=True) for v in col]
                            if col.name == "Max DD %"
                            else [_heatmap_color(v, w_lo, w_hi) for v in col]
                            if col.name == "Win %"
                            else [_signal_color(str(v)) for v in col]
                            if col.name == "Signal"
                            else [""] * len(col),
                axis=0,
            ).format({"Return %":"{:+.1f}", "Sharpe":"{:.2f}",
                      "Max DD %":"{:.1f}", "Win %":"{:.0f}", "LTP ₹":"₹{:,.2f}"})
            st.dataframe(styler, hide_index=True, use_container_width=True)

    if bad:
        with st.expander(f"⚠️ {len(bad)} symbols skipped"):
            st.dataframe(pd.DataFrame(bad), hide_index=True, use_container_width=True)


def render_bulk_live_signals():
    """Bulk latest-signal scan (mounted inside AlgoLab Live Runner area)."""
    st.markdown("##### 📊 Bulk Live Scan")
    st.caption("Get the latest signal for many stocks at once — useful as a daily morning glance.")

    code = st.session_state.get("algolab_code")
    if not code:
        st.info("Write or load a strategy in the AlgoLab editor first.")
        return

    symbols = _universe_picker("bls")
    workers, timeout = _workers_slider("bls")

    if st.button(f"📡 Scan {len(symbols)} symbols", key="bls_run",
                 type="primary", use_container_width=True, disabled=not symbols):
        results = _run_parallel(
            lambda s: _signal_one(s, code),
            symbols, workers, timeout,
            status_caption="Scanning",
        )
        st.session_state["bls_results"] = results

    results = st.session_state.get("bls_results")
    if not results:
        return

    ok  = [r for r in results if "error" not in r]
    bad = [r for r in results if "error" in r]

    if ok:
        df = pd.DataFrame(ok).rename(columns={
            "symbol":"Symbol","signal":"Signal","ltp":"LTP ₹","chg_pct":"Day %",
        }).round(2)
        # Sort: BUY first, then SELL, then HOLD; secondary by abs(Day %) desc
        order = {"BUY": 0, "SELL": 1, "HOLD": 2}
        df["_o"] = df["Signal"].map(order).fillna(3)
        df = df.sort_values(["_o","Day %"], ascending=[True, False]).drop(columns="_o").reset_index(drop=True)

        # Top-line counts
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("BUY",  int((df["Signal"]=="BUY").sum()))
        c2.metric("SELL", int((df["Signal"]=="SELL").sum()))
        c3.metric("HOLD", int((df["Signal"]=="HOLD").sum()))
        c4.metric("Errors", len(bad))

        view = st.radio("View", ["Table", "Heatmap"], horizontal=True, key="bls_view")
        if view == "Table":
            st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            d_lo, d_hi = df["Day %"].min(), df["Day %"].max()
            styler = df.style.apply(
                lambda col: [_signal_color(str(v)) for v in col]
                            if col.name == "Signal"
                            else [_heatmap_color(v, d_lo, d_hi) for v in col]
                            if col.name == "Day %"
                            else [""] * len(col),
                axis=0,
            ).format({"LTP ₹":"₹{:,.2f}", "Day %":"{:+.2f}%"})
            st.dataframe(styler, hide_index=True, use_container_width=True)

    if bad:
        with st.expander(f"⚠️ {len(bad)} symbols skipped"):
            st.dataframe(pd.DataFrame(bad), hide_index=True, use_container_width=True)
