"""
Bulk Technical Setup & Chart Pattern Scanner.
Runs the same pattern-detection logic as the single-symbol checker
across the entire universe in parallel, then presents a ranked table.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import streamlit as st


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PatternResult:
    symbol: str
    price: float
    rsi: float
    atr: float
    vol_ratio: float
    sma20: float
    sma50: float
    sma200: Optional[float]
    w52_high: float
    w52_low: float
    patterns: list[tuple[str, str]]          # (name, desc)
    signal: str                               # BULL / BEAR / NEUTRAL
    score: int                                # 0-10 bullish score
    error: str = ""


_BULL_PATTERNS = {
    "🟢 Golden Cross", "🚀 Near 52-Week High",
    "⚡ Volume Breakout", "🔄 Oversold Bounce", "📈 Uptrend Structure",
}
_BEAR_PATTERNS = {
    "🔴 Death Cross", "⚠️ Overbought", "📉 Downtrend Structure",
}


# ── Data fetch ────────────────────────────────────────────────────────────────

def _fetch(symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
    # 1. Kite Connect
    try:
        from data.kite_client import KiteClient
        from data.instruments import InstrumentManager
        from data.historical import HistoricalDataFetcher
        kite = KiteClient()
        if kite.is_connected():
            fetcher = HistoricalDataFetcher(kite, InstrumentManager())
            to_dt   = date.today().strftime("%Y-%m-%d")
            from_dt = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
            df = fetcher.fetch(symbol, from_dt, to_dt, interval="day")
            if df is not None and len(df) >= 30:
                df.columns = [c.lower() for c in df.columns]
                return df
    except Exception:
        pass
    # 2. yfinance
    try:
        import yfinance as yf
        df = yf.Ticker(f"{symbol}.NS").history(period=f"{days}d", interval="1d")
        if df is not None and len(df) >= 30:
            df.columns = [c.lower() for c in df.columns]
            return df
    except Exception:
        pass
    return None


# ── Pattern computation (pure, no Streamlit) ──────────────────────────────────

def _analyse(symbol: str) -> PatternResult:
    df = _fetch(symbol)
    if df is None or len(df) < 30:
        return PatternResult(symbol=symbol, price=0, rsi=0, atr=0, vol_ratio=0,
                             sma20=0, sma50=0, sma200=None, w52_high=0, w52_low=0,
                             patterns=[], signal="NEUTRAL", score=0,
                             error="no_data")

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    last    = float(close.iloc[-1])
    sma20   = float(close.rolling(20).mean().iloc[-1])
    sma50   = float(close.rolling(50).mean().iloc[-1])
    sma200  = float(close.rolling(200).mean().iloc[-1]) if len(df) >= 200 else None

    delta   = close.diff()
    gain    = delta.where(delta > 0, 0).rolling(14).mean()
    loss    = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs      = gain / loss.replace(0, float("nan"))
    rsi     = float((100 - 100 / (1 + rs)).iloc[-1])

    tr      = pd.concat([high - low,
                         (high - close.shift()).abs(),
                         (low  - close.shift()).abs()], axis=1).max(axis=1)
    atr     = float(tr.rolling(14).mean().iloc[-1])

    vol_avg   = float(volume.rolling(20).mean().iloc[-1])
    vol_ratio = float(volume.iloc[-1]) / vol_avg if vol_avg else 1.0

    w52_high = float(high.tail(252).max()) if len(df) >= 252 else float(high.max())
    w52_low  = float(low.tail(252).min())  if len(df) >= 252 else float(low.min())
    near_52h = abs(last - w52_high) / w52_high < 0.03

    patterns: list[tuple[str, str]] = []

    if sma50 is not None and sma200 is not None:
        if sma50 > sma200:
            prev50  = float(close.rolling(50).mean().iloc[-5])
            prev200 = float(close.rolling(200).mean().iloc[-5]) if len(df) >= 205 else sma200
            if prev50 <= prev200:
                patterns.append(("🟢 Golden Cross", "SMA50 just crossed above SMA200"))
        elif sma50 < sma200:
            prev50  = float(close.rolling(50).mean().iloc[-5])
            prev200 = float(close.rolling(200).mean().iloc[-5]) if len(df) >= 205 else sma200
            if prev50 >= prev200:
                patterns.append(("🔴 Death Cross", "SMA50 just crossed below SMA200"))

    if near_52h:
        patterns.append(("🚀 Near 52-Week High", f"Within 3% of ₹{w52_high:,.2f}"))
    if vol_ratio >= 2.5 and last > sma20:
        patterns.append(("⚡ Volume Breakout", f"Vol {vol_ratio:.1f}× avg, price > SMA20"))
    if rsi < 35 and last > float(close.iloc[-3]):
        patterns.append(("🔄 Oversold Bounce", f"RSI {rsi:.1f}"))
    if rsi > 70:
        patterns.append(("⚠️ Overbought", f"RSI {rsi:.1f}"))
    if last > sma20 > sma50:
        patterns.append(("📈 Uptrend Structure", "Price > SMA20 > SMA50"))
    if last < sma20 < sma50:
        patterns.append(("📉 Downtrend Structure", "Price < SMA20 < SMA50"))

    bull = sum(1 for p in patterns if p[0] in _BULL_PATTERNS)
    bear = sum(1 for p in patterns if p[0] in _BEAR_PATTERNS)
    score = max(0, min(10, 5 + bull * 2 - bear * 2))
    if bull > bear:
        signal = "BULL"
    elif bear > bull:
        signal = "BEAR"
    else:
        signal = "NEUTRAL"

    return PatternResult(
        symbol=symbol, price=last, rsi=rsi, atr=atr, vol_ratio=vol_ratio,
        sma20=sma20, sma50=sma50, sma200=sma200,
        w52_high=w52_high, w52_low=w52_low,
        patterns=patterns, signal=signal, score=score,
    )


# ── DeepSeek one-liner per stock ──────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _deepseek_brief(symbol: str, price: float, rsi: float, atr: float,
                    vol_ratio: float, sma20: float, sma50: float,
                    sma200: Optional[float], w52_high: float,
                    pattern_names: str) -> str:
    key = os.getenv("DEEPSEEK_API_KEY", "")
    if not key:
        return ""
    import requests
    prompt = (
        f"Technical analysis for {symbol}:\n"
        f"Price: ₹{price:,.2f} | SMA20: ₹{sma20:,.0f} | SMA50: ₹{sma50:,.0f}"
        + (f" | SMA200: ₹{sma200:,.0f}" if sma200 else "")
        + f"\nRSI: {rsi:.1f} | ATR: ₹{atr:.2f} | Vol: {vol_ratio:.1f}× avg"
        + f"\n52W High: ₹{w52_high:,.2f}"
        + f"\nPatterns: {pattern_names or 'none'}\n\n"
        "In 4 bullet points: trade bias, entry zone, stop loss, target."
    )
    try:
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": "deepseek-chat",
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.3, "max_tokens": 300},
            timeout=20,
        )
        data = resp.json()
        return data["choices"][0]["message"]["content"] if "choices" in data else ""
    except Exception:
        return ""


# ── Main render ───────────────────────────────────────────────────────────────

def render_bulk_pattern(universe: list[str]) -> None:
    st.markdown("### Bulk Technical Setup & Pattern Scanner")

    col_tf, col_filter, col_run = st.columns([2, 3, 2])
    with col_tf:
        tf = st.selectbox("Timeframe", ["Daily (1D)", "Weekly (1W)"],
                          key="bulk_pat_tf")
    with col_filter:
        signal_filter = st.multiselect(
            "Show signals",
            ["BULL", "NEUTRAL", "BEAR"],
            default=["BULL", "NEUTRAL"],
            key="bulk_pat_filter",
        )
    with col_run:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        run = st.button("⚡ Run Bulk Scan", key="bulk_pat_run",
                        use_container_width=True, type="primary")

    if not run and "bulk_pat_results" not in st.session_state:
        st.caption(f"Universe: {len(universe)} symbols. Click **Run Bulk Scan** to analyse all.")
        return

    if run:
        results: list[PatternResult] = []
        prog = st.progress(0, text="Scanning universe…")
        total = len(universe)

        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = {pool.submit(_analyse, sym): sym for sym in universe}
            done = 0
            for fut in as_completed(futures):
                done += 1
                prog.progress(done / total, text=f"Scanned {done}/{total} symbols…")
                try:
                    results.append(fut.result())
                except Exception:
                    pass

        prog.empty()
        st.session_state["bulk_pat_results"] = results

    results: list[PatternResult] = st.session_state.get("bulk_pat_results", [])
    if not results:
        st.warning("No results yet.")
        return

    # ── Filter & sort ─────────────────────────────────────────────────────────
    filtered = [r for r in results if r.signal in signal_filter and not r.error]
    filtered.sort(key=lambda r: r.score, reverse=True)

    ok    = len(filtered)
    total = len(results)
    bull  = sum(1 for r in results if r.signal == "BULL" and not r.error)
    bear  = sum(1 for r in results if r.signal == "BEAR" and not r.error)
    neu   = sum(1 for r in results if r.signal == "NEUTRAL" and not r.error)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scanned", total)
    c2.metric("Bullish", bull, delta=f"{bull/total*100:.0f}%" if total else None)
    c3.metric("Bearish", bear, delta=f"-{bear/total*100:.0f}%" if total else None,
              delta_color="inverse")
    c4.metric("Showing", ok)

    st.markdown("---")

    if not filtered:
        st.info("No symbols match the selected signal filter.")
        return

    # ── Table ─────────────────────────────────────────────────────────────────
    _SIG_COLOR = {"BULL": "#00d4a0", "BEAR": "#ff4b4b", "NEUTRAL": "#f59e0b"}

    for r in filtered:
        sig_col = _SIG_COLOR.get(r.signal, "#8892a4")
        pat_str = " · ".join(p[0] for p in r.patterns) if r.patterns else "No pattern"

        with st.expander(
            f"**{r.symbol}**  ₹{r.price:,.2f}  |  "
            f"RSI {r.rsi:.0f}  |  Score {r.score}/10  |  {pat_str}",
            expanded=False,
        ):
            # Metrics row
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Price", f"₹{r.price:,.2f}")
            m2.metric("RSI", f"{r.rsi:.1f}")
            m3.metric("Vol Ratio", f"{r.vol_ratio:.1f}×")
            m4.metric("SMA20", f"₹{r.sma20:,.0f}")
            m5.metric("SMA50", f"₹{r.sma50:,.0f}")
            m6.metric("52W High", f"₹{r.w52_high:,.0f}")

            # Signal badge
            st.markdown(
                f"<div style='display:inline-block;background:{sig_col}22;"
                f"border:1px solid {sig_col}55;border-radius:6px;"
                f"padding:.25rem .75rem;font-size:.75rem;font-weight:700;"
                f"color:{sig_col};margin:.4rem 0'>{r.signal}</div>",
                unsafe_allow_html=True,
            )

            # Pattern list
            if r.patterns:
                for name, desc in r.patterns:
                    st.markdown(
                        f"<div style='background:rgba(255,255,255,.03);"
                        f"border-left:3px solid {sig_col};"
                        f"border-radius:0 8px 8px 0;padding:.35rem .8rem;margin:.2rem 0'>"
                        f"<span style='color:#e8eaf0;font-size:.82rem;font-weight:600'>{name}</span>"
                        f"<span style='color:#8892a4;font-size:.76rem;margin-left:.6rem'>{desc}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No strong patterns detected.")

            # DeepSeek analysis (lazy, on-demand)
            if os.getenv("DEEPSEEK_API_KEY"):
                if st.button(f"DeepSeek analysis", key=f"ds_{r.symbol}"):
                    pat_names = ", ".join(p[0] for p in r.patterns)
                    with st.spinner("Analysing…"):
                        brief = _deepseek_brief(
                            r.symbol, r.price, r.rsi, r.atr, r.vol_ratio,
                            r.sma20, r.sma50, r.sma200, r.w52_high, pat_names,
                        )
                    if brief:
                        st.markdown(
                            f"<div style='background:rgba(0,212,255,.05);"
                            f"border:1px solid rgba(0,212,255,.2);"
                            f"border-radius:8px;padding:.7rem 1rem;"
                            f"font-size:.83rem;color:#c8cfe0;line-height:1.7'>{brief}</div>",
                            unsafe_allow_html=True,
                        )
