"""
Homepage — Dashboard: Heatmap, Crude Oil, News, Chart Pattern, Scanner, Watchlist.
"""
from __future__ import annotations

import streamlit as st

from ui.scanner import render_scanner

_SIGNAL_COLOR = {"BUY": "#00d4a0", "SELL": "#ff4b4b", "HOLD": "#8892a4", "WATCH": "#f59e0b"}
_SIGNAL_BG    = {"BUY": "rgba(0,212,160,.08)", "SELL": "rgba(255,75,75,.08)",
                 "HOLD": "rgba(136,146,164,.06)", "WATCH": "rgba(245,158,11,.08)"}


def render_homepage(universe: list[str]) -> None:
    # ── Regime Bar ────────────────────────────────────────────────────────────
    try:
        from ui.regime_bar import render_regime_bar
        render_regime_bar()
    except Exception:
        pass

    # ── Row 1: Heatmap (left) + Crude Oil + News (right) ─────────────────────
    col_heat, col_right = st.columns([3, 2])

    with col_heat:
        st.markdown(
            "<div style='font-size:.7rem;color:#8892a4;text-transform:uppercase;"
            "letter-spacing:.08em;margin-bottom:.4rem'>Sector Heatmap</div>",
            unsafe_allow_html=True,
        )
        try:
            from ui.heatmap import render_heatmap
            render_heatmap()
        except Exception as e:
            st.warning(f"Heatmap unavailable: {e}")

    with col_right:
        # ── Crude Oil ─────────────────────────────────────────────────────────
        _render_crude_oil()
        st.divider()
        # ── News (compact) ────────────────────────────────────────────────────
        _render_news_compact()

    st.divider()

    # ── Row 2: Chart Pattern Check (on-demand) ────────────────────────────────
    _render_pattern_check(universe)

    st.divider()

    # ── Row 3: Market Scanner ─────────────────────────────────────────────────
    render_scanner(universe)

    st.divider()

    # ── Row 4: Watchlist ──────────────────────────────────────────────────────
    st.markdown(
        "<h2 style='color:#00d4ff;font-family:JetBrains Mono,monospace;"
        "font-size:1.3rem;letter-spacing:2px;margin-bottom:4px'>"
        "📋 MY WATCHLIST</h2>"
        "<p style='color:#4a5568;font-size:.72rem;margin-bottom:1rem'>"
        "Add any NSE symbol · signals computed on-demand</p>",
        unsafe_allow_html=True,
    )
    _render_watchlist_controls()
    _render_watchlist_cards()


# ── Crude Oil ─────────────────────────────────────────────────────────────────

def _render_crude_oil():
    st.markdown(
        "<div style='font-size:.7rem;color:#8892a4;text-transform:uppercase;"
        "letter-spacing:.08em;margin-bottom:.4rem'>Crude Oil</div>",
        unsafe_allow_html=True,
    )
    try:
        import yfinance as yf
        crude = yf.Ticker("CL=F")
        info  = crude.fast_info
        price  = float(getattr(info, "last_price", 0) or 0)
        prev   = float(getattr(info, "previous_close", price) or price)
        change = ((price / prev) - 1) * 100 if prev else 0.0
        chg_color = "#00d4a0" if change >= 0 else "#ff4466"
        arrow = "▲" if change >= 0 else "▼"

        brent = yf.Ticker("BZ=F")
        b_info = brent.fast_info
        b_price  = float(getattr(b_info, "last_price", 0) or 0)
        b_prev   = float(getattr(b_info, "previous_close", b_price) or b_price)
        b_change = ((b_price / b_prev) - 1) * 100 if b_prev else 0.0
        b_color = "#00d4a0" if b_change >= 0 else "#ff4466"
        b_arrow = "▲" if b_change >= 0 else "▼"

        st.markdown(
            f"<div style='background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);"
            f"border-radius:10px;padding:.75rem 1rem'>"
            f"<div style='display:flex;justify-content:space-between;margin-bottom:.4rem'>"
            f"<span style='color:#8892a4;font-size:.75rem'>WTI Crude (CL=F)</span>"
            f"<span style='color:{chg_color};font-size:.75rem'>{arrow} {abs(change):.2f}%</span>"
            f"</div>"
            f"<div style='font-family:JetBrains Mono,monospace;font-size:1.2rem;"
            f"font-weight:700;color:#e8eaf0'>${price:,.2f}</div>"
            f"<hr style='border:none;border-top:1px solid rgba(255,255,255,.06);margin:.5rem 0'>"
            f"<div style='display:flex;justify-content:space-between;margin-bottom:.2rem'>"
            f"<span style='color:#8892a4;font-size:.75rem'>Brent (BZ=F)</span>"
            f"<span style='color:{b_color};font-size:.75rem'>{b_arrow} {abs(b_change):.2f}%</span>"
            f"</div>"
            f"<div style='font-family:JetBrains Mono,monospace;font-size:1.1rem;"
            f"font-weight:700;color:#e8eaf0'>${b_price:,.2f}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.warning(f"Crude data unavailable: {e}")


# ── News (compact, top 5 headlines) ──────────────────────────────────────────

def _render_news_compact():
    st.markdown(
        "<div style='font-size:.7rem;color:#8892a4;text-transform:uppercase;"
        "letter-spacing:.08em;margin-bottom:.4rem'>Market News</div>",
        unsafe_allow_html=True,
    )
    try:
        from news.fetcher import NewsFetcher
        articles = NewsFetcher().fetch_all(max_age_hours=24)[:6]
        if not articles:
            st.caption("No recent news.")
            return
        for a in articles:
            sentiment = getattr(a, "sentiment", "neutral") or "neutral"
            dot_color = {"positive": "#00d4a0", "negative": "#ff4466"}.get(sentiment, "#8892a4")
            st.markdown(
                f"<div style='padding:.35rem 0;border-bottom:1px solid rgba(255,255,255,.05)'>"
                f"<span style='color:{dot_color};font-size:.65rem'>●</span> "
                f"<span style='color:#c8cfe0;font-size:.78rem;line-height:1.4'>{a.headline}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
    except Exception:
        # RSS fallback
        try:
            import feedparser
            feed = feedparser.parse("https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms")
            for entry in feed.entries[:6]:
                st.markdown(
                    f"<div style='padding:.35rem 0;border-bottom:1px solid rgba(255,255,255,.05)'>"
                    f"<span style='color:#8892a4;font-size:.65rem'>●</span> "
                    f"<span style='color:#c8cfe0;font-size:.78rem;line-height:1.4'>{entry.title}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        except Exception:
            st.caption("News unavailable.")


# ── Chart Pattern Check (on-demand) ──────────────────────────────────────────

def _render_pattern_check(universe: list[str]):
    st.markdown(
        "<h3 style='color:#e8eaf0;font-size:1rem;margin-bottom:.5rem'>"
        "🔍 Technical Setup & Chart Pattern Check</h3>",
        unsafe_allow_html=True,
    )
    pc1, pc2, pc3 = st.columns([3, 1, 1])
    with pc1:
        pat_sym = st.selectbox(
            "Symbol", options=universe, key="pattern_check_sym",
            label_visibility="collapsed",
        )
    with pc2:
        pat_tf = st.selectbox("Timeframe", ["1D", "1W", "1h"], key="pattern_check_tf",
                              label_visibility="collapsed")
    with pc3:
        run_pat = st.button("🔍 Check Pattern", key="pattern_check_run",
                            use_container_width=True, type="primary")

    if run_pat and pat_sym:
        with st.spinner(f"Analysing {pat_sym} chart patterns…"):
            _run_pattern_check(pat_sym, pat_tf)


@st.cache_data(ttl=300, show_spinner=False)
def _pattern_data(symbol: str, period: str, interval: str):
    import yfinance as yf
    import pandas as pd
    df = yf.download(f"{symbol}.NS", period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]].dropna()


def _run_pattern_check(symbol: str, tf: str):
    import os, requests
    import pandas as pd

    _TF_MAP = {"1D": ("1y", "1d"), "1W": ("2y", "1wk"), "1h": ("1mo", "60m")}
    period, interval = _TF_MAP.get(tf, ("1y", "1d"))

    df = _pattern_data(symbol, period, interval)
    if df is None or len(df) < 30:
        st.error(f"Insufficient data for {symbol}.")
        return

    # Compute key levels
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    last      = float(close.iloc[-1])
    sma20     = float(close.rolling(20).mean().iloc[-1])
    sma50     = float(close.rolling(50).mean().iloc[-1])
    sma200    = float(close.rolling(200).mean().iloc[-1]) if len(df) >= 200 else None

    # RSI
    delta = close.diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs    = gain / loss.replace(0, float("nan"))
    rsi   = float((100 - 100 / (1 + rs)).iloc[-1])

    # ATR
    tr = pd.concat([high - low,
                    (high - close.shift()).abs(),
                    (low  - close.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])

    # Volume surge
    vol_avg = float(volume.rolling(20).mean().iloc[-1])
    vol_ratio = float(volume.iloc[-1]) / vol_avg if vol_avg else 1.0

    # 52-week high/low
    w52_high = float(high.tail(252).max()) if len(df) >= 252 else float(high.max())
    w52_low  = float(low.tail(252).min())  if len(df) >= 252 else float(low.min())
    near_52h = abs(last - w52_high) / w52_high < 0.03

    # Simple pattern detection
    patterns = []
    # Golden cross
    if sma50 > sma200 and sma50 is not None and sma200 is not None:
        prev_sma50  = float(close.rolling(50).mean().iloc[-5])
        prev_sma200 = float(close.rolling(200).mean().iloc[-5]) if len(df) >= 205 else sma200
        if prev_sma50 <= prev_sma200:
            patterns.append(("🟢 Golden Cross", "SMA50 just crossed above SMA200 — strong bullish signal"))
    # Death cross
    if sma50 < sma200 and sma50 is not None and sma200 is not None:
        prev_sma50  = float(close.rolling(50).mean().iloc[-5])
        prev_sma200 = float(close.rolling(200).mean().iloc[-5]) if len(df) >= 205 else sma200
        if prev_sma50 >= prev_sma200:
            patterns.append(("🔴 Death Cross", "SMA50 just crossed below SMA200 — bearish signal"))
    # Breakout above 52W high
    if near_52h:
        patterns.append(("🚀 Near 52-Week High", f"Price within 3% of 52W high ₹{w52_high:,.2f}"))
    # Volume surge breakout
    if vol_ratio >= 2.5 and last > sma20:
        patterns.append(("⚡ Volume Breakout", f"Volume {vol_ratio:.1f}× avg with price above SMA20"))
    # Oversold bounce
    if rsi < 35 and last > close.iloc[-3]:
        patterns.append(("🔄 Oversold Bounce", f"RSI {rsi:.1f} — potential reversal from oversold"))
    # Overbought
    if rsi > 70:
        patterns.append(("⚠️ Overbought", f"RSI {rsi:.1f} — caution, may pull back"))
    # Above all MAs (trending)
    if last > sma20 > sma50:
        patterns.append(("📈 Uptrend Structure", "Price > SMA20 > SMA50 — healthy uptrend"))
    # Below all MAs (downtrend)
    if last < sma20 < sma50:
        patterns.append(("📉 Downtrend Structure", "Price < SMA20 < SMA50 — bearish structure"))

    # Metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Price", f"₹{last:,.2f}")
    m2.metric("RSI (14)", f"{rsi:.1f}")
    m3.metric("ATR", f"₹{atr:,.2f}")
    m4.metric("Vol Ratio", f"{vol_ratio:.1f}×")
    m5.metric("SMA20 / SMA50", f"₹{sma20:,.0f} / ₹{sma50:,.0f}")

    # Patterns found
    if patterns:
        st.markdown(
            "<div style='margin:.6rem 0 .3rem;font-size:.75rem;color:#8892a4;"
            "text-transform:uppercase;letter-spacing:.06em'>Patterns Detected</div>",
            unsafe_allow_html=True,
        )
        for name, desc in patterns:
            st.markdown(
                f"<div style='background:rgba(255,255,255,.03);border-left:3px solid #00d4ff;"
                f"border-radius:0 8px 8px 0;padding:.4rem .8rem;margin:.25rem 0'>"
                f"<span style='color:#e8eaf0;font-size:.82rem;font-weight:600'>{name}</span>"
                f"<span style='color:#8892a4;font-size:.76rem;margin-left:.6rem'>{desc}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No strong patterns detected at current levels. Market is ranging.")

    # DeepSeek pattern analysis
    key = os.getenv("DEEPSEEK_API_KEY")
    if key:
        pat_names = ", ".join(p[0] for p in patterns) or "no strong pattern"
        prompt = (
            f"Technical analysis for {symbol} ({tf} chart):\n"
            f"Price: ₹{last:,.2f} | SMA20: ₹{sma20:,.0f} | SMA50: ₹{sma50:,.0f}"
            + (f" | SMA200: ₹{sma200:,.0f}" if sma200 else "")
            + f"\nRSI: {rsi:.1f} | ATR: ₹{atr:.2f} | Volume: {vol_ratio:.1f}× avg\n"
            f"52W High: ₹{w52_high:,.2f} | 52W Low: ₹{w52_low:,.2f}\n"
            f"Detected patterns: {pat_names}\n\n"
            "Give a concise technical setup: key levels to watch, trade bias (bullish/bearish/neutral), "
            "entry zone, stop loss, and target. Use bullet points."
        )
        with st.spinner("DeepSeek analysing setup…"):
            try:
                resp = requests.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={
                        "model": "deepseek-chat",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                        "max_tokens": 500,
                    },
                    timeout=25,
                )
                data = resp.json()
                if "choices" in data:
                    analysis = data["choices"][0]["message"]["content"]
                    st.markdown(
                        "<div style='margin-top:.5rem;font-size:.7rem;color:#00d4ff;"
                        "text-transform:uppercase;letter-spacing:.06em'>DeepSeek Analysis</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"<div style='background:rgba(0,212,255,.05);border:1px solid rgba(0,212,255,.15);"
                        f"border-radius:8px;padding:.7rem 1rem;font-size:.83rem;color:#c8cfe0;"
                        f"line-height:1.7'>{analysis}</div>",
                        unsafe_allow_html=True,
                    )
            except Exception:
                pass


# ── Watchlist controls ────────────────────────────────────────────────────────

def _render_watchlist_controls() -> None:
    if "watchlist" not in st.session_state:
        st.session_state["watchlist"] = []

    col_input, col_add = st.columns([4, 1])
    with col_input:
        new_sym = st.text_input(
            "Add stock", key="wl_add_input",
            placeholder="Type NSE symbol e.g. RELIANCE",
            label_visibility="collapsed",
        ).strip().upper()
    with col_add:
        if st.button("＋ Add", key="wl_add_btn", type="primary"):
            if new_sym and new_sym not in st.session_state["watchlist"]:
                st.session_state["watchlist"].append(new_sym)
                st.rerun()


def _render_watchlist_cards() -> None:
    wl: list[str] = st.session_state.get("watchlist", [])

    if not wl:
        st.markdown(
            "<div style='text-align:center;padding:40px;color:#4a5568;"
            "font-family:JetBrains Mono,monospace;font-size:.8rem'>"
            "🔍 Your watchlist is empty.<br>"
            "<span style='font-size:.7rem'>Add stocks above</span></div>",
            unsafe_allow_html=True,
        )
        return

    prices = _get_prices_batch(wl)
    cols = st.columns(min(len(wl), 4))
    to_remove = []

    for idx, sym in enumerate(wl):
        col = cols[idx % 4]
        with col:
            price_data = prices.get(sym, {})
            price  = price_data.get("price", 0.0)
            change = price_data.get("change_pct", 0.0)
            signal = price_data.get("signal", "HOLD")

            sig_color = _SIGNAL_COLOR.get(signal, "#8892a4")
            sig_bg    = _SIGNAL_BG.get(signal, "rgba(136,146,164,.06)")
            chg_color = "#00d4a0" if change >= 0 else "#ff4b4b"
            chg_arrow = "▲" if change >= 0 else "▼"

            st.markdown(
                f"<div style='background:{sig_bg};border:1px solid {sig_color}33;"
                f"border-left:3px solid {sig_color};border-radius:10px;"
                f"padding:14px 16px;margin-bottom:8px'>"
                f"<div style='display:flex;justify-content:space-between;align-items:start'>"
                f"<span style='color:#e8eaf0;font-size:.85rem;font-weight:700;"
                f"font-family:JetBrains Mono,monospace'>{sym}</span>"
                f"<span style='color:{sig_color};font-size:.62rem;font-weight:700;"
                f"background:rgba(255,255,255,.05);padding:2px 6px;border-radius:4px;"
                f"border:1px solid {sig_color}44'>{signal}</span></div>"
                f"<div style='margin-top:8px'>"
                f"<span style='color:#e8eaf0;font-size:1.1rem;font-weight:700;"
                f"font-family:JetBrains Mono,monospace'>₹{price:,.2f}</span>"
                f"<span style='color:{chg_color};font-size:.72rem;margin-left:8px'>"
                f"{chg_arrow} {abs(change):.1f}%</span></div></div>",
                unsafe_allow_html=True,
            )
            col_run, col_rm = st.columns([3, 1])
            with col_run:
                if st.button("Run Analysis", key=f"wl_run_{sym}", use_container_width=True):
                    st.session_state["terminal_symbol"] = sym
                    st.session_state["sidebar_nav"] = "⚡  Terminal"
                    st.rerun()
            with col_rm:
                if st.button("✕", key=f"wl_rm_{sym}", use_container_width=True):
                    to_remove.append(sym)

    for sym in to_remove:
        st.session_state["watchlist"].remove(sym)
    if to_remove:
        st.rerun()


# ── Price fetcher ─────────────────────────────────────────────────────────────

def _get_prices_batch(symbols: list[str]) -> dict:
    result: dict[str, dict] = {s: {"price": 0.0, "change_pct": 0.0, "signal": "HOLD"} for s in symbols}
    try:
        import yfinance as yf
        for sym in symbols:
            try:
                info   = yf.Ticker(f"{sym}.NS").fast_info
                price  = float(getattr(info, "last_price", 0) or 0)
                prev   = float(getattr(info, "previous_close", price) or price)
                change = ((price / prev) - 1) * 100 if prev else 0.0
                result[sym]["price"]      = round(price, 2)
                result[sym]["change_pct"] = round(change, 2)
            except Exception:
                pass
    except Exception:
        pass
    return result
