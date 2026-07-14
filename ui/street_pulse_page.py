"""
📰 Daily Pulse page — the system's own "Daily Street Pulse" newsletter.

Rendered live from system data (bhavcopy + scanner + Google Finance +
news). Sections mirror a professional daily note: cover takeaways,
market snapshot, movers, buzzing stock, gaining/losing momentum,
breakouts today & tomorrow, global cues, headlines.
"""
from __future__ import annotations

import streamlit as st

_CARD = ("background:#0d1421;border:1px solid #1e293b;border-radius:10px;"
         "padding:14px 18px;margin-bottom:10px")

# EMA overlay colours mirror the newsletter: 10 green · 20 yellow ·
# 50 purple · 200 red
_EMA_SPANS = ((10, "#22c55e"), (20, "#eab308"), (50, "#a78bfa"),
              (200, "#ef4444"))


@st.cache_data(ttl=600, show_spinner=False)
def _daily_ohlcv(symbol: str, sessions: int = 90):
    """DAILY (1D) candles, latest up to date — bhav store is official NSE
    EOD and carries today's live-overlaid bar during market hours. Returns
    a records list (cache-friendly) or None."""
    try:
        from data.bhavcopy_store import get_ohlcv
        df = get_ohlcv(symbol)
        if df is None or len(df) < 20:
            return None
        df = df.tail(sessions)
        return {
            "idx": [str(i)[:10] for i in df.index],
            "o": df["open"].tolist(), "h": df["high"].tolist(),
            "l": df["low"].tolist(), "c": df["close"].tolist(),
        }
    except Exception:
        return None


def _daily_candle_chart(symbol: str, accent: str = "#00d4a0"):
    """Compact daily candlestick + EMA(10/20/50/200) — the newsletter's
    chart, rendered live. None if no data (caller just shows the text)."""
    d = _daily_ohlcv(symbol)
    if not d:
        return None
    try:
        import pandas as pd
        import plotly.graph_objects as go
        close = pd.Series(d["c"])
        fig = go.Figure(go.Candlestick(
            x=d["idx"], open=d["o"], high=d["h"], low=d["l"], close=d["c"],
            increasing_line_color="#00d4a0", decreasing_line_color="#ff4b4b",
            name=symbol, showlegend=False))
        for span, col in _EMA_SPANS:
            if len(close) >= span // 2:
                ema = close.ewm(span=span, adjust=False).mean()
                fig.add_trace(go.Scatter(
                    x=d["idx"], y=ema.tolist(), mode="lines",
                    line=dict(color=col, width=1), name=f"EMA{span}",
                    hoverinfo="skip", showlegend=False))
        fig.update_layout(
            height=240, margin=dict(l=6, r=6, t=6, b=6),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_rangeslider_visible=False,
            xaxis=dict(gridcolor="#161f2e", color="#8892a4",
                       showticklabels=False),
            yaxis=dict(gridcolor="#161f2e", color="#8892a4"))
        return fig
    except Exception:
        return None


def _render_daily_chart(symbol: str) -> None:
    fig = _daily_candle_chart(symbol)
    if fig is not None:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("Daily (1D) candles · EMA 10🟢 20🟡 50🟣 200🔴 · "
                   "market hours mein aaj ka bar live")


@st.cache_data(ttl=600, show_spinner=False)
def _pulse() -> dict:
    from reports.street_pulse import build_pulse
    return build_pulse()


@st.cache_data(ttl=1800, show_spinner=False)
def _global_cues() -> list[dict]:
    """S&P, Nasdaq, Gold, Crude, Nikkei — few tickers, cached 30 min."""
    out = []
    try:
        import yfinance as yf
        for name, tk in (("S&P 500", "^GSPC"), ("Nasdaq", "^IXIC"),
                         ("Gold", "GC=F"), ("Crude", "CL=F"),
                         ("Nikkei", "^N225"), ("Hang Seng", "^HSI")):
            try:
                fi = yf.Ticker(tk).fast_info
                last = float(getattr(fi, "last_price", 0) or 0)
                prev = float(getattr(fi, "previous_close", 0) or 0)
                if last and prev:
                    out.append({"name": name, "price": last,
                                "chg_pct": (last - prev) / prev * 100})
            except Exception:
                continue
    except Exception:
        pass
    return out


def _chg_span(v: float) -> str:
    color = "#00d4a0" if v >= 0 else "#ff4b4b"
    arrow = "▲" if v >= 0 else "▼"
    return f"<span style='color:{color};font-weight:600'>{arrow} {v:+.2f}%</span>"


def _section_title(txt: str) -> None:
    st.markdown(
        f"<div style='font-size:.68rem;color:#8892a4;text-transform:uppercase;"
        f"letter-spacing:.12em;margin:14px 0 8px 0'>{txt}</div>",
        unsafe_allow_html=True)


def _stock_line(r: dict) -> str:
    reason = (r.get("reasons") or [""])[0]
    return (f"<div style='margin-bottom:8px'>"
            f"<span style='color:#e2e8f0;font-weight:700;"
            f"font-family:JetBrains Mono,monospace'>{r['symbol']}</span> "
            f"<span style='color:#e2e8f0'>₹{r['price']:,.1f}</span> "
            f"{_chg_span(r.get('change_pct', 0))}"
            f"<div style='font-size:.76rem;color:#94a3b8;margin-top:2px'>{reason}</div>"
            f"</div>")


def render_street_pulse() -> None:
    # Market-aware: US hours (ya manual US) → US pulse
    try:
        if st.session_state.get("active_market") == "US":
            _render_us_pulse()
            return
    except Exception:
        pass

    with st.spinner("Aaj ka pulse ban raha hai…"):
        p = _pulse()

    # ── Cover ─────────────────────────────────────────────────────────────────
    take_html = "".join(f"<div style='font-size:.9rem;color:#c9d1d9;"
                        f"margin:4px 0'>• {t}</div>" for t in p["takeaways"])
    st.markdown(
        f"<div style='{_CARD};border-left:3px solid #00d4ff'>"
        f"<div style='font-size:1.25rem;font-weight:800;color:#e6edf3'>"
        f"📰 Daily Street Pulse</div>"
        f"<div style='font-size:.8rem;color:#8892a4;margin-bottom:8px'>{p['date']} · "
        f"{p.get('scanned', 0)} stocks scanned</div>"
        f"{take_html}</div>",
        unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        # ── Buzzing stock ─────────────────────────────────────────────────────
        _section_title("🔥 Buzzing Stock of the Day")
        if p.get("buzzing"):
            b = p["buzzing"]
            st.markdown(
                f"<div style='{_CARD};border-left:3px solid #f59e0b'>"
                f"<span style='font-size:1.05rem;font-weight:800;color:#e6edf3;"
                f"font-family:JetBrains Mono,monospace'>{b['symbol']}</span> "
                f"<span style='color:#e2e8f0'>₹{b['price']:,.1f}</span> "
                f"{_chg_span(b.get('change_pct', 0))}"
                f"<div style='font-size:.82rem;color:#c9d1d9;margin-top:6px'>{b['note']}</div>"
                f"</div>", unsafe_allow_html=True)
            _render_daily_chart(b["symbol"])
        else:
            st.caption("Aaj koi bada buzz nahi.")

        # ── Gaining strength ──────────────────────────────────────────────────
        _section_title("💪 Stock Gaining Strength")
        if p.get("strength"):
            s = p["strength"]
            dist = s.get("pivot_distance_pct", 0)
            st.markdown(
                f"<div style='{_CARD};border-left:3px solid #00d4a0'>"
                f"<span style='font-size:1.05rem;font-weight:800;color:#e6edf3;"
                f"font-family:JetBrains Mono,monospace'>{s['symbol']}</span> "
                f"<span style='color:#e2e8f0'>₹{s['price']:,.1f}</span>"
                f"<div style='font-size:.82rem;color:#c9d1d9;margin-top:6px'>"
                f"{(s.get('reasons') or [''])[0]}</div>"
                f"<div style='font-size:.78rem;color:#94a3b8;margin-top:4px'>"
                f"Pivot ₹{s['entry']:,.0f} se {dist:.1f}% neeche · "
                f"Stop ₹{s['stop']:,.0f} · Target ₹{s['target']:,.0f}</div>"
                f"</div>", unsafe_allow_html=True)
            _render_daily_chart(s["symbol"])
        else:
            st.caption("Koi strong accumulation candidate nahi mila.")

        # ── Losing momentum ───────────────────────────────────────────────────
        _section_title("⚠️ Stock Losing Momentum")
        if p.get("weak"):
            w = p["weak"]
            st.markdown(
                f"<div style='{_CARD};border-left:3px solid #ff4b4b'>"
                f"<span style='font-size:1.05rem;font-weight:800;color:#e6edf3;"
                f"font-family:JetBrains Mono,monospace'>{w['symbol']}</span> "
                f"<span style='color:#e2e8f0'>₹{w['price']:,.1f}</span> "
                f"{_chg_span(w.get('chg_5d', 0))}<span style='font-size:.7rem;"
                f"color:#8892a4'> (5 din)</span>"
                f"<div style='font-size:.82rem;color:#c9d1d9;margin-top:6px'>{w['note']}</div>"
                f"</div>", unsafe_allow_html=True)
            _render_daily_chart(w["symbol"])
        else:
            st.caption("Koi bada breakdown nahi aaj.")

    with c2:
        # ── Market snapshot ───────────────────────────────────────────────────
        _section_title("📊 Market Snapshot")
        snap = p.get("snapshot", {})
        idx_html = "".join(
            f"<div style='margin:4px 0'><span style='color:#e2e8f0;font-weight:600'>"
            f"{i['name']}</span> <span style='color:#e2e8f0'>{i['price']:,.0f}</span> "
            f"{_chg_span(i['chg_pct'])}</div>"
            for i in snap.get("indices", []))
        comm = snap.get("commentary", "")
        st.markdown(
            f"<div style='{_CARD}'>{idx_html or '<span style=color:#8892a4>Index data unavailable</span>'}"
            + (f"<div style='font-size:.8rem;color:#94a3b8;margin-top:6px'>{comm}</div>" if comm else "")
            + "</div>", unsafe_allow_html=True)

        # ── Movers ────────────────────────────────────────────────────────────
        _section_title("📈 Top Gainers / 📉 Losers")
        g_html = "".join(
            f"<div style='font-size:.82rem;margin:2px 0'>"
            f"<span style='color:#e2e8f0;font-family:JetBrains Mono,monospace'>{r['symbol']}</span> "
            f"{_chg_span(r['chg_pct'])}</div>" for r in p.get("gainers", [])[:5])
        l_html = "".join(
            f"<div style='font-size:.82rem;margin:2px 0'>"
            f"<span style='color:#e2e8f0;font-family:JetBrains Mono,monospace'>{r['symbol']}</span> "
            f"{_chg_span(r['chg_pct'])}</div>" for r in p.get("losers", [])[:5])
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown(f"<div style='{_CARD}'>{g_html or 'N/A'}</div>",
                        unsafe_allow_html=True)
        with mc2:
            st.markdown(f"<div style='{_CARD}'>{l_html or 'N/A'}</div>",
                        unsafe_allow_html=True)

        # ── Global cues ───────────────────────────────────────────────────────
        _section_title("🌍 Global Cues")
        cues = _global_cues()
        cue_html = "".join(
            f"<div style='font-size:.82rem;margin:2px 0'>"
            f"<span style='color:#e2e8f0'>{c['name']}</span> "
            f"{_chg_span(c['chg_pct'])}</div>" for c in cues)
        st.markdown(f"<div style='{_CARD}'>{cue_html or 'Unavailable right now'}</div>",
                    unsafe_allow_html=True)

    # ── Breakouts today / tomorrow ────────────────────────────────────────────
    bc1, bc2 = st.columns(2)
    with bc1:
        _section_title("💥 Breakout Stocks — Today")
        today = p.get("breakouts_today", [])
        if today:
            st.markdown(f"<div style='{_CARD}'>"
                        + "".join(_stock_line(r) for r in today)
                        + "</div>", unsafe_allow_html=True)
        else:
            st.caption("Aaj koi confirmed breakout nahi.")
    with bc2:
        _section_title("⏳ Breakout Stocks — Tomorrow?")
        tom = p.get("breakouts_tomorrow", [])
        if tom:
            st.markdown(f"<div style='{_CARD}'>"
                        + "".join(_stock_line(r) for r in tom)
                        + "</div>", unsafe_allow_html=True)
        else:
            st.caption("Pivot ke kareeb koi stock nahi abhi.")

    # ── Headlines ─────────────────────────────────────────────────────────────
    heads = p.get("headlines", [])
    if heads:
        _section_title("🗞 Top Updates")
        st.markdown(f"<div style='{_CARD}'>"
                    + "".join(f"<div style='font-size:.82rem;color:#c9d1d9;"
                              f"margin:4px 0'>• {h}</div>" for h in heads)
                    + "</div>", unsafe_allow_html=True)

    # ── 📈 System Report Card — kya yeh system real paise ke laayak hai? ──────
    _section_title("📈 System Report Card — signals follow karte toh?")
    try:
        from reports.verdict_dashboard import build_equity_curve
        cap = st.session_state.get("user_capital") or 100_000.0
        vd = build_equity_curve(start_capital=float(cap))
        if vd["points"]:
            s = vd["stats"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Signals tracked", s["closed"])
            m2.metric("Win rate", f"{s['win_rate']:.0f}%")
            m3.metric("Return (1% rule)", f"{s['total_return_pct']:+.1f}%")
            m4.metric("Max drawdown", f"{s['max_drawdown_pct']:.1f}%")
            try:
                import plotly.graph_objects as go
                xs = [p[0] for p in vd["points"]]
                ys = [p[1] for p in vd["points"]]
                up = ys[-1] >= ys[0]
                fig = go.Figure(go.Scatter(
                    x=xs, y=ys, mode="lines", fill="tozeroy",
                    line=dict(color="#00d4a0" if up else "#ff4b4b", width=2)))
                fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=10),
                                  paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)",
                                  yaxis=dict(gridcolor="#1e293b"),
                                  xaxis=dict(gridcolor="#1e293b"))
                st.plotly_chart(fig, width="stretch",
                                config={"displayModeBar": False})
            except Exception:
                pass
        st.markdown(
            f"<div style='{_CARD};font-size:.85rem;color:#c9d1d9'>{vd['verdict']}</div>",
            unsafe_allow_html=True)
    except Exception as _vd_exc:
        st.caption(f"Report card unavailable: {_vd_exc}")

    # ── 🧠 Coach ki Raay — tumhare trading patterns ka sach ──────────────────
    _section_title("🧠 Coach ki Raay")
    try:
        @st.cache_data(ttl=86_400, show_spinner=False)
        def _coach_note() -> str:
            from reports.trade_coach import build_coach_review
            return build_coach_review()
        _note = _coach_note()
        st.markdown(
            f"<div style='{_CARD};font-size:.85rem;color:#c9d1d9;"
            f"white-space:pre-line'>{_note}</div>", unsafe_allow_html=True)
    except Exception as _c_exc:
        st.caption(f"Coach unavailable: {_c_exc}")

    # ── Actions ───────────────────────────────────────────────────────────────
    a1, a2, _ = st.columns([1.2, 1.2, 2.6])
    with a1:
        if st.button("📨 Telegram pe bhejo", key="pulse_tg", width="stretch"):
            try:
                from alerts.telegram_alerts import AlertEngine
                from reports.street_pulse import pulse_to_telegram
                if AlertEngine().send(pulse_to_telegram(p)):
                    st.success("Bhej diya ✓")
                else:
                    st.warning("Telegram configured nahi hai (.env dekho)")
            except Exception as exc:
                st.warning(f"Nahi bheja ja saka: {exc}")
    with a2:
        if st.button("⟳ Refresh pulse", key="pulse_refresh", width="stretch"):
            _pulse.clear()
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# 🇺🇸 US Pulse — same layout, US setups (built from the US scanner)
# ══════════════════════════════════════════════════════════════════════════════

def _render_us_pulse() -> None:
    from data.us_data import us_market_open
    try:
        from scan.us_scanner import get_us_results, scan_us
        results, ts, status = get_us_results()
        if not results:
            with st.spinner("US setups scan ho rahe hain…"):
                results = scan_us()
    except Exception:
        results = []

    is_open = us_market_open()
    chip = ("🟢 US OPEN" if is_open else "🔴 US CLOSED")
    st.markdown(
        f"<div style='{_CARD};border-left:3px solid #60a5fa'>"
        f"<div style='font-size:1.25rem;font-weight:800;color:#e6edf3'>"
        f"🇺🇸 US Street Pulse</div>"
        f"<div style='font-size:.8rem;color:#8892a4'>{chip} · "
        f"{len(results)} setups · same conviction engine · S&P benchmark</div>"
        f"</div>", unsafe_allow_html=True)

    if not results:
        st.caption("Abhi US setups nahi mile — US market hours (7pm–1:30am IST) "
                   "mein ya **US Scanner** page se scan karo.")
        return

    buys = [r for r in results if r.get("verdict") in ("STRONG BUY", "BUY")]
    pre = [r for r in results if "PreBreakout" in r.get("categories", [])]

    c1, c2 = st.columns(2)
    with c1:
        _section_title("🔥 Buzzing / Strongest US Setup")
        if buys:
            b = buys[0]
            _us_setup_card(b, accent="#f59e0b")
        else:
            st.caption("Koi strong US setup nahi abhi.")

        _section_title("💪 Gaining Strength (pre-breakout)")
        if pre:
            _us_setup_card(min(pre, key=lambda r: r.get("pivot_distance_pct") or 99),
                           accent="#00d4a0")
        else:
            st.caption("Koi US accumulation candidate nahi.")
    with c2:
        _section_title("💥 US Breakouts")
        brk = [r for r in results
               if any("breakout" in s.lower() for s in r.get("signals", []))][:5]
        if brk:
            st.markdown(f"<div style='{_CARD}'>" + "".join(
                f"<div style='font-size:.82rem;color:#c9d1d9;margin:3px 0'>"
                f"<b>{r['symbol']}</b> ${float(r.get('price') or 0):,.2f} · "
                f"{r.get('verdict')} · conv {float(r.get('breakout_conviction') or 0):.0f}"
                f"</div>" for r in brk) + "</div>", unsafe_allow_html=True)
        else:
            st.caption("Aaj koi confirmed US breakout nahi.")

    _section_title("🎯 High Conviction US")
    hc = [r for r in results if r.get("high_conviction")][:5]
    if hc:
        st.markdown(f"<div style='{_CARD}'>" + "".join(
            f"<div style='font-size:.82rem;color:#fbbf24;margin:3px 0'>"
            f"🎯 <b>{r['symbol']}</b> ${float(r.get('price') or 0):,.2f} · "
            f"score {float(r.get('score') or 0):.0f} · "
            f"conviction {float(r.get('breakout_conviction') or 0):.0f} · "
            f"{(r.get('reasons') or [''])[0][:70]}</div>" for r in hc)
            + "</div>", unsafe_allow_html=True)
    else:
        st.caption("Abhi koi US high-conviction setup nahi (conviction rule).")


def _us_setup_card(r: dict, accent: str) -> None:
    st.markdown(
        f"<div style='{_CARD};border-left:3px solid {accent}'>"
        f"<span style='font-size:1.05rem;font-weight:800;color:#e6edf3;"
        f"font-family:JetBrains Mono,monospace'>{r['symbol']}</span> "
        f"<span style='color:#e2e8f0'>${float(r.get('price') or 0):,.2f}</span> "
        f"<span style='font-size:.72rem;color:{accent}'>{r.get('verdict')} · "
        f"conv {float(r.get('breakout_conviction') or 0):.0f}</span>"
        f"<div style='font-size:.82rem;color:#c9d1d9;margin-top:6px'>"
        f"{(r.get('reasons') or [''])[0]}</div>"
        f"<div style='font-size:.76rem;color:#94a3b8;margin-top:4px'>"
        f"Entry ${float(r.get('entry') or 0):,.2f} · "
        f"Stop ${float(r.get('stop') or 0):,.2f} · "
        f"Target ${float(r.get('target') or 0):,.2f}</div></div>",
        unsafe_allow_html=True)
