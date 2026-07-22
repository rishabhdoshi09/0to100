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


# ══════════════════════════════════════════════════════════════════════════════
# 🎛️ Daily cockpit — Pulse par hi sab kuch: aaj ke setups + book + autopilot.
# Taaki har din alag tabs kholne ki zaroorat na pade (Smart Search · Pulse ·
# Autopilot — bas yahi teen).
# ══════════════════════════════════════════════════════════════════════════════

def _render_scoreboard() -> None:
    """🎯 MACHINE KA HISAAB — page ki pehli cheez. Target vs booked, bas.
    1+1=2: yahi number batata hai ki system aaj productive hai ya nahi."""
    try:
        from execution.autopilot import daily_scoreboard
        sb = daily_scoreboard()
    except Exception:
        return
    if sb["target"] > 0:
        pct = min(100.0, max(0.0, sb["pct"]))
        bar_col = ("#00d4a0" if pct >= 100 else
                   "#22d3ee" if sb["booked_n"] > 0 else "#334155")
        st.markdown(
            f"<div style='background:linear-gradient(135deg,#0d1421,#111827);"
            f"border:1px solid #1e3a5f;border-radius:12px;"
            f"padding:14px 18px;margin-bottom:10px'>"
            f"<div style='display:flex;justify-content:space-between;"
            f"flex-wrap:wrap;align-items:baseline'>"
            f"<span style='font-size:.62rem;color:#8892a4;text-transform:"
            f"uppercase;letter-spacing:.14em'>🎯 Aaj ki machine</span>"
            f"<span style='font-family:JetBrains Mono,monospace;font-size:1.15rem;"
            f"font-weight:800;color:#e6edf3'>₹{sb['booked_net']:,.0f}"
            f"<span style='color:#64748b;font-size:.85rem'> / "
            f"₹{sb['target']:,.0f} NET</span></span></div>"
            f"<div style='background:#1e293b;border-radius:6px;height:8px;"
            f"margin:8px 0 6px'><div style='background:{bar_col};height:8px;"
            f"border-radius:6px;width:{pct:.0f}%'></div></div>"
            f"<div style='font-size:.8rem;color:#c9d1d9'>{sb['line']} "
            f"<span style='color:#64748b'>· {sb['booked_n']} booked · "
            f"{sb['taken']} taken · {sb['slots']} slots</span></div>"
            f"</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div style='{_CARD};border-left:3px solid #64748b;"
            f"font-size:.82rem;color:#c9d1d9'>🎯 {sb['line']}</div>",
            unsafe_allow_html=True)


def _render_brain(market: str = "IN") -> None:
    """🧠 The Brain hero — one authoritative read of the whole board: posture,
    verdict, prioritised directives, vitals. Composes every subsystem; the
    first thing the user sees each day."""
    try:
        from core.brain import assess, posture_meta
        cap = float(st.session_state.get("user_capital") or 100_000.0)
        a = assess(market=market, capital=cap)
    except Exception:
        return
    v = a["vitals"]
    p_col, p_label = posture_meta(a["posture"])
    cur = v.get("cur", "₹")
    # hero: posture + verdict
    st.markdown(
        f"<div style='background:linear-gradient(135deg,{p_col}18,#0d1421);"
        f"border:1px solid {p_col}55;border-left:4px solid {p_col};"
        f"border-radius:12px;padding:14px 18px;margin-bottom:10px'>"
        f"<div style='font-size:.62rem;color:#8892a4;text-transform:uppercase;"
        f"letter-spacing:.14em'>🧠 QuantTerm Brain · aaj ka faisla</div>"
        f"<div style='font-size:1.15rem;font-weight:800;color:{p_col};"
        f"margin:3px 0'>{p_label}</div>"
        f"<div style='font-size:.85rem;color:#c9d1d9'>{a['posture_reason']}</div>"
        f"</div>", unsafe_allow_html=True)
    # vitals strip
    reg = v.get("regime", "UNKNOWN")
    _tr = {"improving": "📈", "decaying": "📉", "stable": "➖"}.get(
        v.get("edge_trend"), "")
    armed = v.get("autopilot_armed")
    day = v.get("day_pnl")
    # autopilot summary lives ONLY here (no separate glance card) — trades +
    # day P&L folded in so the one strip is the complete at-a-glance
    ap_bit = "🤖 ARMED" if armed else "🤖 off"
    if armed:
        ap_bit += f" · {v.get('trades_today', 0)} trades"
        if day is not None:
            ap_bit += f" · {cur}{day:+,.0f}"
    _br = v.get("breadth", "")
    _br_col = {"HEALTHY": "#00d4a0", "MIXED": "#eab308",
               "NARROW": "#ff4b4b"}.get(_br, "#8892a4")
    _br_bit = (f" · breadth <b style='color:{_br_col}'>{_br}</b> "
               f"({v.get('pct_above_50', 0):.0f}%>50DMA)" if _br else "")
    _op = v.get("options_bias", "")
    _op_bit = f" · options {_op}" if _op and _op != "NEUTRAL" else ""
    st.markdown(
        f"<div style='{_CARD};padding:8px 14px;font-size:.76rem;color:#94a3b8;"
        f"font-family:JetBrains Mono,monospace'>"
        f"tape <b style='color:#e2e8f0'>{reg}</b> · edge "
        f"<b style='color:#e2e8f0'>{v.get('expectancy_r', 0):+.2f}R</b> {_tr}"
        f"{_br_bit}{_op_bit} · DD {v.get('drawdown_pct', 0):.0f}% · setups "
        f"<b style='color:#e2e8f0'>{v.get('n_buys', 0)}</b> "
        f"({v.get('n_high_conviction', 0)} 🎯) · book "
        f"<b style='color:#e2e8f0'>{v.get('book_verdict', 'OK')}</b> "
        f"({v.get('open_risk_pct', 0):.1f}% risk) · {ap_bit}</div>",
        unsafe_allow_html=True)
    # directives — prioritised to-do
    _sev = {"critical": "#ff4b4b", "warn": "#f59e0b", "info": "#38bdf8",
            "good": "#00d4a0"}
    rows = "".join(
        f"<div style='font-size:.82rem;color:#c9d1d9;margin:4px 0;"
        f"padding-left:8px;border-left:2px solid {_sev.get(d['severity'],'#8892a4')}'>"
        f"{d['text']}</div>" for d in a["directives"])
    st.markdown(f"<div style='{_CARD}'>{rows}</div>", unsafe_allow_html=True)
    # orient the layers below — each ADDS detail, nothing repeats the summary
    st.caption("↓ Summary upar hai. Neeche: actionable setups → tumhara book → "
               "market context → edge ka saboot → coach — har layer detail "
               "jodti hai, dohraati nahi.")


def _render_autopilot_glance(market: str = "IN") -> None:
    """One compact chip: armed/off · trades today · aaj ka P&L. Read-only —
    arming/limits Autopilot tab par hi (safety)."""
    try:
        if market == "US":
            import execution.us_autopilot as _ap
            cur, flag = "$", "🇺🇸"
        else:
            import execution.autopilot as _ap
            cur, flag = "₹", "🇮🇳"
        s = _ap.get_status()
    except Exception:
        return
    # P&L snapshot is optional (US autopilot may not expose it) — glance still
    # renders the armed state + trade count without it.
    pnl: dict = {}
    try:
        pnl = _ap.pnl_snapshot()
    except Exception:
        pnl = {}
    armed = s.get("armed")
    a_col = "#00d4a0" if armed else "#8892a4"
    a_txt = (f"ARMED · {s.get('mode', 'PAPER')}" if armed else "off")
    day = pnl.get("day_pnl")
    day_col = "#00d4a0" if (day or 0) >= 0 else "#ff4b4b"
    day_bit = (f" · aaj <b style='color:{day_col}'>{cur}{day:+,.0f}</b>"
               if day is not None else "")
    nopen = len(pnl.get("positions", s.get("open_trades", [])))
    st.markdown(
        f"<div style='{_CARD};padding:8px 14px;font-size:.8rem;color:#c9d1d9'>"
        f"<b>{flag} 🤖 Autopilot</b> <span style='color:{a_col};font-weight:700'>"
        f"● {a_txt}</span> · {s.get('trades_today_count', 0)}/"
        f"{s.get('max_trades_per_day', 0)} trades today · {nopen} open{day_bit}"
        f" <span style='color:#64748b'>· manage → Autopilot tab</span></div>",
        unsafe_allow_html=True)


def _render_today_best_setups(limit: int = 5) -> None:
    """Aaj ke actionable BUY setups (shared scanner store se, conviction-ranked)
    + ek-tap PAPER trade. Live hamesha ticket se — yahan se sirf paper."""
    try:
        from scan.auto_scan import get_results
        results, _size, ts, _status = get_results()
    except Exception:
        results, ts = [], 0.0
    buys = [r for r in results if r.get("verdict") in ("STRONG BUY", "BUY")]
    # EV-first (expected return per unit risk); conviction fallback for
    # thin-evidence setups — north star: rank by money, not points
    try:
        from scan.ev_engine import ev_rank_key
        buys.sort(key=ev_rank_key, reverse=True)
    except Exception:
        buys.sort(key=lambda r: float(r.get("conviction_rank")
                                      or r.get("score") or 0), reverse=True)
    buys = buys[:limit]
    _section_title("🎯 Aaj ke Best Setups")
    if not buys:
        st.caption("Abhi koi high-conviction BUY nahi — scanner chalu hai, "
                   "milte hi yahan aayenge (ya market band / regime kamzor).")
        return
    import time as _t
    age = f" · {int((_t.time() - ts) / 60)} min pehle" if ts else ""
    st.caption(f"Top {len(buys)} conviction-ranked · exchange-side stop ke "
               f"saath{age}. Ek-tap paper trade neeche.")
    for r in buys:
        v = r.get("verdict", "WATCH")
        v_col = "#22d3ee" if v == "STRONG BUY" else "#00d4a0"
        hc = ("💎 " if r.get("prime") else
              "🎯 " if r.get("high_conviction") else "")
        conv = r.get("breakout_conviction") or 0
        conv_bit = f" · conviction {conv:.0f}" if conv else ""
        # EV chip — the number that actually ranks it (measured, not vibes);
        # confidence = Bayesian trust in the sample (HIGH ≫ LOW)
        if r.get("bulk_deal"):
            conv_bit += " · 🏦 bulk-deal buying"   # bade paise ka footprint
        if r.get("ev_pct") is not None:
            _cf = r.get("ev_conf", "")
            _cf_col = {"HIGH": "#00d4a0", "MEDIUM": "#eab308",
                       "LOW": "#f59e0b"}.get(_cf, "#8892a4")
            conv_bit += (f" · <b style='color:#00d4a0'>EV {r['ev_pct']:+.1f}%</b>"
                         f" ({r.get('p_win', 0):.0f}% win, n={r.get('ev_n', 0)}"
                         + (f", <span style='color:{_cf_col}'>{_cf}</span>"
                            if _cf else "") + ")")
        why = (r.get("reasons") or [""])[0]
        entry, stop = float(r.get("entry") or 0), float(r.get("stop") or 0)
        target = float(r.get("target") or 0)
        c_card, c_btn = st.columns([4.2, 1])
        with c_card:
            st.markdown(
                f"<div style='{_CARD};margin-bottom:4px'>"
                f"<span style='font-weight:800;font-family:JetBrains Mono,"
                f"monospace;color:#e6edf3'>{hc}{r['symbol']}</span> "
                f"<span style='color:#e2e8f0'>₹{float(r.get('price') or 0):,.1f}</span> "
                f"<span style='color:{v_col};font-weight:700;font-size:.8rem'>{v}</span>"
                f"<span style='font-size:.72rem;color:#8892a4'> · score "
                f"{float(r.get('score') or 0):.0f}{conv_bit}</span>"
                f"<div style='font-size:.76rem;color:#c9d1d9;margin-top:3px'>{why}</div>"
                f"<div style='font-size:.72rem;color:#64748b;margin-top:2px'>"
                f"entry ₹{entry:,.1f} · stop ₹{stop:,.1f} · target ₹{target:,.1f}</div>"
                f"</div>", unsafe_allow_html=True)
        with c_btn:
            if st.button("📝 Paper", key=f"pulse_paper_{r['symbol']}",
                         width="stretch", disabled=not (entry > stop > 0)):
                _paper_trade_from_setup(r)


def _paper_trade_from_setup(r: dict) -> None:
    """Size at the 1% rule and place a PAPER trade (GTT-protected) from a
    cockpit setup. Never live — live stays on the app ticket by design."""
    try:
        from risk.position_sizer import size_position
        from execution.trade_executor import place_trade
        entry, stop = float(r.get("entry") or 0), float(r.get("stop") or 0)
        target = float(r.get("target") or round(entry * 1.03, 1))
        ps = size_position(entry, stop)
        qty = int(ps.get("qty") or 0)
        if qty < 1:
            st.warning(f"{r['symbol']}: 1% rule mein 1 share bhi nahi aata "
                       f"(entry ₹{entry:,.0f}).")
            return
        res = place_trade(symbol=r["symbol"], qty=qty, entry_type="MARKET",
                          entry_price=entry, stop=stop, target=target,
                          product="CNC", paper=True, note="PULSE_COCKPIT")
        if res.get("ok"):
            st.success(f"📝 Paper: {qty} × {r['symbol']} @ ₹{entry:,.1f} "
                       f"(stop ₹{stop:,.1f} · target ₹{target:,.1f})")
        else:
            st.warning(f"Nahi laga: {res.get('message', '')[:80]}")
    except Exception as exc:
        st.warning(f"Paper trade fail: {exc}")


def _render_your_book() -> None:
    """Open positions + risk meter + recent journal — Portfolio tab kholе bina."""
    _section_title("💼 Mera Book")
    try:
        from ui.positions_panel import render_open_positions, render_trade_journal
        n = render_open_positions()
        if not n:
            st.caption("Koi open position nahi. Autopilot ya paper trade lagते "
                       "hi yahan dikhega.")
        render_trade_journal(8)
    except Exception as exc:
        st.caption(f"Book abhi load nahi hua: {exc}")


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

    # ── 🎯 Scoreboard — machine ka purpose: target vs booked, pehli nazar ────
    _render_scoreboard()

    # ── 🧠 Brain — poore board ka ek faisla (autopilot yahin) ────────────────
    _render_brain("IN")

    # ── 🎛️ Cockpit — ACTION layer: aaj ke setups + tumhara book ──────────────
    _render_today_best_setups()
    _render_your_book()

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
        _section_title("💥 Breakouts — poori market (browse)")
        today = p.get("breakouts_today", [])
        if today:
            st.caption("Top actionable + paper-trade upar cockpit mein — yeh "
                       "poore market ki breakout list (context).")
            st.markdown(f"<div style='{_CARD}'>"
                        + "".join(_stock_line(r) for r in today)
                        + "</div>", unsafe_allow_html=True)
        else:
            st.caption("Aaj koi confirmed breakout nahi.")
    with bc2:
        _section_title("⏳ Kal ke liye watch (pre-breakout)")
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
    _section_title("📈 System Report Card — Brain ke edge number ka saboot")
    try:
        from reports.verdict_dashboard import build_equity_curve
        cap = st.session_state.get("user_capital") or 100_000.0
        vd = build_equity_curve(start_capital=float(cap))
        if vd["points"]:
            s = vd["stats"]
            st.caption("Brain ne upar edge ek number mein bola — yeh raha uska "
                       "**poora saboot**: win-rate, profit factor, payoff, "
                       "streak, per-signal aur per-tape breakdown. "
                       f"“Agar tum har tracked signal follow karte toh” — "
                       f"**hypothetical** ({s['closed']} signals, 1% "
                       "rule). Jo trades tumne actually liye woh alag (neeche "
                       "Coach + Autopilot Report Card mein).")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Signals tracked", s["closed"])
            m2.metric("Win rate", f"{s['win_rate']:.0f}%",
                      delta=f"{s['wins']}W / {s['closed'] - s['wins']}L",
                      delta_color="off")
            m3.metric("Return (1% rule)", f"{s['total_return_pct']:+.1f}%")
            m4.metric("Max drawdown", f"{s['max_drawdown_pct']:.1f}%",
                      delta=f"MAR {s.get('mar_ratio', 0):.2f}", delta_color="off",
                      help="MAR = return ÷ worst drawdown. >0.5 achha, "
                           "<0.3 matlab risk zyada, reward kam.")
            # ── Edge metrics — yeh numbers decide 'real paisa laayak?' ────────
            n1, n2, n3, n4 = st.columns(4)
            _er = s.get("expectancy_r", 0.0)
            n1.metric("Expectancy", f"{_er:+.2f}R",
                      delta=f"₹{s.get('expectancy_rupees', 0):+,.0f}/trade",
                      delta_color="normal" if _er >= 0 else "inverse",
                      help="Har trade se average kitna banta/jaata (R-multiple). "
                           "Yahi asli edge hai — positive hona chahiye.")
            _pf = s.get("profit_factor", 0.0)
            n2.metric("Profit factor", f"{_pf:.2f}",
                      help="Gross jeet ÷ gross haar. >1.3 healthy, <1 ghaata.")
            n3.metric("Payoff (win:loss)",
                      f"{s.get('avg_win_r', 0):.2f} : {s.get('avg_loss_r', 0):.2f}",
                      help="Average jeet vs average haar R mein. Low win-rate "
                           "tabhi chalti hai jab jeet badi ho.")
            n4.metric("Max losing streak", f"{s.get('max_loss_streak', 0)}",
                      help="Lagataar sabse lambi haar — isi streak pe log system "
                           "chhod dete hain. Isliye size chhota rakho.")
            # edge-trend chip
            _tr = s.get("edge_trend", "stable")
            _tr_c = {"improving": ("#00d4a0", "📈 Edge improving"),
                     "decaying": ("#ff4b4b", "📉 Edge decaying"),
                     "stable": ("#8892a4", "➖ Edge stable")}[_tr]
            st.markdown(
                f"<span style='background:{_tr_c[0]}18;border:1px solid "
                f"{_tr_c[0]}55;border-radius:6px;padding:2px 10px;font-size:.72rem;"
                f"color:{_tr_c[0]}'>{_tr_c[1]} · recent {s.get('recent_n', 0)} "
                f"signals {s.get('recent_avg_r', 0):+.2f}R vs lifetime "
                f"{_er:+.2f}R</span>", unsafe_allow_html=True)
            try:
                import plotly.graph_objects as go
                ys = [p[1] for p in vd["points"]]
                dates = [p[0] for p in vd["points"]]
                xs = list(range(len(ys)))           # signal SEQUENCE, not clustered dates
                up = ys[-1] >= ys[0]
                base = ys[0]                          # start-capital baseline
                col = "#00d4a0" if up else "#ff4b4b"
                fig = go.Figure(go.Scatter(
                    x=xs, y=ys, mode="lines", line=dict(color=col, width=2),
                    customdata=dates,
                    hovertemplate="Signal %{x}<br>%{customdata}"
                                  "<br>₹%{y:,.0f}<extra></extra>"))
                # dotted line at starting capital — upar = profit, neeche = loss
                fig.add_hline(y=base, line=dict(color="#475569", width=1,
                                                dash="dot"))
                # y-range hugs the ACTUAL equity band so real ups/downs +
                # drawdown dikhein (pehle 0→100k fill sab flat kar deta tha)
                _lo, _hi = min(ys), max(ys)
                _pad = max((_hi - _lo) * 0.15, base * 0.005)
                fig.update_layout(
                    height=240, margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(title="Equity ₹", gridcolor="#1e293b",
                               range=[_lo - _pad, _hi + _pad]),
                    xaxis=dict(title="Signal #", gridcolor="#1e293b"))
                st.plotly_chart(fig, width="stretch",
                                config={"displayModeBar": False})
            except Exception:
                pass
        st.markdown(
            f"<div style='{_CARD};font-size:.85rem;color:#c9d1d9'>{vd['verdict']}</div>",
            unsafe_allow_html=True)

        # ── Edge by signal — kahan se edge aata hai, kahan LEAK hota hai ──────
        # System ab in live results se seekhta hai: proven-negative signals ko
        # scanner khud demote karta hai (≥30 outcomes pe), isliye expectancy
        # upar jaati hai. Yeh woh feedback loop hai jo edge badhata hai.
        try:
            from scan.live_edge import profile_edge
            prof = profile_edge()
            sigs = [(s, a) for s, a in prof["signals"].items() if a["n"] >= 10]
            if sigs:
                sigs.sort(key=lambda x: x[1]["expectancy_r"], reverse=True)
                with st.expander("🔬 Edge by signal — kahan kamai, kahan leak"):
                    st.caption("Har signal ka LIVE record (real tracked "
                               "outcomes). ≥30 outcomes waale proven-negative "
                               "signals ko scanner khud demote kar deta hai — "
                               "isi se expectancy upar jaati hai.")
                    for s, a in sigs:
                        er = a["expectancy_r"]
                        col = ("#00d4a0" if er >= 0.10 else
                               "#f59e0b" if er >= -0.10 else "#ff4b4b")
                        demoted = (" · <span style='color:#ff4b4b'>demoted</span>"
                                   if a["n"] >= 30 and er < -0.10 else
                                   " · <span style='color:#00d4a0'>boosted</span>"
                                   if a["n"] >= 30 and er >= 0.30 else "")
                        st.markdown(
                            f"<div style='font-size:.78rem;color:#c9d1d9;"
                            f"font-family:JetBrains Mono,monospace;padding:2px 0'>"
                            f"<b>{s}</b> · <span style='color:{col}'>"
                            f"{er:+.2f}R</span> · {a['win_rate']:.0f}% WR · "
                            f"n={a['n']}{demoted}</div>", unsafe_allow_html=True)
                    # quality-bucket sanity: does a higher score mean more edge?
                    q = prof["quality"]
                    qbits = [f"{k}: {q[k]['expectancy_r']:+.2f}R (n={q[k]['n']})"
                             for k in ("70+", "50-70", "score<50") if q[k]["n"]]
                    if qbits:
                        st.caption("Quality check — " + " · ".join(qbits)
                                   + ". Higher score = zyada edge hona chahiye.")
                    # edge by market tape — kaunse regime mein signals chalte hain
                    regs = [(rg, a) for rg, a in prof.get("regimes", {}).items()
                            if a["n"] >= 10]
                    if regs:
                        regs.sort(key=lambda x: x[1]["expectancy_r"], reverse=True)
                        rbits = [f"{rg}: {a['expectancy_r']:+.2f}R (n={a['n']})"
                                 for rg, a in regs]
                        st.caption("Edge by tape — " + " · ".join(rbits)
                                   + ". Bad-tape signals se door raho.")
                    # LIVE: what today's tape is demoting in the scorer right now
                    try:
                        from core.regime_engine import compute_regime
                        from scan.live_edge import regime_calibration
                        _reg = str(getattr(compute_regime(), "market_regime", "")
                                   or "")
                        if _reg and _reg.upper() != "UNKNOWN":
                            _rc = regime_calibration(_reg)
                            _dem = [s for s, m in _rc.items() if m < 1.0]
                            if _dem:
                                st.caption(f"🌡️ Aaj ka tape **{_reg}** — scanner "
                                           f"is regime mein ye demote kar raha: "
                                           f"{', '.join(_dem)}.")
                            else:
                                st.caption(f"🌡️ Aaj ka tape **{_reg}** — koi "
                                           f"signal is tape mein proven-negative "
                                           f"nahi, full weight.")
                    except Exception:
                        pass
                    # edge decay: rolling 50/100/250 — mar raha hai ya zinda?
                    try:
                        from scan.live_edge import rolling_expectancy
                        roll = rolling_expectancy()
                        if roll:
                            rbits = [f"last {w}: {r:+.2f}R"
                                     for w, r in sorted(roll.items())]
                            st.caption("⏳ Edge decay check — "
                                       + " · ".join(rbits)
                                       + ". Chhota window doob raha aur bada "
                                         "theek = edge ABHI mar raha hai.")
                    except Exception:
                        pass
                    # decision journal: rejected vs taken + calibration audit
                    try:
                        from core.decision_journal import (decision_report,
                                                           calibration_report)
                        dr = decision_report()
                        if dr.get("verdict"):
                            st.caption(f"🗳️ Decision audit — {dr['verdict']}")
                        cal = calibration_report()
                        _cb = [f"{b['range']}: bola {b['predicted']:.0f}% → "
                               f"nikla {b['actual']:.0f}% (n={b['n']})"
                               for b in cal.get("buckets", [])
                               if b.get("predicted") is not None]
                        if _cb:
                            st.caption("🎯 Calibration — " + " · ".join(_cb)
                                       + f". {cal['verdict']}")
                    except Exception:
                        pass
        except Exception:
            pass
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

    _render_brain("US")

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
        # CONFIRMED breaks only. r["signals"] holds signal_labels (plain-
        # English strings), not the raw SIGNAL_META keys — match the exact
        # labels, same as reports/street_pulse.py's NSE equivalent. A loose
        # "breakout" substring also matches the PRE_BREAKOUT watch label
        # ("Breakout ke kareeb", unconfirmed — conviction never computed
        # for it), which used to leak in here showing a misleading "conv 0".
        _CONFIRMED_BRK_LABELS = ("52-week high breakout",
                                 "Resistance break on volume")
        brk = [r for r in results
               if any(s in _CONFIRMED_BRK_LABELS for s in r.get("signals", []))][:5]
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

    # ── Cockpit tail — shared book (autopilot state Brain hero mein hai) ──────
    _render_your_book()


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
