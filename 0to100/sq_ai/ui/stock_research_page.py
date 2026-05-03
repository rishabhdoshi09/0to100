"""Stock research hub — conviction score + full fundamentals in one view."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sq_ai.ui._api import get, post


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
_VERDICT_COLOR = {
    "STRONG BUY": "#00c853",
    "BUY": "#69f0ae",
    "HOLD": "#ffd740",
    "SELL": "#ff6d00",
    "STRONG SELL": "#d50000",
}

_VERDICT_EMOJI = {
    "STRONG BUY": "🟢",
    "BUY": "🟢",
    "HOLD": "🟡",
    "SELL": "🔴",
    "STRONG SELL": "🔴",
}


def _score_bar(label: str, score: int, weight: int, tags: list[str]) -> None:
    tag_str = "  ·  ".join(t.replace("_", " ") for t in tags[:5]) if tags else "–"
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:4px'>"
        f"<span style='width:120px;font-weight:600;font-size:13px'>{label}</span>"
        f"<div style='flex:1;background:#2d2d2d;border-radius:6px;height:16px'>"
        f"<div style='width:{score}%;background:{'#00c853' if score>=65 else '#ffd740' if score>=45 else '#ff6d00'};"
        f"height:16px;border-radius:6px;transition:width 0.4s'></div></div>"
        f"<span style='width:40px;text-align:right;font-weight:700;font-size:14px'>{score}</span>"
        f"<span style='color:#888;font-size:11px;width:20px'>{weight}%</span>"
        f"</div>"
        f"<div style='margin-left:132px;color:#aaa;font-size:11px;margin-bottom:10px'>{tag_str}</div>",
        unsafe_allow_html=True,
    )


def _candle_chart(history: list[dict]) -> go.Figure | None:
    if not history:
        return None
    df = pd.DataFrame(history)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="OHLC",
        increasing_line_color="#00c853", decreasing_line_color="#d50000",
    ))
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    df["sma200"] = df["close"].rolling(200).mean()
    fig.add_scatter(x=df["date"], y=df["sma20"], name="SMA 20",
                    line=dict(color="#ffd740", width=1.2))
    fig.add_scatter(x=df["date"], y=df["sma50"], name="SMA 50",
                    line=dict(color="#40c4ff", width=1.2))
    fig.add_scatter(x=df["date"], y=df["sma200"], name="SMA 200",
                    line=dict(color="#ea80fc", width=1.2, dash="dot"))
    # volume bars on secondary y
    colors = ["#00c853" if c >= o else "#d50000"
              for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(
        x=df["date"], y=df["volume"], name="Volume",
        marker_color=colors, opacity=0.3,
        yaxis="y2",
    ))
    fig.update_layout(
        height=480,
        margin=dict(t=20, b=0, l=0, r=0),
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.02),
        yaxis=dict(side="right", showgrid=True, gridcolor="#333"),
        yaxis2=dict(overlaying="y", side="left", showgrid=False,
                    showticklabels=False, range=[0, df["volume"].max() * 6]),
        hovermode="x unified",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Conviction panel
# ─────────────────────────────────────────────────────────────────────────────

def _render_conviction(sym: str) -> None:
    with st.spinner("computing conviction…"):
        cv = get(f"/api/stock/conviction/{sym}")
    if not cv or "error" in cv:
        st.warning(cv.get("error", "conviction unavailable") if cv else "conviction unavailable")
        return

    verdict = cv.get("verdict", "HOLD")
    score = cv.get("conviction", 50)
    color = _VERDICT_COLOR.get(verdict, "#ffd740")
    emoji = _VERDICT_EMOJI.get(verdict, "🟡")

    # ── header row ────────────────────────────────────────────────────────────
    hcol1, hcol2, hcol3, hcol4, hcol5 = st.columns([3, 2, 2, 2, 2])
    name = cv.get("name") or sym
    sector = cv.get("sector") or "–"
    mc = cv.get("market_cap") or 0
    mc_str = f"₹{mc/1e7:,.0f} cr" if mc else "–"
    hcol1.markdown(
        f"<div style='font-size:22px;font-weight:700'>{name}</div>"
        f"<div style='color:#aaa;font-size:13px'>{sector} · {mc_str}</div>",
        unsafe_allow_html=True,
    )
    price = cv.get("price", 0)
    hcol2.metric("Price", f"₹{price:,.2f}")
    hcol3.metric("Stop", f"₹{cv.get('stop', 0):,.2f}")
    hcol4.metric("Target", f"₹{cv.get('target', 0):,.2f}")
    hcol5.metric("R:R", f"{cv.get('risk_reward', 0):.1f}:1")

    # ── conviction badge ───────────────────────────────────────────────────────
    st.markdown(
        f"<div style='background:{color}22;border:2px solid {color};"
        f"border-radius:12px;padding:16px 24px;margin:16px 0;display:flex;"
        f"align-items:center;gap:24px'>"
        f"<div style='font-size:42px;font-weight:900;color:{color}'>{score}</div>"
        f"<div><div style='font-size:24px;font-weight:800;color:{color}'>"
        f"{emoji} {verdict}</div>"
        f"<div style='color:#aaa;font-size:12px'>Conviction score out of 100</div></div>"
        f"<div style='margin-left:auto;text-align:right;color:#aaa;font-size:12px'>"
        f"ADX {cv.get('adx', 0):.0f} · "
        f"Vol {cv.get('vol_ratio', 1):.1f}x avg · "
        f"From 52W high {cv.get('from_52w_high_pct', 0):+.1f}%</div></div>",
        unsafe_allow_html=True,
    )

    # ── pillar breakdown ───────────────────────────────────────────────────────
    bd = cv.get("breakdown") or {}
    c_left, c_right = st.columns(2)
    with c_left:
        for key, label in [("technical", "Technical"), ("fundamental", "Fundamental"),
                           ("smart_money", "Smart Money")]:
            p = bd.get(key, {})
            _score_bar(label, p.get("score", 50), p.get("weight", 0),
                       p.get("signals", []))
    with c_right:
        for key, label in [("momentum", "Momentum"), ("valuation", "Valuation")]:
            p = bd.get(key, {})
            _score_bar(label, p.get("score", 50), p.get("weight", 0),
                       p.get("signals", []))

    # ── Claude bullets ─────────────────────────────────────────────────────────
    bullets = cv.get("claude_bullets") or []
    if bullets:
        st.markdown("**Why now — AI analysis:**")
        for b in bullets:
            st.markdown(f"› {b}")

    # ── quick add to watchlist ─────────────────────────────────────────────────
    if st.button(f"⭐ Add {sym} to watchlist", key="wl_add_top"):
        st.json(post("/api/watchlist", json={"symbol": sym}))


# ─────────────────────────────────────────────────────────────────────────────
# Main render
# ─────────────────────────────────────────────────────────────────────────────

def render() -> None:
    st.markdown(
        "<style>.stTabs [data-baseweb='tab']{font-size:13px}</style>",
        unsafe_allow_html=True,
    )

    col_sym, col_btn = st.columns([4, 1])
    sym = col_sym.text_input(
        "Search any NSE stock", value="RELIANCE.NS",
        placeholder="e.g. RELIANCE.NS · TCS.NS · HDFCBANK.NS",
        label_visibility="collapsed",
    ).upper().strip()
    if col_btn.button("Analyse", type="primary"):
        st.cache_data.clear()
    if not sym:
        return

    # ── conviction panel (always at top) ──────────────────────────────────────
    _render_conviction(sym)

    st.divider()

    # ── full profile for tab details ──────────────────────────────────────────
    with st.spinner(f"loading full profile for {sym}…"):
        prof = get(f"/api/stock/profile/{sym}")
    if not prof or "error" in prof:
        st.error(prof.get("error", "no data") if prof else "no data")
        return

    tabs = st.tabs([
        "📈 Chart", "💰 Fundamentals", "🏦 Ownership",
        "📰 News", "🏢 Peers", "📊 Earnings", "🎯 Estimates",
        "📅 Corp Actions",
    ])

    # ── Chart ─────────────────────────────────────────────────────────────────
    with tabs[0]:
        fig = _candle_chart(prof.get("history", []))
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        tech = prof.get("technicals", {})
        kl = tech.get("key_levels", {})
        if kl:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Support", f"₹{kl.get('support', 0):.2f}")
            c2.metric("Resistance", f"₹{kl.get('resistance', 0):.2f}")
            c3.metric("Stop (2×ATR)", f"₹{kl.get('stop', 0):.2f}")
            c4.metric("Target (3×ATR)", f"₹{kl.get('target', 0):.2f}")
        with st.expander("Raw indicators"):
            st.json(tech.get("indicators", {}), expanded=False)

    # ── Fundamentals ──────────────────────────────────────────────────────────
    with tabs[1]:
        r = prof.get("ratios", {}) or {}
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("P/E", f"{r.get('pe') or 0:.1f}")
        c2.metric("P/B", f"{r.get('pb') or 0:.1f}")

        roe = r.get("roe") or 0
        roe_disp = f"{roe*100:.1f}%" if isinstance(roe, float) and roe < 1 else f"{roe:.1f}%"
        c3.metric("ROE", roe_disp)
        c4.metric("Debt/Eq", f"{r.get('debt_to_equity') or 0:.2f}")
        c5.metric("EV/EBITDA", f"{r.get('ev_ebitda') or 0:.1f}")
        mc = r.get("market_cap") or 0
        c6.metric("Mkt cap", f"₹{mc/1e7:,.0f} cr" if mc else "–")

        fin = prof.get("financials", {}) or {}
        st.subheader("Annual P&L")
        if fin.get("income"):
            st.dataframe(pd.DataFrame(fin["income"]),
                         use_container_width=True, hide_index=True)
        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("Balance sheet")
            if fin.get("balance"):
                st.dataframe(pd.DataFrame(fin["balance"]),
                             use_container_width=True, hide_index=True)
        with col_r:
            st.subheader("Cash flow")
            if fin.get("cashflow"):
                st.dataframe(pd.DataFrame(fin["cashflow"]),
                             use_container_width=True, hide_index=True)
        st.subheader("Quarterly results")
        if prof.get("quarterly"):
            st.dataframe(pd.DataFrame(prof["quarterly"]),
                         use_container_width=True, hide_index=True)

    # ── Ownership ─────────────────────────────────────────────────────────────
    with tabs[2]:
        sh = prof.get("shareholding", {}) or {}
        cur = sh.get("current", {}) or {}
        if cur:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Promoter", f"{cur.get('promoter') or 0:.1f}%")
            c2.metric("FII", f"{cur.get('fii') or 0:.1f}%")
            c3.metric("DII", f"{cur.get('dii') or 0:.1f}%")
            c4.metric("Public", f"{cur.get('public') or 0:.1f}%")

            # Stacked area chart for ownership history
            hist = sh.get("history") or []
            if hist:
                df_sh = pd.DataFrame(hist).set_index("quarter")
                fig_sh = go.Figure()
                colors_sh = {"promoter": "#00c853", "fii": "#40c4ff",
                             "dii": "#ffd740", "public": "#aaa"}
                for col in ["promoter", "fii", "dii", "public"]:
                    if col in df_sh.columns:
                        fig_sh.add_trace(go.Scatter(
                            x=df_sh.index, y=df_sh[col], name=col.upper(),
                            stackgroup="one", mode="none",
                            fillcolor=colors_sh.get(col, "#888"),
                            hovertemplate=f"{col}: %{{y:.1f}}%",
                        ))
                fig_sh.update_layout(
                    height=280, margin=dict(t=10, b=0, l=0, r=0),
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(ticksuffix="%"),
                    legend=dict(orientation="h"),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_sh, use_container_width=True)
        else:
            st.caption("ownership data unavailable")

    # ── News ──────────────────────────────────────────────────────────────────
    with tabs[3]:
        news = prof.get("news", []) or []
        if news:
            for n in news:
                sentiment_icon = "📰"
                st.markdown(
                    f"{sentiment_icon} **[{n.get('title', '')}]({n.get('url', '#')})**  \n"
                    f"<span style='color:#888;font-size:12px'>"
                    f"{n.get('source', '')} · {n.get('publishedAt', '')}</span>",
                    unsafe_allow_html=True,
                )
                st.divider()
        else:
            st.caption("no headlines — add NEWSAPI_KEY in .env for live news")

    # ── Peers ─────────────────────────────────────────────────────────────────
    with tabs[4]:
        peers = prof.get("peers") or []
        if peers:
            df_peers = pd.DataFrame(peers)
            # highlight the current symbol
            def _highlight(row):
                return ["background-color: #1e3a1e" if row["symbol"] == sym
                        else "" for _ in row]
            st.dataframe(df_peers.style.apply(_highlight, axis=1),
                         use_container_width=True, hide_index=True)
        else:
            st.caption("no peers configured for this sector")

    # ── Earnings ──────────────────────────────────────────────────────────────
    with tabs[5]:
        calls = get(f"/api/stock/earnings/{sym}") or []
        if calls:
            for c in calls:
                with st.expander(f"{c.get('quarter')} – {c.get('call_date') or 'no date'}"):
                    hl = c.get("highlights", {})
                    for h_ in (hl.get("highlights") or []):
                        st.markdown(f"- {h_}")
                    st.json(c.get("guidance") or {})
        with st.form("analyse_call"):
            st.caption("Paste a transcript PDF URL → Claude extracts highlights")
            qq = st.text_input("Quarter (e.g. Q3-2025)")
            url = st.text_input("Transcript PDF URL")
            if st.form_submit_button("Analyse with Claude"):
                if qq and url:
                    st.json(post("/api/stock/earnings/analyse",
                                 json={"symbol": sym, "quarter": qq,
                                       "transcript_url": url}))

    # ── Estimates ─────────────────────────────────────────────────────────────
    with tabs[6]:
        est = prof.get("estimates", {}) or {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Target high", f"₹{est.get('target_high') or 0:.2f}")
        c2.metric("Target mean", f"₹{est.get('target_mean') or 0:.2f}")
        c3.metric("Target low", f"₹{est.get('target_low') or 0:.2f}")
        c4.metric("EPS (FY)", f"{est.get('eps_current_year') or '–'}")
        rd = est.get("rating_distribution") or {}
        if rd:
            st.bar_chart(pd.Series(rd))

    # ── Corp Actions ──────────────────────────────────────────────────────────
    with tabs[7]:
        a = prof.get("actions", {}) or {}
        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("Dividends")
            if a.get("dividends"):
                st.dataframe(pd.DataFrame(a["dividends"]),
                             use_container_width=True, hide_index=True)
        with col_r:
            st.subheader("Splits & Buybacks")
            if a.get("splits"):
                st.dataframe(pd.DataFrame(a["splits"]),
                             use_container_width=True, hide_index=True)
