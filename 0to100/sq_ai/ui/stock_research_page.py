"""Stock-research hub: technicals + fundamentals + earnings + news + peers."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sq_ai.ui._api import get, post


def _candle_chart(history: list[dict]) -> go.Figure | None:
    if not history:
        return None
    df = pd.DataFrame(history)
    fig = go.Figure(data=[go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="OHLC",
    )])
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    fig.add_scatter(x=df["date"], y=df["sma20"], name="SMA20", line=dict(width=1))
    fig.add_scatter(x=df["date"], y=df["sma50"], name="SMA50", line=dict(width=1))
    fig.update_layout(height=420, margin=dict(t=20, b=0, l=0, r=0),
                      xaxis_rangeslider_visible=False, template="plotly_dark")
    return fig


def render() -> None:
    st.title("Stock research")
    sym = st.text_input("Symbol (e.g. RELIANCE.NS)", value="RELIANCE.NS").upper()
    if not sym:
        return
    with st.spinner(f"fetching {sym} …"):
        prof = get(f"/api/stock/profile/{sym}")
    if not prof or "error" in prof:
        st.error(prof.get("error", "no data"))
        return

    h = prof.get("header", {}) or {}
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(h.get("name") or sym,
              f"₹{h.get('price', 0):.2f}",
              f"{h.get('change_pct', 0)*100:+.2f}%")
    c2.metric("Sector", h.get("sector") or "–")
    c3.metric("52W high", f"₹{h.get('high_52w', 0):.2f}")
    c4.metric("52W low", f"₹{h.get('low_52w', 0):.2f}")

    tabs = st.tabs([
        "Technicals", "Fundamentals", "Earnings & Guidance",
        "Estimates", "Shareholding", "Corp. Actions", "News", "Peers",
    ])

    # ── Technicals ──────────────────────────────────────────────────────
    with tabs[0]:
        tech = prof.get("technicals", {})
        if "error" in tech:
            st.warning(tech["error"])
        else:
            st.subheader(tech.get("label", "—"))
            kl = tech.get("key_levels", {})
            cc = st.columns(4)
            cc[0].metric("Support", f"₹{kl.get('support', 0):.2f}")
            cc[1].metric("Resistance", f"₹{kl.get('resistance', 0):.2f}")
            cc[2].metric("Stop", f"₹{kl.get('stop', 0):.2f}")
            cc[3].metric("Target", f"₹{kl.get('target', 0):.2f}")
            fig = _candle_chart(prof.get("history", []))
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            st.json(tech.get("indicators", {}), expanded=False)

    # ── Fundamentals ────────────────────────────────────────────────────
    with tabs[1]:
        r = prof.get("ratios", {}) or {}
        cc = st.columns(4)
        cc[0].metric("P/E", f"{r.get('pe', 0) or 0:.2f}")
        cc[1].metric("P/B", f"{r.get('pb', 0) or 0:.2f}")
        cc[2].metric("ROE", f"{(r.get('roe') or 0)*100:.2f}%"
                     if isinstance(r.get('roe'), float) and r.get('roe') < 1
                     else f"{r.get('roe') or 0}")
        cc[3].metric("Debt/Eq", f"{r.get('debt_to_equity') or 0:.2f}")
        cc = st.columns(3)
        cc[0].metric("EV/EBITDA", f"{r.get('ev_ebitda') or 0:.2f}")
        cc[1].metric("Div yield", f"{(r.get('dividend_yield') or 0)*100:.2f}%")
        cc[2].metric("Mkt cap", f"₹{(r.get('market_cap') or 0)/1e7:,.0f} cr")

        fin = prof.get("financials", {}) or {}
        st.subheader("Annual P&L")
        if fin.get("income"):
            st.dataframe(pd.DataFrame(fin["income"]),
                         use_container_width=True, hide_index=True)
        st.subheader("Balance sheet")
        if fin.get("balance"):
            st.dataframe(pd.DataFrame(fin["balance"]),
                         use_container_width=True, hide_index=True)
        st.subheader("Cash flow")
        if fin.get("cashflow"):
            st.dataframe(pd.DataFrame(fin["cashflow"]),
                         use_container_width=True, hide_index=True)
        st.subheader("Quarterly")
        if prof.get("quarterly"):
            st.dataframe(pd.DataFrame(prof["quarterly"]),
                         use_container_width=True, hide_index=True)

    # ── Earnings & Guidance ─────────────────────────────────────────────
    with tabs[2]:
        calls = get(f"/api/stock/earnings/{sym}") or []
        if calls:
            for c in calls:
                with st.expander(f"{c.get('quarter')} – "
                                 f"{c.get('call_date') or 'no date'}"):
                    st.write("**Highlights**")
                    hl = c.get("highlights", {})
                    for h_ in (hl.get("highlights") or []):
                        st.markdown(f"- {h_}")
                    st.write("**Guidance**")
                    st.json(c.get("guidance") or {})
                    if c.get("transcript_url"):
                        st.markdown(f"[transcript pdf]({c['transcript_url']})")
        with st.form("analyse_call"):
            st.caption("Add a transcript URL → Claude extracts highlights")
            qq = st.text_input("Quarter (e.g. Q3-2024)")
            url = st.text_input("Transcript PDF URL")
            if st.form_submit_button("Analyse with Claude"):
                if qq and url:
                    res = post("/api/stock/earnings/analyse",
                               json={"symbol": sym, "quarter": qq,
                                     "transcript_url": url})
                    st.json(res)
                else:
                    st.warning("quarter and URL required")

    # ── Estimates ───────────────────────────────────────────────────────
    with tabs[3]:
        est = prof.get("estimates", {}) or {}
        cc = st.columns(4)
        cc[0].metric("Target high", f"₹{est.get('target_high', 0) or 0:.2f}")
        cc[1].metric("Target mean", f"₹{est.get('target_mean', 0) or 0:.2f}")
        cc[2].metric("Target low", f"₹{est.get('target_low', 0) or 0:.2f}")
        cc[3].metric("EPS (FY)", f"{est.get('eps_current_year') or 0}")
        rd = est.get("rating_distribution", {}) or {}
        if rd:
            st.subheader("Rating distribution")
            st.bar_chart(pd.Series(rd))

    # ── Shareholding ────────────────────────────────────────────────────
    with tabs[4]:
        sh = prof.get("shareholding", {}) or {}
        st.write("**Current**")
        st.json(sh.get("current") or {})
        if sh.get("history"):
            df = pd.DataFrame(sh["history"]).set_index("quarter")
            st.line_chart(df)

    # ── Corporate actions ──────────────────────────────────────────────
    with tabs[5]:
        a = prof.get("actions", {}) or {}
        st.subheader("Dividends")
        if a.get("dividends"):
            st.dataframe(pd.DataFrame(a["dividends"]),
                         use_container_width=True, hide_index=True)
        st.subheader("Splits")
        if a.get("splits"):
            st.dataframe(pd.DataFrame(a["splits"]),
                         use_container_width=True, hide_index=True)

    # ── News ────────────────────────────────────────────────────────────
    with tabs[6]:
        for n in prof.get("news", []) or []:
            st.markdown(f"**[{n.get('source', '')}]** "
                        f"[{n.get('title', '')}]({n.get('url', '#')})  "
                        f"_{n.get('publishedAt', '')}_")
        if not prof.get("news"):
            st.caption("no headlines (NewsAPI key missing or rate-limited)")

    # ── Peers ───────────────────────────────────────────────────────────
    with tabs[7]:
        peers = prof.get("peers") or []
        if peers:
            st.dataframe(pd.DataFrame(peers),
                         use_container_width=True, hide_index=True)
        else:
            st.caption("no peers configured for this sector")

    # ── Watchlist toggle ───────────────────────────────────────────────
    st.divider()
    if st.button(f"⭐ Add {sym} to watchlist"):
        st.json(post("/api/watchlist", json={"symbol": sym}))
