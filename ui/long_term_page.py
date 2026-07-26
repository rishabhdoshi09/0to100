"""
💎 Long-Term Picks page — durable compounders to hold for months.

A calm, separate lens from the fast breakout scanner. Screen the market for
long-term uptrends, see the tracked picks and how they're doing, and the honest
record of calls that were REVISED (exited) when their thesis broke. The weekly
daemon runs this automatically and alerts on Telegram; this page is the on-demand
view. Fail-open throughout.
"""
from __future__ import annotations


def _live_price(sym: str) -> float:
    try:
        from data.bhavcopy_store import get_ohlcv
        df = get_ohlcv(sym)
        return float(df["close"].iloc[-1]) if df is not None and len(df) else 0.0
    except Exception:
        return 0.0


def render_long_term() -> None:
    import streamlit as st

    st.subheader("💎 Long-Term Picks")
    st.caption("Mahine bhar rakhne layak stocks — durable uptrend, rising 200-DMA, "
               "momentum, leadership. Yeh short-term breakout se ALAG hai. Har "
               "hafte system khud scan karta hai aur Telegram pe alert deta hai; "
               "thesis toote toh EXIT-alert bhi.")

    # ── the tracked picks (what we're holding + revising) ─────────────────────
    try:
        from core.long_term_tracker import active_picks, exited_picks
        active = active_picks()
    except Exception:
        active = []
    if active:
        st.markdown("#### 📌 Tracked picks")
        for p in active:
            entry = float(p.get("entry_price") or 0)
            cur = _live_price(p["symbol"])
            ret = ((cur - entry) / entry * 100) if entry > 0 and cur > 0 else None
            rcol = "#00d4a0" if (ret or 0) >= 0 else "#ff4b4b"
            rtxt = (f"<span style='color:{rcol};font-weight:700'>{ret:+.1f}%</span>"
                    if ret is not None else "<span style='color:#8892a4'>—</span>")
            st.markdown(
                f"<div style='background:#0d1421;border:1px solid #1e293b;"
                f"border-left:3px solid #22d3ee;border-radius:8px;padding:9px 14px;"
                f"margin-bottom:6px'>"
                f"<b style='font-family:JetBrains Mono,monospace'>{p['symbol']}</b> "
                f"<span style='color:#8892a4;font-size:.8rem'>· added ₹{entry:,.0f} "
                f"· now ₹{cur:,.0f} · </span>{rtxt} "
                f"<span style='color:#8892a4;font-size:.8rem'>· score "
                f"{p.get('score',0):.0f}</span>"
                f"<div style='color:#c9d1d9;font-size:.8rem;margin-top:2px'>"
                f"{p.get('thesis','')}</div></div>", unsafe_allow_html=True)
    else:
        st.info("Abhi koi tracked long-term pick nahi. Neeche scan chala ke "
                "candidates dekho — jo pasand aaye, track pe daal do.")

    # ── scan the market on demand ─────────────────────────────────────────────
    st.markdown("#### 🔍 Screen the market")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        _scope = st.radio("Scope", ["NIFTY 500 (fast)", "Whole market"],
                          horizontal=True, key="lt_scope", label_visibility="collapsed")
    with c2:
        _go = st.button("🔍 Find long-term picks", key="lt_scan", type="primary",
                        width="stretch")

    if _go or st.session_state.get("lt_last_results"):
        @st.cache_data(ttl=1800, show_spinner="💎 Screening the market…")
        def _scan(whole: bool) -> list:
            from scan.long_term import scan_long_term
            syms = None
            if not whole:
                try:
                    from data.nse_universe import get_nifty500_universe
                    syms = get_nifty500_universe()
                except Exception:
                    syms = None
            return scan_long_term(symbols=syms, top=30)

        if _go:
            st.session_state["lt_last_results"] = _scan("Whole" in _scope)
        picks = st.session_state.get("lt_last_results") or []
        if not picks:
            st.caption("Abhi koi strong long-term candidate nahi mila (ya bhavcopy "
                       "store warm nahi hua).")
        else:
            st.caption(f"{len(picks)} candidates — score ke hisaab se ranked.")
            _tracked = {p["symbol"] for p in active}
            for p in picks:
                cc1, cc2 = st.columns([5, 1])
                with cc1:
                    m = (f"12m {p.get('mom_12m_pct',0):+.0f}% · "
                         f"{p.get('from_high_pct',0):.0f}% below high · "
                         f"₹{p.get('turnover_cr',0):.0f}cr/day")
                    st.markdown(
                        f"<div style='padding:6px 0'>"
                        f"<b style='font-family:JetBrains Mono,monospace'>{p['symbol']}</b> "
                        f"<span style='color:#22d3ee;font-weight:700'>score "
                        f"{p['score']:.0f}</span> "
                        f"<span style='color:#8892a4;font-size:.78rem'>₹{p.get('price',0):,.0f} "
                        f"· {m}</span>"
                        f"<div style='color:#c9d1d9;font-size:.8rem'>{p.get('thesis','')}</div>"
                        f"</div>", unsafe_allow_html=True)
                with cc2:
                    if p["symbol"] in _tracked:
                        st.caption("✓ tracked")
                    elif st.button("Track", key=f"lt_track_{p['symbol']}",
                                   width="stretch"):
                        try:
                            from core.long_term_tracker import record_picks
                            record_picks([p])
                            st.rerun()
                        except Exception:
                            pass

    # ── revised (exited) — the honest record ──────────────────────────────────
    try:
        ex = exited_picks(limit=12)
    except Exception:
        ex = []
    if ex:
        with st.expander("🔄 Revised / exited picks (thesis broke)"):
            for e in ex:
                col = "#00d4a0" if (e.get("return_pct") or 0) >= 0 else "#ff4b4b"
                st.markdown(
                    f"<div style='font-size:.82rem;margin-bottom:4px'>"
                    f"<b>{e['symbol']}</b> "
                    f"<span style='color:{col}'>{e.get('return_pct',0):+.1f}%</span> "
                    f"<span style='color:#8892a4'>— {e.get('review_note','')}</span>"
                    f"</div>", unsafe_allow_html=True)
