"""
🇺🇸 US Scanner — the same setup engine, on US equities.

Confirmed breakouts, conviction, base quality and chart patterns, run
over a curated liquid US universe with the S&P 500 as the relative-
strength benchmark. Daily-signal driven; live prices via yfinance with
an honest freshness label (Kite serves no US data; true tick-level US
needs a paid feed).
"""
from __future__ import annotations

import streamlit as st

_CARD = ("background:#0d1421;border:1px solid #1e293b;border-radius:10px;"
         "padding:12px 16px;margin-bottom:8px")
_CAT_COLOR = {"Momentum": "#38bdf8", "Breakout": "#a78bfa",
              "Pattern": "#f472b6", "PreBreakout": "#facc15",
              "Pullback": "#34d399"}


@st.cache_data(ttl=900, show_spinner=False)
def _cached_scan() -> list:
    from scan.us_scanner import scan_us
    return scan_us()


def render_us_scanner() -> None:
    st.markdown("### 🇺🇸 US Scanner")
    try:
        from data.us_data import us_market_open
        is_open = us_market_open()
    except Exception:
        is_open = False
    chip = ("<span style='color:#00d4a0'>🟢 US OPEN</span>" if is_open
            else "<span style='color:#8892a4'>🔴 US CLOSED</span>")
    st.markdown(
        f"<div style='font-size:.75rem;color:#8892a4;margin-bottom:8px'>"
        f"{chip} · same engine as NSE — confirmed breakouts, conviction, "
        f"base quality, patterns · RS benchmark S&P 500 · daily signals, "
        f"yfinance data (delivery% US mein nahi hota → neutral)</div>",
        unsafe_allow_html=True)

    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("🔍 Scan US", key="us_scan_btn", width="stretch"):
            _cached_scan.clear()
            st.session_state["_us_scan_go"] = True
            st.rerun()

    if not st.session_state.get("_us_scan_go") and not _has_cached():
        st.caption("**Scan US** dabao — pehli baar ~1-2 min (yfinance se "
                   "daily data), phir cached. Indian market band ho toh yahan "
                   "US setups dekho.")
        return

    with st.spinner("US universe scan ho raha hai…"):
        results = _cached_scan()

    if not results:
        st.warning("Koi US setup nahi mila (ya data fetch fail — dobara try "
                   "karo). Market band ho toh signals EOD pe bante hain.")
        return

    n_buy = sum(1 for r in results if r["verdict"] in ("STRONG BUY", "BUY"))
    n_hc = sum(1 for r in results if r.get("high_conviction"))
    st.markdown(
        f"<div style='{_CARD};font-size:.82rem;color:#c9d1d9'>"
        f"<b>{len(results)}</b> setups · "
        f"<span style='color:#00d4a0'>⚡ {n_buy} buy</span> · "
        f"<span style='color:#fbbf24'>🎯 {n_hc} high-conviction</span></div>",
        unsafe_allow_html=True)

    # ── Category filter — poora spectrum, sirf breakouts nahi ────────────────
    _cats = ["All", "Breakout", "Pullback", "Pattern", "PreBreakout", "Momentum"]
    _counts = {c: sum(1 for r in results if c in r.get("categories", []))
               for c in _cats[1:]}
    _pick = st.radio(
        "Type", [c + (f" ({_counts[c]})" if c in _counts and _counts[c] else "")
                 for c in _cats],
        horizontal=True, key="us_cat_filter", label_visibility="collapsed")
    _pick_cat = _pick.split(" (")[0]
    if _pick_cat != "All":
        results = [r for r in results if _pick_cat in r.get("categories", [])]
        if not results:
            st.caption(f"Abhi koi '{_pick_cat}' setup nahi.")
            return

    for r in results[:30]:
        v = r.get("verdict", "WATCH")
        v_col = "#22d3ee" if v == "STRONG BUY" else (
            "#00d4a0" if v == "BUY" else "#8892a4")
        hc = "🎯 " if r.get("high_conviction") else ""
        conv = r.get("breakout_conviction") or 0
        conv_bit = (f" · conviction {conv:.0f}" if conv else "")
        sigs = " · ".join((r.get("signals") or [])[:3])
        reason = (r.get("reasons") or [""])[0]
        st.markdown(
            f"<div style='{_CARD}'>"
            f"<span style='font-weight:800;font-family:JetBrains Mono,"
            f"monospace;color:#e6edf3'>{hc}{r['symbol']}</span> "
            f"<span style='color:#e2e8f0'>${float(r.get('price') or 0):,.2f}</span> "
            f"<span style='color:{v_col};font-weight:700;font-size:.8rem'>"
            f"{v}</span> <span style='font-size:.72rem;color:#8892a4'>"
            f"score {float(r.get('score') or 0):.0f}{conv_bit}</span>"
            f"<div style='font-size:.75rem;color:#94a3b8;margin-top:4px'>{sigs}</div>"
            f"<div style='font-size:.76rem;color:#c9d1d9;margin-top:3px'>{reason}</div>"
            f"<div style='font-size:.72rem;color:#64748b;margin-top:3px'>"
            f"entry ${float(r.get('entry') or 0):,.2f} · "
            f"stop ${float(r.get('stop') or 0):,.2f} · "
            f"target ${float(r.get('target') or 0):,.2f}</div>"
            f"</div>", unsafe_allow_html=True)


def _has_cached() -> bool:
    try:
        from scan.us_scanner import get_us_results
        return bool(get_us_results()[0])
    except Exception:
        return False
