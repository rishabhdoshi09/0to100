"""
Smart Scanner — the whole market, always current, zero wait.

A background auto-scan (scan/auto_scan.py) covers the ENTIRE NSE and
refreshes every 15 min during market hours. This page just reads the
store — results appear instantly. Every BUY is logged to the outcome
tracker, so the accuracy shown here is measured, not promised.
"""
from __future__ import annotations

import time
from datetime import datetime

import streamlit as st

_CATEGORY_TABS = ["🔥 All Signals", "🚀 Momentum", "💥 Breakouts", "📐 Chart Patterns"]
_CATEGORY_MAP = {
    "🚀 Momentum": "Momentum",
    "💥 Breakouts": "Breakout",
    "📐 Chart Patterns": "Pattern",
}

_VERDICT_STYLE = {
    "STRONG BUY": ("🔥 Strong Buy", "#22d3ee"),
    "BUY":        ("⚡ Buy Signal", "#00d4a0"),
    "WATCH":      ("👁 Watch",      "#f59e0b"),
}
_BUY_VERDICTS = ("STRONG BUY", "BUY")

_CAT_COLOR = {"Momentum": "#38bdf8", "Breakout": "#a78bfa", "Pattern": "#f472b6"}


@st.cache_data(ttl=600, show_spinner=False)
def _accuracy_stat() -> str:
    """Measured accuracy from the outcome tracker — '' if not enough data yet."""
    try:
        from core.signal_outcome_tracker import get_accuracy_report
        rep = get_accuracy_report()
        closed = (rep.get("wins", 0) or 0) + (rep.get("losses", 0) or 0)
        if closed >= 5:
            return (f"🎯 Verified accuracy: <b>{rep.get('overall_accuracy', 0):.0f}%</b> "
                    f"on {closed} tracked signals")
    except Exception:
        pass
    return ""


def _freshness(ts: float) -> str:
    if not ts:
        return ""
    mins = int((time.time() - ts) / 60)
    if mins < 1:
        return "just now"
    if mins < 60:
        return f"{mins} min ago"
    return datetime.fromtimestamp(ts).strftime("%I:%M %p")


# ── Card renderer ─────────────────────────────────────────────────────────────

def _render_card(s: dict, key_prefix: str = "") -> None:
    label, vcolor = _VERDICT_STYLE.get(s["verdict"], _VERDICT_STYLE["WATCH"])
    chg = s["change_pct"]
    chg_color = "#00d4a0" if chg >= 0 else "#ff4b4b"
    chg_arrow = "▲" if chg >= 0 else "▼"

    chips = "".join(
        f"<span style='background:#1e293b;border-radius:5px;padding:2px 8px;"
        f"font-size:.68rem;color:#94a3b8;margin-right:6px'>{sig}</span>"
        for sig in s["signals"]
    )

    # Conviction checklist (from JARVIS layer) or plain scanner reasons
    checks = s.get("checks") or []
    if checks:
        reason = "<br>".join(
            f"<span style='color:{'#00d4a0' if c.startswith('✓') else '#f59e0b' if c.startswith('⚠') else '#94a3b8'}'>"
            f"{c}</span>"
            for c in checks[:5]
        )
    else:
        reason = " · ".join(s["reasons"][:2])
    plan = (
        f"<div style='margin-top:7px;font-size:.78rem;color:#94a3b8'>"
        f"Entry <span style='color:#f59e0b;font-weight:700'>₹{s['entry']:,.0f}</span>"
        f" &nbsp;·&nbsp; Stop <span style='color:#ff4b4b;font-weight:700'>₹{s['stop']:,.0f}</span>"
        f" &nbsp;·&nbsp; Target <span style='color:#00d4a0;font-weight:700'>₹{s['target']:,.0f}</span>"
        + (f" &nbsp;·&nbsp; <span style='color:#e2e8f0'>Reward {s['rr']:.1f}×</span>" if s["rr"] > 0 else "")
        + "</div>"
    )

    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown(
            f"<div style='background:#0d1421;border:1px solid #1e293b;border-radius:10px;"
            f"padding:12px 16px;margin-bottom:6px'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"  <div>"
            f"    <span style='color:#e2e8f0;font-weight:700;font-size:1rem;"
            f"      font-family:JetBrains Mono,monospace'>{s['symbol']}</span>"
            f"    <span style='color:#e2e8f0;font-size:.95rem;margin-left:12px'>₹{s['price']:,.1f}</span>"
            f"    <span style='color:{chg_color};font-size:.82rem;margin-left:6px;font-weight:600'>"
            f"      {chg_arrow}{abs(chg):.1f}%</span>"
            f"  </div>"
            f"  <span style='background:{vcolor}18;border:1px solid {vcolor}55;border-radius:6px;"
            f"    padding:3px 10px;font-size:.7rem;font-weight:700;color:{vcolor}'>{label}</span>"
            f"</div>"
            f"<div style='margin-top:7px'>{chips}</div>"
            f"<div style='margin-top:6px;font-size:.78rem;color:#c9d1d9'>{reason}</div>"
            f"{plan}"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col2:
        if st.button("Analyse →", key=f"scan_{key_prefix}_{s['symbol']}", use_container_width=True):
            st.session_state["sidebar_nav"] = "Terminal"
            st.session_state["terminal_symbol"] = s["symbol"]
            st.rerun()


# ── Main render ───────────────────────────────────────────────────────────────

def render_scanner(universe: list[str]) -> None:
    from scan.auto_scan import start_background_scan, force_rescan, get_results
    start_background_scan()   # idempotent — first visitor kicks it off

    results, universe_size, last_ts, status = get_results()

    # ── Header ────────────────────────────────────────────────────────────────
    c1, c2 = st.columns([4, 1])
    with c1:
        st.markdown("### 🔍 Smart Scanner — whole market")
        sub = f"Momentum · Breakouts · Chart patterns across **all {universe_size or len(universe)} NSE stocks**"
        if last_ts:
            sub += f" · updated **{_freshness(last_ts)}**"
            if status == "scanning":
                sub += " · ⟳ refreshing…"
        st.caption(sub)
    with c2:
        if st.button("⟳ Rescan", key="scanner_run", type="primary",
                     use_container_width=True, disabled=(status == "scanning")):
            force_rescan()
            st.rerun()

    # ── First scan still running ──────────────────────────────────────────────
    if not results:
        if status in ("scanning", "idle"):
            st.info("⏳ First market scan is running in the background (~3-4 min for "
                    "the full NSE). You can use the rest of the app — results will "
                    "be waiting here.")
            if st.button("⟳ Check again", key="scanner_poll"):
                st.rerun()
        else:
            st.warning("Scan couldn't fetch market data. Check your internet "
                       "connection, then hit **Rescan**.")
        return

    # ── Summary strip ─────────────────────────────────────────────────────────
    n_strong = sum(1 for r in results if r["verdict"] == "STRONG BUY")
    n_buy = sum(1 for r in results if r["verdict"] in _BUY_VERDICTS)
    n_mom = sum(1 for r in results if "Momentum" in r["categories"])
    n_brk = sum(1 for r in results if "Breakout" in r["categories"])
    n_pat = sum(1 for r in results if "Pattern" in r["categories"])
    acc = _accuracy_stat()
    st.markdown(
        f"<div style='background:#0d1421;border:1px solid #1e293b;border-radius:8px;"
        f"padding:9px 14px;margin:4px 0 12px 0;font-size:.84rem;color:#c9d1d9'>"
        f"<b>{universe_size}</b> stocks scanned → <b>{len(results)}</b> signals &nbsp;·&nbsp; "
        + (f"<span style='color:#22d3ee'>🔥 {n_strong} strong buy</span> &nbsp;·&nbsp; " if n_strong else "")
        + f"<span style='color:#00d4a0'>⚡ {n_buy} buy</span> &nbsp;·&nbsp; "
        f"<span style='color:{_CAT_COLOR['Momentum']}'>🚀 {n_mom} momentum</span> &nbsp;·&nbsp; "
        f"<span style='color:{_CAT_COLOR['Breakout']}'>💥 {n_brk} breakouts</span> &nbsp;·&nbsp; "
        f"<span style='color:{_CAT_COLOR['Pattern']}'>📐 {n_pat} patterns</span>"
        + (f" &nbsp;·&nbsp; {acc}" if acc else "")
        + "</div>",
        unsafe_allow_html=True,
    )

    # ── Telegram push hint (only when not configured) ─────────────────────────
    try:
        from alerts.telegram_alerts import AlertEngine
        if not AlertEngine().is_configured():
            st.caption("💡 Naye setups **khud aap tak** pahunch sakte hain — `.env` mein "
                       "`TELEGRAM_BOT_TOKEN` & `TELEGRAM_CHAT_ID` daalo, har scan ke baad "
                       "fresh Buy setups Telegram pe milenge.")
    except Exception:
        pass

    # ── Search within results ─────────────────────────────────────────────────
    q = st.text_input("Filter", placeholder="🔍 Filter by symbol…",
                      key="scanner_filter", label_visibility="collapsed")
    if q:
        qq = q.strip().upper()
        results = [r for r in results if qq in r["symbol"]]

    # ── Category tabs ─────────────────────────────────────────────────────────
    tabs = st.tabs(_CATEGORY_TABS)
    for tab, tab_name in zip(tabs, _CATEGORY_TABS):
        with tab:
            cat = _CATEGORY_MAP.get(tab_name)
            subset = results if cat is None else [
                r for r in results if cat in r["categories"]]
            if not subset:
                st.caption("Nothing in this category right now.")
                continue

            buy_only = st.toggle("⚡ Buy signals only", value=False,
                                 key=f"buyonly_{tab_name}")
            if buy_only:
                subset = [r for r in subset if r["verdict"] in _BUY_VERDICTS]

            buys = [r for r in subset if r["verdict"] in _BUY_VERDICTS]
            watch = [r for r in subset if r["verdict"] == "WATCH"]

            kp = tab_name.split(" ")[-1].lower()
            for r in buys[:20]:
                _render_card(r, key_prefix=kp)
            if watch and not buy_only:
                st.markdown("###### 👁 Worth watching")
                for r in watch[:20]:
                    _render_card(r, key_prefix=kp)
