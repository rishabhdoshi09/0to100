"""
Smart Scanner — every signal engine, one clean page.

Momentum + Breakouts (52W high, resistance, golden cross, squeeze)
+ Chart Patterns (VCP, flat base, cup & handle, high tight flag)
run in a single pass over the whole universe from the bulk cache.

Layman-friendly: plain-English reasons, one card per stock,
entry/stop/target on every idea.
"""
from __future__ import annotations

import streamlit as st

_CATEGORY_TABS = ["🔥 All Signals", "🚀 Momentum", "💥 Breakouts", "📐 Chart Patterns"]
_CATEGORY_MAP = {
    "🚀 Momentum": "Momentum",
    "💥 Breakouts": "Breakout",
    "📐 Chart Patterns": "Pattern",
}

_VERDICT_STYLE = {
    "BUY":   ("⚡ Buy Signal", "#00d4a0"),
    "WATCH": ("👁 Watch",      "#f59e0b"),
}

_CAT_COLOR = {"Momentum": "#38bdf8", "Breakout": "#a78bfa", "Pattern": "#f472b6"}


# ── Cached scan ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _run_unified_scan(symbols_key: str) -> list[dict]:
    """Run the unified scanner and return plain dicts (picklable for cache)."""
    from scan.unified_scanner import UnifiedScanner
    symbols = [s for s in symbols_key.split(",") if s]
    try:
        results = UnifiedScanner(max_workers=8).scan(symbols)
    except Exception:
        return []
    return [{
        "symbol": r.symbol, "price": r.price, "change_pct": r.change_pct,
        "momentum_5d": r.momentum_5d, "volume_ratio": r.volume_ratio,
        "signals": r.signal_labels, "categories": sorted(r.categories),
        "reasons": r.reasons, "score": r.score, "verdict": r.verdict,
        "entry": r.entry, "stop": r.stop, "target": r.target,
        "rr": round(r.risk_reward, 1),
    } for r in results]


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
    try:
        from data.nse_universe import get_nifty500_universe
        nifty500 = get_nifty500_universe()
    except Exception:
        nifty500 = universe[:500]

    # ── Top controls ──────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([2.2, 1.6, 1])
    with c1:
        st.markdown("### 🔍 Smart Scanner")
        st.caption("Momentum · Breakouts · Chart patterns — one scan, whole market")
    with c2:
        scope_options = {
            f"NIFTY 500 · fast (~1 min)": nifty500,
            f"All NSE · {len(universe)} stocks (~3-4 min)": universe,
        }
        scope_label = st.selectbox("Universe", list(scope_options.keys()),
                                   index=0, key="scanner_scope",
                                   label_visibility="collapsed")
        syms = scope_options[scope_label]
    with c3:
        if st.button("🔍 Scan Now", key="scanner_run", type="primary",
                     use_container_width=True):
            _run_unified_scan.clear()

    symbols_key = ",".join(syms)
    with st.spinner(f"Scanning {len(syms)} stocks — downloading data in bulk…"):
        results = _run_unified_scan(symbols_key)

    if not results:
        st.info("No signals right now. Markets may be closed or data is unavailable — "
                "hit **Scan Now** to retry.")
        return

    # ── Summary strip ─────────────────────────────────────────────────────────
    n_buy = sum(1 for r in results if r["verdict"] == "BUY")
    n_mom = sum(1 for r in results if "Momentum" in r["categories"])
    n_brk = sum(1 for r in results if "Breakout" in r["categories"])
    n_pat = sum(1 for r in results if "Pattern" in r["categories"])
    st.markdown(
        f"<div style='background:#0d1421;border:1px solid #1e293b;border-radius:8px;"
        f"padding:9px 14px;margin:4px 0 12px 0;font-size:.84rem;color:#c9d1d9'>"
        f"Scanned <b>{len(syms)}</b> stocks → <b>{len(results)}</b> signals &nbsp;·&nbsp; "
        f"<span style='color:#00d4a0'>⚡ {n_buy} buy</span> &nbsp;·&nbsp; "
        f"<span style='color:{_CAT_COLOR['Momentum']}'>🚀 {n_mom} momentum</span> &nbsp;·&nbsp; "
        f"<span style='color:{_CAT_COLOR['Breakout']}'>💥 {n_brk} breakouts</span> &nbsp;·&nbsp; "
        f"<span style='color:{_CAT_COLOR['Pattern']}'>📐 {n_pat} patterns</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

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
                subset = [r for r in subset if r["verdict"] == "BUY"]

            buys = [r for r in subset if r["verdict"] == "BUY"]
            watch = [r for r in subset if r["verdict"] == "WATCH"]

            kp = tab_name.split(" ")[-1].lower()
            for r in buys[:15]:
                _render_card(r, key_prefix=kp)
            if watch and not buy_only:
                st.markdown("###### 👁 Worth watching")
                for r in watch[:15]:
                    _render_card(r, key_prefix=kp)
