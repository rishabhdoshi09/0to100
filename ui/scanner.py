"""
Market Scanner — simple, actionable stock list.

One unified view: stocks worth watching today, sorted by signal strength.
Each card shows exactly what a trader needs: price, signal, entry/stop/target.
"""
from __future__ import annotations

import streamlit as st


# ── Data fetchers (cached) ────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_elite(symbols_key: str, top_n: int = 20):
    """Full quality pipeline — returns RankedSetup list with entry/stop/target."""
    from scan.pipeline import ScanPipeline
    from core.regime_engine import compute_regime
    symbols = [s.strip() for s in symbols_key.split(",") if s.strip()][:500]
    try:
        regime = compute_regime()
        return ScanPipeline(max_workers=12, min_quality_score=38).run(
            symbols, top_n=top_n, regime_state=regime, skip_liquidity_filter=True,
        )
    except Exception:
        return []


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_momentum(symbols_key: str, top_n: int = 30):
    """Lightweight momentum scan — fallback when elite pipeline is slow."""
    from screener.momentum_scanner import MomentumScanner
    symbols = [s.strip() for s in symbols_key.split(",") if s.strip()]
    try:
        return MomentumScanner().scan_momentum(symbols, top_n=top_n)
    except Exception:
        return []


@st.cache_data(ttl=86_400, show_spinner=False)
def _full_nse_universe() -> list[str]:
    try:
        from screener.universe import StockUniverseFetcher
        return StockUniverseFetcher().get_all_symbols()
    except Exception:
        return []


# ── Card renderers ────────────────────────────────────────────────────────────

def _signal_badge(verdict: str) -> tuple[str, str]:
    """(label, color) for verdict string."""
    return {
        "READY_TO_TRADE": ("⚡ Buy Signal", "#00d4a0"),
        "BUILDING":       ("👁 Watch",       "#f59e0b"),
        "AVOID":          ("✗ Avoid",        "#ff4b4b"),
    }.get(verdict, ("👁 Watch", "#f59e0b"))


def _render_elite_card(s) -> None:
    """One clean trade card from a RankedSetup."""
    verdict = getattr(s, "verdict", "BUILDING")
    label, color = _signal_badge(verdict)

    entry  = getattr(s, "pivot_level", 0) or 0
    stop   = getattr(s, "stop_level",  0) or 0
    target = getattr(s, "target_price", 0) or 0
    price  = getattr(s, "price", entry) or entry
    risk   = abs(entry - stop) if entry and stop else 0
    rr     = abs(target - entry) / risk if (risk > 0 and target) else 0

    chg_pct = getattr(s, "momentum_5d_pct", 0) or 0
    chg_color = "#00d4a0" if chg_pct >= 0 else "#ff4b4b"
    chg_arrow = "▲" if chg_pct >= 0 else "▼"

    # Trade plan line
    plan_html = ""
    if entry and stop and target:
        plan_html = (
            f"<div style='margin-top:6px;font-size:.78rem;color:#94a3b8'>"
            f"Entry <span style='color:#f59e0b;font-weight:700'>₹{entry:,.0f}</span>"
            f" &nbsp;·&nbsp; Stop <span style='color:#ff4b4b;font-weight:700'>₹{stop:,.0f}</span>"
            f" &nbsp;·&nbsp; Target <span style='color:#00d4a0;font-weight:700'>₹{target:,.0f}</span>"
            f" &nbsp;·&nbsp; <span style='color:#e2e8f0'>Reward {rr:.1f}×</span>"
            f"</div>"
        )
    elif entry:
        plan_html = (
            f"<div style='margin-top:6px;font-size:.78rem;color:#94a3b8'>"
            f"Near <span style='color:#f59e0b;font-weight:700'>₹{entry:,.0f}</span> — watch for breakout"
            f"</div>"
        )

    sym = s.symbol
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown(
            f"<div style='background:#0d1421;border:1px solid #1e293b;border-radius:10px;"
            f"padding:12px 16px;margin-bottom:6px'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"  <div>"
            f"    <span style='color:#e2e8f0;font-weight:700;font-size:1rem;"
            f"      font-family:JetBrains Mono,monospace'>{sym}</span>"
            f"    <span style='color:#e2e8f0;font-size:.95rem;margin-left:12px'>₹{price:,.1f}</span>"
            f"    <span style='color:{chg_color};font-size:.82rem;margin-left:6px;font-weight:600'>"
            f"      {chg_arrow}{abs(chg_pct):.1f}%</span>"
            f"  </div>"
            f"  <span style='background:{color}18;border:1px solid {color}55;border-radius:6px;"
            f"    padding:3px 10px;font-size:.7rem;font-weight:700;color:{color}'>{label}</span>"
            f"</div>"
            f"{plan_html}"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col2:
        if st.button("Analyse →", key=f"scan_analyse_{sym}", use_container_width=True):
            st.session_state["sidebar_nav"] = "Terminal"
            st.session_state["terminal_symbol"] = sym
            st.rerun()


def _render_momentum_card(s) -> None:
    """Simpler card for MomentumStock (no full trade plan)."""
    sig = getattr(s, "signal", "NEUTRAL")
    if sig == "BUY":
        label, color = "⚡ Buy Signal", "#00d4a0"
    elif sig == "WATCH":
        label, color = "👁 Watch", "#f59e0b"
    else:
        return  # skip NEUTRAL in simple view

    chg_color = "#00d4a0" if s.change_pct >= 0 else "#ff4b4b"
    chg_arrow = "▲" if s.change_pct >= 0 else "▼"
    sym = s.symbol

    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown(
            f"<div style='background:#0d1421;border:1px solid #1e293b;border-radius:10px;"
            f"padding:12px 16px;margin-bottom:6px'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"  <div>"
            f"    <span style='color:#e2e8f0;font-weight:700;font-size:1rem;"
            f"      font-family:JetBrains Mono,monospace'>{sym}</span>"
            f"    <span style='color:#e2e8f0;font-size:.95rem;margin-left:12px'>₹{s.price:,.1f}</span>"
            f"    <span style='color:{chg_color};font-size:.82rem;margin-left:6px;font-weight:600'>"
            f"      {chg_arrow}{abs(s.change_pct):.1f}%</span>"
            f"  </div>"
            f"  <span style='background:{color}18;border:1px solid {color}55;border-radius:6px;"
            f"    padding:3px 10px;font-size:.7rem;font-weight:700;color:{color}'>{label}</span>"
            f"</div>"
            f"<div style='margin-top:6px;font-size:.75rem;color:#4a5568'>Building momentum — watch for entry</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col2:
        if st.button("Analyse →", key=f"scan_mom_{sym}", use_container_width=True):
            st.session_state["sidebar_nav"] = "Terminal"
            st.session_state["terminal_symbol"] = sym
            st.rerun()


# ── Main render ───────────────────────────────────────────────────────────────

def render_scanner(universe: list[str]) -> None:
    # ── Universe selector ─────────────────────────────────────────────────────
    try:
        from data.nse_universe import get_nifty500_universe
        nifty500 = get_nifty500_universe()
    except Exception:
        nifty500 = universe[:500]

    scope_options = {
        f"NIFTY 500 ({len(nifty500)} stocks — fast)": nifty500,
        f"All NSE ({len(universe)} stocks — slow)": universe,
    }
    scope_label = st.selectbox(
        "Universe",
        list(scope_options.keys()),
        index=0,
        key="scanner_scope",
        label_visibility="collapsed",
    )
    syms = scope_options[scope_label]

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        st.caption(f"Scanning **{len(syms)} stocks** · results cached 5 min")
    with c2:
        run_scan = st.button("🔍 Scan Now", key="scanner_run",
                             type="primary", use_container_width=True)
    with c3:
        buy_only = st.toggle("Buy only", value=False, key="scanner_buy_only")

    if run_scan:
        st.cache_data.clear()

    symbols_key = ",".join(syms)

    # ── Run elite pipeline (entry/stop/target) ────────────────────────────────
    with st.spinner(f"Scanning {len(syms)} stocks…"):
        elite = _fetch_elite(symbols_key, top_n=25)

    # ── Filter ────────────────────────────────────────────────────────────────
    if buy_only:
        elite = [s for s in elite if getattr(s, "verdict", "") == "READY_TO_TRADE"]

    # ── Results ───────────────────────────────────────────────────────────────
    ready   = [s for s in elite if getattr(s, "verdict", "") == "READY_TO_TRADE"]
    watch   = [s for s in elite if getattr(s, "verdict", "") == "BUILDING"]
    avoid   = [s for s in elite if getattr(s, "verdict", "") == "AVOID"]

    total = len(ready) + len(watch)
    if total == 0:
        st.info("No setups found. Try scanning All NSE or check back later.")
        # Fallback to lightweight momentum scan
        with st.spinner("Running quick momentum scan…"):
            mom = _fetch_momentum(symbols_key, top_n=20)
        buys = [s for s in mom if s.signal == "BUY"]
        if buys:
            st.markdown(f"**{len(buys)} momentum signals found:**")
            for s in buys[:10]:
                _render_momentum_card(s)
        return

    # Summary line
    parts = []
    if ready: parts.append(f"⚡ {len(ready)} ready to buy")
    if watch:  parts.append(f"👁 {len(watch)} to watch")
    st.markdown(
        f"<div style='background:#0d1421;border:1px solid #1e293b;border-radius:8px;"
        f"padding:8px 14px;margin-bottom:12px;font-size:.82rem;color:#c9d1d9'>"
        f"{'  ·  '.join(parts)}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Buy signals first
    if ready:
        st.markdown("#### ⚡ Ready to Buy")
        for s in ready:
            _render_elite_card(s)

    # Watch list
    if watch and not buy_only:
        st.markdown("#### 👁 Watch These")
        for s in watch[:8]:
            _render_elite_card(s)

    # Avoid — collapsed
    if avoid and not buy_only:
        with st.expander(f"✗ Avoid ({len(avoid)} stocks)"):
            for s in avoid[:5]:
                _render_elite_card(s)
