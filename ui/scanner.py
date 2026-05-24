"""
Market Scanner UI — Momentum & Breakout panels.

Signal logic:
  BUY   — composite score ≥ 65 AND RSI < 75
           composite = momentum(40%) + RSI(35%) + volume(25%)
  WATCH — composite score 45-64
  NEUTRAL — composite < 45

Two scan modes:
  · Quick Scan  — your configured universe (instant, ~10-30 stocks)
  · Full NSE    — all ~1900 NSE EQ symbols (takes 2-4 min), yfinance
"""
from __future__ import annotations

import streamlit as st

_BREAKOUT_ICONS = {
    "52W_HIGH":         "🔴",
    "GOLDEN_CROSS":     "🟢",
    "VOL_SQUEEZE":      "🔵",
    "RESISTANCE_BREAK": "🟡",
    "CUP_HANDLE":       "⭐",
}
_SIGNAL_COLOR = {"BUY": "#00d4a0", "WATCH": "#f59e0b", "NEUTRAL": "#8892a4"}

# Market cap buckets (approx in Cr)
_MCAP_FILTER = {
    "All":        (0,        99_999_999),
    "Largecap":   (20_000,   99_999_999),
    "Midcap":     (5_000,    19_999),
    "Smallcap":   (500,       4_999),
    "Microcap":   (0,           499),
}


@st.cache_data(ttl=300, show_spinner=False)
def _cached_momentum(symbols_key: str, top_n: int):
    from screener.momentum_scanner import MomentumScanner
    symbols = [s.strip() for s in symbols_key.split(",") if s.strip()]
    return MomentumScanner().scan_momentum(symbols, top_n=top_n)


@st.cache_data(ttl=300, show_spinner=False)
def _cached_breakouts(symbols_key: str, top_n: int):
    from screener.momentum_scanner import MomentumScanner
    symbols = [s.strip() for s in symbols_key.split(",") if s.strip()]
    return MomentumScanner().scan_breakouts(symbols, top_n=top_n)


@st.cache_data(ttl=86_400, show_spinner=False)
def _get_full_universe() -> list[str]:
    try:
        from screener.universe import StockUniverseFetcher
        return StockUniverseFetcher().get_all_symbols()
    except Exception:
        return []


@st.cache_data(ttl=3_600, show_spinner=False)
def _get_mcap_cr(symbol: str) -> float:
    """Fetch market cap in Cr via yfinance. Cached 1h."""
    try:
        import yfinance as yf
        info = yf.Ticker(f"{symbol}.NS").fast_info
        mc = getattr(info, "market_cap", None)
        return round(float(mc) / 1e7, 0) if mc else 0.0
    except Exception:
        return 0.0


def _filter_by_mcap(stocks, mcap_filter: str):
    """Filter a list of MomentumStock/BreakoutStock by market cap bucket."""
    if mcap_filter == "All":
        return stocks
    lo, hi = _MCAP_FILTER[mcap_filter]
    filtered = []
    for s in stocks:
        mc = _get_mcap_cr(s.symbol)
        if mc == 0 or lo <= mc <= hi:   # include unknowns so we don't lose everything
            filtered.append(s)
    return filtered


@st.cache_data(ttl=300, show_spinner=False)
def _cached_elite_setups(symbols_key: str, top_n: int = 20):
    """Run the full quality pipeline and return RankedSetup list with verdict/accum/target."""
    from scan.pipeline import ScanPipeline
    from core.regime_engine import compute_regime
    symbols = [s.strip() for s in symbols_key.split(",") if s.strip()][:500]
    try:
        regime = compute_regime()
        results = ScanPipeline(max_workers=12, min_quality_score=38).run(
            symbols, top_n=top_n, regime_state=regime, skip_liquidity_filter=True
        )
        return results
    except Exception:
        return []


def _verdict_badge(verdict: str) -> str:
    cfg = {
        "READY_TO_TRADE": ("#00d4a0", "#00d4a014", "⚡ Ready to Buy"),
        "BUILDING":       ("#f59e0b", "#f59e0b14", "🔨 Building"),
        "AVOID":          ("#ff4b4b", "#ff4b4b14", "✗ Avoid"),
    }.get(verdict, ("#8892a4", "#8892a414", "👁 Watch"))
    fg, bg, label = cfg
    return (
        f"<span style='background:{bg};border:1px solid {fg}44;border-radius:5px;"
        f"padding:2px 8px;font-size:.62rem;font-weight:700;color:{fg};"
        f"font-family:JetBrains Mono,monospace;white-space:nowrap'>{label}</span>"
    )


def _accum_bar(score: float) -> str:
    pct = min(max(int((score + 100) / 2), 0), 100)  # -100..+100 → 0..100
    color = "#00d4a0" if score >= 0 else "#ff4b4b"
    label = "Buying Pressure" if score >= 0 else "Selling Pressure"
    return (
        f"<div style='font-size:.55rem;color:#4a5568;margin-top:2px'>{label} {score:+.0f}</div>"
        f"<div style='background:#1a2035;border-radius:3px;height:4px;width:60px'>"
        f"<div style='width:{pct}%;background:{color};height:4px;border-radius:3px'></div></div>"
    )


def _tier_label(tier: str) -> str:
    """Convert internal tier names to trader-friendly labels."""
    return {
        "ELITE_A_PLUS": "A+ Setup",
        "ELITE_A":      "A Setup",
        "A_PLUS":       "A+ Setup",
        "A":            "A Setup",
        "B":            "B Setup",
        "WATCHLIST":    "Watch",
    }.get(tier, tier.replace("_", " "))


def _render_elite_setups(setups) -> None:
    """Render the quality-pipeline ranked setups with plain-English trade cards."""
    for s in setups:
        verdict  = getattr(s, "verdict", "BUILDING")
        target   = getattr(s, "target_price", 0.0)
        accum    = getattr(s, "accum_score", 0.0)
        weekly   = getattr(s, "weekly_aligned", False)
        tier_color = {
            "ELITE_A_PLUS": "#f59e0b", "A_PLUS": "#f59e0b",
            "ELITE_A": "#00d4a0", "A": "#00d4a0",
            "B": "#60a5fa", "WATCHLIST": "#8892a4",
        }.get(s.quality_tier, "#4a5568")

        weekly_tag = (
            "<span style='color:#00d4a0;font-size:.58rem;margin-left:6px'>✓ Weekly Trend OK</span>"
            if weekly else ""
        )

        # Calculate R:R
        entry  = s.pivot_level
        stop   = s.stop_level
        risk_pts = abs(entry - stop) if entry and stop else 0
        rr     = abs(target - entry) / risk_pts if (risk_pts > 0 and target) else 0.0

        # Plain-English summary line
        summary_html = ""
        if entry and stop and target:
            summary_html = (
                f"<div style='background:#111827;border-radius:5px;padding:6px 10px;"
                f"margin-top:8px;font-size:.78rem;color:#c9d1e0;font-style:italic;"
                f"border-left:3px solid #f59e0b44'>"
                f"Buy above <span style='color:#f59e0b;font-weight:700'>₹{entry:,.1f}</span> "
                f"with stop at <span style='color:#ff4b4b;font-weight:700'>₹{stop:,.1f}</span>. "
                f"Target <span style='color:#00d4a0;font-weight:700'>₹{target:,.1f}</span> "
                f"({rr:.1f}× reward).</div>"
            )

        ev_r = getattr(s, "expected_value_r", 0.0)
        strength = min(int(getattr(s, "quality_score", 0)), 100)

        st.markdown(
            f"""
            <div style='background:#0d1421;border:1px solid #1e293b;border-radius:8px;
                        padding:12px 16px;margin-bottom:8px'>
              <!-- Header row -->
              <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:4px'>
                <div style='display:flex;align-items:center;gap:8px'>
                  <span style='color:#e2e8f0;font-weight:700;font-size:1rem;
                    font-family:JetBrains Mono,monospace'>{s.symbol}</span>
                  <span style='color:{tier_color};font-size:.65rem;font-weight:700;
                    background:{tier_color}18;border:1px solid {tier_color}44;
                    border-radius:4px;padding:1px 7px'>{_tier_label(s.quality_tier)}</span>
                  {weekly_tag}
                </div>
                {_verdict_badge(verdict)}
              </div>
              <!-- Key numbers — large & prominent -->
              <div style='display:flex;gap:20px;margin-top:8px;flex-wrap:wrap;align-items:flex-end'>
                <div>
                  <div style='font-size:.6rem;color:#4a5568;text-transform:uppercase;letter-spacing:.06em'>Entry</div>
                  <div style='font-size:.95rem;color:#f59e0b;font-weight:700;font-family:JetBrains Mono,monospace'>₹{entry:,.1f}</div>
                </div>
                <div>
                  <div style='font-size:.6rem;color:#4a5568;text-transform:uppercase;letter-spacing:.06em'>Stop</div>
                  <div style='font-size:.95rem;color:#ff4b4b;font-weight:700;font-family:JetBrains Mono,monospace'>₹{stop:,.1f}</div>
                </div>
                <div>
                  <div style='font-size:.6rem;color:#4a5568;text-transform:uppercase;letter-spacing:.06em'>Target</div>
                  <div style='font-size:.95rem;color:#00d4a0;font-weight:700;font-family:JetBrains Mono,monospace'>₹{target:,.1f}</div>
                </div>
                <div style='border-left:1px solid #1e293b;padding-left:16px'>
                  <div style='font-size:.6rem;color:#4a5568;text-transform:uppercase;letter-spacing:.06em'>Risk:Reward</div>
                  <div style='font-size:.88rem;color:#e2e8f0;font-weight:700'>{rr:.1f}×</div>
                </div>
                <div>
                  <div style='font-size:.6rem;color:#4a5568;text-transform:uppercase;letter-spacing:.06em'>Strength</div>
                  <div style='font-size:.88rem;color:#e2e8f0;font-weight:700'>{strength}/100</div>
                </div>
                <div>
                  <div style='font-size:.6rem;color:#4a5568;text-transform:uppercase;letter-spacing:.06em'>Edge per trade</div>
                  <div style='font-size:.88rem;color:#60a5fa;font-weight:700'>+{ev_r:.1f}R</div>
                </div>
              </div>
              <!-- Plain-English summary -->
              {summary_html}
              <!-- Evidence + Buying Pressure bar -->
              <div style='display:flex;align-items:center;gap:16px;margin-top:8px'>
                <div style='font-size:.65rem;color:#4a5568;flex:1'>
                  {" · ".join(s.behavioral_evidence[:3]) if s.behavioral_evidence else s.archetype.replace("_"," ")}
                </div>
                {_accum_bar(accum)}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_scanner(universe: list[str]) -> None:
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        "<h2 style='color:#00d4ff;font-family:JetBrains Mono,monospace;"
        "font-size:1.3rem;letter-spacing:2px;margin-bottom:2px'>"
        "📡 MARKET SCANNER</h2>",
        unsafe_allow_html=True,
    )
    # Regime context pill
    try:
        from ui.regime_bar import get_regime
        _r = get_regime()
        _rc = {"BULL_TREND": "#00d4a0", "EXPANSION": "#00d4ff", "CHOPPY": "#f59e0b",
               "DISTRIBUTION": "#fb923c", "BEAR": "#ff4b4b"}.get(_r["regime"], "#8892a4")
        st.markdown(
            f"<span style='background:{_rc}18;border:1px solid {_rc}44;border-radius:6px;"
            f"padding:2px 10px;font-size:.65rem;color:{_rc};font-family:JetBrains Mono,monospace'>"
            f"{_r['emoji']} {_r['regime'].replace('_',' ')} · Score {_r['regime_score']:.0f} "
            f"· Setup ×{_r['quality_multiplier']:.2f}</span>",
            unsafe_allow_html=True,
        )
    except Exception:
        pass

    # ── Signal legend ─────────────────────────────────────────────────────────
    with st.expander("ℹ️ How signals are calculated", expanded=False):
        st.markdown(
            """
**Composite Score** = Momentum×40% + RSI×35% + Volume×25%

| Signal | Condition |
|--------|-----------|
| 🟢 **BUY ✓MTF** | Composite ≥ 65 AND RSI < 75 AND 1H aligned (price > 1H MA20, 1H RSI > 48) |
| 🟢 **BUY** | Composite ≥ 65 AND RSI < 75 (large-scan mode, 1H not checked) |
| 🟡 **WATCH** | Composite 45–64, or daily BUY but 1H not aligned |
| ⚪ **NEUTRAL** | Composite < 45 |

- **Momentum** — 5-day price change normalised (-5% to +10%)
- **RSI** — 14-day RSI, ideal zone 50–70
- **Volume** — today's volume vs 20-day avg (surge = bullish)
- **MTF (multi-timeframe)** — 1H confirmation fetched for BUY candidates only; skipped in Full NSE scan
- **Breakouts** — 52W high cross, Golden Cross (SMA50 > SMA200), Volume squeeze
            """,
            unsafe_allow_html=True,
        )

    # ── Controls ──────────────────────────────────────────────────────────────
    _c1, _c2, _c3, _c4, _c5 = st.columns([2, 1, 1, 1, 2])
    with _c1:
        scan_mode = st.radio(
            "Scan mode",
            ["My Universe", "Full NSE (~1900 stocks)"],
            horizontal=True,
            key="scanner_mode",
            label_visibility="collapsed",
        )
    with _c2:
        scan_quick = st.button(
            "⚡ Quick Scan", key="scanner_quick",
            type="primary", width="stretch",
        )
    with _c3:
        scan_full = st.button(
            "🌐 Full NSE Scan", key="scanner_full",
            width="stretch",
        )
    with _c4:
        mcap_filter = st.selectbox(
            "Market Cap",
            options=list(_MCAP_FILTER.keys()),
            index=0,
            key="scanner_mcap",
            label_visibility="collapsed",
            help="Filter by market cap: Largecap >₹20k Cr · Midcap ₹5k-20k Cr · Smallcap ₹500-5k Cr",
        )
    with _c5:
        if scan_mode == "Full NSE (~1900 stocks)":
            st.caption("⏱ Full scan: ~3-4 min · 16 workers · yfinance · results cached 5 min")
        else:
            st.caption(f"Scanning {len(universe)} stocks · Kite data · cached 5 min")

    # ── Resolve symbol list ───────────────────────────────────────────────────
    if scan_full or scan_mode == "Full NSE (~1900 stocks)":
        with st.spinner("Loading NSE symbol list…"):
            all_nse = _get_full_universe()
        symbols = all_nse if all_nse else universe
        if all_nse:
            st.caption(
                f"<span style='color:#00d4ff;font-size:.72rem'>✅ {len(symbols)} NSE symbols loaded</span>",
                unsafe_allow_html=True,
            )
    else:
        symbols = universe

    symbols_key = ",".join(symbols)

    if scan_quick or scan_full:
        st.cache_data.clear()

    if len(symbols) > 100:
        st.info(
            f"🔄 Scanning **{len(symbols)} stocks** in parallel (16 workers). "
            "This may take 3-4 minutes. Results appear when complete.",
            icon="⏱",
        )

    # ── Market cap filter note ────────────────────────────────────────────────
    if mcap_filter != "All":
        lo, hi = _MCAP_FILTER[mcap_filter]
        if hi > 90_000_000:
            cap_label = f"Largecap (>₹{lo:,} Cr)"
        else:
            cap_label = f"{mcap_filter} (₹{lo:,}–₹{hi:,} Cr)"
        st.caption(f"🔍 Filtering: **{cap_label}** — fetches market cap per stock (may slow down slightly)")

    # ── Two-column results ────────────────────────────────────────────────────
    col_mom, col_brk = st.columns(2)

    with col_mom:
        st.markdown(
            "<div style='background:rgba(0,212,160,.06);border:1px solid rgba(0,212,160,.25);"
            "border-radius:10px;padding:12px 16px;margin-bottom:8px'>"
            "<span style='color:#00d4a0;font-size:.75rem;font-weight:700;"
            "letter-spacing:2px'>🚀 TOP MOMENTUM</span>"
            "<span style='color:#4a5568;font-size:.65rem;margin-left:.5rem'>"
            "Score = Momentum 40% + RSI 35% + Volume 25%</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        with st.spinner(f"Scanning momentum across {len(symbols)} stocks…"):
            momentum_stocks = _cached_momentum(symbols_key, top_n=40)

        # ── Log BUY/WATCH signals to the feedback tracker ─────────────────
        try:
            from screener.signal_tracker import SignalTracker as _SignalTracker
            _tracker = _SignalTracker()
            for _ms in momentum_stocks:
                if _ms.signal in ("BUY", "WATCH"):
                    _tracker.log_signal(
                        symbol=_ms.symbol,
                        signal=_ms.signal,
                        price=_ms.price,
                        score=_ms.momentum_score,
                        rsi=_ms.rsi,
                        volume_ratio=_ms.volume_ratio,
                    )
        except Exception:
            pass  # never let tracker errors break the scanner UI

        if mcap_filter != "All":
            momentum_stocks = _filter_by_mcap(momentum_stocks, mcap_filter)

        momentum_stocks = momentum_stocks[:20]

        if not momentum_stocks:
            st.markdown(
                "<div style='text-align:center;padding:2rem;color:#4a5568;font-size:.82rem'>"
                "No momentum signals found for this filter.</div>",
                unsafe_allow_html=True,
            )
        else:
            _render_momentum_table(momentum_stocks)

        # ── Telegram alert check (lazy, only when configured) ────────────────
        try:
            from alerts.telegram_alerts import AlertEngine, AlertManager as _AM
            if AlertEngine().is_configured():
                _am = _AM()
                for _s in momentum_stocks:
                    try:
                        _am.check_and_fire(_s.symbol, _s.price, _s.rsi)
                    except Exception:
                        pass
        except Exception:
            pass

    with col_brk:
        st.markdown(
            "<div style='background:rgba(251,146,60,.06);border:1px solid rgba(251,146,60,.25);"
            "border-radius:10px;padding:12px 16px;margin-bottom:8px'>"
            "<span style='color:#f97316;font-size:.75rem;font-weight:700;"
            "letter-spacing:2px'>💥 BREAKOUTS</span>"
            "<span style='color:#4a5568;font-size:.65rem;margin-left:.5rem'>"
            "52W high · Golden cross · Vol squeeze</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        with st.spinner(f"Scanning breakouts across {len(symbols)} stocks…"):
            breakout_stocks = _cached_breakouts(symbols_key, top_n=40)

        if mcap_filter != "All":
            breakout_stocks = _filter_by_mcap(breakout_stocks, mcap_filter)

        breakout_stocks = breakout_stocks[:20]

        if not breakout_stocks:
            st.markdown(
                "<div style='text-align:center;padding:2rem;color:#4a5568;font-size:.82rem'>"
                "No breakout setups for this filter.</div>",
                unsafe_allow_html=True,
            )
        else:
            _render_breakout_table(breakout_stocks)

    # ── Elite Setups (full pipeline) ─────────────────────────────────────────
    st.markdown(
        "<div style='background:rgba(245,158,11,.06);border:1px solid rgba(245,158,11,.25);"
        "border-radius:10px;padding:12px 16px;margin:16px 0 8px'>"
        "<span style='color:#f59e0b;font-size:.75rem;font-weight:700;letter-spacing:2px'>"
        "⭐ ELITE SETUPS — VERDICT + TRADE PLAN</span>"
        "<span style='color:#4a5568;font-size:.65rem;margin-left:.5rem'>"
        "Volume accumulation · Multi-timeframe · Pre-calculated entry/stop/target/size</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Running quality pipeline…"):
        elite = _cached_elite_setups(symbols_key, top_n=20)

    if not elite:
        st.caption("No elite setups found — try Full NSE Scan mode or broaden the universe.")
    else:
        ready  = [s for s in elite if getattr(s, "verdict", "") == "READY_TO_TRADE"]
        build  = [s for s in elite if getattr(s, "verdict", "") == "BUILDING"]
        others = [s for s in elite if getattr(s, "verdict", "") not in ("READY_TO_TRADE", "BUILDING")]

        if ready:
            st.markdown(
                "<div style='font-size:.68rem;color:#00d4a0;font-weight:700;"
                "letter-spacing:1px;margin-bottom:4px'>✅ READY TO TRADE NOW</div>",
                unsafe_allow_html=True,
            )
            _render_elite_setups(ready)

        if build:
            st.markdown(
                "<div style='font-size:.68rem;color:#f59e0b;font-weight:700;"
                "letter-spacing:1px;margin:8px 0 4px'>🔨 BUILDING — WATCH THESE</div>",
                unsafe_allow_html=True,
            )
            _render_elite_setups(build[:8])

        if others:
            with st.expander(f"Show {len(others)} more setups"):
                _render_elite_setups(others)


def _render_momentum_table(stocks) -> None:
    st.markdown(
        "<div style='display:grid;grid-template-columns:90px 75px 60px 50px 55px 55px;"
        "gap:4px;padding:4px 8px;font-size:.6rem;color:#4a5568;font-weight:700;"
        "text-transform:uppercase;letter-spacing:.05em;"
        "border-bottom:1px solid rgba(255,255,255,.06);margin-bottom:4px'>"
        "<span>Stock</span><span>Price</span><span>Chg%</span>"
        "<span>RSI</span><span>Vol✕</span><span>Signal</span></div>",
        unsafe_allow_html=True,
    )
    for s in stocks:
        chg_color = "#00d4a0" if s.change_pct >= 0 else "#ff4b4b"
        sig_color = _SIGNAL_COLOR.get(s.signal, "#8892a4")
        bar_pct   = min(int(s.momentum_score), 100)
        mtf_confirmed = getattr(s, "mtf_confirmed", True)
        mtf_badge = (
            " <span style='font-size:.55rem;color:#00d4a0;opacity:.85'>✓MTF</span>"
            if s.signal == "BUY" and mtf_confirmed
            else ""
        )
        st.markdown(
            f"<div style='display:grid;grid-template-columns:90px 75px 60px 50px 55px 55px;"
            f"gap:4px;padding:5px 8px;font-size:.72rem;font-family:JetBrains Mono,monospace;"
            f"border-bottom:1px solid rgba(255,255,255,.03);align-items:center'>"
            f"<span style='color:#e8eaf0;font-weight:700'>{s.symbol}</span>"
            f"<span style='color:#c9d1e0'>₹{s.price:,.0f}</span>"
            f"<span style='color:{chg_color}'>{s.change_pct:+.1f}%</span>"
            f"<span style='color:#8892a4'>{s.rsi:.0f}</span>"
            f"<span style='color:#8892a4'>{s.volume_ratio:.1f}x</span>"
            f"<span style='color:{sig_color};font-weight:700;font-size:.65rem'>{s.signal}{mtf_badge}</span>"
            f"</div>"
            f"<div style='height:2px;margin:0 8px 3px;background:rgba(255,255,255,.04);border-radius:1px'>"
            f"<div style='height:2px;width:{bar_pct}%;background:{sig_color}88;border-radius:1px'></div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _render_breakout_table(stocks) -> None:
    st.markdown(
        "<div style='display:grid;grid-template-columns:90px 75px 130px 55px 60px;"
        "gap:4px;padding:4px 8px;font-size:.6rem;color:#4a5568;font-weight:700;"
        "text-transform:uppercase;letter-spacing:.05em;"
        "border-bottom:1px solid rgba(255,255,255,.06);margin-bottom:4px'>"
        "<span>Stock</span><span>Price</span><span>Pattern</span>"
        "<span>ATR</span><span>Conf%</span></div>",
        unsafe_allow_html=True,
    )
    for s in stocks:
        icon       = _BREAKOUT_ICONS.get(s.breakout_type, "⚡")
        label      = s.breakout_type.replace("_", " ")
        conf_color = (
            "#00d4a0" if s.confidence >= 70
            else ("#f59e0b" if s.confidence >= 55 else "#8892a4")
        )
        st.markdown(
            f"<div style='display:grid;grid-template-columns:90px 75px 130px 55px 60px;"
            f"gap:4px;padding:5px 8px;font-size:.72rem;font-family:JetBrains Mono,monospace;"
            f"border-bottom:1px solid rgba(255,255,255,.03);align-items:center'>"
            f"<span style='color:#e8eaf0;font-weight:700'>{s.symbol}</span>"
            f"<span style='color:#c9d1e0'>₹{s.price:,.0f}</span>"
            f"<span style='color:#f97316;font-size:.67rem'>{icon} {label}</span>"
            f"<span style='color:#8892a4'>{s.atr:.1f}</span>"
            f"<span style='color:{conf_color};font-weight:700'>{s.confidence:.0f}%</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
