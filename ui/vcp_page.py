"""VCP + Base Formation Scanner UI.

Minervini VCP · Wyckoff Accumulation · Stage 2 Breakouts
"""
from __future__ import annotations

from typing import Optional

import streamlit as st

# ── Category colours / icons ──────────────────────────────────────────────────
_CAT_COLOR = {
    "Elite Setup":  "#00d4a0",
    "Strong Setup": "#f59e0b",
    "Watchlist":    "#60a5fa",
    "Avoid":        "#8892a4",
}
_CAT_BADGE = {
    "Elite Setup":  "#00d4a0",
    "Strong Setup": "#f59e0b",
    "Watchlist":    "#60a5fa",
    "Avoid":        "#4a5568",
}
_BASE_ICONS = {
    "VCP":       "🌀",
    "FLAT":      "📏",
    "CUP":       "☕",
    "ASCENDING": "📈",
    "DEEP":      "🕳",
    "NONE":      "◽",
}


# ── Cached scan function ──────────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def _run_vcp_scan(symbols_key: str, top_n: int) -> list:
    from screener.vcp_scanner import VCPScanner
    symbols = [s.strip() for s in symbols_key.split(",") if s.strip()]
    return VCPScanner().scan(symbols, top_n=top_n)


# ── Public entry point ────────────────────────────────────────────────────────

def render_vcp_page(universe: list[str]) -> None:
    # ── Regime Bar ────────────────────────────────────────────────────────────
    try:
        from ui.regime_bar import render_regime_bar
        render_regime_bar()
    except Exception:
        pass

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        "<h2 style='color:#00d4ff;font-family:JetBrains Mono,monospace;"
        "font-size:1.3rem;letter-spacing:2px;margin-bottom:2px'>"
        "🎯 VCP &amp; BASE FORMATION SCANNER</h2>"
        "<p style='color:#4a5568;font-size:.78rem;margin-top:0'>"
        "Minervini VCP · Wyckoff Accumulation · Stage 2 Breakouts</p>",
        unsafe_allow_html=True,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    with st.expander("ℹ️ How the VCP score is calculated", expanded=False):
        st.markdown(
            """
**Master Score** = Trend(20%) + Base(20%) + Volume Contraction(20%) + Volatility Contraction(15%) + RS(15%) + Breakout(10%)

| Category | Score |
|----------|-------|
| 🟢 **Elite Setup** | ≥ 75 |
| 🟡 **Strong Setup** | 60 – 74 |
| 🔵 **Watchlist** | 45 – 59 |
| ⚪ **Avoid** | < 45 |

- **VCP** — Volatility Contraction Pattern (Minervini): successive pullbacks that get progressively smaller
- **FLAT** — base depth < 12%, consolidation ≥ 4 weeks
- **CUP** — U-shaped base with 12–35% depth
- **ASCENDING** — each pullback holds higher than the last
- **RS** — Relative Strength vs Nifty50 (1M×40% + 3M×35% + 6M×25%)
            """,
            unsafe_allow_html=True,
        )

    # ── Controls ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns([2, 1.2, 1.5, 1.2, 1.5])

    with c1:
        scan_mode = st.radio(
            "Symbol list",
            ["My Universe", "Full NSE"],
            horizontal=True,
            key="vcp_scan_mode",
            label_visibility="collapsed",
        )

    with c2:
        min_score = st.slider(
            "Min score",
            min_value=40,
            max_value=100,
            value=55,
            step=5,
            key="vcp_min_score",
            label_visibility="visible",
        )

    with c3:
        base_type_filter = st.multiselect(
            "Base type",
            options=["VCP", "FLAT", "CUP", "ASCENDING"],
            default=["VCP", "FLAT", "CUP", "ASCENDING"],
            key="vcp_base_filter",
        )

    with c4:
        stage2_only = st.toggle("Stage 2 only", value=True, key="vcp_stage2")

    with c5:
        run_scan = st.button(
            "🔍 Run VCP Scan",
            type="primary",
            use_container_width=True,
            key="vcp_run",
        )

    # ── Resolve symbols ───────────────────────────────────────────────────────
    if scan_mode == "Full NSE":
        try:
            from screener.universe import StockUniverseFetcher
            with st.spinner("Loading NSE symbol list…"):
                all_nse = StockUniverseFetcher().get_all_symbols()
            symbols = all_nse if all_nse else universe
        except Exception:
            symbols = universe
    else:
        symbols = universe

    if run_scan:
        st.cache_data.clear()

    symbols_key = ",".join(symbols)

    # ── Scan ──────────────────────────────────────────────────────────────────
    if run_scan or st.session_state.get("vcp_results_loaded"):
        top_n = min(50, max(10, len(symbols) // 2))

        with st.spinner(f"Scanning {len(symbols)} stocks for VCP setups…"):
            raw_results = _run_vcp_scan(symbols_key, top_n=top_n)

        st.session_state["vcp_results_loaded"] = True
        st.session_state["vcp_raw_results"] = raw_results

    if "vcp_raw_results" not in st.session_state:
        st.info(
            "Click **🔍 Run VCP Scan** to find Minervini VCP and base formation breakout setups.",
            icon="🎯",
        )
        return

    raw_results = st.session_state["vcp_raw_results"]

    # ── Apply filters ─────────────────────────────────────────────────────────
    results = [r for r in raw_results if r.setup_score >= min_score]

    if base_type_filter:
        results = [r for r in results if r.base_type in base_type_filter]

    if stage2_only:
        results = [r for r in results if r.trend_stage == "Stage 2"]

    # ── Summary bar ───────────────────────────────────────────────────────────
    elite = [r for r in results if r.category == "Elite Setup"]
    strong = [r for r in results if r.category == "Strong Setup"]
    watchlist = [r for r in results if r.category == "Watchlist"]

    top_stock = results[0] if results else None
    top_info = f" | Top score: {top_stock.setup_score:.1f} ({top_stock.symbol})" if top_stock else ""

    st.markdown(
        f"<div style='background:rgba(0,212,255,.06);border:1px solid rgba(0,212,255,.2);"
        f"border-radius:8px;padding:10px 16px;margin-bottom:12px;font-size:.8rem;"
        f"font-family:JetBrains Mono,monospace;color:#c9d1e0'>"
        f"Found <span style='color:#00d4a0;font-weight:700'>{len(elite)} Elite</span>"
        f" + <span style='color:#f59e0b;font-weight:700'>{len(strong)} Strong</span>"
        f" + <span style='color:#60a5fa;font-weight:700'>{len(watchlist)} Watchlist</span>"
        f" setups across <b>{len(symbols)}</b> stocks{top_info}"
        f"</div>",
        unsafe_allow_html=True,
    )

    if not results:
        st.warning("No setups match your filters. Try lowering the min score or relaxing filters.")
        return

    # ── Category tabs ─────────────────────────────────────────────────────────
    tab_elite, tab_strong, tab_watch = st.tabs([
        f"🟢 Elite Setup ({len(elite)})",
        f"🟡 Strong Setup ({len(strong)})",
        f"🔵 Watchlist ({len(watchlist)})",
    ])

    with tab_elite:
        if elite:
            _render_setup_cards(elite)
        else:
            st.markdown(
                "<div style='text-align:center;padding:2rem;color:#4a5568;font-size:.82rem'>"
                "No Elite setups match current filters.</div>",
                unsafe_allow_html=True,
            )

    with tab_strong:
        if strong:
            _render_setup_cards(strong)
        else:
            st.markdown(
                "<div style='text-align:center;padding:2rem;color:#4a5568;font-size:.82rem'>"
                "No Strong setups match current filters.</div>",
                unsafe_allow_html=True,
            )

    with tab_watch:
        if watchlist:
            _render_setup_cards(watchlist)
        else:
            st.markdown(
                "<div style='text-align:center;padding:2rem;color:#4a5568;font-size:.82rem'>"
                "No Watchlist setups match current filters.</div>",
                unsafe_allow_html=True,
            )

    # ── Chart section ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<span style='color:#00d4ff;font-size:.9rem;font-weight:700;"
        "letter-spacing:1px'>📊 VCP SETUP CHART</span>",
        unsafe_allow_html=True,
    )

    chart_symbols = [r.symbol for r in results]
    selected_sym = st.selectbox(
        "Select symbol to chart",
        options=chart_symbols,
        key="vcp_chart_sym",
    )

    if selected_sym:
        setup = next((r for r in results if r.symbol == selected_sym), None)
        if setup:
            _render_vcp_chart(setup)

    # ── Telegram alert ────────────────────────────────────────────────────────
    try:
        from alerts.telegram_alerts import AlertEngine
        if AlertEngine().is_configured() and elite:
            if st.button(
                f"📤 Alert {len(elite)} Elite Setups to Telegram",
                key="vcp_tg_alert",
                type="secondary",
            ):
                _send_elite_alerts(elite)
    except Exception:
        pass


# ── Card renderer ─────────────────────────────────────────────────────────────

def _render_setup_cards(setups: list) -> None:
    for s in setups:
        cat_color = _CAT_COLOR.get(s.category, "#8892a4")
        badge_bg = _CAT_BADGE.get(s.category, "#4a5568")
        base_icon = _BASE_ICONS.get(s.base_type, "◽")

        # Contraction sequence string
        if s.contraction_sequence:
            seq_str = " → ".join(f"{v:.1f}%" for v in s.contraction_sequence[:5])
        else:
            seq_str = "—"

        # Score bars (unicode block)
        def _bar(val: float, width: int = 10) -> str:
            filled = int(round(val / 100 * width))
            return "█" * filled + "░" * (width - filled)

        pocket_badge = (
            " <span style='background:#f59e0b22;color:#f59e0b;"
            "border-radius:3px;padding:0 4px;font-size:.55rem'>POCKET</span>"
            if s.pocket_pivot else ""
        )
        tight_badge = (
            " <span style='background:#60a5fa22;color:#60a5fa;"
            "border-radius:3px;padding:0 4px;font-size:.55rem'>TIGHT</span>"
            if s.weekly_tight else ""
        )

        st.markdown(
            f"<div style='background:rgba(8,12,28,.7);border:1px solid {cat_color}33;"
            f"border-left:3px solid {cat_color};border-radius:10px;"
            f"padding:14px 18px;margin-bottom:10px;font-family:JetBrains Mono,monospace'>"

            # Row 1: symbol, category badge, score
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"margin-bottom:6px'>"
            f"<span style='color:#e8eaf0;font-size:1rem;font-weight:700'>{s.symbol}"
            f"{pocket_badge}{tight_badge}</span>"
            f"<span style='display:flex;gap:8px;align-items:center'>"
            f"<span style='background:{badge_bg}22;color:{badge_bg};"
            f"border:1px solid {badge_bg}44;border-radius:4px;"
            f"padding:2px 8px;font-size:.65rem;font-weight:700;letter-spacing:.05em'>"
            f"{s.category.upper()}</span>"
            f"<span style='color:{cat_color};font-size:1.1rem;font-weight:700'>"
            f"Score: {s.setup_score:.0f}</span>"
            f"</span></div>"

            # Row 2: price, stage, base type
            f"<div style='display:flex;gap:18px;margin-bottom:8px;font-size:.78rem'>"
            f"<span style='color:#c9d1e0'>₹{s.price:,.2f}</span>"
            f"<span style='color:{cat_color}'>{s.trend_stage}</span>"
            f"<span style='color:#a78bfa'>{base_icon} {s.base_type} Pattern</span>"
            f"<span style='color:#8892a4'>BP: {s.breakout_probability:.0%}</span>"
            f"</div>"

            # Row 3: base info + contractions
            f"<div style='font-size:.74rem;color:#8892a4;margin-bottom:8px'>"
            f"Base: <span style='color:#c9d1e0'>{s.base_depth_pct:.1f}%</span> deep"
            f" · <span style='color:#c9d1e0'>{s.base_duration_weeks}</span> weeks"
            f" · <span style='color:#c9d1e0'>{s.vcp_contractions}</span> contractions"
            f" [<span style='color:#f59e0b'>{seq_str}</span>]"
            f"</div>"

            # Row 4: levels
            f"<div style='font-size:.74rem;color:#8892a4;margin-bottom:10px'>"
            f"Breakout <span style='color:#00d4a0;font-weight:700'>@ ₹{s.breakout_level:,.2f}</span>"
            f" · Stop <span style='color:#ff4b4b'>₹{s.stop_loss:,.2f}</span>"
            f" · R:R <span style='color:#e8eaf0;font-weight:700'>{s.risk_reward:.1f}×</span>"
            f" · Pivot dist <span style='color:#c9d1e0'>{s.pivot_distance_pct:.1f}%</span>"
            f"</div>"

            # Row 5: score bars
            f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:4px 16px;font-size:.68rem'>"
            f"<div><span style='color:#4a5568'>Vol Contraction: </span>"
            f"<span style='color:#00d4a0;font-family:monospace'>{_bar(s.volume_contraction_score)}</span>"
            f" <span style='color:#c9d1e0'>{s.volume_contraction_score:.0f}</span></div>"
            f"<div><span style='color:#4a5568'>Accumulation: </span>"
            f"<span style='color:#a78bfa;font-family:monospace'>{_bar(s.accumulation_score)}</span>"
            f" <span style='color:#c9d1e0'>{s.accumulation_score:.0f}</span></div>"
            f"<div><span style='color:#4a5568'>Vola Contract: </span>"
            f"<span style='color:#f59e0b;font-family:monospace'>{_bar(s.volatility_contraction_score)}</span>"
            f" <span style='color:#c9d1e0'>{s.volatility_contraction_score:.0f}</span></div>"
            f"<div><span style='color:#4a5568'>RS vs Nifty: </span>"
            f"<span style='color:#60a5fa;font-family:monospace'>{_bar(s.rs_score)}</span>"
            f" <span style='color:#c9d1e0'>{s.rs_score:.0f}</span></div>"
            f"</div>"

            f"</div>",
            unsafe_allow_html=True,
        )


# ── Chart renderer ────────────────────────────────────────────────────────────

def _render_vcp_chart(setup) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import yfinance as yf
        import numpy as np
        import pandas as pd

        ticker = yf.Ticker(f"{setup.symbol}.NS")
        df = ticker.history(period="6mo", interval="1d")
        if df is None or df.empty or len(df) < 20:
            st.warning(f"No chart data for {setup.symbol}")
            return

        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        df = df.reset_index()
        df = df.tail(120)

        close = df["close"].values
        dates = df.iloc[:, 0]  # date column

        # MAs
        sma50 = pd.Series(close).rolling(50).mean().values
        sma200 = pd.Series(close).rolling(200).mean().values

        # Volume colours
        vol_colors = [
            "#00d4a0" if close[i] >= (close[i - 1] if i > 0 else close[i]) else "#ff4b4b"
            for i in range(len(close))
        ]

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.75, 0.25],
        )

        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=dates,
                open=df["open"] if "open" in df.columns else df["close"],
                high=df["high"] if "high" in df.columns else df["close"],
                low=df["low"] if "low" in df.columns else df["close"],
                close=df["close"],
                name=setup.symbol,
                increasing_line_color="#00d4a0",
                decreasing_line_color="#ff4b4b",
                increasing_fillcolor="#00d4a022",
                decreasing_fillcolor="#ff4b4b22",
            ),
            row=1, col=1,
        )

        # SMA50
        valid_sma50 = ~np.isnan(sma50)
        if valid_sma50.any():
            fig.add_trace(
                go.Scatter(
                    x=dates[valid_sma50],
                    y=sma50[valid_sma50],
                    mode="lines",
                    name="SMA50",
                    line=dict(color="#3b82f6", width=1.5),
                ),
                row=1, col=1,
            )

        # SMA200
        valid_sma200 = ~np.isnan(sma200)
        if valid_sma200.any():
            fig.add_trace(
                go.Scatter(
                    x=dates[valid_sma200],
                    y=sma200[valid_sma200],
                    mode="lines",
                    name="SMA200",
                    line=dict(color="#f97316", width=1.5),
                ),
                row=1, col=1,
            )

        # Base zone shaded rectangle
        x_start = dates.iloc[max(0, len(dates) - setup.base_duration_weeks * 5)]
        x_end = dates.iloc[-1]
        base_low_val = setup.stop_loss
        base_high_val = setup.breakout_level

        fig.add_shape(
            type="rect",
            x0=x_start, x1=x_end,
            y0=base_low_val, y1=base_high_val,
            fillcolor="rgba(0,212,160,0.05)",
            line=dict(color="rgba(0,212,160,0.2)", width=1, dash="dot"),
            row=1, col=1,
        )

        # Breakout level (pivot)
        fig.add_hline(
            y=setup.breakout_level,
            line=dict(color="#00d4a0", width=1.5, dash="dash"),
            annotation_text=f"Pivot ₹{setup.breakout_level:,.0f}",
            annotation_position="right",
            annotation_font=dict(color="#00d4a0", size=10),
            row=1, col=1,
        )

        # Stop loss
        fig.add_hline(
            y=setup.stop_loss,
            line=dict(color="#ff4b4b", width=1.5, dash="dash"),
            annotation_text=f"Stop ₹{setup.stop_loss:,.0f}",
            annotation_position="right",
            annotation_font=dict(color="#ff4b4b", size=10),
            row=1, col=1,
        )

        # Contraction annotations
        if setup.contraction_sequence and len(dates) > 5:
            spacing = max(1, len(dates) // (len(setup.contraction_sequence) + 1))
            for idx, depth in enumerate(setup.contraction_sequence[:4]):
                ann_idx = min(spacing * (idx + 1), len(dates) - 1)
                ann_y = float(close[ann_idx]) if ann_idx < len(close) else float(close[-1])
                fig.add_annotation(
                    x=dates.iloc[ann_idx],
                    y=ann_y,
                    text=f"C{idx+1}: {depth:.1f}%",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="#f59e0b",
                    font=dict(color="#f59e0b", size=9),
                    ax=0, ay=-30,
                    row=1, col=1,
                )

        # Volume bars
        if "volume" in df.columns:
            fig.add_trace(
                go.Bar(
                    x=dates,
                    y=df["volume"],
                    name="Volume",
                    marker_color=vol_colors,
                    opacity=0.7,
                ),
                row=2, col=1,
            )
            # Volume MA20
            vol_ma20 = pd.Series(df["volume"].values).rolling(20).mean().values
            valid_vol_ma = ~np.isnan(vol_ma20)
            if valid_vol_ma.any():
                fig.add_trace(
                    go.Scatter(
                        x=dates[valid_vol_ma],
                        y=vol_ma20[valid_vol_ma],
                        mode="lines",
                        name="Vol MA20",
                        line=dict(color="#8892a4", width=1, dash="dot"),
                    ),
                    row=2, col=1,
                )

        cat_color = _CAT_COLOR.get(setup.category, "#00d4ff")
        fig.update_layout(
            title=dict(
                text=(
                    f"{setup.symbol} — {setup.category} | "
                    f"{setup.base_type} | Score {setup.setup_score:.0f} | "
                    f"{setup.trend_stage}"
                ),
                font=dict(color=cat_color, size=14, family="JetBrains Mono"),
            ),
            height=600,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(8,12,28,0.6)",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(
                font=dict(color="#8892a4", size=10),
                bgcolor="rgba(0,0,0,0)",
            ),
            margin=dict(l=50, r=100, t=50, b=20),
        )

        fig.update_xaxes(
            gridcolor="rgba(255,255,255,0.05)",
            tickfont=dict(color="#8892a4", size=9),
            showgrid=True,
        )
        fig.update_yaxes(
            gridcolor="rgba(255,255,255,0.05)",
            tickfont=dict(color="#8892a4", size=9),
            showgrid=True,
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as exc:
        st.warning(f"Chart error for {setup.symbol}: {exc}")


# ── Telegram helpers ──────────────────────────────────────────────────────────

def _send_elite_alerts(setups: list) -> None:
    try:
        from alerts.telegram_alerts import AlertEngine
        engine = AlertEngine()
        sent = 0
        for s in setups:
            try:
                ok = engine.send_signal_alert(
                    symbol=s.symbol,
                    signal="BUY",
                    price=s.price,
                    score=s.setup_score,
                    rsi=50.0,
                    vol_ratio=s.volume_contraction_score / 50,
                )
                if ok:
                    sent += 1
            except Exception:
                pass
        st.success(f"Sent {sent}/{len(setups)} Elite VCP alerts to Telegram.")
    except Exception as exc:
        st.error(f"Telegram alert failed: {exc}")
