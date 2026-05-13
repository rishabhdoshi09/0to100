"""
Institutional Terminal — 3-panel market operating system layout.

Left  (30%): Ranked Setup Queue — VCP + momentum setups ranked by regime-adjusted score
Center(40%): Interactive Chart + Pattern annotations
Right (30%): AI Intelligence Panel — playbook match, DeepSeek analysis, risk matrix
"""
from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _regime_adjusted_score(base_score: float, quality_multiplier: float) -> float:
    return min(100.0, round(base_score * quality_multiplier, 1))


def _score_color(score: float) -> str:
    if score >= 80: return "#00d4a0"
    if score >= 65: return "#f59e0b"
    if score >= 50: return "#fb923c"
    return "#8892a4"


def _category_badge(category: str) -> str:
    colors = {
        "Elite Setup":  ("#00d4a0", "#001a12"),
        "Strong Setup": ("#f59e0b", "#1a1200"),
        "Watchlist":    ("#8892a4", "#111827"),
        # short forms too
        "Elite":        ("#00d4a0", "#001a12"),
        "Strong":       ("#f59e0b", "#1a1200"),
    }
    fg, bg = colors.get(category, ("#8892a4", "#111827"))
    return (
        f"<span style='background:{bg};color:{fg};border:1px solid {fg}44;"
        f"border-radius:4px;padding:1px 6px;font-size:.58rem;font-weight:700;"
        f"letter-spacing:.05em'>{category}</span>"
    )


# ── Left Panel: Setup Queue ───────────────────────────────────────────────────

def _render_setup_queue(universe: list[str], regime: dict) -> None:
    st.markdown(
        "<div style='background:rgba(0,212,160,.04);border:1px solid rgba(0,212,160,.15);"
        "border-radius:10px;padding:10px 12px;margin-bottom:8px'>"
        "<span style='color:#00d4a0;font-size:.72rem;font-weight:700;letter-spacing:.1em'>"
        "⚡ SETUP QUEUE</span>"
        "<span style='color:#4a5568;font-size:.6rem;margin-left:.5rem'>"
        f"Regime ×{regime.get('quality_multiplier', 1.0):.2f}</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Filter controls
    min_score = st.slider(
        "Min Score", 40, 90, 60, step=5,
        key="iq_min_score", label_visibility="collapsed",
        help="Minimum regime-adjusted setup score"
    )
    show_only_stage2 = st.checkbox("Stage 2 only", value=True, key="iq_stage2")

    with st.spinner("Scanning setups…"):
        setups = _load_setups(universe, regime.get("regime_score", 50.0))

    if not setups:
        st.markdown(
            "<div style='text-align:center;padding:2rem;color:#4a5568;font-size:.8rem'>"
            "No setups found. Run a scan first or expand universe.</div>",
            unsafe_allow_html=True,
        )
        return

    # Filter
    filtered = [
        s for s in setups
        if s["adj_score"] >= min_score
        and (not show_only_stage2 or s.get("stage2", False))
    ]

    if not filtered:
        st.caption(f"No setups above score {min_score}. Lower the threshold.")
        filtered = setups[:10]

    # Render each setup card
    for s in filtered[:15]:
        sc   = s["adj_score"]
        col  = _score_color(sc)
        cat  = s.get("category", "Watchlist")
        badge = _category_badge(cat)

        clicked = st.button(
            f"{'⭐ ' if cat == 'Elite' else ''}{s['symbol']}  ₹{s['price']:,.0f}",
            key=f"iq_setup_{s['symbol']}",
            use_container_width=True,
        )
        if clicked:
            st.session_state["iq_selected"] = s["symbol"]
            st.session_state["iq_selected_data"] = s
            st.rerun()

        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"padding:2px 4px 6px;border-bottom:1px solid rgba(255,255,255,.04);margin-top:-6px'>"
            f"<div style='display:flex;gap:8px;align-items:center'>"
            f"{badge}"
            f"<span style='font-size:.6rem;color:#4a5568'>{s.get('base_type','—')}</span>"
            f"</div>"
            f"<div style='display:flex;gap:6px;align-items:center'>"
            f"<span style='font-size:.65rem;color:#8892a4'>R:R {s.get('rr','—')}</span>"
            f"<span style='font-size:.72rem;color:{col};font-weight:700'>{sc:.0f}</span>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.caption(f"Showing {len(filtered)} setups · {len(setups)} total scanned")


@st.cache_data(ttl=300, show_spinner=False)
def _load_setups(universe_key: str, _regime_score: float = 50.0) -> list[dict]:
    """Cache setup scan for 5 min. _regime_score used as cache differentiator only."""
    try:
        universe = [s.strip() for s in universe_key.split(",") if s.strip()]
        from screener.vcp_scanner import VCPScanner
        from ui.regime_bar import get_regime
        regime = get_regime()
        qm = regime.get("quality_multiplier", 1.0)

        scanner = VCPScanner(max_workers=12)
        results = scanner.scan(universe, top_n=50)

        out = []
        for r in results:
            adj = _regime_adjusted_score(r.setup_score, qm)
            out.append({
                "symbol":    r.symbol,
                "price":     r.price,
                "adj_score": adj,
                "base_score": r.setup_score,
                "category":  r.category,
                "base_type": r.base_type,
                "rr":        f"{r.risk_reward:.1f}x",
                "stop":      r.stop_loss,
                "pivot":     r.breakout_level,
                "stage2":    "Stage 2" in r.trend_stage,
                "atr_risk":  r.atr_risk,
                "contraction": r.vcp_contractions,
            })
        return sorted(out, key=lambda x: x["adj_score"], reverse=True)
    except Exception:
        return []


# ── Center Panel: Chart ────────────────────────────────────────────────────────

def _render_chart_panel(symbol: str, setup_data: dict | None) -> None:
    import plotly.graph_objects as go

    st.markdown(
        f"<div style='background:rgba(0,212,255,.04);border:1px solid rgba(0,212,255,.15);"
        f"border-radius:10px;padding:10px 12px;margin-bottom:8px'>"
        f"<span style='color:#00d4ff;font-size:.72rem;font-weight:700;letter-spacing:.1em'>"
        f"📊 {symbol}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        tf = st.radio(
            "Timeframe", ["Daily", "Weekly", "Hourly"],
            horizontal=True, key="iq_tf", label_visibility="collapsed"
        )
    with c2:
        show_sma = st.checkbox("SMA", value=True, key="iq_sma")

    df = _fetch_chart_data(symbol, tf)

    if df is None or df.empty:
        st.info(f"No data available for {symbol}.")
        return

    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name=symbol,
        increasing_line_color="#00d4a0",
        decreasing_line_color="#ff4b4b",
        increasing_fillcolor="#00d4a044",
        decreasing_fillcolor="#ff4b4b44",
    ))

    close = df["close"]

    if show_sma and len(close) >= 50:
        sma50 = close.rolling(50).mean()
        fig.add_trace(go.Scatter(
            x=df.index, y=sma50, name="SMA50",
            line=dict(color="#f59e0b", width=1.2, dash="dot"),
        ))
    if show_sma and len(close) >= 200:
        sma200 = close.rolling(200).mean()
        fig.add_trace(go.Scatter(
            x=df.index, y=sma200, name="SMA200",
            line=dict(color="#8892a4", width=1, dash="dash"),
        ))

    # Setup annotations from VCP scan data
    if setup_data:
        pivot = setup_data.get("pivot")
        stop  = setup_data.get("stop")
        if pivot and pivot > 0:
            fig.add_hline(
                y=pivot, line_color="#00d4ff", line_dash="dash", line_width=1.5,
                annotation_text=f"Pivot ₹{pivot:,.0f}",
                annotation_font_color="#00d4ff",
                annotation_position="right",
            )
        if stop and stop > 0:
            fig.add_hline(
                y=stop, line_color="#ff4b4b", line_dash="dot", line_width=1,
                annotation_text=f"Stop ₹{stop:,.0f}",
                annotation_font_color="#ff4b4b",
                annotation_position="right",
            )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        height=480,
        margin=dict(l=10, r=60, t=10, b=10),
        xaxis_rangeslider_visible=False,
        xaxis=dict(showgrid=False, color="#4a5568"),
        yaxis=dict(showgrid=True, gridcolor="#1e293b", color="#4a5568"),
        legend=dict(
            orientation="h", y=1.02, x=0,
            font=dict(size=10, color="#8892a4"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True, key=f"iq_chart_{symbol}_{tf}")

    # Volume sub-chart
    if "volume" in df.columns:
        vol_colors = ["#00d4a044" if df["close"].iloc[i] >= df["open"].iloc[i]
                      else "#ff4b4b44" for i in range(len(df))]
        vol_fig = go.Figure(go.Bar(
            x=df.index, y=df["volume"], marker_color=vol_colors, name="Volume"
        ))
        avg_vol = df["volume"].rolling(20).mean()
        vol_fig.add_trace(go.Scatter(
            x=df.index, y=avg_vol, name="Avg20",
            line=dict(color="#f59e0b", width=1, dash="dot")
        ))
        vol_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#0d1117",
            height=120,
            margin=dict(l=10, r=60, t=0, b=10),
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
        )
        st.plotly_chart(vol_fig, use_container_width=True, key=f"iq_vol_{symbol}_{tf}")


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_chart_data(symbol: str, timeframe: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf
        tf_map = {"Daily": ("200d", "1d"), "Weekly": ("2y", "1wk"), "Hourly": ("30d", "1h")}
        period, interval = tf_map.get(timeframe, ("200d", "1d"))
        df = yf.Ticker(f"{symbol}.NS").history(period=period, interval=interval)
        if df is None or df.empty:
            return None
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception:
        return None


# ── Right Panel: AI Intelligence ──────────────────────────────────────────────

def _render_ai_panel(symbol: str, setup_data: dict | None, regime: dict) -> None:
    st.markdown(
        "<div style='background:rgba(139,92,246,.06);border:1px solid rgba(139,92,246,.2);"
        "border-radius:10px;padding:10px 12px;margin-bottom:8px'>"
        "<span style='color:#a78bfa;font-size:.72rem;font-weight:700;letter-spacing:.1em'>"
        "🧠 AI INTELLIGENCE</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Playbook match
    _render_playbook_match(regime)

    st.divider()

    # Risk matrix
    if setup_data:
        _render_risk_matrix(symbol, setup_data)
        st.divider()

    # DeepSeek analysis
    _render_deepseek_analysis(symbol, setup_data, regime)


def _render_playbook_match(regime: dict) -> None:
    regime_name = regime.get("regime", "UNKNOWN")
    st.markdown(
        "<span style='font-size:.62rem;color:#8892a4;text-transform:uppercase;"
        "letter-spacing:.1em'>Regime-Aligned Playbooks</span>",
        unsafe_allow_html=True,
    )
    try:
        from analytics.playbook import get_regime_aligned_playbooks, enrich_with_live_stats
        playbooks = get_regime_aligned_playbooks(regime_name)
        playbooks = enrich_with_live_stats(playbooks)

        for pb in playbooks[:3]:
            live = pb.live_stats
            exp_str = (
                f"Live: {live['expectancy']:+.1f}% ({live['sample_size']} trades)"
                if live else
                f"Static: {pb.expectancy:+.1f}%"
            )
            wr_pct = (live["win_rate"] * 100 if live else pb.win_rate * 100)
            st.markdown(
                f"<div style='background:#111827;border:1px solid rgba(167,139,250,.15);"
                f"border-radius:8px;padding:8px 10px;margin-bottom:6px'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                f"<span style='color:#a78bfa;font-size:.72rem;font-weight:700'>{pb.emoji} {pb.name}</span>"
                f"<span style='font-size:.6rem;color:#4a5568'>{pb.category}</span>"
                f"</div>"
                f"<div style='display:flex;gap:12px;margin-top:4px;flex-wrap:wrap'>"
                f"<span style='font-size:.65rem;color:#8892a4'>Win Rate: "
                f"<span style='color:#00d4a0'>{wr_pct:.0f}%</span></span>"
                f"<span style='font-size:.65rem;color:#8892a4'>R:R: "
                f"<span style='color:#f59e0b'>{pb.risk_reward:.1f}x</span></span>"
                f"<span style='font-size:.65rem;color:#8892a4'>Exp: "
                f"<span style='color:#00d4ff'>{exp_str}</span></span>"
                f"</div>"
                f"<div style='font-size:.6rem;color:#4a5568;margin-top:3px'>"
                f"Entry: {pb.entry_rule[:60]}…</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    except Exception as exc:
        st.caption(f"Playbooks unavailable: {exc}")


def _render_risk_matrix(symbol: str, setup_data: dict) -> None:
    st.markdown(
        "<span style='font-size:.62rem;color:#8892a4;text-transform:uppercase;"
        "letter-spacing:.1em'>Risk Matrix</span>",
        unsafe_allow_html=True,
    )
    price  = setup_data.get("price", 0)
    pivot  = setup_data.get("pivot", price * 1.02)
    stop   = setup_data.get("stop", price * 0.95)
    rr_raw = setup_data.get("rr", "2.0x")

    if price > 0 and stop > 0 and pivot > 0:
        risk_pct  = (price - stop) / price * 100
        # Assume standard 2% portfolio risk
        # position size = 2% / risk_pct
        pos_size_pct = min(10.0, 2.0 / max(risk_pct, 0.1) * 100)

        st.markdown(
            f"<div style='background:#111827;border:1px solid rgba(255,75,75,.2);"
            f"border-radius:8px;padding:8px 10px'>"
            f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:6px'>"
            f"<div>"
            f"  <span style='font-size:.58rem;color:#4a5568'>Entry (Pivot)</span><br>"
            f"  <span style='font-size:.75rem;color:#00d4ff;font-weight:700'>₹{pivot:,.0f}</span>"
            f"</div>"
            f"<div>"
            f"  <span style='font-size:.58rem;color:#4a5568'>Stop Loss</span><br>"
            f"  <span style='font-size:.75rem;color:#ff4b4b;font-weight:700'>₹{stop:,.0f}</span>"
            f"</div>"
            f"<div>"
            f"  <span style='font-size:.58rem;color:#4a5568'>Risk per Share</span><br>"
            f"  <span style='font-size:.75rem;color:#f59e0b;font-weight:700'>{risk_pct:.1f}%</span>"
            f"</div>"
            f"<div>"
            f"  <span style='font-size:.58rem;color:#4a5568'>Pos Size (2% risk)</span><br>"
            f"  <span style='font-size:.75rem;color:#00d4a0;font-weight:700'>{pos_size_pct:.1f}%</span>"
            f"</div>"
            f"<div>"
            f"  <span style='font-size:.58rem;color:#4a5568'>Risk:Reward</span><br>"
            f"  <span style='font-size:.75rem;color:#a78bfa;font-weight:700'>{rr_raw}</span>"
            f"</div>"
            f"<div>"
            f"  <span style='font-size:.58rem;color:#4a5568'>VCP Contractions</span><br>"
            f"  <span style='font-size:.75rem;color:#00d4ff;font-weight:700'>"
            f"  {setup_data.get('contraction', '—')}</span>"
            f"</div>"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _render_deepseek_analysis(symbol: str, setup_data: dict | None, regime: dict) -> None:
    st.markdown(
        "<span style='font-size:.62rem;color:#8892a4;text-transform:uppercase;"
        "letter-spacing:.1em'>DeepSeek Analysis</span>",
        unsafe_allow_html=True,
    )

    if st.button("🧠 Analyse Setup", key=f"iq_analyse_{symbol}", use_container_width=True):
        with st.spinner("DeepSeek reasoning…"):
            analysis = _run_deepseek_analysis(symbol, setup_data, regime)
        st.session_state[f"iq_analysis_{symbol}"] = analysis

    analysis = st.session_state.get(f"iq_analysis_{symbol}")
    if analysis:
        st.markdown(
            f"<div style='background:#111827;border:1px solid rgba(167,139,250,.2);"
            f"border-radius:8px;padding:10px 12px;font-size:.72rem;color:#c9d1e0;"
            f"line-height:1.6;font-family:system-ui'>{analysis}</div>",
            unsafe_allow_html=True,
        )


def _run_deepseek_analysis(symbol: str, setup_data: dict | None, regime: dict) -> str:
    import os
    import requests

    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        return "⚠️ DeepSeek API key not configured."

    regime_str = (
        f"Market Regime: {regime.get('regime')} (Score: {regime.get('regime_score')})\n"
        f"Nifty: {regime.get('nifty_price')} ({regime.get('nifty_change_pct'):+.2f}%)\n"
        f"VIX: {regime.get('vix')} — {regime.get('vix_state')}\n"
        f"Breadth: {regime.get('breadth')}, Sector Leader: {regime.get('sector_leader')}"
    )

    setup_str = ""
    if setup_data:
        setup_str = (
            f"\nSetup Data:\n"
            f"  Base Type: {setup_data.get('base_type', '—')}\n"
            f"  Category: {setup_data.get('category', '—')}\n"
            f"  Regime-Adjusted Score: {setup_data.get('adj_score', '—')}\n"
            f"  Pivot: ₹{setup_data.get('pivot', '—')}\n"
            f"  Stop: ₹{setup_data.get('stop', '—')}\n"
            f"  Risk:Reward: {setup_data.get('rr', '—')}\n"
            f"  VCP Contractions: {setup_data.get('contraction', '—')}"
        )

    prompt = (
        f"Analyse the trading setup for {symbol} (NSE).\n\n"
        f"Context:\n{regime_str}{setup_str}\n\n"
        f"Provide a concise institutional-grade assessment covering:\n"
        f"1. Setup Quality (1-2 sentences)\n"
        f"2. Regime Alignment (is this setup favoured by current market conditions?)\n"
        f"3. Key Risk Factors (max 2 bullets)\n"
        f"4. Actionable Verdict: BUY / WAIT / SKIP with specific trigger condition\n\n"
        f"Be precise and probabilistic. No fluff. Max 200 words."
    )

    try:
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are an expert institutional equity trader specialising in NSE India. Be direct, probabilistic, and concise."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 400,
            },
            timeout=25,
        )
        data = resp.json()
        if "choices" not in data:
            return f"API error: {data.get('error', {}).get('message', 'Unknown')}"
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Analysis unavailable: {e}"


# ── Main render function ───────────────────────────────────────────────────────

def render_institutional_terminal(universe: list[str]) -> None:
    from ui.regime_bar import render_regime_bar, get_regime

    # Regime strip at top
    render_regime_bar()

    regime = get_regime()

    # ── Symbol selector (above the 3 columns) ────────────────────────────────
    selected = st.session_state.get("iq_selected", universe[0] if universe else "RELIANCE")

    header_col, sym_col = st.columns([3, 1])
    with header_col:
        st.markdown(
            "<h2 style='color:#00d4ff;font-family:JetBrains Mono,monospace;"
            "font-size:1.15rem;letter-spacing:2px;margin:0 0 4px'>"
            "🏛️ INSTITUTIONAL TERMINAL</h2>"
            "<span style='color:#4a5568;font-size:.7rem'>Setup Quality Engine · "
            "Risk-Expectancy Framework · Regime-Aware Ranking</span>",
            unsafe_allow_html=True,
        )
    with sym_col:
        manual = st.text_input(
            "Jump to symbol", value="", placeholder="e.g. INFY",
            key="iq_manual_sym", label_visibility="collapsed"
        )
        if manual.strip().upper():
            selected = manual.strip().upper()
            st.session_state["iq_selected"] = selected
            st.session_state["iq_selected_data"] = None

    setup_data = st.session_state.get("iq_selected_data")

    # ── 3-panel layout ───────────────────────────────────────────────────────
    col_left, col_center, col_right = st.columns([3, 4, 3])

    universe_key = ",".join(sorted(universe))

    with col_left:
        _render_setup_queue(universe_key, regime)

    with col_center:
        _render_chart_panel(selected, setup_data)

    with col_right:
        _render_ai_panel(selected, setup_data, regime)
