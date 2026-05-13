"""
Institutional Terminal — State-aware market operating system.

3-panel layout:
  Left  (28%): Ranked Setup Queue — 8-stage pipeline output
  Center(44%): Chart workspace with annotations
  Right (28%): AI Intelligence Panel — institutional summary, EV, playbook

ALL outputs are regime-aware. No indicator-first logic.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np


# ── Color constants ────────────────────────────────────────────────────────────
_TIER_COLORS = {
    "ELITE_A_PLUS": ("#00d4a0", "#001a12"),
    "A":            ("#f59e0b", "#1a1200"),
    "B":            ("#60a5fa", "#001020"),
    "WATCHLIST":    ("#8892a4", "#111827"),
    "AVOID":        ("#4a5568", "#0d1117"),
}
_ARCHETYPE_EMOJI = {
    "VCP_BREAKOUT":           "🌀",
    "MOMENTUM_EXPANSION":     "🚀",
    "EARLY_LEADER":           "⭐",
    "ACCUMULATION_BREAKOUT":  "📦",
    "EARNINGS_CONTINUATION":  "📊",
    "FAILED_BREAKOUT":        "🔴",
    "MEAN_REVERSION":         "🔄",
    "TREND_CONTINUATION":     "📈",
    "HIGH_TIGHT_FLAG":        "🏴",
}
_REGIME_COLORS = {
    "TRENDING_BULL": "#00d4a0", "EXPANSION": "#00d4ff",
    "CHOPPY": "#f59e0b", "COMPRESSION": "#60a5fa",
    "DISTRIBUTION": "#fb923c", "TRENDING_BEAR": "#ff4b4b",
}


# ── Cached scan ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _run_pipeline(universe_key: str, top_n: int = 30) -> list[dict]:
    """Run 8-stage scan pipeline. Returns serialisable dicts (not dataclasses)."""
    try:
        universe = [s.strip() for s in universe_key.split(",") if s.strip()]
        from core.regime_engine import compute_regime
        regime = compute_regime()

        from scan.pipeline import ScanPipeline
        setups = ScanPipeline(max_workers=12, min_quality_score=40.0).run(
            universe, top_n=top_n, regime_state=regime,
            skip_liquidity_filter=len(universe) <= 30,
        )
        # Serialise to dicts for cache compatibility
        return [
            {
                "symbol":      s.symbol,
                "archetype":   s.archetype,
                "playbook_id": s.playbook_id,
                "price":       s.price,
                "pivot":       s.pivot_level,
                "stop":        s.stop_level,
                "risk_pct":    s.risk_pct,
                "tier":        s.quality_tier,
                "score":       s.quality_score,
                "adj_score":   s.regime_adjusted_score,
                "ev_r":        s.expected_value_r,
                "win_rate":    s.historical_win_rate,
                "reg_align":   s.regime_alignment,
                "fail_risk":   s.failure_risk,
                "evidence":    s.behavioral_evidence,
                "regime":      s.regime,
                "sector_leader": s.sector_leader,
                "pos_pct":     s.suggested_position_pct,
                "summary":     s.institutional_summary,
            }
            for s in setups
        ]
    except Exception as exc:
        st.session_state["pipeline_error"] = str(exc)
        return []


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_chart(symbol: str, timeframe: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf
        tf_map = {"Daily": ("250d", "1d"), "Weekly": ("3y", "1wk"), "Hourly": ("30d", "1h")}
        period, interval = tf_map.get(timeframe, ("250d", "1d"))
        df = yf.Ticker(f"{symbol}.NS").history(period=period, interval=interval)
        if df is None or df.empty:
            return None
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception:
        return None


# ── Left Panel: Setup Queue ────────────────────────────────────────────────────

def _render_setup_queue(universe_key: str, regime: dict) -> None:
    qm  = regime.get("quality_mult", 1.0)
    reg = regime.get("market", "UNKNOWN")
    rc  = _REGIME_COLORS.get(reg, "#8892a4")

    st.markdown(
        f"<div style='background:#0d1117;border:1px solid {rc}33;"
        f"border-radius:10px;padding:8px 12px;margin-bottom:8px'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center'>"
        f"<span style='color:{rc};font-size:.68rem;font-weight:700;letter-spacing:.1em'>"
        f"⚡ SETUP QUEUE</span>"
        f"<span style='font-size:.6rem;color:#4a5568'>×{qm:.2f} · {reg.replace('_',' ')}</span>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # Controls
    c1, c2 = st.columns([2, 1])
    with c1:
        min_score = st.slider("Min score", 40, 90, 60, 5, key="iq2_min", label_visibility="collapsed")
    with c2:
        only_elite = st.toggle("Elite only", key="iq2_elite")

    if st.button("🔄 Re-scan", key="iq2_rescan", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    with st.spinner("Running 8-stage pipeline…"):
        setups = _run_pipeline(universe_key, top_n=30)

    if err := st.session_state.pop("pipeline_error", None):
        st.warning(f"Pipeline error: {err}")

    filtered = [
        s for s in setups
        if s["adj_score"] >= min_score
        and (not only_elite or s["tier"] == "ELITE_A_PLUS")
    ]

    if not filtered:
        st.markdown(
            "<div style='text-align:center;padding:2rem;color:#4a5568;font-size:.78rem'>"
            "No setups match current filters.</div>",
            unsafe_allow_html=True,
        )
        return

    for s in filtered[:20]:
        fg, bg = _TIER_COLORS.get(s["tier"], ("#8892a4", "#111827"))
        emoji  = _ARCHETYPE_EMOJI.get(s["archetype"], "◆")
        ev_str = f"+{s['ev_r']:.1f}R" if s['ev_r'] >= 0 else f"{s['ev_r']:.1f}R"
        ev_col = "#00d4a0" if s['ev_r'] > 0 else "#ff4b4b"
        sl_badge = " 🏆" if s.get("sector_leader") else ""

        selected = st.session_state.get("iq2_selected") == s["symbol"]
        btn_style = f"border: 1px solid {fg}66;" if selected else ""

        if st.button(
            f"{emoji} {s['symbol']}{sl_badge}  ₹{s['price']:,.0f}",
            key=f"iq2_{s['symbol']}",
            use_container_width=True,
        ):
            st.session_state["iq2_selected"]   = s["symbol"]
            st.session_state["iq2_setup_data"] = s
            st.rerun()

        st.markdown(
            f"<div style='background:{bg};border:1px solid {fg}22;"
            f"border-radius:0 0 6px 6px;padding:4px 8px 6px;margin-top:-8px;"
            f"display:flex;justify-content:space-between;margin-bottom:4px'>"
            f"<div style='display:flex;gap:8px;align-items:center'>"
            f"<span style='background:{fg}22;color:{fg};font-size:.55rem;"
            f"padding:1px 6px;border-radius:4px;font-weight:700'>"
            f"{s['tier'].replace('_',' ')}</span>"
            f"<span style='font-size:.6rem;color:#4a5568'>"
            f"{s['archetype'].replace('_',' ')[:14]}</span>"
            f"</div>"
            f"<div style='display:flex;gap:8px;align-items:center'>"
            f"<span style='font-size:.65rem;color:{ev_col};font-weight:700'>{ev_str}</span>"
            f"<span style='font-size:.65rem;color:{fg};font-weight:700'>{s['adj_score']:.0f}</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    st.caption(f"{len(filtered)} setups · {len(setups)} scanned")


# ── Center Panel: Chart ────────────────────────────────────────────────────────

def _render_chart_panel(symbol: str, setup: dict | None) -> None:
    import plotly.graph_objects as go

    reg_col = _REGIME_COLORS.get(setup.get("regime", ""), "#00d4ff") if setup else "#00d4ff"

    st.markdown(
        f"<div style='background:#0d1117;border:1px solid {reg_col}22;"
        f"border-radius:10px;padding:8px 12px;margin-bottom:8px;display:flex;"
        f"justify-content:space-between;align-items:center'>"
        f"<span style='color:{reg_col};font-size:.72rem;font-weight:700;letter-spacing:.08em'>"
        f"📊 {symbol}</span>"
        + (f"<span style='font-size:.6rem;color:#4a5568'>"
           f"{setup['archetype'].replace('_',' ')} · {setup['tier'].replace('_',' ')}</span>"
           if setup else "") +
        "</div>",
        unsafe_allow_html=True,
    )

    tf_col, overlay_col = st.columns([2, 2])
    with tf_col:
        tf = st.radio("TF", ["Daily", "Weekly", "Hourly"], horizontal=True,
                      key="iq2_tf", label_visibility="collapsed")
    with overlay_col:
        show_sma = st.checkbox("SMA50/200", value=True, key="iq2_sma")
        show_ema = st.checkbox("EMA10/21", value=False, key="iq2_ema")

    df = _fetch_chart(symbol, tf)
    if df is None or df.empty:
        st.info(f"No chart data for {symbol}")
        return

    close  = df["close"]
    n      = len(df)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=close,
        name=symbol,
        increasing_line_color="#00d4a0", decreasing_line_color="#ff4b4b",
        increasing_fillcolor="rgba(0,212,160,.25)", decreasing_fillcolor="rgba(255,75,75,.20)",
        line=dict(width=1),
    ))

    if show_sma and n >= 50:
        sma50 = close.rolling(50).mean()
        fig.add_trace(go.Scatter(x=df.index, y=sma50, name="SMA50",
                                  line=dict(color="#f59e0b", width=1.2, dash="dot")))
    if show_sma and n >= 200:
        sma200 = close.rolling(200).mean()
        fig.add_trace(go.Scatter(x=df.index, y=sma200, name="SMA200",
                                  line=dict(color="#8892a4", width=1, dash="dash")))
    if show_ema and n >= 21:
        ema10 = close.ewm(span=10).mean()
        ema21 = close.ewm(span=21).mean()
        fig.add_trace(go.Scatter(x=df.index, y=ema10, name="EMA10",
                                  line=dict(color="#a78bfa", width=1)))
        fig.add_trace(go.Scatter(x=df.index, y=ema21, name="EMA21",
                                  line=dict(color="#60a5fa", width=1)))

    # Setup annotations
    if setup:
        pivot = setup.get("pivot", 0)
        stop  = setup.get("stop", 0)
        if pivot > 0:
            fig.add_hline(y=pivot, line_color="#00d4ff", line_dash="dash", line_width=1.5,
                          annotation_text=f"Pivot ₹{pivot:,.0f}", annotation_font_color="#00d4ff",
                          annotation_position="right")
        if stop > 0:
            fig.add_hline(y=stop, line_color="#ff4b4b", line_dash="dot", line_width=1,
                          annotation_text=f"Stop ₹{stop:,.0f}", annotation_font_color="#ff4b4b",
                          annotation_position="right")
            # Shade risk zone
            if pivot > stop > 0:
                fig.add_hrect(y0=stop, y1=pivot, fillcolor="rgba(255,75,75,.05)",
                              line_width=0)

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=440, margin=dict(l=10, r=70, t=5, b=5),
        xaxis_rangeslider_visible=False,
        xaxis=dict(showgrid=False, color="#4a5568", showspikes=True),
        yaxis=dict(showgrid=True, gridcolor="#1e293b", color="#4a5568", showspikes=True),
        legend=dict(orientation="h", y=1.01, x=0, font=dict(size=9, color="#8892a4"),
                    bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, key=f"iq2_c_{symbol}_{tf}")

    # Volume bar
    if "volume" in df.columns:
        vols  = df["volume"]
        vcols = ["rgba(0,212,160,.35)" if close.iloc[i] >= df["open"].iloc[i]
                 else "rgba(255,75,75,.3)" for i in range(n)]
        vfig  = go.Figure(go.Bar(x=df.index, y=vols, marker_color=vcols, name="Vol"))
        avg20 = vols.rolling(20).mean()
        vfig.add_trace(go.Scatter(x=df.index, y=avg20, line=dict(color="#f59e0b", width=1)))
        vfig.update_layout(
            template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            height=100, margin=dict(l=10, r=70, t=0, b=5), showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
        )
        st.plotly_chart(vfig, use_container_width=True, key=f"iq2_v_{symbol}_{tf}")


# ── Right Panel: Intelligence ──────────────────────────────────────────────────

def _render_intelligence_panel(symbol: str, setup: dict | None, regime: dict) -> None:
    st.markdown(
        "<div style='background:#0d1117;border:1px solid rgba(167,139,250,.2);"
        "border-radius:10px;padding:8px 12px;margin-bottom:8px'>"
        "<span style='color:#a78bfa;font-size:.68rem;font-weight:700;letter-spacing:.1em'>"
        "🧠 INTELLIGENCE PANEL</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    if setup:
        # Institutional summary (pre-computed by ranking engine)
        _render_institutional_summary(setup)
        st.divider()
        # Risk matrix
        _render_risk_matrix(setup)
        st.divider()

    # Playbook match
    _render_regime_playbooks(regime)
    st.divider()

    # DeepSeek analysis (on demand)
    _render_ai_analysis(symbol, setup, regime)


def _render_institutional_summary(setup: dict) -> None:
    tier  = setup["tier"]
    fg, _ = _TIER_COLORS.get(tier, ("#8892a4", "#111827"))
    emoji = _ARCHETYPE_EMOJI.get(setup["archetype"], "◆")

    summary = setup.get("summary", "")
    # Parse into sections for display
    sections = {}
    current  = None
    for line in summary.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.isupper() and line.endswith(":") or line in (
            "SETUP:", "QUALITY:", "WHY IT MATTERS:", "RISK:", "EXPECTANCY:", "REGIME:"
        ):
            current = line.rstrip(":")
            sections[current] = []
        elif current:
            sections.setdefault(current, []).append(line)

    ev_str = f"+{setup['ev_r']:.1f}R" if setup['ev_r'] >= 0 else f"{setup['ev_r']:.1f}R"
    ev_col = "#00d4a0" if setup['ev_r'] > 0 else "#ff4b4b"
    wr_pct = setup['win_rate'] * 100

    st.markdown(
        f"<div style='background:#111827;border:1px solid {fg}33;"
        f"border-radius:8px;padding:10px 12px;font-family:JetBrains Mono,monospace'>"

        f"<div style='display:flex;justify-content:space-between;margin-bottom:6px'>"
        f"<span style='color:{fg};font-size:.78rem;font-weight:800'>"
        f"{emoji} {setup['archetype'].replace('_',' ')}</span>"
        f"<span style='background:{fg}22;color:{fg};font-size:.62rem;"
        f"padding:2px 8px;border-radius:4px;font-weight:700'>"
        f"{tier.replace('_',' ')}</span>"
        f"</div>"

        f"<div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin:6px 0'>"
        f"<div style='background:#0d1117;border-radius:6px;padding:5px 8px'>"
        f"<div style='font-size:.55rem;color:#4a5568'>EXP VALUE</div>"
        f"<div style='font-size:.82rem;color:{ev_col};font-weight:800'>{ev_str}</div>"
        f"</div>"
        f"<div style='background:#0d1117;border-radius:6px;padding:5px 8px'>"
        f"<div style='font-size:.55rem;color:#4a5568'>WIN RATE</div>"
        f"<div style='font-size:.82rem;color:#f59e0b;font-weight:800'>{wr_pct:.0f}%</div>"
        f"</div>"
        f"<div style='background:#0d1117;border-radius:6px;padding:5px 8px'>"
        f"<div style='font-size:.55rem;color:#4a5568'>REG ALIGN</div>"
        f"<div style='font-size:.82rem;color:#a78bfa;font-weight:800'>{setup['reg_align']}</div>"
        f"</div>"
        f"</div>"

        f"<div style='font-size:.62rem;color:#4a5568;margin:4px 0 2px;text-transform:uppercase;"
        f"letter-spacing:.06em'>Evidence</div>"
        + "".join(f"<div style='font-size:.67rem;color:#c9d1e0;padding:1px 0'>• {e}</div>"
                  for e in setup.get("evidence", [])[:4]) +

        f"<div style='display:flex;justify-content:space-between;margin-top:6px'>"
        f"<span style='font-size:.62rem;color:#4a5568'>Failure Risk: "
        f"<span style='color:{'#ff4b4b' if setup['fail_risk']=='HIGH' else '#f59e0b' if setup['fail_risk']=='MODERATE' else '#00d4a0'}'>"
        f"{setup['fail_risk']}</span></span>"
        f"<span style='font-size:.62rem;color:#4a5568'>Score: "
        f"<span style='color:{fg}'>{setup['score']:.0f}/100</span></span>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_risk_matrix(setup: dict) -> None:
    st.markdown(
        "<span style='font-size:.6rem;color:#8892a4;text-transform:uppercase;"
        "letter-spacing:.08em'>RISK MATRIX</span>",
        unsafe_allow_html=True,
    )
    price = setup.get("price", 0)
    pivot = setup.get("pivot", price * 1.02)
    stop  = setup.get("stop", price * 0.95)
    risk  = setup.get("risk_pct", 5.0)
    pos   = setup.get("pos_pct", 2.0)

    target_1r = pivot * (1 + risk / 100)
    target_2r = pivot * (1 + risk * 2 / 100)

    st.markdown(
        f"<div style='background:#111827;border:1px solid rgba(255,75,75,.15);"
        f"border-radius:8px;padding:8px 12px;font-family:JetBrains Mono,monospace'>"
        f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:5px'>"
        f"<div><span style='font-size:.55rem;color:#4a5568'>Pivot Entry</span><br>"
        f"<span style='font-size:.75rem;color:#00d4ff;font-weight:700'>₹{pivot:,.0f}</span></div>"
        f"<div><span style='font-size:.55rem;color:#4a5568'>Stop Loss</span><br>"
        f"<span style='font-size:.75rem;color:#ff4b4b;font-weight:700'>₹{stop:,.0f}</span></div>"
        f"<div><span style='font-size:.55rem;color:#4a5568'>1R Target</span><br>"
        f"<span style='font-size:.75rem;color:#f59e0b;font-weight:700'>₹{target_1r:,.0f}</span></div>"
        f"<div><span style='font-size:.55rem;color:#4a5568'>2R Target</span><br>"
        f"<span style='font-size:.75rem;color:#00d4a0;font-weight:700'>₹{target_2r:,.0f}</span></div>"
        f"<div><span style='font-size:.55rem;color:#4a5568'>Risk %</span><br>"
        f"<span style='font-size:.75rem;color:#fb923c;font-weight:700'>{risk:.1f}%</span></div>"
        f"<div><span style='font-size:.55rem;color:#4a5568'>Position Size</span><br>"
        f"<span style='font-size:.75rem;color:#a78bfa;font-weight:700'>{pos:.1f}%</span></div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )


def _render_regime_playbooks(regime: dict) -> None:
    st.markdown(
        "<span style='font-size:.6rem;color:#8892a4;text-transform:uppercase;"
        "letter-spacing:.08em'>TODAY'S PLAYBOOKS</span>",
        unsafe_allow_html=True,
    )
    try:
        from playbooks import get_playbooks_for_regime
        pbs = get_playbooks_for_regime(
            regime.get("market", "CHOPPY"),
            regime.get("volatility", "NORMAL"),
            regime.get("breadth", "NEUTRAL"),
        )[:3]
        for pb in pbs:
            ev_str = f"{pb.baseline_expectancy*100:+.1f}%"
            wr     = f"{pb.baseline_win_rate*100:.0f}%"
            aligned_dot = "🟢" if pb.regime_aligned else "🟡"
            st.markdown(
                f"<div style='background:#111827;border:1px solid rgba(167,139,250,.12);"
                f"border-radius:6px;padding:6px 10px;margin-bottom:4px'>"
                f"<div style='display:flex;justify-content:space-between'>"
                f"<span style='color:#a78bfa;font-size:.7rem;font-weight:700'>"
                f"{aligned_dot} {pb.emoji} {pb.name}</span>"
                f"<span style='font-size:.58rem;color:#4a5568'>{pb.category[:10]}</span>"
                f"</div>"
                f"<div style='font-size:.62rem;color:#8892a4;margin-top:2px'>"
                f"EV {ev_str} · WR {wr} · R:R {pb.baseline_risk_reward:.1f}×"
                f"</div></div>",
                unsafe_allow_html=True,
            )
    except Exception:
        st.caption("Playbook data unavailable")


def _render_ai_analysis(symbol: str, setup: dict | None, regime: dict) -> None:
    st.markdown(
        "<span style='font-size:.6rem;color:#a78bfa;text-transform:uppercase;"
        "letter-spacing:.08em'>DEEPSEEK ANALYSIS</span>",
        unsafe_allow_html=True,
    )

    cache_key = f"iq2_analysis_{symbol}"

    if st.button("🧠 Analyse", key=f"iq2_btn_{symbol}", use_container_width=True):
        with st.spinner("DeepSeek reasoning…"):
            analysis = _deepseek_analyse(symbol, setup, regime)
        st.session_state[cache_key] = analysis

    analysis = st.session_state.get(cache_key)
    if analysis:
        st.markdown(
            f"<div style='background:#111827;border:1px solid rgba(167,139,250,.15);"
            f"border-radius:8px;padding:10px 12px;font-size:.7rem;color:#c9d1e0;"
            f"line-height:1.65;font-family:JetBrains Mono,monospace;white-space:pre-wrap'>"
            f"{analysis}</div>",
            unsafe_allow_html=True,
        )


@st.cache_data(ttl=1800, show_spinner=False)
def _deepseek_analyse(symbol: str, setup: dict | None, regime: dict) -> str:
    import os, requests
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        return "⚠️ DEEPSEEK_API_KEY not configured."

    setup_block = ""
    if setup:
        ev_str = f"+{setup['ev_r']:.1f}R" if setup['ev_r'] >= 0 else f"{setup['ev_r']:.1f}R"
        setup_block = (
            f"\nSETUP:\n"
            f"  Archetype: {setup['archetype']}\n"
            f"  Quality Tier: {setup['tier']} ({setup['score']:.0f}/100)\n"
            f"  Pivot: ₹{setup['pivot']:,.0f}  Stop: ₹{setup['stop']:,.0f}\n"
            f"  Risk: {setup['risk_pct']:.1f}%  EV: {ev_str}\n"
            f"  Regime Alignment: {setup['reg_align']}\n"
            f"  Evidence: {'; '.join(setup.get('evidence', [])[:3])}"
        )

    prompt = (
        f"Institutional analysis for {symbol} (NSE India).\n\n"
        f"MARKET REGIME:\n"
        f"  {regime.get('market','?')} · VIX {regime.get('vix',16):.1f} ({regime.get('volatility','?')})\n"
        f"  Breadth: {regime.get('breadth','?')} · Risk: {regime.get('risk_mode','?')}\n"
        f"  Institutional: {regime.get('inst_activity','?')}\n"
        f"  Leaders: {', '.join(regime.get('leaders', ['N/A'])[:3])}"
        f"{setup_block}\n\n"
        f"Output EXACTLY this structure (fill in the blanks, no extra text):\n\n"
        f"SETUP QUALITY: [ELITE / STRONG / SELECTIVE / SKIP]\n\n"
        f"WHY IT MATTERS:\n"
        f"• [key bullish factor 1]\n"
        f"• [key bullish factor 2]\n"
        f"• [optional 3rd factor]\n\n"
        f"RISK:\n"
        f"• [primary risk]\n"
        f"• [secondary risk]\n\n"
        f"EXPECTANCY:\n"
        f"[one sentence on EV and best-case scenario]\n\n"
        f"VERDICT: [BUY AT PIVOT / WAIT FOR CONFIRMATION / SKIP] — [specific trigger]\n\n"
        f"Max 150 words total. Probabilistic language. No retail commentary."
    )

    try:
        resp = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "Senior institutional equity analyst. Terse, probabilistic, no fluff. Terminal output only."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 350,
            },
            timeout=25,
        )
        data = resp.json()
        if "choices" not in data:
            return f"API error: {data.get('error',{}).get('message','Unknown')}"
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Analysis unavailable: {e}"


# ── Main Entry Point ───────────────────────────────────────────────────────────

def render_institutional_terminal(universe: list[str]) -> None:
    from ui.regime_bar import render_regime_bar, get_regime

    # Persistent regime bar
    render_regime_bar()

    # Get regime for downstream use
    regime = get_regime()

    # Header
    hc, sc = st.columns([4, 1])
    with hc:
        st.markdown(
            "<h2 style='color:#00d4ff;font-family:JetBrains Mono,monospace;"
            "font-size:1.1rem;letter-spacing:2px;margin:0 0 2px'>"
            "🏛️ INSTITUTIONAL TERMINAL</h2>"
            "<span style='color:#4a5568;font-size:.65rem'>"
            "8-Stage Pipeline · Regime-Aware Ranking · Expectancy-Based EV · "
            "Behavioral Setup Detection</span>",
            unsafe_allow_html=True,
        )
    with sc:
        manual = st.text_input(
            "Symbol", value="", placeholder="Jump to…",
            key="iq2_manual", label_visibility="collapsed"
        )
        if manual.strip().upper():
            st.session_state["iq2_selected"]   = manual.strip().upper()
            st.session_state["iq2_setup_data"] = None

    selected   = st.session_state.get("iq2_selected", universe[0] if universe else "RELIANCE")
    setup_data = st.session_state.get("iq2_setup_data")

    universe_key = ",".join(sorted(universe))

    # 3-panel layout
    col_left, col_center, col_right = st.columns([28, 44, 28])

    with col_left:
        _render_setup_queue(universe_key, regime)

    with col_center:
        _render_chart_panel(selected, setup_data)

    with col_right:
        _render_intelligence_panel(selected, setup_data, regime)
