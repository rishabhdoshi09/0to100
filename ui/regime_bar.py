"""
Regime Bar — renders a compact persistent strip at the top of every page.
Shows: Market Regime | Regime Score | Nifty | VIX | Breadth | Sector Leader
"""
from __future__ import annotations

import streamlit as st


@st.cache_data(ttl=900, show_spinner=False)
def _get_regime_cached() -> dict:
    """Cache regime for 15 min — regime doesn't change minute to minute."""
    # Try core (production) engine first, fall back to analytics engine
    try:
        from core.regime_engine import compute_regime
        r = compute_regime()
        leaders = getattr(r, "leading_sectors", [])
        return {
            "regime":           r.market_regime,
            "market":           r.market_regime,
            "regime_score":     r.regime_score,
            "emoji":            _regime_emoji(r.market_regime),
            "nifty_price":      r.nifty_price,
            "nifty_change_pct": r.nifty_change_1d,
            "vix":              r.vix,
            "vix_state":        r.volatility_regime,
            "breadth":          r.breadth_label,
            "sector_leader":    leaders[0] if leaders else "N/A",
            "quality_multiplier": r.quality_multiplier,
            "timestamp":        r.timestamp,
            "atr_regime":       r.volatility_regime,
            "volatility":       r.volatility_regime,
            "risk_mode":        r.risk_mode,
            "inst_activity":    r.institutional_activity,
            "leaders":          r.leading_sectors,
            "laggards":         r.lagging_sectors,
            "rotation":         r.rotation_mode,
            "breakout_env":     r.breakout_environment,
            "playbooks":        r.recommended_playbooks,
            "avoid":            r.avoid_patterns,
            "breadth_score":    r.breadth_strength,
            "nifty_5d":         r.nifty_change_5d,
            "sector_returns":   r.sector_returns,
        }
    except Exception:
        pass
    # Fallback to old analytics engine
    try:
        from analytics.regime_engine import compute_regime as _old_compute
        r = _old_compute()
        return {
            "regime": r.regime, "market": r.regime,
            "regime_score": r.regime_score, "emoji": r.emoji,
            "nifty_price": r.nifty_price, "nifty_change_pct": r.nifty_change_pct,
            "vix": r.vix, "vix_state": r.vix_state, "breadth": r.breadth,
            "sector_leader": r.sector_leader, "quality_multiplier": r.quality_multiplier,
            "timestamp": r.timestamp, "atr_regime": r.atr_regime,
            "volatility": r.vix_state, "risk_mode": "NEUTRAL",
            "inst_activity": "NEUTRAL", "leaders": [r.sector_leader],
            "laggards": [], "rotation": "MIXED", "breakout_env": "NEUTRAL",
            "playbooks": [], "avoid": [], "breadth_score": r.regime_score,
            "nifty_5d": 0.0, "sector_returns": {},
        }
    except Exception:
        return {
            "regime": "UNKNOWN", "market": "UNKNOWN", "regime_score": 50.0, "emoji": "⚪",
            "nifty_price": 0.0, "nifty_change_pct": 0.0, "vix": 16.0,
            "vix_state": "NORMAL", "breadth": "NEUTRAL", "sector_leader": "N/A",
            "quality_multiplier": 1.0, "timestamp": "--", "atr_regime": "STABLE",
            "volatility": "NORMAL", "risk_mode": "NEUTRAL", "inst_activity": "NEUTRAL",
            "leaders": [], "laggards": [], "rotation": "MIXED", "breakout_env": "NEUTRAL",
            "playbooks": [], "avoid": [], "breadth_score": 50.0,
            "nifty_5d": 0.0, "sector_returns": {},
        }


def _regime_emoji(regime: str) -> str:
    return {
        "TRENDING_BULL": "🟢", "EXPANSION": "🚀", "CHOPPY": "🟡",
        "COMPRESSION": "🔵", "DISTRIBUTION": "🟠", "TRENDING_BEAR": "🔴",
    }.get(regime, "⚪")


def get_regime() -> dict:
    """Returns current regime dict (cached 15 min)."""
    return _get_regime_cached()


def _pill(label: str, bg: str, fg: str = "#ffffff", emoji: str = "") -> str:
    """Return an inline HTML pill/badge string."""
    prefix = f"{emoji} " if emoji else ""
    return (
        f"<span style='display:inline-block;background:{bg};color:{fg};"
        f"border-radius:4px;padding:2px 8px;font-size:.72rem;font-weight:600;"
        f"font-family:JetBrains Mono,monospace;white-space:nowrap;margin-right:4px'>"
        f"{prefix}{label}</span>"
    )


def render_regime_bar() -> None:
    """
    Renders a compact horizontal regime strip.
    Call this at the top of any page that should show market context.

    Line 1 (metrics row): Regime | Nifty | VIX | Breadth | Sector Leader | ATR | Setup×
    Line 2 (5D command strip): REGIME · VIX STATE · BREADTH · INSTITUTIONAL · RISK MODE · Sector Leaders
    """
    r = get_regime()

    # ── Legacy metrics row (unchanged) ───────────────────────────────────────
    regime_colors = {
        "BULL_TREND":   ("#00d4a0", "#002a1f"),
        "EXPANSION":    ("#00d4ff", "#002030"),
        "CHOPPY":       ("#f59e0b", "#2a1f00"),
        "DISTRIBUTION": ("#fb923c", "#2a1000"),
        "BEAR":         ("#ff4b4b", "#2a0000"),
        "UNKNOWN":      ("#8892a4", "#111827"),
    }

    nifty_chg   = r["nifty_change_pct"]
    chg_str     = f"{nifty_chg:+.2f}%"
    score       = r["regime_score"]
    qm          = r["quality_multiplier"]
    vix         = r["vix"]
    breadth     = r["breadth"]
    sec_leader  = r["sector_leader"]
    ts          = r["timestamp"]

    regime_emoji_map = {
        "BULL_TREND": "🟢", "TRENDING_BULL": "🟢", "EXPANSION": "🚀",
        "CHOPPY": "🟡", "COMPRESSION": "🔵", "DISTRIBUTION": "🟠",
        "BEAR": "🔴", "TRENDING_BEAR": "🔴",
    }
    emoji = regime_emoji_map.get(r["regime"], "⚪")
    regime_name = r["regime"].replace("_", " ")

    st.caption(f"📡 Regime Strip — {ts}")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    with c1:
        st.metric(f"{emoji} Regime", regime_name, f"{score:.0f}/100")
    with c2:
        st.metric("Nifty 50", f"{r['nifty_price']:,.0f}", chg_str)
    with c3:
        vix_lbl = r.get("vix_state", "").replace("_", " ")
        st.metric("India VIX", f"{vix:.1f}", vix_lbl)
    with c4:
        b_score = r.get("breadth_score", 0)
        st.metric("Breadth", breadth, f"{b_score:.0f}/100")
    with c5:
        st.metric("Sector Leader", sec_leader)
    with c6:
        st.metric("ATR Regime", r.get("atr_regime", "—").replace("_", " "))
    with c7:
        qm_icon = "🟢" if qm >= 1.1 else ("🟡" if qm >= 0.9 else "🔴")
        st.metric("Setup ×", f"{qm_icon} ×{qm:.2f}")

    # ── 5D Command Strip ─────────────────────────────────────────────────────
    # Pill 1 — Regime
    regime_pill_bg = {
        "TRENDING_BULL": "#00d4a0", "BULL_TREND": "#00d4a0", "EXPANSION": "#00d4ff",
        "CHOPPY": "#f59e0b", "COMPRESSION": "#3b82f6",
        "DISTRIBUTION": "#fb923c", "BEAR": "#ff4b4b", "TRENDING_BEAR": "#ff4b4b",
    }.get(r["regime"], "#8892a4")
    p1 = _pill(regime_name, regime_pill_bg, "#001a10" if "BULL" in r["regime"] or "EXPAN" in r["regime"] else "#ffffff", emoji)

    # Pill 2 — VIX
    vix_state = r.get("vix_state", "NORMAL")
    vix_bg = {"LOW": "#00d4a0", "NORMAL": "#f59e0b", "HIGH": "#ff4b4b"}.get(vix_state, "#8892a4")
    vix_fg = "#001a10" if vix_state == "LOW" else "#ffffff"
    p2 = _pill(f"VIX {vix:.1f} {vix_state.replace('_', ' ')}", vix_bg, vix_fg)

    # Pill 3 — Breadth
    breadth_bg = {"STRONG": "#00d4a0", "NEUTRAL": "#f59e0b", "WEAK": "#ff4b4b"}.get(breadth, "#8892a4")
    breadth_fg = "#001a10" if breadth == "STRONG" else "#ffffff"
    p3 = _pill(f"BREADTH {breadth}", breadth_bg, breadth_fg)

    # Pill 4 — Institutional Activity
    inst = r.get("inst_activity", "NEUTRAL")
    inst_cfg = {
        "ACCUMULATION": ("#3b82f6", "#e0f2fe", "🔵"),
        "DISTRIBUTION": ("#ef4444", "#ffffff", "🔴"),
        "RISK_ON":       ("#00d4a0", "#001a10", "🟢"),
        "RISK_OFF":      ("#f97316", "#ffffff", "🟠"),
        "NEUTRAL":       ("#6b7280", "#ffffff", "⚪"),
    }
    ibg, ifg, iemoji = inst_cfg.get(inst, ("#6b7280", "#ffffff", "⚪"))
    p4 = _pill(inst.replace("_", " "), ibg, ifg, iemoji)

    # Pill 5 — Risk Mode
    risk_mode = r.get("risk_mode", "NEUTRAL")
    risk_cfg = {
        "OFFENSIVE": ("#00d4a0", "#001a10", "⚔️"),
        "DEFENSIVE": ("#ef4444", "#ffffff", "🛡️"),
        "NEUTRAL":   ("#6b7280", "#ffffff", ""),
    }
    rbg, rfg, remoji = risk_cfg.get(risk_mode, ("#6b7280", "#ffffff", ""))
    p5 = _pill(risk_mode, rbg, rfg, remoji)

    # Pill 6 — Sector Leaders (top 2)
    leaders = r.get("leaders", [])
    if not leaders and r.get("sector_leader") and r["sector_leader"] != "N/A":
        leaders = [r["sector_leader"]]
    top2 = leaders[:2]
    if top2:
        p6 = _pill("▲ " + " · ".join(top2), "#1e3a5f", "#93c5fd")
    else:
        p6 = ""

    pills_html = (
        "<div style='display:flex;flex-wrap:wrap;align-items:center;gap:2px;"
        "padding:4px 0;border-top:1px solid rgba(255,255,255,0.07);margin-top:4px'>"
        + p1 + p2 + p3 + p4 + p5 + p6
        + "</div>"
    )
    st.markdown(pills_html, unsafe_allow_html=True)
