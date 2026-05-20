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


def render_regime_bar() -> None:
    """
    Renders a compact horizontal regime strip.
    Call this at the top of any page that should show market context.
    """
    r = get_regime()

    regime_colors = {
        "BULL_TREND":   ("#00d4a0", "#002a1f"),
        "EXPANSION":    ("#00d4ff", "#002030"),
        "CHOPPY":       ("#f59e0b", "#2a1f00"),
        "DISTRIBUTION": ("#fb923c", "#2a1000"),
        "BEAR":         ("#ff4b4b", "#2a0000"),
        "UNKNOWN":      ("#8892a4", "#111827"),
    }
    fg, bg = regime_colors.get(r["regime"], ("#8892a4", "#111827"))

    nifty_chg   = r["nifty_change_pct"]
    chg_color   = "#00d4a0" if nifty_chg >= 0 else "#ff4b4b"
    chg_arrow   = "▲" if nifty_chg >= 0 else "▼"
    breadth_col = {"STRONG": "#00d4a0", "NEUTRAL": "#f59e0b", "WEAK": "#ff4b4b"}.get(r["breadth"], "#8892a4")
    vix_col     = {"LOW": "#00d4a0", "NORMAL": "#f59e0b", "HIGH": "#ff4b4b"}.get(r["vix_state"], "#8892a4")

    regime_name = r["regime"].replace("_", " ")
    nifty_chg   = r["nifty_change_pct"]
    chg_str     = f"{nifty_chg:+.2f}%"
    score       = r["regime_score"]
    qm          = r["quality_multiplier"]
    vix         = r["vix"]
    breadth     = r["breadth"]
    sec_leader  = r["sector_leader"]
    ts          = r["timestamp"]

    regime_emoji = {
        "BULL_TREND": "🟢", "TRENDING_BULL": "🟢", "EXPANSION": "🚀",
        "CHOPPY": "🟡", "COMPRESSION": "🔵", "DISTRIBUTION": "🟠",
        "BEAR": "🔴", "TRENDING_BEAR": "🔴",
    }
    emoji = regime_emoji.get(r["regime"], "⚪")

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
